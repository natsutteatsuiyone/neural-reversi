use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion};

use reversi_core::bitboard::Bitboard;
use reversi_core::constants::SCORE_MAX;
use reversi_core::count_last_flip::{count_last_flip, solve1};
use reversi_core::square::Square;
use reversi_core::types::Score;

const FIXTURE_BYTES: &[u8] = include_bytes!("fixtures/solve1_trace_v1.bin");
const MAGIC: &[u8; 8] = b"NRS1TRC\0";
const FORMAT_VERSION: u32 = 1;
const RECORD_SIZE: u32 = 16;

const SECTION_FROM_SOLVE2: u32 = 1;
const SECTION_FROM_SOLVE3: u32 = 2;
const SECTION_FROM_SOLVE4: u32 = 3;
const SECTION_MIXED_ORDERED: u32 = 4;
const SECTION_MIXED_SHUFFLED: u32 = 5;

const SCRATCH_WORDS_64K: usize = 8192;

#[derive(Clone, Copy)]
struct Solve1TraceCase {
    player: u64,
    alpha: Score,
    sq: Square,
    root_empty_count: u8,
}

#[derive(Default)]
struct Solve1Fixture {
    trace2: Vec<Solve1TraceCase>,
    trace3: Vec<Solve1TraceCase>,
    trace4: Vec<Solve1TraceCase>,
    mixed_ordered: Vec<Solve1TraceCase>,
    mixed_shuffled: Vec<Solve1TraceCase>,
}

#[derive(Debug, Default)]
struct TraceStats {
    total: usize,
    hot_flip: usize,
    player_pass_pruned: usize,
    opponent_checked: usize,
    both_pass: usize,
}

#[repr(align(64))]
struct CacheScratch {
    words: [u64; SCRATCH_WORDS_64K],
}

impl CacheScratch {
    fn new() -> Self {
        let mut words = [0u64; SCRATCH_WORDS_64K];
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for word in &mut words {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *word = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        }
        Self { words }
    }
}

struct FixtureReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> FixtureReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_bytes<const N: usize>(&mut self, what: &str) -> [u8; N] {
        let end = self
            .offset
            .checked_add(N)
            .unwrap_or_else(|| panic!("solve1 fixture offset overflow while reading {what}"));
        let bytes = self
            .bytes
            .get(self.offset..end)
            .unwrap_or_else(|| panic!("solve1 fixture ended while reading {what}"));
        self.offset = end;

        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        out
    }

    fn read_u8(&mut self, what: &str) -> u8 {
        self.read_bytes::<1>(what)[0]
    }

    fn read_u32(&mut self, what: &str) -> u32 {
        u32::from_le_bytes(self.read_bytes::<4>(what))
    }

    fn read_i32(&mut self, what: &str) -> i32 {
        i32::from_le_bytes(self.read_bytes::<4>(what))
    }

    fn read_u64(&mut self, what: &str) -> u64 {
        u64::from_le_bytes(self.read_bytes::<8>(what))
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn load_fixture(bytes: &[u8]) -> Solve1Fixture {
    let mut reader = FixtureReader::new(bytes);
    assert_eq!(
        &reader.read_bytes::<8>("magic"),
        MAGIC,
        "unexpected solve1 fixture magic"
    );
    assert_eq!(
        reader.read_u32("format version"),
        FORMAT_VERSION,
        "unsupported solve1 fixture version"
    );
    assert_eq!(
        reader.read_u32("record size"),
        RECORD_SIZE,
        "unsupported solve1 fixture record size"
    );

    let section_count = reader.read_u32("section count");
    let mut fixture = Solve1Fixture::default();
    for _ in 0..section_count {
        let section_id = reader.read_u32("section id");
        let trace = read_trace_section(&mut reader);
        match section_id {
            SECTION_FROM_SOLVE2 => fixture.trace2 = trace,
            SECTION_FROM_SOLVE3 => fixture.trace3 = trace,
            SECTION_FROM_SOLVE4 => fixture.trace4 = trace,
            SECTION_MIXED_ORDERED => fixture.mixed_ordered = trace,
            SECTION_MIXED_SHUFFLED => fixture.mixed_shuffled = trace,
            _ => panic!("unknown solve1 fixture section id: {section_id}"),
        }
    }

    assert!(reader.is_empty(), "solve1 fixture has trailing bytes");
    assert!(!fixture.trace2.is_empty(), "solve1 trace2 fixture is empty");
    assert!(!fixture.trace3.is_empty(), "solve1 trace3 fixture is empty");
    assert!(!fixture.trace4.is_empty(), "solve1 trace4 fixture is empty");
    assert!(
        !fixture.mixed_ordered.is_empty(),
        "solve1 mixed ordered fixture is empty"
    );
    assert!(
        !fixture.mixed_shuffled.is_empty(),
        "solve1 mixed shuffled fixture is empty"
    );

    fixture
}

fn read_trace_section(reader: &mut FixtureReader<'_>) -> Vec<Solve1TraceCase> {
    let count = reader.read_u32("section record count") as usize;
    let mut trace = Vec::with_capacity(count);
    for _ in 0..count {
        let player = reader.read_u64("record player");
        let alpha = reader.read_i32("record alpha");
        let sq_index = reader.read_u8("record square");
        let root_empty_count = reader.read_u8("record root empty count");
        let padding = reader.read_bytes::<2>("record padding");
        assert_eq!(
            padding,
            [0, 0],
            "solve1 fixture record padding must be zero"
        );

        let sq = Square::from_u8(sq_index)
            .filter(|&sq| sq != Square::None)
            .unwrap_or_else(|| panic!("invalid solve1 fixture square index: {sq_index}"));
        trace.push(Solve1TraceCase {
            player,
            alpha,
            sq,
            root_empty_count,
        });
    }
    trace
}

fn trace_stats(trace: &[Solve1TraceCase]) -> TraceStats {
    let mut stats = TraceStats::default();
    for case in trace {
        stats.total += 1;
        let player = Bitboard::new(case.player);
        let n_flipped = count_last_flip(player, case.sq);
        if n_flipped != 0 {
            stats.hot_flip += 1;
            continue;
        }

        let score_base = 2 * player.count() as Score - SCORE_MAX + 2;
        let score_if_opp_passes = if score_base > 0 {
            score_base
        } else {
            score_base - 2
        };
        if score_if_opp_passes <= case.alpha {
            stats.player_pass_pruned += 1;
        } else if count_last_flip(!player, case.sq) != 0 {
            stats.opponent_checked += 1;
        } else {
            stats.both_pass += 1;
        }
    }
    stats
}

#[inline(always)]
fn checksum_trace(cases: &[Solve1TraceCase]) -> Score {
    let mut acc = 0;
    for &case in cases {
        let player = Bitboard::new(black_box(case.player));
        let score = solve1(player, black_box(case.alpha), black_box(case.sq));
        acc ^= score ^ case.root_empty_count as Score;
    }
    black_box(acc)
}

#[inline(never)]
fn touch_scratch(scratch: &mut CacheScratch, key: u64) -> u64 {
    let mut idx = key as usize & (SCRATCH_WORDS_64K - 1);
    let mut acc = key.rotate_left(17);
    for _ in 0..2 {
        // SAFETY: `idx` is masked by `SCRATCH_WORDS_64K - 1`, and the length is exactly
        // `SCRATCH_WORDS_64K`.
        let word = unsafe { scratch.words.get_unchecked_mut(idx) };
        let v = *word;
        *word = v.wrapping_add(acc).rotate_left(9);
        acc ^= v;
        idx = (idx.wrapping_mul(131).wrapping_add(acc as usize)) & (SCRATCH_WORDS_64K - 1);
    }
    black_box(acc)
}

#[inline(always)]
fn checksum_trace_with_cache_pressure(
    cases: &[Solve1TraceCase],
    scratch: &mut CacheScratch,
) -> Score {
    let mut acc = 0;
    for &case in cases {
        let salt = case.player ^ ((case.alpha as u64) << 48) ^ case.sq.index() as u64;
        acc ^= touch_scratch(scratch, salt) as Score;
        let player = Bitboard::new(black_box(case.player));
        let score = solve1(player, black_box(case.alpha), black_box(case.sq));
        acc ^= score ^ case.root_empty_count as Score;
    }
    black_box(acc)
}

pub(crate) fn bench_solve1(c: &mut Criterion) {
    let fixture = load_fixture(FIXTURE_BYTES);

    let mut group = c.benchmark_group("count_last_flip::solve1/leaf_trace");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(2));

    eprintln!(
        "solve1 trace 2-empty stats: {:?}",
        trace_stats(&fixture.trace2)
    );
    eprintln!(
        "solve1 trace 3-empty stats: {:?}",
        trace_stats(&fixture.trace3)
    );
    eprintln!(
        "solve1 trace 4-empty stats: {:?}",
        trace_stats(&fixture.trace4)
    );
    eprintln!(
        "solve1 trace mixed stats:  {:?}",
        trace_stats(&fixture.mixed_ordered)
    );

    group.bench_with_input(
        BenchmarkId::new("direct", "2_empty"),
        &fixture.trace2,
        |b, cases| b.iter(|| checksum_trace(black_box(cases))),
    );
    group.bench_with_input(
        BenchmarkId::new("direct", "3_empty"),
        &fixture.trace3,
        |b, cases| b.iter(|| checksum_trace(black_box(cases))),
    );
    group.bench_with_input(
        BenchmarkId::new("direct", "4_empty"),
        &fixture.trace4,
        |b, cases| b.iter(|| checksum_trace(black_box(cases))),
    );
    group.bench_with_input(
        BenchmarkId::new("direct", "mixed_ordered"),
        &fixture.mixed_ordered,
        |b, cases| b.iter(|| checksum_trace(black_box(cases))),
    );
    group.bench_with_input(
        BenchmarkId::new("direct", "mixed_shuffled"),
        &fixture.mixed_shuffled,
        |b, cases| b.iter(|| checksum_trace(black_box(cases))),
    );
    group.bench_with_input(
        BenchmarkId::new("direct_cache64k", "mixed_ordered"),
        &fixture.mixed_ordered,
        |b, cases| {
            b.iter_batched_ref(
                CacheScratch::new,
                |scratch| checksum_trace_with_cache_pressure(black_box(cases), black_box(scratch)),
                BatchSize::SmallInput,
            )
        },
    );
    group.finish();
}
