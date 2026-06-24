use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

mod common;

use common::random_square;
use reversi_core::bitboard::Bitboard;
use reversi_core::constants::{SCORE_MAX, SCORE_MIN};
use reversi_core::count_last_flip::{count_last_flip, solve1};
use reversi_core::square::Square;
use reversi_core::types::Score;

const N_CASES: usize = 4096;
const N_CASES_PER_SQUARE: usize = 256;

#[derive(Clone, Copy)]
struct CountCase {
    sq: Square,
    player: u64,
}

#[derive(Clone, Copy)]
struct SolveCase {
    player: u64,
    sq: Square,
    alpha: Score,
}

#[derive(Clone, Copy, Debug)]
enum CallMode {
    Plain,
    Reg8,
    Reg12,
}

impl CallMode {
    const ALL: [CallMode; 3] = [CallMode::Plain, CallMode::Reg8, CallMode::Reg12];

    const fn name(self) -> &'static str {
        match self {
            CallMode::Plain => "plain",
            CallMode::Reg8 => "reg8",
            CallMode::Reg12 => "reg12",
        }
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! touch_live8 {
    ($v0:ident, $v1:ident, $v2:ident, $v3:ident, $v4:ident, $v5:ident, $v6:ident, $v7:ident) => {
        unsafe {
            core::arch::asm!(
                "",
                inout("rax") $v0,
                inout("rcx") $v1,
                inout("rdx") $v2,
                inout("rsi") $v3,
                inout("rdi") $v4,
                inout("r8") $v5,
                inout("r9") $v6,
                inout("r10") $v7,
                options(nomem, nostack, preserves_flags),
            );
        }
    };
}

#[cfg(not(target_arch = "x86_64"))]
macro_rules! touch_live8 {
    ($($v:ident),+) => {
        $(black_box($v);)+
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! touch_live12 {
    (
        $v0:ident, $v1:ident, $v2:ident, $v3:ident,
        $v4:ident, $v5:ident, $v6:ident, $v7:ident,
        $v8:ident, $v9:ident, $v10:ident, $v11:ident
    ) => {
        unsafe {
            core::arch::asm!(
                "",
                inout("rax") $v0,
                inout("rcx") $v1,
                inout("rdx") $v2,
                inout("rsi") $v3,
                inout("rdi") $v4,
                inout("r8") $v5,
                inout("r9") $v6,
                inout("r10") $v7,
                inout("r11") $v8,
                inout("r12") $v9,
                inout("r13") $v10,
                inout("r14") $v11,
                options(nomem, nostack, preserves_flags),
            );
        }
    };
}

#[cfg(not(target_arch = "x86_64"))]
macro_rules! touch_live12 {
    ($($v:ident),+) => {
        $(black_box($v);)+
    };
}

fn random_count_cases(seed: u64) -> Vec<CountCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);
        out.push(CountCase { sq, player });
    }
    out
}

fn random_empty_square_case(rng: &mut StdRng) -> (Square, u64) {
    loop {
        let player: u64 = rng.random();
        let empty = !player;
        if empty == 0 {
            continue;
        }

        let sq = loop {
            let candidate = random_square(rng);
            if empty & (1u64 << candidate.index()) != 0 {
                break candidate;
            }
        };

        return (sq, player);
    }
}

fn random_player_for_square(rng: &mut StdRng, sq: Square) -> u64 {
    rng.random::<u64>() & !(1u64 << sq.index())
}

fn random_player_with_disc_count(rng: &mut StdRng, sq: Square, count: u32) -> u64 {
    debug_assert!(count <= 63);
    let sq_idx = sq.index();
    let mut player = 0u64;

    while player.count_ones() < count {
        let bit_idx = rng.random_range(0..64);
        if bit_idx != sq_idx {
            player |= 1u64 << bit_idx;
        }
    }

    player
}

fn classify_player_can_move(player: u64, sq: Square) -> bool {
    count_last_flip(Bitboard::new(player), sq) != 0
}

fn classify_opponent_can_move(player: u64, sq: Square) -> bool {
    count_last_flip(Bitboard::new(!player), sq) != 0
}

fn random_mixed_cases(seed: u64) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);
        out.push(SolveCase {
            player,
            sq,
            alpha: rng.random_range(SCORE_MIN..=SCORE_MAX),
        });
    }
    out
}

fn random_hot_flip_cases(seed: u64) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);
        if classify_player_can_move(player, sq) {
            out.push(SolveCase {
                player,
                sq,
                alpha: rng.random_range(SCORE_MIN..=SCORE_MAX),
            });
        }
    }
    out
}

fn player_cannot_move_cases(seed: u64, alpha: Score) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);
        if !classify_player_can_move(player, sq) {
            out.push(SolveCase { player, sq, alpha });
        }
    }
    out
}

fn both_pass_cases(seed: u64) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);

    // Random sampling makes true both-pass positions extremely rare. Use simple
    // constructed one-empty boards that still exercise the same solve1 branch:
    // either the player has no discs, or the player owns every non-empty square.
    while out.len() < N_CASES {
        let sq = random_square(&mut rng);
        let sq_bit = 1u64 << sq.index();
        let player = if rng.random::<bool>() { 0 } else { !sq_bit };
        debug_assert!(!classify_player_can_move(player, sq));
        debug_assert!(!classify_opponent_can_move(player, sq));

        // Force the opponent check path. The second count returns zero.
        out.push(SolveCase {
            player,
            sq,
            alpha: SCORE_MIN - 1,
        });
    }

    out
}

fn square_class_cases(seed: u64, squares: &[Square]) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        for &sq in squares {
            if out.len() == N_CASES {
                break;
            }
            out.push(SolveCase {
                player: random_player_for_square(&mut rng, sq),
                sq,
                alpha: rng.random_range(SCORE_MIN..=SCORE_MAX),
            });
        }
    }
    out
}

fn per_square_cases(seed: u64, sq: Square) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed ^ sq.index() as u64);
    let mut out = Vec::with_capacity(N_CASES_PER_SQUARE);
    while out.len() < N_CASES_PER_SQUARE {
        out.push(SolveCase {
            player: random_player_for_square(&mut rng, sq),
            sq,
            alpha: rng.random_range(SCORE_MIN..=SCORE_MAX),
        });
    }
    out
}

fn density_cases(seed: u64, min_discs: u32, max_discs: u32) -> Vec<SolveCase> {
    debug_assert!(min_discs <= max_discs && max_discs <= 63);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let sq = random_square(&mut rng);
        let count = rng.random_range(min_discs..=max_discs);
        out.push(SolveCase {
            player: random_player_with_disc_count(&mut rng, sq, count),
            sq,
            alpha: rng.random_range(SCORE_MIN..=SCORE_MAX),
        });
    }
    out
}

fn square_from_index(index: u32) -> Square {
    Square::from_u32(index).expect("square index must be in 0..64")
}

fn validate_cases(name: &str, cases: &[SolveCase]) {
    assert_eq!(
        cases.len(),
        N_CASES,
        "{name} must contain exactly N_CASES cases"
    );
    for case in cases {
        assert_eq!(
            case.player & (1u64 << case.sq.index()),
            0,
            "{name}: sq must be empty"
        );
    }
}

fn validate_per_square_cases(name: &str, cases: &[SolveCase], sq: Square) {
    assert_eq!(
        cases.len(),
        N_CASES_PER_SQUARE,
        "{name} must contain exactly N_CASES_PER_SQUARE cases",
    );
    for case in cases {
        assert_eq!(case.sq.index(), sq.index(), "{name}: wrong square");
        assert_eq!(
            case.player & (1u64 << case.sq.index()),
            0,
            "{name}: sq must be empty"
        );
    }
}

#[inline(always)]
fn call_solve1_plain(case: SolveCase) -> Score {
    solve1(
        Bitboard::new(black_box(case.player)),
        black_box(case.alpha),
        black_box(case.sq),
    )
}

/// Synthetic caller with 8 live u64 values across `solve1`.
///
/// This is not meant to model one exact search frame. It is a controlled probe
/// for regressions caused by larger metadata records, extra branches, or longer
/// live ranges inside `solve1`.
#[inline(never)]
fn call_solve1_reg8(case: SolveCase, salt: u64) -> Score {
    let mut v0 = case.player.wrapping_add(salt.rotate_left(1));
    let mut v1 = case.player ^ 0x9e37_79b9_7f4a_7c15 ^ salt.rotate_left(7);
    let mut v2 = case.player.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let mut v3 = salt.wrapping_mul(0x94d0_49bb_1331_11eb);
    let mut v4 = case.player.rotate_left(17) ^ salt;
    let mut v5 = case.player.rotate_right(23).wrapping_add(v1);
    let mut v6 = v2 ^ v3.rotate_left(31);
    let mut v7 = v4.wrapping_add(v5).rotate_right(11);

    touch_live8!(v0, v1, v2, v3, v4, v5, v6, v7);
    let score = call_solve1_plain(case);
    touch_live8!(v0, v1, v2, v3, v4, v5, v6, v7);

    black_box(v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7 ^ score as u64);
    score
}

/// Synthetic caller with 12 live u64 values across `solve1`.
#[inline(never)]
fn call_solve1_reg12(case: SolveCase, salt: u64) -> Score {
    let mut v0 = case.player.wrapping_add(salt.rotate_left(1));
    let mut v1 = case.player ^ 0x9e37_79b9_7f4a_7c15 ^ salt.rotate_left(7);
    let mut v2 = case.player.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let mut v3 = salt.wrapping_mul(0x94d0_49bb_1331_11eb);
    let mut v4 = case.player.rotate_left(17) ^ salt;
    let mut v5 = case.player.rotate_right(23).wrapping_add(v1);
    let mut v6 = v2 ^ v3.rotate_left(31);
    let mut v7 = v4.wrapping_add(v5).rotate_right(11);
    let mut v8 = v6.wrapping_mul(0xd6e8_feb8_6659_fd93);
    let mut v9 = v7 ^ v8.rotate_left(19);
    let mut v10 = v0.wrapping_add(v9).rotate_right(29);
    let mut v11 = v10 ^ v3.wrapping_mul(0xa076_1d64_78bd_642f);

    touch_live12!(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11);
    let score = call_solve1_plain(case);
    touch_live12!(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11);

    black_box(v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7 ^ v8 ^ v9 ^ v10 ^ v11 ^ score as u64);
    score
}

#[inline(never)]
fn checksum_cases(cases: &[SolveCase], mode: CallMode) -> i32 {
    let mut acc = 0i32;
    for (i, &case) in cases.iter().enumerate() {
        let score = match mode {
            CallMode::Plain => call_solve1_plain(case),
            CallMode::Reg8 => call_solve1_reg8(case, i as u64),
            CallMode::Reg12 => call_solve1_reg12(case, i as u64),
        };
        acc = acc.rotate_left(1) ^ score;
    }
    black_box(acc)
}

#[inline(always)]
fn direct_solve1_without_specialization(player: Bitboard, alpha: Score, sq: Square) -> Score {
    let mut n_flipped = count_last_flip(player, sq);
    let mut score = 2 * player.count() as Score - SCORE_MAX + 2 + n_flipped;

    if n_flipped == 0 {
        let score_if_opp_passes = if score > 0 { score } else { score - 2 };
        if score_if_opp_passes > alpha {
            n_flipped = count_last_flip(!player, sq);
            score = if n_flipped > 0 {
                score - 2 - n_flipped
            } else {
                score_if_opp_passes
            };
        } else {
            score = score_if_opp_passes;
        }
    }

    score
}

#[inline(never)]
fn checksum_direct_count_calls(cases: &[SolveCase]) -> i32 {
    let mut acc = 0i32;
    for &case in cases {
        let score = direct_solve1_without_specialization(
            Bitboard::new(black_box(case.player)),
            black_box(case.alpha),
            black_box(case.sq),
        );
        acc = acc.rotate_left(1) ^ score;
    }
    black_box(acc)
}

fn bench_count_last_flip(c: &mut Criterion, cases: &[CountCase]) {
    c.bench_function("count_last_flip::count_last_flip", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for &case in cases {
                let player = Bitboard::new(black_box(case.player));
                acc ^= count_last_flip(player, black_box(case.sq));
            }
            black_box(acc)
        })
    });
}

fn bench_solve_case_set(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dataset_name: &str,
    cases: &[SolveCase],
    modes: &[CallMode],
) {
    group.throughput(Throughput::Elements(cases.len() as u64));

    for &mode in modes {
        group.bench_with_input(
            BenchmarkId::new(mode.name(), dataset_name),
            cases,
            |b, cases| b.iter(|| checksum_cases(black_box(cases), mode)),
        );
    }
}

fn bench_direct_baseline(
    group: &mut BenchmarkGroup<'_, WallTime>,
    dataset_name: &str,
    cases: &[SolveCase],
) {
    group.throughput(Throughput::Elements(cases.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("direct_count_calls", dataset_name),
        cases,
        |b, cases| b.iter(|| checksum_direct_count_calls(black_box(cases))),
    );
}

fn bench_solve_branch_profiles(c: &mut Criterion) {
    let mixed = random_mixed_cases(0x51d3_f00d);
    let hot_flip = random_hot_flip_cases(0xaced_f00d);
    let pass_pruned = player_cannot_move_cases(0xdead_beef, SCORE_MAX + 1);
    let opponent_checked = player_cannot_move_cases(0xfade_5eed, SCORE_MIN - 1);
    let both_pass = both_pass_cases(0xb07a_aa55);

    validate_cases("mixed", &mixed);
    validate_cases("hot_flip", &hot_flip);
    validate_cases("pass_pruned", &pass_pruned);
    validate_cases("opponent_checked", &opponent_checked);
    validate_cases("both_pass", &both_pass);

    let mut group = c.benchmark_group("count_last_flip::solve1/branch_profile");
    bench_solve_case_set(&mut group, "mixed", &mixed, &CallMode::ALL);
    bench_solve_case_set(&mut group, "hot_flip", &hot_flip, &CallMode::ALL);
    bench_solve_case_set(
        &mut group,
        "player_pass_pruned",
        &pass_pruned,
        &CallMode::ALL,
    );
    bench_solve_case_set(
        &mut group,
        "opponent_checked",
        &opponent_checked,
        &CallMode::ALL,
    );
    bench_solve_case_set(&mut group, "both_pass", &both_pass, &CallMode::ALL);

    // Keep the direct-count baseline plain-only: it is for judging whether the
    // specialized solve1 path is worth its tables, not for call-site pressure.
    bench_direct_baseline(&mut group, "mixed", &mixed);
    bench_direct_baseline(&mut group, "player_pass_pruned", &pass_pruned);
    bench_direct_baseline(&mut group, "opponent_checked", &opponent_checked);
    group.finish();
}

fn bench_solve_square_classes(c: &mut Criterion) {
    let corners = [
        square_from_index(0),
        square_from_index(7),
        square_from_index(56),
        square_from_index(63),
    ];
    let edges = [
        square_from_index(1),
        square_from_index(2),
        square_from_index(5),
        square_from_index(6),
        square_from_index(8),
        square_from_index(15),
        square_from_index(48),
        square_from_index(55),
    ];
    let x_squares = [
        square_from_index(9),
        square_from_index(14),
        square_from_index(49),
        square_from_index(54),
    ];
    let inner = [
        square_from_index(18),
        square_from_index(19),
        square_from_index(20),
        square_from_index(21),
        square_from_index(26),
        square_from_index(27),
        square_from_index(28),
        square_from_index(29),
        square_from_index(34),
        square_from_index(35),
        square_from_index(36),
        square_from_index(37),
        square_from_index(42),
        square_from_index(43),
        square_from_index(44),
        square_from_index(45),
    ];

    let class_sets = [
        ("corner", square_class_cases(0x5000_0001, &corners)),
        ("edge", square_class_cases(0x5000_0002, &edges)),
        ("x_square", square_class_cases(0x5000_0003, &x_squares)),
        ("inner_4x4", square_class_cases(0x5000_0004, &inner)),
    ];

    let mut group = c.benchmark_group("count_last_flip::solve1/square_class");
    for (name, cases) in &class_sets {
        validate_cases(name, cases);
        bench_solve_case_set(&mut group, name, cases, &[CallMode::Plain, CallMode::Reg8]);
    }
    group.finish();
}

fn bench_solve_density(c: &mut Criterion) {
    let sparse = density_cases(0xd300_0001, 4, 16);
    let balanced = density_cases(0xd300_0002, 24, 39);
    let dense = density_cases(0xd300_0003, 48, 63);

    validate_cases("sparse", &sparse);
    validate_cases("balanced", &balanced);
    validate_cases("dense", &dense);

    let mut group = c.benchmark_group("count_last_flip::solve1/player_density");
    bench_solve_case_set(&mut group, "sparse_4_16", &sparse, &[CallMode::Plain]);
    bench_solve_case_set(&mut group, "balanced_24_39", &balanced, &[CallMode::Plain]);
    bench_solve_case_set(&mut group, "dense_48_63", &dense, &[CallMode::Plain]);
    group.finish();
}

fn bench_solve_per_square_smoke(c: &mut Criterion) {
    let mut group = c.benchmark_group("count_last_flip::solve1/per_square_smoke");
    group.sample_size(50);

    // Representative squares with very different diagonal-union lengths and edge behavior.
    let squares = [
        ("a1", square_from_index(0)),
        ("b2", square_from_index(9)),
        ("c3", square_from_index(18)),
        ("d4", square_from_index(27)),
        ("e4", square_from_index(28)),
        ("h8", square_from_index(63)),
    ];

    for (name, sq) in squares {
        let cases = per_square_cases(0x5a00_0000, sq);
        validate_per_square_cases(name, &cases, sq);
        bench_solve_case_set(&mut group, name, &cases, &[CallMode::Plain]);
    }

    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let count_cases = random_count_cases(0xc01d_cafe);

    bench_count_last_flip(c, &count_cases);
    bench_solve_branch_profiles(c);
    bench_solve_square_classes(c);
    bench_solve_density(c);
    bench_solve_per_square_smoke(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(750))
        .measurement_time(Duration::from_secs(4))
        .sample_size(100);
    targets = criterion_benchmark
}
criterion_main!(benches);
