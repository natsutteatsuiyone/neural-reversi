use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use rand::{SeedableRng, rngs::StdRng};

mod common;

use common::{choose_square, exact_endgame_score, playout_to_n_empty};
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::eval::Eval;
use reversi_core::flip::flip;
use reversi_core::obf::ObfPosition;
use reversi_core::probcut::Selectivity;
use reversi_core::search::context::SearchContext;
use reversi_core::search::{EndGameCaches, null_window_search};
use reversi_core::square::Square;
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::types::Score;

const MOVE_CHAIN_COUNT: usize = 256;
const ENDGAME_CASE_COUNT: usize = 512;
const CORPUS_SEED: u64 = 0xf11f_5e4c_1a7e_2026;

const ENDGAME_SOURCES: &[&str] = &[
    include_str!("../../../problem/fforum-1-19.obf"),
    include_str!("../../../problem/fforum-20-39.obf"),
    include_str!("../../../problem/fforum-40-59.obf"),
    include_str!("../../../problem/fforum-60-79.obf"),
    include_str!("../../../problem/hard-20.obf"),
    include_str!("../../../problem/hard-25.obf"),
    include_str!("../../../problem/hard-30.obf"),
    include_str!("../../../problem/small-35.txt"),
];

#[derive(Clone, Copy)]
enum PlayoutStep {
    Move(Square),
    Pass,
}

struct MoveChain {
    start: Board,
    steps: Vec<PlayoutStep>,
    expected_final: Board,
}

#[derive(Clone, Copy)]
struct FlipCase {
    square: Square,
    player: Bitboard,
    opponent: Bitboard,
}

#[derive(Clone, Copy)]
struct EndgameCase<const N_EMPTY: u32> {
    board: Board,
    alpha: Score,
    expected: Score,
}

#[derive(Clone, Copy)]
struct ChainedEndgameCase<const N_EMPTY: u32> {
    board: Board,
    encoded_alpha: Score,
    expected: Score,
}

struct EndgameBenchState {
    contexts: Vec<SearchContext>,
    caches: EndGameCaches,
}

fn source_boards() -> Vec<Board> {
    let mut boards = Vec::new();
    for source in ENDGAME_SOURCES {
        for line in source.lines() {
            let Some(position) = ObfPosition::parse(line).expect("benchmark OBF line must parse")
            else {
                continue;
            };
            boards.push(position.board);
        }
    }
    assert!(!boards.is_empty(), "flip consumer corpus must not be empty");
    boards
}

fn move_chains(seed: u64) -> Vec<MoveChain> {
    let sources = source_boards();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chains = Vec::with_capacity(MOVE_CHAIN_COUNT);
    let mut source_index = 0usize;

    while chains.len() < MOVE_CHAIN_COUNT {
        let start = sources[source_index % sources.len()];
        source_index += 1;
        let mut board = start;
        let mut steps = Vec::with_capacity(board.get_empty_count() as usize + 2);

        loop {
            let moves = board.get_moves();
            if !moves.is_empty() {
                let sq = choose_square(moves, &mut rng);
                steps.push(PlayoutStep::Move(sq));
                board = board.make_move(sq);
                continue;
            }

            let passed = board.switch_players();
            if passed.get_moves().is_empty() {
                break;
            }
            steps.push(PlayoutStep::Pass);
            board = passed;
        }

        if steps
            .iter()
            .any(|&step| matches!(step, PlayoutStep::Move(_)))
        {
            chains.push(MoveChain {
                start,
                steps,
                expected_final: board,
            });
        }
    }

    chains
}

#[inline(never)]
fn replay_move_chains(chains: &[MoveChain]) -> u64 {
    let mut checksum = 0u64;

    for chain in chains {
        let mut board = chain.start;
        for &step in &chain.steps {
            board = match step {
                PlayoutStep::Move(sq) => board.make_move(sq),
                PlayoutStep::Pass => board.switch_players(),
            };
        }
        debug_assert_eq!(board, chain.expected_final);
        checksum = checksum.rotate_left(11)
            ^ board.player().bits()
            ^ board.opponent().bits().rotate_left(1);
    }

    black_box(checksum)
}

fn flip_cases(chains: &[MoveChain]) -> Vec<FlipCase> {
    let case_count = chains
        .iter()
        .map(|chain| {
            chain
                .steps
                .iter()
                .filter(|step| matches!(step, PlayoutStep::Move(_)))
                .count()
        })
        .sum();
    let mut cases = Vec::with_capacity(case_count);

    for chain in chains {
        let mut board = chain.start;
        for &step in &chain.steps {
            board = match step {
                PlayoutStep::Move(square) => {
                    cases.push(FlipCase {
                        square,
                        player: board.player(),
                        opponent: board.opponent(),
                    });
                    board.make_move(square)
                }
                PlayoutStep::Pass => board.switch_players(),
            };
        }
        debug_assert_eq!(board, chain.expected_final);
    }

    assert!(!cases.is_empty(), "single-flip corpus must not be empty");
    cases
}

#[inline(never)]
fn flip_checksum(cases: &[FlipCase]) -> u64 {
    let mut checksum = 0u64;

    for &case in cases {
        let flipped = flip(
            black_box(case.square),
            black_box(case.player),
            black_box(case.opponent),
        );
        checksum = checksum.rotate_left(7) ^ flipped.bits();
    }

    black_box(checksum)
}

fn bench_single_flip(c: &mut Criterion, cases: &[FlipCase]) {
    let mut group = c.benchmark_group("flip");
    group.throughput(Throughput::Elements(cases.len() as u64));
    group.bench_function("single", |b| {
        b.iter(|| flip_checksum(black_box(cases)));
    });
    group.finish();
}

fn endgame_cases<const N_EMPTY: u32>(seed: u64) -> Vec<EndgameCase<N_EMPTY>> {
    let sources = source_boards();
    let mut rng = StdRng::seed_from_u64(seed ^ N_EMPTY as u64);
    let mut cases = Vec::with_capacity(ENDGAME_CASE_COUNT);
    let mut source_index = 0usize;

    while cases.len() < ENDGAME_CASE_COUNT {
        let source = sources[source_index % sources.len()];
        source_index += 1;
        let Some(board) = playout_to_n_empty::<N_EMPTY>(source, &mut rng) else {
            continue;
        };
        let expected = exact_endgame_score(&board);
        let alpha = if cases.len().is_multiple_of(2) {
            expected - 1
        } else {
            expected
        };
        cases.push(EndgameCase {
            board,
            alpha,
            expected,
        });
    }

    cases
}

#[inline(always)]
fn score_dependency_key(score: Score) -> Score {
    ((score as u32).rotate_left(13) ^ 0x9e37_79b9) as Score
}

fn chained_endgame_cases<const N_EMPTY: u32>(
    cases: &[EndgameCase<N_EMPTY>],
) -> Vec<ChainedEndgameCase<N_EMPTY>> {
    let mut preceding_score = 0;
    let mut chained = Vec::with_capacity(cases.len());

    for &case in cases {
        chained.push(ChainedEndgameCase {
            board: case.board,
            encoded_alpha: case.alpha ^ score_dependency_key(preceding_score),
            expected: case.expected,
        });
        preceding_score = case.expected;
    }

    chained
}

fn make_context(board: &Board, eval: &Arc<Eval>, tt: &Arc<TranspositionTable>) -> SearchContext {
    SearchContext::new(board, Selectivity::None, tt.clone(), eval.clone())
}

fn validate_endgame_chain<const N_EMPTY: u32>(
    cases: &[ChainedEndgameCase<N_EMPTY>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    let mut preceding_score = 0;
    let mut caches = EndGameCaches::for_thread_count(1);

    for case in cases {
        let alpha = case.encoded_alpha ^ score_dependency_key(preceding_score);
        let mut ctx = make_context(&case.board, eval, tt);
        let score = null_window_search(&mut ctx, &case.board, alpha, &mut caches);
        assert_eq!(score, case.expected, "endgame benchmark score mismatch");
        preceding_score = score;
    }
}

fn bench_endgame_chain<const N_EMPTY: u32>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    cases: &[ChainedEndgameCase<N_EMPTY>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    validate_endgame_chain(cases, eval, tt);
    group.throughput(Throughput::Elements(cases.len() as u64));
    group.bench_with_input(
        BenchmarkId::new("alpha_chained_endgame", name),
        cases,
        |b, cases| {
            b.iter_batched_ref(
                || EndgameBenchState {
                    contexts: cases
                        .iter()
                        .map(|case| make_context(&case.board, eval, tt))
                        .collect(),
                    caches: EndGameCaches::for_thread_count(1),
                },
                |state| {
                    let EndgameBenchState { contexts, caches } = state;
                    let mut preceding_score = 0;
                    let mut checksum = 0i32;

                    for (case, ctx) in cases.iter().zip(contexts.iter_mut()) {
                        let alpha = case.encoded_alpha ^ score_dependency_key(preceding_score);
                        let score = null_window_search(
                            black_box(ctx),
                            black_box(&case.board),
                            black_box(alpha),
                            black_box(&mut *caches),
                        );
                        debug_assert_eq!(score, case.expected);
                        checksum = checksum.rotate_left(5) ^ score;
                        preceding_score = score;
                    }

                    black_box(checksum)
                },
                BatchSize::SmallInput,
            )
        },
    );
}

fn flip_consumer_benchmark(c: &mut Criterion) {
    let chains = move_chains(CORPUS_SEED);
    let total_steps: u64 = chains.iter().map(|chain| chain.steps.len() as u64).sum();
    let single_flip_cases = flip_cases(&chains);

    bench_single_flip(c, &single_flip_cases);

    let cases2 = chained_endgame_cases(&endgame_cases::<2>(CORPUS_SEED));
    let cases3 = chained_endgame_cases(&endgame_cases::<3>(CORPUS_SEED));
    let cases4 = chained_endgame_cases(&endgame_cases::<4>(CORPUS_SEED));

    let eval = Arc::new(
        Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
    );
    let tt = Arc::new(TranspositionTable::new(0));

    let mut group = c.benchmark_group("flip/consumer");
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(750));
    group.measurement_time(Duration::from_secs(4));

    group.throughput(Throughput::Elements(total_steps));
    group.bench_with_input(
        BenchmarkId::new("make_move_chain", "problem_playouts"),
        &chains,
        |b, chains| b.iter(|| replay_move_chains(black_box(chains))),
    );

    bench_endgame_chain(&mut group, "2_empty", &cases2, &eval, &tt);
    bench_endgame_chain(&mut group, "3_empty", &cases3, &eval, &tt);
    bench_endgame_chain(&mut group, "4_empty", &cases4, &eval, &tt);

    group.finish();
}

criterion_group!(benches, flip_consumer_benchmark);
criterion_main!(benches);
