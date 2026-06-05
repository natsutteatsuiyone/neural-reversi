use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::board::Board;
use reversi_core::eval::Eval;
use reversi_core::obf::ObfPosition;
use reversi_core::probcut::Selectivity;
use reversi_core::search::search_context::SearchContext;
use reversi_core::search::{EndGameCaches, null_window_search};
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::types::Score;

const REALISTIC_ENDGAME_SEED: u64 = 0x5012_0002;
const REALISTIC_SOLVE_TARGET_CASES: usize = 4096;
const REALISTIC_SOLVE_CASE_NAME: &str = "legal_playout";
const REALISTIC_SOLVE_BENCH_NAME: &str = "legal_playouts_4096";
const REALISTIC_CACHED_SEARCH_TARGET_CASES: usize = 256;
const REALISTIC_CACHED_SEARCH_CASE_NAME: &str = "legal_playout";
const REALISTIC_CACHED_SEARCH_BENCH_NAME: &str = "legal_playouts_256_mixed_window";
const REALISTIC_ENDGAME_SOURCES: &[&str] = &[
    include_str!("../../../problem/fforum-1-19.obf"),
    include_str!("../../../problem/fforum-20-39.obf"),
    include_str!("../../../problem/fforum-40-59.obf"),
    include_str!("../../../problem/fforum-60-79.obf"),
    include_str!("../../../problem/hard-20.obf"),
    include_str!("../../../problem/hard-25.obf"),
    include_str!("../../../problem/hard-30.obf"),
    include_str!("../../../problem/small-35.txt"),
];
struct EndgameCase<const N_EMPTY: u32> {
    name: &'static str,
    board: Board,
    alpha: Score,
    expected: Score,
}
impl<const N_EMPTY: u32> EndgameCase<N_EMPTY> {
    fn from_board(name: &'static str, board: Board, alpha: Score, expected: Score) -> Self {
        assert_eq!(
            board.get_empty_count(),
            N_EMPTY,
            "benchmark case {name} must have exactly {N_EMPTY} empties"
        );

        Self {
            name,
            board,
            alpha,
            expected,
        }
    }
}

// Aggregate endgame benchmarks use deterministic legal playouts from real
// problem files, without symmetry expansion, so the corpus stays close to
// positions the search can actually create.
fn realistic_solve_cases<const N_EMPTY: u32>() -> Vec<EndgameCase<N_EMPTY>> {
    let cases = legal_playout_cases(
        REALISTIC_SOLVE_TARGET_CASES,
        REALISTIC_ENDGAME_SEED ^ N_EMPTY as u64,
        REALISTIC_SOLVE_CASE_NAME,
        fail_high_alpha,
    );

    let branch_mix = solve_branch_mix(&cases);
    assert_eq!(
        branch_mix.total(),
        cases.len(),
        "realistic solve branch classifier must account for every case"
    );
    assert!(
        branch_mix.has_core_paths(),
        "realistic solve benchmark must include one-move, multi-move, and pass branches: {branch_mix:?}"
    );

    cases
}

fn realistic_cached_search_cases<const N_EMPTY: u32>() -> Vec<EndgameCase<N_EMPTY>> {
    legal_playout_cases(
        REALISTIC_CACHED_SEARCH_TARGET_CASES,
        (REALISTIC_ENDGAME_SEED ^ 0xca_c4_ed) ^ N_EMPTY as u64,
        REALISTIC_CACHED_SEARCH_CASE_NAME,
        mixed_window_alpha,
    )
}

fn legal_playout_cases<const N_EMPTY: u32>(
    target_cases: usize,
    seed: u64,
    case_name: &'static str,
    alpha_for: fn(usize, Score) -> Score,
) -> Vec<EndgameCase<N_EMPTY>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cases = Vec::with_capacity(target_cases);

    while cases.len() < target_cases {
        let len_before_pass = cases.len();
        for source in REALISTIC_ENDGAME_SOURCES {
            for line in source.lines() {
                let Some(position) =
                    ObfPosition::parse(line).expect("benchmark OBF line must parse")
                else {
                    continue;
                };
                let Some(board) = playout_to_n_empty::<N_EMPTY>(position.board, &mut rng) else {
                    continue;
                };
                let expected = exact_endgame_score(&board);

                cases.push(EndgameCase::from_board(
                    case_name,
                    board,
                    alpha_for(cases.len(), expected),
                    expected,
                ));
                if cases.len() >= target_cases {
                    break;
                }
            }
            if cases.len() >= target_cases {
                break;
            }
        }
        assert!(
            cases.len() > len_before_pass,
            "realistic endgame sources must produce at least one {N_EMPTY}-empty playout"
        );
    }

    assert_eq!(
        cases.len(),
        target_cases,
        "realistic endgame benchmark corpus must stay large enough"
    );
    cases
}

fn fail_high_alpha(_: usize, expected: Score) -> Score {
    expected - 1
}

fn mixed_window_alpha(index: usize, expected: Score) -> Score {
    if index.is_multiple_of(2) {
        expected - 1
    } else {
        expected
    }
}

#[derive(Debug, Default)]
struct SolveBranchMix {
    one_player_move: usize,
    multiple_player_moves: usize,
    player_pass: usize,
    both_pass: usize,
}

impl SolveBranchMix {
    fn total(&self) -> usize {
        self.one_player_move + self.multiple_player_moves + self.player_pass + self.both_pass
    }

    fn has_core_paths(&self) -> bool {
        self.one_player_move > 0 && self.multiple_player_moves > 0 && self.player_pass > 0
    }
}

fn solve_branch_mix<const N_EMPTY: u32>(cases: &[EndgameCase<N_EMPTY>]) -> SolveBranchMix {
    let mut mix = SolveBranchMix::default();

    for case in cases {
        match case.board.get_moves().count() {
            0 => {
                if case.board.switch_players().get_moves().is_empty() {
                    mix.both_pass += 1;
                } else {
                    mix.player_pass += 1;
                }
            }
            1 => mix.one_player_move += 1,
            _ => mix.multiple_player_moves += 1,
        }
    }

    mix
}

fn playout_to_n_empty<const N_EMPTY: u32>(mut board: Board, rng: &mut StdRng) -> Option<Board> {
    while board.get_empty_count() > N_EMPTY {
        let moves = board.get_moves();
        if moves.is_empty() {
            let passed = board.switch_players();
            if passed.get_moves().is_empty() {
                return None;
            }
            board = passed;
            continue;
        }

        let move_index = rng.random_range(0..moves.count()) as usize;
        let sq = moves
            .iter()
            .nth(move_index)
            .expect("random move index must be inside legal move bitboard");
        board = board.make_move(sq);
    }

    (board.get_empty_count() == N_EMPTY).then_some(board)
}

fn exact_endgame_score(board: &Board) -> Score {
    let moves = board.get_moves();
    if !moves.is_empty() {
        return moves
            .iter()
            .map(|sq| -exact_endgame_score(&board.make_move(sq)))
            .max()
            .expect("non-empty move bitboard must yield at least one move");
    }

    let passed = board.switch_players();
    if !passed.get_moves().is_empty() {
        return -exact_endgame_score(&passed);
    }

    board.solve(board.get_empty_count())
}

fn make_context(board: &Board, eval: &Arc<Eval>, tt: &Arc<TranspositionTable>) -> SearchContext {
    SearchContext::new(board, Selectivity::None, tt.clone(), eval.clone())
}

fn assert_expected<const N_EMPTY: u32>(
    case: &EndgameCase<N_EMPTY>,
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    let mut caches = EndGameCaches::for_thread_count(1);
    let mut ctx = make_context(&case.board, eval, tt);
    assert_eq!(
        null_window_search(&mut ctx, &case.board, case.alpha, &mut caches),
        case.expected,
        "benchmark case {} expected score mismatch",
        case.name
    );
}

fn bench_direct_solver_case_set<const N_EMPTY: u32>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    solver_name: &str,
    case_set_name: &str,
    cases: &[EndgameCase<N_EMPTY>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    for case in cases {
        assert_expected(case, eval, tt);
    }

    let mut caches = EndGameCaches::for_thread_count(1);
    group.bench_with_input(
        BenchmarkId::new(solver_name, case_set_name),
        cases,
        |b, cases| {
            b.iter_batched_ref(
                || {
                    cases
                        .iter()
                        .map(|case| make_context(&case.board, eval, tt))
                        .collect::<Vec<_>>()
                },
                |contexts| {
                    let mut checksum = 0;
                    for (case, ctx) in cases.iter().zip(contexts.iter_mut()) {
                        let score = null_window_search(
                            black_box(ctx),
                            black_box(&case.board),
                            black_box(case.alpha),
                            black_box(&mut caches),
                        );
                        debug_assert_eq!(score, case.expected);
                        checksum ^= score;
                    }
                    black_box(checksum)
                },
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_cached_search_case_set<const N_EMPTY: u32>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    search_name: &str,
    case_set_name: &str,
    cases: &[EndgameCase<N_EMPTY>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    for case in cases {
        assert_expected(case, eval, tt);
    }

    group.bench_with_input(
        BenchmarkId::new(search_name, case_set_name),
        cases,
        |b, cases| {
            b.iter_batched_ref(
                || {
                    cases
                        .iter()
                        .map(|case| {
                            (
                                make_context(&case.board, eval, tt),
                                EndGameCaches::for_thread_count(1),
                            )
                        })
                        .collect::<Vec<_>>()
                },
                |states| {
                    let mut checksum = 0;
                    for (case, (ctx, caches)) in cases.iter().zip(states.iter_mut()) {
                        let score = null_window_search(
                            black_box(ctx),
                            black_box(&case.board),
                            black_box(case.alpha),
                            black_box(caches),
                        );
                        debug_assert_eq!(score, case.expected);
                        checksum ^= score;
                    }
                    black_box(checksum)
                },
                BatchSize::LargeInput,
            );
        },
    );
}

struct DirectSolverCases<'a> {
    realistic_solve2: &'a [EndgameCase<2>],
    realistic_solve3: &'a [EndgameCase<3>],
    realistic_solve4: &'a [EndgameCase<4>],
}

struct CachedSearchCases<'a> {
    cached_5_empty: &'a [EndgameCase<5>],
    cached_6_empty: &'a [EndgameCase<6>],
    cached_9_empty: &'a [EndgameCase<9>],
}

fn bench_direct_solvers(
    c: &mut Criterion,
    cases: DirectSolverCases<'_>,
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    let mut group = c.benchmark_group("endgame::direct_solvers");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(3));

    bench_direct_solver_case_set(
        &mut group,
        "solve2",
        REALISTIC_SOLVE_BENCH_NAME,
        cases.realistic_solve2,
        eval,
        tt,
    );
    bench_direct_solver_case_set(
        &mut group,
        "solve3",
        REALISTIC_SOLVE_BENCH_NAME,
        cases.realistic_solve3,
        eval,
        tt,
    );
    bench_direct_solver_case_set(
        &mut group,
        "solve4",
        REALISTIC_SOLVE_BENCH_NAME,
        cases.realistic_solve4,
        eval,
        tt,
    );
    group.finish();
}

fn bench_cached_search(
    c: &mut Criterion,
    cases: CachedSearchCases<'_>,
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    let mut group = c.benchmark_group("endgame::cached_search");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    bench_cached_search_case_set(
        &mut group,
        "5_empty",
        REALISTIC_CACHED_SEARCH_BENCH_NAME,
        cases.cached_5_empty,
        eval,
        tt,
    );
    bench_cached_search_case_set(
        &mut group,
        "6_empty",
        REALISTIC_CACHED_SEARCH_BENCH_NAME,
        cases.cached_6_empty,
        eval,
        tt,
    );
    bench_cached_search_case_set(
        &mut group,
        "9_empty",
        REALISTIC_CACHED_SEARCH_BENCH_NAME,
        cases.cached_9_empty,
        eval,
        tt,
    );
    group.finish();
}

fn endgame_benchmark(c: &mut Criterion) {
    let eval = Arc::new(
        Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
    );
    let tt = Arc::new(TranspositionTable::new(0));
    let realistic_solve2_cases = realistic_solve_cases::<2>();
    let realistic_solve3_cases = realistic_solve_cases::<3>();
    let realistic_solve4_cases = realistic_solve_cases::<4>();
    let cached_5_empty_cases = realistic_cached_search_cases::<5>();
    let cached_6_empty_cases = realistic_cached_search_cases::<6>();
    let cached_9_empty_cases = realistic_cached_search_cases::<9>();

    bench_direct_solvers(
        c,
        DirectSolverCases {
            realistic_solve2: &realistic_solve2_cases,
            realistic_solve3: &realistic_solve3_cases,
            realistic_solve4: &realistic_solve4_cases,
        },
        &eval,
        &tt,
    );
    bench_cached_search(
        c,
        CachedSearchCases {
            cached_5_empty: &cached_5_empty_cases,
            cached_6_empty: &cached_6_empty_cases,
            cached_9_empty: &cached_9_empty_cases,
        },
        &eval,
        &tt,
    );
}
criterion_group!(benches, endgame_benchmark);
criterion_main!(benches);
