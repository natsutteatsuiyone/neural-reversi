use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::eval::Eval;
use reversi_core::probcut::Selectivity;
use reversi_core::search::search_context::SearchContext;
use reversi_core::search::{EndGameCaches, null_window_search};
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::types::Score;

struct EndgameCase<const N_EMPTY: u32> {
    name: &'static str,
    board: Board,
    alpha: Score,
    expected: Score,
}

impl<const N_EMPTY: u32> EndgameCase<N_EMPTY> {
    fn new(
        name: &'static str,
        board: &'static str,
        side_to_move: Disc,
        alpha: Score,
        expected: Score,
    ) -> Self {
        let board = Board::from_string(board, side_to_move).expect("benchmark board must parse");
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

fn solve2_cases() -> Vec<EndgameCase<2>> {
    vec![
        EndgameCase::new(
            "case1",
            "XXXXXXXXXXXXXXXXXXOOXOXXXXXXOXXXXXXOXOXXXXOXOXOXXOOOOOOX--OOOOOX",
            Disc::Black,
            45,
            46,
        ),
        EndgameCase::new(
            "case2",
            "X-XXXXOXOOOOOOOXOOXXOXOOOOXXXXOOOOOXXOXOOOOOXXXOOOOOOX-OOOOOOOOO",
            Disc::Black,
            -33,
            -32,
        ),
        EndgameCase::new(
            "case3",
            "-OXOOOX-XXXXOOXXXOXOXXXXXOXXXOOXXOOXXOOXXOXOXXOXXXOOOXXXXXXXXXXX",
            Disc::White,
            -21,
            -20,
        ),
    ]
}

fn solve3_cases() -> Vec<EndgameCase<3>> {
    vec![
        EndgameCase::new(
            "case1",
            "XXXXXXXXXXXXXXXXXXOOXOXXXXXXOXXXXXXOXOXXXXOXOXOX-OOOOOOX--OOOOOX",
            Disc::Black,
            43,
            44,
        ),
        EndgameCase::new(
            "case2",
            "X-XXXXOXOOOOOOOXOOXXOXOOOOXXXXOOOOOXXOXOOOOOXXXOOOOOOX-OOOOOOO-O",
            Disc::Black,
            -39,
            -38,
        ),
        EndgameCase::new(
            "case3",
            "-OXOOO--XXXXOOXXXOXOXXXXXOXXXOOXXOOXXOOXXOXOXXOXXXOOOXXXXXXXXXXX",
            Disc::White,
            -29,
            -28,
        ),
        EndgameCase::new(
            "pass",
            concat!(
                "OOOOOOOO", "OOOOOOOO", "OOX--OOO", "OO-XOOOO", "OOOOOOOO", "OOOOOOOO", "OOOOOOOO",
                "OOOOOOOO"
            ),
            Disc::Black,
            -62,
            -64,
        ),
    ]
}

fn solve4_cases() -> Vec<EndgameCase<4>> {
    vec![
        EndgameCase::new(
            "case1",
            "XOOOOOO-XXOOOOOOXXXOXOOOXXOOOOOOXXXOOOOOXXOOXOOOXO-OOOOOOOO-XXX-",
            Disc::Black,
            31,
            32,
        ),
        EndgameCase::new(
            "case2",
            "XXXXXX-OXXXXXXOOXXXOXOOOXXXXOOOOXXXXOOOOXXXOOOOOXXXXOO-XOOOOOO--",
            Disc::Black,
            19,
            20,
        ),
        EndgameCase::new(
            "case3",
            "XXXXXXXXXXOXOOXXXXXXXXOXXXXXXXXOXXXXXXX-XXXXXXX-XXXXXX-XXXXXXOO-",
            Disc::White,
            -53,
            -52,
        ),
        EndgameCase::new(
            "pass",
            concat!(
                "OOOOOOOO", "OOOOOOOO", "OO---OOO", "OO-XOOOO", "OOOOOOOO", "OOOOOOOO", "OOOOOOOO",
                "OOOOOOOO"
            ),
            Disc::Black,
            -62,
            -64,
        ),
    ]
}

fn shallow5_cases() -> Vec<EndgameCase<5>> {
    vec![
        EndgameCase::new(
            "case1",
            "-OOOOOO-XXXXXXX---XOXXXOXXXOXXOOXXXXXOXOXXXOOOXOXOOOOXXOXOOOOOOO",
            Disc::White,
            31,
            32,
        ),
        EndgameCase::new(
            "case2",
            "XXXXXXXXXX-XOOXXXXXXXXOXXXXXXXXOXXXXXXX-XXXXXXX-XXXXXX-XXXXXXOO-",
            Disc::White,
            -53,
            -42,
        ),
    ]
}

fn shallow6_cases() -> Vec<EndgameCase<6>> {
    vec![
        EndgameCase::new(
            "case1",
            "--XXXX-OXXXXXXOOXXXOXOOOXXXXOOOOXXXXOOOOXXXOOOOOXXXXOO-XOOOOOO--",
            Disc::Black,
            19,
            -26,
        ),
        EndgameCase::new(
            "pass",
            concat!(
                "--OOOOOO", "OOOOOOOO", "OO---OOO", "OO-XOOOO", "OOOOOOOO", "OOOOOOOO", "OOOOOOOO",
                "OOOOOOOO"
            ),
            Disc::Black,
            -62,
            -64,
        ),
    ]
}

fn ec9_cases() -> Vec<EndgameCase<9>> {
    vec![EndgameCase::new(
        "case1",
        "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX",
        Disc::Black,
        49,
        50,
    )]
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

fn bench_direct_solver_cases<const N_EMPTY: u32>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    solver_name: &str,
    cases: &[EndgameCase<N_EMPTY>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    for case in cases {
        assert_expected(case, eval, tt);

        let mut caches = EndGameCaches::for_thread_count(1);
        group.bench_with_input(BenchmarkId::new(solver_name, case.name), case, |b, case| {
            b.iter_batched_ref(
                || make_context(&case.board, eval, tt),
                |ctx| {
                    let score = null_window_search(
                        black_box(ctx),
                        black_box(&case.board),
                        black_box(case.alpha),
                        black_box(&mut caches),
                    );
                    debug_assert_eq!(score, case.expected);
                    black_box(score)
                },
                BatchSize::SmallInput,
            );
        });
    }
}

fn bench_cached_search_cases<const N_EMPTY: u32>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    search_name: &str,
    cases: &[EndgameCase<N_EMPTY>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    for case in cases {
        assert_expected(case, eval, tt);

        group.bench_with_input(BenchmarkId::new(search_name, case.name), case, |b, case| {
            b.iter_batched_ref(
                || {
                    (
                        make_context(&case.board, eval, tt),
                        EndGameCaches::for_thread_count(1),
                    )
                },
                |(ctx, caches)| {
                    let score = null_window_search(
                        black_box(ctx),
                        black_box(&case.board),
                        black_box(case.alpha),
                        black_box(caches),
                    );
                    debug_assert_eq!(score, case.expected);
                    black_box(score)
                },
                BatchSize::LargeInput,
            );
        });
    }
}

fn bench_direct_solvers(
    c: &mut Criterion,
    solve2_cases: &[EndgameCase<2>],
    solve3_cases: &[EndgameCase<3>],
    solve4_cases: &[EndgameCase<4>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    let mut group = c.benchmark_group("endgame::direct_solvers");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(3));

    bench_direct_solver_cases(&mut group, "solve2", solve2_cases, eval, tt);
    bench_direct_solver_cases(&mut group, "solve3", solve3_cases, eval, tt);
    bench_direct_solver_cases(&mut group, "solve4", solve4_cases, eval, tt);
    group.finish();
}

fn bench_cached_search(
    c: &mut Criterion,
    shallow5_cases: &[EndgameCase<5>],
    shallow6_cases: &[EndgameCase<6>],
    ec9_cases: &[EndgameCase<9>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    let mut group = c.benchmark_group("endgame::cached_search");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    bench_cached_search_cases(&mut group, "shallow5", shallow5_cases, eval, tt);
    bench_cached_search_cases(&mut group, "shallow6", shallow6_cases, eval, tt);
    bench_cached_search_cases(&mut group, "ec9", ec9_cases, eval, tt);
    group.finish();
}

fn endgame_benchmark(c: &mut Criterion) {
    let eval = Arc::new(
        Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
    );
    let tt = Arc::new(TranspositionTable::new(0));
    let solve2_cases = solve2_cases();
    let solve3_cases = solve3_cases();
    let solve4_cases = solve4_cases();
    let shallow5_cases = shallow5_cases();
    let shallow6_cases = shallow6_cases();
    let ec9_cases = ec9_cases();

    bench_direct_solvers(c, &solve2_cases, &solve3_cases, &solve4_cases, &eval, &tt);
    bench_cached_search(c, &shallow5_cases, &shallow6_cases, &ec9_cases, &eval, &tt);
}

criterion_group!(benches, endgame_benchmark);
criterion_main!(benches);
