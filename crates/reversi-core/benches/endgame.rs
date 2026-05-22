use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
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

fn solve4_pass_cases() -> Vec<EndgameCase<4>> {
    let base = Board::from_string(
        concat!(
            "OOOOOOOO", "OOOOOOOO", "OO---OOO", "OO-XOOOO", "OOOOOOOO", "OOOOOOOO", "OOOOOOOO",
            "OOOOOOOO"
        ),
        Disc::Black,
    )
    .expect("benchmark board must parse");
    let cases = [
        ("base", base),
        ("flip_h", base.flip_horizontal()),
        ("flip_v", base.flip_vertical()),
        ("rot90", base.rotate_90_clockwise()),
        ("rot180", base.rotate_180_clockwise()),
        ("rot270", base.rotate_270_clockwise()),
        ("diag_a1h8", base.flip_diag_a1h8()),
        ("diag_a8h1", base.flip_diag_a8h1()),
    ];

    cases
        .into_iter()
        .map(|(name, board)| EndgameCase::from_board(name, board, -62, -64))
        .collect()
}

fn solve3_cases() -> Vec<EndgameCase<3>> {
    vec![
        EndgameCase::new(
            "case1",
            "XXXXXXXXXXOOOOOOXXXOXOOOXXOOOOOOXXXOOOOOXXOOXOOOXO-OOOOOOOO-XXX-",
            Disc::White,
            31,
            -12,
        ),
        EndgameCase::new(
            "case2",
            "XXXXXX-OXXXXXXOOXXXOXOOOXXXXOOOOXXXXOOOOXXXOXOOOXXXXOX-XOOOOOOX-",
            Disc::White,
            19,
            -10,
        ),
        EndgameCase::new(
            "case3",
            "XXXXXXXXXXOXOOXXXXXXXOOXXXXXXXOOXXXXXXXOXXXXXXX-XXXXXX-XXXXXXOO-",
            Disc::Black,
            -53,
            52,
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

fn shallow5_cases() -> Vec<EndgameCase<5>> {
    vec![
        EndgameCase::new(
            "case1",
            "-OOOOOO-XXOOOOOOXXXOXOOOXXOOOOOOXXXOOOOOXXOOXOOOXO-OOOOOOOO-XXX-",
            Disc::Black,
            31,
            -8,
        ),
        EndgameCase::new(
            "case3",
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
            "case2",
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

fn make_context(board: &Board, eval: &Arc<Eval>, tt: &Arc<TranspositionTable>) -> SearchContext {
    SearchContext::new(board, Selectivity::None, tt.clone(), eval.clone())
}

fn bench_cases<const N_EMPTY: u32>(
    c: &mut Criterion,
    group_name: &str,
    cases: &[EndgameCase<N_EMPTY>],
    eval: &Arc<Eval>,
    tt: &Arc<TranspositionTable>,
) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(5));

    for case in cases {
        let mut caches = EndGameCaches::for_thread_count(1);
        let mut ctx = make_context(&case.board, eval, tt);
        assert_eq!(
            null_window_search(&mut ctx, &case.board, case.alpha, &mut caches),
            case.expected,
            "benchmark case {} expected score mismatch",
            case.name
        );

        let mut ctx = make_context(&case.board, eval, tt);
        group.bench_with_input(BenchmarkId::from_parameter(case.name), case, |b, case| {
            b.iter(|| {
                let score = null_window_search(
                    black_box(&mut ctx),
                    black_box(&case.board),
                    black_box(case.alpha),
                    black_box(&mut caches),
                );
                black_box(score)
            });
        });
    }

    group.finish();
}

fn endgame_benchmark(c: &mut Criterion) {
    let eval = Arc::new(
        Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
    );
    let tt = Arc::new(TranspositionTable::new(0));
    let solve3_cases = solve3_cases();
    let solve4_cases = solve4_cases();
    let solve4_pass_cases = solve4_pass_cases();
    let shallow5_cases = shallow5_cases();
    let shallow6_cases = shallow6_cases();

    bench_cases(c, "endgame::solve3", &solve3_cases, &eval, &tt);
    bench_cases(c, "endgame::solve4", &solve4_cases, &eval, &tt);
    bench_cases(c, "endgame::solve4_pass", &solve4_pass_cases, &eval, &tt);
    bench_cases(c, "endgame::shallow_search::5", &shallow5_cases, &eval, &tt);
    bench_cases(c, "endgame::shallow_search::6", &shallow6_cases, &eval, &tt);
}

criterion_group!(benches, endgame_benchmark);
criterion_main!(benches);
