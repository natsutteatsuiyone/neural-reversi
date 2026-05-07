use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::eval::Eval;
use reversi_core::probcut::Selectivity;
use reversi_core::search::null_window_search;
use reversi_core::search::search_context::SearchContext;
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::types::Score;

struct Solve4Case {
    name: &'static str,
    board: Board,
    alpha: Score,
    expected: Score,
}

impl Solve4Case {
    fn new(
        name: &'static str,
        board: &'static str,
        side_to_move: Disc,
        alpha: Score,
        expected: Score,
    ) -> Self {
        Self {
            name,
            board: Board::from_string(board, side_to_move).expect("benchmark board must parse"),
            alpha,
            expected,
        }
    }
}

fn solve4_cases() -> Vec<Solve4Case> {
    vec![
        Solve4Case::new(
            "case1",
            "XOOOOOO-XXOOOOOOXXXOXOOOXXOOOOOOXXXOOOOOXXOOXOOOXO-OOOOOOOO-XXX-",
            Disc::Black,
            31,
            32,
        ),
        Solve4Case::new(
            "case2",
            "XXXXXX-OXXXXXXOOXXXOXOOOXXXXOOOOXXXXOOOOXXXOOOOOXXXXOO-XOOOOOO--",
            Disc::Black,
            19,
            20,
        ),
        Solve4Case::new(
            "case3",
            "XXXXXXXXXXOXOOXXXXXXXXOXXXXXXXXOXXXXXXX-XXXXXXX-XXXXXX-XXXXXXOO-",
            Disc::White,
            -53,
            -52,
        ),
        Solve4Case::new(
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

fn make_context(board: &Board, eval: &Arc<Eval>, tt: &Arc<TranspositionTable>) -> SearchContext {
    SearchContext::new(board, Selectivity::None, tt.clone(), eval.clone())
}

fn solve4_benchmark(c: &mut Criterion) {
    let eval = Arc::new(
        Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
    );
    let tt = Arc::new(TranspositionTable::new(0));
    let cases = solve4_cases();

    let mut group = c.benchmark_group("endgame::solve4");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(5));

    for case in &cases {
        assert_eq!(
            case.board.get_empty_count(),
            4,
            "solve4 case must have exactly 4 empties"
        );

        let mut ctx = make_context(&case.board, &eval, &tt);
        assert_eq!(
            null_window_search(&mut ctx, &case.board, case.alpha),
            case.expected,
            "benchmark case {} expected score mismatch",
            case.name
        );

        let mut ctx = make_context(&case.board, &eval, &tt);
        group.bench_with_input(BenchmarkId::from_parameter(case.name), case, |b, case| {
            b.iter(|| {
                let score = null_window_search(
                    black_box(&mut ctx),
                    black_box(&case.board),
                    black_box(case.alpha),
                );
                black_box(score)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, solve4_benchmark);
criterion_main!(benches);
