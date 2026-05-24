use std::sync::{Arc, OnceLock};

use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::eval::Eval;
use reversi_core::level::Level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::search_context::SearchContext;
use reversi_core::search::search_result::SearchResult;
use reversi_core::search::{EndGameCaches, Search, SearchRunOptions, null_window_search};
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::types::Score;

fn score(result: &SearchResult) -> i32 {
    result.score().expect("expected best move") as i32
}

fn eval() -> Arc<Eval> {
    static EVAL: OnceLock<Arc<Eval>> = OnceLock::new();
    EVAL.get_or_init(|| {
        Arc::new(
            Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
        )
    })
    .clone()
}

fn direct_endgame_score(board: &Board, alpha: Score) -> Score {
    let tt = Arc::new(TranspositionTable::new(0));
    let mut ctx = SearchContext::new(board, Selectivity::None, tt, eval());
    let mut caches = EndGameCaches::for_thread_count(1);

    null_window_search(&mut ctx, board, alpha, &mut caches)
}

fn symmetry_cases(base: Board) -> [(&'static str, Board); 8] {
    [
        ("base", base),
        ("flip_h", base.flip_horizontal()),
        ("flip_v", base.flip_vertical()),
        ("rot90", base.rotate_90_clockwise()),
        ("rot180", base.rotate_180_clockwise()),
        ("rot270", base.rotate_270_clockwise()),
        ("diag_a1h8", base.flip_diag_a1h8()),
        ("diag_a8h1", base.flip_diag_a8h1()),
    ]
}

fn assert_null_window_symmetry_scores<const N_EMPTY: u32>(
    base: Board,
    alpha: Score,
    expected: Score,
) {
    assert_eq!(
        base.get_empty_count(),
        N_EMPTY,
        "base board must have exactly {N_EMPTY} empties"
    );

    for (name, board) in symmetry_cases(base) {
        assert_eq!(
            board.get_empty_count(),
            N_EMPTY,
            "{name} board must have exactly {N_EMPTY} empties"
        );
        assert_eq!(
            direct_endgame_score(&board, alpha),
            expected,
            "{name} symmetry should preserve the pass-position score"
        );
    }
}

#[test]
fn test_solve_2_case1() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "XXXXXXXXXXXXXXXXXXOOXOXXXXXXOXXXXXXOXOXXXXOXOXOXXOOOOOOX--OOOOOX",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 46);
}

#[test]
fn test_solve_2_case2() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "X-XXXXOXOOOOOOOXOOXXOXOOOOXXXXOOOOOXXOXOOOOOXXXOOOOOOX-OOOOOOOOO",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -32);
}

#[test]
fn test_solve_2_case3() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "-OXOOOX-XXXXOOXXXOXOXXXXXOXXXOOXXOOXXOOXXOXOXXOXXXOOOXXXXXXXXXXX",
        Disc::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -20);
}

#[test]
fn test_solve_3_case1() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "XXXXXXXXXXXXXXXXXXOOXOXXXXXXOXXXXXXOXOXXXXOXOXOX-OOOOOOX--OOOOOX",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 44);
}

#[test]
fn test_solve_3_case2() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "X-XXXXOXOOOOOOOXOOXXOXOOOOXXXXOOOOOXXOXOOOOOXXXOOOOOOX-OOOOOOO-O",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -38);
}

#[test]
fn test_solve_3_case3() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "-OXOOO--XXXXOOXXXOXOXXXXXOXXXOOXXOOXXOOXXOXOXXOXXXOOOXXXXXXXXXXX",
        Disc::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -28);
}

#[test]
fn test_solve_3_pass_symmetries() {
    let board = Board::from_string(
        concat!(
            "OOOOOOOO", "OOOOOOOO", "OOX--OOO", "OO-XOOOO", "OOOOOOOO", "OOOOOOOO", "OOOOOOOO",
            "OOOOOOOO"
        ),
        Disc::Black,
    )
    .unwrap();

    assert_null_window_symmetry_scores::<3>(board, -62, -64);
}

#[test]
fn test_solve_4_case1() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "XOOOOOO-XXOOOOOOXXXOXOOOXXOOOOOOXXXOOOOOXXOOXOOOXO-OOOOOOOO-XXX-",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 32);
}

#[test]
fn test_solve_4_case2() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "XXXXXX-OXXXXXXOOXXXOXOOOXXXXOOOOXXXXOOOOXXXOOOOOXXXXOO-XOOOOOO--",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 20);
}

#[test]
fn test_solve_4_case3() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "XXXXXXXXXXOXOOXXXXXXXXOXXXXXXXXOXXXXXXX-XXXXXXX-XXXXXX-XXXXXXOO-",
        Disc::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -52);
}

#[test]
fn test_solve_4_pass_symmetries() {
    let board = Board::from_string(
        concat!(
            "OOOOOOOO", "OOOOOOOO", "OO---OOO", "OO-XOOOO", "OOOOOOOO", "OOOOOOOO", "OOOOOOOO",
            "OOOOOOOO"
        ),
        Disc::Black,
    )
    .unwrap();

    assert_null_window_symmetry_scores::<4>(board, -62, -64);
}

#[test]
fn test_solve_5_case1() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "-OOOOOO-XXXXXXX---XOXXXOXXXOXXOOXXXXXOXOXXXOOOXOXOOOOXXOXOOOOOOO",
        Disc::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 32);
}

#[test]
fn test_solve_5_case2() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--O--O----OOOOO-XOOOOOOOXXOOXOOOXXXXXOXXXOXXOOXXXXXXOXOXXOOOOOOX",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 28);
}

#[test]
fn test_solve_9() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 50);
}

#[test]
fn test_solve_15() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 8);
}

#[test]
fn test_solve_20() {
    let mut search = Search::new(&SearchOptions::default().with_threads(Some(1)));
    let board = Board::from_string(
        "-XXXXX-----OXX---OOOOOO-XXOOXOO-XXXOXOO-XXXXXOOO--XXXO----OXXO--",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 6);
}

#[test]
fn test_solve_20_case2() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "----------XOO---XOOOOOX-XOOOXXXXXOOXOOXXXXOXOOXXX-OOXOO---OXX-O-",
        Disc::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 0);
}
