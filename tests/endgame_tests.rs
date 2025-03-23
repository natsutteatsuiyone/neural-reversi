use reversi::engine::board::Board;
use reversi::engine::piece::Piece;
use reversi::engine::search::SearchOptions;
use reversi::engine::{self, search};

#[test]
fn test_solve_5() {
    engine::init();
    let mut search = search::Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--O--O----OOOOO-XOOOOOOOXXOOXOOOXXXXXOXXXOXXOOXXXXXXOXOXXOOOOOOX",
        Piece::Black,
    );
    let result = search.solve(&board);

    assert_eq!(result.score, 28);
}

#[test]
fn test_solve_15() {
    engine::init();
    let mut search = search::Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-",
        Piece::Black,
    );
    let result = search.solve(&board);

    assert_eq!(result.score, 8);
}
