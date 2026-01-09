use reversi_core::board::Board;
use reversi_core::level::Level;
use reversi_core::piece::Piece;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::Search;
use reversi_core::types::Selectivity;

#[test]
fn test_solve_5() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--O--O----OOOOO-XOOOOOOOXXOOXOOOXXXXXOXXXOXXOOXXXXXXOXOXXOOOOOOX",
        Piece::Black,
    );
    let result = search.test(&board, Level::perfect(), Selectivity::None);

    assert_eq!(result.score as i32, 28);
}

#[test]
fn test_solve_15() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-",
        Piece::Black,
    );
    let result = search.test(&board, Level::perfect(), Selectivity::None);

    assert_eq!(result.score as i32, 8);
}
