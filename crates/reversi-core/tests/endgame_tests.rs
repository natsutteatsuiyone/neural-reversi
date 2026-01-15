use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::level::Level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::{Search, SearchRunOptions};

#[test]
fn test_solve_5() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--O--O----OOOOO-XOOOOOOOXXOOXOOOXXXXXOXXXOXXOOXXXXXXOXOXXOOOOOOX",
        Disc::Black,
    );
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(result.score as i32, 28);
}

#[test]
fn test_solve_15() {
    let mut search = Search::new(&SearchOptions::default());
    let board = Board::from_string(
        "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-",
        Disc::Black,
    );
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);

    assert_eq!(result.score as i32, 8);
}
