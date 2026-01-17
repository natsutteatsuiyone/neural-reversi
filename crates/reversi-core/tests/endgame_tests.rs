use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::level::Level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::{Search, SearchRunOptions};

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

    assert_eq!(result.score as i32, 46);
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

    assert_eq!(result.score as i32, -32);
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

    assert_eq!(result.score as i32, -20);
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

    assert_eq!(result.score as i32, 44);
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

    assert_eq!(result.score as i32, -38);
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

    assert_eq!(result.score as i32, -28);
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

    assert_eq!(result.score as i32, 32);
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

    assert_eq!(result.score as i32, 20);
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

    assert_eq!(result.score as i32, -52);
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

    assert_eq!(result.score as i32, 32);
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

    assert_eq!(result.score as i32, 28);
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

    assert_eq!(result.score as i32, 50);
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

    assert_eq!(result.score as i32, 8);
}
