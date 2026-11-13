use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use reversi_core::board::Board;
use reversi_core::level::Level;
use reversi_core::player_color::PlayerColor;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::result::SearchResult;
use reversi_core::search::{Search, SearchRunOptions, SearchSharedResources, StopToken};
use reversi_core::square::Square;

const TEST_TT_MB_SIZE: usize = 8;

fn search() -> Search {
    static SHARED: OnceLock<SearchSharedResources> = OnceLock::new();
    Search::from_shared_resources(SHARED.get_or_init(|| {
        SearchSharedResources::new(&SearchOptions::new(TEST_TT_MB_SIZE).with_threads(Some(1)))
    }))
}

fn score(result: &SearchResult) -> i32 {
    result.score().expect("expected best move") as i32
}

#[test]
fn test_solve_2_case1() {
    let mut search = search();
    let board = Board::from_string(
        "XXXXXXXXXXXXXXXXXXOOXOXXXXXXOXXXXXXOXOXXXXOXOXOXXOOOOOOX--OOOOOX",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 46);
}

#[test]
fn aborted_endgame_reports_last_completed_selectivity() {
    let mut search = search();
    let board = Board::from_string(
        "XXXXXXXXXXXXXXXXXXOOXOXXXXXXOXXXXXXOXOXXXXOXOXOXXOOOOOOX--OOOOOX",
        PlayerColor::Black,
    )
    .unwrap();
    let completed = Arc::new(Mutex::new(None));
    let abort_requested = Arc::new(AtomicBool::new(false));
    let callback_completed = completed.clone();
    let callback_abort_requested = abort_requested.clone();
    let stop = StopToken::new();
    let callback_stop = stop.clone();
    let options = SearchRunOptions::with_level(Level::perfect())
        .stop_token(stop)
        .callback(move |progress| {
            if !callback_abort_requested.swap(true, Ordering::SeqCst) {
                *callback_completed.lock().unwrap() = Some((
                    progress.best_move,
                    progress.score,
                    progress.probability,
                    progress.pv_line,
                ));
                callback_stop.stop();
            }
        });

    let result = search.run(&board, &options);
    let (best_move, score, probability, pv_line) = completed
        .lock()
        .unwrap()
        .clone()
        .expect("the first endgame selectivity must complete before abort");

    assert_eq!(result.best_move(), best_move);
    assert_eq!(result.score(), Some(score));
    assert_eq!(result.probability(), probability);
    assert_eq!(result.pv_line(), pv_line);
}

#[test]
fn multi_pv_solve_18_reports_each_legal_root_move() {
    let mut search = search();
    let board = Board::from_string(
        "--O-------OOX---OOOXXXO-OOOOXOXXXXXOOXOXXXXXXOOXX-XXXXOX--XXXX--",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect()).multi_pv(true);
    let result = search.run(&board, &options);

    let mut pv_moves: Vec<_> = result
        .pv_moves()
        .iter()
        .map(|pv_move| (pv_move.sq, pv_move.score as i32))
        .collect();

    assert!(
        pv_moves.windows(2).all(|window| window[0].1 >= window[1].1),
        "multi-pv scores should be non-increasing: {pv_moves:?}"
    );

    pv_moves.sort_by_key(|(sq, _score)| sq.index());
    let mut expected = vec![
        (Square::A2, 4),
        (Square::D1, -20),
        (Square::H3, -20),
        (Square::B1, -30),
        (Square::G2, -30),
        (Square::E1, -30),
        (Square::F2, -34),
        (Square::G8, -34),
        (Square::B2, -36),
        (Square::H2, -38),
    ];
    expected.sort_by_key(|(sq, _score)| sq.index());

    assert_eq!(pv_moves, expected);
}

#[test]
fn test_solve_2_case2() {
    let mut search = search();
    let board = Board::from_string(
        "X-XXXXOXOOOOOOOXOOXXOXOOOOXXXXOOOOOXXOXOOOOOXXXOOOOOOX-OOOOOOOOO",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -32);
}

#[test]
fn test_solve_2_case3() {
    let mut search = search();
    let board = Board::from_string(
        "-OXOOOX-XXXXOOXXXOXOXXXXXOXXXOOXXOOXXOOXXOXOXXOXXXOOOXXXXXXXXXXX",
        PlayerColor::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -20);
}

#[test]
fn test_solve_3_case1() {
    let mut search = search();
    let board = Board::from_string(
        "XXXXXXXXXXXXXXXXXXOOXOXXXXXXOXXXXXXOXOXXXXOXOXOX-OOOOOOX--OOOOOX",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 44);
}

#[test]
fn test_solve_3_case2() {
    let mut search = search();
    let board = Board::from_string(
        "X-XXXXOXOOOOOOOXOOXXOXOOOOXXXXOOOOOXXOXOOOOOXXXOOOOOOX-OOOOOOO-O",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -38);
}

#[test]
fn test_solve_3_case3() {
    let mut search = search();
    let board = Board::from_string(
        "-OXOOO--XXXXOOXXXOXOXXXXXOXXXOOXXOOXXOOXXOXOXXOXXXOOOXXXXXXXXXXX",
        PlayerColor::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -28);
}

#[test]
fn test_solve_4_case1() {
    let mut search = search();
    let board = Board::from_string(
        "XOOOOOO-XXOOOOOOXXXOXOOOXXOOOOOOXXXOOOOOXXOOXOOOXO-OOOOOOOO-XXX-",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 32);
}

#[test]
fn test_solve_4_case2() {
    let mut search = search();
    let board = Board::from_string(
        "XXXXXX-OXXXXXXOOXXXOXOOOXXXXOOOOXXXXOOOOXXXOOOOOXXXXOO-XOOOOOO--",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 20);
}

#[test]
fn test_solve_4_case3() {
    let mut search = search();
    let board = Board::from_string(
        "XXXXXXXXXXOXOOXXXXXXXXOXXXXXXXXOXXXXXXX-XXXXXXX-XXXXXX-XXXXXXOO-",
        PlayerColor::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), -52);
}

#[test]
fn test_solve_5_case1() {
    let mut search = search();
    let board = Board::from_string(
        "-OOOOOO-XXXXXXX---XOXXXOXXXOXXOOXXXXXOXOXXXOOOXOXOOOOXXOXOOOOOOO",
        PlayerColor::White,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 32);
}

#[test]
fn test_solve_9_case2() {
    let mut search = search();
    let board = Board::from_string(
        "--O--O----OOOOO-XOOOOOOOXXOOXOOOXXXXXOXXXOXXOOXXXXXXOXOXXOOOOOOX",
        PlayerColor::Black,
    )
    .unwrap();
    assert_eq!(board.empty_count(), 9);
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 28);
}

#[test]
fn test_solve_16() {
    let mut search = search();
    let board = Board::from_string(
        "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-",
        PlayerColor::Black,
    )
    .unwrap();
    assert_eq!(board.empty_count(), 16);
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 8);
}

#[test]
fn test_solve_20() {
    let mut search = search();
    let board = Board::from_string(
        "-XXXXX-----OXX---OOOOOO-XXOOXOO-XXXOXOO-XXXXXOOO--XXXO----OXXO--",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 6);
}

#[test]
fn test_solve_20_case2() {
    let mut search = search();
    let board = Board::from_string(
        "----------XOO---XOOOOOX-XOOOXXXXXOOXOOXXXXOXOOXXX-OOXOO---OXX-O-",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
    let result = search.run(&board, &options);

    assert_eq!(score(&result), 0);
}
