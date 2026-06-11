use std::time::{Duration, Instant};

use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::level::Level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::search_result::SearchResult;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{Search, SearchRunOptions};

const BOARD_20_EMPTIES: &str = "-XXXXX-----OXX---OOOOOO-XXOOXOO-XXXOXOO-XXXXXOOO--XXXO----OXXO--";
const BOARD_15_EMPTIES: &str = "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-";
const BOARD_9_EMPTIES: &str = "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX";

fn score(result: &SearchResult) -> i32 {
    result.score().expect("expected best move") as i32
}

/// Solves `board_str` with `n_threads` and asserts the exact score.
fn assert_parallel_solve(board_str: &str, side: Disc, n_threads: usize, expected: i32) {
    let mut search = Search::new(&SearchOptions::default().with_threads(Some(n_threads)));
    let board = Board::from_string(board_str, side).unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);
    let result = search.run(&board, &options);
    assert_eq!(score(&result), expected, "threads={n_threads}");
}

#[test]
fn parallel_solve_matches_known_score_20_empties() {
    assert_parallel_solve(BOARD_20_EMPTIES, Disc::Black, 4, 6);
}

#[test]
fn parallel_solve_matches_known_score_15_empties() {
    assert_parallel_solve(BOARD_15_EMPTIES, Disc::Black, 4, 8);
}

#[test]
fn parallel_solve_matches_single_threaded() {
    assert_parallel_solve(BOARD_9_EMPTIES, Disc::Black, 1, 50);
    assert_parallel_solve(BOARD_9_EMPTIES, Disc::Black, 4, 50);
}

#[test]
fn parallel_solve_reused_search_instance_is_consistent() {
    let mut search = Search::new(&SearchOptions::default().with_threads(Some(4)));
    let board = Board::from_string(BOARD_15_EMPTIES, Disc::Black).unwrap();
    let options = SearchRunOptions::with_level(Level::perfect(), Selectivity::None);

    // The second run hits a warm transposition table.
    for run in 0..2 {
        let result = search.run(&board, &options);
        assert_eq!(score(&result), 8, "run={run}");
    }
}

#[test]
fn timed_search_terminates_within_deadline_margin() {
    let mut search = Search::new(&SearchOptions::default().with_threads(Some(4)));
    let board = Board::new();
    let options = SearchRunOptions::with_time(
        TimeControlMode::Byoyomi {
            time_per_move_ms: 500,
        },
        Selectivity::None,
    );

    let start = Instant::now();
    let result = search.run(&board, &options);
    let elapsed = start.elapsed();

    // Generous margin: this guards against a hung abort/timer, not
    // time-management precision.
    assert!(
        elapsed < Duration::from_secs(5),
        "timed search took {elapsed:?}"
    );
    assert!(result.score().is_some(), "expected a best move");
}
