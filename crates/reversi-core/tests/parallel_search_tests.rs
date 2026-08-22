use std::sync::OnceLock;
use std::time::{Duration, Instant};

use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::level::Level;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::result::SearchResult;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{Search, SearchRunOptions, SearchSharedResources};

const BOARD_20_EMPTIES: &str = "-XXXXX-----OXX---OOOOOO-XXOOXOO-XXXOXOO-XXXXXOOO--XXXO----OXXO--";
const BOARD_15_EMPTIES: &str = "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-";
const BOARD_9_EMPTIES: &str = "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX";
const TEST_TT_MB_SIZE: usize = 8;

fn search(n_threads: usize) -> Search {
    static SINGLE: OnceLock<SearchSharedResources> = OnceLock::new();
    static DOUBLE: OnceLock<SearchSharedResources> = OnceLock::new();
    static QUAD: OnceLock<SearchSharedResources> = OnceLock::new();
    let shared = match n_threads {
        1 => &SINGLE,
        2 => &DOUBLE,
        4 => &QUAD,
        _ => panic!("unsupported test thread count: {n_threads}"),
    };
    Search::from_shared_resources(shared.get_or_init(|| {
        SearchSharedResources::new(
            &SearchOptions::new(TEST_TT_MB_SIZE).with_threads(Some(n_threads)),
        )
    }))
}

fn score(result: &SearchResult) -> i32 {
    result.score().expect("expected best move") as i32
}

/// Solves `board_str` with `n_threads` and asserts the exact score.
fn assert_parallel_solve(board_str: &str, side: Disc, n_threads: usize, expected: i32) {
    let mut search = search(n_threads);
    let board = Board::from_string(board_str, side).unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());
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
    let mut search = search(4);
    let board = Board::from_string(BOARD_15_EMPTIES, Disc::Black).unwrap();
    let options = SearchRunOptions::with_level(Level::perfect());

    // The second run hits a warm transposition table.
    for run in 0..2 {
        let result = search.run(&board, &options);
        assert_eq!(score(&result), 8, "run={run}");
    }
}

#[test]
fn timed_search_terminates_within_deadline_margin() {
    let mut search = search(4);
    let board = Board::new();
    let options = SearchRunOptions::with_time(TimeControlMode::Byoyomi {
        time_per_move_ms: 500,
    });

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

/// Full-width midgame search must return the same score for any thread count:
/// with ProbCut disabled, LMR is disabled too, and aspiration windows
/// re-search until the score is strictly inside the window, so the root score
/// at the final depth is the exact minimax value.
#[test]
fn parallel_midgame_score_matches_single_threaded() {
    let level = Level::uniform(8, 0);
    let options = SearchRunOptions::with_level(level).disable_probcut();
    let board = Board::new();

    let mut single = search(1);
    let expected = single
        .run(&board, &options)
        .score()
        .expect("expected best move");

    let mut parallel = search(4);
    let actual = parallel
        .run(&board, &options)
        .score()
        .expect("expected best move");

    assert_eq!(actual, expected);
}

/// Covers the GUI's manual abort path (`ThreadPool::abort_search`), which the
/// timer-based test does not exercise. A single abort must not be lost; this
/// pins the regression that previously required re-firing every 100 ms to
/// paper over the start/reset race.
#[test]
fn manual_abort_stops_deep_solve_promptly() {
    let mut search = search(4);
    let pool = search.thread_pool();
    let board = Board::new();
    let options = SearchRunOptions::with_level(Level::perfect());

    let aborter = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        pool.abort_search();
    });

    let start = Instant::now();
    let result = search.run(&board, &options);
    let elapsed = start.elapsed();
    aborter.join().unwrap();

    // Generous margin: this guards against a hung abort, not abort latency.
    assert!(elapsed < Duration::from_secs(5), "abort took {elapsed:?}");
    let best = result.best_move().expect("expected fallback best move");
    assert!(board.is_legal_move(best));
}

/// Guards against shutdown deadlocks or state leaking across pool lifetimes.
#[test]
fn repeated_pool_create_solve_drop_is_clean() {
    for run in 0..3 {
        let mut search = search(4);
        let board = Board::from_string(BOARD_15_EMPTIES, Disc::Black).unwrap();
        let options = SearchRunOptions::with_level(Level::perfect());
        let result = search.run(&board, &options);
        assert_eq!(score(&result), 8, "run={run}");
    }
}

/// Two engines from one `SearchSharedResources` share only the evaluation
/// network; concurrent solves must both stay exact.
#[test]
fn concurrent_engines_from_shared_resources_solve_correctly() {
    let options = SearchRunOptions::with_level(Level::perfect());

    std::thread::scope(|s| {
        let first = s.spawn(|| {
            let mut search = search(2);
            let board = Board::from_string(BOARD_20_EMPTIES, Disc::Black).unwrap();
            score(&search.run(&board, &options))
        });
        let second = s.spawn(|| {
            let mut search = search(2);
            let board = Board::from_string(BOARD_15_EMPTIES, Disc::Black).unwrap();
            score(&search.run(&board, &options))
        });
        assert_eq!(first.join().unwrap(), 6);
        assert_eq!(second.join().unwrap(), 8);
    });
}
