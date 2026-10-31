use std::io::{self, Write};
use std::sync::{OnceLock, mpsc};
use std::time::{Duration, Instant};

use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::level::Level;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::result::SearchResult;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{Search, SearchRunOptions, SearchSharedResources, StopToken};
use reversi_core::square::Square;

const BOARD_20_EMPTIES: &str = "-XXXXX-----OXX---OOOOOO-XXOOXOO-XXXOXOO-XXXXXOOO--XXXO----OXXO--";
const BOARD_16_EMPTIES: &str = "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-";
const BOARD_9_EMPTIES: &str = "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX";
const TEST_TT_MB_SIZE: usize = 8;

// Aborts the whole test process if `run` outlives 30 seconds. Create engines
// before calling it: loading the weights in a debug build can take that long.
fn with_watchdog(name: &'static str, run: impl FnOnce()) {
    let (_done, deadline) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        if deadline.recv_timeout(Duration::from_secs(30)) == Err(mpsc::RecvTimeoutError::Timeout) {
            // libtest captures this thread's `eprintln!`, which abort would discard.
            let _ = writeln!(io::stderr(), "{name} exceeded the 30-second deadline");
            std::process::abort();
        }
    });
    run();
}

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
fn parallel_solve_matches_single_threaded() {
    assert_parallel_solve(BOARD_9_EMPTIES, Disc::Black, 1, 50);
    assert_parallel_solve(BOARD_9_EMPTIES, Disc::Black, 4, 50);
}

#[test]
fn parallel_solve_reused_search_instance_is_consistent() {
    let mut search = search(4);
    let board = Board::from_string(BOARD_16_EMPTIES, Disc::Black).unwrap();
    assert_eq!(board.empty_count(), 16);
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
    with_watchdog(
        "timed_search_terminates_within_deadline_margin",
        move || {
            let board = Board::new().make_move(Square::F5);
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
            assert!(result.n_nodes() > 0);
        },
    );
}

/// Full-width midgame search must return the same score for any thread count:
/// with ProbCut disabled, LMR is disabled too, and aspiration windows
/// re-search until the score is strictly inside the window, so the root score
/// at the final depth is the exact minimax value.
#[test]
fn parallel_midgame_score_matches_single_threaded() {
    const DEPTH: u32 = 8;
    let options = SearchRunOptions::with_level(Level::uniform(DEPTH, 0)).disable_probcut();
    let mut single = search(1);
    let mut parallel = search(4);

    for moves in ["f5", "f5d6", "f5f6", "f5f6e6"] {
        let board = Square::parse_sequence(moves)
            .unwrap()
            .into_iter()
            .fold(Board::new(), |board, sq| board.make_move(sq));

        let expected = single.run(&board, &options);
        let actual = parallel.run(&board, &options);

        for result in [&expected, &actual] {
            assert_eq!(result.depth(), DEPTH, "moves={moves}");
            assert!(result.n_nodes() > 0, "moves={moves}");
        }
        assert_eq!(actual.score(), expected.score(), "moves={moves}");
    }
}

/// An infinite search (GGS ponder) has no time to bank, so a forced move must
/// keep deepening until stopped.
#[test]
fn infinite_search_keeps_deepening_a_forced_move() {
    let mut search = search(4);
    let stop = StopToken::new();
    let callback_stop = stop.clone();
    let board = Square::parse_sequence("d3e3f6c2d2c3f2f3b2g1g2d1e1f1f4h3")
        .unwrap()
        .into_iter()
        .fold(Board::new(), |board, sq| board.make_move(sq));
    assert_eq!(board.get_moves().count(), 1);

    let options = SearchRunOptions::with_time(TimeControlMode::Infinite)
        .stop_token(stop)
        .callback(move |progress| {
            if progress.depth > 4 {
                callback_stop.stop();
            }
        });
    let result = search.run(&board, &options);

    assert!(result.depth() > 4, "stopped at depth {}", result.depth());
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
            let board = Board::from_string(BOARD_16_EMPTIES, Disc::Black).unwrap();
            score(&search.run(&board, &options))
        });
        assert_eq!(first.join().unwrap(), 6);
        assert_eq!(second.join().unwrap(), 8);
    });
}

/// Engines from a `share_thread_pool` bundle reuse one pool but keep their own
/// transposition tables, so searches taking turns on the pool stay exact.
#[test]
fn engines_sharing_a_pool_search_in_turn() {
    let mut shared =
        SearchSharedResources::new(&SearchOptions::new(TEST_TT_MB_SIZE).with_threads(Some(2)));
    shared.share_thread_pool();
    let mut first = Search::from_shared_resources(&shared);
    let mut second = Search::from_shared_resources(&shared);
    assert!(!std::sync::Arc::ptr_eq(first.tt(), second.tt()));

    let options = SearchRunOptions::with_level(Level::perfect());
    let board_20 = Board::from_string(BOARD_20_EMPTIES, Disc::Black).unwrap();
    let board_16 = Board::from_string(BOARD_16_EMPTIES, Disc::Black).unwrap();
    for _ in 0..2 {
        assert_eq!(score(&first.run(&board_20, &options)), 6);
        assert_eq!(score(&second.run(&board_16, &options)), 8);
    }
}
