use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use reversi_core::board::Board;
use reversi_core::eval::Eval;
use reversi_core::level::Level;
use reversi_core::obf::ObfPosition;
use reversi_core::player_color::PlayerColor;
use reversi_core::probcut::Selectivity;
use reversi_core::score::{DiscScore, Score};
use reversi_core::search::result::SearchResult;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{Search, SearchRunOptions, SearchSharedResources, StopToken};
use reversi_core::square::Square;
use reversi_core::transposition_table::Bound;

const TEST_TT_MB_SIZE: usize = 8;

fn shared_eval() -> Arc<Eval> {
    static EVAL: OnceLock<Arc<Eval>> = OnceLock::new();
    EVAL.get_or_init(|| {
        Arc::new(
            Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
        )
    })
    .clone()
}

fn search_with_threads(n_threads: usize) -> Search {
    let shared = SearchSharedResources::from_eval(shared_eval(), TEST_TT_MB_SIZE, Some(n_threads));
    Search::from_shared_resources(&shared)
}

fn search() -> Search {
    search_with_threads(1)
}

fn assert_exact_score(result: &SearchResult, discs: i32) {
    let SearchResult::BestMove {
        score,
        is_endgame: true,
        selectivity: Selectivity::None,
        ..
    } = result
    else {
        panic!("expected a completed exact endgame result");
    };
    assert_eq!(score.value(), 256 * discs);
}

#[test]
fn exact_solve_reuses_a_selective_root_entry_across_generation_wrap() {
    let mut engine = search();
    let full = SearchRunOptions::with_level(Level::perfect());
    let weak = SearchRunOptions::with_level(Level::perfect().stopping_at(Selectivity::Level1));
    let warmup = Board::from_string(
        "-OXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-",
        PlayerColor::Black,
    )
    .unwrap();
    for _ in 0..126 {
        engine.run(&warmup, &full);
    }

    let board = Board::from_bitboards(0x0012060f3c000080_u64, 0x04e9b0f0c27efc7f_u64);
    let previous = engine.run(&board, &weak);
    assert!(previous.is_endgame());
    assert_eq!(previous.selectivity(), Selectivity::Level1);
    let stored = engine.tt().lookup(&board, board.hash()).unwrap();
    assert_eq!(stored.bound(), Bound::Exact);
    assert_eq!(stored.selectivity(), Selectivity::Level1);
    assert_eq!(stored.score(), previous.score().unwrap());
    assert_eq!(stored.generation(), 127);

    // Ensure the selective root entry carries a score distinct from the exact +20
    // even when Level1 ProbCut happens to match on this position.
    let probe = engine.tt().probe(&board, board.hash());
    let stale_score = Score::from_disc_diff(DiscScore::new(0));
    engine.tt().store(
        probe.slot(),
        &board,
        stale_score,
        Bound::Exact,
        stored.depth(),
        stored.best_move(),
        Selectivity::Level1,
        true,
        true,
    );
    let seeded = engine.tt().lookup(&board, board.hash()).unwrap();
    assert_eq!(seeded.score(), stale_score);
    assert_eq!(seeded.generation(), 127);

    // Independent directional minimax gives +20 on this board.
    let exact = engine.run(&board, &full);
    assert_exact_score(&exact, 20);
    assert_eq!(engine.tt().generation(), 0);
}

#[test]
fn midgame_root_score_can_seed_an_exact_solve() {
    let mut engine = search();
    let board = Board::from_bitboards(0x0012060f3c000080_u64, 0x04e9b0f0c27efc7f_u64);
    let shallow = engine.run(&board, &SearchRunOptions::with_level(Level::uniform(1, 0)));
    assert!(!shallow.is_endgame());
    let prior = engine.tt().lookup(&board, board.hash()).unwrap();
    assert!(!prior.is_endgame());
    assert_eq!(prior.bound(), Bound::Exact);
    assert_eq!(prior.score(), shallow.score().unwrap());
    assert_ne!(prior.score().value(), 20 * 256);

    let exact = engine.run(&board, &SearchRunOptions::with_level(Level::perfect()));
    assert_exact_score(&exact, 20);
}

fn replay_pv(mut board: Board, best_move: Square, line: &[Square]) -> usize {
    assert_eq!(line.first(), Some(&best_move));
    let mut passes = 0;
    for &square in line {
        if !board.has_legal_moves() {
            board = board.switch_players();
            passes += 1;
        }
        assert!(
            board.is_legal_move(square),
            "illegal move {square} in PV {line:?}"
        );
        board = board.make_move(square);
    }
    passes
}

#[test]
fn multi_pv_lines_replay_across_a_forced_pass() {
    let mut search = search();
    let board = Board::from_bitboards(
        Square::A1.bitboard() | Square::A8.bitboard(),
        Square::B1.bitboard() | Square::B8.bitboard(),
    );
    let options = SearchRunOptions::with_level(Level::perfect()).multi_pv(true);
    let result = search.run(&board, &options);

    assert_exact_score(&result, 64);
    assert_eq!(result.pv_moves().len(), 2);
    for row in result.pv_moves() {
        assert_eq!(row.pv_line.len(), 2);
        assert_eq!(replay_pv(board, row.sq, &row.pv_line), 1);
    }
    assert_eq!(
        replay_pv(board, result.best_move().unwrap(), result.pv_line()),
        1
    );
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

    assert_exact_score(&result, 46);
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
    replay_pv(board, best_move.unwrap(), result.pv_line());
}

#[test]
fn multi_pv_solve_18_reports_each_legal_root_move() {
    let board = Board::from_string(
        "--O-------OOX---OOOXXXO-OOOOXOXXXXXOOXOXXXXXXOOXX-XXXXOX--XXXX--",
        PlayerColor::Black,
    )
    .unwrap();
    let options = SearchRunOptions::with_level(Level::perfect()).multi_pv(true);
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

    let parallel = search_with_threads(2);
    for mut engine in [search(), parallel] {
        let result = engine.run(&board, &options);
        assert_exact_score(&result, 4);
        assert_eq!(result.pv_moves().len(), expected.len());
        for row in result.pv_moves() {
            replay_pv(board, row.sq, &row.pv_line);
        }
        replay_pv(board, result.best_move().unwrap(), result.pv_line());

        let mut pv_moves: Vec<_> = result
            .pv_moves()
            .iter()
            .map(|pv_move| (pv_move.sq, pv_move.score.to_disc_diff().value()))
            .collect();
        assert!(
            pv_moves.windows(2).all(|window| window[0].1 >= window[1].1),
            "multi-pv scores should be non-increasing: {pv_moves:?}"
        );
        pv_moves.sort_by_key(|(sq, _score)| sq.index());
        assert_eq!(pv_moves, expected);
    }
}

#[test]
fn timed_multi_pv_handoff_reports_every_score_after_prior_searches() {
    let ffo9 = include_str!("../../../problem/fforum-1-19.obf")
        .lines()
        .nth(8)
        .unwrap();
    let problem = ObfPosition::parse(ffo9).unwrap().unwrap();
    let mut engine = search();
    let weak = SearchRunOptions::with_level(Level::perfect().stopping_at(Selectivity::Level1))
        .multi_pv(true);
    let warm = engine.run(&problem.board, &weak);
    assert_eq!(warm.selectivity(), Selectivity::Level1);

    let stop = StopToken::new();
    let reports = Arc::new(AtomicUsize::new(0));
    let seen = reports.clone();
    let steps = Selectivity::ENDGAME_LADDER.len();
    let interrupted = SearchRunOptions::with_level(Level::perfect())
        .multi_pv(true)
        .stop_token(stop.clone())
        .callback(move |p| {
            if seen.fetch_add(1, Ordering::Relaxed) == steps {
                assert_eq!(p.probability, Selectivity::Level1.probability());
                stop.stop();
            }
        });
    engine.run(&problem.board, &interrupted);
    // One complete line, one level of the next line, and the final callback.
    assert_eq!(reports.load(Ordering::Relaxed), steps + 2);

    let events = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    let timed = SearchRunOptions::with_time(TimeControlMode::Infinite)
        .multi_pv(true)
        .callback(move |p| capture.lock().unwrap().push((p.depth, p.is_endgame)));
    let result = engine.run(&problem.board, &timed);
    assert!(
        events
            .lock()
            .unwrap()
            .windows(2)
            .any(|w| w == [(15, false), (15, true)])
    );
    assert_exact_score(&result, problem.expected_score().unwrap());

    let rows = result.pv_moves();
    assert_eq!(rows.first().map(|r| r.sq), result.best_move());
    assert!(rows.windows(2).all(|pair| pair[0].score >= pair[1].score));
    assert_eq!(rows.len(), problem.board.legal_moves().count() as usize);
    for sq in problem.board.legal_moves().iter() {
        let row = rows
            .iter()
            .find(|r| r.sq == sq)
            .expect("every legal move has a row");
        assert_eq!(row.score.value(), 256 * problem.score_of(sq).unwrap());
    }
}

#[test]
fn multi_pv_solve_two_open_replay_position() {
    let mut search = search();
    let board = Board::from_bitboards(0xde52_7100_7814_a8a2_u64, 0x21ad_0eff_87eb_175d_u64);
    let options = SearchRunOptions::with_level(Level::perfect()).multi_pv(true);
    let result = search.run(&board, &options);
    assert_exact_score(&result, -4);
    let moves: Vec<_> = result
        .pv_moves()
        .iter()
        .map(|pv_move| (pv_move.sq, pv_move.score.to_disc_diff().value()))
        .collect();

    assert_eq!(moves, [(Square::H6, -4), (Square::G2, -16)]);
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

    assert_exact_score(&result, -32);
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

    assert_exact_score(&result, -20);
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

    assert_exact_score(&result, 44);
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

    assert_exact_score(&result, -38);
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

    assert_exact_score(&result, -28);
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

    assert_exact_score(&result, 32);
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

    assert_exact_score(&result, 20);
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

    assert_exact_score(&result, -52);
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

    assert_exact_score(&result, 32);
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

    assert_exact_score(&result, 28);
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

    assert_exact_score(&result, 8);
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

    assert_exact_score(&result, 6);
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

    assert_exact_score(&result, 0);
}

/// White has no move in either position; Black does.
#[test]
fn pass_position_reports_no_legal_move_from_both_roots() {
    let mut search = search();
    let few_empties = "-OXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-";
    let many_empties = "-OXXXXXX------------------------------------------------XXX-----";

    for (name, board, level) in [
        ("endgame root", few_empties, Level::perfect()),
        ("midgame root", many_empties, Level::uniform(4, 0)),
    ] {
        let board = Board::from_string(board, PlayerColor::White).unwrap();
        assert!(!board.has_legal_moves(), "{name}");
        assert!(board.switch_players().has_legal_moves(), "{name}");

        let result = search.run(&board, &SearchRunOptions::with_level(level));

        assert!(matches!(result, SearchResult::NoLegalMove), "{name}");
    }
}
