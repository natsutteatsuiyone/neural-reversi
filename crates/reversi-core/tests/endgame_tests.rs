use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use reversi_core::board::Board;
use reversi_core::eval::Eval;
use reversi_core::level::Level;
use reversi_core::player_color::PlayerColor;
use reversi_core::search::result::SearchResult;
use reversi_core::search::{Search, SearchRunOptions, SearchSharedResources, StopToken};
use reversi_core::square::Square;

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

fn score(result: &SearchResult) -> i32 {
    result
        .score()
        .expect("expected best move")
        .to_disc_diff()
        .value()
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
fn multi_pv_solve_two_open_replay_position() {
    let mut search = search();
    let board = Board::from_bitboards(0xde52_7100_7814_a8a2_u64, 0x21ad_0eff_87eb_175d_u64);
    let options = SearchRunOptions::with_level(Level::perfect()).multi_pv(true);
    let result = search.run(&board, &options);
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
