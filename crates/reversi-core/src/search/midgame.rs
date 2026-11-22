//! Midgame search algorithms.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/midgame.c>

mod lazy_smp;

use std::hash::{BuildHasher, Hasher, RandomState};
use std::sync::Arc;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::constants::INITIAL_EMPTY_COUNT;
use crate::eval::Eval;
use crate::flip;
use crate::move_list::MoveList;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::score::Score;
use crate::search::aspiration::{ASPIRATION_MIN_DEPTH, midgame_window, retry_midgame};
use crate::search::context::SearchContext;
use crate::search::endgame;
use crate::search::node_type::{NonPV, Root};
use crate::search::result::{CompletedState, SearchResult};
use crate::search::root_move::RootMoves;
use crate::search::strategy::MidGameStrategy;
use crate::search::threading::Thread;
use crate::search::time_control::{record_abort, should_stop_iteration};
use crate::search::{SearchProgress, SearchTask, search};
use crate::square::Square;
use crate::transposition_table::Bound;
use crate::types::Depth;

pub(in crate::search) use lazy_smp::Helper;
use lazy_smp::{Helpers, vote_best};

/// Depth threshold for switching iteration step from +2 to +1.
const DEPTH_STEP_THRESHOLD: Depth = 10;
/// Square-value tiers (18, 16, ..., 0) used to search depth-1 moves from strongest to weakest.
const DEPTH1_MOVE_ORDER_MASKS: [Bitboard; 10] = [
    Bitboard::new(0x8100_0000_0000_0081),
    Bitboard::new(0x2400_8100_0081_0024),
    Bitboard::new(0x0000_2400_0024_0000),
    Bitboard::new(0x1800_0081_8100_0018),
    Bitboard::new(0x0000_1824_2418_0000),
    Bitboard::new(0x0018_0042_4200_1800),
    Bitboard::new(0x0024_4200_0042_2400),
    Bitboard::new(0x4281_0000_0000_8142),
    Bitboard::new(0x0042_0000_0000_4200),
    Bitboard::new(0x0000_0018_1800_0000),
];

/// Performs the root search using iterative deepening with aspiration windows.
///
/// The other threads of a parallel pool run Lazy SMP helper searches on the
/// same root until this search returns or hands the root to the endgame
/// solver, whose split points then share the work among the same threads.
pub fn search_root(task: SearchTask, max_depth: Depth, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let root_moves = RootMoves::new(&board);

    let mut ctx = SearchContext::new(
        &board,
        task.mid_selectivity,
        &root_moves,
        &task.tt,
        &task.eval,
    );

    if let Some(mode) = task.eval_mode {
        ctx.eval_mode = mode;
    }

    if ctx.root_moves.count() == 0 {
        return SearchResult::NoLegalMove;
    }

    if board == Board::new() && !task.multi_pv {
        return SearchResult::new_random_move(random_move(&board));
    }

    let n_empties = ctx.empty_list.count();

    let max_depth = max_depth.max(1).min(n_empties);
    ctx.correction_history = Some(task.corrections.take(thread.idx()));
    let helpers = Helpers::start(&task, max_depth);
    let outcome = iterate(&mut ctx, &board, &task, max_depth, &helpers, thread);
    let helper_results = helpers.stop();
    if let Some(history) = ctx.correction_history.take() {
        task.corrections.put(thread.idx(), history);
    }
    ctx.merge_counters(&helper_results.counters);

    match outcome {
        Outcome::Done(mut completed) => {
            // A fixed-level search reports the depth it was asked for and a
            // Multi-PV search every line, so both keep the main thread's
            // iteration; a timed search lets the helpers' iterations vote.
            if !task.multi_pv && task.time_manager.is_some() {
                completed.last = vote_best(completed.last.take(), helper_results.completed);
            }
            completed.into_result(&ctx, &board)
        }
        Outcome::Endgame(completed, base_score) => endgame::solve_root(
            &mut ctx,
            &task,
            thread,
            base_score,
            completed,
            super::ENDGAME_TRACKING_MIN_SELECTIVITY,
            Selectivity::None,
        ),
    }
}

/// How the main thread's iterative deepening ended.
enum Outcome {
    /// Report the last completed iteration.
    Done(CompletedState),
    /// Hand off to the endgame selectivity ladder, centered on the score.
    Endgame(CompletedState, Score),
}

/// Runs the main thread's iterative deepening up to `max_depth`.
fn iterate(
    ctx: &mut SearchContext,
    board: &Board,
    task: &SearchTask,
    max_depth: Depth,
    helpers: &Helpers<'_>,
    thread: &Arc<Thread>,
) -> Outcome {
    let time_manager = task.time_manager.as_deref();
    let n_empties = ctx.empty_list.count();
    let pv_count = task.pv_count(ctx.root_moves.count());

    let mut depth = compute_start_depth(max_depth);
    let mut completed = CompletedState::new(pv_count);
    while depth <= max_depth {
        let mut iteration_window_clean = false;
        let mut best = None;
        for pv_idx in 0..pv_count {
            ctx.root_moves.set_pv_idx(pv_idx);

            let (mut alpha, mut beta, delta) = aspiration_window(ctx, depth);
            let (score, window_clean) =
                aspiration_search(ctx, board, depth, &mut alpha, &mut beta, delta, thread);

            ctx.root_moves.sort_from_pv_idx();

            if thread.is_search_aborted() {
                break;
            }

            if let Some(ref callback) = task.callback
                && let Some(rm) = ctx.root_moves.current_pv()
            {
                let nodes = ctx.n_nodes + helpers.nodes();
                callback(SearchProgress::from_iteration(
                    ctx, &rm, depth, max_depth, score, nodes, false,
                ));
            }

            iteration_window_clean = window_clean;
            best = completed
                .line_completed(ctx, depth, false)
                .map(|rm| (rm.sq, rm.score));
            if thread.is_search_aborted() {
                break;
            }
        }

        let Some((best_sq, best_score)) = best else {
            record_abort(time_manager);
            return Outcome::Done(completed);
        };
        ctx.root_moves.sort_all();

        if let Some(tm) = time_manager {
            let window_clean =
                !task.multi_pv && depth >= ASPIRATION_MIN_DEPTH && iteration_window_clean;
            tm.report_iteration(best_sq, best_score.to_disc_diff_f32(), depth, window_clean);
        }

        // A forced move needs no deep search under a deadline: answer after the
        // first completed iteration and bank the time. Score and PV are already
        // available for reporting.
        if ctx.root_moves.count() == 1 && time_manager.is_some_and(|tm| tm.has_deadline()) {
            return Outcome::Done(completed);
        }

        if thread.is_search_aborted() {
            return Outcome::Done(completed);
        }

        if let Some(tm) = time_manager
            && depth >= max_depth
        {
            debug_assert_eq!(
                max_depth, n_empties,
                "time-controlled midgame searches always target all empties"
            );
            if tm.should_enter_endgame(ctx.n_nodes + helpers.nodes()) {
                return Outcome::Endgame(completed, best_score);
            }
            return Outcome::Done(completed);
        }

        if should_stop_iteration(time_manager) {
            return Outcome::Done(completed);
        }

        depth = next_iteration_depth(depth);
    }

    Outcome::Done(completed)
}

/// Returns the aspiration window `(alpha, beta, delta)` for the current PV
/// line, centered on its running average score once deep enough.
fn aspiration_window(ctx: &SearchContext, depth: Depth) -> (Score, Score, Score) {
    let history = ctx
        .root_moves
        .current_pv()
        .map(|rm| (rm.average_score, rm.mean_squared_score));
    midgame_window(depth, history, ctx.helper.map_or(0, Helper::index))
}

/// Computes the starting depth for iterative deepening.
pub(super) fn compute_start_depth(max_depth: Depth) -> Depth {
    if max_depth.is_multiple_of(2) { 2 } else { 1 }
}

/// Performs aspiration window search at the given depth.
///
/// Returns the score and whether the search completed inside its initial
/// window (i.e. without any fail-low/fail-high re-search). A clean window is
/// evidence that the best score is stable across iterations.
fn aspiration_search(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    alpha: &mut Score,
    beta: &mut Score,
    delta: Score,
    thread: &Arc<Thread>,
) -> (Score, bool) {
    retry_midgame(alpha, beta, delta, |alpha, beta| {
        let score = search::<Root, MidGameStrategy>(ctx, board, depth, alpha, beta, thread, false);
        (score, thread.is_search_aborted())
    })
}

/// Computes the next iteration depth.
fn next_iteration_depth(current_depth: Depth) -> Depth {
    if current_depth <= DEPTH_STEP_THRESHOLD {
        current_depth + 2
    } else {
        current_depth + 1
    }
}

/// Selects a random legal move from the current position.
fn random_move(board: &Board) -> Square {
    let moves = board.legal_moves();
    let index = RandomState::new().build_hasher().finish() % u64::from(moves.count());
    moves.iter().nth(index as usize).unwrap()
}

/// Attempts ProbCut pruning for midgame positions.
///
/// `eval_score` is the corrected static evaluation of `board`.
pub fn try_probcut(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    beta: Score,
    cut_node: bool,
    eval_score: Score,
    thread: &Arc<Thread>,
) -> Option<Score> {
    if !beta.is_finite() {
        return None;
    }
    let ply = ctx.ply();
    let pc_depth = probcut::probcut_depth(depth);
    let pc = probcut::midgame_coefficients(ply, depth);
    let t = ctx.selectivity.t_value();

    let pc_beta = probcut::compute_probcut_beta(beta, t, pc).max(-Score::INF + 1);
    if pc_beta >= Score::MAX {
        return None;
    }

    let eval_beta = probcut::compute_eval_beta(beta, t, pc, cut_node);

    if eval_score < eval_beta {
        return None;
    }

    if pc_depth == 0 {
        return (eval_score >= pc_beta).then_some(beta);
    }

    let current_selectivity = ctx.selectivity;
    ctx.selectivity = Selectivity::None; // Disable nested probcut
    let score =
        search::<NonPV, MidGameStrategy>(ctx, board, pc_depth, pc_beta - 1, pc_beta, thread, true);
    ctx.selectivity = current_selectivity;

    if score >= pc_beta {
        return Some(beta);
    }

    None
}

/// Searches positions at depth 2 or 3 for the midgame search, with the
/// transposition table.
pub fn evaluate_shallow_with_tt(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: Score,
    beta: Score,
) -> Score {
    debug_assert!(depth == 2 || depth == 3);

    let Some(moves) = board.legal_moves().non_empty() else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_shallow_with_tt(ctx, &next, depth, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve(ctx.empty_list.count());
        }
    };

    let org_alpha = alpha;
    let tt_probe_result = ctx.tt.probe(board, board.hash());
    let tt_move = tt_probe_result.best_move();

    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.can_cut(beta, depth, Selectivity::None, false)
    {
        return tt_data.score();
    }

    let mut move_list = MoveList::with_at_least_one_move(board, moves);
    if move_list.wipeout_move().is_some() {
        return Score::MAX;
    }

    if move_list.count() >= 2 {
        move_list.evaluate_moves_fast(ctx, board, tt_move);
    }

    let mut best_score = -Score::INF;
    let mut best_move = None;
    for mv in move_list.best_first_iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);

        ctx.update(mv.sq, mv.flipped);
        let score = if depth == 2 {
            -evaluate_depth1(ctx, &next, -beta, -alpha)
        } else {
            -evaluate_shallow_with_tt(ctx, &next, depth - 1, -beta, -alpha)
        };
        ctx.undo(mv.sq);

        if score >= beta {
            best_score = score;
            best_move = Some(mv.sq);
            break;
        }
        best_score = best_score.max(score);
        if score > alpha {
            best_move = Some(mv.sq);
            alpha = score;
        }
    }

    ctx.tt.store(
        tt_probe_result.slot(),
        board,
        best_score,
        Bound::classify::<NonPV>(best_score, org_alpha, beta),
        depth,
        best_move,
        Selectivity::None,
        false,
        tt_probe_result.data().is_some_and(|data| data.is_pv()),
    );

    best_score
}

/// Searches positions at depth 2 with a specialized alpha-beta routine.
pub fn evaluate_depth2(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: Score,
    beta: Score,
) -> Score {
    let Some(moves) = board.legal_moves().non_empty() else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve(ctx.empty_list.count());
        }
    };

    let mut move_list = MoveList::with_at_least_one_move(board, moves);
    if move_list.wipeout_move().is_some() {
        return Score::MAX;
    }
    if move_list.count() >= 2 {
        move_list.evaluate_moves_fast(ctx, board, None);
    }

    let mut best_score = -Score::INF;
    for mv in move_list.best_first_iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);

        ctx.update(mv.sq, mv.flipped);
        let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
        ctx.undo(mv.sq);

        if score >= beta {
            best_score = score;
            break;
        }
        best_score = best_score.max(score);
        alpha = alpha.max(score);
    }

    best_score
}

/// Searches positions at depth 1 with a specialized alpha-beta routine.
pub fn evaluate_depth1(ctx: &mut SearchContext, board: &Board, alpha: Score, beta: Score) -> Score {
    let moves = board.legal_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve(ctx.empty_list.count());
        }
    }

    let mut best_score = -Score::INF;

    for &mask in &DEPTH1_MOVE_ORDER_MASKS {
        for sq in (moves & mask).iter() {
            let flipped = flip::flip(sq, board.player(), board.opponent());
            if flipped == board.opponent() {
                return Score::MAX;
            }
            let next = board.make_move_with_flipped(flipped, sq);

            ctx.update(sq, flipped);
            let score = -evaluate(ctx, &next);
            ctx.undo(sq);

            if score >= beta {
                return score;
            }
            best_score = best_score.max(score);
        }
    }

    best_score
}

/// Evaluates a leaf node position using the neural network.
///
/// Selects the network via [`Eval::should_use_main_network`]; only main-network
/// evaluations go through the eval cache and the search's corrections.
#[inline(always)]
pub fn evaluate(ctx: &SearchContext, board: &Board) -> Score {
    if ctx.ply() == INITIAL_EMPTY_COUNT {
        return board.final_score();
    }

    if Eval::should_use_main_network(ctx.eval_mode, ctx.ply()) {
        ctx.evaluate_main_with_key(board, board.hash())
    } else {
        ctx.eval.evaluate_small(ctx.pattern_feature(), ctx.ply())
    }
}

#[cfg(test)]
mod schedule_tests {
    use super::*;
    use std::sync::Arc;

    use crate::player_color::PlayerColor;
    use crate::score::DiscScore;
    use crate::search::RootPhase;
    use crate::search::threading::ThreadPool;
    use crate::search::time_control::{TimeControlMode, TimeManager};
    use crate::transposition_table::TranspositionTable;

    #[test]
    fn depth1_move_order_masks_partition_the_board() {
        let mut covered = 0;
        for &mask in &DEPTH1_MOVE_ORDER_MASKS {
            let mask = mask.bits();
            assert_eq!(covered & mask, 0, "depth-1 move-order masks overlap");
            covered |= mask;
        }
        assert_eq!(
            covered,
            u64::MAX,
            "depth-1 move-order masks do not cover the board"
        );
    }

    #[test]
    fn aspiration_window_accepts_history_near_the_score_sentinel() {
        let board = Board::new();
        let roots = RootMoves::new(&board);
        let tt = TranspositionTable::new(0);
        let eval = shared_eval();
        let ctx = SearchContext::new(&board, Selectivity::Mid, &roots, &tt, &eval);
        let first = roots.get(0).unwrap().sq;

        roots.update(
            first,
            Score::INF - 1,
            true,
            &[None; crate::constants::MAX_PLY],
        );

        let (alpha, beta, delta) = aspiration_window(&ctx, 5);
        assert_eq!((alpha, beta), (-Score::INF, Score::INF));
        assert_eq!(delta.value(), 105_840);
    }

    #[test]
    fn aspiration_search_retries_both_boundaries_on_a_real_root() {
        let board = Board::new().make_move(Square::D3);
        let eval = shared_eval();
        let pool = ThreadPool::new(1);
        let search_window = |mut alpha, mut beta| {
            let roots = RootMoves::new(&board);
            let tt = TranspositionTable::new(0);
            let mut ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
            let delta = Score::ZERO + 512;
            let (score, clean) = aspiration_search(
                &mut ctx,
                &board,
                5,
                &mut alpha,
                &mut beta,
                delta,
                pool.main(),
            );
            assert!(alpha < score && score < beta);
            (score, clean)
        };
        let (expected, clean) = search_window(-Score::INF, Score::INF);
        assert!(clean);
        assert_eq!(search_window(-Score::INF, expected), (expected, false));
        assert_eq!(search_window(expected, Score::INF), (expected, false));
    }

    #[test]
    fn probcut_search_rejects_sentinel_scores_and_restores_selectivity() {
        let board = Board::new().make_move(Square::D3);
        let roots = RootMoves::new(&board);
        let tt = TranspositionTable::new(0);
        let eval = shared_eval();
        let pool = ThreadPool::new(1);
        let mut ctx = SearchContext::new(&board, Selectivity::Mid, &roots, &tt, &eval);
        let static_eval = evaluate(&ctx, &board);

        for depth in [4, 6] {
            let before = ctx.n_nodes;
            for sentinel in [-Score::INF, Score::INF] {
                assert_eq!(
                    try_probcut(
                        &mut ctx,
                        &board,
                        depth,
                        sentinel,
                        true,
                        static_eval,
                        pool.main()
                    ),
                    None,
                );
            }
            assert_eq!(ctx.n_nodes, before);
            let beta = -Score::INF + 1;
            assert_eq!(
                try_probcut(
                    &mut ctx,
                    &board,
                    depth,
                    beta,
                    true,
                    static_eval,
                    pool.main()
                ),
                Some(beta),
            );
            assert_eq!(ctx.n_nodes > before, depth >= 6);
            assert_eq!(ctx.selectivity, Selectivity::Mid);
            assert_eq!(ctx.empty_list.count(), board.empty_count());
        }
    }

    #[test]
    fn midgame_search_keeps_finite_scores_after_a_rejected_tt_write() {
        let board = Board::new().make_move(Square::D3);
        let roots = RootMoves::new(&board);
        let tt = TranspositionTable::new(0);
        let eval = shared_eval();
        let pool = ThreadPool::new(1);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
        tt.store(
            tt.probe(&board, board.hash()).slot(),
            &board,
            Score::INF,
            Bound::Exact,
            12,
            None,
            Selectivity::None,
            false,
            false,
        );
        for (alpha, beta) in [(-Score::INF, -Score::INF + 1), (Score::INF - 1, Score::INF)] {
            let shallow = evaluate_shallow_with_tt(&mut ctx, &board, 3, alpha, beta);
            assert!(shallow.is_finite());
            let root = search::<Root, MidGameStrategy>(
                &mut ctx,
                &board,
                5,
                alpha,
                beta,
                pool.main(),
                false,
            );
            assert!(root.is_finite());
        }
    }

    #[test]
    fn shallow_verification_matches_exhaustive_thresholds() {
        fn exhaustive(ctx: &mut SearchContext<'_>, board: &Board, depth: Depth) -> Score {
            if depth == 0 {
                return evaluate(ctx, board);
            }
            let moves = board.legal_moves();
            if moves.is_empty() {
                let next = board.switch_players();
                if !next.has_legal_moves() {
                    return board.solve(ctx.empty_list.count());
                }
                ctx.update_pass();
                let score = -exhaustive(ctx, &next, depth);
                ctx.undo_pass();
                return score;
            }
            moves
                .iter()
                .map(|square| {
                    let flipped = flip::flip(square, board.player(), board.opponent());
                    if flipped == board.opponent() {
                        return Score::MAX;
                    }
                    let next = board.make_move_with_flipped(flipped, square);
                    ctx.update(square, flipped);
                    let score = -exhaustive(ctx, &next, depth - 1);
                    ctx.undo(square);
                    score
                })
                .max()
                .unwrap()
        }

        let forced_pass = Board::from_bitboards(0xfffe, !0xffff);
        let wipeout = Board::from_bitboards(
            Square::E4.bitboard() | Square::D5.bitboard(),
            Square::D4.bitboard() | Square::C4.bitboard(),
        );
        let terminal = wipeout.make_move(Square::B4);
        assert!(!forced_pass.has_legal_moves() && forced_pass.switch_players().has_legal_moves());
        assert!(MoveList::new(&wipeout).wipeout_move().is_some());
        assert!(terminal.is_game_over());

        let tt = TranspositionTable::new(0);
        let eval = shared_eval();
        for board in [
            Board::new(),
            Board::new().make_move(Square::D3),
            forced_pass,
            wipeout,
            terminal,
        ] {
            let roots = RootMoves::new(&board);
            let mut ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
            let side = ctx.side_to_move;
            for (depth, verify) in [
                (
                    1,
                    evaluate_depth1 as fn(&mut SearchContext<'_>, &Board, Score, Score) -> Score,
                ),
                (2, evaluate_depth2),
            ] {
                let exact = exhaustive(&mut ctx, &board, depth);
                assert_eq!(verify(&mut ctx, &board, -Score::INF, Score::INF), exact);
                for beta in [exact, exact + 1] {
                    let result = verify(&mut ctx, &board, beta - 1, beta);
                    assert_eq!(result >= beta, exact >= beta);
                }
                assert_eq!(ctx.empty_list.count(), board.empty_count());
                assert!(ctx.side_to_move == side);
            }
        }
    }

    fn shared_eval() -> Arc<Eval> {
        Eval::shared_for_tests()
    }

    #[test]
    fn boundary_abort_after_completed_iteration_reports_reached_depth() {
        let pool = ThreadPool::new(1);
        let abort_pool = pool.clone();
        let board = Board::new().make_move(Square::D3);
        let task = SearchTask {
            board,
            mid_selectivity: Selectivity::None,
            tt: Arc::new(TranspositionTable::new(0)),
            pool: pool.clone(),
            eval: shared_eval(),
            phase: RootPhase::Midgame { max_depth: 1 },
            multi_pv: false,
            callback: Some(Arc::new(move |progress| {
                if progress.depth == 1 {
                    abort_pool.abort_search();
                }
            })),
            time_manager: None,
            eval_mode: None,
            corrections: Default::default(),
        };

        let result = search_root(task, 1, pool.main());

        assert_eq!(result.depth(), 1);
        assert!(result.best_move().is_some());
    }

    #[test]
    fn start_depth_is_two_for_even_targets_and_one_for_odd() {
        for (target, expected) in [(1, 1), (2, 2), (7, 1), (8, 2), (59, 1), (60, 2)] {
            assert_eq!(compute_start_depth(target), expected, "target={target}");
        }
    }

    #[test]
    fn iteration_depth_steps_by_two_then_by_one() {
        // At/below the threshold: +2.
        assert_eq!(next_iteration_depth(4), 6);
        assert_eq!(
            next_iteration_depth(DEPTH_STEP_THRESHOLD),
            DEPTH_STEP_THRESHOLD + 2
        );
        // Above the threshold: +1.
        assert_eq!(
            next_iteration_depth(DEPTH_STEP_THRESHOLD + 1),
            DEPTH_STEP_THRESHOLD + 2
        );
        assert_eq!(
            next_iteration_depth(DEPTH_STEP_THRESHOLD + 2),
            DEPTH_STEP_THRESHOLD + 3
        );
    }

    #[test]
    fn time_control_hands_off_to_endgame_solve_after_the_last_depth() {
        let pool = ThreadPool::new(1);
        // FFO position #1: 14 empties, Black to move, exact score +18 (G8).
        let board = Board::from_string(
            "--XXXXX--OOOXX-O-OOOXXOX-OXOXOXXOXXXOXXX--XOXOXX-XXXOOO--OOOOO--",
            PlayerColor::Black,
        )
        .unwrap();
        let n_empties = board.empty_count();
        let time_manager = Arc::new(TimeManager::new(
            TimeControlMode::Byoyomi {
                time_per_move_ms: 600_000,
            },
            pool.abort_state(),
            n_empties,
        ));
        let task = SearchTask {
            board,
            mid_selectivity: Selectivity::Mid,
            tt: Arc::new(TranspositionTable::new(1)),
            pool: pool.clone(),
            eval: shared_eval(),
            phase: RootPhase::Midgame {
                max_depth: n_empties,
            },
            multi_pv: false,
            callback: None,
            time_manager: Some(time_manager),
            eval_mode: None,
            corrections: Default::default(),
        };

        let result = search_root(task, n_empties, pool.main());

        assert!(result.is_endgame());
        assert_eq!(result.depth(), n_empties);
        assert_eq!(result.probability(), 100);
        assert_eq!(result.best_move(), Some(Square::G8));
        assert_eq!(
            result.score(),
            Some(Score::from_disc_diff(DiscScore::new(18)))
        );
    }
}
