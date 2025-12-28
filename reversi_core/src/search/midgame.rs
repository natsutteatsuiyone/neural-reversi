//! # References:
//! - https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/midgame.c
//! - https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/search.cpp

use std::sync::Arc;

use rand::seq::IteratorRandom;

use super::SearchTask;
use super::search_result::SearchResult;
use super::threading::Thread;
use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::constants::{MID_SCORE_MAX, SCORE_INF, scale_score, unscale_score, unscale_score_f32};
use crate::flip;
use crate::move_list::ConcurrentMoveIterator;
use crate::move_list::MoveList;
use crate::probcut;
use crate::search::endgame;
use crate::search::enhanced_transposition_cutoff;
use crate::search::node_type::{NodeType, NonPV, PV, Root};
use crate::search::search_context::GamePhase;
use crate::search::search_context::SearchContext;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::stability;
use crate::transposition_table::Bound;
use crate::types::Depth;
use crate::types::Score;
use crate::types::Selectivity;

/// Minimum depth required before considering parallel split.
const MIN_SPLIT_DEPTH: Depth = 5;

/// Minimum depth for enhanced transposition table cutoff.
const MIN_ETC_DEPTH: Depth = 6;

/// Initial aspiration window delta.
const ASPIRATION_DELTA: Score = scale_score(3);

/// Performs the root search using iterative deepening with aspiration windows.
///
/// # Arguments
///
/// * `task` - Search task containing board position, search parameters, and callbacks
/// * `thread` - Thread handle for parallel search coordination
///
/// # Returns
///
/// SearchResult containing the best move, score, and search statistics
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let time_manager = task.time_manager.clone();
    let use_time_control = time_manager.is_some();

    let mut ctx = SearchContext::new(
        &board,
        task.generation,
        task.selectivity,
        task.tt.clone(),
        task.eval.clone(),
    );
    ctx.game_phase = GamePhase::MidGame;

    let n_empties = ctx.empty_list.count;

    // Handle opening position with random move
    if n_empties == 60 && !task.multi_pv {
        return SearchResult::new_random_move(random_move(&board));
    }

    if let Some(ref callback) = task.callback {
        ctx.set_callback(callback.clone());
    }

    // Search configuration
    let org_selectivity = ctx.selectivity;
    let pv_count = if task.multi_pv {
        ctx.root_moves_count()
    } else {
        1
    };
    let max_depth = task.level.mid_depth.max(1).min(n_empties);
    let mut prev_best_move: Option<Square> = None;

    let mut depth = compute_start_depth(max_depth);
    let mut best_score = 0;
    let mut alpha = -SCORE_INF;
    let mut beta = SCORE_INF;

    while depth <= max_depth {
        // Adjust selectivity based on remaining depth (only without time control)
        if !use_time_control {
            let depth_diff = (max_depth - depth) as u8;
            ctx.selectivity =
                Selectivity::from_u8(org_selectivity.as_u8().saturating_sub(depth_diff));
        }
        ctx.reset_root_move_searched();

        // Reset aspiration window for shallow depths
        if depth <= 10 {
            alpha = -SCORE_INF;
            beta = SCORE_INF;
        }

        // Multi-PV loop
        for pv_idx in 0..pv_count {
            if pv_idx >= 1 {
                alpha = -SCORE_INF;
                beta = best_score;
            }

            best_score = aspiration_search(&mut ctx, &board, depth, &mut alpha, &mut beta, thread);

            let best_move = ctx.get_best_root_move(true).unwrap();
            ctx.mark_root_move_searched(best_move.sq);

            if thread.is_search_aborted() {
                break;
            }

            ctx.notify_progress(
                depth,
                unscale_score_f32(best_score),
                best_move.sq,
                ctx.selectivity,
            );
        }

        let best_move = ctx.get_best_root_move(false).unwrap();

        // Update aspiration window for next iteration
        alpha = (best_move.average_score - ASPIRATION_DELTA).max(-SCORE_INF);
        beta = (best_move.average_score + ASPIRATION_DELTA).min(SCORE_INF);

        // Notify time manager about search progress
        if let Some(ref tm) = time_manager {
            let pv_changed = prev_best_move.is_some_and(|sq| sq != best_move.sq);
            tm.try_extend_time(unscale_score_f32(best_move.score), pv_changed, depth);
        }
        prev_best_move = Some(best_move.sq);

        // Check termination conditions
        if thread.is_search_aborted() || should_stop_iteration(&time_manager) {
            return build_search_result(&ctx, &best_move, depth.min(n_empties));
        }

        // Advance to next depth
        depth = next_iteration_depth(depth, max_depth, &mut ctx.selectivity, use_time_control);
        if depth == 0 {
            break; // selectivity maxed out, exit loop
        }
    }

    let rm = ctx.get_best_root_move(false).unwrap();
    build_search_result(&ctx, &rm, max_depth.min(n_empties))
}

/// Computes the starting depth for iterative deepening.
fn compute_start_depth(max_depth: Depth) -> Depth {
    if max_depth.is_multiple_of(2) { 2 } else { 1 }
}

/// Performs aspiration window search at the given depth.
fn aspiration_search(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    alpha: &mut Score,
    beta: &mut Score,
    thread: &Arc<Thread>,
) -> Score {
    let mut delta = ASPIRATION_DELTA;

    loop {
        let score = search::<Root>(ctx, board, depth, *alpha, *beta, thread);

        if thread.is_search_aborted() {
            return score;
        }

        if score <= *alpha {
            *beta = *alpha;
            *alpha = (score - delta).max(-SCORE_INF);
        } else if score >= *beta {
            *alpha = (*beta - delta).max(*alpha);
            *beta = (score + delta).min(SCORE_INF);
        } else {
            return score;
        }

        delta += delta / 2;
    }
}

/// Determines whether to stop the current iteration based on time control.
fn should_stop_iteration(time_manager: &Option<Arc<super::time_control::TimeManager>>) -> bool {
    match time_manager {
        Some(tm) => tm.check_time() || !tm.should_continue_iteration(),
        None => false,
    }
}

/// Computes the next iteration depth, handling selectivity progression.
///
/// Returns 0 if the search should terminate (selectivity maxed out).
fn next_iteration_depth(
    current_depth: Depth,
    max_depth: Depth,
    selectivity: &mut Selectivity,
    use_time_control: bool,
) -> Depth {
    // At max_depth - 1 with time control, try increasing selectivity instead of depth
    if use_time_control && current_depth == max_depth - 1 {
        if selectivity.is_enabled() {
            *selectivity = Selectivity::from_u8(selectivity.as_u8() + 1);
            return current_depth; // Stay at same depth with higher selectivity
        } else {
            return 0; // Signal to exit
        }
    }

    // Normal depth progression: +2 for shallow, +1 for deep
    if current_depth <= 10 {
        current_depth + 2
    } else {
        current_depth + 1
    }
}

/// Builds the final search result from the current state.
fn build_search_result(
    ctx: &SearchContext,
    best_move: &super::root_move::RootMove,
    reported_depth: Depth,
) -> SearchResult {
    SearchResult {
        score: unscale_score_f32(best_move.score),
        best_move: Some(best_move.sq),
        n_nodes: ctx.n_nodes,
        pv_line: best_move.pv.clone(),
        depth: reported_depth,
        selectivity: ctx.selectivity,
        game_phase: GamePhase::MidGame,
    }
}

/// Selects a random legal move from the current position.
///
/// # Arguments
///
/// * `board` - Current board position
///
/// # Returns
///
/// A randomly selected legal move square
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    BitboardIterator::new(board.get_moves())
        .choose(&mut rng)
        .unwrap()
}

/// Alpha-beta search function for midgame positions.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state and statistics.
/// * `board` - Current board position to search.
/// * `depth` - Remaining search depth.
/// * `alpha` - Lower bound of the search window
/// * `beta` - Upper bound of the search window
/// * `thread` - Thread handle for parallel search coordination.
///
/// # Returns
///
/// The best score found for the position.
pub fn search<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: Score,
    beta: Score,
    thread: &Arc<Thread>,
) -> Score {
    let org_alpha = alpha;
    let n_empties = ctx.empty_list.count;

    if NT::PV_NODE {
        if depth == 0 {
            return evaluate(ctx, board);
        }
    } else {
        match depth {
            0 => return evaluate(ctx, board),
            1 => return evaluate_depth1(ctx, board, alpha, beta),
            2 => return evaluate_depth2(ctx, board, alpha, beta),
            _ => {}
        }

        if let Some(score) = stability_cutoff(board, n_empties, alpha) {
            return score;
        }
    }

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -search::<NT>(ctx, &next, depth, -beta, -alpha, thread);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, n_empties);
        }
    } else if let Some(sq) = move_list.wipeout_move {
        if NT::ROOT_NODE {
            ctx.update_root_move(sq, MID_SCORE_MAX, 1, alpha);
        } else if NT::PV_NODE {
            ctx.update_pv(sq);
        }
        return MID_SCORE_MAX;
    }

    // Look up position in transposition table
    let tt_probe_result = ctx.tt.probe(tt_key, ctx.generation);
    let tt_move = tt_probe_result.best_move();

    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && tt_data.depth() >= depth
            && tt_data.selectivity() >= ctx.selectivity
            && tt_data.can_cut(beta)
        {
            return tt_data.score();
        }

        if depth >= MIN_ETC_DEPTH
            && let Some(score) = enhanced_transposition_cutoff(
                ctx,
                board,
                &move_list,
                depth,
                alpha,
                tt_key,
                tt_probe_result.index(),
            )
        {
            return score;
        }

        if let Some(score) = probcut::probcut_midgame(ctx, board, depth, beta, thread) {
            return score;
        }
    }

    if move_list.count() > 1 {
        move_list.evaluate_moves::<NT>(ctx, board, depth, tt_move);
        move_list.sort();
    }

    let move_iter = Arc::new(ConcurrentMoveIterator::new(move_list));
    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;

    while let Some((mv, move_count)) = move_iter.next() {
        if NT::ROOT_NODE && ctx.is_move_searched(mv.sq) {
            continue;
        }

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -SCORE_INF;
        if depth >= 2 && mv.reduction_depth > 0 {
            let d = (depth - 1).saturating_sub(mv.reduction_depth);
            score = -search::<NonPV>(ctx, &next, d, -(alpha + 1), -alpha, thread);
            if score > alpha {
                score = -search::<NonPV>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
            }
        } else if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
        }

        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            ctx.clear_pv();
            score = -search::<PV>(ctx, &next, depth - 1, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return 0;
        }

        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, alpha);
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                best_move = mv.sq;

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                }

                if NT::PV_NODE && score < beta {
                    alpha = score;
                } else {
                    break;
                }
            }
        }

        if depth >= MIN_SPLIT_DEPTH && move_iter.count() > 1 && thread.can_split() {
            let (s, m, n) = thread.split(
                ctx,
                board,
                alpha,
                beta,
                best_score,
                best_move,
                depth,
                &move_iter,
                NT::TYPE_ID,
            );
            best_score = s;
            best_move = m;
            ctx.n_nodes += n;

            if thread.is_search_aborted() || thread.cutoff_occurred() {
                return 0;
            }

            if best_score >= beta {
                break;
            }
        }
    }

    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        best_score,
        Bound::determine_bound::<NT>(best_score, org_alpha, beta),
        depth,
        best_move,
        ctx.selectivity,
        ctx.generation,
        false, // midgame search
    );

    best_score
}

/// Alpha-beta search function for splitpoint nodes in parallel search.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state and statistics.
/// * `board` - Current board position to search.
/// * `depth` - Remaining search depth.
/// * `thread` - Thread handle for parallel search coordination.
/// * `split_point` - Split point for parallel search coordination.
///
/// # Returns
///
/// The best score found for the position.
pub fn search_sp<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    thread: &Arc<Thread>,
    split_point: &Arc<SplitPoint>,
) -> Score {
    let beta = split_point.state().beta;
    let move_iter = split_point.state().move_iter.clone().unwrap();

    while let Some((mv, move_count)) = move_iter.next() {
        if NT::ROOT_NODE && ctx.is_move_searched(mv.sq) {
            continue;
        }

        split_point.unlock();

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let alpha = split_point.state().alpha();
        let mut score = -SCORE_INF;
        if depth >= 2 && mv.reduction_depth > 0 {
            let d = (depth - 1).saturating_sub(mv.reduction_depth);
            score = -search::<NonPV>(ctx, &next, d, -(alpha + 1), -alpha, thread);
            if score > alpha {
                let alpha = split_point.state().alpha();
                score = -search::<NonPV>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
            }
        } else if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
        }

        if NT::PV_NODE && score > alpha {
            ctx.clear_pv();
            let alpha = split_point.state().alpha();
            score = -search::<PV>(ctx, &next, depth - 1, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        split_point.lock();

        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return 0;
        }

        let sp = split_point.state();

        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, sp.alpha());
        }

        if score > sp.best_score() {
            sp.set_best_score(score);

            if score > sp.alpha() {
                sp.set_best_move(mv.sq);

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                    split_point.state_mut().copy_pv(ctx.get_pv());
                }

                if NT::PV_NODE && score < beta {
                    sp.set_alpha(score);
                } else {
                    sp.set_cutoff(true);
                    break;
                }
            }
        }
    }

    split_point.state().best_score()
}

/// Specialized evaluation function for positions at depth 2.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
pub fn evaluate_depth2(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: Score,
    beta: Score,
) -> Score {
    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, ctx.empty_list.count);
        }
    }

    let mut best_score = -SCORE_INF;
    if move_list.count() >= 3 {
        move_list.evaluate_moves_fast(ctx, board, Square::None);
        for mv in move_list.into_best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);

            ctx.update(mv.sq, mv.flipped);
            let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
            ctx.undo(mv.sq);

            if score > best_score {
                best_score = score;
                if score >= beta {
                    break;
                }
                if score > alpha {
                    alpha = score;
                }
            }
        }
    } else {
        for mv in move_list.iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);

            ctx.update(mv.sq, mv.flipped);
            let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
            ctx.undo(mv.sq);

            if score > best_score {
                best_score = score;
                if score >= beta {
                    break;
                }
                if score > alpha {
                    alpha = score;
                }
            }
        }
    }

    best_score
}

/// Specialized evaluation function for positions at depth 1.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
pub fn evaluate_depth1(ctx: &mut SearchContext, board: &Board, alpha: Score, beta: Score) -> Score {
    let moves = board.get_moves();
    if moves == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, ctx.empty_list.count);
        }
    }

    let mut best_score = -SCORE_INF;
    for sq in BitboardIterator::new(moves) {
        let flipped = flip::flip(sq, board.player, board.opponent);
        if flipped == board.opponent {
            return MID_SCORE_MAX;
        }
        let next = board.make_move_with_flipped(flipped, sq);

        ctx.update(sq, flipped);
        let score = -evaluate(ctx, &next);
        ctx.undo(sq);

        if score > best_score {
            best_score = score;
            if score >= beta {
                break;
            }
        }
    }

    best_score
}

/// Evaluates a leaf node position using the neural network evaluator.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position
///
/// # Returns
///
/// Position score in internal units
#[inline(always)]
pub fn evaluate(ctx: &SearchContext, board: &Board) -> Score {
    if ctx.ply() == 60 {
        return scale_score(endgame::calculate_final_score(board));
    }

    ctx.eval.evaluate(ctx, board)
}

/// Calls the endgame solver for terminal nodes where both players must pass.
///
/// # Arguments
///
/// * `board` - The terminal board position to be evaluated.
/// * `n_empties` - The number of empty squares remaining on the board.
///
/// # Returns
///
/// The exact final score of the position, scaled to internal units.
fn solve(board: &Board, n_empties: Depth) -> Score {
    scale_score(endgame::solve(board, n_empties))
}

/// Checks for stability-based cutoffs in the search.
///
/// # Arguments
///
/// * `board` - Current board position
/// * `n_empties` - Number of empty squares
/// * `alpha` - Current alpha bound for pruning decision
///
/// # Returns
///
/// * `Some(score)` - If position can be pruned with this score
/// * `None` - If no stability cutoff is possible
fn stability_cutoff(board: &Board, n_empties: Depth, alpha: Score) -> Option<Score> {
    stability::stability_cutoff(board, n_empties, unscale_score(alpha)).map(scale_score)
}
