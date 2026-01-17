//! Midgame search algorithms.
//!
//! # References
//!
//! - <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/midgame.c>

use std::sync::Arc;

use rand::seq::IteratorRandom;

use crate::board::Board;
use crate::eval::EvalMode;
use crate::flip;
use crate::move_list::MoveList;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::search::node_type::NonPV;
use crate::search::node_type::Root;
use crate::search::search_context::SearchContext;
use crate::search::search_result::SearchResult;
use crate::search::search_strategy::MidGameStrategy;
use crate::search::threading::Thread;
use crate::search::time_control::should_stop_iteration;
use crate::search::{SearchProgress, SearchTask, search};
use crate::square::Square;
use crate::types::{Depth, ScaledScore};

/// Initial aspiration window delta.
const ASPIRATION_DELTA: ScaledScore = ScaledScore::from_disc_diff(3);

/// Performs the root search using iterative deepening with aspiration windows.
///
/// # Arguments
///
/// * `task` - Search task containing board position, search parameters, and callbacks
/// * `thread` - Thread handle for parallel search coordination
///
/// # Returns
///
/// SearchResult containing the best move, score, and search statistics.
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let time_manager = task.time_manager.clone();
    let use_time_control = time_manager.is_some();

    let mut ctx = SearchContext::new(&board, task.selectivity, task.tt.clone(), task.eval.clone());
    ctx.eval_mode = EvalMode::Large;

    let n_empties = ctx.empty_list.count;

    // Handle opening position with random move
    if n_empties == 60 && !task.multi_pv {
        return SearchResult::new_random_move(random_move(&board));
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

    while depth <= max_depth {
        // Adjust selectivity based on remaining depth (only without time control)
        if !use_time_control {
            let depth_diff = (max_depth - depth) as u8;
            ctx.selectivity =
                Selectivity::from_u8(org_selectivity.as_u8().saturating_sub(depth_diff));
        }

        // Save previous iteration scores for aspiration windows
        ctx.save_previous_scores();

        // Multi-PV loop: search each PV line with its own aspiration window
        for pv_idx in 0..pv_count {
            ctx.set_pv_idx(pv_idx);

            // Set up aspiration window based on previous score at this PV index
            let (mut alpha, mut beta) = ctx
                .get_current_pv_root_move()
                .filter(|_| depth >= 5)
                .map(|rm| {
                    (
                        (rm.previous_score - ASPIRATION_DELTA).max(-ScaledScore::INF),
                        (rm.previous_score + ASPIRATION_DELTA).min(ScaledScore::INF),
                    )
                })
                .unwrap_or((-ScaledScore::INF, ScaledScore::INF));

            let score = aspiration_search(&mut ctx, &board, depth, &mut alpha, &mut beta, thread);

            // Stable sort moves from pv_idx to end, bringing best to pv_idx position
            ctx.sort_root_moves_from_pv_idx();

            if thread.is_search_aborted() {
                break;
            }

            // Notify progress with the move now at pv_idx (the best for this PV line)
            if let Some(ref callback) = task.callback
                && let Some(rm) = ctx.get_current_pv_root_move()
            {
                callback(SearchProgress {
                    depth,
                    target_depth: max_depth,
                    score: score.to_disc_diff_f32(),
                    best_move: rm.sq,
                    probability: ctx.selectivity.probability(),
                    nodes: ctx.n_nodes,
                    pv_line: rm.pv.clone(),
                    eval_mode: ctx.eval_mode,
                });
            }
        }

        // Sort all root moves by score for consistent ordering
        ctx.sort_all_root_moves();
        let best_move = ctx.get_best_root_move().unwrap();

        // Notify time manager about search progress
        if let Some(ref tm) = time_manager {
            let pv_changed = prev_best_move.is_some_and(|sq| sq != best_move.sq);
            tm.try_extend_time(best_move.score.to_disc_diff_f32(), pv_changed, depth);
        }
        prev_best_move = Some(best_move.sq);

        // Check termination conditions
        if thread.is_search_aborted() || should_stop_iteration(&time_manager) {
            return SearchResult::from_root_move(
                &ctx.root_moves,
                &best_move,
                ctx.n_nodes,
                depth.min(n_empties),
                ctx.selectivity,
                EvalMode::Large,
            );
        }

        // Advance to next depth
        depth = next_iteration_depth(depth, max_depth, &mut ctx.selectivity, use_time_control);
        if depth == 0 {
            break; // selectivity maxed out, exit loop
        }
    }

    let rm = ctx.get_best_root_move().unwrap();
    SearchResult::from_root_move(
        &ctx.root_moves,
        &rm,
        ctx.n_nodes,
        max_depth.min(n_empties),
        ctx.selectivity,
        EvalMode::Large,
    )
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
    alpha: &mut ScaledScore,
    beta: &mut ScaledScore,
    thread: &Arc<Thread>,
) -> ScaledScore {
    let mut delta = ASPIRATION_DELTA;

    loop {
        let score = search::<Root, MidGameStrategy>(ctx, board, depth, *alpha, *beta, thread);

        if thread.is_search_aborted() {
            return score;
        }

        if score <= *alpha {
            *beta = *alpha;
            *alpha = (score - delta).max(-ScaledScore::INF);
        } else if score >= *beta {
            *alpha = (*beta - delta).max(*alpha);
            *beta = (score + delta).min(ScaledScore::INF);
        } else {
            return score;
        }

        delta += delta / 2;
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

/// Selects a random legal move from the current position.
///
/// # Arguments
///
/// * `board` - Current board position.
///
/// # Returns
///
/// Randomly selected legal move.
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    board.get_moves().iter().choose(&mut rng).unwrap()
}

/// Attempts ProbCut pruning for midgame positions
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `depth` - Depth of the deep search.
/// * `beta` - Beta bound.
/// * `thread` - Thread handle for parallel search.
///
/// # Returns
///
/// * `Some(score)` - If probcut triggers, returns the predicted bound.
/// * `None` - If probcut doesn't trigger, deep search should be performed.
pub fn probcut(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    beta: ScaledScore,
    thread: &Arc<Thread>,
) -> Option<ScaledScore> {
    if !ctx.selectivity.is_enabled() {
        return None;
    }

    let ply = ctx.ply();
    let pc_depth = 2 * (depth as f64 * 0.2).floor() as Depth;
    let mean = probcut::get_mean(ply, pc_depth, depth);
    let sigma = probcut::get_sigma(ply, pc_depth, depth);
    let t = ctx.selectivity.t_value();

    let pc_beta = probcut::compute_probcut_beta(beta, t, mean, sigma);
    if pc_beta >= ScaledScore::MAX {
        return None;
    }

    let eval_score = evaluate(ctx, board);
    let mean0 = probcut::get_mean(ply, 0, depth);
    let sigma0 = probcut::get_sigma(ply, 0, depth);
    let eval_beta = probcut::compute_eval_beta(beta, t, mean, sigma, mean0, sigma0);

    if eval_score >= eval_beta {
        let current_selectivity = ctx.selectivity;
        ctx.selectivity = Selectivity::None; // Disable nested probcut
        let score =
            search::<NonPV, MidGameStrategy>(ctx, board, pc_depth, pc_beta - 1, pc_beta, thread);
        ctx.selectivity = current_selectivity;

        if score >= pc_beta {
            return Some(ScaledScore::from_raw((beta.value() + pc_beta.value()) / 2));
        }
    }

    None
}

/// Specialized evaluation function for positions at depth 2.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Alpha bound.
/// * `beta` - Beta bound.
///
/// # Returns
///
/// Best score found.
pub fn evaluate_depth2(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count);
        }
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move().is_some() {
        return ScaledScore::MAX;
    }

    let mut best_score = -ScaledScore::INF;
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
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Alpha bound.
/// * `beta` - Beta bound.
///
/// # Returns
///
/// Best score found.
pub fn evaluate_depth1(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count);
        }
    }

    let mut best_score = -ScaledScore::INF;

    // Process corner moves first
    for sq in moves.corners().iter() {
        if let Some(score) = search_move_in_evaluate_depth1(ctx, board, sq, beta, &mut best_score) {
            return score;
        }
    }

    // Process non-corner moves
    for sq in moves.non_corners().iter() {
        if let Some(score) = search_move_in_evaluate_depth1(ctx, board, sq, beta, &mut best_score) {
            return score;
        }
    }

    best_score
}

/// Searches a move and updates the best score if it's better.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `sq` - Square to search.
/// * `beta` - Beta bound.
/// * `best_score` - Best score found so far.
///
/// # Returns
///
/// Best score found.
#[inline(always)]
fn search_move_in_evaluate_depth1(
    ctx: &mut SearchContext,
    board: &Board,
    sq: Square,
    beta: ScaledScore,
    best_score: &mut ScaledScore,
) -> Option<ScaledScore> {
    let flipped = flip::flip(sq, board.player, board.opponent);
    if flipped == board.opponent {
        return Some(ScaledScore::MAX);
    }
    let next = board.make_move_with_flipped(flipped, sq);

    ctx.update(sq, flipped);
    let score = -evaluate(ctx, &next);
    ctx.undo(sq);

    if score > *best_score {
        *best_score = score;
        if score >= beta {
            return Some(score);
        }
    }
    None
}

/// Evaluates a leaf node position using the neural network evaluator.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
///
/// # Returns
///
/// Position score.
#[inline(always)]
pub fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore {
    if ctx.ply() == 60 {
        return board.final_score_scaled();
    }

    ctx.eval.evaluate(ctx, board)
}
