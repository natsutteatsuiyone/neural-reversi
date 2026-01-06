//! # References:
//! - https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/midgame.c
//! - https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/search.cpp

use std::sync::Arc;

use rand::seq::IteratorRandom;

use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::flip;
use crate::move_list::ConcurrentMoveIterator;
use crate::move_list::MoveList;
use crate::probcut;
use crate::search::SearchTask;
use crate::search::endgame;
use crate::search::enhanced_transposition_cutoff;
use crate::search::node_type::{NodeType, NonPV, PV, Root};
use crate::search::search_context::GamePhase;
use crate::search::search_context::SearchContext;
use crate::search::search_result::SearchResult;
use crate::search::threading::SplitPoint;
use crate::search::threading::Thread;
use crate::search::time_control::should_stop_iteration;
use crate::square::Square;
use crate::stability;
use crate::transposition_table::Bound;
use crate::types::{Depth, ScaledScore, Selectivity};

/// Minimum depth required before considering parallel split.
const MIN_SPLIT_DEPTH: Depth = 5;

/// Minimum depth for enhanced transposition table cutoff.
const MIN_ETC_DEPTH: Depth = 6;

/// Minimum depth for probcut.
const MIN_PROBCUT_DEPTH: Depth = 3;

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
/// SearchResult containing the best move, score, and search statistics
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let time_manager = task.time_manager.clone();
    let use_time_control = time_manager.is_some();

    let mut ctx = SearchContext::new(&board, task.selectivity, task.tt.clone(), task.eval.clone());
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
            if let Some(rm) = ctx.get_current_pv_root_move() {
                ctx.notify_progress(
                    depth,
                    max_depth,
                    score.to_disc_diff_f32(),
                    rm.sq,
                    ctx.selectivity,
                    ctx.n_nodes,
                    rm.pv.clone(),
                );
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
                GamePhase::MidGame,
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
        GamePhase::MidGame,
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
        let score = search::<Root>(ctx, board, depth, *alpha, *beta, thread);

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
    mut alpha: ScaledScore,
    beta: ScaledScore,
    thread: &Arc<Thread>,
) -> ScaledScore {
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
            ctx.update_root_move(sq, ScaledScore::MAX, 1, alpha);
        } else if NT::PV_NODE {
            ctx.update_pv(sq);
        }
        return ScaledScore::MAX;
    }

    // Look up position in transposition table
    let tt_probe_result = ctx.tt.probe(tt_key);
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

        if depth >= MIN_PROBCUT_DEPTH
            && let Some(score) = probcut(ctx, board, depth, beta, thread)
        {
            return score;
        }
    }

    if NT::ROOT_NODE {
        move_list.exclude_earlier_pv_moves(ctx);
    }

    if move_list.count() > 1 {
        move_list.evaluate_moves(ctx, board, depth, tt_move);
        move_list.sort();
    }

    let move_iter = Arc::new(ConcurrentMoveIterator::new(move_list));
    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;

    while let Some((mv, move_count)) = move_iter.next() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -ScaledScore::INF;
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
            return ScaledScore::ZERO;
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
                return ScaledScore::ZERO;
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
        Bound::determine_bound::<NT>(best_score.value(), org_alpha.value(), beta.value()),
        depth,
        best_move,
        ctx.selectivity,
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
/// The best score found for the position (raw i32 for threading compatibility).
pub fn search_sp<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    thread: &Arc<Thread>,
    split_point: &Arc<SplitPoint>,
) -> ScaledScore {
    let beta = split_point.state().beta;
    let move_iter = split_point.state().move_iter.clone().unwrap();

    while let Some((mv, move_count)) = move_iter.next() {
        split_point.unlock();

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let alpha = split_point.state().alpha();
        let mut score = -ScaledScore::INF;
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
            return ScaledScore::ZERO;
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

/// Attempts ProbCut pruning for midgame positions
///
/// # Arguments
///
/// * `ctx` - Search context containing selectivity settings and search state
/// * `board` - Current board position to evaluate
/// * `depth` - Depth of the deep search that would be performed
/// * `beta` - Beta bound for the search window
/// * `thread` - Search thread used to run the shallow verification search
///
/// # Returns
///
/// * `Some(score)` - If probcut triggers, returns the predicted bound (alpha or beta)
/// * `None` - If probcut doesn't trigger, deep search should be performed
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
        let score = search::<NonPV>(ctx, board, pc_depth, pc_beta - 1, pc_beta, thread);
        ctx.selectivity = current_selectivity;

        if score >= pc_beta {
            return Some(ScaledScore::new((beta.value() + pc_beta.value()) / 2));
        }
    }

    None
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
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
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
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
pub fn evaluate_depth1(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
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

    let mut best_score = -ScaledScore::INF;
    for sq in BitboardIterator::new(moves) {
        let flipped = flip::flip(sq, board.player, board.opponent);
        if flipped == board.opponent {
            return ScaledScore::MAX;
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
pub fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore {
    if ctx.ply() == 60 {
        return ScaledScore::from_disc_diff(endgame::calculate_final_score(board));
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
fn solve(board: &Board, n_empties: Depth) -> ScaledScore {
    ScaledScore::from_disc_diff(endgame::solve(board, n_empties))
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
fn stability_cutoff(board: &Board, n_empties: Depth, alpha: ScaledScore) -> Option<ScaledScore> {
    stability::stability_cutoff(board, n_empties, alpha.to_disc_diff())
        .map(ScaledScore::from_disc_diff)
}
