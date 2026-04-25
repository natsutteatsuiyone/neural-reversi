//! Midgame search algorithms.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/midgame.c>

use std::sync::Arc;

use rand::seq::IteratorRandom;

use crate::board::Board;
use crate::flip;
use crate::move_list::MoveList;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::search::node_type::{NodeType, NonPV, Root};
use crate::search::search_context::SearchContext;
use crate::search::search_result::SearchResult;
use crate::search::search_strategy::MidGameStrategy;
use crate::search::threading::Thread;
use crate::search::time_control::should_stop_iteration;
use crate::search::{SearchProgress, SearchTask, search};
use crate::square::Square;
use crate::transposition_table::Bound;
use crate::types::{Depth, ScaledScore};

/// Initial aspiration window delta.
const ASPIRATION_DELTA: ScaledScore = ScaledScore::from_disc_diff(3);

/// Minimum depth to enable aspiration windows.
const ASPIRATION_MIN_DEPTH: Depth = 5;

/// Depth threshold for switching iteration step from +2 to +1.
const DEPTH_STEP_THRESHOLD: Depth = 10;

/// Minimum depth to enable Late Move Reductions.
pub const LMR_MIN_DEPTH: Depth = 4;

/// Depth threshold for deeper LMR (reduction = 2).
pub const LMR_DEEPER_DEPTH: Depth = 8;

/// Performs the root search using iterative deepening with aspiration windows.
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let time_manager = task.time_manager.clone();
    let use_time_control = time_manager.is_some();

    let mut ctx = SearchContext::new(&board, task.selectivity, task.tt.clone(), task.eval.clone());

    if let Some(mode) = task.eval_mode {
        ctx.eval_mode = mode;
    }

    if ctx.root_moves_count() == 0 {
        return SearchResult::new_no_moves(false);
    }

    let n_empties = ctx.empty_list.count();
    if n_empties == 60 && !task.multi_pv {
        return SearchResult::new_random_move(random_move(&board));
    }

    let pv_count = if task.multi_pv {
        ctx.root_moves_count()
    } else {
        1
    };
    let max_depth = task.level.mid_depth.max(1).min(n_empties);

    let mut depth = compute_start_depth(max_depth);
    while depth <= max_depth {
        ctx.save_previous_scores();

        for pv_idx in 0..pv_count {
            ctx.set_pv_idx(pv_idx);

            let (mut alpha, mut beta) = ctx
                .get_current_pv_root_move()
                .filter(|_| depth >= ASPIRATION_MIN_DEPTH)
                .map(|rm| {
                    (
                        (rm.previous_score - ASPIRATION_DELTA).max(-ScaledScore::INF),
                        (rm.previous_score + ASPIRATION_DELTA).min(ScaledScore::INF),
                    )
                })
                .unwrap_or((-ScaledScore::INF, ScaledScore::INF));

            let score = aspiration_search(&mut ctx, &board, depth, &mut alpha, &mut beta, thread);

            ctx.sort_root_moves_from_pv_idx();

            if thread.is_search_aborted() {
                break;
            }

            if let Some(ref callback) = task.callback
                && let Some(rm) = ctx.get_current_pv_root_move()
            {
                callback(SearchProgress {
                    depth,
                    target_depth: max_depth,
                    score: score.to_disc_diff_f32(),
                    best_move: rm.sq,
                    probability: ctx.selectivity.probability(),
                    nodes: ctx.counters.n_nodes,
                    pv_line: rm.pv.clone(),
                    is_endgame: false,
                    counters: ctx.counters.clone(),
                });
            }
        }

        ctx.sort_all_root_moves();
        let best_move = ctx
            .get_best_root_move()
            .expect("internal error: no root moves after search");

        if let Some(ref tm) = time_manager {
            tm.report_iteration(best_move.sq, best_move.score.to_disc_diff_f32(), depth);
        }

        if thread.is_search_aborted() || should_stop_iteration(&time_manager) {
            return SearchResult::from_root_move(
                &ctx.root_moves,
                &best_move,
                depth.min(n_empties),
                ctx.selectivity,
                false,
                ctx.counters.clone(),
            );
        }

        depth = next_iteration_depth(depth, max_depth, &mut ctx.selectivity, use_time_control);
        if depth == 0 {
            break;
        }
    }

    let rm = ctx
        .get_best_root_move()
        .expect("internal error: no root moves after search");
    SearchResult::from_root_move(
        &ctx.root_moves,
        &rm,
        max_depth.min(n_empties),
        ctx.selectivity,
        false,
        ctx.counters.clone(),
    )
}

/// Computes the starting depth for iterative deepening.
pub(super) fn compute_start_depth(max_depth: Depth) -> Depth {
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
    if use_time_control && current_depth == max_depth - 1 {
        if selectivity.is_enabled() {
            *selectivity = Selectivity::from_u8(selectivity.as_u8() + 1);
            return current_depth;
        } else {
            return 0;
        }
    }

    if current_depth <= DEPTH_STEP_THRESHOLD {
        current_depth + 2
    } else {
        current_depth + 1
    }
}

/// Selects a random legal move from the current position.
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    board.get_moves().iter().choose(&mut rng).unwrap()
}

/// Attempts ProbCut pruning for midgame positions.
pub fn try_probcut(
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
    let pc_depth = 2 * (depth / 5);
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
            return Some(beta);
        }
    }

    None
}

/// Specialized alpha-beta search for positions at depth 3.
pub fn evaluate_depth3<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let org_alpha = alpha;

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    let moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth3::<NT>(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    let tt_probe_result = ctx.tt.probe(board, tt_key);
    let tt_move = tt_probe_result.best_move();

    if !NT::PV_NODE
        && let Some(tt_data) = tt_probe_result.data()
        && tt_data.depth() >= 3
        && tt_data.selectivity() == Selectivity::None
        && tt_data.can_cut(beta)
    {
        return tt_data.score();
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move().is_some() {
        return ScaledScore::MAX;
    }

    if move_list.count() >= 2 {
        move_list.evaluate_moves_fast(ctx, board, tt_move);
    }

    let mut best_score = -ScaledScore::INF;
    let mut best_move = Square::None;
    for mv in move_list.best_first_iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);

        ctx.update(mv.sq, mv.flipped);
        let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
        ctx.undo(mv.sq);

        if score > best_score {
            best_score = score;
            if score >= beta {
                best_move = mv.sq;
                break;
            }
            if score > alpha {
                best_move = mv.sq;
                alpha = score;
            }
        }
    }

    ctx.tt.store(
        tt_probe_result.index(),
        board,
        best_score,
        Bound::classify::<NT>(best_score, org_alpha, beta),
        3,
        best_move,
        Selectivity::None,
        false,
    );

    best_score
}

/// Specialized alpha-beta search for positions at depth 2.
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
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move().is_some() {
        return ScaledScore::MAX;
    }

    if move_list.count() >= 2 {
        move_list.evaluate_moves_fast(ctx, board, Square::None);
    }

    let mut best_score = -ScaledScore::INF;
    for mv in move_list.best_first_iter() {
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

    best_score
}

/// Specialized alpha-beta search for positions at depth 1.
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
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    let mut best_score = -ScaledScore::INF;

    for sq in moves.corners().iter() {
        if let Some(score) = search_move_in_evaluate_depth1(ctx, board, sq, beta, &mut best_score) {
            return score;
        }
    }

    for sq in moves.non_corners().iter() {
        if let Some(score) = search_move_in_evaluate_depth1(ctx, board, sq, beta, &mut best_score) {
            return score;
        }
    }

    best_score
}

/// Searches a single move within [`evaluate_depth1`], returning on beta cutoff.
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

/// Evaluates a leaf node position using the neural network.
#[inline(always)]
pub fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore {
    if ctx.ply() == 60 {
        return board.final_score_scaled();
    }

    ctx.eval.evaluate(ctx, board)
}
