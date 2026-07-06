//! Shared principal variation search core.
//!
//! Node-level alpha-beta search used by both the midgame and endgame phases,
//! specialized at compile time by [`NodeType`] and [`SearchStrategy`].

use std::sync::Arc;

use crate::board::Board;
use crate::flip;
use crate::move_list::MoveList;
use crate::probcut::Selectivity;
use crate::search::context::SearchContext;
use crate::search::midgame;
use crate::search::node_type::{NodeType, NonPV, PV};
use crate::search::strategy::SearchStrategy;
use crate::search::threading::{SplitPoint, SplitRequest, Thread};
use crate::square::Square;
use crate::stability::stability_cutoff;
use crate::transposition_table::Bound;
use crate::types::{Depth, ScaledScore};

/// Searches both midgame and endgame positions using Principal Variation Search.
pub fn search<NT: NodeType, SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: ScaledScore,
    beta: ScaledScore,
    thread: &Arc<Thread>,
    cut_node: bool,
) -> ScaledScore {
    let all_node = !NT::PV_NODE && !cut_node;
    let org_alpha = alpha;

    if NT::PV_NODE {
        if depth == 0 {
            ctx.clear_pv();
            return SS::evaluate(ctx, board);
        }
        ctx.prepare_pv();
    } else {
        if depth <= SS::DEPTH_TO_SHALLOW {
            return SS::shallow_search(ctx, board, depth, alpha, beta, thread);
        }

        if let Some(score) = stability_cutoff(board, ctx.empty_list.count(), alpha.to_disc_diff()) {
            ctx.counters.increment_stability_cut();
            return ScaledScore::from_disc_diff(score);
        }
    }

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    // Move generation
    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let child_cut_node = !NT::PV_NODE && !cut_node;
            let score = -search::<NT, SS>(ctx, &next, depth, -beta, -alpha, thread, child_cut_node);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    // Root node: exclude earlier PV moves (before wipeout/TT shortcuts)
    if NT::ROOT_NODE {
        move_list.exclude_earlier_pv_moves(ctx, board);
        if move_list.count() == 0 {
            return -ScaledScore::INF;
        }
    }

    if let Some(sq) = move_list.wipeout_move() {
        if NT::ROOT_NODE {
            ctx.update_root_move(sq, ScaledScore::MAX, true);
        } else if NT::PV_NODE {
            ctx.update_pv(sq);
        }
        return ScaledScore::MAX;
    }

    let use_etc = !NT::PV_NODE && depth >= SS::MIN_ETC_DEPTH;
    if use_etc {
        // Hide child TT latency behind the parent TT probe.
        for mv in move_list.iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.tt.prefetch(next.hash());
        }
    }

    // Transposition table probe
    let tt_probe_result = ctx.tt.probe(board, tt_key);
    ctx.counters.increment_tt_probe();
    let tt_move = tt_probe_result.best_move();

    // NonPV cutoffs
    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && tt_data.can_cut(beta, depth, ctx.selectivity, SS::IS_ENDGAME)
        {
            ctx.counters.increment_tt_hit();
            return tt_data.score();
        }

        // Enhanced Transposition Cutoff
        if use_etc {
            ctx.counters.increment_etc_attempt();
            if let Some(score) = enhanced_transposition_cutoff::<SS>(
                ctx,
                board,
                &move_list,
                depth,
                alpha,
                tt_probe_result.index(),
            ) {
                ctx.counters.increment_etc_cut();
                return score;
            }
        }

        // ProbCut
        if depth >= SS::MIN_PROBCUT_DEPTH {
            ctx.counters.increment_probcut_attempt();
            if let Some(score) = SS::try_probcut(ctx, board, depth, beta, cut_node, thread) {
                ctx.counters.increment_probcut_cut();
                return score;
            }
        }
    }

    let n_moves = move_list.count();
    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;
    let mut move_count: usize = 0;

    if !NT::PV_NODE && (n_moves == 1 || tt_move != Square::None) {
        let (sq, flipped) = if tt_move != Square::None {
            (
                tt_move,
                flip::flip(tt_move, board.player(), board.opponent()),
            )
        } else {
            let mv = move_list.get_move(0);
            (mv.sq, mv.flipped)
        };
        move_count = 1;

        let next = board.make_move_with_flipped(flipped, sq);
        ctx.update(sq, flipped);
        let score = -search::<NonPV, SS>(
            ctx,
            &next,
            depth - 1,
            -(alpha + 1),
            -alpha,
            thread,
            !cut_node,
        );
        ctx.undo(sq);

        if thread.should_stop() {
            return ScaledScore::ZERO;
        }

        best_score = score;
        if score > alpha {
            best_move = sq;
            if score >= beta {
                ctx.tt.store(
                    tt_probe_result.index(),
                    board,
                    best_score,
                    Bound::Lower,
                    depth,
                    best_move,
                    ctx.selectivity,
                    SS::IS_ENDGAME,
                );
                return best_score;
            }
            alpha = score;
        }
    }

    // Move ordering
    // Both branches must ensure the TT move ends up at index 0 when present,
    // so the main loop (starting at move_count=1) skips it correctly.
    if n_moves - move_count > 1 {
        move_list.evaluate_moves::<NT, SS>(ctx, board, depth, tt_move, alpha, cut_node);
        move_list.sort();
    } else if n_moves == 2 && move_list.get_move(0).sq != tt_move {
        move_list.swap_moves(0, 1);
    }

    // Main move loop
    let allow_speculative_split = all_node && depth <= SS::SPECULATIVE_SPLIT_MAX_DEPTH;
    while move_count < n_moves {
        // Parallel search split
        if (move_count >= 1 || allow_speculative_split)
            && depth >= SS::MIN_SPLIT_DEPTH
            && (n_moves - move_count) >= 2
            && thread.can_split()
        {
            let (s, m, c) = thread.split(
                ctx,
                SplitRequest {
                    board,
                    alpha,
                    beta,
                    best_score,
                    best_move,
                    depth,
                    move_list,
                    move_count,
                    node_type: NT::ID,
                    is_endgame: SS::IS_ENDGAME,
                    cut_node,
                },
            );
            best_score = s;
            best_move = m;
            ctx.counters.merge(&c);

            if thread.should_stop() {
                return ScaledScore::ZERO;
            }

            break; // Split consumed all remaining moves
        }

        let mv = move_list.get_move(move_count);
        move_count += 1;

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -ScaledScore::INF;

        if !NT::PV_NODE || move_count > 1 {
            let reduction = compute_lmr_reduction::<NT, SS>(
                ctx.selectivity,
                depth,
                move_count,
                n_moves,
                cut_node,
            );

            score = -search::<NonPV, SS>(
                ctx,
                &next,
                depth - 1 - reduction,
                -(alpha + 1),
                -alpha,
                thread,
                reduction > 0 || !cut_node,
            );

            if reduction > 0 && score > alpha {
                score = -search::<NonPV, SS>(
                    ctx,
                    &next,
                    depth - 1,
                    -(alpha + 1),
                    -alpha,
                    thread,
                    !cut_node,
                );
            }
        }

        // PV re-search
        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread, false);
        }

        ctx.undo(mv.sq);

        // Abort check
        if thread.should_stop() {
            return ScaledScore::ZERO;
        }

        // Root move update
        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count == 1 || score > alpha);
        }

        // Best score update
        if score > best_score {
            best_score = score;

            if score > alpha {
                best_move = mv.sq;

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                }

                if NT::PV_NODE && score < beta {
                    alpha = score;
                    if alpha >= ScaledScore::MAX {
                        break;
                    }
                } else {
                    break; // Beta cutoff
                }
            }
        }
    }

    // Store in transposition table
    ctx.tt.store(
        tt_probe_result.index(),
        board,
        best_score,
        Bound::classify::<NT>(best_score, org_alpha, beta),
        depth,
        best_move,
        ctx.selectivity,
        SS::IS_ENDGAME,
    );

    best_score
}

/// Searches remaining moves at a split point in parallel search.
///
/// Called by helper threads that join an existing split point. Picks moves from
/// the shared move iterator, searches them, and updates the split
/// point's best score/move under its lock.
pub(super) fn search_split_point<NT: NodeType, SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    thread: &Arc<Thread>,
    split_point: &Arc<SplitPoint>,
) -> ScaledScore {
    let beta = split_point.state().beta;
    let cut_node = split_point.state().cut_node;
    let move_iter = split_point.move_iter();
    let n_moves = move_iter.count();

    while let Some((mv, move_count)) = move_iter.next() {
        split_point.unlock();

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let alpha = split_point.state().alpha();

        debug_assert!(!NT::PV_NODE || move_count > 1);
        let reduction =
            compute_lmr_reduction::<NT, SS>(ctx.selectivity, depth, move_count, n_moves, cut_node);

        let mut score = -search::<NonPV, SS>(
            ctx,
            &next,
            depth - 1 - reduction,
            -(alpha + 1),
            -alpha,
            thread,
            reduction > 0 || !cut_node,
        );

        if reduction > 0 && score > alpha {
            score = -search::<NonPV, SS>(
                ctx,
                &next,
                depth - 1,
                -(alpha + 1),
                -alpha,
                thread,
                !cut_node,
            );
        }

        // PV re-search
        if NT::PV_NODE && score > alpha {
            let alpha = split_point.state().alpha();
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread, false);
        }

        ctx.undo(mv.sq);

        split_point.lock();

        // Abort check
        if thread.should_stop() {
            return ScaledScore::ZERO;
        }

        let sp = split_point.state();

        // Root move update
        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, score > sp.alpha());
        }

        // Best score update
        if score > sp.best_score() {
            sp.set_best_score(score);

            if score > sp.alpha() {
                sp.set_best_move(mv.sq);

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                    split_point.copy_pv(ctx.get_pv());
                }

                if NT::PV_NODE && score < beta {
                    sp.set_alpha(score);
                    if score >= ScaledScore::MAX {
                        thread.mark_split_point_cutoff(sp);
                        break;
                    }
                } else {
                    thread.mark_split_point_cutoff(sp);
                    break;
                }
            }
        }
    }

    split_point.state().best_score()
}

/// Performs enhanced transposition cutoff (ETC) by probing child positions.
///
/// For each move in the move list, checks the transposition table for the resulting
/// position. If a child entry produces a score above alpha with sufficient depth
/// and selectivity, stores a lower-bound entry at the parent and returns the cutoff score.
fn enhanced_transposition_cutoff<SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    move_list: &MoveList,
    depth: Depth,
    alpha: ScaledScore,
    tt_entry_index: usize,
) -> Option<ScaledScore> {
    let etc_depth = depth - 1;

    for mv in move_list.iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.increment_nodes();

        let etc_tt_key = next.hash();
        if let Some(etc_tt_data) = ctx.tt.lookup(&next, etc_tt_key)
            && (!SS::IS_ENDGAME || etc_tt_data.is_endgame())
            && etc_tt_data.depth() >= etc_depth
            && etc_tt_data.selectivity() >= ctx.selectivity
        {
            let score = -etc_tt_data.score();
            if (etc_tt_data.bound() == Bound::Exact || etc_tt_data.bound() == Bound::Upper)
                && score > alpha
            {
                ctx.tt.store(
                    tt_entry_index,
                    board,
                    score,
                    Bound::Lower,
                    depth,
                    mv.sq,
                    ctx.selectivity,
                    SS::IS_ENDGAME,
                );
                return Some(score);
            }
        }
    }
    None
}

/// Computes the LMR depth reduction for late moves.
///
/// Disabled for endgame search, PV nodes, and ProbCut verification search
/// (selectivity disabled). The schedule preserves the proven shallow 1/2-ply
/// reductions and adds a third ply only for moves that are late in both
/// absolute and relative order.
#[inline(always)]
fn compute_lmr_reduction<NT: NodeType, SS: SearchStrategy>(
    selectivity: Selectivity,
    depth: Depth,
    move_count: usize,
    n_moves: usize,
    cut_node: bool,
) -> Depth {
    if SS::IS_ENDGAME
        || NT::PV_NODE
        || !selectivity.is_enabled()
        || depth < midgame::LMR_MIN_DEPTH
        || move_count <= 2
        || n_moves < 4
    {
        return 0;
    }

    let mut reduction = lmr_base_reduction(depth, move_count, n_moves) as i32;
    if !cut_node && reduction > 2 {
        reduction -= 1;
    }

    let max_reduction = lmr_max_reduction(depth);
    reduction.max(0).min(max_reduction as i32) as Depth
}

#[inline(always)]
fn lmr_base_reduction(depth: Depth, move_count: usize, n_moves: usize) -> Depth {
    let mut reduction = 1;
    if depth >= midgame::LMR_DEEPER_DEPTH && move_count > 5 {
        reduction += 1;
    }
    if depth >= 16 && move_count > 10 && move_count * 5 >= n_moves * 3 {
        reduction += 1;
    }
    reduction
}

#[inline(always)]
fn lmr_max_reduction(depth: Depth) -> Depth {
    (depth / 3).max(1).min(depth - 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probcut::Selectivity;
    use crate::search::midgame::{LMR_DEEPER_DEPTH, LMR_MIN_DEPTH};
    use crate::search::node_type::{NonPV, PV};
    use crate::search::strategy::{EndGameStrategy, MidGameStrategy};

    #[test]
    fn no_reduction_below_the_gating_thresholds() {
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_MIN_DEPTH - 1,
                10,
                10,
                true
            ),
            0
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                2,
                10,
                true
            ),
            0
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                6,
                3,
                true
            ),
            0
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::None,
                LMR_DEEPER_DEPTH,
                6,
                10,
                true
            ),
            0
        );
    }

    #[test]
    fn staged_lmr_base_scales_with_depth_and_relative_move_index() {
        assert_eq!(lmr_base_reduction(LMR_MIN_DEPTH, 3, 4), 1);
        assert_eq!(lmr_base_reduction(LMR_DEEPER_DEPTH, 6, 10), 2);
        assert_eq!(lmr_base_reduction(15, 16, 20), 2);
        assert_eq!(lmr_base_reduction(16, 11, 20), 2);
        assert_eq!(lmr_base_reduction(16, 12, 20), 3);
        assert_eq!(lmr_base_reduction(24, 21, 34), 3);
    }

    #[test]
    fn cut_nodes_use_the_full_lmr_schedule() {
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_MIN_DEPTH,
                3,
                4,
                true
            ),
            1
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                6,
                10,
                true
            ),
            2
        );
    }

    #[test]
    fn all_nodes_reduce_less_aggressively_while_enabled_selectivity_levels_share_schedule() {
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(Selectivity::Level1, 16, 11, 20, true),
            2
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(Selectivity::Level1, 16, 12, 20, true),
            3
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(Selectivity::Level1, 16, 12, 20, false),
            2
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(Selectivity::Level2, 16, 12, 20, true),
            3
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(Selectivity::Level3, 16, 12, 20, true),
            3
        );
    }

    #[test]
    fn pv_nodes_are_never_reduced() {
        assert_eq!(
            compute_lmr_reduction::<PV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                6,
                10,
                false
            ),
            0
        );
        assert_eq!(
            compute_lmr_reduction::<PV, MidGameStrategy>(Selectivity::Level1, 15, 16, 20, false),
            0
        );
    }

    #[test]
    fn endgame_nodes_are_never_reduced() {
        assert_eq!(
            compute_lmr_reduction::<NonPV, EndGameStrategy>(Selectivity::Level1, 30, 16, 20, true),
            0
        );
    }

    #[test]
    fn reduction_is_capped_by_depth_and_node_type() {
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_MIN_DEPTH,
                16,
                20,
                true
            ),
            1
        );
        assert_eq!(lmr_max_reduction(30), 10);
    }
}
