//! Search strategy trait and implementations for midgame and endgame specialization.

use std::sync::Arc;

use crate::board::Board;
use crate::probcut::Selectivity;
use crate::search::context::SearchContext;
use crate::search::endgame;
use crate::search::midgame;
use crate::search::node_type::NonPV;
use crate::search::threading::Thread;
use crate::transposition_table::Bound;
use crate::types::{Depth, ScaledScore};

/// Marker trait for search strategy specialization.
pub trait SearchStrategy: Copy + Clone + 'static {
    /// Whether this is an endgame search (used for TT storage flag).
    const IS_ENDGAME: bool;

    /// Minimum depth for Enhanced Transposition Cutoff (0 = always apply).
    const MIN_ETC_DEPTH: Depth;

    /// Minimum depth for ProbCut (0 = always apply).
    const MIN_PROBCUT_DEPTH: Depth;

    /// Maximum depth for shallow search specialization.
    const DEPTH_TO_SHALLOW: Depth;

    /// Minimum node depth at which children are claimed in the in-flight table.
    const MIN_CLAIM_DEPTH: Depth;

    /// Evaluates a PV leaf node position (depth == 0).
    fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore;

    /// Shallow-depth specialized search for NonPV nodes.
    ///
    /// Called when `depth <= DEPTH_TO_SHALLOW`.
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: ScaledScore,
        beta: ScaledScore,
        thread: &Arc<Thread>,
    ) -> ScaledScore;

    /// Calls the phase-specific probcut implementation.
    ///
    /// `static_eval` is the corrected static evaluation, present only where
    /// midgame ProbCut can run.
    fn try_probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
        cut_node: bool,
        static_eval: Option<ScaledScore>,
        thread: &Arc<Thread>,
    ) -> Option<ScaledScore>;
}

/// Midgame search strategy marker.
#[derive(Copy, Clone)]
pub struct MidGameStrategy;

/// Endgame search strategy marker.
#[derive(Copy, Clone)]
pub struct EndGameStrategy;

/// Minimum NWS subtree size, in nodes, required to share a boundary result.
const BOUNDARY_STORE_MIN_SUBTREE_NODES: u64 = 256;

impl SearchStrategy for MidGameStrategy {
    const IS_ENDGAME: bool = false;
    const MIN_ETC_DEPTH: Depth = 5;
    const MIN_PROBCUT_DEPTH: Depth = 4;
    const DEPTH_TO_SHALLOW: Depth = 3;
    const MIN_CLAIM_DEPTH: Depth = Depth::MAX;

    #[inline(always)]
    fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore {
        midgame::evaluate(ctx, board)
    }

    #[inline(always)]
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: ScaledScore,
        beta: ScaledScore,
        _thread: &Arc<Thread>,
    ) -> ScaledScore {
        match depth {
            0 => midgame::evaluate(ctx, board),
            1 => midgame::evaluate_depth1(ctx, board, alpha, beta),
            _ => midgame::evaluate_shallow_with_tt(ctx, board, depth, alpha, beta),
        }
    }

    #[inline(always)]
    fn try_probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
        cut_node: bool,
        static_eval: Option<ScaledScore>,
        thread: &Arc<Thread>,
    ) -> Option<ScaledScore> {
        midgame::try_probcut(ctx, board, depth, beta, cut_node, static_eval?, thread)
    }
}

impl SearchStrategy for EndGameStrategy {
    const IS_ENDGAME: bool = true;
    const MIN_ETC_DEPTH: Depth = endgame::DEPTH_TO_NWS + 2;
    const MIN_PROBCUT_DEPTH: Depth = endgame::DEPTH_TO_NWS + 1;
    const DEPTH_TO_SHALLOW: Depth = endgame::DEPTH_TO_NWS;
    const MIN_CLAIM_DEPTH: Depth = 19;

    #[inline(always)]
    fn evaluate(_ctx: &SearchContext, board: &Board) -> ScaledScore {
        board.final_score_scaled()
    }

    #[inline(always)]
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: ScaledScore,
        beta: ScaledScore,
        thread: &Arc<Thread>,
    ) -> ScaledScore {
        let alpha_d = alpha.to_disc_diff();

        // ProbCut never runs below this depth, so a boundary result is
        // selectivity-independent and can be shared at Selectivity::None.
        let tt_slot = if depth == endgame::DEPTH_TO_NWS && thread.is_parallel_pool() {
            let probe = ctx.tt.probe(board, board.hash());
            if let Some(tt_data) = probe.data()
                && tt_data.can_cut(beta, depth, ctx.selectivity, true)
            {
                return tt_data.score();
            }
            Some(probe.slot())
        } else {
            None
        };

        let nodes_before = ctx.counters.n_nodes;
        let caches = thread.endgame_caches();
        let n_empties = ctx.empty_list.count();
        let score_d = if n_empties > endgame::DEPTH_TO_SHALLOW_SEARCH {
            let moves = board.legal_moves();
            if thread.has_active_split_point() {
                endgame::null_window_search(ctx, board, alpha_d, caches, &**thread, moves)
            } else {
                endgame::null_window_search(
                    ctx,
                    board,
                    alpha_d,
                    caches,
                    &endgame::IgnoreSplitPointCutoff,
                    moves,
                )
            }
        } else {
            match n_empties {
                0 => board.final_score(),
                1 => {
                    let sq = ctx.empty_list.first();
                    endgame::solve1(ctx, board.player(), alpha_d, sq)
                }
                2 => {
                    let sq1 = ctx.empty_list.first();
                    let sq2 = ctx.empty_list.next(sq1);
                    endgame::solve2(ctx, board, alpha_d, sq1, sq2)
                }
                3 => {
                    let sq1 = ctx.empty_list.first();
                    let sq2 = ctx.empty_list.next(sq1);
                    let sq3 = ctx.empty_list.next(sq2);
                    endgame::solve3(ctx, board, alpha_d, sq1, sq2, sq3)
                }
                4 => {
                    let (sq1, sq2, sq3, sq4) = endgame::sort_last4(&ctx.empty_list);
                    endgame::solve4(ctx, board, alpha_d, sq1, sq2, sq3, sq4)
                }
                _ => endgame::shallow_search(
                    ctx,
                    board,
                    alpha_d,
                    &mut caches.shallow,
                    board.legal_moves(),
                ),
            }
        };

        let score = ScaledScore::from_disc_diff(score_d);

        if let Some(tt_index) = tt_slot
            && ctx.counters.n_nodes - nodes_before >= BOUNDARY_STORE_MIN_SUBTREE_NODES
            && !thread.should_stop()
        {
            ctx.tt.store(
                tt_index,
                board,
                score,
                Bound::classify::<NonPV>(score, alpha, beta),
                depth,
                None,
                Selectivity::None,
                true,
                false,
            );
        }

        score
    }

    #[inline(always)]
    fn try_probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
        cut_node: bool,
        _static_eval: Option<ScaledScore>,
        _thread: &Arc<Thread>,
    ) -> Option<ScaledScore> {
        endgame::try_probcut(ctx, board, depth, beta, cut_node)
    }
}
