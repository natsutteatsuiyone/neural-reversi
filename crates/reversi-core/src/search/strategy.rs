//! Search strategy trait and implementations for midgame and endgame specialization.

use std::sync::Arc;

use crate::board::Board;
use crate::probcut::Selectivity;
use crate::score::Score;
use crate::search::context::SearchContext;
use crate::search::endgame;
use crate::search::midgame;
use crate::search::node_type::NonPV;
use crate::search::threading::Thread;
use crate::transposition_table::Bound;
use crate::types::Depth;

/// Marker trait for search strategy specialization.
pub trait SearchStrategy: 'static {
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
    fn evaluate(ctx: &SearchContext, board: &Board) -> Score;

    /// Shallow-depth specialized search for NonPV nodes.
    ///
    /// Called when `depth <= DEPTH_TO_SHALLOW`.
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: Score,
        beta: Score,
        thread: &Arc<Thread>,
    ) -> Score;

    /// Calls the phase-specific probcut implementation.
    ///
    /// `static_eval` is the corrected static evaluation, present only where
    /// midgame ProbCut can run.
    fn try_probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: Score,
        cut_node: bool,
        static_eval: Option<Score>,
        thread: &Arc<Thread>,
    ) -> Option<Score>;
}

/// Midgame search strategy marker.
pub struct MidGameStrategy;

/// Endgame search strategy marker.
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
    fn evaluate(ctx: &SearchContext, board: &Board) -> Score {
        midgame::evaluate(ctx, board)
    }

    #[inline(always)]
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: Score,
        beta: Score,
        _thread: &Arc<Thread>,
    ) -> Score {
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
        beta: Score,
        cut_node: bool,
        static_eval: Option<Score>,
        thread: &Arc<Thread>,
    ) -> Option<Score> {
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
    fn evaluate(_ctx: &SearchContext, board: &Board) -> Score {
        board.final_score()
    }

    #[inline(always)]
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: Score,
        beta: Score,
        thread: &Arc<Thread>,
    ) -> Score {
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

        let nodes_before = ctx.n_nodes;
        let caches = thread.endgame_caches();
        let empties = endgame::Empties::new(board);
        let n_empties = empties.count();
        let score_d = if n_empties > endgame::DEPTH_TO_SHALLOW_SEARCH {
            let moves = board.legal_moves();
            if thread.has_active_split_point() {
                endgame::null_window_search(ctx, board, alpha_d, caches, &**thread, moves, empties)
            } else {
                endgame::null_window_search(
                    ctx,
                    board,
                    alpha_d,
                    caches,
                    &endgame::IgnoreSplitPointCutoff,
                    moves,
                    empties,
                )
            }
        } else {
            match n_empties {
                0 => board.final_disc_diff(),
                1 => {
                    let [sq] = empties.first();
                    endgame::solve1(ctx, board.player(), alpha_d, sq)
                }
                2 => {
                    let [sq1, sq2] = empties.first();
                    endgame::solve2(ctx, board, alpha_d, sq1, sq2)
                }
                3 => {
                    let [sq1, sq2, sq3] = empties.first();
                    endgame::solve3(ctx, board, alpha_d, sq1, sq2, sq3)
                }
                4 => {
                    let odd_quadrants = endgame::odd_quadrants(board.empty_squares());
                    let (sq1, sq2, sq3, sq4) = endgame::sort_last4(empties, odd_quadrants);
                    endgame::solve4(ctx, board, alpha_d, sq1, sq2, sq3, sq4)
                }
                5 => endgame::shallow_search::<5>(
                    ctx,
                    board,
                    alpha_d,
                    &mut caches.shallow,
                    board.legal_moves(),
                    empties,
                ),
                _ => endgame::shallow_search::<6>(
                    ctx,
                    board,
                    alpha_d,
                    &mut caches.shallow,
                    board.legal_moves(),
                    empties,
                ),
            }
        };

        let score = Score::from_disc_diff(score_d);

        if let Some(slot) = tt_slot
            && ctx.n_nodes - nodes_before >= BOUNDARY_STORE_MIN_SUBTREE_NODES
            && !thread.should_stop()
        {
            ctx.tt.store(
                slot,
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
        beta: Score,
        cut_node: bool,
        _static_eval: Option<Score>,
        _thread: &Arc<Thread>,
    ) -> Option<Score> {
        endgame::try_probcut(ctx, board, depth, beta, cut_node)
    }
}
