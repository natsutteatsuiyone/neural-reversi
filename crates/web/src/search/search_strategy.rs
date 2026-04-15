use reversi_core::{
    board::Board,
    types::{Depth, ScaledScore},
};

use crate::probcut as web_probcut;

use super::{endgame, search_context::SearchContext};

/// Marker trait for search strategy specialization.
pub(crate) trait SearchStrategy: Copy + Clone + 'static {
    /// Whether this strategy stores positions as endgame entries in the TT.
    const IS_ENDGAME: bool;

    /// Minimum depth for Enhanced Transposition Cutoff.
    const MIN_ETC_DEPTH: Depth;

    /// Minimum depth for ProbCut.
    const MIN_PROBCUT_DEPTH: Depth;

    /// Maximum depth handled by strategy-specific shallow search.
    const DEPTH_TO_SHALLOW: Depth;

    /// Evaluates a leaf position.
    fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore;

    /// Performs a shallow search for NonPV nodes.
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: ScaledScore,
        beta: ScaledScore,
    ) -> ScaledScore;

    /// Attempts a strategy-specific ProbCut.
    fn try_probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
    ) -> Option<ScaledScore>;
}

/// Midgame search strategy marker.
#[derive(Copy, Clone)]
pub(crate) struct MidGameStrategy;

/// Endgame search strategy marker.
#[derive(Copy, Clone)]
pub(crate) struct EndGameStrategy;

impl SearchStrategy for MidGameStrategy {
    const IS_ENDGAME: bool = false;
    const MIN_ETC_DEPTH: Depth = 6;
    const MIN_PROBCUT_DEPTH: Depth = 3;
    const DEPTH_TO_SHALLOW: Depth = 2;

    #[inline(always)]
    fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore {
        super::evaluate(ctx, board)
    }

    #[inline(always)]
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        alpha: ScaledScore,
        beta: ScaledScore,
    ) -> ScaledScore {
        match depth {
            0 => super::evaluate(ctx, board),
            1 => super::evaluate_depth1(ctx, board, alpha, beta),
            2 => super::evaluate_depth2(ctx, board, alpha, beta),
            _ => unreachable!("midgame shallow search only supports depth <= 2"),
        }
    }

    #[inline(always)]
    fn try_probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
    ) -> Option<ScaledScore> {
        web_probcut::probcut_midgame(ctx, board, depth, beta)
    }
}

impl SearchStrategy for EndGameStrategy {
    const IS_ENDGAME: bool = true;
    const MIN_ETC_DEPTH: Depth = 15;
    const MIN_PROBCUT_DEPTH: Depth = 13;
    const DEPTH_TO_SHALLOW: Depth = endgame::DEPTH_TO_NWS;

    #[inline(always)]
    fn evaluate(_ctx: &SearchContext, board: &Board) -> ScaledScore {
        board.final_score_scaled()
    }

    #[inline(always)]
    fn shallow_search(
        ctx: &mut SearchContext,
        board: &Board,
        _depth: Depth,
        alpha: ScaledScore,
        _beta: ScaledScore,
    ) -> ScaledScore {
        let score = endgame::null_window_search(ctx, board, alpha.to_disc_diff());
        ScaledScore::from_disc_diff(score)
    }

    #[inline(always)]
    fn try_probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
    ) -> Option<ScaledScore> {
        web_probcut::probcut_midgame(ctx, board, depth, beta)
    }
}
