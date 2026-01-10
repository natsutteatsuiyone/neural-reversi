//! Search phase trait and implementations for midgame and endgame specialization.

use std::sync::Arc;

use crate::board::Board;
use crate::search::endgame;
use crate::search::midgame;
use crate::search::search_context::SearchContext;
use crate::search::threading::Thread;
use crate::types::{Depth, ScaledScore};

/// Marker trait for game phase specialization in search.
pub trait SearchPhase: Copy + Clone + 'static {
    /// Whether this is an endgame search (used for TT storage flag).
    const IS_ENDGAME: bool;

    /// Whether to use Late Move Reduction (LMR).
    const USE_SBR: bool;

    /// Minimum depth for Enhanced Transposition Cutoff (0 = always apply).
    const MIN_ETC_DEPTH: Depth;

    /// Minimum depth for ProbCut (0 = always apply).
    const MIN_PROBCUT_DEPTH: Depth;

    /// Maximum depth for shallow search specialization.
    const DEPTH_TO_SHALLOW: Depth;

    /// Minimum depth required for parallel search splitting.
    const MIN_SPLIT_DEPTH: Depth;

    /// Evaluates a terminal PV node position.
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
    ) -> ScaledScore;

    /// Calls the phase-specific probcut implementation.
    fn probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
        thread: &Arc<Thread>,
    ) -> Option<ScaledScore>;
}

/// Midgame search phase marker.
#[derive(Copy, Clone)]
pub struct MidGamePhase;

/// Endgame search phase marker.
#[derive(Copy, Clone)]
pub struct EndGamePhase;

impl SearchPhase for MidGamePhase {
    const IS_ENDGAME: bool = false;
    const USE_SBR: bool = true;
    const MIN_ETC_DEPTH: Depth = 6;
    const MIN_PROBCUT_DEPTH: Depth = 3;
    const DEPTH_TO_SHALLOW: Depth = 2;
    const MIN_SPLIT_DEPTH: Depth = 5;

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
    ) -> ScaledScore {
        match depth {
            0 => midgame::evaluate(ctx, board),
            1 => midgame::evaluate_depth1(ctx, board, alpha, beta),
            _ => midgame::evaluate_depth2(ctx, board, alpha, beta),
        }
    }

    #[inline(always)]
    fn probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
        thread: &Arc<Thread>,
    ) -> Option<ScaledScore> {
        midgame::probcut(ctx, board, depth, beta, thread)
    }
}

impl SearchPhase for EndGamePhase {
    const IS_ENDGAME: bool = true;
    const USE_SBR: bool = false;
    const MIN_ETC_DEPTH: Depth = 0;
    const MIN_PROBCUT_DEPTH: Depth = 0;
    const DEPTH_TO_SHALLOW: Depth = endgame::DEPTH_TO_NWS;
    const MIN_SPLIT_DEPTH: Depth = 9;

    #[inline(always)]
    fn evaluate(_ctx: &SearchContext, board: &Board) -> ScaledScore {
        ScaledScore::from_disc_diff(endgame::calculate_final_score(board))
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
    fn probcut(
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        beta: ScaledScore,
        thread: &Arc<Thread>,
    ) -> Option<ScaledScore> {
        endgame::probcut(ctx, board, depth, beta, thread)
    }
}
