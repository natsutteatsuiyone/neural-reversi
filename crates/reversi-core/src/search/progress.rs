//! Search progress reporting.

use crate::search::context::SearchContext;
use crate::search::counters::SearchCounters;
use crate::search::result::SearchResult;
use crate::search::root_move::RootMove;
use crate::square::Square;
use crate::types::{Depth, ScaledScore, Scoref};

/// Progress information reported during an ongoing search.
pub struct SearchProgress {
    /// Current search depth completed.
    pub depth: Depth,
    /// Target search depth for this iteration.
    pub target_depth: Depth,
    /// Best score found so far (in disc difference).
    pub score: Scoref,
    /// Best move found so far.
    pub best_move: Square,
    /// Probability percentage from the current [`Selectivity`] level.
    ///
    /// [`Selectivity`]: crate::probcut::Selectivity
    pub probability: i32,
    /// Total nodes searched.
    pub nodes: u64,
    /// Principal variation (sequence of best moves).
    pub pv_line: Vec<Square>,
    /// Whether the search is in endgame phase.
    pub is_endgame: bool,
    /// Snapshot of search counters at this point.
    pub counters: SearchCounters,
}

/// Callback invoked to report [`SearchProgress`] during a search.
pub type SearchProgressCallback = dyn Fn(SearchProgress) + Send + Sync + 'static;

impl SearchProgress {
    /// Builds the progress report for a root iteration from the search context
    /// and the iteration's best root move.
    pub(crate) fn from_iteration(
        ctx: &SearchContext,
        rm: &RootMove,
        depth: Depth,
        target_depth: Depth,
        score: ScaledScore,
        is_endgame: bool,
    ) -> Self {
        Self {
            depth,
            target_depth,
            score: score.to_disc_diff_f32(),
            best_move: rm.sq,
            probability: ctx.selectivity.probability(),
            nodes: ctx.counters.n_nodes,
            pv_line: rm.pv.clone(),
            is_endgame,
            counters: ctx.counters.clone(),
        }
    }

    /// Builds the final progress report from a completed search result.
    pub(crate) fn from_result(result: &SearchResult) -> Self {
        Self {
            depth: result.depth(),
            target_depth: result.depth(),
            score: result.score().unwrap_or(0.0),
            probability: result.get_probability(),
            best_move: result.best_move().unwrap_or(Square::None),
            nodes: result.n_nodes(),
            pv_line: result.pv_line().to_vec(),
            is_endgame: result.is_endgame(),
            counters: result.counters(),
        }
    }
}
