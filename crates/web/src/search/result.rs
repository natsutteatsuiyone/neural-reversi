use reversi_core::{score::Score, square::Square};

/// Holds the outcome of a completed search.
pub struct SearchResult {
    /// Best score found.
    pub score: Score,
    /// Best move, or [`None`] if no legal moves exist.
    pub best_move: Option<Square>,
    /// Total nodes visited.
    pub n_nodes: u64,
    /// Per-root-move final scores (side-to-move perspective).
    /// Populated only by multi-PV (hint) searches; empty otherwise.
    pub multi_pv_scores: Vec<(Square, Score)>,
}
