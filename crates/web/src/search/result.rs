use reversi_core::{
    probcut::Selectivity,
    square::Square,
    types::{Depth, Scoref},
};

/// Holds the outcome of a completed search.
pub struct SearchResult {
    /// Best score found (disc-difference scale).
    #[allow(dead_code)]
    pub score: Scoref,
    /// Best move, or [`None`] if no legal moves exist.
    pub best_move: Option<Square>,
    /// Search depth reached.
    #[allow(dead_code)]
    pub depth: Depth,
    /// Total nodes visited.
    #[allow(dead_code)]
    pub n_nodes: u64,
    /// Selectivity level used.
    #[allow(dead_code)]
    pub selectivity: Selectivity,
    /// Per-root-move final scores (disc diff, side-to-move perspective).
    /// Populated only by multi-PV (hint) searches; empty otherwise.
    pub multi_pv_scores: Vec<(Square, Scoref)>,
}
