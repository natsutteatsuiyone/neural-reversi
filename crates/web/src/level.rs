use reversi_core::types::Depth;

/// Holds search depth settings for each game phase.
#[derive(Debug, Clone, Copy)]
pub struct Level {
    /// Search depth used during the midgame.
    pub mid_depth: Depth,
    /// Empty-square threshold at which endgame search begins.
    pub end_depth: Depth,
    /// Empty-square threshold at which exact (no-selectivity) search begins.
    pub perfect_depth: Depth,
}
