use reversi_core::types::Depth;

/// Holds search depth settings for each game phase.
#[derive(Debug, Clone)]
pub struct Level {
    /// Search depth for the midgame phase.
    pub mid_depth: Depth,

    /// Search depth for the endgame phase.
    pub end_depth: Depth,

    /// Search depth for the perfect-play phase.
    pub perfect_depth: Depth,
}
