use reversi_core::types::Depth;

/// Level settings for the AI player
#[derive(Debug, Clone)]
pub struct Level {
    /// Search depth for mid game phase
    pub mid_depth: Depth,

    /// Search depth for end game phase
    pub end_depth: Depth,

    /// Search depth for perfect play phase
    pub perfect_depth: Depth,
}
