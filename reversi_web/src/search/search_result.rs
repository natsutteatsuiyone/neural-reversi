use reversi_core::{
    square::Square,
    types::{Depth, Scoref},
};

pub struct SearchResult {
    #[allow(dead_code)]
    pub score: Scoref,
    pub best_move: Option<Square>,
    #[allow(dead_code)]
    pub depth: Depth,
    #[allow(dead_code)]
    pub n_nodes: u64,
    #[allow(dead_code)]
    pub selectivity: u8,
}
