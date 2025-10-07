use crate::{
    probcut,
    square::Square,
    types::{Depth, Scoref},
};

pub struct SearchResult {
    pub score: Scoref,
    pub best_move: Option<Square>,
    pub n_nodes: u64,
    pub pv_line: Vec<Square>,
    pub depth: Depth,
    pub selectivity: u8,
}

impl SearchResult {
    pub fn get_probability(&self) -> i32 {
        probcut::get_probability(self.selectivity)
    }
}
