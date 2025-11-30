use crate::{
    probcut,
    square::Square,
    types::{Depth, Scoref},
};

use super::search_context::GamePhase;

pub struct SearchResult {
    pub score: Scoref,
    pub best_move: Option<Square>,
    pub n_nodes: u64,
    pub pv_line: Vec<Square>,
    pub depth: Depth,
    pub selectivity: u8,
    pub game_phase: GamePhase,
}

impl SearchResult {
    /// Creates a result for a random move in the opening position.
    pub fn new_random_move(mv: Square) -> Self {
        Self {
            score: 0.0,
            best_move: Some(mv),
            n_nodes: 0,
            pv_line: vec![],
            depth: 0,
            selectivity: probcut::NO_SELECTIVITY,
            game_phase: GamePhase::MidGame,
        }
    }

    pub fn get_probability(&self) -> i32 {
        probcut::get_probability(self.selectivity)
    }
}
