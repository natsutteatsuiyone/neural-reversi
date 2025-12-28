use crate::{
    square::Square,
    types::{Depth, Scoref, Selectivity},
};

use super::search_context::GamePhase;

/// Represents a single move with its evaluation score for Multi-PV results.
#[derive(Clone, Debug)]
pub struct PvMove {
    pub sq: Square,
    pub score: Scoref,
    pub pv_line: Vec<Square>,
}

pub struct SearchResult {
    pub score: Scoref,
    pub best_move: Option<Square>,
    pub n_nodes: u64,
    pub pv_line: Vec<Square>,
    pub depth: Depth,
    pub selectivity: Selectivity,
    pub game_phase: GamePhase,
    /// All evaluated moves with scores (populated in Multi-PV mode).
    pub pv_moves: Vec<PvMove>,
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
            selectivity: Selectivity::None,
            game_phase: GamePhase::MidGame,
            pv_moves: vec![],
        }
    }

    pub fn get_probability(&self) -> i32 {
        self.selectivity.probability()
    }
}
