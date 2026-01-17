//! Search result types.

use crate::{
    probcut::Selectivity,
    search::root_move::{RootMove, RootMoves},
    square::Square,
    types::{Depth, Scoref},
};

/// Represents a single move with its evaluation score for Multi-PV results.
#[derive(Clone, Debug)]
pub struct PvMove {
    pub sq: Square,
    pub score: Scoref,
    pub pv_line: Vec<Square>,
}

/// Result of a search operation.
pub struct SearchResult {
    pub score: Scoref,
    pub best_move: Option<Square>,
    pub n_nodes: u64,
    pub pv_line: Vec<Square>,
    pub depth: Depth,
    pub selectivity: Selectivity,
    pub is_endgame: bool,
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
            is_endgame: false,
            pv_moves: vec![],
        }
    }

    /// Creates a SearchResult from the search context state.
    ///
    /// # Arguments
    ///
    /// * `root_moves` - Container for root moves.
    /// * `best_move` - Best move found during search.
    /// * `n_nodes` - Total nodes searched.
    /// * `depth` - Search depth reached.
    /// * `selectivity` - Selectivity level used.
    /// * `is_endgame` - Whether this is an endgame search.
    pub fn from_root_move(
        root_moves: &RootMoves,
        best_move: &RootMove,
        n_nodes: u64,
        depth: Depth,
        selectivity: Selectivity,
        is_endgame: bool,
    ) -> Self {
        let pv_moves: Vec<PvMove> = root_moves.map(|rm| PvMove {
            sq: rm.sq,
            score: rm.score.to_disc_diff_f32(),
            pv_line: rm.pv.clone(),
        });

        Self {
            score: best_move.score.to_disc_diff_f32(),
            best_move: Some(best_move.sq),
            n_nodes,
            pv_line: best_move.pv.clone(),
            depth,
            selectivity,
            is_endgame,
            pv_moves,
        }
    }

    /// Returns the probability percentage based on selectivity.
    pub fn get_probability(&self) -> i32 {
        self.selectivity.probability()
    }
}
