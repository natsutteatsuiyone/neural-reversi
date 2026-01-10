use crate::{
    search::{
        GamePhase,
        root_move::{RootMove, RootMoves},
    },
    square::Square,
    types::{Depth, Scoref, Selectivity},
};

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

    /// Creates a SearchResult from the search context state.
    ///
    /// # Arguments
    ///
    /// * `root_moves` - Container for root moves
    /// * `best_move` - The best move found during search
    /// * `n_nodes` - Total nodes searched
    /// * `depth` - Search depth reached
    /// * `selectivity` - Selectivity level used
    /// * `game_phase` - Current game phase (MidGame or EndGame)
    pub fn from_root_move(
        root_moves: &RootMoves,
        best_move: &RootMove,
        n_nodes: u64,
        depth: Depth,
        selectivity: Selectivity,
        game_phase: GamePhase,
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
            game_phase,
            pv_moves,
        }
    }

    pub fn get_probability(&self) -> i32 {
        self.selectivity.probability()
    }
}
