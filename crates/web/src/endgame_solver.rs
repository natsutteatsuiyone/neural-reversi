use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::types::Depth;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::config::{DEFAULT_TT_MB, MIDGAME_SELECTIVITY};
use crate::eval::Eval;
use crate::level::Level;
use crate::search::Search;
use crate::transposition_table::TranspositionTable;

/// Outcome of a single endgame solve exposed to JavaScript.
#[wasm_bindgen]
pub struct EndgameSolveResult {
    score: f32,
    best_move: String,
    n_nodes: f64,
}

#[wasm_bindgen]
impl EndgameSolveResult {
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f32 {
        self.score
    }

    #[wasm_bindgen(getter)]
    pub fn best_move(&self) -> String {
        self.best_move.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn n_nodes(&self) -> f64 {
        self.n_nodes
    }
}

/// Endgame solver entry point used by the CLI benchmarking front end.
#[wasm_bindgen]
pub struct EndgameSolver {
    search: Search,
}

#[wasm_bindgen]
impl EndgameSolver {
    /// Creates an endgame solver with the given transposition table size in MB.
    ///
    /// # Errors
    ///
    /// Returns an error if the evaluation network fails to load.
    #[wasm_bindgen(constructor)]
    pub fn new(tt_mb: Option<u32>) -> Result<EndgameSolver, JsValue> {
        console_error_panic_hook::set_once();

        let tt_mb = tt_mb.unwrap_or(DEFAULT_TT_MB as u32) as usize;
        let tt = Rc::new(TranspositionTable::new(tt_mb));
        let eval = Rc::new(Eval::new().map_err(|e| {
            JsValue::from_str(&format!("Failed to load evaluation network: {}", e))
        })?);

        Ok(EndgameSolver {
            search: Search::new(tt, eval),
        })
    }

    pub fn solve(&mut self, board_str: &str, side: u8) -> Result<EndgameSolveResult, JsValue> {
        let disc = if side == 0 { Disc::Black } else { Disc::White };
        let board = Board::from_string(board_str, disc)
            .map_err(|e| JsValue::from_str(&format!("Invalid board: {}", e)))?;

        let is_pass = !board.has_legal_moves();
        let search_board = if is_pass {
            board.switch_players()
        } else {
            board
        };
        if !search_board.has_legal_moves() {
            return Ok(EndgameSolveResult {
                score: board.solve(board.empty_count()) as f32,
                best_move: String::new(),
                n_nodes: 0.0,
            });
        }

        let empty_count = board.empty_count() as Depth;
        let level = Level {
            mid_depth: empty_count,
            end_depth: empty_count,
            perfect_depth: empty_count,
        };

        let result = self
            .search
            .run(&search_board, level, MIDGAME_SELECTIVITY, None, false);

        if is_pass {
            return Ok(EndgameSolveResult {
                score: -result.score,
                best_move: String::new(),
                n_nodes: result.n_nodes as f64,
            });
        }
        Ok(EndgameSolveResult {
            score: result.score,
            best_move: result
                .best_move
                .map(|sq| sq.to_string())
                .unwrap_or_default(),
            n_nodes: result.n_nodes as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_scores_positions_without_a_legal_move() {
        let mut solver = EndgameSolver::new(Some(1)).unwrap();

        let full = solver.solve(&"X".repeat(64), 0).unwrap();
        assert_eq!(full.score(), 64.0);
        assert_eq!(full.best_move(), "");

        // White must pass; Black then plays g1 and takes every disc.
        let pass = solver
            .solve(&format!("XXXOOO--{}", "-".repeat(56)), 1)
            .unwrap();
        assert_eq!(pass.score(), -64.0);
        assert_eq!(pass.best_move(), "");
    }
}
