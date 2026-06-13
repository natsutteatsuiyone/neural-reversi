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
    depth: u32,
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

    #[wasm_bindgen(getter)]
    pub fn depth(&self) -> u32 {
        self.depth
    }
}

/// Endgame solver entry point used by the CLI benchmarking front end.
#[wasm_bindgen]
pub struct EndgameSolver {
    search: Search,
    tt: Rc<TranspositionTable>,
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
        let search = Search::new(Rc::clone(&tt), eval);

        Ok(EndgameSolver { search, tt })
    }

    pub fn solve(&mut self, board_str: &str, side: u8) -> Result<EndgameSolveResult, JsValue> {
        let disc = if side == 0 { Disc::Black } else { Disc::White };
        let board = Board::from_string(board_str, disc)
            .map_err(|e| JsValue::from_str(&format!("Invalid board: {}", e)))?;

        let empty_count = board.get_empty_count() as Depth;
        let level = Level {
            mid_depth: empty_count,
            end_depth: empty_count,
            perfect_depth: empty_count,
        };

        let result = self.search.run(&board, level, MIDGAME_SELECTIVITY, None);

        Ok(EndgameSolveResult {
            score: result.score,
            best_move: result
                .best_move
                .map(|sq| sq.to_string())
                .unwrap_or_default(),
            n_nodes: result.n_nodes as f64,
            depth: result.depth,
        })
    }

    pub fn clear_tt(&self) {
        self.tt.clear();
    }
}
