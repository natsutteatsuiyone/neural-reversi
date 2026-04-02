use js_sys::Function;
use std::rc::Rc;

use reversi_core::{board::Board, probcut::Selectivity};

use crate::transposition_table::TranspositionTable;

use crate::{eval::Eval, level::Level};

/// Bundles the inputs needed to launch a single search.
pub struct SearchTask {
    /// Board position to search.
    pub board: Board,
    /// Depth limits for each game phase.
    pub level: Level,
    /// Selectivity level for ProbCut pruning.
    pub selectivity: Selectivity,
    /// Shared transposition table.
    pub tt: Rc<TranspositionTable>,
    /// Shared neural network evaluator.
    pub eval: Rc<Eval>,
    /// Optional JavaScript callback for progress reporting.
    pub progress_callback: Option<Function>,
}
