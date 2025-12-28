use js_sys::Function;
use std::rc::Rc;

use reversi_core::{board::Board, transposition_table::TranspositionTable, types::Selectivity};

use crate::{eval::Eval, level::Level};

pub struct SearchTask {
    pub board: Board,
    pub level: Level,
    pub selectivity: Selectivity,
    pub tt: Rc<TranspositionTable>,
    pub eval: Rc<Eval>,
    pub progress_callback: Option<Function>,
}
