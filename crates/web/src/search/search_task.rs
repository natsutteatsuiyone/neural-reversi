use js_sys::Function;
use std::rc::Rc;

use reversi_core::{board::Board, probcut::Selectivity, transposition_table::TranspositionTable};

use crate::{eval::Eval, level::Level};

pub struct SearchTask {
    pub board: Board,
    pub level: Level,
    pub selectivity: Selectivity,
    pub tt: Rc<TranspositionTable>,
    pub eval: Rc<Eval>,
    pub progress_callback: Option<Function>,
}
