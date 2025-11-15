use js_sys::Function;
use std::rc::Rc;

use reversi_core::{board::Board, transposition_table::TranspositionTable};

use crate::{eval::Eval, level::Level};

pub struct SearchTask {
    pub board: Board,
    pub level: Level,
    pub generation: u8,
    pub selectivity: u8,
    pub tt: Rc<TranspositionTable>,
    pub eval: Rc<Eval>,
    pub progress_callback: Option<Function>,
}
