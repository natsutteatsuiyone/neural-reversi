// Runs the native aspiration loop on one worker with initially empty tables.
use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::score::Score;
use std::path::Path;

pub fn run(weights: &Path) {
    let eval = Arc::new(
        Eval::with_weight_files(
            Some(&weights.join(EVAL_FILE_NAME)),
            Some(&weights.join(EVAL_SM_FILE_NAME)),
        )
        .unwrap(),
    );
    let board = Board::from_bitboards(0x7e5032204c1e3c24, 0x81afcddfb3e1c080);
    let tt = TranspositionTable::new(0);
    let pool = ThreadPool::new(1);
    let moves = root_move::RootMoves::new(&board);
    let mut ctx = context::SearchContext::new(&board, Selectivity::None, &moves, &tt, &eval);
    let mut alpha = Score::from_raw(0);
    let mut beta = Score::from_raw(256);
    let value = endgame::aspiration_search(&mut ctx, &board, &mut alpha, &mut beta, pool.main());
    assert_eq!(value.value(), -5632);
}
