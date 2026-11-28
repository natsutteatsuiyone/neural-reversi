use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use std::path::Path;

pub fn run(weights: &Path) {
    let eval = Arc::new(Eval::with_weight_files(
        Some(&weights.join(EVAL_FILE_NAME)),
        Some(&weights.join(EVAL_SM_FILE_NAME)),
    ).unwrap());
    let resources = crate::engine::Engine::from_eval(eval, 0, Some(1));
    let search = resources.open();
    let board = Board::from_bitboards(0x7a2e541b1fa14181, 0x81d1abe4e05e2e5e);
    assert_eq!(board.empty_count(), 4);
    for cap in 1..=5 {
        println!("CAP {cap}");
        let depths = std::array::from_fn(|i| if i < cap { 4 } else { 0 });
        let options = SearchRunOptions::with_level(Level::with_depths(1, depths)).multi_pv(true);
        let result = resources.search(search, &board, &options);
        assert!(result.is_endgame());
        println!("PUBLIC {} {}", result.selectivity() as usize, result.score().unwrap().value());
        for row in result.pv_moves() {
            println!("PV {} {}", row.sq.index(), row.score.value());
        }
    }
}
