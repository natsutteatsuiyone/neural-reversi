use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use std::path::Path;

pub fn run(weights: &Path) {
    let eval = Arc::new(
        Eval::with_weight_files(
            Some(&weights.join(EVAL_FILE_NAME)),
            Some(&weights.join(EVAL_SM_FILE_NAME)),
        )
        .unwrap(),
    );
    let shared = SearchSharedResources::from_eval(eval, 0, Some(1));
    let mut search = Search::from_shared_resources(&shared);
    let board = Board::from_bitboards(0x3778d008b0488010, 0x00062ef64f367d4c);
    let SearchResult::BestMove {
        score,
        is_endgame: true,
        selectivity: Selectivity::None,
        ..
    } = search.run(&board, &SearchRunOptions::with_level(Level::perfect()))
    else {
        panic!("expected a completed public exact solve")
    };
    eprintln!("PUBLIC {}", score.value());
}
