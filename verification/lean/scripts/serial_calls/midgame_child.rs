use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::score::Score;
use std::path::Path;

pub fn run(weights: &Path) {
    let eval = Eval::with_weight_files(
        Some(&weights.join(EVAL_FILE_NAME)),
        Some(&weights.join(EVAL_SM_FILE_NAME)),
    )
    .unwrap();
    let pos = Board::from_bitboards(0x0020000000000100, 0xf7dfffffffff8c00);
    assert_eq!(pos.empty_count(), 13);
    assert_eq!(pos.legal_moves().count(), 1);
    let roots = root_move::RootMoves::new(&pos);
    let tt = TranspositionTable::new(0);
    let pool = ThreadPool::new(1);
    let abort = pool.abort_state();
    for (stop, alpha) in [(true, 0), (false, 0), (false, 0), (false, -32767)] {
        abort.begin_search();
        if stop {
            abort.request_abort();
        }
        let mut ctx = context::SearchContext::new(&pos, Selectivity::None, &roots, &tt, &eval);
        let score = pvs::search::<node_type::NonPV, strategy::MidGameStrategy>(
            &mut ctx, &pos, 13, Score::from_raw(alpha), Score::from_raw(alpha + 1), pool.main(), false,
        );
        println!("SCORE {stop} {alpha} {}", score.value());
        tt.midgame_image();
        pool.main().endgame_caches().midgame_image();
    }
}
