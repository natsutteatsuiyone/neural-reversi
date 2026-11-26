// Exact searches on one worker. Start once from the board and once from its child.
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
    let board = Board::from_bitboards(0xf3d7ffffffff8d00, 0x0028000000000000);
    let child = board.make_move(crate::square::Square::C8);
    let mut tt = TranspositionTable::new(0);
    let pool = ThreadPool::new(1);
    for (name, pos, a, z, fresh) in [
        ("root", board, -Score::INF.value(), Score::INF.value(), true),
        ("scout", board, 0, 1, true),
        ("hit", board, 0, 1, false),
        ("seed", child, -1, 0, true),
        ("etc", board, 0, 1, false),
    ] {
        if fresh {
            tt.clear();
        }
        let moves = root_move::RootMoves::new(&pos);
        let mut ctx = context::SearchContext::new(&pos, Selectivity::None, &moves, &tt, &eval);
        eprintln!("BEGIN {name} {}", fresh as u8);
        let a = Score::from_raw(a);
        let z = Score::from_raw(z);
        let v = if name == "root" {
            pvs::search::<node_type::Root, strategy::EndGameStrategy>(
                &mut ctx,
                &pos,
                pos.empty_count(),
                a,
                z,
                pool.main(),
                false,
            )
        } else {
            pvs::search::<node_type::NonPV, strategy::EndGameStrategy>(
                &mut ctx,
                &pos,
                pos.empty_count(),
                a,
                z,
                pool.main(),
                false,
            )
        };
        tt.threshold_snapshot();
        eprintln!("SCORE {}", v.value());
    }
}
