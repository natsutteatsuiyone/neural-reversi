// Compiled as a child of search in the temporary copy.
use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
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
    for (p, o, expected, cold) in [
        (0x0100000000000001, 0x0200000000000002, 16384, false), // forced pass
        (0xfe9eb8a0c888007e, 0x0161475f37747f00, 8192, true),
        (0x81afcddf13010000, 0x7e503220ecfe7c24, 7168, true),
        (0x3778d008b0488010, 0x00062ef64f367d4c, 4096, true), // 14 empties, ordinary NonPV
    ] {
        let board = Board::from_bitboards(p, o);
        if cold {
            let tt = TranspositionTable::new(0);
            let pool = ThreadPool::new(1);
            let moves = root_move::RootMoves::new(&board);
            let mut ctx =
                context::SearchContext::new(&board, Selectivity::None, &moves, &tt, &eval);
            let value = pvs::search::<node_type::PV, strategy::EndGameStrategy>(
                &mut ctx,
                &board,
                board.empty_count(),
                -Score::INF,
                Score::INF,
                pool.main(),
                false,
            );
            assert_eq!(value.value(), expected);
        }
        let shared = SearchSharedResources::from_eval(eval.clone(), 2, Some(1));
        let mut search = Search::from_shared_resources(&shared);
        let options = SearchRunOptions::with_level(Level::perfect());
        let SearchResult::BestMove {
            score,
            is_endgame: true,
            selectivity: Selectivity::None,
            ..
        } = search.run(&board, &options)
        else {
            panic!("expected completed exact result")
        };
        assert_eq!(score.value(), expected);
    }
}
