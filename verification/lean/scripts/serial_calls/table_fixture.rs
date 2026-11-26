use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use std::path::Path;

fn sample(seed: u32, empties: u32) -> Board {
    let mut board = Board::new();
    while board.empty_count() > empties {
        let moves = board.legal_moves();
        if moves.is_empty() {
            board = board.switch_players();
            assert!(board.has_legal_moves());
        } else {
            let i = ((seed * 17 + board.empty_count() * 13) % moves.count()) as usize;
            board = board.make_move(moves.iter().nth(i).unwrap());
        }
    }
    board
}

pub fn run(weights: &Path) {
    let eval = Arc::new(
        Eval::with_weight_files(
            Some(&weights.join(EVAL_FILE_NAME)),
            Some(&weights.join(EVAL_SM_FILE_NAME)),
        )
        .unwrap(),
    );
    let shared = SearchSharedResources::from_eval(eval, 2, Some(1));
    let mut search = Search::from_shared_resources(&shared);
    let options = SearchRunOptions::with_level(Level::perfect());
    let fourteen = Board::from_bitboards(0x3778d008b0488010, 0x00062ef64f367d4c);
    // Solve related boards through the same public Search to retain genuine warm entries.
    let fifteen = sample(1, 15);
    let child = fifteen.make_move(fifteen.legal_moves().iter().next().unwrap());
    let reply = fifteen
        .make_move(crate::square::Square::H4)
        .make_move(crate::square::Square::H1);
    let sixteen = sample(0, 16);
    let normal = [fourteen, child, reply, fifteen, fifteen, sixteen];
    let mut small =
        Search::from_shared_resources_with_tt(&shared, Arc::new(TranspositionTable::new(0)));
    for (engine, boards) in [
        (&mut search, normal.as_slice()),
        (&mut small, std::slice::from_ref(&sixteen)),
    ] {
        for board in boards {
            eprintln!("RUN {} {}", board.player().bits(), board.opponent().bits());
            let SearchResult::BestMove {
                score,
                selectivity: Selectivity::None,
                is_endgame: true,
                ..
            } = engine.run(board, &options)
            else {
                panic!("expected an exact result")
            };
            eprintln!("PUBLIC {}", score.value());
        }
    }
}
