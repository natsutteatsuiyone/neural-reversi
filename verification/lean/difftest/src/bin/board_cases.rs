#[allow(dead_code, clippy::enum_variant_names, clippy::wrong_self_convention)]
#[path = "../../../../../crates/reversi-core/src"]
mod engine {
    #[path = "util/align.rs"]
    pub mod align;
    pub mod analysis;
    pub mod bitboard;
    pub mod board;
    pub mod constants;
    pub mod disc;
    #[path = "search/endgame/empties.rs"]
    pub mod empties;
    #[allow(unused_imports)]
    pub mod flip;
    pub mod legal_moves;
    #[path = "search/node_type.rs"]
    pub mod node_type;
    pub mod obf;
    pub mod player_color;
    #[allow(unused_imports)]
    pub mod probcut;
    pub mod score;
    pub mod square;
    pub mod types;
}

use std::cell::Cell;
use std::io::{self, Write};

use flip_scalar_difftest::{csv, run_buffered};

use engine::analysis::{Analysis, AnalysisError, GameAnalysisMove, analyze_game};
use engine::board::{self, Board};
use engine::score::Score;
use engine::square::Square;
use engine::{
    bitboard, constants, disc, flip, legal_moves, player_color, probcut, score, square, types,
};

mod util {
    pub use crate::engine::align;
}

#[allow(dead_code, unused_imports)]
mod move_list {
    include!(concat!(env!("OUT_DIR"), "/move_list_source.rs"));
}

#[path = "../empties_cases.rs"]
mod empties_cases;
#[path = "../move_generation_cases.rs"]
mod move_generation_cases;
#[path = "../move_storage_cases.rs"]
mod move_storage_cases;
#[path = "../obf_cases.rs"]
mod obf_cases;
#[path = "../stability_cases.rs"]
mod stability_cases;
#[allow(dead_code)]
#[path = "../tt_save_cases.rs"]
mod tt_save_cases;

fn write_cases(out: &mut impl Write, board: Board, moves: &[GameAnalysisMove]) -> io::Result<()> {
    let encoded_moves = csv(moves.iter().map(|action| match action {
        GameAnalysisMove::Pass => "p".to_owned(),
        GameAnalysisMove::Play(square) => square.index().to_string(),
    }));

    for cancel_at in [0, 1, 2, 3, 4, 8, 256] {
        for mode in ["score", "invalid", "error"] {
            let polls = Cell::new(0);
            let mut reports = Vec::new();
            let result = analyze_game(
                board,
                moves,
                |searched| {
                    let square = 3 * searched.player().count() + searched.opponent().count();
                    let best_move = Square::from_u8((square % 64) as u8).unwrap();
                    let next_move = Square::from_u8(((square + 17) % 64) as u8).unwrap();
                    let score = match mode {
                        "invalid" => Score::ZERO + i32::MIN,
                        "error" => return Err("fixture".into()),
                        _ => Score::from_raw(
                            64 * (searched.player().count() as i32
                                - searched.opponent().count() as i32)
                                + 17,
                        ),
                    };
                    Ok(Analysis {
                        pv: vec![best_move, next_move],
                        best_move,
                        score,
                    })
                },
                || {
                    let poll = polls.get();
                    polls.set(poll + 1);
                    poll >= cancel_at
                },
                |progress| {
                    let pv = csv(progress.pv.iter().map(|square| square.index()));
                    reports.push(format!(
                        "{},{},{},{},{}:{pv}",
                        progress.move_index,
                        progress.best_score.value(),
                        progress.played_score.value(),
                        progress.score_loss.value(),
                        progress.best_move.index()
                    ));
                },
            );
            let outcome = match result {
                Ok(()) => "ok".to_owned(),
                Err(AnalysisError::IllegalMove(square)) => format!("illegal:{}", square.index()),
                Err(AnalysisError::PassWithLegalMoves) => "pass-with-moves".to_owned(),
                Err(AnalysisError::PassAfterGameOver) => "pass-after-game".to_owned(),
                Err(AnalysisError::Search(message)) => format!("search:{message}"),
                Err(AnalysisError::InvalidSearchScore(score)) => {
                    format!("invalid:{}", score.value())
                }
            };
            writeln!(
                out,
                "{} {} {encoded_moves} {cancel_at} {mode}|{outcome}|{}",
                board.player().bits(),
                board.opponent().bits(),
                reports.join(";")
            )?;
        }
    }
    Ok(())
}

fn write_game_analysis_cases(out: &mut impl Write) -> io::Result<()> {
    use GameAnalysisMove::{Pass, Play};

    for (board, moves) in [
        (Board::new(), vec![]),
        (Board::new(), vec![Play(Square::A1)]),
        (Board::new(), vec![Play(Square::D3), Pass]),
        (Board::from_bitboards(!0, 0), vec![Pass]),
        (Board::from_bitboards(2_u64, 1_u64), vec![]),
        (
            Board::from_bitboards(2_u64, 1_u64),
            vec![Pass, Play(Square::C1)],
        ),
        (Board::from_bitboards(1_u64, 2_u64), vec![Play(Square::C1)]),
        (
            Board::from_bitboards(1_u64 << 47, (1_u64 << 55) | (1_u64 << 62)),
            vec![Play(Square::H8)],
        ),
    ] {
        write_cases(out, board, &moves)?;
    }

    for game in 0..4 {
        let initial = Board::new();
        let mut board = initial;
        let mut moves = Vec::new();
        loop {
            let legal = board.legal_moves();
            if legal.is_empty() {
                if !board.switch_players().has_legal_moves() {
                    break;
                }
                moves.push(Pass);
                board = board.switch_players();
            } else {
                let index = (moves.len() * (game + 3) + game) % legal.count() as usize;
                let square = legal.iter().nth(index).unwrap();
                moves.push(Play(square));
                board = board.make_move(square);
            }
            if moves.len().is_multiple_of(8) {
                write_cases(out, initial, &moves)?;
            }
        }
        write_cases(out, initial, &moves)?;
    }
    Ok(())
}

fn write_board_hash_cases(out: &mut impl Write) -> io::Result<()> {
    use rapidhash::v3::{DEFAULT_RAPID_SECRETS, rapidhash_v3_nano_inline};

    let samples = [0, 1, u64::MAX, 0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210];
    for raw in samples
        .into_iter()
        .chain((0..64).map(|bit| 1u64 << bit))
        .chain((0u64..256).map(|seed| seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
    {
        for opponent in [0, !raw, raw.rotate_left(17) & !raw] {
            let board = Board::from_bitboards(raw, opponent);
            for big_endian in [false, true] {
                let bytes = if big_endian {
                    [raw.to_be_bytes(), opponent.to_be_bytes()].concat()
                } else {
                    [raw.to_le_bytes(), opponent.to_le_bytes()].concat()
                };
                let hash = if big_endian == cfg!(target_endian = "big") {
                    board.hash()
                } else {
                    rapidhash_v3_nano_inline::<false, false>(&bytes, &DEFAULT_RAPID_SECRETS)
                };
                writeln!(out, "{} {raw} {opponent} {hash}", u8::from(big_endian))?;
            }
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    run_buffered(|out| match std::env::args().nth(1).as_deref() {
        Some("game-analysis-cases") => write_game_analysis_cases(out),
        Some("obf-cases") => obf_cases::write_cases(out),
        Some("empties-cases") => empties_cases::write_cases(out),
        Some("board-hash-cases") => write_board_hash_cases(out),
        Some("move-storage-cases") => move_storage_cases::write_cases(out),
        Some("move-generation-cases") => move_generation_cases::write_cases(out),
        Some("tt-save-cases") => tt_save_cases::write_cases(out),
        Some("stability-cases") => stability_cases::write_cases(out),
        name => panic!("unknown board case generator: {name:?}"),
    })
}
