use byteorder::{LittleEndian, WriteBytesExt};
use rand::seq::IteratorRandom;
use rand::Rng;
use regex::Regex;
use reversi_core::bitboard::BitboardIterator;
use reversi_core::board::Board;
use reversi_core::level::get_level;
use reversi_core::piece::Piece;
use reversi_core::search::{self, SearchOptions};
use reversi_core::square::Square;
use reversi_core::types::{Scoref, Selectivity};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

const RECORD_SIZE: u64 = 8 * 2 + 4 * 2 + 1 + 1;

struct GameRecord {
    ply: u8,
    board: Board,
    score: Scoref,
    game_score: Scoref,
    side_to_move: Piece,
    is_random: bool,
}

pub fn execute(
    num_games: u32,
    games_per_file: u32,
    hash_size: i32,
    level: usize,
    selectivity: Selectivity,
    prefix: &str,
    output_dir: &str,
) {
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let options = SearchOptions {
        tt_mb_size: hash_size,
        ..Default::default()
    };

    let lv = get_level(level);
    let mut search = search::Search::new(&options);

    for game_id in 0..num_games {
        let game_start = Instant::now();
        search.init();

        let mut board = Board::new();
        let mut side_to_move = Piece::Black;

        let mut game_records = Vec::new();
        let num_random = rand::rng().random_range(10..30);
        let mut is_random = true;
        while !board.is_game_over() {
            if !board.has_legal_moves() {
                board = board.switch_players();
                side_to_move = side_to_move.opposite();
            }
            let result = search.run(
                &board,
                lv,
                selectivity,
                false,
                None::<fn(reversi_core::search::SearchProgress) -> ()>,
            );
            let ply = 60 - board.get_empty_count() as u8;

            let record = GameRecord {
                ply,
                board,
                score: result.score,
                game_score: 0.0,
                side_to_move,
                is_random,
            };

            game_records.push(record);

            if ply < num_random {
                let sq = random_move(&board);
                board = board.make_move(sq);
                side_to_move = side_to_move.opposite();
                is_random = true;
            } else {
                let best_move = result.best_move.unwrap();
                board = board.make_move(best_move);
                side_to_move = side_to_move.opposite();
                is_random = false;
            }
        }

        let last_record = game_records.last().unwrap();
        let last_side = last_record.side_to_move;
        let game_score = last_record.score;
        for record in game_records.iter_mut() {
            record.game_score = if record.side_to_move == last_side {
                game_score
            } else {
                -game_score
            };
        }

        let duration = game_start.elapsed();
        println!(
            "Game {} completed in {:.2} seconds",
            game_id + 1,
            duration.as_secs_f64()
        );

        save_game(game_records, prefix, output_dir, games_per_file).expect("Failed to save game");
    }
}

fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    BitboardIterator::new(board.get_moves())
        .choose(&mut rng)
        .unwrap()
}

fn save_game(game_records: Vec<GameRecord>, prefix: &str, output_dir: &str, games_per_file: u32) -> std::io::Result<()> {
    let pattern = format!(r"^{}_\d{{4}}\.bin$", prefix);
    let re = Regex::new(&pattern).unwrap();
    let latest_file = fs::read_dir(output_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            re.is_match(&name)
        })
        .max_by_key(|entry| entry.file_name());

    let filename = match latest_file {
        Some(file) => {
            let path = file.path();
            let game_count = count_games_in_file(&path)?;
            if game_count >= games_per_file * 60 {
                let new_id = path
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .replace(&format!("{}_", prefix), "")
                    .parse::<u32>()
                    .unwrap()
                    + 1;
                format!("{}/{}_{:05}.bin", output_dir, prefix, new_id)
            } else {
                path.to_string_lossy().to_string()
            }
        }
        None => format!("{}/{}_0000.bin", output_dir, prefix),
    };

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(&filename))?;
    let mut writer = BufWriter::new(file);

    for record in game_records.iter() {
        writer.write_u64::<LittleEndian>(record.board.player)?;
        writer.write_u64::<LittleEndian>(record.board.opponent)?;
        writer.write_f32::<LittleEndian>(record.score)?;
        writer.write_f32::<LittleEndian>(record.game_score)?;
        writer.write_u8(record.ply)?;
        writer.write_u8(if record.is_random { 1 } else { 0 })?;
    }

    writer.flush()?;
    Ok(())
}

fn count_games_in_file(path: &Path) -> std::io::Result<u32> {
    let metadata = fs::metadata(path)?;
    Ok((metadata.len() / RECORD_SIZE) as u32)
}
