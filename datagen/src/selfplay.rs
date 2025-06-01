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

use crate::opening;

const RECORD_SIZE: u64 = 24;

struct GameRecord {
    ply: u8,
    board: Board,
    score: Scoref,
    game_score: i8,
    side_to_move: Piece,
    is_random: bool,
    sq: Square,
}

pub fn execute(
    num_games: u32,
    records_per_file: u32,
    hash_size: usize,
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
                game_score: 0,
                side_to_move,
                is_random,
                sq: result.best_move.unwrap(),
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
                game_score as i8
            } else {
                -game_score as i8
            };
        }

        let duration = game_start.elapsed();
        println!(
            "Game {} completed in {:.2} seconds",
            game_id + 1,
            duration.as_secs_f64()
        );

        save_game(game_records, prefix, output_dir, records_per_file).expect("Failed to save game");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn execute_with_openings(
    openings_path: &str,
    resume: bool,
    records_per_file: u32,
    hash_size: usize,
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

    let opening_sequences = opening::load_openings(openings_path).unwrap();

    // If no openings, exit early
    if opening_sequences.is_empty() {
        eprintln!("No valid opening sequences found in the file");
        return;
    }

    // Determine the starting game ID
    let mut start_game_id = 0;
    if resume {
        // Read the last game ID from resume.txt file
        let resume_file_path = format!("{}/resume.txt", output_dir);
        if Path::new(&resume_file_path).exists() {
            match fs::read_to_string(&resume_file_path) {
                Ok(content) => {
                    let last_game_id = content.lines().next().and_then(|line| line.parse::<usize>().ok());
                    if let Some(id) = last_game_id {
                        start_game_id = id + 1; // Start from the next game
                        println!("Resuming from game ID: {}", start_game_id);
                    }
                },
                Err(e) => {
                    eprintln!("Failed to read resume.txt: {}", e);
                }
            }
        }
    }

    // Start iteration from the specified game ID
    for (game_id, opening_sequence) in opening_sequences.iter().enumerate().skip(start_game_id) {
        let game_start = Instant::now();
        search.init();

        let mut board = Board::new();
        let mut side_to_move = Piece::Black;
        let mut game_records = Vec::new();

        for &sq in opening_sequence {
            if !board.is_legal_move(sq) {
                // If the move isn't legal, switch players and try again
                if !board.has_legal_moves() {
                    board = board.switch_players();
                    side_to_move = side_to_move.opposite();

                    // If it's still not legal after switching, skip this opening
                    if !board.is_legal_move(sq) {
                        eprintln!("Warning: Invalid move in opening sequence: {}", sq);
                        continue;
                    }
                } else {
                    eprintln!("Warning: Invalid move in opening sequence: {}", sq);
                    continue;
                }
            }

            // Record the move
            let result = search.run(
                &board,
                lv,
                selectivity,
                false,
                None::<fn(reversi_core::search::SearchProgress) -> ()>,
            );
            let ply = 60 - board.get_empty_count() as u8;
            let best_move = result.best_move.unwrap();

            let record = GameRecord {
                ply,
                board,
                score: result.score,
                game_score: 0,
                side_to_move,
                is_random: true, // Opening moves are considered "random"
                sq: best_move,
            };

            game_records.push(record);

            // Make the move
            board = board.make_move(sq);
            side_to_move = side_to_move.opposite();
        }

        while !board.is_game_over() {
            if !board.has_legal_moves() {
                board = board.switch_players();
                side_to_move = side_to_move.opposite();
                continue; // Skip evaluation for pass moves
            }

            let result = search.run(
                &board,
                lv,
                selectivity,
                false,
                None::<fn(reversi_core::search::SearchProgress) -> ()>,
            );

            let ply = 60 - board.get_empty_count() as u8;
            let best_move = result.best_move.unwrap();

            // Record the position before the move
            let record = GameRecord {
                ply,
                board,
                score: result.score,
                game_score: 0,
                side_to_move,
                is_random: false,
                sq: best_move,
            };
            game_records.push(record);

            board = board.make_move(best_move);
            side_to_move = side_to_move.opposite();
        }

        // Calculate final game scores
        if !game_records.is_empty() {
            let last_record = game_records.last().unwrap();
            let last_side = last_record.side_to_move;
            let game_score = last_record.score;

            for record in game_records.iter_mut() {
                record.game_score = if record.side_to_move == last_side {
                    game_score as i8
                } else {
                    -game_score as i8
                };
            }
        }

        let duration = game_start.elapsed();
        println!(
            "Game {} completed in {:.2} seconds",
            game_id + 1,
            duration.as_secs_f64()
        );

        save_game(game_records, prefix, output_dir, records_per_file).expect("Failed to save game");

        // Save the current game ID (progress) to resume.txt
        if resume {
            let resume_file_path = format!("{}/resume.txt", output_dir);
            fs::write(&resume_file_path, format!("{}", game_id)).expect("Failed to write resume.txt");
        }
    }
}

fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    BitboardIterator::new(board.get_moves())
        .choose(&mut rng)
        .unwrap()
}

fn write_records_to_file(path_str: &str, records: &[GameRecord]) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(path_str))?;
    let mut writer = BufWriter::new(file);

    for record in records {
        writer.write_u64::<LittleEndian>(record.board.player)?;
        writer.write_u64::<LittleEndian>(record.board.opponent)?;
        writer.write_f32::<LittleEndian>(record.score)?;
        writer.write_i8(record.game_score)?;
        writer.write_u8(record.ply)?;
        writer.write_u8(if record.is_random { 1 } else { 0 })?;
        writer.write_u8(record.sq as u8)?;
    }
    writer.flush()?;
    Ok(())
}

fn save_game(
    game_records: Vec<GameRecord>,
    prefix: &str,
    output_dir: &str,
    records_per_file: u32,
) -> std::io::Result<()> {
    let pattern = format!(r"^{}_\d{{5}}\.bin$", prefix); // Use 5 digits for file ID
    let re = Regex::new(&pattern).unwrap();
    let latest_file_entry = fs::read_dir(output_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            re.is_match(&name)
        })
        .max_by_key(|entry| entry.file_name());

    let mut current_file_id: u32 = 0;
    let mut current_record_count: u32 = 0;
    let mut current_file_path_str: String;

    if let Some(entry) = latest_file_entry {
        let path = entry.path();
        current_file_path_str = path.to_string_lossy().to_string();
        current_file_id = Path::new(&current_file_path_str)
            .file_stem()
            .and_then(|stem| {
                stem.to_string_lossy()
                    .split('_')
                    .last()
                    .and_then(|id_str| id_str.parse::<u32>().ok())
            })
            .unwrap_or(0);

        if path.exists() {
            current_record_count = count_records_in_file(&path)?;
            if current_record_count >= records_per_file {
                current_file_id += 1;
                current_record_count = 0;
            }
        } else {
            current_record_count = 0;
        }
    }

    current_file_path_str = format!("{}/{}_{:05}.bin", output_dir, prefix, current_file_id);

    let mut records_processed = 0;
    while records_processed < game_records.len() {
        let remaining_capacity = records_per_file.saturating_sub(current_record_count);

        if remaining_capacity == 0 {
            current_file_id += 1;
            current_file_path_str = format!("{}/{}_{:05}.bin", output_dir, prefix, current_file_id);
            current_record_count = 0;
            continue;
        }

        let records_to_write_now = std::cmp::min(
            remaining_capacity as usize,
            game_records.len() - records_processed,
        );

        if records_to_write_now > 0 {
            let start_index = records_processed;
            let end_index = records_processed + records_to_write_now;
            write_records_to_file(
                &current_file_path_str,
                &game_records[start_index..end_index],
            )?;

            current_record_count += records_to_write_now as u32;
            records_processed += records_to_write_now;
        } else {
            eprintln!("Warning: No records processed in loop iteration. Breaking.");
            break;
        }
    }

    Ok(())
}

fn count_records_in_file(path: &Path) -> std::io::Result<u32> {
    if !path.exists() {
        return Ok(0);
    }
    let metadata = fs::metadata(path)?;
    let file_size = metadata.len();

    if RECORD_SIZE == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "RECORD_SIZE is zero",
        ));
    }
    // Check for potential file corruption or incorrect RECORD_SIZE
    if file_size % RECORD_SIZE != 0 {
        eprintln!("Warning: File size {} is not a multiple of RECORD_SIZE {} for file {}. File might be corrupted or RECORD_SIZE is incorrect.", file_size, RECORD_SIZE, path.display());
        // Decide handling: error out or proceed with floor division?
        // Proceeding with floor division, assuming partial records at the end are invalid/ignored.
    }

    Ok((file_size / RECORD_SIZE) as u32)
}
