//! Self-play module for generating training data.
//!
//! This module implements self-play game generation for neural network training.
//! It supports both random game generation and generation from predefined opening
//! sequences, with configurable search parameters.

use byteorder::{LittleEndian, WriteBytesExt};
use rand::Rng;
use rand::seq::IteratorRandom;
use regex::Regex;
use reversi_core::bitboard::BitboardIterator;
use reversi_core::board::Board;
use reversi_core::level::{Level, get_level};
use reversi_core::piece::Piece;
use reversi_core::search::{self, SearchOptions};
use reversi_core::square::Square;
use reversi_core::types::{Scoref, Selectivity};
use std::cmp::Ordering;
use std::fs;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::opening;

/// Size of each record in bytes (24 bytes per record)
const RECORD_SIZE: u64 = 24;

/// Minimum number of random moves at the start of each game
const MIN_RANDOM_MOVES: u8 = 10;

/// Maximum number of random moves at the start of each game
const MAX_RANDOM_MOVES: u8 = 30;

/// Number of digits used in output file naming
const FILE_ID_DIGITS: usize = 5;

/// Represents a single position record from a self-play game.
///
/// Each record captures the board state, evaluation, and move information
/// at a specific point in the game.
struct GameRecord {
    /// Move number in the game (0-59)
    ply: u8,
    /// Current board state
    board: Board,
    /// Evaluation score from the search
    score: Scoref,
    /// Final game score from this player's perspective
    game_score: i8,
    /// Current player to move
    side_to_move: Piece,
    /// Whether this position resulted from a random move
    is_random: bool,
    /// The move played from this position
    sq: Square,
}

/// Executes self-play with random openings to generate training data.
///
/// # Arguments
///
/// * `num_games` - Number of games to generate
/// * `records_per_file` - Maximum records per output file
/// * `hash_size` - Transposition table size in MB
/// * `level` - Search depth level
/// * `selectivity` - Search selectivity parameter
/// * `prefix` - Output file prefix
/// * `output_dir` - Directory for output files
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if file operations fail.
pub fn execute(
    num_games: u32,
    records_per_file: u32,
    hash_size: usize,
    level: usize,
    selectivity: Selectivity,
    prefix: &str,
    output_dir: &str,
) -> io::Result<()> {
    fs::create_dir_all(output_dir)?;

    let options = SearchOptions::new(hash_size);

    let lv = get_level(level);
    let mut search = search::Search::new(&options);

    for game_id in 0..num_games {
        // Generate random opening for this game
        let num_random = rand::rng().random_range(MIN_RANDOM_MOVES..MAX_RANDOM_MOVES);
        let opening_sequence = generate_random_opening(num_random);

        // Play the game using the common function
        let game_records = play_game(
            &opening_sequence,
            &mut search,
            lv,
            selectivity,
            game_id as usize,
        );

        // Save the game records
        save_game(game_records, prefix, output_dir, records_per_file)?;
    }
    Ok(())
}

/// Executes self-play using predefined opening sequences.
///
/// # Arguments
///
/// * `openings_path` - Path to file containing opening sequences
/// * `resume` - Whether to resume from last processed opening
/// * `records_per_file` - Maximum records per output file
/// * `hash_size` - Transposition table size in MB
/// * `level` - Search depth level
/// * `selectivity` - Search selectivity parameter
/// * `prefix` - Output file prefix
/// * `output_dir` - Directory for output files
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if operations fail.
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
) -> io::Result<()> {
    fs::create_dir_all(output_dir)?;

    let options = SearchOptions::new(hash_size);

    let lv = get_level(level);
    let mut search = search::Search::new(&options);

    let opening_sequences = opening::load_openings(openings_path)?;

    // If no openings, exit early
    if opening_sequences.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No valid opening sequences found in the file",
        ));
    }

    // Determine the starting game ID
    let mut start_game_id = 0;
    if resume {
        // Read the last game ID from resume.txt file
        let resume_file_path = format!("{output_dir}/resume.txt");
        if Path::new(&resume_file_path).exists() {
            match fs::read_to_string(&resume_file_path) {
                Ok(content) => {
                    let last_game_id = content
                        .lines()
                        .next()
                        .and_then(|line| line.parse::<usize>().ok());
                    if let Some(id) = last_game_id {
                        start_game_id = id + 1; // Start from the next game
                        println!("Resuming from game ID: {start_game_id}");
                    }
                }
                Err(e) => {
                    eprintln!("Failed to read resume.txt: {e}");
                }
            }
        }
    }

    // Start iteration from the specified game ID
    for (game_id, opening_sequence) in opening_sequences.iter().enumerate().skip(start_game_id) {
        // Play the game using the common function
        let game_records = play_game(opening_sequence, &mut search, lv, selectivity, game_id);

        // Save the game records
        save_game(game_records, prefix, output_dir, records_per_file)?;

        // Save the current game ID (progress) to resume.txt
        if resume {
            let resume_file_path = format!("{output_dir}/resume.txt");
            fs::write(&resume_file_path, format!("{game_id}"))?;
        }
    }
    Ok(())
}

/// Generates a random opening sequence for a game.
///
/// Returns a vector of moves to play at the beginning of the game.
fn generate_random_opening(num_moves: u8) -> Vec<Square> {
    let mut opening = Vec::new();
    let mut board = Board::new();
    let mut side_to_move = Piece::Black;

    for _ in 0..num_moves {
        if board.is_game_over() {
            break;
        }

        if !board.has_legal_moves() {
            board = board.switch_players();
            side_to_move = side_to_move.opposite();
            if !board.has_legal_moves() {
                break;
            }
        }

        let sq = random_move(&board);
        opening.push(sq);
        board = board.make_move(sq);
        side_to_move = side_to_move.opposite();
    }

    opening
}

/// Plays a single game with the given opening sequence.
///
/// # Arguments
///
/// * `opening_sequence` - Sequence of moves to play at the start
/// * `search` - Search engine instance
/// * `lv` - Search level
/// * `selectivity` - Search selectivity
/// * `game_id` - Game identifier for logging
///
/// # Returns
///
/// Returns a vector of game records for the played game.
fn play_game(
    opening_sequence: &[Square],
    search: &mut search::Search,
    lv: Level,
    selectivity: Selectivity,
    game_id: usize,
) -> Vec<GameRecord> {
    let game_start = Instant::now();
    search.init();

    let mut board = Board::new();
    let mut side_to_move = Piece::Black;
    let mut game_records = Vec::new();

    // Play opening moves
    for &sq in opening_sequence {
        if board.is_game_over() {
            break;
        }

        // Handle pass moves
        if !board.has_legal_moves() {
            board = board.switch_players();
            side_to_move = side_to_move.opposite();
            if !board.has_legal_moves() {
                break;
            }
        }

        // Skip invalid moves
        if !board.is_legal_move(sq) {
            eprintln!("Warning: Invalid move in opening sequence: {sq}");
            continue;
        }

        // Evaluate position before making the move
        let result = search.run(&board, lv, selectivity, false);
        let ply = 60 - board.get_empty_count() as u8;

        let record = GameRecord {
            ply,
            board,
            score: result.score,
            game_score: 0,
            side_to_move,
            is_random: true, // Opening moves are considered "random"
            sq,
        };

        game_records.push(record);

        // Make the move
        board = board.make_move(sq);
        side_to_move = side_to_move.opposite();
    }

    // Continue playing with search
    while !board.is_game_over() {
        if !board.has_legal_moves() {
            board = board.switch_players();
            side_to_move = side_to_move.opposite();
            continue; // Skip evaluation for pass moves
        }

        let result = search.run(&board, lv, selectivity, false);

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
        let final_score = calculate_final_score(&board);

        for record in game_records.iter_mut() {
            let score = match record.side_to_move {
                Piece::Black | Piece::White => {
                    if record.side_to_move == side_to_move {
                        final_score
                    } else {
                        -final_score
                    }
                }
                Piece::Empty => unreachable!("record side_to_move should never be empty"),
            };
            record.game_score = score;
        }
    }

    let duration = game_start.elapsed();
    println!(
        "Game {} completed in {:.2} seconds",
        game_id + 1,
        duration.as_secs_f64()
    );

    game_records
}

/// Calculates the final score of the game from the board state.
///
/// Arguments
/// * `board` - The final board state of the game
///
/// Returns
/// The final score from the perspective of the player to move.
fn calculate_final_score(board: &Board) -> i8 {
    let n_empties = board.get_empty_count();
    let score = board.get_player_count() as i8 * 2 - 64;
    let diff = score + n_empties as i8;

    match diff.cmp(&0) {
        Ordering::Equal => diff,
        Ordering::Greater => diff + n_empties as i8,
        Ordering::Less => score,
    }
}

/// Selects a random legal move from the current board position.
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    BitboardIterator::new(board.get_moves())
        .choose(&mut rng)
        .unwrap()
}

/// Writes game records to a binary file.
///
/// Records are appended to the file if it already exists.
fn write_records_to_file(path_str: &str, records: &[GameRecord]) -> io::Result<()> {
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

/// Saves game records to output files with automatic file rotation.
///
/// Records are distributed across multiple files, creating new files
/// when the record limit is reached.
fn save_game(
    game_records: Vec<GameRecord>,
    prefix: &str,
    output_dir: &str,
    records_per_file: u32,
) -> io::Result<()> {
    let pattern = format!(r"^{prefix}_\d{{{FILE_ID_DIGITS}}}\.bin$");
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
                    .next_back()
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

    current_file_path_str = format!("{output_dir}/{prefix}_{current_file_id:0FILE_ID_DIGITS$}.bin");

    let mut records_processed = 0;
    while records_processed < game_records.len() {
        let remaining_capacity = records_per_file.saturating_sub(current_record_count);

        if remaining_capacity == 0 {
            current_file_id += 1;
            current_file_path_str =
                format!("{output_dir}/{prefix}_{current_file_id:0FILE_ID_DIGITS$}.bin");
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

/// Counts the number of records in a binary file.
///
/// Returns 0 if the file doesn't exist.
fn count_records_in_file(path: &Path) -> io::Result<u32> {
    if !path.exists() {
        return Ok(0);
    }
    let metadata = fs::metadata(path)?;
    let file_size = metadata.len();

    if RECORD_SIZE == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "RECORD_SIZE is zero",
        ));
    }
    // Check for potential file corruption or incorrect RECORD_SIZE
    if file_size % RECORD_SIZE != 0 {
        eprintln!(
            "Warning: File size {} is not a multiple of RECORD_SIZE {} for file {}. File might be corrupted or RECORD_SIZE is incorrect.",
            file_size,
            RECORD_SIZE,
            path.display()
        );
        // Decide handling: error out or proceed with floor division?
        // Proceeding with floor division, assuming partial records at the end are invalid/ignored.
    }

    Ok((file_size / RECORD_SIZE) as u32)
}
