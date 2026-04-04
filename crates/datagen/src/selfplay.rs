//! Self-play module for generating training data.
//!
//! This module implements self-play game generation for neural network training.
//! It supports both random game generation and generation from predefined opening
//! sequences, with configurable search parameters.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand::RngExt;
use rand::seq::IteratorRandom;
use regex::Regex;
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::game_state::GameState;
use reversi_core::level::Level;
use reversi_core::move_list::MoveList;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::{self, SearchRunOptions};
use reversi_core::square::Square;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::opening;
use crate::record::{
    GameRecord, read_last_game_id, read_records_from_file, truncate_incomplete_record,
    write_records_to_file,
};

/// Minimum number of random moves at the start of each game
const MIN_RANDOM_MOVES: u8 = 10;

/// Maximum number of random moves at the start of each game
const MAX_RANDOM_MOVES: u8 = 50;

/// Number of digits used in output file naming
const FILE_ID_DIGITS: usize = 5;

/// Maximum size of the record cache
const MAX_CACHE_SIZE: usize = 1_000_000;

/// Sentinel value for `game_score` when the true game outcome is unavailable.
const GAME_SCORE_UNAVAILABLE: i8 = i8::MIN;

/// Tracks file rotation state across games to avoid re-scanning the output directory.
struct FileState {
    prefix: String,
    output_dir: String,
    games_per_file: u32,
    file_id: u32,
    game_id: u16,
}

impl FileState {
    fn new(prefix: &str, output_dir: &str, games_per_file: u32) -> io::Result<Self> {
        let escaped_prefix = regex::escape(prefix);
        let pattern = format!(r"^{escaped_prefix}_\d{{{FILE_ID_DIGITS}}}\.bin$");
        let re = Regex::new(&pattern).unwrap();
        let latest_file_entry = fs::read_dir(output_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                re.is_match(&name)
            })
            .max_by_key(|entry| entry.file_name());

        let mut file_id: u32 = 0;
        let mut game_id: u16 = 0;

        if let Some(entry) = latest_file_entry {
            let path = entry.path();
            file_id = path
                .file_stem()
                .and_then(|stem| {
                    stem.to_string_lossy()
                        .split('_')
                        .next_back()
                        .and_then(|id_str| id_str.parse::<u32>().ok())
                })
                .unwrap_or(0);

            // Remove any trailing incomplete record left by a previous interrupted run
            truncate_incomplete_record(&path)?;

            if let Some(last_id) = read_last_game_id(&path)? {
                let game_count = last_id as u32 + 1;
                if game_count >= games_per_file {
                    file_id += 1;
                } else {
                    game_id = last_id + 1;
                }
            }
        }

        Ok(Self {
            prefix: prefix.to_owned(),
            output_dir: output_dir.to_owned(),
            games_per_file,
            file_id,
            game_id,
        })
    }

    fn file_path(&self, file_id: u32) -> PathBuf {
        Path::new(&self.output_dir)
            .join(format!("{}_{:0FILE_ID_DIGITS$}.bin", self.prefix, file_id))
    }

    fn next_game_id(&mut self) -> u16 {
        let id = self.game_id;
        self.game_id += 1;
        id
    }

    fn rotate(&mut self) {
        self.file_id += 1;
        self.game_id = 0;
    }

    fn is_full(&self) -> bool {
        self.game_id as u32 >= self.games_per_file
    }

    /// Counts the total number of games written across all files.
    fn total_games(&self) -> io::Result<usize> {
        let mut total = 0usize;
        for fid in 0..self.file_id {
            match read_last_game_id(&self.file_path(fid)) {
                Ok(Some(last_id)) => total += last_id as usize + 1,
                Ok(None) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        total += self.game_id as usize;
        Ok(total)
    }

    fn write_records(&mut self, game_records: &[GameRecord]) -> io::Result<()> {
        let file_path = self.file_path(self.file_id);
        write_records_to_file(&file_path, game_records)
    }
}

/// Executes self-play with random openings to generate training data.
///
/// # Arguments
///
/// * `num_games` - Number of games to generate
/// * `games_per_file` - Maximum games per output file
/// * `hash_size` - Transposition table size in MB
/// * `level` - Search level configuration
/// * `selectivity` - Search selectivity parameter
/// * `prefix` - Output file prefix
/// * `output_dir` - Directory for output files
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if file operations fail.
pub fn execute(
    num_games: u32,
    games_per_file: u32,
    hash_size: usize,
    level: Level,
    selectivity: Selectivity,
    prefix: &str,
    output_dir: &str,
) -> io::Result<()> {
    fs::create_dir_all(output_dir)?;

    let options = SearchOptions::new(hash_size);

    let mut search = search::Search::new(&options);
    let mut record_cache: HashMap<Board, GameRecord> = HashMap::new();
    let mut file_state = FileState::new(prefix, output_dir, games_per_file)?;

    for _ in 0..num_games {
        if file_state.is_full() {
            file_state.rotate();
        }
        let game_id = file_state.next_game_id();

        // Generate random opening for this game
        let mut rng = rand::rng();
        let num_random = std::cmp::min(
            rng.random_range(MIN_RANDOM_MOVES..MAX_RANDOM_MOVES),
            rng.random_range(MIN_RANDOM_MOVES..MAX_RANDOM_MOVES),
        );
        let opening_sequence = generate_random_opening(num_random);

        // Play the game using the common function
        let game_records = play_game(
            &opening_sequence,
            &mut search,
            level,
            selectivity,
            game_id,
            &mut record_cache,
        );

        // Save the game records
        file_state.write_records(&game_records)?;
    }
    Ok(())
}

/// Executes self-play using predefined opening sequences.
///
/// # Arguments
///
/// * `openings_path` - Path to file containing opening sequences
/// * `resume` - Whether to resume from last processed opening
/// * `games_per_file` - Maximum games per output file
/// * `hash_size` - Transposition table size in MB
/// * `level` - Search level configuration
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
    games_per_file: u32,
    hash_size: usize,
    level: Level,
    selectivity: Selectivity,
    prefix: &str,
    output_dir: &str,
) -> io::Result<()> {
    fs::create_dir_all(output_dir)?;

    let options = SearchOptions::new(hash_size);

    let mut search = search::Search::new(&options);
    let mut record_cache: HashMap<Board, GameRecord> = HashMap::new();
    let mut file_state = FileState::new(prefix, output_dir, games_per_file)?;

    let opening_sequences = opening::load_openings(openings_path)?;

    // If no openings, exit early
    if opening_sequences.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No valid opening sequences found in the file",
        ));
    }

    // Determine the starting opening index from the last written record
    let start_index = if resume {
        let total = file_state.total_games()?;
        if total > 0 {
            println!("Resuming from opening index: {total}");
        }
        total
    } else {
        0
    };

    for opening_sequence in opening_sequences.iter().skip(start_index) {
        if file_state.is_full() {
            file_state.rotate();
        }
        let game_id = file_state.next_game_id();
        let game_records = play_game(
            opening_sequence,
            &mut search,
            level,
            selectivity,
            game_id,
            &mut record_cache,
        );

        // Save the game records
        file_state.write_records(&game_records)?;
    }
    Ok(())
}

/// Generates a random opening sequence for a game.
///
/// Creates a sequence of random moves for the start of a game.
/// The sequence may end with either player to move.
///
/// # Arguments
///
/// * `num_moves` - The number of random moves to generate. May be fewer
///   if the game ends before this many moves have been played.
///
/// # Returns
///
/// A `Vec<Square>` containing the sequence of moves.
fn generate_random_opening(num_moves: u8) -> Vec<Square> {
    let mut opening = Vec::new();
    let mut game = GameState::new();

    // A helper closure to handle a single move generation.
    // Returns `false` if the game ends.
    let mut play_random_move = |g: &mut GameState| -> bool {
        if g.is_game_over() {
            return false;
        }

        if !g.board().has_legal_moves() {
            if g.make_pass().is_err() {
                return false;
            }
            if g.is_game_over() {
                return false;
            }
        }

        let sq = random_move(g.board());
        opening.push(sq);
        if g.make_move(sq).is_err() {
            return false;
        }
        true
    };

    for _ in 0..num_moves {
        if !play_random_move(&mut game) {
            return opening;
        }
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
/// * `record_cache` - Cache for game records to avoid redundant searches
///
/// # Returns
///
/// Returns a vector of game records for the played game.
fn play_game(
    opening_sequence: &[Square],
    search: &mut search::Search,
    lv: Level,
    selectivity: Selectivity,
    game_id: u16,
    record_cache: &mut HashMap<Board, GameRecord>,
) -> Vec<GameRecord> {
    let game_start = Instant::now();
    search.init();

    let mut game = GameState::new();
    let mut game_records = Vec::new();

    // Play opening moves
    for &sq in opening_sequence {
        if game.is_game_over() {
            break;
        }

        // Handle pass moves
        if !game.board().has_legal_moves() {
            let _ = game.make_pass();
            if game.is_game_over() {
                break;
            }
        }

        let board = *game.board();
        let side_to_move = game.side_to_move();

        let mut record = if let Some(cached_record) = record_cache.get(&board) {
            cached_record.clone()
        } else {
            let options = SearchRunOptions::with_level(lv, selectivity);
            let result = search.run(&board, &options);
            let ply = 60 - board.get_empty_count() as u8;

            let record = GameRecord {
                game_id,
                ply,
                board,
                score: result.score,
                game_score: 0,
                side_to_move,
                is_random: true,
                sq: result.best_move.unwrap_or(sq),
            };
            record_cache.insert(board, record.clone());
            record
        };
        record.game_id = game_id;

        game_records.push(record);

        let _ = game.make_move(sq);
    }

    if record_cache.len() > MAX_CACHE_SIZE {
        record_cache.clear();
    }

    // Continue playing with search
    while !game.is_game_over() {
        if !game.board().has_legal_moves() {
            let _ = game.make_pass();
            continue;
        }

        let board = *game.board();
        let side_to_move = game.side_to_move();
        let options = SearchRunOptions::with_level(lv, selectivity);
        let result = search.run(&board, &options);

        let ply = 60 - board.get_empty_count() as u8;
        let best_move = result.best_move.unwrap();

        let record = GameRecord {
            game_id,
            ply,
            board,
            score: result.score,
            game_score: 0,
            side_to_move,
            is_random: false,
            sq: best_move,
        };
        game_records.push(record);

        let _ = game.make_move(best_move);
    }

    // Calculate final game scores
    if !game_records.is_empty() {
        let final_side_to_move = game.side_to_move();
        let final_board = *game.board();
        let final_score = final_board.solve(final_board.get_empty_count()) as i8;

        for record in game_records.iter_mut() {
            record.game_score = if record.side_to_move == final_side_to_move {
                final_score
            } else {
                -final_score
            };
        }
    }

    let random_moves = game_records.iter().filter(|r| r.is_random).count();
    let total_moves = game_records.len();
    let black_score = game_records.first().map_or(0, |r| r.game_score);
    let duration = game_start.elapsed();
    println!(
        "Game {}: score {:+}, moves {}/{} (random/total), {:.2}s",
        game_id + 1,
        black_score,
        random_moves,
        total_moves,
        duration.as_secs_f64()
    );

    game_records
}

/// Enumerates all unique positions reachable within `depth` plies and scores each one.
///
/// Duplicate positions (reached via different move orders) are eliminated in memory.
/// If the output file already exists, previously scored positions are loaded and skipped.
pub fn execute_score_openings(
    depth: u8,
    hash_size: usize,
    level: Level,
    selectivity: Selectivity,
    output: &str,
) -> io::Result<()> {
    if let Some(parent) = Path::new(output).parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Enumerating positions up to depth {depth}...");
    let mut positions: Vec<(Board, Disc)> = Vec::new();
    let mut visited = HashSet::new();
    enumerate_positions(
        &Board::new(),
        Disc::Black,
        depth,
        &mut visited,
        &mut positions,
    );
    drop(visited);
    println!("Found {} unique positions", positions.len());

    // Load previously scored positions for resume
    let output_path = Path::new(output);
    let mut scored: HashSet<Board> = if output_path.exists() {
        let records = read_records_from_file(output_path)?;
        println!("Loaded {} existing records, resuming...", records.len());
        records.into_iter().map(|r| r.board).collect()
    } else {
        HashSet::new()
    };

    let options = SearchOptions::new(hash_size);
    let mut search = search::Search::new(&options);
    let run_options = SearchRunOptions::with_level(level, selectivity);

    let total = positions.len();
    let already_scored = scored.len();
    let to_score = total.saturating_sub(already_scored);
    let mut new_count = 0usize;
    let mut batch: Vec<GameRecord> = Vec::with_capacity(1000);

    let pb = ProgressBar::with_draw_target(
        Some(to_score as u64),
        ProgressDrawTarget::stderr_with_hz(10),
    );
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({per_sec}) ETA:{eta_precise}",
        )
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
        .progress_chars("#>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    for &(board, side_to_move) in &positions {
        if !scored.insert(board) {
            continue;
        }

        let result = search.run(&board, &run_options);
        let ply = 60 - board.get_empty_count() as u8;

        batch.push(GameRecord {
            game_id: 0,
            ply,
            board,
            score: result.score,
            game_score: GAME_SCORE_UNAVAILABLE,
            side_to_move,
            is_random: true,
            sq: result.best_move.unwrap_or(Square::A1),
        });
        new_count += 1;
        pb.inc(1);

        if batch.len() >= 1000 {
            write_records_to_file(output_path, &batch)?;
            batch.clear();
        }
    }

    if !batch.is_empty() {
        write_records_to_file(output_path, &batch)?;
    }

    pb.finish_and_clear();

    println!(
        "Done. {} positions total, {} newly scored.",
        total, new_count
    );
    Ok(())
}

/// Recursively enumerates all unique board positions reachable within `depth` plies.
///
/// Positions are canonicalized via [`Board::unique`] so that symmetric variants
/// are treated as duplicates, reducing the total count by up to 8x.
fn enumerate_positions(
    board: &Board,
    side_to_move: Disc,
    depth: u8,
    visited: &mut HashSet<Board>,
    positions: &mut Vec<(Board, Disc)>,
) {
    let canonical = board.unique();
    if !visited.insert(canonical) {
        return;
    }
    positions.push((canonical, side_to_move));

    if depth == 0 {
        return;
    }

    let move_list = MoveList::new(board);
    if move_list.count() > 0 {
        for m in move_list.iter() {
            let next = board.make_move_with_flipped(m.flipped, m.sq);
            enumerate_positions(
                &next,
                side_to_move.opposite(),
                depth - 1,
                visited,
                positions,
            );
        }
    } else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            enumerate_positions(&next, side_to_move.opposite(), depth, visited, positions);
        }
    }
}

/// Selects a random legal move from the current board position.
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    board.get_moves().iter().choose(&mut rng).unwrap()
}
