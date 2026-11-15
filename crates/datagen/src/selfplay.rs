//! Self-play module for generating training data.
//!
//! Generates neural-network training games from either random openings or
//! predefined opening sequences with configurable search depths.

use std::cmp;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use rand::RngExt;
use rand::seq::IteratorRandom;
use reversi_core::board::Board;
use reversi_core::game_state::GameState;
use reversi_core::level::Level;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::{self, SearchRunOptions};
use reversi_core::square::Square;
use reversi_core::types::Depth;

use crate::opening;
use crate::record::{GameRecord, append_records, read_last_game_id, truncate_incomplete_record};

/// Minimum number of random moves at the start of each game.
const MIN_RANDOM_MOVES: u8 = 14;

/// Exclusive upper bound on the number of random moves at the start of each game.
const MAX_RANDOM_MOVES: u8 = 50;

/// Minimum number of digits used in output file naming.
const FILE_ID_DIGITS: usize = 5;

/// Maximum size of the record cache.
const MAX_CACHE_SIZE: usize = 1_000_000;

/// Parameters controlling how random opening moves of a game are drawn.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RandomMoveChoice {
    /// Depth of the Multi-PV search that scores the candidates.
    pub depth: Depth,
    /// Maximum number of best moves to draw from.
    pub top: usize,
    /// Maximum score loss, in discs, of a candidate against the best move.
    pub margin: f32,
    /// Number of leading random moves drawn uniformly from all legal moves.
    pub uniform_moves: u8,
}

/// File rotation state across games to avoid re-scanning the output directory.
struct FileState {
    prefix: String,
    output_dir: PathBuf,
    games_per_file: u32,
    file_id: u32,
    game_id: u16,
}

impl FileState {
    fn new(prefix: &str, output_dir: &Path, games_per_file: u32) -> anyhow::Result<Self> {
        let latest_file = fs::read_dir(output_dir)
            .with_context(|| format!("failed to read output directory '{}'", output_dir.display()))?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let name = entry.file_name();
                let id = name
                    .to_str()?
                    .strip_prefix(prefix)?
                    .strip_prefix('_')?
                    .strip_suffix(".bin")?;
                if id.len() < FILE_ID_DIGITS || !id.bytes().all(|digit| digit.is_ascii_digit()) {
                    return None;
                }
                Some((id.parse::<u32>().ok()?, entry.path()))
            })
            .max_by_key(|&(id, _)| id);

        let mut file_id: u32 = 0;
        let mut game_id: u16 = 0;

        if let Some((latest_id, path)) = latest_file {
            file_id = latest_id;

            truncate_incomplete_record(&path)?;

            if let Some(last_id) = read_last_game_id(&path)? {
                let game_count = u32::from(last_id) + 1;
                if game_count >= games_per_file {
                    file_id += 1;
                } else {
                    game_id = last_id + 1;
                }
            }
        }

        Ok(Self {
            prefix: prefix.to_owned(),
            output_dir: output_dir.to_path_buf(),
            games_per_file,
            file_id,
            game_id,
        })
    }

    fn file_path(&self, file_id: u32) -> PathBuf {
        self.output_dir
            .join(format!("{}_{:0FILE_ID_DIGITS$}.bin", self.prefix, file_id))
    }

    /// Returns the next game id, advancing to a new file when the current one is full.
    fn next_game_id(&mut self) -> u16 {
        if u32::from(self.game_id) >= self.games_per_file {
            self.file_id += 1;
            self.game_id = 0;
        }
        let id = self.game_id;
        self.game_id += 1;
        id
    }

    /// Counts the total number of games written across all files.
    fn total_games(&self) -> anyhow::Result<usize> {
        let mut total = 0usize;
        for fid in 0..self.file_id {
            if let Some(last_id) = read_last_game_id(&self.file_path(fid))? {
                total += usize::from(last_id) + 1;
            }
        }
        total += usize::from(self.game_id);
        Ok(total)
    }

    fn write_records(&mut self, game_records: &[GameRecord]) -> anyhow::Result<()> {
        let file_path = self.file_path(self.file_id);
        append_records(&file_path, game_records)
    }
}

pub(crate) fn execute(
    num_games: u32,
    games_per_file: u32,
    hash_size: usize,
    level: Level,
    random_choice: RandomMoveChoice,
    prefix: &str,
    output_dir: &Path,
) -> anyhow::Result<()> {
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "failed to create output directory '{}'",
            output_dir.display()
        )
    })?;

    let options = SearchOptions::new(hash_size);

    let mut search = search::Search::new(&options);
    let mut record_cache: HashMap<Board, GameRecord> = HashMap::new();
    let mut file_state = FileState::new(prefix, output_dir, games_per_file)?;

    for _ in 0..num_games {
        let game_id = file_state.next_game_id();

        let mut rng = rand::rng();
        let num_random = cmp::min(
            rng.random_range(MIN_RANDOM_MOVES..MAX_RANDOM_MOVES),
            rng.random_range(MIN_RANDOM_MOVES..MAX_RANDOM_MOVES),
        );
        let opening_sequence = generate_random_opening(num_random, &mut search, random_choice);

        let game_records = play_game(
            &opening_sequence,
            &mut search,
            level,
            game_id,
            &mut record_cache,
        );
        file_state.write_records(&game_records)?;
    }
    Ok(())
}

pub(crate) fn execute_with_openings(
    openings_path: &Path,
    resume: bool,
    games_per_file: u32,
    hash_size: usize,
    level: Level,
    prefix: &str,
    output_dir: &Path,
) -> anyhow::Result<()> {
    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "failed to create output directory '{}'",
            output_dir.display()
        )
    })?;

    let options = SearchOptions::new(hash_size);

    let mut search = search::Search::new(&options);
    let mut record_cache: HashMap<Board, GameRecord> = HashMap::new();
    let mut file_state = FileState::new(prefix, output_dir, games_per_file)?;

    let opening_sequences = opening::load_openings(openings_path)?;

    if opening_sequences.is_empty() {
        anyhow::bail!("No valid opening sequences found in the file");
    }

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
        let game_id = file_state.next_game_id();
        let game_records = play_game(
            opening_sequence,
            &mut search,
            level,
            game_id,
            &mut record_cache,
        );
        file_state.write_records(&game_records)?;
    }
    Ok(())
}

/// Generates a random opening move sequence of up to `num_moves` plies.
///
/// Draws the first `choice.uniform_moves` moves uniformly from legal moves and
/// the remaining moves via [`candidate_move`].
fn generate_random_opening(
    num_moves: u8,
    search: &mut search::Search,
    choice: RandomMoveChoice,
) -> Vec<Square> {
    let mut opening = Vec::new();
    let mut game = GameState::new();

    while opening.len() < usize::from(num_moves) && !game.is_game_over() {
        let sq = if opening.len() < usize::from(choice.uniform_moves) {
            random_move(game.board())
        } else {
            candidate_move(game.board(), search, choice)
        };
        game.make_move(sq).expect("random moves are legal");
        opening.push(sq);
    }

    opening
}

/// Plays a single game starting from `opening_sequence` and returns its [`GameRecord`] entries.
fn play_game(
    opening_sequence: &[Square],
    search: &mut search::Search,
    lv: Level,
    game_id: u16,
    record_cache: &mut HashMap<Board, GameRecord>,
) -> Vec<GameRecord> {
    let game_start = Instant::now();
    search.init();

    let mut game = GameState::new();
    let mut game_records = Vec::new();

    for &sq in opening_sequence {
        if game.is_game_over() {
            break;
        }

        let mut record = record_cache
            .entry(*game.board())
            .or_insert_with(|| search_record(search, &game, lv, game_id, true))
            .clone();
        record.game_id = game_id;
        record.side_to_move = game.side_to_move();
        game_records.push(record);

        game.make_move(sq).expect("opening moves are legal");
    }

    if record_cache.len() > MAX_CACHE_SIZE {
        record_cache.clear();
    }

    while !game.is_game_over() {
        let record = search_record(search, &game, lv, game_id, false);
        game.make_move(record.sq)
            .expect("search returned an illegal move");
        game_records.push(record);
    }

    if !game_records.is_empty() {
        let final_side_to_move = game.side_to_move();
        let final_board = *game.board();
        let final_score = final_board
            .solve_disc_diff(final_board.empty_count())
            .value() as i8;

        for record in &mut game_records {
            let outcome = if record.side_to_move == final_side_to_move {
                final_score
            } else {
                -final_score
            };
            record.game_score = Some(outcome);
        }
    }

    let random_moves = game_records.iter().filter(|r| r.is_random).count();
    let total_moves = game_records.len();
    let black_score = game_records.first().and_then(|r| r.game_score).unwrap_or(0);
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

/// Searches the current position of `game` and returns a [`GameRecord`] with `game_score: None`.
fn search_record(
    search: &mut search::Search,
    game: &GameState,
    lv: Level,
    game_id: u16,
    is_random: bool,
) -> GameRecord {
    let board = *game.board();
    let result = search.run(&board, &SearchRunOptions::with_level(lv));
    GameRecord {
        game_id,
        ply: 60 - board.empty_count() as u8,
        board,
        score: result
            .score()
            .expect("search returned no legal move")
            .to_disc_diff_f32(),
        game_score: None,
        side_to_move: game.side_to_move(),
        is_random,
        sq: result.best_move().expect("search returned no legal move"),
    }
}

/// Selects a random legal move from `board`.
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    board.legal_moves().iter().choose(&mut rng).unwrap()
}

/// Draws a move uniformly from the best `choice.top` moves of a Multi-PV
/// search that score within `choice.margin` discs of the best move.
fn candidate_move(board: &Board, search: &mut search::Search, choice: RandomMoveChoice) -> Square {
    let level = Level::uniform(choice.depth, choice.depth);
    let result = search.run(board, &SearchRunOptions::with_level(level).multi_pv(true));
    let moves = result.pv_moves();
    let best_score = moves
        .first()
        .expect("search returned no legal move")
        .score
        .to_disc_diff_f32();
    moves
        .iter()
        .take(choice.top)
        .take_while(|m| m.score.to_disc_diff_f32() >= best_score - choice.margin)
        .choose(&mut rand::rng())
        .expect("search returned no legal move")
        .sq
}
