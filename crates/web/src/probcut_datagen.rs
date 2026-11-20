//! ProbCut training data generation for WebAssembly.
//!
//! This module provides functionality to generate ProbCut training data in the browser.
//! It analyzes game positions at multiple search depths and records the correlation
//! between shallow and deep search results for training regression models.

use std::collections::HashMap;
use std::rc::Rc;

use js_sys::Function;
use wasm_bindgen::prelude::*;

use reversi_core::{
    board::Board, game_state::GameState, player_color::PlayerColor, score::Score, square::Square,
    types::Depth,
};

use crate::{
    eval::Eval, level::Level, probcut::Selectivity, search::Search,
    transposition_table::TranspositionTable,
};

/// Transposition table size in MB for search.
const TT_SIZE_MB: usize = 128;

/// Maximum search depth to test (depths 0 through this value).
const NUM_SEARCH_DEPTHS: usize = 10;

/// Maximum shallow depth for ProbCut analysis.
const MAX_SHALLOW_DEPTH: usize = 5;

/// Depth difference that a deep search must exceed over the shallow search.
const MIN_DEPTH_DIFFERENCE: Depth = 2;

/// Search selectivity level (none for accurate measurements).
const SELECTIVITY: Selectivity = Selectivity::None;

/// Shallow search depth used for endgame ProbCut analysis.
const ENDGAME_SHALLOW_DEPTH: Depth = 2;

/// Starting ply for endgame ProbCut analysis (positions earlier than this are skipped).
const ENDGAME_START_PLY: u32 = 30;

/// Represents a single ProbCut training data sample.
#[derive(Debug)]
struct ProbCutSample {
    game: u64,
    /// Move number in the game (0-59)
    ply: u32,
    /// Shallow search depth
    shallow_depth: Depth,
    /// Score from shallow search
    shallow_score: Score,
    /// Deep search depth
    deep_depth: Depth,
    /// Score from deep search
    deep_score: Score,
}

/// Result returned to JavaScript containing generated samples and statistics.
#[wasm_bindgen]
pub struct ProbCutDatagenResult {
    samples_csv: String,
    total_samples: u32,
    total_positions: u32,
    skipped_games: u32,
    cache_hit_rate: f64,
}

#[wasm_bindgen]
impl ProbCutDatagenResult {
    /// Returns the samples as a CSV string.
    ///
    /// Format: `game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff`.
    /// The game ID is the one-based source input line number, including blank lines.
    #[wasm_bindgen(getter)]
    pub fn samples_csv(&self) -> String {
        self.samples_csv.clone()
    }

    /// Returns the total number of samples generated.
    #[wasm_bindgen(getter)]
    pub fn total_samples(&self) -> u32 {
        self.total_samples
    }

    /// Returns the total number of positions analyzed.
    #[wasm_bindgen(getter)]
    pub fn total_positions(&self) -> u32 {
        self.total_positions
    }

    /// Returns the number of games left out of the endgame data because
    /// they stop before the end and so have no final result.
    #[wasm_bindgen(getter)]
    pub fn skipped_games(&self) -> u32 {
        self.skipped_games
    }

    /// Returns the cache hit rate (0.0 to 1.0).
    #[wasm_bindgen(getter)]
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache_hit_rate
    }
}

/// ProbCut training data generator for WebAssembly.
///
/// This struct manages the search engine and caching for efficient
/// generation of training data across multiple games.
#[wasm_bindgen]
pub struct ProbCutDatagen {
    search: Search,
    score_cache: HashMap<(Board, Depth), Score>,
    cache_hits: usize,
    cache_misses: usize,
    positions: u32,
}

#[wasm_bindgen]
impl ProbCutDatagen {
    /// Creates a new ProbCut data generator.
    ///
    /// # Errors
    ///
    /// Returns an error if the evaluation network fails to load.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<ProbCutDatagen, JsValue> {
        console_error_panic_hook::set_once();

        let tt = Rc::new(TranspositionTable::new(TT_SIZE_MB));
        let eval = Rc::new(Eval::new().map_err(|e| {
            JsValue::from_str(&format!("Failed to load evaluation network: {}", e))
        })?);
        Ok(ProbCutDatagen {
            search: Search::new(tt, eval),
            score_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            positions: 0,
        })
    }

    /// Processes multiple newline-separated game sequences with an optional progress callback.
    ///
    /// A game may stop before the end.
    ///
    /// # Errors
    ///
    /// Returns an error if any game sequence is not a legal game (an invalid
    /// move token, an illegal move, or a move after the game has ended), or if
    /// the progress callback invocation fails.
    pub fn process_games(
        &mut self,
        games: &str,
        progress_callback: Option<Function>,
    ) -> Result<ProbCutDatagenResult, JsValue> {
        let lines: Vec<_> = games
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .collect();
        let total_games = lines.len();
        let mut all_samples = Vec::new();

        for (idx, &(line_index, line)) in lines.iter().enumerate() {
            let samples = self
                .process_game_internal(line_index as u64 + 1, line)
                .map_err(|e| JsValue::from_str(&e))?;
            all_samples.extend(samples);

            // Call progress callback if provided
            if let Some(ref callback) = progress_callback {
                let progress = js_sys::Object::new();
                js_sys::Reflect::set(
                    &progress,
                    &JsValue::from_str("game_index"),
                    &JsValue::from_f64((idx + 1) as f64),
                )?;
                js_sys::Reflect::set(
                    &progress,
                    &JsValue::from_str("total_games"),
                    &JsValue::from_f64(total_games as f64),
                )?;
                js_sys::Reflect::set(
                    &progress,
                    &JsValue::from_str("samples_so_far"),
                    &JsValue::from_f64(all_samples.len() as f64),
                )?;
                callback.call1(&JsValue::NULL, &progress)?;
            }
        }

        Ok(self.build_result(all_samples, 0))
    }

    /// Processes multiple newline-separated game sequences for endgame ProbCut analysis.
    ///
    /// Only positions with ply >= 30 are analyzed. The shallow search runs at a fixed
    /// depth and the deep score is the exact final game result (disc difference).
    /// A game that stops before the end has no final result, so it is left out
    /// and counted in [`ProbCutDatagenResult::skipped_games`].
    ///
    /// # Errors
    ///
    /// Returns an error if any game sequence is not a legal game (an invalid
    /// move token, an illegal move, or a move after the game has ended), or if
    /// the progress callback invocation fails.
    pub fn process_games_endgame(
        &mut self,
        games: &str,
        progress_callback: Option<Function>,
    ) -> Result<ProbCutDatagenResult, JsValue> {
        let lines: Vec<_> = games
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .collect();
        let total_games = lines.len();
        let mut all_samples = Vec::new();
        let mut skipped_games = 0;

        for (idx, &(line_index, line)) in lines.iter().enumerate() {
            match self
                .process_game_endgame_internal(line_index as u64 + 1, line)
                .map_err(|e| JsValue::from_str(&e))?
            {
                Some(samples) => all_samples.extend(samples),
                None => skipped_games += 1,
            }

            // Call progress callback if provided
            if let Some(ref callback) = progress_callback {
                let progress = js_sys::Object::new();
                js_sys::Reflect::set(
                    &progress,
                    &JsValue::from_str("game_index"),
                    &JsValue::from_f64((idx + 1) as f64),
                )?;
                js_sys::Reflect::set(
                    &progress,
                    &JsValue::from_str("total_games"),
                    &JsValue::from_f64(total_games as f64),
                )?;
                js_sys::Reflect::set(
                    &progress,
                    &JsValue::from_str("samples_so_far"),
                    &JsValue::from_f64(all_samples.len() as f64),
                )?;
                callback.call1(&JsValue::NULL, &progress)?;
            }
        }

        Ok(self.build_result(all_samples, skipped_games))
    }
}

/// Replays the game on input line `game` and returns every position a move
/// was played from, with its side to move, and the state after the last move.
///
/// # Errors
///
/// Returns an error naming the line if a move token is not a square, a move is
/// illegal when reached, or a move follows the end of the game.
fn replay_game(
    game: u64,
    game_sequence: &str,
) -> Result<(Vec<(Board, PlayerColor)>, GameState), String> {
    let moves = Square::parse_sequence(game_sequence).map_err(|e| format!("Line {game}: {e}"))?;
    let mut state = GameState::new();
    let mut positions = Vec::with_capacity(moves.len());

    for (i, &sq) in moves.iter().enumerate() {
        if !state.board().is_legal_move(sq) {
            let reason = if state.is_game_over() {
                "the game is already over"
            } else {
                "illegal move"
            };
            return Err(format!("Line {game}: move {} ({sq}): {reason}", i + 1));
        }
        positions.push((*state.board(), state.side_to_move()));
        state.make_move(sq).map_err(|e| e.to_string())?;
    }

    Ok((positions, state))
}

impl ProbCutDatagen {
    /// Processes a single game and returns the collected samples.
    fn process_game_internal(
        &mut self,
        game: u64,
        game_sequence: &str,
    ) -> Result<Vec<ProbCutSample>, String> {
        let (positions, _) = replay_game(game, game_sequence)?;
        let mut samples = Vec::new();

        for (board, _) in positions {
            let ply = 60 - board.empty_count();

            // Search at multiple depths and collect scores
            let depth_scores = self.search_all_depths(&board);

            // Generate samples from shallow/deep pairs
            for (shallow_depth, shallow_score) in depth_scores.iter().take(MAX_SHALLOW_DEPTH + 1) {
                for (deep_depth, deep_score) in depth_scores.iter() {
                    if *deep_depth > *shallow_depth + MIN_DEPTH_DIFFERENCE {
                        samples.push(ProbCutSample {
                            game,
                            ply,
                            shallow_depth: *shallow_depth,
                            shallow_score: *shallow_score,
                            deep_depth: *deep_depth,
                            deep_score: *deep_score,
                        });
                    }
                }
            }
        }

        Ok(samples)
    }

    /// Processes a single game for endgame ProbCut and returns the collected
    /// samples, or `None` if the game stops before the end.
    ///
    /// For each position at or beyond [`ENDGAME_START_PLY`], runs a single fixed-depth
    /// shallow search and pairs it with the exact final game result as the deep score.
    fn process_game_endgame_internal(
        &mut self,
        game: u64,
        game_sequence: &str,
    ) -> Result<Option<Vec<ProbCutSample>>, String> {
        let (positions, end) = replay_game(game, game_sequence)?;
        if !end.is_game_over() {
            return Ok(None);
        }

        // The deep score is the exact final result, oriented to each sample's side to move.
        let final_score = end.board().solve(end.board().empty_count());
        let mut samples = Vec::new();
        for (board, side_to_move) in positions {
            let n_empties = board.empty_count();
            let ply = 60 - n_empties;
            if ply >= ENDGAME_START_PLY && ENDGAME_SHALLOW_DEPTH < n_empties {
                samples.push(ProbCutSample {
                    game,
                    ply,
                    shallow_depth: ENDGAME_SHALLOW_DEPTH,
                    shallow_score: self.search_endgame_shallow(&board),
                    deep_depth: n_empties,
                    deep_score: if side_to_move == end.side_to_move() {
                        final_score
                    } else {
                        -final_score
                    },
                });
            }
        }

        Ok(Some(samples))
    }

    /// Runs the fixed-depth shallow endgame search and returns its score.
    ///
    /// The transposition table is cleared first so the score reflects a clean
    /// fixed-depth search, uncontaminated by deeper entries from earlier positions.
    fn search_endgame_shallow(&mut self, board: &Board) -> Score {
        self.positions += 1;
        self.search.init();
        let level = Level {
            mid_depth: ENDGAME_SHALLOW_DEPTH,
            end_depth: ENDGAME_SHALLOW_DEPTH,
            perfect_depth: ENDGAME_SHALLOW_DEPTH,
        };
        self.search
            .run(board, level, SELECTIVITY, None, false)
            .score
    }

    /// Searches at all depths from 0 to `NUM_SEARCH_DEPTHS`.
    fn search_all_depths(&mut self, board: &Board) -> Vec<(Depth, Score)> {
        self.positions += 1;
        self.search.init();
        (0..=NUM_SEARCH_DEPTHS)
            .map(|d| {
                let depth = d as Depth;
                let score = self.search_at_depth(board, depth);
                (depth, score)
            })
            .collect()
    }

    /// Searches at a specific depth, using the cache if available.
    fn search_at_depth(&mut self, board: &Board, depth: Depth) -> Score {
        let cache_key = (board.unique(), depth);

        if let Some(&cached) = self.score_cache.get(&cache_key) {
            self.cache_hits += 1;
            return cached;
        }

        self.cache_misses += 1;

        let level = Level {
            mid_depth: depth,
            end_depth: depth,
            perfect_depth: depth,
        };

        let score = self
            .search
            .run(board, level, SELECTIVITY, None, false)
            .score;

        self.score_cache.insert(cache_key, score);
        score
    }

    /// Builds the result object from collected samples.
    fn build_result(
        &self,
        samples: Vec<ProbCutSample>,
        skipped_games: u32,
    ) -> ProbCutDatagenResult {
        // Convert samples to CSV
        let samples_csv = Self::samples_to_csv(&samples);

        let total_lookups = self.cache_hits + self.cache_misses;
        let cache_hit_rate = if total_lookups > 0 {
            self.cache_hits as f64 / total_lookups as f64
        } else {
            0.0
        };

        ProbCutDatagenResult {
            samples_csv,
            total_samples: samples.len() as u32,
            total_positions: self.positions,
            skipped_games,
            cache_hit_rate,
        }
    }

    /// Converts samples to a CSV string.
    ///
    /// Columns: `game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff`.
    fn samples_to_csv(samples: &[ProbCutSample]) -> String {
        let mut csv =
            String::from("game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n");

        for s in samples {
            let diff = (s.deep_score - s.shallow_score).to_disc_diff_f32();
            csv.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                s.game,
                s.ply,
                s.shallow_depth,
                s.shallow_score.to_disc_diff_f32(),
                s.deep_depth,
                s.deep_score.to_disc_diff_f32(),
                diff
            ));
        }

        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator() -> ProbCutDatagen {
        ProbCutDatagen {
            search: Search::new(
                Rc::new(TranspositionTable::new(1)),
                Rc::new(Eval::new().unwrap()),
            ),
            score_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            positions: 0,
        }
    }

    /// Returns the moves of a game played to the end.
    fn finished_game() -> Vec<Square> {
        let mut state = GameState::new();
        let mut moves = Vec::new();
        while !state.is_game_over() {
            let sq = state.board().legal_moves().iter().next().unwrap();
            moves.push(sq);
            state.make_move(sq).unwrap();
        }
        moves
    }

    fn record(moves: &[Square]) -> String {
        moves.iter().map(Square::to_string).collect()
    }

    #[test]
    fn a_record_that_is_not_a_legal_game_is_rejected_with_its_line() {
        let finished = record(&finished_game());
        let cases = [
            ("f5f5", "Line 7: move 2 (f5): illegal move"),
            ("f5a1", "Line 7: move 2 (a1): illegal move"),
            ("f5d", "Line 7: "),
            ("f5zz", "Line 7: "),
            (&format!("{finished}a1"), "(a1): the game is already over"),
        ];

        let mut generator = generator();
        for (game_sequence, expected) in cases {
            let midgame = generator
                .process_game_internal(7, game_sequence)
                .map(|_| ())
                .unwrap_err();
            let endgame = generator
                .process_game_endgame_internal(7, game_sequence)
                .map(|_| ())
                .unwrap_err();
            assert!(midgame.contains(expected), "{game_sequence}: {midgame}");
            assert_eq!(midgame, endgame);
        }
        assert_eq!(generator.positions, 0);
    }

    #[test]
    fn endgame_leaves_out_a_game_that_stops_before_the_end() {
        let moves = finished_game();
        let finished = record(&moves);
        let cut = record(&moves[..moves.len() - 1]);
        assert!(moves.len() > ENDGAME_START_PLY as usize);

        let mut generator = generator();
        let result = generator
            .process_games_endgame(&format!("{cut}\n{finished}"), None)
            .unwrap();

        assert_eq!(result.skipped_games, 1);
        assert!(result.total_samples > 0);
        assert!(
            result
                .samples_csv
                .lines()
                .skip(1)
                .all(|line| line.starts_with("2,"))
        );
    }

    #[test]
    fn midgame_csv_preserves_source_game_line_numbers() {
        let mut generator = generator();
        let result = generator.process_games("D3C3\n\nD3C5", None).unwrap();
        let mut lines = result.samples_csv.lines();
        assert_eq!(
            lines.next(),
            Some("game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff")
        );

        let mut actual = std::collections::BTreeMap::new();
        let mut row_count = 0;
        for line in lines {
            let columns: Vec<_> = line.split(',').collect();
            assert_eq!(columns.len(), 7, "unexpected CSV row: {line}");
            let game = columns[0].parse::<u32>().unwrap();
            let ply = columns[1].parse::<u32>().unwrap();
            let shallow_depth = columns[2].parse::<u32>().unwrap();
            let deep_depth = columns[4].parse::<u32>().unwrap();
            *actual
                .entry((game, ply, shallow_depth, deep_depth))
                .or_insert(0) += 1;
            row_count += 1;
        }

        let mut expected = std::collections::BTreeMap::new();
        for game in [1, 3] {
            for ply in [0, 1] {
                for shallow_depth in 0..=5 {
                    for deep_depth in (shallow_depth + 3)..=10 {
                        expected.insert((game, ply, shallow_depth, deep_depth), 1);
                    }
                }
            }
        }
        assert_eq!(row_count, 132);
        assert_eq!(row_count, result.total_samples);
        assert_eq!(actual, expected);
    }

    #[test]
    fn endgame_csv_preserves_source_game_line_numbers() {
        let mut board = Board::new();
        let mut game_sequence = String::new();
        let mut expected_plies = std::collections::BTreeSet::new();
        loop {
            if !board.has_legal_moves() {
                board = board.switch_players();
                if !board.has_legal_moves() {
                    break;
                }
            }
            let n_empties = board.empty_count();
            let ply = 60 - n_empties;
            if ply >= ENDGAME_START_PLY && ENDGAME_SHALLOW_DEPTH < n_empties {
                expected_plies.insert(ply);
            }
            let sq = board.legal_moves().iter().next().unwrap();
            game_sequence.push_str(&sq.to_string());
            board = board.make_move(sq);
        }
        assert!(!expected_plies.is_empty());

        let mut generator = generator();
        let games = format!("{game_sequence}\n\n{game_sequence}");
        let result = generator.process_games_endgame(&games, None).unwrap();
        let mut lines = result.samples_csv.lines();
        assert_eq!(
            lines.next(),
            Some("game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff")
        );
        let mut actual = std::collections::BTreeMap::<_, Vec<_>>::new();
        for line in lines {
            assert_eq!(line.split(',').count(), 7, "unexpected CSV row: {line}");
            let (game, sample) = line.split_once(',').unwrap();
            actual
                .entry(game.parse::<u64>().unwrap())
                .or_default()
                .push(sample);
        }
        assert_eq!(actual.keys().copied().collect::<Vec<_>>(), [1, 3]);
        assert_eq!(actual[&1], actual[&3]);
        let actual_plies: std::collections::BTreeSet<_> = actual[&1]
            .iter()
            .map(|sample| sample.split(',').next().unwrap().parse::<u32>().unwrap())
            .collect();
        assert_eq!(actual_plies, expected_plies);
        assert_eq!(actual[&1].len(), expected_plies.len());
        assert_eq!(result.total_samples as usize, expected_plies.len() * 2);
    }

    #[test]
    fn midgame_scores_do_not_depend_on_previous_positions() {
        let eval = Rc::new(Eval::new().unwrap());
        let mut warm = ProbCutDatagen {
            search: Search::new(Rc::new(TranspositionTable::new(1)), Rc::clone(&eval)),
            score_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            positions: 0,
        };
        let mut cold = ProbCutDatagen {
            search: Search::new(Rc::new(TranspositionTable::new(1)), eval),
            score_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            positions: 0,
        };
        let previous = Board::new().make_move(Square::D3);
        let board = previous.make_move(Square::C3);
        warm.search_all_depths(&previous);
        let actual = warm.search_all_depths(&board);
        let expected = cold.search_all_depths(&board);
        assert_eq!(actual, expected);
    }
}
