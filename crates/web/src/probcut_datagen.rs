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
    board::Board, disc::Disc, probcut::Selectivity, square::Square,
    transposition_table::TranspositionTable, types::Depth,
};

use crate::{eval::Eval, level::Level, search::Search};

/// Transposition table size in MB for search
const TT_SIZE_MB: usize = 64;

/// Total number of search depths to test
const NUM_SEARCH_DEPTHS: usize = 10;

/// Maximum shallow depth for ProbCut analysis
const MAX_SHALLOW_DEPTH: usize = 5;

/// Minimum depth difference between shallow and deep search
const MIN_DEPTH_DIFFERENCE: Depth = 2;

/// Search selectivity level (none for accurate measurements)
const SELECTIVITY: Selectivity = Selectivity::None;

/// Represents a single ProbCut training data sample.
#[derive(Debug)]
struct ProbCutSample {
    /// Move number in the game (0-59)
    ply: u32,
    /// Shallow search depth
    shallow_depth: Depth,
    /// Score from shallow search
    shallow_score: f32,
    /// Deep search depth
    deep_depth: Depth,
    /// Score from deep search
    deep_score: f32,
}

/// Result returned to JavaScript containing generated samples and statistics.
#[wasm_bindgen]
pub struct ProbCutDatagenResult {
    samples_csv: String,
    total_samples: u32,
    total_positions: u32,
    cache_hit_rate: f64,
}

#[wasm_bindgen]
impl ProbCutDatagenResult {
    /// Get the samples as a CSV string.
    ///
    /// Format: `ply,shallow_depth,shallow_score,deep_depth,deep_score,diff`
    #[wasm_bindgen(getter)]
    pub fn samples_csv(&self) -> String {
        self.samples_csv.clone()
    }

    /// Get the total number of samples generated.
    #[wasm_bindgen(getter)]
    pub fn total_samples(&self) -> u32 {
        self.total_samples
    }

    /// Get the total number of positions analyzed.
    #[wasm_bindgen(getter)]
    pub fn total_positions(&self) -> u32 {
        self.total_positions
    }

    /// Get the cache hit rate (0.0 to 1.0).
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
    tt: Rc<TranspositionTable>,
    score_cache: HashMap<(Board, Depth), f32>,
    cache_hits: usize,
    cache_misses: usize,
}

#[wasm_bindgen]
impl ProbCutDatagen {
    /// Create a new ProbCut data generator.
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
        let search = Search::new(Rc::clone(&tt), eval);

        Ok(ProbCutDatagen {
            search,
            tt,
            score_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
        })
    }

    /// Process a single game sequence and generate training samples.
    ///
    /// # Arguments
    ///
    /// * `game_sequence` - Move sequence string (e.g., "D3C5F6F5E6C6D6...")
    ///
    /// # Returns
    ///
    /// Result containing generated samples and statistics.
    pub fn process_game(&mut self, game_sequence: &str) -> Result<ProbCutDatagenResult, JsValue> {
        let samples = self.process_game_internal(game_sequence)?;
        self.build_result(samples)
    }

    /// Process multiple games with optional progress callback.
    ///
    /// # Arguments
    ///
    /// * `games` - Newline-separated game sequences
    /// * `progress_callback` - Optional JS function called with progress info
    ///
    /// # Returns
    ///
    /// Result containing all generated samples and statistics.
    pub fn process_games(
        &mut self,
        games: &str,
        progress_callback: Option<Function>,
    ) -> Result<ProbCutDatagenResult, JsValue> {
        let lines: Vec<&str> = games.lines().filter(|l| !l.trim().is_empty()).collect();
        let total_games = lines.len();
        let mut all_samples = Vec::new();

        for (idx, line) in lines.iter().enumerate() {
            let samples = self.process_game_internal(line)?;
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

        self.build_result(all_samples)
    }

    /// Clear all caches and reset statistics.
    pub fn clear(&mut self) {
        self.score_cache.clear();
        self.tt.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Get the current number of cached scores.
    #[wasm_bindgen(getter)]
    pub fn cache_size(&self) -> u32 {
        self.score_cache.len() as u32
    }
}

impl ProbCutDatagen {
    /// Internal method to process a single game and return samples.
    fn process_game_internal(
        &mut self,
        game_sequence: &str,
    ) -> Result<Vec<ProbCutSample>, JsValue> {
        let game_sequence = game_sequence.trim();
        if game_sequence.is_empty() {
            return Ok(Vec::new());
        }

        let mut samples = Vec::new();
        let mut board = Board::new();
        let mut side_to_move = Disc::Black;

        // Parse and process each move
        for token in game_sequence.as_bytes().chunks_exact(2) {
            let move_str = std::str::from_utf8(token)
                .map_err(|e| JsValue::from_str(&format!("Invalid UTF-8 in move token: {}", e)))?;

            let sq = move_str
                .parse::<Square>()
                .map_err(|e| JsValue::from_str(&format!("Invalid move '{}': {}", move_str, e)))?;

            // Handle pass if no legal moves
            if !board.has_legal_moves() {
                board = board.switch_players();
                side_to_move = side_to_move.opposite();
                if !board.has_legal_moves() {
                    break;
                }
            }

            let ply = 60 - board.get_empty_count();

            // Search at multiple depths and collect scores
            let depth_scores = self.search_all_depths(&board);

            // Generate samples from shallow/deep pairs
            for (shallow_depth, shallow_score) in depth_scores.iter().take(MAX_SHALLOW_DEPTH + 1) {
                for (deep_depth, deep_score) in depth_scores.iter() {
                    if *deep_depth > *shallow_depth + MIN_DEPTH_DIFFERENCE {
                        samples.push(ProbCutSample {
                            ply,
                            shallow_depth: *shallow_depth,
                            shallow_score: *shallow_score,
                            deep_depth: *deep_depth,
                            deep_score: *deep_score,
                        });
                    }
                }
            }

            // Make the move
            board = board.make_move(sq);
            side_to_move = side_to_move.opposite();
        }

        Ok(samples)
    }

    /// Search at all depths from 0 to NUM_SEARCH_DEPTHS.
    fn search_all_depths(&mut self, board: &Board) -> Vec<(Depth, f32)> {
        (0..=NUM_SEARCH_DEPTHS)
            .map(|d| {
                let depth = d as Depth;
                let score = self.search_at_depth(board, depth);
                (depth, score)
            })
            .collect()
    }

    /// Search at a specific depth, using cache if available.
    fn search_at_depth(&mut self, board: &Board, depth: Depth) -> f32 {
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

        let result = self.search.run(board, level, SELECTIVITY, None);
        let score = result.score;

        self.score_cache.insert(cache_key, score);
        score
    }

    /// Build the result object from collected samples.
    fn build_result(&self, samples: Vec<ProbCutSample>) -> Result<ProbCutDatagenResult, JsValue> {
        // Convert samples to CSV
        let samples_csv = Self::samples_to_csv(&samples);

        let total_lookups = self.cache_hits + self.cache_misses;
        let cache_hit_rate = if total_lookups > 0 {
            self.cache_hits as f64 / total_lookups as f64
        } else {
            0.0
        };

        // Calculate unique positions from samples
        let total_positions = samples
            .iter()
            .map(|s| s.ply)
            .collect::<std::collections::HashSet<_>>()
            .len() as u32;

        Ok(ProbCutDatagenResult {
            samples_csv,
            total_samples: samples.len() as u32,
            total_positions,
            cache_hit_rate,
        })
    }

    /// Convert samples to CSV string.
    ///
    /// Format matches `crates/datagen/src/probcut.rs`:
    /// `ply,shallow_depth,shallow_score,deep_depth,deep_score,diff`
    fn samples_to_csv(samples: &[ProbCutSample]) -> String {
        let mut csv = String::from("ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n");

        for s in samples {
            let diff = s.deep_score - s.shallow_score;
            csv.push_str(&format!(
                "{},{},{},{},{},{}\n",
                s.ply, s.shallow_depth, s.shallow_score, s.deep_depth, s.deep_score, diff
            ));
        }

        csv
    }
}
