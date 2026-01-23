//! ProbCut training data generation module.
//!
//! This module generates training data for calculating ProbCut parameters by analyzing
//! game positions with multiple search depths. For each position, it performs searches
//! at various depths and records the correlation between shallow and deep search results.
//!
//! The generated CSV data can then be used to train regression models that predict
//! deep search scores from shallow search results, enabling ProbCut optimizations
//! in the search algorithm.

use clap::Parser;
use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use reversi_core::{
    board::Board,
    disc::Disc,
    eval::EvalMode,
    level::get_level,
    probcut::Selectivity,
    search::{Search, SearchRunOptions, options::SearchOptions},
    square::Square,
    types::{Depth, Scoref},
};

/// Transposition table size in MB for search
const TT_SIZE_MB: usize = 256;

/// Total number of search depths to test
const NUM_SEARCH_DEPTHS: usize = 15;

/// Maximum shallow depth for ProbCut analysis
const MAX_SHALLOW_DEPTH: usize = 7;

/// Minimum depth difference between shallow and deep search
const MIN_DEPTH_DIFFERENCE: Depth = 2;

/// Search selectivity level
const SELECTIVITY: Selectivity = Selectivity::None;

/// Starting ply for endgame ProbCut analysis
const ENDGAME_START_PLY: u32 = 30;

/// Command line arguments for ProbCut training data generation.
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Input file containing game sequences (move sequences per line)
    #[arg(short, long)]
    input: PathBuf,

    /// Output CSV file for ProbCut training data
    #[arg(short, long)]
    output: PathBuf,
}

/// Represents a single ProbCut training data sample.
///
/// Each sample contains the shallow and deep search results for a position,
/// which will be used to train regression models for ProbCut parameter calculation.
#[derive(Debug)]
struct ProbCutSample {
    /// Move number in the game (0-59)
    ply: u32,
    /// Shallow search depth
    shallow_depth: Depth,
    /// Score from shallow search
    shallow_score: Scoref,
    /// Deep search depth
    deep_depth: Depth,
    /// Score from deep search
    deep_score: Scoref,
    /// Side to move
    #[allow(dead_code)]
    side_to_move: Disc,
}

/// Generates ProbCut training data.
///
/// Reads game sequences from the input file, analyzes each position with multiple
/// search depths, and outputs training data as CSV. The generated data includes
/// shallow/deep search correlations that can be used to train regression models
/// for ProbCut parameter calculation.
///
/// # Arguments
///
/// * `input` - Path to input file containing game sequences (one per line)
/// * `output` - Path to output CSV file for training data
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if file operations fail.
pub fn execute(input: &str, output: &str) -> io::Result<()> {
    let options = SearchOptions::new(TT_SIZE_MB);
    let mut search = Search::new(&options);

    // Cache for (board, depth) -> score to avoid redundant searches
    // This is especially effective for opening positions that are common across games
    let mut score_cache: HashMap<(Board, Depth), Scoref> = HashMap::new();
    let mut cache_hits: usize = 0;
    let mut cache_misses: usize = 0;

    let input_file = File::open(input).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open input file '{input}': {e}"),
        )
    })?;
    let reader = BufReader::new(input_file);

    let output_file = File::create(output).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{output}': {e}"),
        )
    })?;
    let mut writer = BufWriter::new(output_file);
    writer.write_all(b"ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n")?;

    for (line_no, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to read line {}: {}", line_no + 1, e),
            )
        })?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut samples = Vec::new();
        let mut board = Board::new();
        let mut side_to_move = Disc::Black;

        // Note: We don't reset TT between games to allow TT reuse for common positions
        for token in line.as_bytes().chunks_exact(2) {
            let move_str = std::str::from_utf8(token).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid UTF-8 in move token: {e}"),
                )
            })?;
            let sq = move_str.parse::<Square>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid move '{move_str}': {e}"),
                )
            })?;

            if !board.has_legal_moves() {
                board = board.switch_players();
                side_to_move = side_to_move.opposite();
                if !board.has_legal_moves() {
                    break;
                }
            }

            let num_depth = NUM_SEARCH_DEPTHS;
            let max_shallow_depth = MAX_SHALLOW_DEPTH;

            let ply = 60 - board.get_empty_count();

            let depth_scores: Vec<(Depth, Scoref)> = (0..=num_depth)
                .map(|depth| {
                    let cache_key = (board.unique(), depth as Depth);
                    if let Some(&cached_score) = score_cache.get(&cache_key) {
                        cache_hits += 1;
                        (depth as Depth, cached_score)
                    } else {
                        cache_misses += 1;
                        let mut level = get_level(depth);
                        level.end_depth = [depth as Depth; 6];
                        let run_options = SearchRunOptions::with_level(level, SELECTIVITY);
                        let result = search.run(&board, &run_options);
                        score_cache.insert(cache_key, result.score);
                        (depth as Depth, result.score)
                    }
                })
                .collect();

            for (shallow_depth, shallow_score) in depth_scores.iter().take(max_shallow_depth + 1) {
                samples.extend(
                    depth_scores
                        .iter()
                        .filter(|(deep_depth, _)| {
                            *deep_depth > *shallow_depth + MIN_DEPTH_DIFFERENCE
                        })
                        .map(|(deep_depth, deep_score)| ProbCutSample {
                            ply,
                            shallow_depth: *shallow_depth,
                            shallow_score: *shallow_score,
                            deep_depth: *deep_depth,
                            deep_score: *deep_score,
                            side_to_move,
                        }),
                );
            }

            board = board.make_move(sq);
            side_to_move = side_to_move.opposite();
        }

        for sample in samples.iter() {
            let line = format!(
                "{},{},{},{},{},{}\n",
                sample.ply,
                sample.shallow_depth,
                sample.shallow_score,
                sample.deep_depth,
                sample.deep_score,
                sample.deep_score - sample.shallow_score
            );
            writer.write_all(line.as_bytes())?;
        }
        writer.flush()?;

        println!("Processed {} lines", line_no + 1);
        let hit_rate = if cache_hits + cache_misses > 0 {
            cache_hits as f64 / (cache_hits + cache_misses) as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "Cache stats: {} hits, {} misses ({:.1}% hit rate)",
            cache_hits, cache_misses, hit_rate
        );
    }

    println!("ProbCut training data generation completed successfully");
    Ok(())
}

/// Generates endgame ProbCut training data.
///
/// Reads game sequences from the input file, analyzes each position with multiple
/// search depths, and outputs training data as CSV. Only positions with ply >= 30
/// are processed. The deep score is the final game score (disc difference).
///
/// # Arguments
///
/// * `input` - Path to input file containing game sequences (one per line)
/// * `output` - Path to output CSV file for training data
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if file operations fail.
pub fn execute_endgame(input: &str, output: &str) -> io::Result<()> {
    let options = SearchOptions {
        tt_mb_size: TT_SIZE_MB,
        ..Default::default()
    };
    let mut search = Search::new(&options);

    let input_file = File::open(input).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open input file '{input}': {e}"),
        )
    })?;
    let reader = BufReader::new(input_file);

    let output_file = File::create(output).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{output}': {e}"),
        )
    })?;
    let mut writer = BufWriter::new(output_file);
    writer.write_all(b"ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n")?;

    for (line_no, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to read line {}: {}", line_no + 1, e),
            )
        })?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut samples = Vec::new();
        let mut board = Board::new();
        let mut side_to_move = Disc::Black;

        for token in line.as_bytes().chunks_exact(2) {
            let move_str = std::str::from_utf8(token).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid UTF-8 in move token: {e}"),
                )
            })?;
            let sq = move_str.parse::<Square>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid move '{move_str}': {e}"),
                )
            })?;

            if !board.has_legal_moves() {
                board = board.switch_players();
                side_to_move = side_to_move.opposite();
                if !board.has_legal_moves() {
                    break;
                }
            }

            let num_depth = 12;
            let n_empties = board.get_empty_count();
            let ply = 60 - n_empties;
            if ply >= ENDGAME_START_PLY {
                search.init();
                let depth_scores: Vec<(Depth, Scoref)> = (0..=num_depth)
                    .filter(|depth| *depth < n_empties as usize)
                    .map(|depth| {
                        let mut level = get_level(depth);
                        level.end_depth = [depth as Depth; 6];
                        let run_options = SearchRunOptions::with_level(level, Selectivity::None)
                            .with_eval_mode(EvalMode::Small);
                        let result = search.run(&board, &run_options);
                        (depth as Depth, result.score)
                    })
                    .collect();

                for (shallow_depth, shallow_score) in depth_scores.iter() {
                    samples.push(ProbCutSample {
                        ply,
                        shallow_depth: *shallow_depth,
                        shallow_score: *shallow_score,
                        deep_depth: n_empties,
                        deep_score: 0f32,
                        side_to_move,
                    });
                }
            }

            board = board.make_move(sq);
            side_to_move = side_to_move.opposite();
        }

        let score = board.solve(board.get_empty_count()) as f32;

        for sample in samples.iter() {
            let deep_score = if sample.side_to_move == side_to_move {
                score
            } else {
                -score
            };

            let line = format!(
                "{},{},{},{},{},{}\n",
                sample.ply,
                sample.shallow_depth,
                sample.shallow_score,
                sample.deep_depth,
                deep_score,
                deep_score - sample.shallow_score
            );
            writer.write_all(line.as_bytes())?;
        }
        writer.flush()?;

        println!("Processed {} lines", line_no + 1);
    }

    println!("Endgame ProbCut training data generation completed successfully");
    Ok(())
}
