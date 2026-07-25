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
    collections::HashSet,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
};

use reversi_core::{
    board::Board,
    constants::INITIAL_EMPTY_COUNT,
    disc::Disc,
    eval::{Eval, EvalMode},
    level::Level,
    probcut::Selectivity,
    search::{
        Search, SearchRunOptions,
        options::SearchOptions,
        strategy::{EndGameStrategy, SearchStrategy},
    },
    square::Square,
    types::{Depth, Scoref},
};

/// Transposition table size in MB for search
const TT_SIZE_MB: usize = 256;

/// Maximum deep search depth for midgame ProbCut analysis
const MAX_SEARCH_DEPTH: Depth = 14;

/// Maximum shallow depth for midgame ProbCut analysis
const MAX_SHALLOW_DEPTH: Depth = 7;

/// Maximum search depth sampled for endgame ProbCut analysis
const MAX_ENDGAME_SEARCH_DEPTH: Depth = 12;

/// Minimum depth difference between shallow and deep search
const MIN_DEPTH_DIFFERENCE: Depth = 2;

/// Search selectivity level
const SELECTIVITY: Selectivity = Selectivity::None;

/// Starting ply for endgame ProbCut analysis
const ENDGAME_START_PLY: u32 = 30;

/// CSV header shared by both generators
const CSV_HEADER: &[u8] = b"ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n";

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
    /// Side to move
    side_to_move: Disc,
}

/// Opens the game-sequence input and the CSV output, writing the CSV header.
fn open_csv_io(input: &str, output: &str) -> io::Result<(BufReader<File>, BufWriter<File>)> {
    let input_file = File::open(input).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open input file '{input}': {e}"),
        )
    })?;

    let output_file = File::create(output).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{output}': {e}"),
        )
    })?;
    let mut writer = BufWriter::new(output_file);
    writer.write_all(CSV_HEADER)?;

    Ok((BufReader::new(input_file), writer))
}

/// Replays `moves` from the initial position, passing every position reached to
/// `visit` before its move is played, and returns the final position with the
/// side to move there.
fn replay(moves: &[Square], mut visit: impl FnMut(&Board, Disc)) -> (Board, Disc) {
    let mut board = Board::new();
    let mut side_to_move = Disc::Black;

    for &sq in moves {
        if !board.has_legal_moves() {
            board = board.switch_players();
            side_to_move = side_to_move.opposite();
            if !board.has_legal_moves() {
                break;
            }
        }

        visit(&board, side_to_move);
        board = board.make_move(sq);
        side_to_move = side_to_move.opposite();
    }

    (board, side_to_move)
}

/// Scores `board` at every depth in `0..=max_depth` that stays below the empty
/// count, so no sample comes from a position the endgame solver read out exactly.
///
/// Depth 0 is the static evaluation rather than a one-ply search, matching how
/// the engine resolves a shallow depth of 0.
fn depth_scores(
    search: &mut Search,
    eval: &Eval,
    board: &Board,
    max_depth: Depth,
    eval_mode: EvalMode,
) -> Vec<(Depth, Scoref)> {
    let n_empties = board.get_empty_count();

    (0..=max_depth)
        .filter(|&depth| depth < n_empties)
        .map(|depth| {
            let score = if depth == 0 {
                eval.evaluate_simple(board, eval_mode).to_disc_diff_f32()
            } else {
                let level = Level::with_depths(depth, [depth; 4]);
                let run_options =
                    SearchRunOptions::with_level(level, SELECTIVITY).with_eval_mode(eval_mode);
                search
                    .run(board, &run_options)
                    .score()
                    .expect("search returned no legal move")
            };
            (depth, score)
        })
        .collect()
}

/// Writes one CSV row, pairing the sample's shallow result with `deep_score`.
fn write_sample(
    writer: &mut BufWriter<File>,
    sample: &ProbCutSample,
    deep_score: Scoref,
) -> io::Result<()> {
    writeln!(
        writer,
        "{},{},{},{},{},{}",
        sample.ply,
        sample.shallow_depth,
        sample.shallow_score,
        sample.deep_depth,
        deep_score,
        deep_score - sample.shallow_score
    )
}

/// Generates ProbCut training data.
///
/// Reads game sequences from the input file, analyzes each position with multiple
/// search depths, and outputs training data as CSV. Each unique position (up to
/// symmetry) is sampled once, and only depths below the position's empty count are
/// searched so every sample reflects a genuine midgame search. The generated data
/// includes shallow/deep search correlations that can be used to train regression
/// models for ProbCut parameter calculation.
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
    let eval = search.eval().clone();
    let mut visited: HashSet<Board> = HashSet::new();

    let (reader, mut writer) = open_csv_io(input, output)?;

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

        let moves = Square::parse_sequence(line).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid move: {e}"))
        })?;

        let mut samples: Vec<(ProbCutSample, Scoref)> = Vec::new();
        replay(&moves, |board, side_to_move| {
            if !visited.insert(board.unique()) {
                return;
            }

            // A fresh TT per position keeps deep results of previously analyzed
            // positions from leaking into the shallow searches.
            search.init();
            let depth_scores =
                depth_scores(&mut search, &eval, board, MAX_SEARCH_DEPTH, EvalMode::Main);

            let ply = INITIAL_EMPTY_COUNT as u32 - board.get_empty_count();
            for (shallow_depth, shallow_score) in depth_scores
                .iter()
                .filter(|(depth, _)| *depth <= MAX_SHALLOW_DEPTH)
            {
                samples.extend(
                    depth_scores
                        .iter()
                        .filter(|(deep_depth, _)| {
                            *deep_depth > *shallow_depth + MIN_DEPTH_DIFFERENCE
                        })
                        .map(|(deep_depth, deep_score)| {
                            (
                                ProbCutSample {
                                    ply,
                                    shallow_depth: *shallow_depth,
                                    shallow_score: *shallow_score,
                                    deep_depth: *deep_depth,
                                    side_to_move,
                                },
                                *deep_score,
                            )
                        }),
                );
            }
        });

        for (sample, deep_score) in samples.iter() {
            write_sample(&mut writer, sample, *deep_score)?;
        }
        writer.flush()?;

        println!("Processed {} lines", line_no + 1);
    }

    println!("ProbCut training data generation completed successfully");
    Ok(())
}

/// Generates endgame ProbCut training data.
///
/// Reads game sequences from the input file, analyzes each position with multiple
/// search depths, and outputs training data as CSV. Only positions inside the window
/// where endgame ProbCut can fire are processed (ply >= 30 and enough empties for
/// [`EndGameStrategy::MIN_PROBCUT_DEPTH`]), and each unique position (up to symmetry)
/// is sampled once. The deep score is the final game score (disc difference), so input
/// games must play the endgame perfectly from ply 30 on (e.g. rewritten by
/// `correct-endgames`).
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
    let eval = search.eval().clone();
    let mut visited: HashSet<Board> = HashSet::new();

    let (reader, mut writer) = open_csv_io(input, output)?;

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

        let moves = Square::parse_sequence(line).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid move: {e}"))
        })?;

        let mut samples: Vec<ProbCutSample> = Vec::new();
        let (final_board, final_side_to_move) = replay(&moves, |board, side_to_move| {
            let n_empties = board.get_empty_count();
            let ply = INITIAL_EMPTY_COUNT as u32 - n_empties;
            if ply < ENDGAME_START_PLY
                || n_empties < EndGameStrategy::MIN_PROBCUT_DEPTH
                || !visited.insert(board.unique())
            {
                return;
            }

            // A fresh TT per position keeps deep results of previously analyzed
            // positions from leaking into the shallow searches.
            search.init();
            let depth_scores = depth_scores(
                &mut search,
                &eval,
                board,
                MAX_ENDGAME_SEARCH_DEPTH,
                EvalMode::Small,
            );

            samples.extend(depth_scores.iter().map(|(shallow_depth, shallow_score)| {
                ProbCutSample {
                    ply,
                    shallow_depth: *shallow_depth,
                    shallow_score: *shallow_score,
                    deep_depth: n_empties,
                    side_to_move,
                }
            }));
        });

        let score = final_board.solve(final_board.get_empty_count()) as f32;

        for sample in samples.iter() {
            let deep_score = if sample.side_to_move == final_side_to_move {
                score
            } else {
                -score
            };
            write_sample(&mut writer, sample, deep_score)?;
        }
        writer.flush()?;

        println!("Processed {} lines", line_no + 1);
    }

    println!("Endgame ProbCut training data generation completed successfully");
    Ok(())
}
