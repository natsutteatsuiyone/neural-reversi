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
    game_state::GameState,
    level::Level,
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

/// Starting ply for endgame ProbCut analysis
const ENDGAME_START_PLY: u32 = 30;

/// CSV header used by the existing endgame pipeline.
const CSV_HEADER: &[u8] = b"ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n";

/// Midgame CSV header with an explicit game-blocking key.
const MIDGAME_CSV_HEADER: &[u8] =
    b"game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n";

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
fn open_csv_io(
    input: &str,
    output: &str,
    header: &[u8],
) -> io::Result<(BufReader<File>, BufWriter<File>)> {
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
    writer.write_all(header)?;

    Ok((BufReader::new(input_file), writer))
}

/// Replays `moves` from the initial position, passing every position reached to
/// `visit` before its move is played, and returns the final position with the
/// side to move there.
///
/// Moves left over after the game has ended are ignored.
///
/// # Errors
///
/// Returns an error if a move is illegal when reached.
fn replay(moves: &[Square], mut visit: impl FnMut(&Board, Disc)) -> Result<(Board, Disc), String> {
    let mut game = GameState::new();

    for &sq in moves {
        if game.is_game_over() {
            break;
        }

        visit(game.board(), game.side_to_move());
        game.make_move(sq)?;
    }

    Ok((*game.board(), game.side_to_move()))
}

/// Records an exact midgame position and returns whether it has not been sampled yet.
///
/// Symmetric positions remain distinct because the neural evaluation is not
/// guaranteed to produce identical depth-limited scores for every orientation.
fn insert_midgame_position(visited: &mut HashSet<Board>, board: &Board) -> bool {
    visited.insert(*board)
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
                let run_options = SearchRunOptions::with_level(level)
                    .disable_probcut()
                    .with_eval_mode(eval_mode);
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
    writer: &mut impl Write,
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

/// Writes one midgame CSV row with the source game line number.
fn write_midgame_sample(
    writer: &mut impl Write,
    game: usize,
    sample: &ProbCutSample,
    deep_score: Scoref,
) -> io::Result<()> {
    write!(writer, "{game},")?;
    write_sample(writer, sample, deep_score)
}

/// Generates ProbCut training data.
///
/// Reads game sequences from the input file, analyzes each position with multiple
/// search depths, and outputs training data as CSV. Each exact position is sampled
/// once, while symmetric positions remain distinct because their depth-limited neural
/// evaluations can differ. Only depths below the position's empty count are searched,
/// so every sample reflects a genuine midgame search. The generated data includes
/// shallow/deep search correlations that can be used to train regression models for
/// ProbCut parameter calculation.
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

    let (reader, mut writer) = open_csv_io(input, output, MIDGAME_CSV_HEADER)?;

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

        let game = line_no + 1;
        let mut samples: Vec<(ProbCutSample, Scoref)> = Vec::new();
        replay(&moves, |board, side_to_move| {
            if !insert_midgame_position(&mut visited, board) {
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
        })
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Line {game}: {e}")))?;

        for (sample, deep_score) in samples.iter() {
            write_midgame_sample(&mut writer, game, sample, *deep_score)?;
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

    let (reader, mut writer) = open_csv_io(input, output, CSV_HEADER)?;

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
        })
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Line {}: {e}", line_no + 1),
            )
        })?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_rows_include_the_input_game_line_number() {
        let sample = ProbCutSample {
            ply: 12,
            shallow_depth: 2,
            shallow_score: 1.25,
            deep_depth: 7,
            side_to_move: Disc::Black,
        };
        let mut output = Vec::new();

        write_midgame_sample(&mut output, 17, &sample, -0.5).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            "17,12,2,1.25,7,-0.5,-1.75\n"
        );
    }

    #[test]
    fn endgame_csv_contract_remains_six_columns() {
        let sample = ProbCutSample {
            ply: 40,
            shallow_depth: 4,
            shallow_score: -2.0,
            deep_depth: 20,
            side_to_move: Disc::White,
        };
        let mut output = Vec::new();

        write_sample(&mut output, &sample, 6.0).unwrap();

        assert_eq!(
            CSV_HEADER,
            b"ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n"
        );
        assert_eq!(String::from_utf8(output).unwrap(), "40,4,-2,20,6,8\n");
    }

    #[test]
    fn replay_rejects_illegal_move() {
        let moves = Square::parse_sequence("f5f5").unwrap();

        assert!(replay(&moves, |_, _| {}).is_err());
    }

    #[test]
    fn midgame_dedup_keeps_symmetric_positions_distinct() {
        let moves = Square::parse_sequence("f5d6c4d3").unwrap();
        let rotated_moves: Vec<_> = moves
            .iter()
            .map(|sq| Square::from_usize(63 - sq.index()).unwrap())
            .collect();
        let board = replay(&moves, |_, _| {}).unwrap().0;
        let rotated_board = replay(&rotated_moves, |_, _| {}).unwrap().0;

        assert_eq!(rotated_board, board.rotate_180_clockwise());
        assert_eq!(board.unique(), rotated_board.unique());

        let mut visited = HashSet::new();
        assert!(insert_midgame_position(&mut visited, &board));
        assert!(insert_midgame_position(&mut visited, &rotated_board));
        assert!(!insert_midgame_position(&mut visited, &board));
    }
}
