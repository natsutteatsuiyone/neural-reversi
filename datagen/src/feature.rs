//! Feature extraction module for neural network training.
//!
//! This module processes game records from binary files and extracts pattern features
//! for training the evaluation neural network. It handles board symmetries to augment
//! the training data and outputs compressed feature files.

use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufWriter},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use byteorder::{LittleEndian, WriteBytesExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle, ProgressDrawTarget};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use zstd::stream::write::Encoder;

use reversi_core::board::Board;
use reversi_core::eval::pattern_feature;

const COMPRESSION_LEVEL: i32 = 7;
const BATCH_SIZE: usize = 1;

/// Represents a single game position with evaluation and metadata.
///
/// Each record contains the board state (as bitboards) along with the
/// evaluation score and game progress information.
#[derive(Debug)]
struct GameRecord {
    /// Player's pieces as a bitboard (current player to move)
    player: u64,
    /// Opponent's pieces as a bitboard
    opponent: u64,
    /// Evaluation score from the engine's perspective (-64.0 to 64.0)
    score: f32,
    /// Move number in the game (0-59)
    ply: u8,
}

/// Processes game records from binary files and extracts pattern features.
///
/// This function reads game records from `.bin` files in the input directory,
/// extracts pattern features from each position (including all 8 symmetries),
/// and writes compressed feature files to the output directory.
///
/// # Arguments
///
/// * `input_dir` - Directory containing binary game record files
/// * `output_dir` - Directory where compressed feature files will be written
/// * `threads` - Number of parallel threads to use for processing
/// * `score_correction` - Whether to apply endgame score correction
///
/// # Returns
///
/// Returns `Ok(())` on success, or an `io::Error` if file operations fail.
pub fn execute(input_dir: &str, output_dir: &str, threads: usize, score_correction: bool) -> io::Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let input_dir = Path::new(input_dir);
    let output_dir = Path::new(output_dir);

    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    println!("Scanning input directory: {}", input_dir.display());
    let mut entries = fs::read_dir(input_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let total_files = entries.len();
    if total_files == 0 {
        println!("No bin files found to process.");
        return Ok(());
    }
    println!("Found {} bin files to process", total_files);
    entries.shuffle(&mut rand::rng());

    let entry_groups: Vec<_> = entries.chunks(BATCH_SIZE).collect();
    let processed_files_count = AtomicUsize::new(0);
    let start_time = std::time::Instant::now();

    let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    let pb = mp.add(ProgressBar::new(total_files as u64));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) ETA:{eta_precise}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));


    entry_groups
        .par_iter()
        .enumerate()
        .for_each(|(group_idx, entry_group)| {
            if let Err(e) = process_file_group(group_idx, entry_group, output_dir, score_correction) {
                eprintln!("Failed to process group {}: {}", group_idx, e);
            }

            let completed_files =
                processed_files_count.fetch_add(entry_group.len(), Ordering::SeqCst) + entry_group.len();
            pb.set_position(completed_files as u64);
        });

    pb.finish_with_message("Feature generation completed");
    mp.clear()?;

    let total_time = start_time.elapsed();
    println!(
        "Feature generation completed in {:.2?} - Processed {} files",
        total_time, total_files
    );
    Ok(())
}

/// Processes a group of game files and generates a single feature file.
///
/// This function reads multiple game files, extracts unique positions with
/// all 8 symmetrical variations, and writes them to a compressed feature file.
/// Duplicate positions are automatically deduplicated.
///
/// # Arguments
///
/// * `group_idx` - Index of this file group (used for output filename)
/// * `entry_paths` - Paths to the game files to process
/// * `output_dir` - Directory where the feature file will be written
/// * `score_correction` - Whether to apply endgame score correction
fn process_file_group(
    group_idx: usize,
    entry_paths: &[std::path::PathBuf],
    output_dir: &Path,
    score_correction: bool,
) -> io::Result<()> {
    let mut unique_positions = HashMap::new();
    for path in entry_paths {
        let input_path_str = path.to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Path is not valid UTF-8")
        })?;

        let game_records = load_game_records(input_path_str, score_correction)
            .map_err(|e| io::Error::new(e.kind(), format!("Failed to load game records from {}: {}", input_path_str, e)))?;

        for record in game_records {
            let base_board = Board::from_bitboards(record.player, record.opponent);
            let score = record.score;
            let mobility = base_board.get_moves().count_ones() as u8;
            let ply = record.ply;

            let b90 = base_board.rotate_90_clockwise();
            let b180 = b90.rotate_90_clockwise();
            let b270 = b180.rotate_90_clockwise();
            let boards = [
                base_board,
                b90,
                b180,
                b270,
                base_board.flip_vertical(),
                base_board.flip_horizontal(),
                base_board.flip_diag_a1h8(),
                base_board.flip_diag_a8h1(),
            ];

            for board in boards {
                unique_positions
                    .entry(board)
                    .or_insert((score, mobility, ply));
            }
        }
    }

    let mut all_positions: Vec<_> = unique_positions.into_iter().collect();
    all_positions.shuffle(&mut rand::rng());

    let file_path = output_dir.join(format!("features_{}.zst", group_idx));
    let file = File::create(file_path)?;
    let buf_writer = BufWriter::new(file);
    let mut encoder = Encoder::new(buf_writer, COMPRESSION_LEVEL)?;
    for (board, (score, mobility, ply)) in all_positions {
        let mut features = [0; pattern_feature::NUM_PATTERN_FEATURES];
        pattern_feature::set_features(&board, &mut features);

        write_feature_record(&mut encoder, score, &features, mobility, ply)?;
    }

    encoder.finish()?;

    Ok(())
}

/// Loads game records from a binary file.
///
/// The binary file format contains 24-byte records with the following structure:
/// - Bytes 0-7: Player bitboard (u64, little-endian)
/// - Bytes 8-15: Opponent bitboard (u64, little-endian)
/// - Bytes 16-19: Evaluation score (f32, little-endian)
/// - Byte 20: Final game score (i8)
/// - Byte 21: Ply number (u8)
/// - Byte 22: Random move flag (u8, 1 if random)
/// - Byte 23: Move played (u8, unused)
///
/// # Arguments
///
/// * `file_path` - Path to the binary game file
/// * `score_correction` - Whether to blend evaluation scores with game outcomes
///
/// # Returns
///
/// Returns a vector of `GameRecord` structs on success.
fn load_game_records(file_path: &str, score_correction: bool) -> io::Result<Vec<GameRecord>> {
    let metadata = fs::metadata(file_path)?;
    let file_size = metadata.len() as usize;
    let entry_size = 24;

    if file_size % entry_size != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid data size. {}", file_path),
        ));
    }

    let num_entries = file_size / entry_size;
    let mut records = Vec::with_capacity(num_entries);
    let buffer = fs::read(file_path)?;

    for chunk in buffer.chunks_exact(entry_size) {
        let player = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let opponent = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let mut score = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
        let game_score = chunk[20] as i8;
        let ply = chunk[21];
        let is_random = chunk[22] == 1;

        if ply <= 1 {
            score = 0.0;
        } else if !is_random && score_correction {
            score = ((ply as f32 * game_score as f32) + (59.0 - ply as f32) * score) / 59.0;
        }

        assert!(player & opponent == 0, "Player and opponent bitboards overlap: {} {}", player, opponent);
        assert!((-64.0..=64.0).contains(&score), "Score out of range: {}", score);
        assert!((-64..=64).contains(&game_score), "Game score out of range: {}", game_score);
        assert!(ply <= 59, "Ply value out of range: {}", ply);

        records.push(GameRecord {
            player,
            opponent,
            score,
            ply,
        });
    }

    Ok(records)
}

/// Writes a single feature record to the compressed output file.
///
/// The output format for each record is:
/// - 4 bytes: Evaluation score (f32, little-endian)
/// - N*2 bytes: Pattern features (u16 array, little-endian)
/// - 1 byte: Mobility count (number of legal moves)
/// - 1 byte: Ply number
///
/// # Arguments
///
/// * `encoder` - Zstandard encoder for compression
/// * `score` - Evaluation score for this position
/// * `features` - Array of pattern feature indices
/// * `mobility` - Number of legal moves in this position
/// * `ply` - Move number in the game
fn write_feature_record(
    encoder: &mut Encoder<BufWriter<File>>,
    score: f32,
    features: &[u16],
    mobility: u8,
    ply: u8,
) -> io::Result<()> {
    encoder.write_f32::<LittleEndian>(score)?;
    for &feature in features {
        encoder.write_u16::<LittleEndian>(feature)?;
    }
    encoder.write_u8(mobility)?;
    encoder.write_u8(ply)?;
    Ok(())
}
