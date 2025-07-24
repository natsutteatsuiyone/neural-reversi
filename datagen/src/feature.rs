//! Feature extraction module for neural network training.
//!
//! This module processes game records from binary files and extracts pattern features
//! for training the evaluation neural network. It handles board symmetries to augment
//! the training data and outputs compressed feature files.

use std::{
    fs::{self, File},
    io::{self},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use byteorder::{LittleEndian, WriteBytesExt};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle, ProgressDrawTarget};
use rayon::prelude::*;

use reversi_core::board::Board;
use reversi_core::eval::pattern_feature;

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
/// * `ply_min` - Minimum ply value to include (0-59)
/// * `ply_max` - Maximum ply value to include (0-59)
///
/// # Returns
///
/// Returns `Ok(())` on success, or an `io::Error` if file operations fail.
pub fn execute(input_dir: &str, output_dir: &str, threads: usize, score_correction: bool, ply_min: u8, ply_max: u8) -> io::Result<()> {
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
    let entries = fs::read_dir(input_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("bin") {
                // Get file size for better work distribution
                let size = entry.metadata().ok()?.len();
                Some((path, size))
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
    println!("Found {total_files} bin files to process");

    let processed_files_count = AtomicUsize::new(0);
    let start_time = std::time::Instant::now();

    let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));

    // Main progress bar showing overall completion
    let main_pb = mp.add(ProgressBar::new(total_files as u64));
    main_pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} Overall [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) ETA:{eta_precise} | {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    main_pb.enable_steady_tick(Duration::from_millis(100));

    // Individual progress bars for each thread
    let thread_pbs: Vec<_> = (0..threads)
        .map(|i| {
            let pb = mp.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::with_template(
                    "  Thread {prefix:>2} {spinner:.yellow} {msg}"
                )
                .unwrap(),
            );
            pb.set_prefix(format!("{}", i + 1));
            pb.enable_steady_tick(Duration::from_millis(100));
            pb
        })
        .collect();

    let active_threads = AtomicUsize::new(0);


    entries
        .par_iter()
        .enumerate()
        .for_each(|(file_idx, (entry_path, file_size))| {
            let thread_id = rayon::current_thread_index().unwrap_or(0);
            let thread_pb = &thread_pbs[thread_id % threads];

            // Get filename for display
            let file_name = entry_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            // Update thread progress with current file
            let size_mb = *file_size as f64 / 1_048_576.0;
            thread_pb.set_message(format!("Processing: {file_name} ({size_mb:.1}MB)"));

            // Update active thread count
            let active = active_threads.fetch_add(1, Ordering::SeqCst) + 1;
            main_pb.set_message(format!("Active threads: {active}/{threads}"));

            let start = std::time::Instant::now();

            if let Err(e) = process_file(file_idx, entry_path, output_dir, score_correction, ply_min, ply_max) {
                eprintln!("Failed to process file {file_idx}: {e}");
                thread_pb.set_message(format!("Error: {file_name}"));
            } else {
                let duration = start.elapsed();
                let throughput = size_mb / duration.as_secs_f64();
                thread_pb.set_message(format!("Completed: {file_name} ({throughput:.1}MB/s)"));
            }

            let completed_files =
                processed_files_count.fetch_add(1, Ordering::SeqCst) + 1;
            main_pb.set_position(completed_files as u64);

            // Decrease active thread count
            active_threads.fetch_sub(1, Ordering::SeqCst);

            // Brief delay before clearing progress message
            std::thread::sleep(Duration::from_millis(500));
            thread_pb.set_message("");
        });

    // Finish all progress bars
    for pb in thread_pbs {
        pb.finish_and_clear();
    }
    main_pb.finish_with_message("Feature generation completed");
    mp.clear()?;

    let total_time = start_time.elapsed();
    let throughput = total_files as f64 / total_time.as_secs_f64();
    println!(
        "Feature generation completed in {total_time:.2?} - Processed {total_files} files ({throughput:.1} files/sec)"
    );
    Ok(())
}

/// Processes a single game file and generates 8 separate feature files for each symmetry.
///
/// This function reads a game file, extracts positions with their 8 symmetrical variations,
/// and writes each symmetry pattern to a separate compressed feature file.
/// Each symmetry pattern gets its own output file for better parallel processing.
///
/// # Arguments
///
/// * `file_idx` - Index of this file (used for output filename)
/// * `entry_path` - Path to the game file to process
/// * `output_dir` - Directory where the feature files will be written
/// * `score_correction` - Whether to apply endgame score correction
/// * `ply_min` - Minimum ply value to include (0-59)
/// * `ply_max` - Maximum ply value to include (0-59)
fn process_file(
    file_idx: usize,
    entry_path: &Path,
    output_dir: &Path,
    score_correction: bool,
    ply_min: u8,
    ply_max: u8,
) -> io::Result<()> {
    let input_path_str = entry_path.to_str().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "Path is not valid UTF-8")
    })?;

    let game_records = load_game_records(input_path_str, score_correction, ply_min, ply_max)
        .map_err(|e| io::Error::new(e.kind(), format!("Failed to load game records from {input_path_str}: {e}")))?;

    // Create 8 separate data vectors for each symmetry pattern
    let mut symmetry_data = vec![Vec::new(); 8];
    let symmetry_names = [
        "base", "rot90", "rot180", "rot270",
        "flip_v", "flip_h", "flip_diag_a1h8", "flip_diag_a8h1"
    ];

    // Process each game record and generate all 8 symmetries
    for record in game_records {
        let base_board = Board::from_bitboards(record.player, record.opponent);
        let score = record.score;
        let ply = record.ply;

        // Generate all 8 symmetrical boards
        let symmetrical_boards = [
            base_board,                          // 0: base
            base_board.rotate_90_clockwise(),    // 1: 90° rotation
            base_board.rotate_180_clockwise(),   // 2: 180° rotation
            base_board.rotate_270_clockwise(),   // 3: 270° rotation
            base_board.flip_vertical(),          // 4: vertical flip
            base_board.flip_horizontal(),        // 5: horizontal flip
            base_board.flip_diag_a1h8(),         // 6: diagonal a1-h8 flip
            base_board.flip_diag_a8h1(),         // 7: diagonal a8-h1 flip
        ];

        // Process each symmetry and add to corresponding data vector
        for (i, board) in symmetrical_boards.iter().enumerate() {
            let mobility = board.get_moves().count_ones() as u8;
            let mut features = [0; pattern_feature::NUM_PATTERN_FEATURES];
            pattern_feature::set_features(board, &mut features);

            write_feature_record(&mut symmetry_data[i], score, &features, mobility, ply)?;
        }
    }

    // Write each symmetry pattern to a separate file
    for (i, data) in symmetry_data.into_iter().enumerate() {
        if !data.is_empty() {
            let file_path = output_dir.join(format!("features_{}_{}.zst", file_idx, symmetry_names[i]));
            let file = File::create(file_path)?;

            // Compress and write data
            let compressed = zstd::bulk::compress(&data, 3)?;
            std::io::Write::write_all(&mut std::io::BufWriter::new(file), &compressed)?;
        }
    }

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
/// * `ply_min` - Minimum ply value to include (0-59)
/// * `ply_max` - Maximum ply value to include (0-59)
///
/// # Returns
///
/// Returns a vector of `GameRecord` structs on success.
fn load_game_records(file_path: &str, score_correction: bool, ply_min: u8, ply_max: u8) -> io::Result<Vec<GameRecord>> {
    let metadata = fs::metadata(file_path)?;
    let file_size = metadata.len() as usize;
    let entry_size = 24;

    if file_size % entry_size != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid data size. {file_path}"),
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

        debug_assert!(player & opponent == 0, "Player and opponent bitboards overlap: {player} {opponent}");
        debug_assert!((-64.0..=64.0).contains(&score), "Score out of range: {score}");
        debug_assert!((-64..=64).contains(&game_score), "Game score out of range: {game_score}");
        debug_assert!(ply <= 59, "Ply value out of range: {ply}");

        // Filter records by ply range
        if ply >= ply_min && ply <= ply_max {
            records.push(GameRecord {
                player,
                opponent,
                score,
                ply,
            });
        }
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
/// * `data` - Vector to write binary data to
/// * `score` - Evaluation score for this position
/// * `features` - Array of pattern feature indices
/// * `mobility` - Number of legal moves in this position
/// * `ply` - Move number in the game
fn write_feature_record(
    data: &mut Vec<u8>,
    score: f32,
    features: &[u16],
    mobility: u8,
    ply: u8,
) -> io::Result<()> {
    data.write_f32::<LittleEndian>(score)?;
    for &feature in features {
        data.write_u16::<LittleEndian>(feature)?;
    }
    data.write_u8(mobility)?;
    data.write_u8(ply)?;
    Ok(())
}
