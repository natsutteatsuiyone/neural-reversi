use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, BufWriter},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

use byteorder::{LittleEndian, WriteBytesExt};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use zstd::stream::write::Encoder;

use reversi_core::board::Board;
use reversi_core::eval::pattern_feature;

const COMPRESSION_LEVEL: i32 = 7;
const BATCH_SIZE: usize = 1;

#[derive(Debug)]
struct GameRecord {
    player: u64,
    opponent: u64,
    score: f32,
    ply: u8,
}

pub fn execute(input_dir: &str, output_dir: &str, threads: usize) -> io::Result<()> {
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
    println!("Found {} bin files to process", total_files);
    entries.shuffle(&mut rand::rng());

    let entry_groups: Vec<_> = entries.chunks(BATCH_SIZE).collect();
    let processed_files = AtomicUsize::new(0);
    let start_time = std::time::Instant::now();

    entry_groups
        .par_iter()
        .enumerate()
        .for_each(|(group_idx, entry_group)| {
            let group_start_time = std::time::Instant::now();

            if let Err(e) = process_file_group(group_idx, entry_group, output_dir) {
                eprintln!("Failed to process group {}: {}", group_idx, e);
            }

            let completed_files =
                processed_files.fetch_add(entry_group.len(), Ordering::SeqCst) + entry_group.len();
            let elapsed = group_start_time.elapsed();

            println!(
                "Processed {}/{} files ({:.1}% complete) in {:.2?}",
                completed_files,
                total_files,
                (completed_files as f64 / total_files as f64) * 100.0,
                elapsed
            );
        });

    let total_time = start_time.elapsed();
    println!(
        "Feature generation completed in {:.2?} - Processed {} files",
        total_time, total_files
    );
    Ok(())
}

fn process_file_group(
    group_idx: usize,
    entry_paths: &[std::path::PathBuf],
    output_dir: &Path,
) -> io::Result<()> {
    let mut unique_positions = HashMap::new();
    for path in entry_paths {
        let input_path_str = path.to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "Path is not valid UTF-8")
        })?;

        let game_records = load_game_records(input_path_str)?;

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

        if write_feature_record(&mut encoder, score, &features, mobility, ply).is_err() {
            eprintln!("Failed to write feature",);
        }
    }

    encoder.finish()?;

    Ok(())
}

fn load_game_records(file_path: &str) -> io::Result<Vec<GameRecord>> {
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

    buffer.chunks_exact(entry_size).for_each(|chunk| {
        let player = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let opponent = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let mut score = f32::from_le_bytes(chunk[16..20].try_into().unwrap());
        let game_score = chunk[20] as i8;
        let ply = chunk[21];
        let is_random = chunk[22] == 1;
        // let mv = chunk[23];

        if ply <= 1 {
            score = 0.0;
        } else if !is_random {
            score = ((ply as f32 * game_score as f32) + (59.0 - ply as f32) * score) / 59.0;
        }

        records.push(GameRecord {
            player,
            opponent,
            score,
            ply,
        });
    });

    Ok(records)
}

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
