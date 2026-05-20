//! Overwrite-scores module.
//!
//! Reads a source binary record file, builds a map from canonical board
//! ([`Board::unique`]) to `(score, game_score, is_random)`, then scans every
//! file matching a glob pattern in a target directory and overwrites those
//! three fields of each record whose canonical board appears in the source.

use glob::glob;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::types::Scoref;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::record::{
    GAME_SCORE_OFFSET, IS_RANDOM_OFFSET, RECORD_SIZE, SCORE_OFFSET, read_records_from_file,
};

const RECORD_SIZE_USIZE: usize = RECORD_SIZE as usize;

#[derive(Clone, Copy)]
struct Patch {
    score: Scoref,
    game_score: i8,
    is_random: bool,
}

/// Overwrites the `score`, `game_score`, and `is_random` fields of records in
/// `target_dir` whose canonical board matches a record in `source`.
pub fn execute(source: &str, target_dir: &str, pattern: &str) -> anyhow::Result<()> {
    let source_path = Path::new(source);
    let target_dir_path = Path::new(target_dir);

    println!("Loading source records from {}...", source_path.display());
    let source_records = read_records_from_file(source_path)?;
    println!("Loaded {} source records", source_records.len());

    let mut score_map: HashMap<Board, Patch> = HashMap::with_capacity(source_records.len());
    let mut duplicates = 0usize;
    for r in &source_records {
        let patch = Patch {
            score: r.score,
            game_score: r.game_score,
            is_random: r.is_random,
        };
        if score_map.insert(r.board.unique(), patch).is_some() {
            duplicates += 1;
        }
    }
    drop(source_records);
    if duplicates > 0 {
        eprintln!("Warning: {duplicates} duplicate canonical boards in source; later record wins.");
    }
    println!("{} unique canonical boards in source", score_map.len());

    let target_files = find_target_files(target_dir_path, pattern)?;
    if target_files.is_empty() {
        println!("No target files found - nothing to do.");
        return Ok(());
    }
    println!("Found {} target files", target_files.len());

    let pb = ProgressBar::with_draw_target(
        Some(target_files.len() as u64),
        ProgressDrawTarget::stderr_with_hz(10),
    );
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} files ETA:{eta_precise}",
        )
        .map_err(io::Error::other)?
        .progress_chars("#>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let mut total_records: u64 = 0;
    let mut total_updated: u64 = 0;
    let mut files_modified: u64 = 0;

    for path in &target_files {
        let (scanned, updated) = update_file(path, &score_map)?;
        total_records += scanned;
        total_updated += updated;
        if updated > 0 {
            files_modified += 1;
        }
        pb.inc(1);
    }
    pb.finish_and_clear();

    println!("------------- Summary -------------");
    println!("Files processed : {}", target_files.len());
    println!("Files modified  : {files_modified}");
    println!("Records scanned : {total_records}");
    println!("Records updated : {total_updated}");
    println!("-----------------------------------");
    Ok(())
}

fn find_target_files(dir: &Path, pattern: &str) -> anyhow::Result<Vec<PathBuf>> {
    let full_pattern = dir.join(pattern).to_string_lossy().into_owned();
    let paths = glob(&full_pattern)
        .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", full_pattern, e))?;
    let mut files = Vec::new();
    for entry in paths {
        match entry {
            Ok(path) if path.is_file() => files.push(path),
            Ok(_) => {}
            Err(e) => eprintln!(
                "Warning: Failed to access path matched by glob ({}): {}",
                e.path().display(),
                e
            ),
        }
    }
    files.sort();
    Ok(files)
}

fn update_file(path: &Path, score_map: &HashMap<Board, Patch>) -> io::Result<(u64, u64)> {
    let file_size = fs::metadata(path)?.len();
    if file_size == 0 || file_size % RECORD_SIZE != 0 {
        eprintln!(
            "Warning: {} skipped (size {} is not a multiple of RECORD_SIZE {})",
            path.display(),
            file_size,
            RECORD_SIZE
        );
        return Ok((0, 0));
    }

    let num_records = (file_size / RECORD_SIZE) as usize;
    let mut bytes = Vec::with_capacity(file_size as usize);
    File::open(path)?.read_to_end(&mut bytes)?;

    let mut updated = 0u64;
    for i in 0..num_records {
        let offset = i * RECORD_SIZE_USIZE;
        let chunk = &bytes[offset..offset + RECORD_SIZE_USIZE];
        let player_bits = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let opponent_bits = u64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let board = Board::from_bitboards(Bitboard::new(player_bits), Bitboard::new(opponent_bits));
        if let Some(&patch) = score_map.get(&board.unique()) {
            let score_bytes = patch.score.to_le_bytes();
            bytes[offset + SCORE_OFFSET..offset + SCORE_OFFSET + 4].copy_from_slice(&score_bytes);
            bytes[offset + GAME_SCORE_OFFSET] = patch.game_score as u8;
            bytes[offset + IS_RANDOM_OFFSET] = u8::from(patch.is_random);
            updated += 1;
        }
    }

    if updated > 0 {
        let temp_path = path.with_extension("bin.tmp");
        {
            let mut file = File::create(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp_path, path)?;
    }

    Ok((num_records as u64, updated))
}
