//! Canonical-board score patching across binary record files.
//!
//! Reads a source binary record file, maps each canonical board
//! ([`Board::unique`]) to `(score, game_score, is_random)`, and overwrites
//! those three fields in matching target records while preserving original
//! bitboards and untouched metadata.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use glob::glob;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use reversi_core::board::Board;

use crate::record::{self, read_records_from_file};

#[derive(Clone, Copy)]
struct Patch {
    score: f32,
    game_score: Option<i8>,
    is_random: bool,
}

/// Overwrites `score`, `game_score`, and `is_random` on records in `target_dir`
/// whose canonical board matches a record in `source`.
pub(crate) fn execute(source: &Path, target_dir: &Path, pattern: &str) -> anyhow::Result<()> {
    println!("Loading source records from {}...", source.display());
    let source_records = read_records_from_file(source)?;
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

    let target_files = find_target_files(target_dir, pattern)?;
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
        )?
        .progress_chars("#>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let mut total_records: u64 = 0;
    let mut total_updated: u64 = 0;
    let mut files_modified: u64 = 0;

    for path in &target_files {
        let stats = record::rewrite_raw_records(path, |raw| {
            let Some(&patch) = score_map.get(&raw.board().unique()) else {
                return false;
            };
            raw.set_scores(patch.score, patch.game_score, patch.is_random);
            true
        })?;
        total_records += stats.scanned;
        total_updated += stats.updated;
        if stats.updated > 0 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{GameRecord, append_records};
    use reversi_core::{player_color::PlayerColor, square::Square};
    use std::fs;
    use std::io;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> io::Result<Self> {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "neural-reversi-datagen-overwrite-{name}-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path)?;
            Ok(Self { path })
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn execute_patches_matching_canonical_boards_and_preserves_other_fields() -> anyhow::Result<()>
    {
        use reversi_core::game_state::GameState;

        let dir = TempDir::new("canonical")?;
        let source_path = dir.path.join("source.bin");
        let target_dir = dir.path.join("targets");
        fs::create_dir(&target_dir)?;
        let target_path = target_dir.join("part_00000.bin");

        let after_d3 = *GameState::from_moves(&[Square::D3]).unwrap().board();
        let after_f5 = *GameState::from_moves(&[Square::F5]).unwrap().board();
        assert_eq!(after_d3.unique(), after_f5.unique());
        assert_ne!(after_d3, after_f5);

        let unmatched = *GameState::from_moves(&[Square::D3, Square::C3])
            .unwrap()
            .board();

        let source_records = [
            GameRecord {
                game_id: 1,
                ply: 1,
                board: after_d3,
                score: 1.0,
                game_score: Some(2),
                side_to_move: PlayerColor::White,
                is_random: true,
                sq: Square::C3,
            },
            GameRecord {
                game_id: 2,
                ply: 1,
                board: after_d3.unique(),
                score: -3.5,
                game_score: Some(-4),
                side_to_move: PlayerColor::Black,
                is_random: false,
                sq: Square::A1,
            },
        ];
        append_records(&source_path, &source_records)?;

        let target_records = [
            GameRecord {
                game_id: 77,
                ply: 9,
                board: after_f5,
                score: 99.0,
                game_score: Some(64),
                side_to_move: PlayerColor::White,
                is_random: true,
                sq: Square::C5,
            },
            GameRecord {
                game_id: 88,
                ply: 10,
                board: unmatched,
                score: 0.25,
                game_score: Some(1),
                side_to_move: PlayerColor::Black,
                is_random: true,
                sq: Square::D6,
            },
        ];
        append_records(&target_path, &target_records)?;

        execute(&source_path, &target_dir, "*.bin")?;

        let actual = read_records_from_file(&target_path)?;
        assert_eq!(actual.len(), 2);

        assert_eq!(actual[0].board, after_f5);
        assert_eq!(actual[0].score, -3.5);
        assert_eq!(actual[0].game_score, Some(-4));
        assert!(!actual[0].is_random);
        assert_eq!(actual[0].ply, 9);
        assert_eq!(actual[0].side_to_move, PlayerColor::White);
        assert_eq!(actual[0].sq, Square::C5);
        assert_eq!(actual[0].game_id, 77);

        assert_eq!(actual[1].board, unmatched);
        assert_eq!(actual[1].score, 0.25);
        assert_eq!(actual[1].game_score, Some(1));
        assert!(actual[1].is_random);
        assert_eq!(actual[1].game_id, 88);
        Ok(())
    }
}
