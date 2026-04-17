//! Score-openings module.
//!
//! Enumerates every unique board position reachable within a given number of plies
//! from the initial position, scores each one with the search algorithm, and writes
//! the results in the shared binary record format. Supports resume by skipping
//! positions already present in an existing output file.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::level::Level;
use reversi_core::move_list::MoveList;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::{self, SearchRunOptions};
use reversi_core::square::Square;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;

use crate::record::{GameRecord, read_records_from_file, write_records_to_file};

/// Enumerates all unique positions reachable within `depth` plies and scores each one.
///
/// Duplicate positions (reached via different move orders) are eliminated in memory.
/// If the output file already exists, previously scored positions are loaded and skipped.
pub fn execute(
    depth: u8,
    hash_size: usize,
    level: Level,
    selectivity: Selectivity,
    output: &str,
) -> io::Result<()> {
    if let Some(parent) = Path::new(output).parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Enumerating positions up to depth {depth}...");
    let mut positions: Vec<(Board, Disc)> = Vec::new();
    let mut visited = HashSet::new();
    enumerate_positions(
        &Board::new(),
        Disc::Black,
        depth,
        &mut visited,
        &mut positions,
    );
    drop(visited);
    println!("Found {} unique positions", positions.len());

    // Load previously scored positions for resume
    let output_path = Path::new(output);
    let mut scored: HashSet<Board> = if output_path.exists() {
        let records = read_records_from_file(output_path)?;
        println!("Loaded {} existing records, resuming...", records.len());
        records.into_iter().map(|r| r.board).collect()
    } else {
        HashSet::new()
    };

    let options = SearchOptions::new(hash_size);
    let mut search = search::Search::new(&options);
    let run_options = SearchRunOptions::with_level(level, selectivity);

    let total = positions.len();
    let already_scored = scored.len();
    let to_score = total.saturating_sub(already_scored);
    let mut new_count = 0usize;
    let mut batch: Vec<GameRecord> = Vec::with_capacity(1000);

    let pb = ProgressBar::with_draw_target(
        Some(to_score as u64),
        ProgressDrawTarget::stderr_with_hz(10),
    );
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({per_sec}) ETA:{eta_precise}",
        )
        .map_err(io::Error::other)?
        .progress_chars("#>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    for &(board, side_to_move) in &positions {
        if !scored.insert(board) {
            continue;
        }

        let result = search.run(&board, &run_options);
        let ply = 60 - board.get_empty_count() as u8;

        batch.push(GameRecord {
            game_id: 0,
            ply,
            board,
            score: result.score,
            game_score: result.score.round() as i8,
            side_to_move,
            is_random: false,
            sq: result.best_move.unwrap_or(Square::A1),
        });
        new_count += 1;
        pb.inc(1);

        if batch.len() >= 1000 {
            write_records_to_file(output_path, &batch)?;
            batch.clear();
        }
    }

    if !batch.is_empty() {
        write_records_to_file(output_path, &batch)?;
    }

    pb.finish_and_clear();

    println!(
        "Done. {} positions total, {} newly scored.",
        total, new_count
    );
    Ok(())
}

/// Recursively enumerates all unique board positions reachable within `depth` plies.
///
/// Positions are canonicalized via [`Board::unique`] so that symmetric variants
/// are treated as duplicates, reducing the total count by up to 8x.
fn enumerate_positions(
    board: &Board,
    side_to_move: Disc,
    depth: u8,
    visited: &mut HashSet<Board>,
    positions: &mut Vec<(Board, Disc)>,
) {
    let canonical = board.unique();
    if !visited.insert(canonical) {
        return;
    }

    let move_list = MoveList::new(board);
    let has_moves = move_list.count() > 0;
    if has_moves {
        positions.push((canonical, side_to_move));
    }

    if depth == 0 {
        return;
    }

    if has_moves {
        for m in move_list.iter() {
            let next = board.make_move_with_flipped(m.flipped, m.sq);
            enumerate_positions(
                &next,
                side_to_move.opposite(),
                depth - 1,
                visited,
                positions,
            );
        }
    } else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            enumerate_positions(&next, side_to_move.opposite(), depth, visited, positions);
        }
    }
}
