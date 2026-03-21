//! Rescore module for correcting training data with exact endgame solving.

use reversi_core::disc::Disc;
use reversi_core::level::Level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::{self, SearchRunOptions};
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::record::{self, GameRecord, RECORD_SIZE};

/// Detects game boundaries by finding where ply decreases.
fn find_game_boundaries(records: &[GameRecord]) -> Vec<std::ops::Range<usize>> {
    if records.is_empty() {
        return Vec::new();
    }
    let mut boundaries = Vec::new();
    let mut start = 0;
    for i in 1..records.len() {
        if records[i].ply <= records[i - 1].ply {
            boundaries.push(start..i);
            start = i;
        }
    }
    boundaries.push(start..records.len());
    boundaries
}

/// Reconstructs the actual side-to-move for each record in a game by replaying moves.
///
/// The binary format does not store side-to-move, and `ply % 2` is unreliable when
/// passes occur. This function replays each move via `Board::make_move` and detects
/// passes by checking if the resulting board has legal moves for the opponent.
fn reconstruct_side_to_move(
    records: &[GameRecord],
    game_range: &std::ops::Range<usize>,
) -> Vec<Disc> {
    let len = game_range.end - game_range.start;
    let mut sides = Vec::with_capacity(len);
    let mut side = Disc::Black;
    for (j, i) in game_range.clone().enumerate() {
        sides.push(side);
        if j + 1 < len {
            let after_move = records[i].board.make_move(records[i].sq);
            if after_move.has_legal_moves() {
                side = side.opposite();
            }
            // If opponent has no legal moves, same side continues (pass)
        }
    }
    sides
}

/// Returns the number of complete games already written to the output file.
///
/// If the output contains a partial game (e.g. due to a crash mid-write),
/// truncates it to the last complete game boundary.
fn find_resume_point(
    output_path: &Path,
    game_ranges: &[std::ops::Range<usize>],
) -> io::Result<usize> {
    let existing = record::count_records_in_file(output_path)? as usize;
    if existing == 0 {
        return Ok(0);
    }

    let mut accumulated = 0usize;
    for (idx, range) in game_ranges.iter().enumerate() {
        accumulated += range.len();
        if accumulated == existing {
            return Ok(idx + 1);
        }
        if accumulated > existing {
            // Partial game detected — truncate to last complete boundary
            let safe = accumulated - range.len();
            let safe_bytes = (safe as u64) * RECORD_SIZE;
            let file = fs::OpenOptions::new().write(true).open(output_path)?;
            file.set_len(safe_bytes)?;
            eprintln!(
                "Warning: truncated {} from {} to {} records (partial game removed)",
                output_path.display(),
                existing,
                safe,
            );
            return Ok(idx);
        }
    }

    // existing > total input records — should not happen, but treat as fully done
    Ok(game_ranges.len())
}

/// Executes rescoring on input file(s).
///
/// Reads binary training data, performs exact endgame solving on positions
/// with empty squares <= `empties`, and writes corrected data to `output_dir`.
/// Also propagates the corrected game_score to earlier positions in each game.
pub fn execute(
    input: &str,
    output_dir: &str,
    empties: u32,
    hash_size: usize,
    verbose: bool,
) -> io::Result<()> {
    fs::create_dir_all(output_dir)?;

    let input_path = Path::new(input);
    let files = enumerate_bin_files(input_path)?;

    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No .bin files found in input",
        ));
    }

    let options = SearchOptions::new(hash_size);
    let mut search = search::Search::new(&options);
    let level = Level {
        mid_depth: 60,
        end_depth: [empties, empties, empties - 2, empties - 4],
    };
    let run_options = SearchRunOptions::with_level(level, Selectivity::None);

    let total_start = Instant::now();
    let mut total_records = 0u64;
    let mut total_corrected = 0u64;
    let mut total_propagated = 0u64;

    for file_path in &files {
        let file_start = Instant::now();

        let mut records = match record::read_records_from_file(file_path) {
            Ok(r) => r,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                eprintln!("Warning: skipping {}: {}", file_path.display(), e);
                continue;
            }
            Err(e) => return Err(e),
        };
        let record_count = records.len();
        let mut corrected_count = 0usize;
        let mut propagated_count = 0usize;
        let file_name_display = file_path.file_name().unwrap_or_default().to_string_lossy();

        let game_ranges = find_game_boundaries(&records);

        // Determine resume point from existing output
        let file_name = file_path
            .file_name()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid file name"))?;
        let output_path = Path::new(output_dir).join(file_name);
        let skip_games = find_resume_point(&output_path, &game_ranges)?;

        if skip_games == game_ranges.len() {
            println!("{}: already completed, skipping", file_path.display());
            total_records += record_count as u64;
            continue;
        }
        if skip_games > 0 {
            println!(
                "{}: resuming from game {}/{}",
                file_path.display(),
                skip_games + 1,
                game_ranges.len()
            );
        }

        let out_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)?;
        let mut writer = BufWriter::new(out_file);

        for game_range in &game_ranges[skip_games..] {
            let sides = reconstruct_side_to_move(&records, game_range);

            // Phase 1: exact solve for empties <= threshold
            let mut propagation_ref: Option<(i8, Disc)> = None;

            for (j, i) in game_range.clone().enumerate() {
                let n_empties = records[i].board.get_empty_count();
                if records[i].board.is_game_over() || n_empties > empties {
                    continue;
                }

                let rec_start = Instant::now();
                let old_score = records[i].score;
                let (board, sign) = if records[i].board.has_legal_moves() {
                    (records[i].board, 1.0)
                } else {
                    (records[i].board.switch_players(), -1.0)
                };
                let result = search.run(&board, &run_options);
                let exact_score = result.score * sign;

                records[i].board = board;
                records[i].score = exact_score;
                records[i].game_score = exact_score.round() as i8;
                corrected_count += 1;

                // Record the first solved position as the reference for propagation
                if propagation_ref.is_none() {
                    propagation_ref = Some((records[i].game_score, sides[j]));
                }

                if verbose {
                    let new_score = records[i].game_score;
                    let diff = if old_score != new_score as f32 {
                        format!(" ({:+} -> {:+})", old_score, new_score)
                    } else {
                        String::new()
                    };
                    println!(
                        "{file_name_display}: [{}/{}] empties={}, side={}, score={:+}, {:.2}s{}",
                        i + 1,
                        record_count,
                        n_empties,
                        sides[j].to_char(),
                        new_score,
                        rec_start.elapsed().as_secs_f64(),
                        diff,
                    );
                }
            }

            // Phase 2: propagate corrected game_score to pre-threshold positions
            // Skip propagation if the game doesn't start at ply 0, since
            // reconstruct_side_to_move assumes the first record is Black's turn.
            if records[game_range.start].ply == 0
                && let Some((ref_score, ref_side)) = propagation_ref
            {
                for (j, i) in game_range.clone().enumerate() {
                    let n_empties = records[i].board.get_empty_count();
                    if n_empties <= empties || records[i].is_random {
                        continue;
                    }

                    let old_score = records[i].game_score;
                    let new_score = if sides[j] == ref_side {
                        ref_score
                    } else {
                        -ref_score
                    };

                    if old_score != new_score {
                        records[i].game_score = new_score;
                        propagated_count += 1;

                        if verbose {
                            println!(
                                "{file_name_display}: [{}/{}] empties={}, side={}, propagated ({:+2}, {:+} -> {:+})",
                                i + 1,
                                record_count,
                                n_empties,
                                sides[j].to_char(),
                                records[i].score,
                                old_score,
                                new_score,
                            );
                        }
                    }
                }
            }

            // Append this game's records and flush for crash safety
            record::write_records(&mut writer, &records[game_range.clone()])?;
            writer.flush()?;

            search.init();
        }

        let elapsed = file_start.elapsed();
        println!(
            "{}: {} records, {} corrected, {} propagated, {:.2}s",
            file_path.display(),
            record_count,
            corrected_count,
            propagated_count,
            elapsed.as_secs_f64()
        );

        total_records += record_count as u64;
        total_corrected += corrected_count as u64;
        total_propagated += propagated_count as u64;
    }

    let total_elapsed = total_start.elapsed();
    println!(
        "\nDone: {} files, {} records, {} corrected, {} propagated, {:.2}s",
        files.len(),
        total_records,
        total_corrected,
        total_propagated,
        total_elapsed.as_secs_f64()
    );

    Ok(())
}

/// Enumerates .bin files from a path (file or directory).
fn enumerate_bin_files(path: &Path) -> io::Result<Vec<std::path::PathBuf>> {
    if path.is_file() {
        if path.extension().is_none_or(|ext| ext != "bin") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Input file is not a .bin file: {}", path.display()),
            ));
        }
        return Ok(vec![path.to_path_buf()]);
    }

    if path.is_dir() {
        let mut files: Vec<_> = fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "bin"))
            .collect();
        files.sort();
        return Ok(files);
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("Input path does not exist: {}", path.display()),
    ))
}
