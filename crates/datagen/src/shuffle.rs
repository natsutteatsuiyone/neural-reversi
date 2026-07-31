//! Game record shuffling module.
//!
//! This module provides functionality to shuffle and redistribute game records
//! from multiple input files into a new set of output files. This is useful for
//! preparing training data by randomizing the order of game records and potentially
//! redistributing them across a different number of files.

use std::{
    fs::{File, OpenOptions, metadata},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use glob::glob;
use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

use crate::record::{
    self, GAME_SCORE_OFFSET, GAME_SCORE_UNAVAILABLE, IS_RANDOM_OFFSET, PLY_OFFSET, SCORE_OFFSET,
};

/// Size of each game record in bytes
const RECORD_SIZE: usize = record::RECORD_SIZE as usize;

pub(crate) struct FilterConfig {
    pub min_ply: u8,
    pub max_score_diff: Option<f32>,
    pub drop_random: bool,
    pub keep_above_ply: Option<u8>,
}

#[derive(Default)]
struct FilterStats {
    dropped_min_ply: u64,
    dropped_random: u64,
    dropped_score_diff: u64,
}

/// Buffer size for reading files (in number of records)
const READ_BUFFER_RECORDS: usize = 4096;

/// Random seed for reproducible shuffling
const SHUFFLE_SEED: u64 = 42;

/// Number of digits used in output file naming
const OUTPUT_FILE_DIGITS: usize = 5;

/// Represents a single game record as a fixed-size byte array
type Record = [u8; RECORD_SIZE];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExistingShuffleOutput {
    index: usize,
    record_count: u64,
}

/// Shuffles and redistributes game records from input files.
///
/// # Arguments
///
/// * `input_dir` - Directory containing input files to shuffle
/// * `output_dir` - Directory where shuffled files will be written
/// * `pattern` - Glob pattern to match input files (e.g., "*.bin")
/// * `files_per_chunk` - Number of input files to process in each chunk
/// * `num_output_files` - Number of output files to create (defaults to input file count)
/// * `append` - Whether to validate and append to existing shuffle output files
///
/// # Returns
///
/// Returns `Ok(())` on success, or an error if file operations fail.
pub fn execute(
    input_dir: &str,
    output_dir: &str,
    pattern: &str,
    files_per_chunk: usize,
    num_output_files: Option<usize>,
    append: bool,
    filter: FilterConfig,
) -> anyhow::Result<()> {
    let mut stats = FilterStats::default();

    let input_dir_path = Path::new(input_dir);
    let output_dir_path = Path::new(output_dir);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir_path)?;
    let existing_outputs =
        inspect_existing_shuffle_outputs(output_dir_path, append, num_output_files)?;

    let mut rng = SmallRng::seed_from_u64(SHUFFLE_SEED);
    let input_files = find_input_files(input_dir_path, pattern, &mut rng)?;
    if input_files.is_empty() {
        println!("No input files found – nothing to do.");
        return Ok(());
    }

    let mut records_per_output_file =
        resolve_output_layout(&existing_outputs, num_output_files, input_files.len())?;
    let num_output_files = records_per_output_file.len();

    println!("Input  folder : {input_dir:?}");
    println!("Output folder : {output_dir:?}");
    println!("Input files   : {}", input_files.len());
    println!("Output files  : {num_output_files}");
    println!("Append mode   : {}", if append { "yes" } else { "no" });
    println!("Files/chunk   : {files_per_chunk}");
    println!("Min ply       : {}", filter.min_ply);
    println!(
        "Drop random   : {}",
        if filter.drop_random { "yes" } else { "no" }
    );
    match filter.max_score_diff {
        Some(t) => println!("Max |Δscore|  : {t}"),
        None => println!("Max |Δscore|  : off"),
    }
    match filter.keep_above_ply {
        Some(p) => println!("Keep above ply: {p}"),
        None => println!("Keep above ply: off"),
    }
    println!("----------------------------------------");

    let mp = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    let chunk_pb = mp.add(ProgressBar::new(
        input_files.len().div_ceil(files_per_chunk) as u64,
    ));
    chunk_pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] chunks {pos}/{len} ETA:{eta_precise}",
        )?
        .progress_chars("#>-"),
    );
    chunk_pb.enable_steady_tick(Duration::from_millis(100));

    let mut added_records: u64 = 0;
    let mut added_bytes: u64 = 0;

    for (chunk_id, chunk) in input_files.chunks(files_per_chunk).enumerate() {
        let mut chunk_records: Vec<Record> = Vec::new();

        for path in chunk {
            read_records(path, &mut chunk_records, &filter, &mut stats)?;
        }

        chunk_records.shuffle(&mut rng);

        distribute_records(
            output_dir_path,
            &chunk_records,
            &mut records_per_output_file,
            chunk_id,
        )?;

        added_records += chunk_records.len() as u64;
        added_bytes += (chunk_records.len() * RECORD_SIZE) as u64;
        chunk_pb.set_message(format!(
            "added {added_records} recs / {}",
            HumanBytes(added_bytes)
        ));
        chunk_pb.inc(1);
    }

    chunk_pb.finish_with_message("done");
    mp.clear()?;

    println!("------------- Summary -------------");
    println!(
        "Added records : {}  ({})",
        added_records,
        HumanBytes(added_bytes)
    );
    let total_records = records_per_output_file.iter().sum::<u64>();
    let total_bytes = total_records * RECORD_SIZE as u64;
    println!(
        "Total records : {}  ({})",
        total_records,
        HumanBytes(total_bytes)
    );
    let total_dropped = stats.dropped_min_ply + stats.dropped_random + stats.dropped_score_diff;
    println!("Dropped       : {total_dropped} recs");
    println!("  min_ply     : {}", stats.dropped_min_ply);
    println!("  random      : {}", stats.dropped_random);
    println!("  score_diff  : {}", stats.dropped_score_diff);
    for (i, record_count) in records_per_output_file.iter().enumerate() {
        println!("shuffled_{i:0OUTPUT_FILE_DIGITS$}.bin : {record_count} recs");
    }
    println!("-----------------------------------");
    Ok(())
}

fn inspect_existing_shuffle_outputs(
    output_dir: &Path,
    append: bool,
    explicit_num_output_files: Option<usize>,
) -> anyhow::Result<Vec<ExistingShuffleOutput>> {
    let mut candidates = Vec::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if entry.file_type()?.is_file()
            && file_name.starts_with("shuffled_")
            && file_name.ends_with(".bin")
        {
            candidates.push((entry.path(), file_name.into_owned()));
        }
    }

    if !append {
        anyhow::ensure!(
            candidates.is_empty(),
            "output dir '{}' already contains {} shuffled_*.bin file(s); a re-run would append to them and corrupt the dataset. Remove them, use a fresh --output-dir, or pass --append to validate and append safely.",
            output_dir.display(),
            candidates.len()
        );
        return Ok(Vec::new());
    }

    let mut existing_outputs = Vec::with_capacity(candidates.len());
    for (path, file_name) in candidates {
        let index = parse_shuffle_output_index(&file_name)?;
        if let Some(num_output_files) = explicit_num_output_files {
            anyhow::ensure!(
                index < num_output_files,
                "existing shuffle output '{}' has index {index}, which is outside the {num_output_files}-file output layout",
                path.display()
            );
        }

        let file_size = path.metadata()?.len();
        anyhow::ensure!(
            file_size % RECORD_SIZE as u64 == 0,
            "existing shuffle output '{}' has size {file_size}, which is not a multiple of record size {RECORD_SIZE}; refusing to append",
            path.display()
        );
        existing_outputs.push(ExistingShuffleOutput {
            index,
            record_count: file_size / RECORD_SIZE as u64,
        });
    }
    existing_outputs.sort_unstable_by_key(|output| output.index);
    Ok(existing_outputs)
}

fn parse_shuffle_output_index(file_name: &str) -> anyhow::Result<usize> {
    let index_text = file_name
        .strip_prefix("shuffled_")
        .and_then(|name| name.strip_suffix(".bin"))
        .unwrap_or_default();
    let index = index_text.parse::<usize>().map_err(|_| {
        anyhow::anyhow!(
            "'{file_name}' is not a canonical shuffle output name; expected shuffled_{{index:05}}.bin"
        )
    })?;
    let canonical_name = format!("shuffled_{index:0OUTPUT_FILE_DIGITS$}.bin");
    anyhow::ensure!(
        file_name == canonical_name,
        "'{file_name}' is not a canonical shuffle output name; expected '{canonical_name}'"
    );
    Ok(index)
}

fn resolve_output_layout(
    existing_outputs: &[ExistingShuffleOutput],
    explicit_num_output_files: Option<usize>,
    default_num_output_files: usize,
) -> anyhow::Result<Vec<u64>> {
    let num_output_files = if let Some(num_output_files) = explicit_num_output_files {
        anyhow::ensure!(
            num_output_files > 0,
            "--num-output-files must be at least 1"
        );
        num_output_files
    } else if let Some(max_index) = existing_outputs.iter().map(|output| output.index).max() {
        max_index
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("existing shuffle output index is too large"))?
    } else {
        default_num_output_files.max(1)
    };

    let mut records_per_output_file = vec![0u64; num_output_files];
    for output in existing_outputs {
        anyhow::ensure!(
            output.index < num_output_files,
            "existing shuffle output index {} is outside the {num_output_files}-file output layout",
            output.index
        );
        records_per_output_file[output.index] = output.record_count;
    }
    Ok(records_per_output_file)
}

/// Finds and shuffles input files matching the given pattern.
///
/// # Arguments
///
/// * `dir` - Directory to search for files
/// * `pattern` - Glob pattern to match files
/// * `rng` - Random number generator for shuffling file order
///
/// # Returns
///
/// Returns a vector of file paths in random order.
fn find_input_files(dir: &Path, pattern: &str, rng: &mut SmallRng) -> anyhow::Result<Vec<PathBuf>> {
    let full_pattern = dir.join(pattern).to_string_lossy().into_owned();
    let mut file_paths = Vec::new();

    let paths = glob(&full_pattern)
        .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", full_pattern, e))?;

    for entry in paths {
        match entry {
            Ok(path) if path.is_file() => file_paths.push(path),
            Ok(_) => {}
            Err(e) => eprintln!(
                "Warning: Failed to access path matched by glob ({}): {}",
                e.path().display(),
                e
            ),
        }
    }
    file_paths.shuffle(rng);
    Ok(file_paths)
}

/// Reads game records from a binary file.
///
/// # Arguments
///
/// * `path` - Path to the binary file to read
/// * `out` - Vector to append the read records to
///
/// # Returns
///
/// Returns `Ok(())` on success, or an I/O error if reading fails.
fn read_records(
    path: &Path,
    out: &mut Vec<Record>,
    filter: &FilterConfig,
    stats: &mut FilterStats,
) -> io::Result<()> {
    let md = metadata(path)?;
    if md.len() == 0 || md.len() % RECORD_SIZE as u64 != 0 {
        eprintln!(
            "Warning: {} skipped (size not multiple of {})",
            path.display(),
            RECORD_SIZE
        );
        return Ok(());
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; RECORD_SIZE * READ_BUFFER_RECORDS];

    // `md.len()` is guaranteed to be a multiple of RECORD_SIZE by the check above,
    // so we can read in exact record-batch-sized chunks without losing trailing bytes.
    let mut records_remaining = (md.len() / RECORD_SIZE as u64) as usize;
    while records_remaining > 0 {
        let batch = records_remaining.min(READ_BUFFER_RECORDS);
        let batch_bytes = batch * RECORD_SIZE;
        reader.read_exact(&mut buffer[..batch_bytes])?;
        for chunk in buffer[..batch_bytes].as_chunks::<RECORD_SIZE>().0 {
            let ply = chunk[PLY_OFFSET];
            if ply < filter.min_ply {
                stats.dropped_min_ply += 1;
                continue;
            }
            let dominated = filter
                .keep_above_ply
                .is_none_or(|threshold| ply < threshold);
            if dominated {
                if filter.drop_random && chunk[IS_RANDOM_OFFSET] != 0 {
                    stats.dropped_random += 1;
                    continue;
                }
                if let Some(threshold) = filter.max_score_diff {
                    let game_score = chunk[GAME_SCORE_OFFSET] as i8;
                    if game_score != GAME_SCORE_UNAVAILABLE {
                        let score_bytes: [u8; 4] = chunk[SCORE_OFFSET..SCORE_OFFSET + 4]
                            .try_into()
                            .expect("4-byte score slice");
                        let score = f32::from_le_bytes(score_bytes);
                        if (score - f32::from(game_score)).abs() > threshold {
                            stats.dropped_score_diff += 1;
                            continue;
                        }
                    }
                }
            }
            out.push(*chunk);
        }
        records_remaining -= batch;
    }
    Ok(())
}

fn distribute_records(
    output_dir: &Path,
    records: &[Record],
    records_per_file: &mut [u64],
    chunk_offset: usize,
) -> io::Result<()> {
    if records_per_file.is_empty() {
        return Ok(());
    }

    let num_output_files = records_per_file.len();
    let base_records_per_file = records.len() / num_output_files;
    let extra_records = records.len() % num_output_files;

    let mut record_index = 0;
    for file_index in 0..num_output_files {
        let output_file_index = (file_index + chunk_offset) % num_output_files;
        let records_to_write = base_records_per_file + usize::from(file_index < extra_records);
        if records_to_write == 0 {
            continue;
        }

        let output_path = output_dir.join(format!(
            "shuffled_{output_file_index:0OUTPUT_FILE_DIGITS$}.bin"
        ));
        let output_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)?;
        let mut writer = BufWriter::new(output_file);

        for record in &records[record_index..record_index + records_to_write] {
            writer.write_all(record)?;
        }
        writer.flush()?;

        records_per_file[output_file_index] += records_to_write as u64;
        record_index += records_to_write;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> io::Result<Self> {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "neural-reversi-datagen-shuffle-{name}-{}-{id}",
                std::process::id()
            ));
            std::fs::create_dir(&path)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn distribute_records_accounts_every_record() -> io::Result<()> {
        let output_dir = TempDir::new("accounting")?;
        let records: Vec<Record> = (0..10).map(|value| [value; RECORD_SIZE]).collect();
        let mut records_per_file = [0u64; 3];

        distribute_records(output_dir.path(), &records, &mut records_per_file, 0)?;

        assert_eq!(records_per_file.iter().sum::<u64>(), 10);
        let min_records = records_per_file
            .iter()
            .min()
            .copied()
            .expect("three output files");
        let max_records = records_per_file
            .iter()
            .max()
            .copied()
            .expect("three output files");
        assert!(max_records - min_records <= 1);

        let total_output_size =
            (0..records_per_file.len()).try_fold(0u64, |total, file_index| {
                let path = output_dir
                    .path()
                    .join(format!("shuffled_{file_index:0OUTPUT_FILE_DIGITS$}.bin"));
                Ok::<_, io::Error>(total + std::fs::metadata(path)?.len())
            })?;
        assert_eq!(total_output_size, 10 * RECORD_SIZE as u64);
        Ok(())
    }

    #[test]
    fn default_mode_rejects_prior_shuffle_output() -> anyhow::Result<()> {
        let canonical_dir = TempDir::new("guard-canonical")?;
        std::fs::write(canonical_dir.path().join("other.bin"), [])?;
        inspect_existing_shuffle_outputs(canonical_dir.path(), false, None)?;

        std::fs::write(canonical_dir.path().join("shuffled_00000.bin"), [])?;
        let error = inspect_existing_shuffle_outputs(canonical_dir.path(), false, None)
            .expect_err("canonical shuffle output must be rejected");
        assert!(
            error
                .to_string()
                .contains("already contains 1 shuffled_*.bin file(s)")
        );

        let malformed_dir = TempDir::new("guard-malformed")?;
        std::fs::write(malformed_dir.path().join("shuffled_bad.bin"), [])?;
        inspect_existing_shuffle_outputs(malformed_dir.path(), false, None)
            .expect_err("shuffle-looking output must be rejected");
        Ok(())
    }

    #[test]
    fn append_preserves_existing_bytes_and_seeds_counts() -> anyhow::Result<()> {
        let output_dir = TempDir::new("append")?;
        let output_path = output_dir.path().join("shuffled_00000.bin");
        let existing_record = [0x11; RECORD_SIZE];
        std::fs::write(&output_path, existing_record)?;
        let existing = inspect_existing_shuffle_outputs(output_dir.path(), true, None)?;
        let mut records_per_file = resolve_output_layout(&existing, None, 10)?;
        let new_records = [[0x22; RECORD_SIZE], [0x33; RECORD_SIZE]];

        distribute_records(output_dir.path(), &new_records, &mut records_per_file, 0)?;

        assert_eq!(records_per_file, [3]);
        let bytes = std::fs::read(output_path)?;
        assert_eq!(&bytes[..RECORD_SIZE], &existing_record);
        assert_eq!(&bytes[RECORD_SIZE..], new_records.as_flattened());
        Ok(())
    }

    #[test]
    fn append_infers_layout_from_highest_existing_index() -> anyhow::Result<()> {
        let output_dir = TempDir::new("infer-layout")?;
        std::fs::write(
            output_dir.path().join("shuffled_00000.bin"),
            [0u8; RECORD_SIZE],
        )?;
        std::fs::write(
            output_dir.path().join("shuffled_00002.bin"),
            [0u8; RECORD_SIZE * 2],
        )?;
        let existing = inspect_existing_shuffle_outputs(output_dir.path(), true, None)?;

        let records_per_file = resolve_output_layout(&existing, None, 99)?;

        assert_eq!(records_per_file, [1, 0, 2]);
        Ok(())
    }

    #[test]
    fn append_rejects_partial_record_shard() -> anyhow::Result<()> {
        let output_dir = TempDir::new("partial")?;
        std::fs::write(
            output_dir.path().join("shuffled_00000.bin"),
            [0u8; RECORD_SIZE + 1],
        )?;

        let error = inspect_existing_shuffle_outputs(output_dir.path(), true, None)
            .expect_err("partial shard must be rejected");

        assert!(error.to_string().contains("not a multiple of record size"));
        Ok(())
    }

    #[test]
    fn append_rejects_noncanonical_shard_names() -> anyhow::Result<()> {
        assert_eq!(parse_shuffle_output_index("shuffled_100000.bin")?, 100_000);

        for file_name in [
            "shuffled_1.bin",
            "shuffled_000000.bin",
            "shuffled_nonnumeric.bin",
        ] {
            let output_dir = TempDir::new("noncanonical")?;
            std::fs::write(output_dir.path().join(file_name), [])?;

            let error = inspect_existing_shuffle_outputs(output_dir.path(), true, None)
                .expect_err("noncanonical shard name must be rejected");

            assert!(
                error
                    .to_string()
                    .contains("not a canonical shuffle output name")
            );
        }
        Ok(())
    }

    #[test]
    fn append_rejects_index_outside_explicit_layout() -> anyhow::Result<()> {
        let output_dir = TempDir::new("outside-layout")?;
        std::fs::write(output_dir.path().join("shuffled_00002.bin"), [])?;

        let error = inspect_existing_shuffle_outputs(output_dir.path(), true, Some(2))
            .expect_err("index equal to output count must be rejected");

        assert!(
            error
                .to_string()
                .contains("outside the 2-file output layout")
        );
        Ok(())
    }
}
