//! Game record shuffling module.
//!
//! This module provides functionality to shuffle and redistribute game records
//! from multiple input files into a new set of output files. This is useful for
//! preparing training data by randomizing the order of game records and potentially
//! redistributing them across a different number of files.

use std::{
    fs::{metadata, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use glob::glob;
use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

/// Size of each game record in bytes
const RECORD_SIZE: usize = 24;

/// Buffer size for reading files (in number of records)
const READ_BUFFER_RECORDS: usize = 4096;

/// Random seed for reproducible shuffling
const SHUFFLE_SEED: u64 = 42;

/// Number of digits used in output file naming
const OUTPUT_FILE_DIGITS: usize = 5;

/// Represents a single game record as a fixed-size byte array
type Record = [u8; RECORD_SIZE];

/// Shuffles and redistributes game records from input files.
///
/// # Arguments
///
/// * `input_dir` - Directory containing input files to shuffle
/// * `output_dir` - Directory where shuffled files will be written
/// * `pattern` - Glob pattern to match input files (e.g., "*.bin")
/// * `files_per_chunk` - Number of input files to process in each chunk
/// * `num_output_files` - Number of output files to create (defaults to input file count)
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
) -> anyhow::Result<()> {
    let input_dir_path = Path::new(input_dir);
    let output_dir_path = Path::new(output_dir);

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir_path)?;

    let mut rng = SmallRng::seed_from_u64(SHUFFLE_SEED);
    let input_files = find_input_files(input_dir_path, pattern, &mut rng)?;
    if input_files.is_empty() {
        println!("No input files found – nothing to do.");
        return Ok(());
    }

    let num_output_files = num_output_files.unwrap_or(input_files.len()).max(1);

    println!("Input  folder : {input_dir:?}");
    println!("Output folder : {output_dir:?}");
    println!("Input files   : {}", input_files.len());
    println!("Output files  : {num_output_files}");
    println!("Files/chunk   : {files_per_chunk}");
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

    let mut records_per_output_file = vec![0u64; num_output_files];
    let mut total_records: u64 = 0;
    let mut total_bytes: u64 = 0;

    for (chunk_id, chunk) in input_files.chunks(files_per_chunk).enumerate() {
        let mut chunk_records: Vec<Record> = Vec::new();

        for path in chunk {
            read_records(path, &mut chunk_records)?;
        }

        chunk_records.shuffle(&mut rng);

        distribute_records(
            output_dir_path,
            &chunk_records,
            &mut records_per_output_file,
            chunk_id,
        )?;

        total_records += chunk_records.len() as u64;
        total_bytes += (chunk_records.len() * RECORD_SIZE) as u64;
        chunk_pb.set_message(format!(
            "total {total_records} recs / {}",
            HumanBytes(total_bytes)
        ));
        chunk_pb.inc(1);
    }

    chunk_pb.finish_with_message("done");
    mp.clear()?;

    println!("------------- Summary -------------");
    println!(
        "Total records : {}  ({})",
        total_records,
        HumanBytes(total_bytes)
    );
    for (i, record_count) in records_per_output_file.iter().enumerate() {
        println!("shuffled_{i:0OUTPUT_FILE_DIGITS$}.bin : {record_count} recs");
    }
    println!("-----------------------------------");
    Ok(())
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
fn read_records(path: &Path, out: &mut Vec<Record>) -> io::Result<()> {
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

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        let record_count = bytes_read / RECORD_SIZE;
        for chunk in buffer[..record_count * RECORD_SIZE].chunks_exact(RECORD_SIZE) {
            out.push(chunk.try_into().expect("slice length == RECORD_SIZE"));
        }
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

        let output_path = output_dir.join(format!("shuffled_{output_file_index:0OUTPUT_FILE_DIGITS$}.bin"));
        let output_file = OpenOptions::new().create(true).append(true).open(&output_path)?;
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
