//! Game record shuffling and filtering across binary shard files.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context;
use glob::glob;
use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::record::{self, RECORD_SIZE, RawRecord};

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

/// Random seed for reproducible shuffling.
const SHUFFLE_SEED: u64 = 42;

/// Number of digits used in output file naming.
const OUTPUT_FILE_DIGITS: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExistingShuffleOutput {
    index: usize,
    record_count: u64,
}

pub(crate) fn execute(
    input_dir: &Path,
    output_dir: &Path,
    pattern: &str,
    files_per_chunk: usize,
    num_output_files: Option<usize>,
    append: bool,
    filter: FilterConfig,
) -> anyhow::Result<()> {
    let mut stats = FilterStats::default();

    fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "failed to create output directory '{}'",
            output_dir.display()
        )
    })?;
    let existing_outputs = inspect_existing_shuffle_outputs(output_dir, append, num_output_files)?;

    let mut rng = SmallRng::seed_from_u64(SHUFFLE_SEED);
    let input_files = find_input_files(input_dir, pattern, &mut rng)?;
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
        let mut chunk_records: Vec<RawRecord> = Vec::new();

        for path in chunk {
            read_records(path, &mut chunk_records, &filter, &mut stats)?;
        }

        chunk_records.shuffle(&mut rng);

        distribute_records(
            output_dir,
            &chunk_records,
            &mut records_per_output_file,
            chunk_id,
        )?;

        added_records += chunk_records.len() as u64;
        added_bytes += (chunk_records.len() * RECORD_SIZE) as u64;
        chunk_pb.inc(1);
    }

    chunk_pb.finish();
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
    for entry in fs::read_dir(output_dir)
        .with_context(|| format!("failed to read output directory '{}'", output_dir.display()))?
    {
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

        let record_count = record::count_aligned_records(&path)?;
        existing_outputs.push(ExistingShuffleOutput {
            index,
            record_count,
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

fn read_records(
    path: &Path,
    out: &mut Vec<RawRecord>,
    filter: &FilterConfig,
    stats: &mut FilterStats,
) -> anyhow::Result<()> {
    record::visit_aligned_raw_records_or_warn(path, |raw| {
        let ply = raw.ply();
        if ply < filter.min_ply {
            stats.dropped_min_ply += 1;
            return;
        }
        if filter
            .keep_above_ply
            .is_none_or(|threshold| ply < threshold)
        {
            if filter.drop_random && raw.is_random() {
                stats.dropped_random += 1;
                return;
            }
            if let (Some(threshold), Some(game_score)) = (filter.max_score_diff, raw.game_score())
                && (raw.score() - f32::from(game_score)).abs() > threshold
            {
                stats.dropped_score_diff += 1;
                return;
            }
        }
        out.push(raw);
    })
}

fn distribute_records(
    output_dir: &Path,
    records: &[RawRecord],
    records_per_file: &mut [u64],
    chunk_offset: usize,
) -> anyhow::Result<()> {
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
        record::append_raw_records(
            &output_path,
            &records[record_index..record_index + records_to_write],
        )?;

        records_per_file[output_file_index] += records_to_write as u64;
        record_index += records_to_write;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
                "neural-reversi-datagen-shuffle-{name}-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path)?;
            Ok(Self { path })
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn distribute_records_accounts_every_record() -> anyhow::Result<()> {
        let output_dir = TempDir::new("accounting")?;
        let records: Vec<RawRecord> = (0..10)
            .map(|value| RawRecord::from_bytes([value; RECORD_SIZE]))
            .collect();
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
                Ok::<_, io::Error>(total + fs::metadata(path)?.len())
            })?;
        assert_eq!(total_output_size, 10 * RECORD_SIZE as u64);
        let mut output_records = Vec::<RawRecord>::new();
        for (file_index, &count) in records_per_file.iter().enumerate() {
            let bytes = fs::read(
                output_dir
                    .path()
                    .join(format!("shuffled_{file_index:0OUTPUT_FILE_DIGITS$}.bin")),
            )?;
            assert_eq!(bytes.len(), count as usize * RECORD_SIZE);
            output_records.extend(
                bytes
                    .as_chunks::<RECORD_SIZE>()
                    .0
                    .iter()
                    .copied()
                    .map(RawRecord::from_bytes),
            );
        }
        output_records.sort_unstable();
        let mut expected = records;
        expected.sort_unstable();
        assert_eq!(output_records, expected);
        Ok(())
    }

    #[test]
    fn default_mode_rejects_prior_shuffle_output() -> anyhow::Result<()> {
        let canonical_dir = TempDir::new("guard-canonical")?;
        fs::write(canonical_dir.path().join("other.bin"), [])?;
        inspect_existing_shuffle_outputs(canonical_dir.path(), false, None)?;

        fs::write(canonical_dir.path().join("shuffled_00000.bin"), [])?;
        let error = inspect_existing_shuffle_outputs(canonical_dir.path(), false, None)
            .expect_err("canonical shuffle output must be rejected");
        assert!(
            error
                .to_string()
                .contains("already contains 1 shuffled_*.bin file(s)")
        );

        let malformed_dir = TempDir::new("guard-malformed")?;
        fs::write(malformed_dir.path().join("shuffled_bad.bin"), [])?;
        inspect_existing_shuffle_outputs(malformed_dir.path(), false, None)
            .expect_err("shuffle-looking output must be rejected");
        Ok(())
    }

    #[test]
    fn append_preserves_existing_bytes_and_seeds_counts() -> anyhow::Result<()> {
        let output_dir = TempDir::new("append")?;
        let output_path = output_dir.path().join("shuffled_00000.bin");
        let existing_record = [0x11; RECORD_SIZE];
        fs::write(&output_path, existing_record)?;
        let existing = inspect_existing_shuffle_outputs(output_dir.path(), true, None)?;
        let mut records_per_file = resolve_output_layout(&existing, None, 10)?;
        let new_records = [
            RawRecord::from_bytes([0x22; RECORD_SIZE]),
            RawRecord::from_bytes([0x33; RECORD_SIZE]),
        ];

        distribute_records(output_dir.path(), &new_records, &mut records_per_file, 0)?;

        assert_eq!(records_per_file, [3]);
        let bytes = fs::read(output_path)?;
        assert_eq!(&bytes[..RECORD_SIZE], &existing_record);
        assert_eq!(
            &bytes[RECORD_SIZE..RECORD_SIZE * 2],
            new_records[0].as_bytes()
        );
        assert_eq!(&bytes[RECORD_SIZE * 2..], new_records[1].as_bytes());
        Ok(())
    }

    #[test]
    fn append_infers_layout_from_highest_existing_index() -> anyhow::Result<()> {
        let output_dir = TempDir::new("infer-layout")?;
        fs::write(
            output_dir.path().join("shuffled_00000.bin"),
            [0u8; RECORD_SIZE],
        )?;
        fs::write(
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
        fs::write(
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
            fs::write(output_dir.path().join(file_name), [])?;

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
        fs::write(output_dir.path().join("shuffled_00002.bin"), [])?;

        let error = inspect_existing_shuffle_outputs(output_dir.path(), true, Some(2))
            .expect_err("index equal to output count must be rejected");

        assert!(
            error
                .to_string()
                .contains("outside the 2-file output layout")
        );
        Ok(())
    }

    #[test]
    fn execute_filters_records_by_ply_random_and_score_diff() -> anyhow::Result<()> {
        use reversi_core::{
            bitboard::Bitboard, board::Board, player_color::PlayerColor, square::Square,
        };

        let input_dir = TempDir::new("filter-in")?;
        let output_dir = TempDir::new("filter-out")?;
        let make = |game_id: u16, ply: u8, score: f32, game_score: Option<i8>, is_random: bool| {
            record::GameRecord {
                game_id,
                ply,
                board: Board::from_bitboards(Bitboard::new(1 << game_id), Bitboard::new(1 << 20)),
                score,
                game_score,
                side_to_move: PlayerColor::Black,
                is_random,
                sq: Square::D3,
            }
        };

        let input_records = [
            make(1, 5, 0.0, Some(0), false),
            make(2, 12, 0.0, Some(0), true),
            make(3, 12, 10.0, Some(0), false),
            make(4, 12, 2.5, Some(0), false),
            make(5, 12, 50.0, None, false),
            make(6, 30, 50.0, Some(0), true),
        ];
        record::append_records(&input_dir.path().join("shard.bin"), &input_records)?;

        execute(
            input_dir.path(),
            output_dir.path(),
            "*.bin",
            1,
            Some(1),
            false,
            FilterConfig {
                min_ply: 8,
                max_score_diff: Some(4.0),
                drop_random: true,
                keep_above_ply: Some(25),
            },
        )?;

        let mut actual =
            record::read_records_from_file(&output_dir.path().join("shuffled_00000.bin"))?;
        actual.sort_by_key(|r| r.game_id);
        let kept_ids: Vec<u16> = actual.iter().map(|r| r.game_id).collect();
        assert_eq!(kept_ids, vec![4, 5, 6]);
        Ok(())
    }
}
