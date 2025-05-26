use std::{
    fs::{metadata, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use glob::glob;
use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};

const RECORD_SIZE: usize = 24;
type Record = [u8; RECORD_SIZE];

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

    let mut rng = SmallRng::seed_from_u64(42);
    let input_files = find_input_files(input_dir_path, pattern, &mut rng)?;
    if input_files.is_empty() {
        println!("No input files found – nothing to do.");
        return Ok(());
    }

    let num_outputs = num_output_files.unwrap_or(input_files.len()).max(1);

    println!("Input  folder : {:?}", input_dir);
    println!("Output folder : {:?}", output_dir);
    println!("Input files   : {}", input_files.len());
    println!("Output files  : {}", num_outputs);
    println!("Files/chunk   : {}", files_per_chunk);
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

    let mut records_per_output = vec![0u64; num_outputs];
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
            &mut records_per_output,
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
    for (i, n) in records_per_output.iter().enumerate() {
        println!("shuffled_{i:05}.bin : {n} recs");
    }
    println!("-----------------------------------");
    Ok(())
}

fn find_input_files(dir: &Path, pattern: &str, rng: &mut SmallRng) -> anyhow::Result<Vec<PathBuf>> {
    let full_pattern = dir.join(pattern).to_string_lossy().into_owned();
    let mut v = Vec::new();

    let paths = glob(&full_pattern)
        .map_err(|e| anyhow::anyhow!("Invalid glob pattern '{}': {}", full_pattern, e))?;

    for entry in paths {
        match entry {
            Ok(p) if p.is_file() => v.push(p),
            Ok(_) => {}
            Err(e) => eprintln!(
                "Warning: Failed to access path matched by glob ({}): {}",
                e.path().display(),
                e
            ),
        }
    }
    v.shuffle(rng);
    Ok(v)
}

fn read_records(path: &Path, out: &mut Vec<Record>) -> io::Result<()> {
    let md = metadata(path)?;
    if md.len() == 0 || md.len() % RECORD_SIZE as u64 != 0 {
        eprintln!(
            "Warning: {} skipped (size not multiple of 24)",
            path.display()
        );
        return Ok(());
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = vec![0u8; RECORD_SIZE * 4096]; // 96 KiB

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let rec_cnt = n / RECORD_SIZE;
        for chunk in buf[..rec_cnt * RECORD_SIZE].chunks_exact(RECORD_SIZE) {
            out.push(chunk.try_into().expect("slice length == RECORD_SIZE"));
        }
    }
    Ok(())
}

fn distribute_records(
    out_dir: &Path,
    records: &[Record],
    per_file_counter: &mut [u64],
    offset: usize,
) -> io::Result<()> {
    if per_file_counter.is_empty() {
        return Ok(());
    }

    let n_out = per_file_counter.len();
    let base = records.len() / n_out;
    let extra = records.len() % n_out;

    let mut idx = 0;
    for local_i in 0..n_out {
        let file_no = (local_i + offset) % n_out;
        let take = base + usize::from(local_i < extra);
        if take == 0 {
            continue;
        }

        let path = out_dir.join(format!("shuffled_{file_no:05}.bin"));
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let mut writer = BufWriter::new(file);

        for rec in &records[idx..idx + take] {
            writer.write_all(rec)?;
        }
        writer.flush()?;

        per_file_counter[file_no] += take as u64;
        idx += take;
    }
    Ok(())
}
