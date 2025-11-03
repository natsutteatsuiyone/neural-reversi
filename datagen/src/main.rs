mod feature;
mod opening;
mod probcut;
mod selfplay;
mod shuffle;

use clap::{Parser, Subcommand};
use reversi_core::types::{Depth, Selectivity};

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: SubCommands,
}

#[derive(Debug, Subcommand)]
enum SubCommands {
    Feature {
        #[arg(short, long)]
        input_dir: String,

        #[arg(short, long)]
        output_dir: String,

        #[arg(short, long, default_value = "1")]
        threads: usize,

        #[arg(long, default_value = "false")]
        score_correction: bool,

        #[arg(long, default_value = "0")]
        ply_min: u8,

        #[arg(long, default_value = "59")]
        ply_max: u8,

        #[arg(long, default_value = "false")]
        dedup: bool,
    },
    Selfplay {
        #[arg(long, default_value = "100000000")]
        games: u32,

        #[arg(long, default_value = "1000000")]
        records_per_file: u32,

        #[arg(long, default_value = "128")]
        hash_size: usize,

        #[arg(long, default_value = "12")]
        level: usize,

        #[arg(long, default_value = "1", value_parser = clap::value_parser!(Selectivity).range(1..=6))]
        selectivity: Selectivity,

        #[arg(long, default_value = "game")]
        prefix: String,

        #[arg(short, long)]
        output_dir: String,

        #[arg(long)]
        openings: Option<String>,

        #[arg(long, default_value = "false")]
        resume: bool,
    },
    Opening {
        #[arg(short, long)]
        depth: Depth,
    },
    Probcut {
        #[arg(short, long)]
        input: String,

        #[arg(short, long)]
        output: String,
    },
    Shuffle {
        #[arg(short, long)]
        input_dir: String,

        #[arg(short, long)]
        output_dir: String,

        #[arg(short = 'p', long, default_value = "*.bin")]
        pattern: String,

        #[arg(short = 'c', long, default_value_t = 10)]
        files_per_chunk: usize,

        #[arg(short = 'n', long)]
        num_output_files: Option<usize>,
    },
}

fn main() {
    let args = Cli::parse();
    match args.command {
        SubCommands::Feature {
            input_dir,
            output_dir,
            threads,
            score_correction,
            ply_min,
            ply_max,
            dedup,
        } => {
            feature::execute(
                &input_dir,
                &output_dir,
                threads,
                score_correction,
                ply_min,
                ply_max,
                dedup,
            )
            .unwrap();
        }
        SubCommands::Selfplay {
            games,
            records_per_file,
            hash_size,
            level,
            selectivity,
            prefix,
            output_dir,
            openings,
            resume,
        } => {
            if let Some(openings_path) = openings {
                selfplay::execute_with_openings(
                    &openings_path,
                    resume,
                    records_per_file,
                    hash_size,
                    level,
                    selectivity,
                    &prefix,
                    &output_dir,
                )
                .expect("Failed to execute selfplay with openings");
            } else {
                selfplay::execute(
                    games,
                    records_per_file,
                    hash_size,
                    level,
                    selectivity,
                    &prefix,
                    &output_dir,
                )
                .expect("Failed to execute selfplay");
            }
        }
        SubCommands::Opening { depth } => {
            opening::generate(depth);
        }
        SubCommands::Probcut { input, output } => {
            probcut::execute(&input, &output).expect("Failed to execute probcut");
        }
        SubCommands::Shuffle {
            input_dir,
            output_dir,
            pattern,
            files_per_chunk,
            num_output_files,
        } => {
            shuffle::execute(
                &input_dir,
                &output_dir,
                &pattern,
                files_per_chunk,
                num_output_files,
            )
            .unwrap();
        }
    }
}
