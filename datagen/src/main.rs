mod feature;
mod opening;
mod probcut;
mod selfplay;

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
    },
    Selfplay {
        #[arg(long, default_value = "100000000")]
        games: u32,

        #[arg(long, default_value = "1000000")]
        records_per_file: u32,

        #[arg(long, default_value = "128")]
        hash_size: i32,

        #[arg(long, default_value = "12")]
        level: usize,

        #[arg(long, default_value = "1", value_parser = clap::value_parser!(Selectivity).range(1..=6))]
        selectivity: Selectivity,

        #[arg(long, default_value = "game")]
        prefix: String,

        #[arg(short, long)]
        output_dir: String,
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
    }
}

fn main() {
    reversi_core::init();

    let args = Cli::parse();
    match args.command {
        SubCommands::Feature {
            input_dir,
            output_dir,
            threads,
            score_correction,
        } => {
            feature::execute(&input_dir, &output_dir, threads, score_correction).unwrap();
        }
        SubCommands::Selfplay {
            games,
            records_per_file,
            hash_size,
            level,
            selectivity,
            prefix,
            output_dir,
        } => {
            selfplay::execute(games, records_per_file, hash_size, level, selectivity, &prefix, &output_dir);
        }
        SubCommands::Opening { depth } => {
            opening::generate(depth);
        }
        SubCommands::Probcut {
            input,
            output,
        } => {
            probcut::execute(&input, &output);
        }
    }
}
