mod feature;
mod probcut;
mod selfplay;

use clap::{Parser, Subcommand};
use reversi_core::types::Selectivity;

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
    },
    Selfplay {
        #[arg(long, default_value = "100000000")]
        games: u32,

        #[arg(long, default_value = "10000")]
        games_per_file: u32,

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
        } => {
            feature::execute(&input_dir, &output_dir, threads).unwrap();
        }
        SubCommands::Selfplay {
            games,
            games_per_file,
            hash_size,
            level,
            selectivity,
            prefix,
            output_dir,
        } => {
            selfplay::execute(games, games_per_file, hash_size, level, selectivity, &prefix, &output_dir);
        }
        SubCommands::Probcut {
            input,
            output,
        } => {
            probcut::execute(&input, &output);
        }
    }
}
