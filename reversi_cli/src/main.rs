mod game;
mod gtp;
mod solve;
mod ui;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use reversi_core::types::Selectivity;

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Option<SubCommands>,

    #[arg(long, default_value = "64")]
    hash_size: usize,

    #[arg(short, long, default_value = "21")]
    level: usize,

    #[arg(long, default_value = "1", value_parser = clap::value_parser!(Selectivity).range(1..=6))]
    selectivity: Selectivity,

    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Debug, Subcommand)]
enum SubCommands {
    Gtp {
        #[arg(long, default_value = "64")]
        hash_size: usize,

        #[arg(long, default_value = "21")]
        level: usize,

        #[arg(long, default_value = "1", value_parser = clap::value_parser!(Selectivity).range(1..=6))]
        selectivity: Selectivity,

        #[arg(long)]
        threads: Option<usize>,
    },
    Solve {
        #[arg()]
        file: PathBuf,

        #[arg(long, default_value = "64")]
        hash_size: usize,

        #[arg(short, long, default_value = "21")]
        level: usize,

        #[arg(long, default_value = "1", value_parser = clap::value_parser!(Selectivity).range(1..=6))]
        selectivity: Selectivity,

        #[arg(long)]
        threads: Option<usize>,
    },
}

fn main() {
    reversi_core::init();

    let args = Cli::parse();
    match args.command {
        Some(SubCommands::Gtp { hash_size, level, selectivity, threads }) => {
            let mut gtp_engine = gtp::GtpEngine::new(hash_size, level, selectivity, threads);
            gtp_engine.run();
        }
        Some(SubCommands::Solve {
            file,
            hash_size,
            level,
            selectivity,
            threads,
        }) => {
            if !file.exists() {
                eprintln!("File does not exist: {}", file.display());
                return;
            }
            let path = Path::new(&file);
            if let Err(e) = solve::solve(path, hash_size, level, selectivity, threads) {
                eprintln!("Error solving game: {e}");
            }
        }
        None => {
            ui::ui_loop(args.hash_size, args.level, args.selectivity, args.threads);
        }
    }
}
