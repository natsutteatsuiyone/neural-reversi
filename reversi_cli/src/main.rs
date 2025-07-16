mod game;
mod gtp;
mod solve;
mod ui;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use reversi_core::types::Selectivity;

#[derive(Parser, Debug, Clone)]
struct EngineParams {
    #[arg(long, default_value = "64")]
    hash_size: usize,

    #[arg(short, long, default_value = "21")]
    level: usize,

    #[arg(long, default_value = "1", value_parser = clap::value_parser!(Selectivity).range(1..=6))]
    selectivity: Selectivity,

    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Option<SubCommands>,

    #[command(flatten)]
    engine_params: EngineParams,
}

#[derive(Debug, Subcommand)]
enum SubCommands {
    Gtp {
        #[command(flatten)]
        engine_params: EngineParams,
    },
    Solve {
        #[arg()]
        file: PathBuf,

        #[command(flatten)]
        engine_params: EngineParams,
    },
}

fn main() {
    reversi_core::init();

    let args = Cli::parse();
    match args.command {
        Some(SubCommands::Gtp { engine_params }) => {
            let mut gtp_engine = gtp::GtpEngine::new(
                engine_params.hash_size,
                engine_params.level,
                engine_params.selectivity,
                engine_params.threads,
            );
            gtp_engine.run();
        }
        Some(SubCommands::Solve { file, engine_params }) => {
            if !file.exists() {
                eprintln!("File does not exist: {}", file.display());
                return;
            }
            let path = Path::new(&file);
            if let Err(e) = solve::solve(
                path,
                engine_params.hash_size,
                engine_params.level,
                engine_params.selectivity,
                engine_params.threads,
            ) {
                eprintln!("Error solving game: {e}");
            }
        }
        None => {
            ui::ui_loop(
                args.engine_params.hash_size,
                args.engine_params.level,
                args.engine_params.selectivity,
                args.engine_params.threads,
            );
        }
    }
}
