mod game;
mod gtp;
mod ui;

use clap::{Parser, Subcommand};
use reversi_core::types::Selectivity;

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Option<SubCommands>,

    #[arg(long, default_value = "1", value_parser = clap::value_parser!(Selectivity).range(1..=6))]
    selectivity: Selectivity,

    #[arg(long, default_value = "1")]
    hash_size: usize,
}

#[derive(Debug, Subcommand)]
enum SubCommands {
    Gtp {
        #[arg(long, default_value = "10")]
        level: usize,
    },
}

fn main() {
    reversi_core::init();

    let args = Cli::parse();
    match args.command {
        Some(SubCommands::Gtp { level}) => {
            let mut gtp_engine = gtp::GtpEngine::new(level, args.selectivity);
            gtp_engine.run();
        }
        None => {
            ui::ui_loop(args.selectivity);
        }
    }
}
