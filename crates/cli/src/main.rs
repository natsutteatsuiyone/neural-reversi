mod game;
mod gtp;
mod solve;
mod tui;
mod ui;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use reversi_core::probcut::Selectivity;

#[derive(Parser, Debug, Clone)]
struct EngineParams {
    #[arg(long, default_value = "64")]
    hash_size: usize,

    #[arg(short, long, default_value = "21")]
    level: usize,

    #[arg(long, default_value = "0", value_parser = clap::value_parser!(u8).range(0..=5))]
    selectivity: u8,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long = "eval-file", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    eval_file: Option<PathBuf>,

    #[arg(long = "eval-sm-file", value_name = "FILE", value_hint = clap::ValueHint::FilePath)]
    eval_sm_file: Option<PathBuf>,
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

        /// Solve for exact score with perfect play (ignores level setting)
        #[arg(long)]
        exact: bool,

        #[command(flatten)]
        engine_params: EngineParams,
    },
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Some(SubCommands::Gtp { engine_params }) => {
            validate_weight_paths(
                engine_params.eval_file.as_deref(),
                engine_params.eval_sm_file.as_deref(),
            );
            let mut gtp_engine = gtp::GtpEngine::new(
                engine_params.hash_size,
                engine_params.level,
                Selectivity::from_u8(engine_params.selectivity),
                engine_params.threads,
                engine_params.eval_file.as_deref(),
                engine_params.eval_sm_file.as_deref(),
            )
            .unwrap_or_else(|err| {
                eprintln!("Failed to initialize engine: {err}");
                std::process::exit(1);
            });
            gtp_engine.run();
        }
        Some(SubCommands::Solve {
            file,
            exact,
            engine_params,
        }) => {
            if !file.exists() {
                eprintln!("File does not exist: {}", file.display());
                return;
            }
            validate_weight_paths(
                engine_params.eval_file.as_deref(),
                engine_params.eval_sm_file.as_deref(),
            );
            let path = Path::new(&file);
            if let Err(e) = solve::solve(
                path,
                engine_params.hash_size,
                engine_params.level,
                Selectivity::from_u8(engine_params.selectivity),
                engine_params.threads,
                engine_params.eval_file.as_deref(),
                engine_params.eval_sm_file.as_deref(),
                exact,
            ) {
                eprintln!("Error solving game: {e}");
            }
        }
        None => {
            validate_weight_paths(
                args.engine_params.eval_file.as_deref(),
                args.engine_params.eval_sm_file.as_deref(),
            );
            ui::ui_loop(
                args.engine_params.hash_size,
                args.engine_params.level,
                Selectivity::from_u8(args.engine_params.selectivity),
                args.engine_params.threads,
                args.engine_params.eval_file.as_deref(),
                args.engine_params.eval_sm_file.as_deref(),
            )
            .unwrap_or_else(|err| {
                eprintln!("Failed to initialize UI: {err}");
            });
        }
    }
}

fn validate_weight_paths(main: Option<&Path>, small: Option<&Path>) {
    for path in [main, small].into_iter().flatten() {
        if !path.exists() {
            eprintln!("Weight file does not exist: {}", path.display());
            std::process::exit(1);
        }
    }
}
