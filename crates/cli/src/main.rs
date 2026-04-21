mod game;
mod gtp;
mod solve;
mod tui;

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use reversi_core::level::MAX_LEVEL;
use reversi_core::probcut::Selectivity;

fn parse_usize_range<const LO: usize, const HI: usize>(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|e| format!("{e}"))?;
    if (LO..=HI).contains(&v) {
        Ok(v)
    } else {
        Err(format!("value must be between {LO} and {HI}"))
    }
}

#[derive(Parser, Debug, Clone)]
struct EngineParams {
    #[arg(
        long,
        default_value = "512",
        value_parser = parse_usize_range::<1, 16384>,
        help = "Transposition table size in MB"
    )]
    hash_size: usize,

    #[arg(
        short,
        long,
        default_value = "21",
        value_parser = parse_usize_range::<1, MAX_LEVEL>,
        help = "Search level (affects midgame search depth)"
    )]
    level: usize,

    #[arg(
        long,
        default_value = "0",
        value_parser = clap::value_parser!(u8).range(0..=5),
        help = "Search selectivity for ProbCut pruning (0 = most selective, 5 = least selective)"
    )]
    selectivity: u8,

    #[arg(long, help = "Number of search threads [default: CPU count]")]
    threads: Option<usize>,

    #[arg(
        long = "eval-file",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath,
        help = "Path to the main network weight file"
    )]
    eval_file: Option<PathBuf>,

    #[arg(
        long = "eval-sm-file",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath,
        help = "Path to the small network weight file"
    )]
    eval_sm_file: Option<PathBuf>,
}

#[derive(Parser, Debug)]
#[command(
    about = "Neural Reversi — a high-performance Reversi (Othello) engine powered by neural networks"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<SubCommands>,

    #[command(flatten)]
    engine_params: EngineParams,
}

#[derive(Debug, Subcommand)]
enum SubCommands {
    #[command(about = "Start the GTP (Go Text Protocol) interface for engine communication")]
    Gtp {
        #[command(flatten)]
        engine_params: EngineParams,
    },
    #[command(about = "Solve endgame positions from a file")]
    Solve {
        #[arg(help = "Path to the file containing positions to solve")]
        file: PathBuf,

        #[arg(
            long,
            help = "Solve for exact score with perfect play (ignores level setting)"
        )]
        exact: bool,

        #[command(flatten)]
        engine_params: EngineParams,
    },
    #[command(about = "Display version information")]
    Version,
    #[command(about = "Print the GPL-3.0 license covering Neural Reversi itself")]
    ShowLicense,
    #[command(about = "Print license texts of all bundled third-party crates")]
    ShowLicenses,
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
            validate_weight_paths(
                engine_params.eval_file.as_deref(),
                engine_params.eval_sm_file.as_deref(),
            );
            if let Err(e) = solve::solve(
                &file,
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
        Some(SubCommands::Version) => {
            println!(
                "neural-reversi {} ({})",
                env!("CARGO_PKG_VERSION"),
                env!("TARGET")
            );
        }
        Some(SubCommands::ShowLicense) => {
            print!("{}", include_str!("../../../NOTICE"));
            println!();
            print!("{}", include_str!("../../../LICENSE"));
        }
        Some(SubCommands::ShowLicenses) => {
            print!("{}", include_str!("../THIRD_PARTY_LICENSES.txt"));
        }
        None => {
            validate_weight_paths(
                args.engine_params.eval_file.as_deref(),
                args.engine_params.eval_sm_file.as_deref(),
            );
            tui::run(
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
