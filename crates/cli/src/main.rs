mod config;
mod ggs;
mod gtp;
mod solve;
mod text;
mod tui;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use config::EngineConfig;
use reversi_core::level::MAX_LEVEL;

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

impl From<EngineParams> for EngineConfig {
    fn from(params: EngineParams) -> Self {
        EngineConfig {
            hash_size: params.hash_size,
            level: params.level,
            threads: params.threads,
            eval_file: params.eval_file,
            eval_sm_file: params.eval_sm_file,
        }
    }
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

        #[arg(
            long,
            help = "Report the evaluation for every legal move in each position"
        )]
        all_moves: bool,

        #[command(flatten)]
        engine_params: EngineParams,
    },
    #[command(about = "Connect to a GGS server (default localhost:5000) and play via /os")]
    Ggs {
        #[arg(long, default_value = "localhost", help = "GGS server hostname")]
        host: String,

        #[arg(
            long,
            default_value = "5000",
            value_parser = clap::value_parser!(u16),
            help = "GGS server port"
        )]
        port: u16,

        #[arg(
            long,
            value_name = "NAME",
            help = "GGS login name; also identifies our side in match events"
        )]
        user: String,

        #[arg(long, help = "GGS password, sent verbatim at the password prompt")]
        password: String,

        #[arg(
            long,
            default_value = "1",
            value_parser = parse_usize_range::<1, 64>,
            help = "Maximum simultaneous GGS matches (synchro counts as one); configured hash is split across up to two boards per match, which share one pool of search threads and search one at a time"
        )]
        max_games: usize,

        #[arg(
            long,
            help = "While every game waits for the opponent, search the opponent-to-move board to warm each game's transposition table"
        )]
        ponder: bool,

        #[arg(
            long,
            help = "Use plain line input and output instead of the interactive GGS terminal interface"
        )]
        plain: bool,

        #[arg(
            long,
            help = "Send `tell /td join <id>` or `tell /os join <id>` when any user tells us to, for unattended tournament entry"
        )]
        join_tournaments: bool,

        #[arg(
            long = "command",
            value_name = "TEXT",
            help = "Send a GGS command after login setup; repeat to send multiple commands"
        )]
        startup_commands: Vec<String>,

        #[arg(
            long,
            value_name = "PATH",
            help = "Append every line sent to and received from the server after the password, with Unix timestamps"
        )]
        log_file: Option<PathBuf>,

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
            let config = EngineConfig::from(engine_params);
            let mut gtp_engine = gtp::GtpEngine::new(&config).unwrap_or_else(|err| {
                eprintln!("Failed to initialize engine: {err}");
                std::process::exit(1);
            });
            gtp_engine.run();
        }
        Some(SubCommands::Solve {
            file,
            exact,
            all_moves,
            engine_params,
        }) => {
            let config = EngineConfig::from(engine_params);
            if let Err(e) = solve::solve(&file, &config, exact, all_moves) {
                eprintln!("Error solving game: {e}");
            }
        }
        Some(SubCommands::Ggs {
            host,
            port,
            user,
            password,
            max_games,
            ponder,
            plain,
            join_tournaments,
            startup_commands,
            log_file,
            engine_params,
        }) => {
            let config = EngineConfig::from(engine_params);
            if let Err(e) = ggs::run_ggs(
                ggs::GgsOptions {
                    host: &host,
                    port,
                    user: &user,
                    password: &password,
                    max_games,
                    ponder,
                    plain,
                    join_tournaments,
                    startup_commands: &startup_commands,
                    log_file: log_file.as_deref(),
                },
                &config,
            ) {
                eprintln!("GGS session error: {}", text::visible(&e.to_string()));
                // A distinct code lets reconnect loops stop on a bad login.
                let code = match e {
                    ggs::GgsError::LoginRejected(_) => 2,
                    ggs::GgsError::Other(_) => 1,
                };
                std::process::exit(code);
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
            let config = EngineConfig::from(args.engine_params);
            tui::run(&config).unwrap_or_else(|err| {
                eprintln!("Failed to initialize UI: {err}");
            });
        }
    }
}
