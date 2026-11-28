mod config;
mod flipsight;
mod ggs;
mod gtp;
mod solve;
mod text;
mod tui;

use std::path::PathBuf;

use clap::builder::TypedValueParser as _;
use clap::{Parser, Subcommand};
use config::EngineConfig;
use reversi_core::constants::MAX_THREADS;
use reversi_core::level::MAX_LEVEL;

fn parse_usize_range<const LO: usize, const HI: usize>(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|e| format!("{e}"))?;
    if (LO..=HI).contains(&v) {
        Ok(v)
    } else {
        Err(format!("value must be between {LO} and {HI}"))
    }
}

/// Accepts a path that names something other than a directory, so a mistyped
/// file argument fails before the engine starts.
fn existing_file(path: PathBuf) -> Result<PathBuf, String> {
    match std::fs::metadata(&path) {
        Ok(meta) if meta.is_dir() => Err("is a directory".to_string()),
        Ok(_) => Ok(path),
        Err(e) => Err(e.to_string()),
    }
}

/// Applies the login-name rule of the GGS server, which answers any other
/// name with a new login prompt. A `\` is refused as well: the server reads it
/// as a line break.
fn parse_ggs_login(s: &str) -> Result<String, String> {
    if let Some(c) = s
        .chars()
        .find(|c| !c.is_ascii_graphic() || matches!(c, ',' | '\\'))
    {
        return Err(format!("must not contain {c:?}"));
    }
    if !(2..=8).contains(&s.len()) {
        return Err("must be 2 to 8 characters".to_string());
    }
    if s.starts_with(['.', '_']) {
        return Err("must not start with '.' or '_'".to_string());
    }
    Ok(s.to_string())
}

/// Accepts text the GGS server reads as one line; a line break would send the
/// rest as a separate command.
fn parse_single_line(s: &str) -> Result<String, String> {
    if s.contains(['\r', '\n']) {
        Err("must be a single line".to_string())
    } else {
        Ok(s.to_string())
    }
}

fn parse_host(s: &str) -> Result<String, String> {
    if s.is_empty() {
        Err("must not be empty".to_string())
    } else if s.contains(|c: char| c.is_whitespace() || c.is_control()) {
        Err("must not contain spaces or control characters".to_string())
    } else {
        Ok(s.to_string())
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
        value_parser = parse_usize_range::<1, MAX_THREADS>,
        help = "Number of search threads, lowered to the CPU count [default: CPU count]"
    )]
    threads: Option<usize>,

    #[arg(
        long = "eval-file",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath,
        value_parser = clap::builder::PathBufValueParser::new().try_map(existing_file),
        help = "Path to the main network weight file"
    )]
    eval_file: Option<PathBuf>,

    #[arg(
        long = "eval-sm-file",
        value_name = "FILE",
        value_hint = clap::ValueHint::FilePath,
        value_parser = clap::builder::PathBufValueParser::new().try_map(existing_file),
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
    about = "Neural Reversi — a high-performance Reversi (Othello) engine powered by neural networks",
    args_conflicts_with_subcommands = true
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
        #[arg(
            value_hint = clap::ValueHint::FilePath,
            value_parser = clap::builder::PathBufValueParser::new().try_map(existing_file),
            help = "Path to the file containing positions to solve"
        )]
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
        #[arg(
            long,
            default_value = "localhost",
            value_parser = parse_host,
            help = "GGS server hostname"
        )]
        host: String,

        #[arg(
            long,
            default_value = "5000",
            value_parser = clap::value_parser!(u16).range(1..),
            help = "GGS server port"
        )]
        port: u16,

        #[arg(
            long,
            value_name = "NAME",
            value_parser = parse_ggs_login,
            help = "GGS login name; also identifies our side in match events"
        )]
        user: String,

        #[arg(
            long,
            value_parser = parse_single_line,
            help = "GGS password, sent verbatim at the password prompt"
        )]
        password: String,

        #[arg(
            long,
            default_value = "1",
            value_parser = parse_usize_range::<1, 64>,
            help = "Maximum simultaneous GGS matches (synchro counts as one); each match can allocate up to two transposition tables of the configured hash size (one per board once a synchro pair diverges); all boards share one pool of search threads and search one at a time"
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
            help = "Join a /td tournament when its director tells us `tell /td join <id>`, for unattended tournament entry; the director, an /os service, and a game type and clock the engine plays are verified with `tell /td finger <id>` first"
        )]
        join_tournaments: bool,

        #[arg(
            long = "command",
            value_name = "TEXT",
            value_parser = parse_single_line,
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
    #[command(
        about = "Answer FlipSight bot connector requests (its --engine JSON-lines protocol); --threads defaults to the request's engine_threads and is capped at 4, and --hash-size applies to each game's process"
    )]
    Flipsight {
        #[arg(
            long,
            help = "While the opponent thinks, search the position after our move to warm the transposition table; reported to FlipSight"
        )]
        ponder: bool,

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
    let process_start = std::time::Instant::now();
    let args = Cli::parse();
    match args.command {
        Some(SubCommands::Flipsight {
            ponder,
            engine_params,
        }) => {
            if let Err(e) = flipsight::run(EngineConfig::from(engine_params), ponder, process_start)
            {
                eprintln!("FlipSight engine error: {e}");
                std::process::exit(1);
            }
        }
        Some(SubCommands::Gtp { engine_params }) => {
            let config = EngineConfig::from(engine_params);
            match gtp::GtpEngine::new(&config) {
                Ok(mut gtp_engine) => gtp_engine.run(),
                Err(e) => {
                    eprintln!("GTP engine error: {e}");
                    std::process::exit(1);
                }
            }
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
                std::process::exit(1);
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
        Some(SubCommands::Version) => match env!("TARGET_CPU") {
            "" => println!(
                "neural-reversi {} ({})",
                env!("CARGO_PKG_VERSION"),
                env!("TARGET")
            ),
            cpu => println!(
                "neural-reversi {} ({}, target-cpu={cpu})",
                env!("CARGO_PKG_VERSION"),
                env!("TARGET")
            ),
        },
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
                eprintln!("TUI error: {err}");
                std::process::exit(1);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::error::ErrorKind;

    use super::*;

    fn value_rejected(args: &[&str]) -> bool {
        let args = std::iter::once("cli").chain(args.iter().copied());
        matches!(Cli::try_parse_from(args), Err(e) if e.kind() == ErrorKind::ValueValidation)
    }

    fn accepted(args: &[&str]) -> bool {
        Cli::try_parse_from(std::iter::once("cli").chain(args.iter().copied())).is_ok()
    }

    #[test]
    fn threads_must_be_within_the_engine_limit() {
        let max = MAX_THREADS.to_string();
        let over = (MAX_THREADS + 1).to_string();
        assert!(accepted(&["--threads", "1"]));
        assert!(accepted(&["gtp", "--threads", &max]));
        assert!(value_rejected(&["--threads", "0"]));
        assert!(value_rejected(&["gtp", "--threads", &over]));
    }

    #[test]
    fn file_arguments_must_name_an_existing_non_directory() {
        let dir = std::env::temp_dir();
        let dir = dir.to_str().unwrap();
        let missing = format!("{dir}/neural-reversi-cli-no-such-file");
        for path in [dir, missing.as_str()] {
            assert!(value_rejected(&["--eval-file", path]), "{path}");
            assert!(value_rejected(&["gtp", "--eval-sm-file", path]), "{path}");
            assert!(value_rejected(&["solve", path]), "{path}");
        }
    }

    #[test]
    fn ggs_login_follows_the_server_rule() {
        let ggs = |user: &'static str| ["ggs", "--password", "p", "--user", user];
        for user in ["ab", "Rhapsody", "n.r_a", "a-b+c"] {
            assert!(accepted(&ggs(user)), "{user:?}");
        }
        for user in [
            "",
            "a",
            "Rhapsody9",
            ".ab",
            "_ab",
            "a,b",
            "a b",
            "a\\b",
            "a\nb",
            "a\tb",
            "あい",
        ] {
            assert!(value_rejected(&ggs(user)), "{user:?}");
        }
    }

    #[test]
    fn ggs_lines_and_address_cannot_inject_or_misroute() {
        let ggs = |option: &'static str, value: &'static str| {
            ["ggs", "--user", "nra", "--password", "p", option, value]
        };
        // An unregistered login takes any password, the empty one included.
        assert!(accepted(&["ggs", "--user", "nra", "--password", ""]));
        assert!(accepted(&ggs("--command", "")));
        assert!(accepted(&ggs("--host", "::1")));
        assert!(accepted(&ggs("--port", "65535")));
        let injected = ["ggs", "--user", "nra", "--password", "p\nquit"];
        assert!(value_rejected(&injected));
        assert!(value_rejected(&ggs("--command", "info a\rinfo b")));
        assert!(value_rejected(&ggs("--host", "")));
        assert!(value_rejected(&ggs("--host", "local host")));
        assert!(value_rejected(&ggs("--port", "0")));
    }
}
