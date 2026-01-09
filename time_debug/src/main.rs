//! Time control debugging tool.

use std::time::Instant;

use clap::{Parser, ValueEnum};
use colored::Colorize;

use reversi_core::game_state::GameState;
use reversi_core::piece::Piece;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::search_context::GamePhase;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{Search, SearchProgress, SearchRunOptions};
use reversi_core::types::Selectivity;

/// Time control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TimeMode {
    None,
    Byoyomi,
    Fischer,
}

/// Time control debugging tool.
#[derive(Parser)]
#[command(author, version, about = "Debug time control in self-play games")]
struct Args {
    /// Number of games to play
    #[arg(short, long, default_value_t = 1)]
    games: u32,

    /// Time control mode
    #[arg(short, long, value_enum, default_value_t = TimeMode::Byoyomi)]
    time_mode: TimeMode,

    /// Main time in milliseconds (for Fischer mode)
    #[arg(long, default_value_t = 60000)]
    main_time: u64,

    /// Time per move in milliseconds (byoyomi) or increment (Fischer)
    #[arg(long, default_value_t = 0)]
    byoyomi: u64,

    /// Hash table size in MB
    #[arg(long, default_value_t = 256)]
    hash_size: usize,

    /// Opening moves (e.g., "f5d6c3")
    #[arg(short, long, default_value = "")]
    opening: String,

    /// Show search progress for each move
    #[arg(short, long)]
    verbose: bool,
}

/// Game statistics.
#[derive(Default)]
struct GameStats {
    black_wins: u32,
    white_wins: u32,
    draws: u32,
    black_timeouts: u32,
    white_timeouts: u32,
    total_black_time_ms: u64,
    total_white_time_ms: u64,
    total_black_moves: u32,
    total_white_moves: u32,
    max_black_time_ms: u64,
    max_white_time_ms: u64,
}

/// Player time tracker.
struct PlayerTime {
    /// Remaining main time in milliseconds.
    main_time_ms: u64,
    /// Byoyomi time per move in milliseconds.
    byoyomi_ms: u64,
    /// Whether the player has entered byoyomi phase.
    in_byoyomi: bool,
    /// Time control mode.
    mode: TimeMode,
}

impl PlayerTime {
    fn new(mode: TimeMode, main_time_ms: u64, byoyomi_ms: u64) -> Self {
        match mode {
            TimeMode::None => Self {
                main_time_ms: u64::MAX,
                byoyomi_ms: 0,
                in_byoyomi: false,
                mode,
            },
            TimeMode::Byoyomi => Self {
                main_time_ms,
                byoyomi_ms,
                in_byoyomi: main_time_ms == 0,
                mode,
            },
            TimeMode::Fischer => Self {
                main_time_ms,
                byoyomi_ms, // increment
                in_byoyomi: false,
                mode,
            },
        }
    }

    fn use_time(&mut self, elapsed_ms: u64) -> bool {
        match self.mode {
            TimeMode::None => true,
            TimeMode::Byoyomi => {
                if self.in_byoyomi {
                    // In byoyomi: succeed if within byoyomi time
                    elapsed_ms <= self.byoyomi_ms
                } else {
                    // In main time phase
                    if elapsed_ms <= self.main_time_ms {
                        self.main_time_ms -= elapsed_ms;
                        true
                    } else {
                        // Main time exhausted, transition to byoyomi
                        let overtime = elapsed_ms - self.main_time_ms;
                        self.main_time_ms = 0;
                        self.in_byoyomi = true;
                        // Check if this move exceeded byoyomi
                        overtime <= self.byoyomi_ms
                    }
                }
            }
            TimeMode::Fischer => {
                if elapsed_ms > self.main_time_ms {
                    self.main_time_ms = 0;
                    false
                } else {
                    self.main_time_ms = self.main_time_ms.saturating_sub(elapsed_ms);
                    self.main_time_ms = self.main_time_ms.saturating_add(self.byoyomi_ms);
                    true
                }
            }
        }
    }

    fn get_time_control(&self) -> TimeControlMode {
        match self.mode {
            TimeMode::None => TimeControlMode::Infinite,
            TimeMode::Byoyomi => {
                if self.byoyomi_ms == 0 {
                    // No byoyomi means sudden death (same as Fischer with no increment)
                    TimeControlMode::Fischer {
                        main_time_ms: self.main_time_ms,
                        increment_ms: 0,
                    }
                } else if self.in_byoyomi {
                    // Already in byoyomi phase
                    TimeControlMode::Byoyomi {
                        time_per_move_ms: self.byoyomi_ms,
                    }
                } else {
                    // Main time phase with byoyomi safety net
                    TimeControlMode::JapaneseByo {
                        main_time_ms: self.main_time_ms,
                        time_per_move_ms: self.byoyomi_ms,
                    }
                }
            }
            TimeMode::Fischer => TimeControlMode::Fischer {
                main_time_ms: self.main_time_ms,
                increment_ms: self.byoyomi_ms,
            },
        }
    }

    /// Returns the remaining time string for display.
    fn remaining_str(&self) -> String {
        match self.mode {
            TimeMode::None => "-".to_string(),
            TimeMode::Byoyomi => {
                if self.in_byoyomi {
                    format!("Byo:{}", self.byoyomi_ms)
                } else {
                    format!("{}", self.main_time_ms)
                }
            }
            TimeMode::Fischer => format!("{}", self.main_time_ms),
        }
    }

    /// Returns the available time for this move in milliseconds.
    fn available_time_ms(&self) -> u64 {
        match self.mode {
            TimeMode::None => u64::MAX,
            TimeMode::Byoyomi => {
                if self.in_byoyomi {
                    self.byoyomi_ms
                } else {
                    // In main time: use main time or byoyomi as safety net
                    self.main_time_ms.max(self.byoyomi_ms)
                }
            }
            TimeMode::Fischer => self.main_time_ms,
        }
    }
}

fn main() {
    let args = Args::parse();

    // Initialize reversi_core
    reversi_core::probcut::init();
    reversi_core::stability::init();

    println!("{}", "=".repeat(80).cyan());
    println!("{}", " Time Control Debug Tool ".bold().cyan());
    println!("{}", "=".repeat(80).cyan());
    println!();

    // Print configuration
    println!("{}", "Configuration:".bold());
    println!("  Games:       {}", args.games);
    println!("  Time mode:   {:?}", args.time_mode);
    match args.time_mode {
        TimeMode::None => println!("  Time limit:  None"),
        TimeMode::Byoyomi => {
            println!("  Main time:   {} ms", args.main_time);
            println!("  Byoyomi:     {} ms/move", args.byoyomi);
        }
        TimeMode::Fischer => {
            println!("  Main time:   {} ms", args.main_time);
            println!("  Increment:   {} ms/move", args.byoyomi);
        }
    }
    println!("  Hash size:   {} MB", args.hash_size);
    if !args.opening.is_empty() {
        println!("  Opening:     {}", args.opening);
    }
    println!();

    // Create search instance
    let options = SearchOptions::new(args.hash_size);
    let mut black_search = Search::new(&options);
    let mut white_search = Search::new(&options);
    let selectivity = Selectivity::Level1;

    let mut stats = GameStats::default();

    for game_num in 1..=args.games {
        println!("{}", "-".repeat(80).yellow());
        println!(
            "{} ",
            format!("Game {game_num}/{}:", args.games).bold().yellow()
        );
        println!("{}", "-".repeat(80).yellow());

        let result = play_game(
            &mut black_search,
            &mut white_search,
            selectivity,
            &args,
            game_num,
            &mut stats,
        );

        println!();
        match result {
            GameResult::BlackWin(score) => {
                println!(
                    "{}",
                    format!("Result: Black wins by {score} discs")
                        .green()
                        .bold()
                );
                stats.black_wins += 1;
            }
            GameResult::WhiteWin(score) => {
                println!(
                    "{}",
                    format!("Result: White wins by {score} discs").red().bold()
                );
                stats.white_wins += 1;
            }
            GameResult::Draw => {
                println!("{}", "Result: Draw".yellow().bold());
                stats.draws += 1;
            }
            GameResult::BlackTimeout => {
                println!("{}", "Result: Black timeout!".red().bold());
                stats.black_timeouts += 1;
                stats.white_wins += 1;
            }
            GameResult::WhiteTimeout => {
                println!("{}", "Result: White timeout!".green().bold());
                stats.white_timeouts += 1;
                stats.black_wins += 1;
            }
        }
        println!();
    }

    // Print final statistics
    print_final_stats(&stats, &args);
}

#[derive(Debug)]
enum GameResult {
    BlackWin(i32),
    WhiteWin(i32),
    Draw,
    BlackTimeout,
    WhiteTimeout,
}

fn play_game(
    black_search: &mut Search,
    white_search: &mut Search,
    selectivity: Selectivity,
    args: &Args,
    _game_num: u32,
    stats: &mut GameStats,
) -> GameResult {
    let mut game = GameState::new();

    // Apply opening moves
    if !args.opening.is_empty() {
        let opening = &args.opening;
        let mut i = 0;
        while i + 1 < opening.len() {
            let file = opening.chars().nth(i).unwrap();
            let rank = opening.chars().nth(i + 1).unwrap();
            if let Ok(sq) = format!("{file}{rank}").parse() {
                let _ = game.make_move(sq);
            }
            i += 2;
        }
        println!("  Opening applied: {}", args.opening);
    }

    // Initialize time trackers
    let mut black_time = PlayerTime::new(args.time_mode, args.main_time, args.byoyomi);
    let mut white_time = PlayerTime::new(args.time_mode, args.main_time, args.byoyomi);

    let mut move_num = 0;

    println!();
    println!(
        "  {:>4} {:>6} {:>8} {:>10} {:>10} {:>8} {:>6} {:>7} {:>10} {:>10}",
        "Move",
        "Side",
        "Square",
        "Time(ms)",
        "Remaining",
        "Depth",
        "Score",
        "Phase",
        "NPS",
        "Nodes"
    );
    println!("  {}", "-".repeat(115));

    loop {
        if game.is_game_over() {
            break;
        }

        let board = game.board();
        let legal_moves = board.get_moves();

        if legal_moves == 0 {
            let _ = game.make_pass();
            continue;
        }

        move_num += 1;
        let is_black = game.side_to_move() == Piece::Black;
        let side_str = if is_black { "Black" } else { "White" };

        // Get time control for this player
        let player_time = if is_black {
            &mut black_time
        } else {
            &mut white_time
        };

        let search: &mut Search = if is_black {
            &mut *black_search
        } else {
            &mut *white_search
        };

        // Minimum time threshold for full search (in milliseconds)
        const MIN_TIME_FOR_SEARCH_MS: u64 = 15;

        // Check if time is critically low - use quick_move fallback
        let (result, elapsed_ms) = if player_time.available_time_ms() < MIN_TIME_FOR_SEARCH_MS {
            let start = Instant::now();
            let result = search.quick_move(board);
            let elapsed_ms = start.elapsed().as_millis() as u64;
            (result, elapsed_ms)
        } else {
            let time_control = player_time.get_time_control();

            // Create progress callback if verbose
            let callback: Option<Box<dyn Fn(SearchProgress) + Send + Sync>> = if args.verbose {
                Some(Box::new(move |progress: SearchProgress| {
                    println!(
                        "    {} depth={} score={:.2} pv={:?}",
                        if is_black { "B" } else { "W" },
                        progress.depth,
                        progress.score,
                        progress.best_move
                    );
                }))
            } else {
                None
            };

            // Run search with time control
            let start = Instant::now();
            let options = if let Some(cb) = callback {
                SearchRunOptions::with_time(time_control, selectivity).callback(cb)
            } else {
                SearchRunOptions::with_time(time_control, selectivity)
            };
            let result = search.run(board, &options);
            let elapsed_ms = start.elapsed().as_millis() as u64;
            (result, elapsed_ms)
        };

        // Check for timeout
        let has_time = player_time.use_time(elapsed_ms);
        if !has_time {
            let phase_str = match result.game_phase {
                GamePhase::MidGame => "Mid",
                GamePhase::EndGame => "End",
            };
            let nps = if elapsed_ms > 0 {
                result.n_nodes * 1000 / elapsed_ms
            } else {
                0
            };
            println!(
                "  {:>4} {:>6} {:>8} {:>10} {:>10} {:>8} {:>6} {:>7} {:>10} {:>10}",
                move_num,
                side_str.red(),
                "TIMEOUT".red(),
                format!("{elapsed_ms}").red(),
                "0".red(),
                "-",
                "-",
                phase_str.red(),
                format_nps(nps).red(),
                "-"
            );
            return if is_black {
                GameResult::BlackTimeout
            } else {
                GameResult::WhiteTimeout
            };
        }

        // Update statistics
        if is_black {
            stats.total_black_time_ms += elapsed_ms;
            stats.total_black_moves += 1;
            stats.max_black_time_ms = stats.max_black_time_ms.max(elapsed_ms);
        } else {
            stats.total_white_time_ms += elapsed_ms;
            stats.total_white_moves += 1;
            stats.max_white_time_ms = stats.max_white_time_ms.max(elapsed_ms);
        }

        // Get remaining time display
        let remaining_str = player_time.remaining_str();

        // Make the move
        if let Some(best_move) = result.best_move {
            let time_color = if elapsed_ms > args.byoyomi * 90 / 100 {
                format!("{elapsed_ms}").red()
            } else if elapsed_ms > args.byoyomi * 50 / 100 {
                format!("{elapsed_ms}").yellow()
            } else {
                format!("{elapsed_ms}").green()
            };

            let phase_str = match result.game_phase {
                GamePhase::MidGame => "Mid".cyan(),
                GamePhase::EndGame => "End".magenta(),
            };

            let (nps, nodes_str) = if elapsed_ms > 0 {
                (
                    result.n_nodes * 1000 / elapsed_ms,
                    format!("{}", result.n_nodes),
                )
            } else {
                (0, "-".to_string())
            };

            // Format depth with selectivity probability (like ffotest)
            let depth_str = {
                let probability = result.get_probability();
                if probability == 100 {
                    format!("{}", result.depth)
                } else {
                    format!("{}@{}%", result.depth, probability)
                }
            };

            println!(
                "  {:>4} {:>6} {:>8} {:>10} {:>10} {:>8} {:>6.2} {:>7} {:>10} {:>10}",
                move_num,
                if is_black {
                    side_str.white()
                } else {
                    side_str.bright_black()
                },
                format!("{best_move:?}"),
                time_color,
                remaining_str,
                depth_str,
                result.score,
                phase_str,
                format_nps(nps),
                nodes_str
            );

            let _ = game.make_move(best_move);
        } else {
            println!("  {:>4} {:>6} {:>8}", move_num, side_str, "NO MOVE".red());
            break;
        }
    }

    // Game over - count discs
    let (black_count, white_count) = game.get_score();

    if black_count > white_count {
        GameResult::BlackWin((black_count - white_count) as i32)
    } else if white_count > black_count {
        GameResult::WhiteWin((white_count - black_count) as i32)
    } else {
        GameResult::Draw
    }
}

fn print_final_stats(stats: &GameStats, args: &Args) {
    println!("{}", "=".repeat(80).cyan());
    println!("{}", " Final Statistics ".bold().cyan());
    println!("{}", "=".repeat(80).cyan());
    println!();

    let total_games = stats.black_wins + stats.white_wins + stats.draws;
    println!("{}", "Results:".bold());
    println!(
        "  Black wins:    {} ({:.1}%)",
        stats.black_wins,
        stats.black_wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "  White wins:    {} ({:.1}%)",
        stats.white_wins,
        stats.white_wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "  Draws:         {} ({:.1}%)",
        stats.draws,
        stats.draws as f64 / total_games as f64 * 100.0
    );
    println!();

    if stats.black_timeouts > 0 || stats.white_timeouts > 0 {
        println!("{}", "Timeouts:".bold().red());
        println!("  Black timeouts: {}", stats.black_timeouts);
        println!("  White timeouts: {}", stats.white_timeouts);
        println!();
    }

    println!("{}", "Time Statistics:".bold());
    if stats.total_black_moves > 0 {
        let avg_black = stats.total_black_time_ms as f64 / stats.total_black_moves as f64;
        println!("  Black:");
        println!("    Total time:   {} ms", stats.total_black_time_ms);
        println!("    Total moves:  {}", stats.total_black_moves);
        println!("    Avg time:     {:.1} ms/move", avg_black);
        println!("    Max time:     {} ms", stats.max_black_time_ms);
        if args.time_mode != TimeMode::None {
            let budget = match args.time_mode {
                TimeMode::Byoyomi | TimeMode::Fischer => {
                    args.main_time as f64 + args.byoyomi as f64 * stats.total_black_moves as f64
                }
                TimeMode::None => 0.0,
            };

            if budget > 0.0 {
                println!(
                    "    Budget usage: {:.1}% (total budget {:.0} ms)",
                    stats.total_black_time_ms as f64 / budget * 100.0,
                    budget
                );
            }
        }
    }
    println!();
    if stats.total_white_moves > 0 {
        let avg_white = stats.total_white_time_ms as f64 / stats.total_white_moves as f64;
        println!("  White:");
        println!("    Total time:   {} ms", stats.total_white_time_ms);
        println!("    Total moves:  {}", stats.total_white_moves);
        println!("    Avg time:     {:.1} ms/move", avg_white);
        println!("    Max time:     {} ms", stats.max_white_time_ms);
        if args.time_mode != TimeMode::None {
            let budget = match args.time_mode {
                TimeMode::Byoyomi | TimeMode::Fischer => {
                    args.main_time as f64 + args.byoyomi as f64 * stats.total_white_moves as f64
                }
                TimeMode::None => 0.0,
            };

            if budget > 0.0 {
                println!(
                    "    Budget usage: {:.1}% (total budget {:.0} ms)",
                    stats.total_white_time_ms as f64 / budget * 100.0,
                    budget
                );
            }
        }
    }
    println!();
}

/// Format NPS with K/M suffix for readability.
fn format_nps(nps: u64) -> String {
    if nps >= 1_000_000 {
        format!("{:.1}M", nps as f64 / 1_000_000.0)
    } else if nps >= 1_000 {
        format!("{:.1}K", nps as f64 / 1_000.0)
    } else {
        format!("{}", nps)
    }
}
