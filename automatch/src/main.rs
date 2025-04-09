use clap::Parser;
use colored::*;
use reversi_core::piece::Piece;
use reversi_core::square::Square;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

mod engine;
mod game;

use engine::GtpEngine;
use game::GameState;

const ELO_K: f64 = 400.0; // Elo K-factor

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Tool for running matches between GTP-compatible Reversi engines"
)]
struct Cli {
    /// Command for the first engine (program path and arguments)
    #[arg(short = '1', long)]
    engine1: String,

    /// Working directory for the first program
    #[arg(long)]
    engine1_working_dir: Option<PathBuf>,

    /// Command for the second engine (program path and arguments)
    #[arg(short = '2', long)]
    engine2: String,

    /// Working directory for the second program
    #[arg(long)]
    engine2_working_dir: Option<PathBuf>,

    /// Opening file (required)
    #[arg(short, long)]
    opening_file: Option<PathBuf>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
}

fn parse_move(move_str: &str) -> Option<Square> {
    move_str.parse::<Square>().ok()
}

fn read_opening_file(path: &Path) -> io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut openings = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            openings.push(line.to_string());
        }
    }

    Ok(openings)
}

fn play_game(
    black_engine: &mut GtpEngine,
    white_engine: &mut GtpEngine,
    opening_moves: Option<&str>,
) -> io::Result<(GameResult, i32)> {
    let result = play_game_internal(black_engine, white_engine, opening_moves);
    let (game_result, score) = result.unwrap();
    Ok((game_result, score))
}

fn play_game_internal(
    black_engine: &mut GtpEngine,
    white_engine: &mut GtpEngine,
    opening_moves: Option<&str>,
) -> io::Result<(GameResult, i32)> {
    black_engine.clear_board()?;
    white_engine.clear_board()?;

    let mut game_state = GameState::new();
    let mut moves = Vec::new();

    if let Some(opening) = opening_moves {
        let mut i = 0;
        while i + 1 < opening.len() {
            let file = opening.chars().nth(i).unwrap();
            let rank = opening.chars().nth(i + 1).unwrap();

            if !('a'..='h').contains(&file) || !('1'..='8').contains(&rank) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid move in opening sequence: {}{}", file, rank),
                ));
            }

            let mv = format!("{}{}", file, rank);
            let square = parse_move(&mv).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Invalid move: {}", mv))
            })?;

            let color = if game_state.side_to_move() == Piece::Black {
                "black"
            } else {
                "white"
            };

            game_state
                .make_move(Some(square))
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            black_engine.play(color, &mv)?;
            white_engine.play(color, &mv)?;

            moves.push(mv);
            i += 2;
        }
    }

    // Main game loop
    while !game_state.is_game_over() {
        let current_color = if game_state.side_to_move() == Piece::Black {
            "black"
        } else {
            "white"
        };

        let mv = if current_color == "black" {
            black_engine.genmove("black")?
        } else {
            white_engine.genmove("white")?
        };

        if mv.to_lowercase() == "pass" {
            game_state
                .make_move(None)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            if current_color == "black" {
                white_engine.play("black", "pass")?;
            } else {
                black_engine.play("white", "pass")?;
            }
        } else {
            let square = parse_move(&mv).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Invalid move: {}", mv))
            })?;

            game_state
                .make_move(Some(square))
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            if current_color == "black" {
                white_engine.play("black", &mv)?;
            } else {
                black_engine.play("white", &mv)?;
            }
        }

        moves.push(mv);
    }

    let (black_count, white_count) = game_state.get_score();
    let score = match black_count.cmp(&white_count) {
        std::cmp::Ordering::Greater => 64 - (white_count as i32) * 2,
        std::cmp::Ordering::Less => (black_count as i32) * 2 - 64,
        std::cmp::Ordering::Equal => 0,
    };

    let result = match black_count.cmp(&white_count) {
        std::cmp::Ordering::Greater => GameResult::BlackWin,
        std::cmp::Ordering::Less => GameResult::WhiteWin,
        std::cmp::Ordering::Equal => GameResult::Draw,
    };

    Ok((result, score))
}

fn main() -> io::Result<()> {
    let args = Cli::parse();

    let openings = if let Some(ref path) = args.opening_file {
        read_opening_file(path)?
    } else {
        eprintln!("Error: Opening file not specified. Please specify a file with the -o or --opening-file option.");
        return Ok(());
    };

    // Error if opening file is empty
    if openings.is_empty() {
        eprintln!("Error: The opening file doesn't contain any valid positions.");
        return Ok(());
    }

    let engine1_parts = args.engine1.split_whitespace().collect::<Vec<&str>>();
    let engine1_program = engine1_parts[0];
    let engine1_args = engine1_parts[1..]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let engine2_parts = args.engine2.split_whitespace().collect::<Vec<&str>>();
    let engine2_program = engine2_parts[0];
    let engine2_args = engine2_parts[1..]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let mut engine1 = GtpEngine::new(engine1_program, &engine1_args, args.engine1_working_dir)?;
    let mut engine2 = GtpEngine::new(engine2_program, &engine2_args, args.engine2_working_dir)?;

    let engine1_name = engine1.name()?;
    let engine2_name = engine2.name()?;

    let mut engine1_wins = 0;
    let mut engine2_wins = 0;
    let mut draws = 0;
    let total_games = openings.len() * 2;
    let mut total_score = 0;
    let mut current_game = 0;

    let mut last_update = Instant::now();

    for (opening_idx, opening_str) in openings.iter().enumerate() {
        for game_round in 0..2 {
            let is_swapped = game_round == 1;
            let game_number = opening_idx * 2 + game_round + 1;
            current_game += 1;

            let (black_engine, white_engine) = if is_swapped {
                (&mut engine2, &mut engine1)
            } else {
                (&mut engine1, &mut engine2)
            };

            match play_game(black_engine, white_engine, Some(opening_str)) {
                Ok((result, black_score)) => {
                    match result {
                        GameResult::BlackWin => {
                            if is_swapped {
                                engine2_wins += 1;
                            } else {
                                engine1_wins += 1;
                            }
                        }
                        GameResult::WhiteWin => {
                            if is_swapped {
                                engine1_wins += 1;
                            } else {
                                engine2_wins += 1;
                            }
                        }
                        GameResult::Draw => {
                            draws += 1;
                        }
                    };

                    if is_swapped {
                        total_score -= black_score;
                    } else {
                        total_score += black_score;
                    }

                    if last_update.elapsed() >= Duration::from_secs(1) {
                        print!(
                            "\r\x1B[2KGame {}-{} | {} ({:2.1}%) {}-{}-{} {} ({:2.1}%) | Score: {}",
                            opening_idx + 1,
                            game_round + 1,
                            engine1_name.bold(),
                            ((engine1_wins as f64 / current_game as f64) * 1000.0).round() / 10.0,
                            engine1_wins,
                            draws,
                            engine2_wins,
                            engine2_name.bold(),
                            ((engine2_wins as f64 / current_game as f64) * 1000.0).round() / 10.0,
                            total_score
                        );

                        io::stdout().flush()?;
                        last_update = Instant::now();
                    }
                }
                Err(e) => {
                    eprintln!("\nFatal error in game {}: {}", game_number, e);
                    break;
                }
            }
        }
    }

    if total_games == 0 {
        eprintln!("No games were played.");
        return Ok(());
    }

    let total_games = engine1_wins + engine2_wins + draws;

    // Get the maximum length of engine names to adjust column width
    let name_max_len = std::cmp::max(engine1_name.len(), engine2_name.len());
    let name_width = std::cmp::max(name_max_len, 10);

    println!("\r\x1B[2K### Match Results");
    println!();
    println!("Total games: {}", total_games);

    let header = format!(
        "| {:Name$} | {:^5} | {:^7} | {:^7} | {:^8} | {:^5} |",
        "Engine",
        "Wins",
        "Losses",
        "Draws",
        "Win Rate",
        "Score",
        Name = name_width
    );
    let separator = format!(
        "|-{:-<Name$}-|-------|---------|---------|----------|-------|",
        "",
        Name = name_width
    );

    println!("{}", header);
    println!("{}", separator);

    let engine1_win_rate = (engine1_wins as f64 / total_games as f64) * 100.0;
    let engine2_win_rate = (engine2_wins as f64 / total_games as f64) * 100.0;

    // ELO計算と信頼区間の計算
    let elo_stats = calculate_elo_stats(engine1_wins, engine2_wins, draws, total_games);

    println!(
        "| {:Name$} | {:5} | {:7} | {:7} | {:7.1}% | {:5.2} |",
        engine1_name,
        engine1_wins,
        engine2_wins,
        draws,
        engine1_win_rate,
        total_score,
        Name = name_width
    );

    println!(
        "| {:Name$} | {:5} | {:7} | {:7} | {:7.1}% | {:5.2} |",
        engine2_name,
        engine2_wins,
        engine1_wins,
        draws,
        engine2_win_rate,
        -total_score,
        Name = name_width
    );

    println!();
    println!("Engine | {}", engine1_name);
    println!(
        "Elo    | {:+.2} ± {:.2} (95%)",
        elo_stats.elo_diff, elo_stats.confidence_interval
    );
    println!(
        "Games  | N: {} W: {} L: {} D: {}",
        total_games, engine1_wins, engine2_wins, draws
    );

    Ok(())
}

struct EloStats {
    elo_diff: f64,
    confidence_interval: f64,
}

fn calculate_elo_stats(wins: u32, losses: u32, draws: u32, total_games: u32) -> EloStats {
    if total_games == 0 {
        return EloStats {
            elo_diff: 0.0,
            confidence_interval: 0.0,
        };
    }

    let n = total_games as f64;
    let w = wins as f64;
    let d = draws as f64;
    let l = losses as f64;

    let p_hat = (w + 0.5 * d) / n;

    let elo_diff = if p_hat == 0.0 || p_hat == 1.0 {
        if p_hat > 0.5 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        -ELO_K * (-(p_hat / (1.0 - p_hat)).ln()) / std::f64::consts::LN_10
    };

    let wld_var = w * (1.0 - p_hat).powi(2) + l * p_hat.powi(2) + d * (0.5 - p_hat).powi(2);
    let se_elo = if p_hat == 0.0 || p_hat == 1.0 {
        f64::INFINITY
    } else {
        (ELO_K / (std::f64::consts::LN_10 * n)) * wld_var.sqrt() / (p_hat * (1.0 - p_hat))
    };

    let z_score = 1.96;
    let confidence_interval = z_score * se_elo;

    EloStats {
        elo_diff,
        confidence_interval,
    }
}
