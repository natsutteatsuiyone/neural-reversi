//! Interactive command-line user interface for the reversi engine.
//!
//! This module provides a REPL-style interface for playing reversi games,
//! configuring the AI, and analyzing positions. It supports both human vs AI
//! and AI vs AI gameplay modes.

use reversi_core::{
    self,
    level::{self},
    piece::Piece,
    search::{self, SearchOptions},
    search::search_result::SearchResult,
    square::Square,
    types::Selectivity,
};
use rustyline::{error::ReadlineError, DefaultEditor};
use std::fmt;

use crate::game::GameState;

// Game mode constants
const MODE_BLACK_HUMAN_WHITE_AI: usize = 0;
const MODE_BLACK_AI_WHITE_HUMAN: usize = 1;
const MODE_BOTH_AI: usize = 2;
const MODE_BOTH_HUMAN: usize = 3;
const MAX_MODE: usize = 3;

/// Commands available in the interactive UI.
#[derive(Debug)]
enum Command {
    /// Initialize/reset the game to starting position
    Init,
    /// Start a new game (alias for Init)
    New,
    /// Undo the last move
    Undo,
    /// Set AI search level
    Level(usize),
    /// Get/set game mode (human vs AI configuration)
    Mode(Option<usize>),
    /// Force AI to make a move with detailed analysis
    Go,
    /// Play a sequence of moves
    Play(String),
    /// Set board position from string representation
    SetBoard(String),
    /// Make a move at the specified square
    Move(Square),
    /// Exit the program
    Quit,
    /// Show help information
    Help,
}

impl Command {
    /// Parse a command string into a Command enum.
    ///
    /// Supports both full command names and single-character shortcuts.
    /// If the input doesn't match a known command, attempts to parse it as a move.
    fn parse(input: &str) -> Result<Self, String> {
        let mut parts = input.split_whitespace();
        let cmd = parts.next().ok_or("No command provided")?;

        match cmd {
            "init" | "i" => Ok(Command::Init),
            "new" | "n" => Ok(Command::New),
            "undo" | "u" => Ok(Command::Undo),
            "level" | "l" => {
                if let Some(lvl_str) = parts.next() {
                    match lvl_str.parse::<usize>() {
                        Ok(lvl) => Ok(Command::Level(lvl)),
                        Err(_) => Err("Invalid level format. Please provide a number.".to_string()),
                    }
                } else {
                    Err("Level command requires a number argument.".to_string())
                }
            }
            "mode" | "m" => {
                if let Some(mode_str) = parts.next() {
                    match mode_str.parse::<usize>() {
                        Ok(mode) if mode <= MAX_MODE => Ok(Command::Mode(Some(mode))),
                        Ok(_) => Err(format!(
                            "Invalid mode. Please specify a value between 0-{MAX_MODE}."
                        )),
                        Err(_) => Err("Invalid mode format. Please provide a number.".to_string()),
                    }
                } else {
                    Ok(Command::Mode(None))
                }
            }
            "go" => Ok(Command::Go),
            "play" => {
                if let Some(moves) = parts.next() {
                    Ok(Command::Play(moves.to_string()))
                } else {
                    Err("Play command requires move sequence.".to_string())
                }
            }
            "setboard" => {
                if let Some(board_str) = parts.next() {
                    Ok(Command::SetBoard(board_str.to_string()))
                } else {
                    Err("SetBoard command requires board position string.".to_string())
                }
            }
            "quit" | "q" => Ok(Command::Quit),
            "help" | "h" => Ok(Command::Help),
            _ => {
                // Try to parse as a move
                match cmd.parse::<Square>() {
                    Ok(sq) => Ok(Command::Move(sq)),
                    Err(_) => Err(format!("Unknown command: {cmd}")),
                }
            }
        }
    }
}

/// Wrapper for game mode configuration.
///
/// Determines which players (human or AI) control each side.
struct GameMode(usize);

impl GameMode {
    /// Determines if the AI should play for the given side based on current mode.
    fn should_ai_play(&self, side: Piece) -> bool {
        matches!(
            (self.0, side),
            (MODE_BLACK_HUMAN_WHITE_AI, Piece::White)
                | (MODE_BLACK_AI_WHITE_HUMAN, Piece::Black)
                | (MODE_BOTH_AI, _)
        )
    }
}

impl fmt::Display for GameMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self.0 {
            MODE_BLACK_HUMAN_WHITE_AI => "Black-Human, White-AI",
            MODE_BLACK_AI_WHITE_HUMAN => "Black-AI, White-Human",
            MODE_BOTH_AI => "Black-AI, White-AI",
            MODE_BOTH_HUMAN => "Black-Human, White-Human",
            _ => "Unknown",
        };
        write!(f, "{description}")
    }
}

/// Main interactive UI loop.
///
/// Runs the command-line interface, handling user input and game state.
/// Automatically triggers AI moves based on the current game mode.
///
/// # Arguments
/// * `hash_size` - Size of transposition table in MB
/// * `initial_level` - Initial AI search level
/// * `selectivity` - Search selectivity setting
pub fn ui_loop(hash_size: usize, initial_level: usize, selectivity: Selectivity) {
    let mut rl = DefaultEditor::new().unwrap();
    let mut game = GameState::new();
    let mut search = search::Search::new(&SearchOptions {
        tt_mb_size: hash_size,
        ..Default::default()
    });
    let mut game_mode = GameMode(MODE_BOTH_HUMAN);
    let mut level = initial_level;

    loop {
        game.print();
        println!();

        let current_side = game.get_side_to_move();

        if game_mode.should_ai_play(current_side)
            && !game.board.is_game_over()
            && execute_ai_move(&mut game, &mut search, level, selectivity, false)
        {
            continue;
        }

        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(&line);

                match Command::parse(&line) {
                    Ok(command) => {
                        println!();
                        match handle_command(
                            command,
                            &mut game,
                            &mut search,
                            &mut level,
                            &mut game_mode,
                            selectivity,
                        ) {
                            Ok(should_continue) => {
                                if !should_continue {
                                    break;
                                }
                            }
                            Err(err) => println!("Error: {err}"),
                        }
                    }
                    Err(err) => println!("Error: {err}\n"),
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {err:?}");
                break;
            }
        }
    }
}

/// Handle execution of a parsed command.
///
/// # Returns
/// * `Ok(true)` - Continue the main loop
/// * `Ok(false)` - Exit the program
/// * `Err(msg)` - Command failed with error message
fn handle_command(
    command: Command,
    game: &mut GameState,
    search: &mut search::Search,
    level: &mut usize,
    game_mode: &mut GameMode,
    selectivity: Selectivity,
) -> Result<bool, String> {
    match command {
        Command::Init | Command::New => {
            *game = GameState::new();
            search.init();
            Ok(true)
        }
        Command::Undo => {
            if !game.undo() {
                println!("Cannot undo.");
            }
            Ok(true)
        }
        Command::Level(new_level) => {
            *level = new_level;
            println!("Level set to: {new_level}");
            Ok(true)
        }
        Command::Mode(Some(mode)) => {
            game_mode.0 = mode;
            println!("Mode changed to: {game_mode}");
            Ok(true)
        }
        Command::Mode(None) => {
            print_mode_info(game_mode);
            Ok(true)
        }
        Command::Go => {
            execute_ai_move(game, search, *level, selectivity, false);
            Ok(true)
        }
        Command::Play(moves) => {
            execute_play_sequence(game, &moves)?;
            Ok(true)
        }
        Command::SetBoard(board_str) => {
            match parse_setboard(&board_str) {
                Ok((board, side_to_move)) => {
                    *game = GameState::from_board(board, side_to_move);
                    search.init();
                    println!("Board position set successfully.");
                    Ok(true)
                }
                Err(err) => Err(format!("Invalid board format: {err}")),
            }
        }
        Command::Move(sq) => {
            if game.board.is_legal_move(sq) {
                game.make_move(sq);
            } else {
                return Err(format!("Illegal move: {sq:?}"));
            }
            Ok(true)
        }
        Command::Help => {
            print_help();
            Ok(true)
        }
        Command::Quit => Ok(false),
    }
}

/// Execute AI search on the given board position.
///
/// # Arguments
/// * `board` - Current board position
/// * `search` - Search engine instance
/// * `level` - Search depth/level
/// * `selectivity` - Search selectivity setting
fn execute_ai_search(
    board: &reversi_core::board::Board,
    search: &mut search::Search,
    level: usize,
    selectivity: Selectivity,
) -> SearchResult {
    search.run(
        board,
        level::get_level(level),
        selectivity,
        false,
    )
}

/// Display the results of an AI search in either verbose or compact format.
///
/// # Arguments
/// * `result` - Search result containing evaluation, depth, nodes searched
/// * `mv` - The best move found
/// * `verbose` - Whether to show detailed formatted output
fn display_search_result(result: &SearchResult, mv: Square, verbose: bool) {
    let depth = if result.get_probability() == 100 {
        format!("{}", result.depth)
    } else {
        format!("{}@{}%", result.depth, result.get_probability())
    };

    if verbose {
        println!("╔═══════════════════════════════════════════╗");
        println!("║           AI Search Results               ║");
        println!("╠═══════════════════════════════════════════╣");
        println!("║ Depth      : {depth:>28} ║");
        println!(
            "║ Evaluation : {:>28} ║",
            format!("{:+.2}", result.score)
        );
        println!("║ Nodes      : {:>28} ║", format!("{}", result.n_nodes));
        println!("║ Best Move  : {:>28} ║", format!("{:?}", mv));
        println!("╚═══════════════════════════════════════════╝");
    } else {
        println!(
            "AI plays {:?} (eval: {:+.2}, depth: {}, nodes: {})",
            mv,
            result.score,
            depth,
            result.n_nodes
        );
    }
}

/// Execute an AI move and display the results.
///
/// # Returns
/// * `true` - AI successfully made a move
/// * `false` - No legal moves available for AI
fn execute_ai_move(
    game: &mut GameState,
    search: &mut search::Search,
    level: usize,
    selectivity: Selectivity,
    verbose: bool,
) -> bool {
    let result = execute_ai_search(&game.board, search, level, selectivity);

    if let Some(mv) = result.best_move {
        game.make_move(mv);
        display_search_result(&result, mv, verbose);
        println!();
        true
    } else {
        if verbose {
            println!("No legal moves available for the AI.");
        }
        false
    }
}

/// Parse a board position string into Board and side-to-move.
///
/// Format: 64 characters for board + optional spaces + 1 character for side to move
///
/// Board characters:
/// * 'b', 'B', 'x', 'X', '*' - Black pieces
/// * 'o', 'O', 'w', 'W' - White pieces
/// * '-', '.' - Empty squares
/// * Other characters are ignored
///
/// Side to move characters:
/// * 'b', 'B', 'x', 'X', '*' - Black to move
/// * 'o', 'O', 'w', 'W' - White to move
///
/// # Example
/// ```
/// // Standard starting position with Black to move
/// let pos = "----------------------------ox------xo----------------------------b";
/// ```
fn parse_setboard(board_str: &str) -> Result<(reversi_core::board::Board, Piece), String> {
    if board_str.len() < 65 {
        return Err("Board string must be at least 65 characters (64 for board + 1 for side_to_move)".to_string());
    }

    let board_part = &board_str[..64];
    let remaining = &board_str[64..];

    // Skip any spaces after the board part
    let side_to_move_part = remaining.trim_start();
    if side_to_move_part.is_empty() {
        return Err("Side to move character is missing after board position".to_string());
    }

    let side_to_move_char = side_to_move_part.chars().next().unwrap();

    // Parse the side_to_move
    let side_to_move = match side_to_move_char {
        'b' | 'B' | 'x' | 'X' | '*' => Piece::Black,
        'o' | 'O' | 'w' | 'W' => Piece::White,
        _ => return Err(format!("Invalid side to move character: {side_to_move_char}")),
    };

    // Parse the board
    let mut black_bitboard = 0u64;
    let mut white_bitboard = 0u64;

    for (i, c) in board_part.chars().enumerate() {
        if i >= 64 {
            break;
        }

        let bit = 1u64 << i;
        match c {
            'b' | 'B' | 'x' | 'X' | '*' => {
                black_bitboard |= bit;
            }
            'o' | 'O' | 'w' | 'W' => {
                white_bitboard |= bit;
            }
            '-' | '.' => {
                // Empty square, do nothing
            }
            _ => {
                // Ignore other characters as per specification
            }
        }
    }

    // Create board with the parsed bitboards
    let board = reversi_core::board::Board::from_bitboards(black_bitboard, white_bitboard);
    Ok((board, side_to_move))
}

/// Execute a sequence of moves from a string.
///
/// Moves should be in standard notation (e.g., "d3e4f5").
/// Each move is 2 characters (column + row).
fn execute_play_sequence(game: &mut GameState, moves: &str) -> Result<(), String> {
    for sq_str in moves.chars().collect::<Vec<_>>().chunks(2) {
        let sq_str: String = sq_str.iter().collect();
        let sq = sq_str
            .parse::<Square>()
            .map_err(|_| format!("Invalid move notation: {sq_str}"))?;

        if game.board.is_legal_move(sq) {
            game.make_move(sq);
        } else {
            return Err(format!("Illegal move in sequence: {sq_str}"));
        }
    }
    Ok(())
}

/// Display current game mode and available mode options.
fn print_mode_info(game_mode: &GameMode) {
    println!("Current mode: {} ({})", game_mode.0, game_mode);
    println!("0: Black-Human, White-AI");
    println!("1: Black-AI, White-Human");
    println!("2: Black-AI, White-AI");
    println!("3: Black-Human, White-Human");
}

/// Display help information for all available commands.
fn print_help() {
    println!("Available commands:");
    println!("  <square>        - Make a move (e.g., d3, e4)");
    println!("  init, i         - Initialize a new game");
    println!("  new, n          - Start a new game");
    println!("  undo, u         - Undo last move");
    println!("  level, l <n>    - Set AI level");
    println!("  mode, m [n]     - Show/set game mode");
    println!("  go              - Let AI make a move with analysis");
    println!("  play <moves>    - Play a sequence of moves");
    println!("  setboard <pos>  - Set board position (64 board chars + optional spaces + 1 side to move char)");
    println!("  help, h         - Show this help");
    println!("  quit, q         - Exit the program");
}
