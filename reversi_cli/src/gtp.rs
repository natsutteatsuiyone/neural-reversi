//! GTP (Go Text Protocol) implementation for Neural Reversi.
//!
//! This module provides a GTP-compliant interface for the Neural Reversi engine,
//! allowing it to communicate with GTP-compatible GUI applications and tournament
//! management software. While originally designed for Go, GTP is also used for
//! other board games including Reversi/Othello.
//!
//! The implementation supports all essential GTP commands for game play and
//! engine configuration, including moves, board setup, and engine information.

use reversi_core::{
    level,
    piece::Piece,
    search::{self, SearchOptions},
    square::Square, types::Selectivity,
};

use crate::game::GameState;
use std::io::{self, BufRead, Write};
use std::env;

/// Represents a parsed GTP command with its arguments.
///
/// Each variant encapsulates the command type and any required parameters,
/// providing type safety and eliminating the need for runtime argument validation
/// in most cases.
#[derive(Debug, Clone)]
pub enum Command {
    /// Returns the GTP protocol version (always 2)
    ProtocolVersion,
    /// Returns the engine name
    Name,
    /// Returns the engine version
    Version,
    /// Checks if a command is known to the engine
    #[allow(clippy::enum_variant_names)]
    KnownCommand(String),
    /// Lists all supported commands
    ListCommands,
    /// Terminates the GTP session
    Quit,
    /// Sets the board size (only 8x8 supported)
    Boardsize(usize),
    /// Clears the board to initial position
    ClearBoard,
    /// Plays a move for the specified color
    Play { color: String, move_str: String },
    /// Generates and plays a move for the specified color
    Genmove(String),
    /// Displays the current board state
    Showboard,
    /// Undoes the last move
    Undo,
    /// Sets the engine's playing strength level (1-20)
    SetLevel(usize),
    /// Represents an unknown or malformed command
    Unknown(String),
}

impl Command {
    /// Parses a command string and arguments into a Command enum variant.
    ///
    /// This method performs argument validation and converts string arguments
    /// to appropriate types. Invalid commands or malformed arguments result
    /// in the `Unknown` variant.
    ///
    /// # Arguments
    /// * `cmd` - The command name as a string
    /// * `args` - Command arguments as string slices
    ///
    /// # Returns
    /// A `Command` enum variant representing the parsed command
    fn from_str_with_args(cmd: &str, args: &[&str]) -> Self {
        match cmd {
            "protocol_version" => Command::ProtocolVersion,
            "name" => Command::Name,
            "version" => Command::Version,
            "known_command" => {
                if args.len() == 1 {
                    Command::KnownCommand(args[0].to_string())
                } else {
                    Command::Unknown(cmd.to_string())
                }
            }
            "list_commands" => Command::ListCommands,
            "quit" => Command::Quit,
            "boardsize" => {
                if args.len() == 1 {
                    if let Ok(size) = args[0].parse::<usize>() {
                        Command::Boardsize(size)
                    } else {
                        Command::Unknown(cmd.to_string())
                    }
                } else {
                    Command::Unknown(cmd.to_string())
                }
            }
            "clear_board" => Command::ClearBoard,
            "play" => {
                if args.len() == 2 {
                    Command::Play {
                        color: args[0].to_lowercase(),
                        move_str: args[1].to_lowercase(),
                    }
                } else {
                    Command::Unknown(cmd.to_string())
                }
            }
            "genmove" => {
                if args.len() == 1 {
                    Command::Genmove(args[0].to_lowercase())
                } else {
                    Command::Unknown(cmd.to_string())
                }
            }
            "showboard" => Command::Showboard,
            "undo" => Command::Undo,
            "set_level" => {
                if args.len() == 1 {
                    if let Ok(level) = args[0].parse::<usize>() {
                        Command::SetLevel(level)
                    } else {
                        Command::Unknown(cmd.to_string())
                    }
                } else {
                    Command::Unknown(cmd.to_string())
                }
            }
            _ => Command::Unknown(cmd.to_string()),
        }
    }
}

/// List of all supported GTP command names.
/// Used for the `list_commands` response and command validation.
const COMMAND_NAMES: &[&str] = &[
    "protocol_version", "name", "version", "known_command", "list_commands",
    "quit", "boardsize", "clear_board", "play", "genmove", "showboard",
    "undo", "set_level",
];

/// Represents a GTP response that can be either successful or an error.
///
/// GTP responses are formatted with specific prefixes:
/// - Success responses start with "="
/// - Error responses start with "?"
pub enum GtpResponse {
    /// A successful command response with optional message
    Success(String),
    /// An error response with error message
    Error(String),
}

impl std::fmt::Display for GtpResponse {
    /// Formats the response according to GTP protocol specifications.
    ///
    /// Success responses are prefixed with "=", error responses with "?".
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success(msg) => write!(f, "= {}", msg),
            Self::Error(msg) => write!(f, "? {}", msg),
        }
    }
}

/// The main GTP engine that handles command processing and game state.
///
/// This struct maintains the current game state, search engine configuration,
/// and engine metadata required for GTP communication.
pub struct GtpEngine {
    /// Current game state including board position and move history
    game: GameState,
    /// Neural network search engine for move generation
    search: search::Search,
    /// Current playing strength level (1-20)
    level: usize,
    /// Search selectivity setting
    selectivity: Selectivity,
    /// Engine name reported to GTP clients
    name: String,
    /// Engine version reported to GTP clients
    version: String,
}

impl GtpEngine {
    /// Creates a new GTP engine with the specified configuration.
    ///
    /// # Arguments
    /// * `hash_size` - Size of the transposition table in MB
    /// * `level` - Initial playing strength level (1-20)
    /// * `selectivity` - Search selectivity setting
    ///
    /// # Returns
    /// A new `GtpEngine` instance ready to process commands
    pub fn new(hash_size: usize, level: usize, selectivity: Selectivity) -> Self {
        Self {
            game: GameState::new(),
            search: search::Search::new(&SearchOptions {
                tt_mb_size: hash_size,
                ..Default::default()
            }),
            level,
            selectivity,
            name: "Neural Reversi".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Runs the main GTP command loop.
    ///
    /// This method reads commands from stdin, processes them, and writes
    /// responses to stdout according to the GTP protocol. The loop continues
    /// until a `quit` command is received or an I/O error occurs.
    ///
    /// # Protocol Details
    /// - Commands may be prefixed with an optional ID number
    /// - Empty lines and lines starting with '#' are ignored
    /// - Responses are formatted with '=' for success or '?' for errors
    /// - Each response is followed by a blank line
    pub fn run(&mut self) {
        reversi_core::init();

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            match line {
                Ok(input) => {
                    let input = input.trim();
                    if input.is_empty() || input.starts_with('#') {
                        continue;
                    }

                    let (id, cmd, args) = self.parse_input_line(input);
                    if cmd.is_empty() {
                        continue;
                    }

                    let command = Command::from_str_with_args(cmd, &args);
                    let is_quit = matches!(command, Command::Quit);
                    let response = self.handle_command(command);

                    if let Err(e) = self.output_response(&mut stdout, id, &response) {
                        eprintln!("Error writing output: {}", e);
                        break;
                    }

                    if is_quit {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    break;
                }
            }
        }
    }

    /// Parses a GTP input line into command ID, command name, and arguments.
    ///
    /// # Arguments
    /// * `input` - The raw input line from stdin
    ///
    /// # Returns
    /// A tuple containing:
    /// - Optional command ID (if present)
    /// - Command name as string slice
    /// - Vector of argument string slices
    fn parse_input_line<'a>(&self, input: &'a str) -> (Option<usize>, &'a str, Vec<&'a str>) {
        let (id, command) = self.extract_id_and_command(input);
        self.split_command_and_args(id, command)
    }

    /// Extracts the optional command ID and command portion from input.
    ///
    /// GTP commands may optionally start with a numeric ID followed by whitespace.
    /// This method separates the ID (if present) from the actual command.
    fn extract_id_and_command<'a>(&self, input: &'a str) -> (Option<usize>, &'a str) {
        if let Some(idx) = input.find(|c: char| c.is_whitespace()) {
            let (id_str, rest) = input.split_at(idx);
            if let Ok(id) = id_str.parse::<usize>() {
                (Some(id), rest.trim_start())
            } else {
                (None, input)
            }
        } else {
            (None, input)
        }
    }

    /// Splits the command portion into command name and arguments.
    ///
    /// # Arguments
    /// * `id` - Optional command ID from previous parsing step
    /// * `command` - Command string containing command name and arguments
    ///
    /// # Returns
    /// Tuple of (id, command_name, arguments)
    fn split_command_and_args<'a>(&self, id: Option<usize>, command: &'a str) -> (Option<usize>, &'a str, Vec<&'a str>) {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return (id, "", Vec::new());
        }

        let cmd = parts[0];
        let args = parts[1..].to_vec();

        (id, cmd, args)
    }

    /// Outputs a GTP response to stdout with proper formatting.
    ///
    /// Formats the response according to GTP protocol:
    /// - Includes command ID if present
    /// - Adds appropriate success/error prefix
    /// - Follows with double newline for protocol compliance
    ///
    /// # Arguments
    /// * `stdout` - Output stream to write to
    /// * `id` - Optional command ID to include in response
    /// * `response` - The response to output
    fn output_response(&self, stdout: &mut impl Write, id: Option<usize>, response: &GtpResponse) -> io::Result<()> {
        let response_str = response.to_string();
        match id {
            Some(id) => {
                if let GtpResponse::Success(_) = response {
                    write!(stdout, "={} {}\n\n", id, &response_str[2..])
                } else {
                    write!(stdout, "?{} {}\n\n", id, &response_str[2..])
                }
            }
            None => {
                writeln!(stdout, "{}\n", response_str)
            }
        }?;
        stdout.flush()
    }

    /// Processes a parsed command and returns the appropriate response.
    ///
    /// This is the main command dispatcher that routes commands to their
    /// specific handler methods.
    ///
    /// # Arguments
    /// * `command` - The parsed command to process
    ///
    /// # Returns
    /// A `GtpResponse` containing the result of command execution
    fn handle_command(&mut self, command: Command) -> GtpResponse {
        match command {
            Command::ProtocolVersion => self.handle_protocol_version(),
            Command::Name => self.handle_name(),
            Command::Version => self.handle_version(),
            Command::KnownCommand(cmd) => self.handle_known_command(&cmd),
            Command::ListCommands => self.handle_list_commands(),
            Command::Quit => self.handle_quit(),
            Command::Boardsize(size) => self.handle_boardsize(size),
            Command::ClearBoard => self.handle_clear_board(),
            Command::Play { color, move_str } => self.handle_play(&color, &move_str),
            Command::Genmove(color) => self.handle_genmove(&color),
            Command::Showboard => self.handle_showboard(),
            Command::Undo => self.handle_undo(),
            Command::SetLevel(level) => self.handle_set_level(level),
            Command::Unknown(cmd) => GtpResponse::Error(format!("unknown command: {}", cmd)),
        }
    }

    /// Handles the `protocol_version` command.
    ///
    /// Returns the GTP protocol version supported by this engine (always 2).
    fn handle_protocol_version(&self) -> GtpResponse {
        GtpResponse::Success("2".to_string())
    }

    /// Handles the `name` command.
    ///
    /// Returns the name of this engine.
    fn handle_name(&self) -> GtpResponse {
        GtpResponse::Success(self.name.clone())
    }

    /// Handles the `version` command.
    ///
    /// Returns the version string of this engine.
    fn handle_version(&self) -> GtpResponse {
        GtpResponse::Success(self.version.clone())
    }

    /// Handles the `known_command` command.
    ///
    /// Checks if the specified command is supported by this engine.
    ///
    /// # Arguments
    /// * `cmd` - The command name to check
    ///
    /// # Returns
    /// "true" if the command is supported, "false" otherwise
    fn handle_known_command(&self, cmd: &str) -> GtpResponse {
        let known = self.is_known_command(cmd);
        GtpResponse::Success(if known { "true" } else { "false" }.to_string())
    }

    /// Handles the `list_commands` command.
    ///
    /// Returns a newline-separated list of all supported commands.
    fn handle_list_commands(&self) -> GtpResponse {
        let list = COMMAND_NAMES.join("\n");
        GtpResponse::Success(list)
    }

    /// Handles the `quit` command.
    ///
    /// Returns a successful response. The main loop will terminate after
    /// processing this command.
    fn handle_quit(&self) -> GtpResponse {
        GtpResponse::Success("".to_string())
    }

    /// Handles the `boardsize` command.
    ///
    /// Currently only supports 8x8 boards as required for Reversi/Othello.
    ///
    /// # Arguments
    /// * `size` - The requested board size
    ///
    /// # Returns
    /// Success if size is 8, error otherwise
    fn handle_boardsize(&self, size: usize) -> GtpResponse {
        if size == 8 {
            GtpResponse::Success("".to_string())
        } else {
            GtpResponse::Error("unacceptable size (only 8x8 is supported)".to_string())
        }
    }

    /// Handles the `clear_board` command.
    ///
    /// Resets the game to the initial position and reinitializes the search engine.
    fn handle_clear_board(&mut self) -> GtpResponse {
        self.game = GameState::new();
        self.search.init();
        GtpResponse::Success("".to_string())
    }

    /// Handles the `play` command.
    ///
    /// Executes a move for the specified color. Validates that it's the correct
    /// player's turn and that the move is legal.
    ///
    /// # Arguments
    /// * `color` - The color making the move ("b"/"black" or "w"/"white")
    /// * `move_str` - The move in coordinate notation (e.g., "d3") or "pass"
    ///
    /// # Returns
    /// Success if the move was played, error if invalid
    fn handle_play(&mut self, color: &str, move_str: &str) -> GtpResponse {
        if move_str == "pass" {
            if !self.game.board.has_legal_moves() {
                self.game.make_pass();
                return GtpResponse::Success("".to_string());
            } else {
                return GtpResponse::Error("pass not allowed when legal moves exist".to_string());
            }
        }

        if let Err(msg) = self.validate_color(color) {
            return GtpResponse::Error(msg);
        }

        match move_str.parse::<Square>() {
            Ok(sq) => {
                if self.game.board.is_legal_move(sq) {
                    self.game.make_move(sq);
                    GtpResponse::Success("".to_string())
                } else {
                    GtpResponse::Error("illegal move".to_string())
                }
            }
            Err(_) => GtpResponse::Error("invalid move format (use a1, b2, etc.)".to_string()),
        }
    }

    /// Handles the `genmove` command.
    ///
    /// Generates and plays the best move for the specified color using the
    /// neural network search engine.
    ///
    /// # Arguments
    /// * `color` - The color to generate a move for
    ///
    /// # Returns
    /// The generated move in coordinate notation, or "pass" if no legal moves
    fn handle_genmove(&mut self, color: &str) -> GtpResponse {
        if let Err(msg) = self.validate_color(color) {
            return GtpResponse::Error(msg);
        }

        if !self.game.board.has_legal_moves() {
            self.game.make_pass();
            return GtpResponse::Success("pass".to_string());
        }

        let result = self.search.run(
            &self.game.board,
            level::get_level(self.level),
            self.selectivity,
            false,
        );

        if let Some(computer_move) = result.pv_line.first() {
            self.game.make_move(*computer_move);
            GtpResponse::Success(format!("{:?}", computer_move))
        } else {
            GtpResponse::Error("failed to generate move".to_string())
        }
    }

    /// Handles the `showboard` command.
    ///
    /// Returns a text representation of the current board state.
    fn handle_showboard(&self) -> GtpResponse {
        let board_display = self.game.get_board_string();
        GtpResponse::Success(format!("\n{}", board_display))
    }

    /// Handles the `undo` command.
    ///
    /// Undoes the last move if possible.
    ///
    /// # Returns
    /// Success if a move was undone, error if no moves to undo
    fn handle_undo(&mut self) -> GtpResponse {
        if self.game.undo() {
            GtpResponse::Success("".to_string())
        } else {
            GtpResponse::Error("cannot undo".to_string())
        }
    }

    /// Handles the `set_level` command.
    ///
    /// Sets the engine's playing strength level, which affects search depth
    /// and time allocation.
    ///
    /// # Arguments
    /// * `level` - The strength level (1-20, where 20 is strongest)
    ///
    /// # Returns
    /// Success if level is valid (1-20), error otherwise
    fn handle_set_level(&mut self, level: usize) -> GtpResponse {
        if level > 0 && level <= 20 {
            self.level = level;
            GtpResponse::Success("".to_string())
        } else {
            GtpResponse::Error("level must be between 1 and 20".to_string())
        }
    }

    /// Validates that the specified color matches the current player to move.
    ///
    /// Accepts multiple formats: "b", "black", "w", "white" (case insensitive).
    ///
    /// # Arguments
    /// * `color` - The color string to validate
    ///
    /// # Returns
    /// `Ok(())` if valid, `Err(message)` if invalid or wrong player
    fn validate_color(&self, color: &str) -> Result<(), String> {
        let expected_color = match self.game.get_side_to_move() {
            Piece::Black => "b",
            Piece::White => "w",
            _ => unreachable!(),
        };

        if color != "b" && color != "w" && color != "black" && color != "white" {
            return Err("invalid color (must be 'b', 'w', 'black', or 'white')".to_string());
        }

        let is_correct_color = (color == "b" || color == "black") && expected_color == "b" ||
                              (color == "w" || color == "white") && expected_color == "w";

        if !is_correct_color {
            return Err(format!("wrong color, expected {}", expected_color));
        }

        Ok(())
    }

    /// Checks if a command name is in the list of supported commands.
    ///
    /// # Arguments
    /// * `cmd` - The command name to check
    ///
    /// # Returns
    /// `true` if the command is supported, `false` otherwise
    fn is_known_command(&self, cmd: &str) -> bool {
        COMMAND_NAMES.contains(&cmd)
    }
}
