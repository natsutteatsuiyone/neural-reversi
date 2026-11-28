//! GTP (Go Text Protocol) implementation for Neural Reversi.
//!
//! This module provides a GTP-compliant interface for the Neural Reversi engine,
//! allowing it to communicate with GTP-compatible GUI applications and tournament
//! management software. While originally designed for Go, GTP is also used for
//! other board games including Reversi/Othello.
//!
//! The implementation supports all essential GTP commands for game play and
//! engine configuration, including moves, board setup, and engine information.

use std::fmt::Write as _;
use std::io::{self, BufRead, Write};

use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::engine::{Engine, GameId};
use reversi_core::game_state::GameState;
use reversi_core::level::{Level, MAX_LEVEL};
use reversi_core::player_color::PlayerColor;
use reversi_core::search::SearchRunOptions;
use reversi_core::search::time_control::{TimeControlMode, TimeSettings};
use reversi_core::square::Square;

use crate::config::EngineConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    ProtocolVersion,
    Name,
    Version,
    #[allow(clippy::enum_variant_names)]
    KnownCommand(String),
    ListCommands,
    Quit,
    Boardsize(usize),
    ClearBoard,
    SetBoard {
        board: String,
        side: PlayerColor,
    },
    Play {
        color: PlayerColor,
        move_str: String,
    },
    Genmove(PlayerColor),
    Showboard,
    Undo,
    SetLevel(usize),
    TimeSettings(TimeSettings),
    TimeLeft {
        color: PlayerColor,
        time: u32,
        stones: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommandParseError {
    UnknownOrMalformed(String),
    InvalidColor,
    InvalidSideToMove,
}

fn parse_setboard_side(side: &str) -> Option<PlayerColor> {
    let side = side.trim();
    if side.eq_ignore_ascii_case("x") {
        Some(PlayerColor::Black)
    } else if side.eq_ignore_ascii_case("o") {
        Some(PlayerColor::White)
    } else {
        side.parse().ok()
    }
}

impl Command {
    fn from_str_with_args(cmd: &str, args: &[&str]) -> Result<Self, CommandParseError> {
        let unknown = || CommandParseError::UnknownOrMalformed(cmd.to_string());
        match (cmd, args) {
            ("protocol_version", _) => Ok(Self::ProtocolVersion),
            ("name", _) => Ok(Self::Name),
            ("version", _) => Ok(Self::Version),
            ("known_command", [name]) => Ok(Self::KnownCommand((*name).to_string())),
            ("list_commands", _) => Ok(Self::ListCommands),
            ("quit", _) => Ok(Self::Quit),
            ("boardsize", [size]) => size.parse().map(Self::Boardsize).map_err(|_| unknown()),
            ("clear_board", _) => Ok(Self::ClearBoard),
            ("setboard", [board, side]) => Ok(Self::SetBoard {
                board: (*board).to_string(),
                side: parse_setboard_side(side).ok_or(CommandParseError::InvalidSideToMove)?,
            }),
            ("play", [color, mv]) => Ok(Self::Play {
                color: color.parse().map_err(|_| CommandParseError::InvalidColor)?,
                move_str: mv.to_ascii_lowercase(),
            }),
            ("genmove", [color]) => color
                .parse()
                .map(Self::Genmove)
                .map_err(|_| CommandParseError::InvalidColor),
            ("showboard", _) => Ok(Self::Showboard),
            ("undo", _) => Ok(Self::Undo),
            ("set_level", [level]) => level.parse().map(Self::SetLevel).map_err(|_| unknown()),
            ("time_settings", [main_time, byoyomi_time, byoyomi_stones]) => {
                let main_time: u32 = main_time.parse().map_err(|_| unknown())?;
                let byoyomi_time: u32 = byoyomi_time.parse().map_err(|_| unknown())?;
                let byoyomi_stones: u32 = byoyomi_stones.parse().map_err(|_| unknown())?;
                Ok(Self::TimeSettings(TimeSettings::from_gtp_secs(
                    u64::from(main_time),
                    u64::from(byoyomi_time),
                    byoyomi_stones,
                )))
            }
            ("time_left", [color, time, stones]) => {
                let time: u32 = time.parse().map_err(|_| unknown())?;
                let stones: u32 = stones.parse().map_err(|_| unknown())?;
                let color = color.parse().map_err(|_| CommandParseError::InvalidColor)?;
                Ok(Self::TimeLeft {
                    color,
                    time,
                    stones,
                })
            }
            _ => Err(unknown()),
        }
    }
}

/// List of all supported GTP command names.
/// Used for the `list_commands` response and command validation.
const COMMAND_NAMES: &[&str] = &[
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "clear_board",
    "setboard",
    "play",
    "genmove",
    "showboard",
    "undo",
    "set_level",
    "time_settings",
    "time_left",
];

enum GtpResponse {
    Success(String),
    Error(String),
}

impl From<CommandParseError> for GtpResponse {
    fn from(err: CommandParseError) -> Self {
        let msg = match err {
            CommandParseError::UnknownOrMalformed(cmd) => format!("unknown command: {cmd}"),
            CommandParseError::InvalidColor => {
                "invalid color (must be 'b', 'w', 'black', or 'white')".to_string()
            }
            CommandParseError::InvalidSideToMove => "invalid side to move".to_string(),
        };
        Self::Error(msg)
    }
}

impl std::fmt::Display for GtpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success(msg) => write!(f, "= {msg}"),
            Self::Error(msg) => write!(f, "? {msg}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PlayerClock {
    remaining_time_ms: u64,
    byo_stones_left: u32,
}

/// GTP engine that processes commands and owns game and search state.
pub struct GtpEngine {
    game: GameState,
    engine: Engine,
    search: GameId,
    level: usize,
    name: String,
    version: String,
    time_settings: TimeSettings,
    clocks: [Option<PlayerClock>; 2],
}

impl GtpEngine {
    /// Creates a new GTP engine with the specified configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if a configured weight file cannot be loaded.
    pub fn new(config: &EngineConfig) -> Result<Self, String> {
        Ok(Self::from_engine(config.engine()?, config.level))
    }

    fn from_engine(engine: Engine, level: usize) -> Self {
        Self {
            game: GameState::new(),
            search: engine.open(),
            engine,
            level,
            name: "Neural Reversi".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            time_settings: TimeSettings::Infinite,
            clocks: [None; 2],
        }
    }

    /// Runs the main GTP command loop over stdin and stdout until `quit` or EOF.
    pub fn run(&mut self) {
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

                    let (is_quit, response) = match Command::from_str_with_args(cmd, &args) {
                        Ok(command) => (
                            matches!(command, Command::Quit),
                            self.handle_command(command),
                        ),
                        Err(err) => (false, GtpResponse::from(err)),
                    };

                    if let Err(e) = self.output_response(&mut stdout, id, &response) {
                        eprintln!("Error writing output: {e}");
                        break;
                    }

                    if is_quit {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Error reading input: {e}");
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
    fn split_command_and_args<'a>(
        &self,
        id: Option<usize>,
        command: &'a str,
    ) -> (Option<usize>, &'a str, Vec<&'a str>) {
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
    fn output_response(
        &self,
        stdout: &mut impl Write,
        id: Option<usize>,
        response: &GtpResponse,
    ) -> io::Result<()> {
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
                writeln!(stdout, "{response_str}\n")
            }
        }?;
        stdout.flush()
    }

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
            Command::SetBoard { board, side } => self.handle_setboard(&board, side),
            Command::Play { color, move_str } => self.handle_play(color, &move_str),
            Command::Genmove(color) => self.handle_genmove(color),
            Command::Showboard => self.handle_showboard(),
            Command::Undo => self.handle_undo(),
            Command::SetLevel(level) => self.handle_set_level(level),
            Command::TimeSettings(settings) => self.handle_time_settings(settings),
            Command::TimeLeft {
                color,
                time,
                stones,
            } => self.handle_time_left(color, time, stones),
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
        self.clocks = [None; 2];
        self.engine.reset(self.search);
        GtpResponse::Success("".to_string())
    }

    fn handle_setboard(&mut self, text: &str, side: PlayerColor) -> GtpResponse {
        match Board::from_string(text, side) {
            // The search assumes at most 60 empties.
            Ok(board) if board.player_count() + board.opponent_count() < 4 => {
                GtpResponse::Error("at least 4 discs required".to_string())
            }
            Ok(board) => {
                self.game = GameState::from_board(board, side);
                self.engine.reset(self.search);
                GtpResponse::Success(String::new())
            }
            Err(error) => GtpResponse::Error(error.to_string()),
        }
    }

    fn handle_play(&mut self, color: PlayerColor, move_str: &str) -> GtpResponse {
        if move_str == "pass" && self.game.confirm_automatic_pass(color) {
            return GtpResponse::Success(String::new());
        }
        if let Err(msg) = self.ensure_turn(color) {
            return GtpResponse::Error(msg);
        }

        if move_str == "pass" {
            if self.game.board().has_legal_moves() {
                return GtpResponse::Error("pass not allowed when legal moves exist".to_string());
            }
            return match self.game.make_pass() {
                Ok(()) => GtpResponse::Success(String::new()),
                Err(e) => GtpResponse::Error(e.to_string()),
            };
        }

        match move_str.parse::<Square>() {
            Ok(sq) => match self.game.make_move(sq) {
                Ok(()) => GtpResponse::Success(String::new()),
                Err(_) => GtpResponse::Error("illegal move".to_string()),
            },
            Err(_) => GtpResponse::Error("invalid move format (use a1, b2, etc.)".to_string()),
        }
    }

    fn handle_genmove(&mut self, color: PlayerColor) -> GtpResponse {
        if self.game.confirm_automatic_pass(color) {
            return GtpResponse::Success("pass".to_string());
        }
        if let Err(msg) = self.ensure_turn(color) {
            return GtpResponse::Error(msg);
        }

        if !self.game.board().has_legal_moves() {
            return match self.game.make_pass() {
                Ok(()) => GtpResponse::Success("pass".to_string()),
                Err(e) => GtpResponse::Error(e.to_string()),
            };
        }

        let options = match self.current_time_control() {
            TimeControlMode::Infinite => {
                let level_idx = self.level.min(MAX_LEVEL);
                SearchRunOptions::with_level(Level::preset(level_idx))
            }
            mode => SearchRunOptions::with_time(mode),
        };
        let result = self.engine.search(self.search, self.game.board(), &options);

        if let Some(computer_move) = result.best_move() {
            self.game
                .make_move(computer_move)
                .expect("search returned an illegal move");
            GtpResponse::Success(format!("{computer_move}"))
        } else {
            GtpResponse::Error("failed to generate move".to_string())
        }
    }

    fn current_time_control(&self) -> TimeControlMode {
        let side = self.game.side_to_move();
        match self.clocks[side.index()] {
            Some(clock) => self
                .time_settings
                .move_time_control(clock.remaining_time_ms, clock.byo_stones_left),
            None => self.time_settings.initial_time_control(),
        }
    }

    /// Handles the `showboard` command.
    ///
    /// Returns a text representation of the current board state.
    fn handle_showboard(&self) -> GtpResponse {
        let mut result = String::from("\n");
        let board = self.game.board();
        let side_to_move = self.game.side_to_move();
        let (black_count, white_count) = self.game.score();

        // Header
        result.push_str("   a b c d e f g h\n");
        result.push_str("  +-+-+-+-+-+-+-+-+\n");

        // Board rows
        for y in 0..8 {
            let _ = write!(result, "{} |", y + 1);

            for x in 0..8 {
                let sq = Square::from_file_rank(x as u8, y as u8);
                let piece = board.disc_at(sq, side_to_move);
                let symbol = match piece {
                    Disc::Black => "X",
                    Disc::White => "O",
                    Disc::Empty => {
                        if board.is_legal_move(sq) {
                            "."
                        } else {
                            " "
                        }
                    }
                };
                let _ = write!(result, "{symbol}|");
            }

            // Side information
            match y {
                0 => {
                    let _ = write!(
                        result,
                        " {}'s turn",
                        if side_to_move == PlayerColor::Black {
                            "Black(X)"
                        } else {
                            "White(O)"
                        }
                    );
                }
                1 => {
                    let _ = write!(result, " Black: {black_count}");
                }
                2 => {
                    let _ = write!(result, " White: {white_count}");
                }
                _ => {}
            }

            result.push('\n');
            if y < 7 {
                result.push_str("  +-+-+-+-+-+-+-+-+\n");
            }
        }

        // Footer
        result.push_str("  +-+-+-+-+-+-+-+-+\n");
        GtpResponse::Success(result)
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
    /// Sets the engine's playing strength level, which sets the search depth
    /// when no time control is set.
    ///
    /// # Arguments
    /// * `level` - The strength level (1 to [`MAX_LEVEL`])
    ///
    /// # Returns
    /// Success if level is valid, error otherwise
    fn handle_set_level(&mut self, level: usize) -> GtpResponse {
        if level > 0 && level <= MAX_LEVEL {
            self.level = level;
            GtpResponse::Success("".to_string())
        } else {
            GtpResponse::Error(format!("level must be between 1 and {MAX_LEVEL}"))
        }
    }

    fn ensure_turn(&self, color: PlayerColor) -> Result<(), String> {
        let expected = self.game.side_to_move();
        if color == expected {
            Ok(())
        } else {
            let short = match expected {
                PlayerColor::Black => 'b',
                PlayerColor::White => 'w',
            };
            Err(format!("wrong color, expected {short}"))
        }
    }

    fn handle_time_settings(&mut self, settings: TimeSettings) -> GtpResponse {
        self.time_settings = settings;
        self.clocks = [None; 2];
        GtpResponse::Success(String::new())
    }

    fn handle_time_left(&mut self, color: PlayerColor, time: u32, stones: u32) -> GtpResponse {
        self.clocks[color.index()] = Some(PlayerClock {
            remaining_time_ms: u64::from(time) * 1000,
            byo_stones_left: stones,
        });
        GtpResponse::Success(String::new())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setboard_preserves_absolute_colors_and_rejects_invalid_input_atomically() {
        let mut engine = test_engine();
        // White (b1) has no move; Black can play c1.
        let text = format!("XO------X-------X-------{}", "-".repeat(40));
        assert!(matches!(
            engine.handle_setboard(&text, PlayerColor::White),
            GtpResponse::Success(_)
        ));
        assert_eq!(engine.game.side_to_move(), PlayerColor::White);
        assert_eq!(
            engine.game.board().disc_at(Square::A1, PlayerColor::White),
            Disc::Black
        );
        assert_eq!(
            engine.game.board().disc_at(Square::B1, PlayerColor::White),
            Disc::White
        );
        let before = *engine.game.board();
        assert_eq!(
            Command::from_str_with_args("setboard", &[text.as_str(), "invalid"]),
            Err(CommandParseError::InvalidSideToMove)
        );
        assert_eq!(
            exec_line(&mut engine, &format!("setboard {text} invalid")),
            "? invalid side to move\n\n"
        );
        assert_eq!(*engine.game.board(), before);
        assert_eq!(engine.game.side_to_move(), PlayerColor::White);
        for board in ["X", &"?".repeat(64), &format!("XO{}", "-".repeat(62))] {
            assert!(matches!(
                engine.handle_setboard(board, PlayerColor::Black),
                GtpResponse::Error(_)
            ));
            assert_eq!(*engine.game.board(), before);
            assert_eq!(engine.game.side_to_move(), PlayerColor::White);
        }
        assert!(engine.is_known_command("setboard"));
        assert!(matches!(
            engine.handle_play(PlayerColor::White, "C1"),
            GtpResponse::Error(_)
        ));
        assert!(matches!(
            engine.handle_play(PlayerColor::White, "pass"),
            GtpResponse::Success(_)
        ));
        assert!(matches!(
            engine.handle_play(PlayerColor::Black, "C1"),
            GtpResponse::Success(_)
        ));
    }

    #[test]
    fn play_pass_and_genmove_confirm_an_automatic_pass() {
        let mut engine = test_engine();
        // Black a1 leaves White without a move while Black can still play g1.
        let text = format!("-OXOOO--{}", "-".repeat(56));
        assert!(matches!(
            engine.handle_setboard(&text, PlayerColor::Black),
            GtpResponse::Success(_)
        ));
        assert!(matches!(
            engine.handle_play(PlayerColor::Black, "a1"),
            GtpResponse::Success(_)
        ));
        assert!(matches!(
            engine.handle_play(PlayerColor::White, "g1"),
            GtpResponse::Error(_)
        ));
        assert!(matches!(
            engine.handle_play(PlayerColor::White, "pass"),
            GtpResponse::Success(_)
        ));
        assert!(matches!(
            engine.handle_play(PlayerColor::White, "pass"),
            GtpResponse::Error(_)
        ));
        assert!(matches!(engine.handle_undo(), GtpResponse::Success(_)));
        assert_eq!(engine.game.side_to_move(), PlayerColor::White);
        assert!(matches!(
            engine.handle_play(PlayerColor::Black, "g1"),
            GtpResponse::Error(_)
        ));
        assert!(matches!(
            engine.handle_genmove(PlayerColor::White),
            GtpResponse::Success(mv) if mv == "pass"
        ));
        assert!(matches!(engine.handle_undo(), GtpResponse::Success(_)));
        assert!(matches!(engine.handle_undo(), GtpResponse::Success(_)));
        assert!(engine.game.move_history().is_empty());
        assert!(matches!(
            engine.handle_play(PlayerColor::Black, "a1"),
            GtpResponse::Success(_)
        ));
        assert!(matches!(
            engine.handle_genmove(PlayerColor::White),
            GtpResponse::Success(mv) if mv == "pass"
        ));
        assert!(matches!(engine.handle_undo(), GtpResponse::Success(_)));
        assert_eq!(engine.game.side_to_move(), PlayerColor::White);
    }

    #[test]
    fn genmove_and_pass_on_finished_game_return_errors() {
        let mut engine = test_engine();
        let full = format!("{}O", "X".repeat(63));
        assert!(matches!(
            engine.handle_setboard(&full, PlayerColor::Black),
            GtpResponse::Success(_)
        ));
        assert!(matches!(
            engine.handle_genmove(PlayerColor::Black),
            GtpResponse::Error(_)
        ));
        assert!(matches!(
            engine.handle_play(PlayerColor::Black, "pass"),
            GtpResponse::Error(_)
        ));
    }

    #[test]
    fn parses_zero_arg_commands() {
        assert_eq!(
            Command::from_str_with_args("protocol_version", &[]),
            Ok(Command::ProtocolVersion)
        );
        assert_eq!(
            Command::from_str_with_args("list_commands", &[]),
            Ok(Command::ListCommands)
        );
        assert_eq!(Command::from_str_with_args("quit", &[]), Ok(Command::Quit));
    }

    #[test]
    fn parses_boardsize_and_rejects_bad_args() {
        assert_eq!(
            Command::from_str_with_args("boardsize", &["8"]),
            Ok(Command::Boardsize(8))
        );
        assert_eq!(
            Command::from_str_with_args("boardsize", &["abc"]),
            Err(CommandParseError::UnknownOrMalformed(
                "boardsize".to_string()
            ))
        );
        assert_eq!(
            Command::from_str_with_args("boardsize", &[]),
            Err(CommandParseError::UnknownOrMalformed(
                "boardsize".to_string()
            ))
        );
    }

    #[test]
    fn parses_play_and_lowercases_args() {
        assert_eq!(
            Command::from_str_with_args("play", &["B", "D3"]),
            Ok(Command::Play {
                color: PlayerColor::Black,
                move_str: "d3".to_string(),
            })
        );
        assert_eq!(
            Command::from_str_with_args("play", &["b"]),
            Err(CommandParseError::UnknownOrMalformed("play".to_string()))
        );
        assert_eq!(
            Command::from_str_with_args("play", &["x", "d3"]),
            Err(CommandParseError::InvalidColor)
        );
    }

    #[test]
    fn parses_set_level() {
        assert_eq!(
            Command::from_str_with_args("set_level", &["10"]),
            Ok(Command::SetLevel(10))
        );
        assert_eq!(
            Command::from_str_with_args("set_level", &["abc"]),
            Err(CommandParseError::UnknownOrMalformed(
                "set_level".to_string()
            ))
        );
    }

    #[test]
    fn parses_time_commands() {
        assert_eq!(
            Command::from_str_with_args("time_settings", &["300", "5", "1"]),
            Ok(Command::TimeSettings(TimeSettings::from_gtp_secs(
                300, 5, 1
            )))
        );
        assert_eq!(
            Command::from_str_with_args("time_settings", &["300", "5", "x"]),
            Err(CommandParseError::UnknownOrMalformed(
                "time_settings".to_string()
            ))
        );
        assert_eq!(
            Command::from_str_with_args("time_left", &["b", "60", "0"]),
            Ok(Command::TimeLeft {
                color: PlayerColor::Black,
                time: 60,
                stones: 0,
            })
        );
    }

    #[test]
    fn response_display_uses_gtp_prefixes() {
        assert_eq!(GtpResponse::Success("foo".to_string()).to_string(), "= foo");
        assert_eq!(GtpResponse::Error("bar".to_string()).to_string(), "? bar");
    }

    fn test_engine() -> GtpEngine {
        GtpEngine::from_engine(crate::config::test_engine(1, 1), 1)
    }

    fn exec_line(engine: &mut GtpEngine, line: &str) -> String {
        let (id, cmd, args) = engine.parse_input_line(line.trim());
        let response = match Command::from_str_with_args(cmd, &args) {
            Ok(command) => engine.handle_command(command),
            Err(err) => GtpResponse::from(err),
        };
        let mut out = Vec::new();
        engine.output_response(&mut out, id, &response).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn gtp_wire_contract_pins_command_arity_colors_and_error_precedence() {
        let mut engine = test_engine();

        assert_eq!(
            exec_line(&mut engine, "1 known_command genmove"),
            "=1 true\n\n"
        );
        assert_eq!(exec_line(&mut engine, "known_command foo"), "= false\n\n");
        assert_eq!(
            exec_line(&mut engine, "known_command"),
            "? unknown command: known_command\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "known_command a b"),
            "? unknown command: known_command\n\n"
        );
        assert_eq!(exec_line(&mut engine, "protocol_version extra"), "= 2\n\n");
        assert_eq!(exec_line(&mut engine, "name extra"), "= Neural Reversi\n\n");
        assert_eq!(
            exec_line(&mut engine, "NAME"),
            "? unknown command: NAME\n\n"
        );
        assert_eq!(exec_line(&mut engine, "42"), "? unknown command: 42\n\n");

        assert_eq!(
            exec_line(&mut engine, "boardsize abc"),
            "? unknown command: boardsize\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "boardsize 9"),
            "? unacceptable size (only 8x8 is supported)\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "set_level abc"),
            "? unknown command: set_level\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "set_level 0"),
            "? level must be between 1 and 30\n\n"
        );

        assert_eq!(
            exec_line(&mut engine, "play x"),
            "? unknown command: play\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "play b d3 extra"),
            "? unknown command: play\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "play x d3"),
            "? invalid color (must be 'b', 'w', 'black', or 'white')\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "play o d3"),
            "? invalid color (must be 'b', 'w', 'black', or 'white')\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "play x z9"),
            "? invalid color (must be 'b', 'w', 'black', or 'white')\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "play w z9"),
            "? wrong color, expected b\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "play b z9"),
            "? invalid move format (use a1, b2, etc.)\n\n"
        );
        assert_eq!(exec_line(&mut engine, "play BLACK D3"), "= \n\n");
        assert_eq!(
            exec_line(&mut engine, "play b c3"),
            "? wrong color, expected w\n\n"
        );

        assert_eq!(
            exec_line(&mut engine, "genmove"),
            "? unknown command: genmove\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "genmove w extra"),
            "? unknown command: genmove\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "genmove x"),
            "? invalid color (must be 'b', 'w', 'black', or 'white')\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "genmove black"),
            "? wrong color, expected w\n\n"
        );

        assert_eq!(
            exec_line(&mut engine, "time_left x 10 0"),
            "? invalid color (must be 'b', 'w', 'black', or 'white')\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "time_left x abc 0"),
            "? unknown command: time_left\n\n"
        );
        assert_eq!(exec_line(&mut engine, "time_left BLACK 45 0"), "= \n\n");
        assert_eq!(exec_line(&mut engine, "time_left WHITE 50 0"), "= \n\n");

        assert_eq!(
            exec_line(&mut engine, "setboard short invalid"),
            "? invalid side to move\n\n"
        );
        assert_eq!(
            exec_line(&mut engine, "setboard xo b"),
            "? Board string too short: expected 64 characters, got 2\n\n"
        );
        let start_board = format!("{}OX------XO{}", "-".repeat(27), "-".repeat(27));
        assert_eq!(
            exec_line(&mut engine, &format!("setboard {start_board} x")),
            "= \n\n"
        );
        assert_eq!(engine.game.side_to_move(), PlayerColor::Black);
        assert_eq!(
            exec_line(&mut engine, &format!("setboard {start_board} O")),
            "= \n\n"
        );
        assert_eq!(engine.game.side_to_move(), PlayerColor::White);
    }

    #[test]
    fn gtp_time_settings_and_time_left_pin_all_eight_mode_combinations() {
        let mut engine = test_engine();

        assert_eq!(exec_line(&mut engine, "time_settings 0 0 0"), "= \n\n");
        assert_eq!(engine.current_time_control(), TimeControlMode::Infinite);

        assert_eq!(exec_line(&mut engine, "time_settings 0 0 3"), "= \n\n");
        assert_eq!(engine.current_time_control(), TimeControlMode::Infinite);

        assert_eq!(exec_line(&mut engine, "time_settings 0 5 0"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Byoyomi {
                time_per_move_ms: 5_000,
            }
        );
        assert_eq!(exec_line(&mut engine, "time_left b 99 4"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Byoyomi {
                time_per_move_ms: 5_000,
            }
        );

        assert_eq!(exec_line(&mut engine, "time_settings 0 10 2"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms: 5_000,
            }
        );
        assert_eq!(exec_line(&mut engine, "time_left b 6 2"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms: 3_000,
            }
        );

        assert_eq!(exec_line(&mut engine, "time_settings 60 0 0"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 60_000,
                increment_ms: 0,
            }
        );

        assert_eq!(exec_line(&mut engine, "time_settings 60 0 3"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 60_000,
                increment_ms: 0,
            }
        );
        assert_eq!(exec_line(&mut engine, "time_left b 42 5"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 42_000,
                increment_ms: 0,
            }
        );

        assert_eq!(exec_line(&mut engine, "time_settings 60 5 0"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 60_000,
                increment_ms: 5_000,
            }
        );

        assert_eq!(exec_line(&mut engine, "time_settings 60 10 2"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::JapaneseByo {
                main_time_ms: 60_000,
                time_per_move_ms: 5_000,
            }
        );
        assert_eq!(exec_line(&mut engine, "time_left b 25 0"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::JapaneseByo {
                main_time_ms: 25_000,
                time_per_move_ms: 5_000,
            }
        );
        assert_eq!(exec_line(&mut engine, "time_left b 8 2"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms: 4_000,
            }
        );
    }

    #[test]
    fn clear_board_and_setboard_reset_player_clocks() {
        let mut engine = test_engine();
        assert_eq!(exec_line(&mut engine, "time_settings 60 5 0"), "= \n\n");
        assert_eq!(exec_line(&mut engine, "time_left b 12 0"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 12_000,
                increment_ms: 5_000,
            }
        );

        assert_eq!(exec_line(&mut engine, "clear_board"), "= \n\n");
        assert_eq!(
            engine.current_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 60_000,
                increment_ms: 5_000,
            }
        );
    }
}
