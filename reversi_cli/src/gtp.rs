use reversi_core::{
    level,
    piece::Piece,
    search::{self, SearchOptions, SearchProgress},
    square::Square, types::Selectivity,
};

use crate::game::GameState;
use std::io::{self, BufRead, Write};
use std::env;

const COMMANDS: &[&str] = &[
    "protocol_version", "name", "version", "known_command", "list_commands",
    "quit", "boardsize", "clear_board", "play", "genmove", "showboard",
    "undo", "set_level",
];

pub enum GtpResponse {
    Success(String),
    Error(String),
}

impl std::fmt::Display for GtpResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success(msg) => write!(f, "= {}", msg),
            Self::Error(msg) => write!(f, "? {}", msg),
        }
    }
}

pub struct GtpEngine {
    game: GameState,
    search: search::Search,
    level: usize,
    selectivity: Selectivity,
    name: String,
    version: String,
}

impl GtpEngine {
    pub fn new(level: usize, selectivity: Selectivity) -> Self {
        Self {
            game: GameState::new(),
            search: search::Search::new(&SearchOptions::default()),
            level,
            selectivity,
            name: "Neural Reversi".to_string(),
            version: env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.1".to_string()),
        }
    }

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

                    let response = self.handle_command(cmd, &args);

                    if let Err(e) = self.output_response(&mut stdout, id, &response) {
                        eprintln!("Error writing output: {}", e);
                        break;
                    }

                    if cmd == "quit" {
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

    fn parse_input_line<'a>(&self, input: &'a str) -> (Option<usize>, &'a str, Vec<&'a str>) {
        let (id, command) = if let Some(idx) = input.find(|c: char| c.is_whitespace()) {
            let (id_str, rest) = input.split_at(idx);
            if let Ok(id) = id_str.parse::<usize>() {
                (Some(id), rest.trim_start())
            } else {
                (None, input)
            }
        } else {
            (None, input)
        };

        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return (id, "", Vec::new());
        }

        let cmd = parts[0];
        let args = parts[1..].to_vec();

        (id, cmd, args)
    }

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

    fn handle_command(&mut self, cmd: &str, args: &[&str]) -> GtpResponse {
        match cmd {
            "protocol_version" => GtpResponse::Success("2".to_string()),

            "name" => GtpResponse::Success(self.name.clone()),

            "version" => GtpResponse::Success(self.version.clone()),

            "known_command" => {
                let known = if args.len() == 1 {
                    self.is_known_command(args[0])
                } else {
                    false
                };
                GtpResponse::Success(if known { "true" } else { "false" }.to_string())
            }

            "list_commands" => {
                let list = COMMANDS.join("\n");
                GtpResponse::Success(list)
            }

            "quit" => GtpResponse::Success("".to_string()),

            "boardsize" => {
                if args.len() != 1 {
                    return GtpResponse::Error("boardsize requires exactly one argument".to_string());
                }

                if let Ok(size) = args[0].parse::<usize>() {
                    if size == 8 {
                        GtpResponse::Success("".to_string())
                    } else {
                        GtpResponse::Error("unacceptable size (only 8x8 is supported)".to_string())
                    }
                } else {
                    GtpResponse::Error("argument is not a valid number".to_string())
                }
            }

            "clear_board" => {
                self.game = GameState::new();
                self.search.init();
                GtpResponse::Success("".to_string())
            }

            "play" => {
                if args.len() != 2 {
                    return GtpResponse::Error("play requires exactly two arguments: color and move".to_string());
                }

                let color = args[0].to_lowercase();
                let move_str = args[1].to_lowercase();

                if move_str == "pass" {
                    if !self.game.board.has_legal_moves() {
                        self.game.make_pass();
                        return GtpResponse::Success("".to_string());
                    } else {
                        return GtpResponse::Error("pass not allowed when legal moves exist".to_string());
                    }
                }

                let expected_color = match self.game.get_side_to_move() {
                    Piece::Black => "b",
                    Piece::White => "w",
                    _ => unreachable!(),
                };

                if color != "b" && color != "w" && color != "black" && color != "white" {
                    return GtpResponse::Error("invalid color (must be 'b', 'w', 'black', or 'white')".to_string());
                }

                let is_correct_color = (color == "b" || color == "black") && expected_color == "b" ||
                                      (color == "w" || color == "white") && expected_color == "w";

                if !is_correct_color {
                    return GtpResponse::Error(format!("wrong color, expected {}", expected_color));
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

            "genmove" => {
                if args.len() != 1 {
                    return GtpResponse::Error("genmove requires exactly one argument: color".to_string());
                }

                let color = args[0].to_lowercase();

                let expected_color = match self.game.get_side_to_move() {
                    Piece::Black => "b",
                    Piece::White => "w",
                    _ => unreachable!(),
                };

                if color != "b" && color != "w" && color != "black" && color != "white" {
                    return GtpResponse::Error("invalid color (must be 'b', 'w', 'black', or 'white')".to_string());
                }

                let is_correct_color = (color == "b" || color == "black") && expected_color == "b" ||
                                      (color == "w" || color == "white") && expected_color == "w";

                if !is_correct_color {
                    return GtpResponse::Error(format!("wrong color, expected {}", expected_color));
                }

                if !self.game.board.has_legal_moves() {
                    self.game.make_pass();
                    return GtpResponse::Success("pass".to_string());
                }

                let result = self.search.run(
                    &self.game.board,
                    level::get_level(self.level),
                    self.selectivity,
                    None::<fn(SearchProgress)>,
                );

                if let Some(computer_move) = result.pv_line.first() {
                    self.game.make_move(*computer_move);
                    GtpResponse::Success(format!("{:?}", computer_move))
                } else {
                    GtpResponse::Error("failed to generate move".to_string())
                }
            }

            "showboard" => {
                let board_display = self.game.get_board_string();
                GtpResponse::Success(format!("\n{}", board_display))
            }

            "undo" => {
                if self.game.undo() {
                    GtpResponse::Success("".to_string())
                } else {
                    GtpResponse::Error("cannot undo".to_string())
                }
            }

            "set_level" => {
                if args.len() != 1 {
                    return GtpResponse::Error("set_level requires exactly one argument".to_string());
                }

                if let Ok(lvl) = args[0].parse::<usize>() {
                    if lvl > 0 && lvl <= 20 {
                        self.level = lvl;
                        GtpResponse::Success("".to_string())
                    } else {
                        GtpResponse::Error("level must be between 1 and 20".to_string())
                    }
                } else {
                    GtpResponse::Error("argument is not a valid number".to_string())
                }
            }

            _ => GtpResponse::Error(format!("unknown command: {}", cmd)),
        }
    }

    fn is_known_command(&self, cmd: &str) -> bool {
        COMMANDS.contains(&cmd)
    }
}
