//! Game state management for the Neural Reversi CLI.
//!
//! This module provides the `GameState` struct which maintains the current
//! game position, move history, and handles game logic such as automatic
//! passing when no legal moves are available.
//!
//! The module supports both text-based and colored terminal display of
//! the game board, making it suitable for both human players and GTP
//! protocol communication.

use colored::Colorize;
use reversi_core::{board::Board, piece::Piece, square::Square};

/// Represents the state of a Reversi/Othello game.
pub struct GameState {
    /// The current board position
    pub board: Board,
    /// Which player's turn it is to move
    side_to_move: Piece,
    /// History of moves for undo functionality: (move, board_before_move)
    history: Vec<(Square, Board)>,
    /// The last move played (for highlighting in display)
    last_move: Square,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    /// Creates a new game in the initial position.
    ///
    /// The initial position has 4 pieces in the center (2 black, 2 white)
    /// with Black to move first, following standard Reversi rules.
    ///
    /// # Returns
    /// A new `GameState` in the starting position
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            side_to_move: Piece::Black,
            history: Vec::new(),
            last_move: Square::None,
        }
    }

    /// Creates a new game state from an existing board position.
    ///
    /// This is useful for setting up specific positions for analysis
    /// or continuing a game from a saved state.
    ///
    /// # Arguments
    /// * `board` - The board position to start from
    /// * `side_to_move` - Which player moves next
    ///
    /// # Returns
    /// A new `GameState` with the specified position
    pub fn from_board(board: Board, side_to_move: Piece) -> Self {
        Self {
            board,
            side_to_move,
            history: Vec::new(),
            last_move: Square::None,
        }
    }

    /// Executes a move and updates the game state.
    ///
    /// # Arguments
    /// * `sq` - The square to place a piece on
    ///
    /// # Panics
    /// Panics if the move is not legal on the current board
    pub fn make_move(&mut self, sq: Square) {
        if !self.board.is_legal_move(sq) {
            panic!("Attempted to make illegal move: {sq:?}");
        }

        self.history.push((sq, self.board));
        self.board = self.board.make_move(sq);
        self.last_move = sq;
        self.side_to_move = self.side_to_move.opposite();

        self.handle_automatic_passes();
    }

    /// Handles automatic pass moves when players have no legal moves.
    fn handle_automatic_passes(&mut self) {
        // If current player has no moves, they must pass
        if !self.board.has_legal_moves() {
            self.make_pass();

            // If the opponent also has no moves after the pass, pass again
            // This handles the case where both players are blocked
            if !self.board.has_legal_moves() {
                self.make_pass();
            }
        }
    }

    /// Executes a pass move (switching players without placing a piece).
    pub fn make_pass(&mut self) {
        self.board = self.board.switch_players();
        self.side_to_move = self.side_to_move.opposite();
    }

    /// Returns which player's turn it is to move.
    ///
    /// # Returns
    /// The `Piece` representing the current player (Black or White)
    pub fn get_side_to_move(&self) -> Piece {
        self.side_to_move
    }

    /// Returns a plain text representation of the board suitable for GTP.
    ///
    /// # Returns
    /// A string containing the text representation of the board
    pub fn get_board_string(&self) -> String {
        let mut result = String::new();

        // Header
        result.push_str("   a b c d e f g h\n");
        result.push_str("  +-+-+-+-+-+-+-+-+\n");

        // Board rows
        for y in 0..8 {
            result.push_str(&format!("{} |", y + 1));

            // Board squares
            for x in 0..8 {
                let sq = Square::from_usize_unchecked(y * 8 + x);
                let piece = self.board.get_piece_at(sq, self.side_to_move);
                let symbol = match piece {
                    Piece::Black => "X",
                    Piece::White => "O",
                    Piece::Empty => {
                        if self.board.is_legal_move(sq) {
                            "."
                        } else {
                            " "
                        }
                    }
                };
                result.push_str(&format!("{symbol}|"));
            }

            // Side information
            match y {
                0 => result.push_str(&format!(
                    " {}'s turn",
                    if self.side_to_move == Piece::Black {
                        "Black(X)"
                    } else {
                        "White(O)"
                    }
                )),
                1 => result.push_str(&format!(" Black: {}", self.get_black_count())),
                2 => result.push_str(&format!(" White: {}", self.get_white_count())),
                _ => {}
            }

            result.push('\n');
            if y < 7 {
                result.push_str("  +-+-+-+-+-+-+-+-+\n");
            }
        }

        // Footer
        result.push_str("  +-+-+-+-+-+-+-+-+\n");
        result
    }

    /// Undoes the last move if possible.
    ///
    /// # Returns
    /// `true` if a move was successfully undone, `false` if no moves to undo
    pub fn undo(&mut self) -> bool {
        match self.history.pop() {
            Some((_, prev_board)) => {
                self.board = prev_board;
                self.last_move = self.history.last().map_or(Square::None, |(sq, _)| *sq);
                self.side_to_move = self.side_to_move.opposite();
                true
            }
            None => false,
        }
    }

    /// Prints a colored representation of the board to the terminal.
    ///
    /// This is designed for human players using a terminal interface.
    pub fn print(&self) {
        // Header
        println!("      a   b   c   d   e   f   g   h");
        println!("    ┌───┬───┬───┬───┬───┬───┬───┬───┐");

        // Board rows
        for y in 0..8 {
            print!("  {} │", y + 1);

            // Board squares
            for x in 0..8 {
                let sq = Square::from_usize_unchecked(y * 8 + x);
                let piece = self.board.get_piece_at(sq, self.side_to_move);
                let is_legal = self.board.is_legal_move(sq);
                let is_last_move = sq == self.last_move;

                let symbol = match piece {
                    Piece::Black if is_last_move => " X ".on_bright_black().bright_green(),
                    Piece::White if is_last_move => " O ".on_bright_black().bright_yellow(),
                    Piece::Black => " X ".bright_green(),
                    Piece::White => " O ".bright_yellow(),
                    Piece::Empty if is_legal => " · ".bright_cyan(),
                    Piece::Empty => "   ".black(),
                };
                print!("{symbol}│");
            }

            // Side information
            match y {
                2 => {
                    let player_info = match self.side_to_move {
                        Piece::Black => "Black's turn (X)".bright_green(),
                        Piece::White => "White's turn (O)".bright_yellow(),
                        _ => unreachable!(),
                    };
                    println!("   {player_info}");
                }
                3 => println!(
                    "   Black: {}",
                    format!("{:2}", self.get_black_count()).bright_green()
                ),
                4 => println!(
                    "   White: {}",
                    format!("{:2}", self.get_white_count()).bright_yellow()
                ),
                6 => {
                    if self.board.is_game_over() {
                        let black_count = self.get_black_count();
                        let white_count = self.get_white_count();
                        match black_count.cmp(&white_count) {
                            std::cmp::Ordering::Greater => {
                                println!("   {}", "Black wins!".bright_green())
                            }
                            std::cmp::Ordering::Less => {
                                println!("   {}", "White wins!".bright_yellow())
                            }
                            std::cmp::Ordering::Equal => {
                                println!("  {}", "Draw".bright_cyan())
                            }
                        }
                    } else {
                        println!();
                    }
                }
                7 => {
                    if self.board.is_game_over() {
                        println!("   {}", "*** Game Over ***".bright_red());
                    } else {
                        println!();
                    }
                }
                _ => println!(),
            }

            if y < 7 {
                println!("    ├───┼───┼───┼───┼───┼───┼───┼───┤");
            }
        }

        // Footer
        println!("    └───┴───┴───┴───┴───┴───┴───┴───┘");
    }

    /// Returns the number of black pieces on the board.
    ///
    /// # Returns
    /// The number of black pieces on the board
    fn get_black_count(&self) -> u32 {
        if self.side_to_move == Piece::Black {
            self.board.get_player_count()
        } else {
            self.board.get_opponent_count()
        }
    }

    /// Returns the number of white pieces on the board.
    ///
    /// # Returns
    /// The number of white pieces on the board
    fn get_white_count(&self) -> u32 {
        if self.side_to_move == Piece::White {
            self.board.get_player_count()
        } else {
            self.board.get_opponent_count()
        }
    }
}
