//! Game state management for the Neural Reversi CLI.
//!
//! This module provides the `GameState` struct which wraps the core
//! game state and adds CLI-specific display capabilities.

use colored::Colorize;
use reversi_core::{board::Board, game_state, piece::Piece, square::Square};

/// Represents the state of a Reversi/Othello game with CLI-specific features.
///
/// This is a thin wrapper around the core `GameState` that adds
/// colored terminal display functionality.
pub struct GameState {
    /// Core game state with history and undo support
    core: game_state::GameState,
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
            core: game_state::GameState::new(),
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
    #[allow(dead_code)]
    pub fn from_board(board: Board, side_to_move: Piece) -> Self {
        Self {
            core: game_state::GameState::from_board(board, side_to_move),
        }
    }

    /// Returns a reference to the current board position.
    pub fn board(&self) -> &Board {
        self.core.board()
    }

    /// Executes a move and updates the game state.
    ///
    /// # Arguments
    /// * `sq` - The square to place a piece on
    ///
    /// # Panics
    /// Panics if the move is not legal on the current board
    pub fn make_move(&mut self, sq: Square) {
        self.core
            .make_move(sq)
            .expect("Attempted to make illegal move");
    }

    /// Executes a pass move (switching players without placing a piece).
    ///
    /// This is primarily used in GTP protocol where pass moves need to be
    /// explicitly managed.
    pub fn make_pass(&mut self) {
        let _ = self.core.make_pass();
    }

    /// Returns which player's turn it is to move.
    ///
    /// # Returns
    /// The `Piece` representing the current player (Black or White)
    pub fn get_side_to_move(&self) -> Piece {
        self.core.side_to_move()
    }

    /// Returns a plain text representation of the board suitable for GTP.
    ///
    /// # Returns
    /// A string containing the text representation of the board
    pub fn get_board_string(&self) -> String {
        let mut result = String::new();
        let board = self.core.board();
        let side_to_move = self.core.side_to_move();

        // Header
        result.push_str("   a b c d e f g h\n");
        result.push_str("  +-+-+-+-+-+-+-+-+\n");

        // Board rows
        for y in 0..8 {
            result.push_str(&format!("{} |", y + 1));

            // Board squares
            for x in 0..8 {
                let sq = Square::from_usize_unchecked(y * 8 + x);
                let piece = board.get_piece_at(sq, side_to_move);
                let symbol = match piece {
                    Piece::Black => "X",
                    Piece::White => "O",
                    Piece::Empty => {
                        if board.is_legal_move(sq) {
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
                    if side_to_move == Piece::Black {
                        "Black(X)"
                    } else {
                        "White(O)"
                    }
                )),
                1 => {
                    let (black_count, _) = self.core.get_score();
                    result.push_str(&format!(" Black: {black_count}"));
                }
                2 => {
                    let (_, white_count) = self.core.get_score();
                    result.push_str(&format!(" White: {white_count}"));
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
        result
    }

    /// Undoes the last move if possible.
    ///
    /// # Returns
    /// `true` if a move was successfully undone, `false` if no moves to undo
    pub fn undo(&mut self) -> bool {
        self.core.undo()
    }

    /// Returns the last move played.
    ///
    /// # Returns
    /// `Some(Square)` if a regular move was played, `None` if the last move was a pass
    /// or if no moves have been played yet
    pub fn last_move(&self) -> Option<Square> {
        self.core.last_move()
    }

    /// Returns the disc count for both players.
    ///
    /// # Returns
    /// A tuple `(black_count, white_count)` representing the number of
    /// discs each player has on the board
    pub fn get_score(&self) -> (u32, u32) {
        self.core.get_score()
    }

    /// Checks if the game has ended.
    ///
    /// # Returns
    /// `true` if the game is over, `false` otherwise
    #[allow(dead_code)]
    pub fn is_game_over(&self) -> bool {
        self.core.is_game_over()
    }

    /// Returns the move history as a list of squares.
    ///
    /// # Returns
    /// A vector of moves played (excluding passes)
    pub fn get_move_history(&self) -> Vec<Square> {
        self.core
            .move_history()
            .iter()
            .filter_map(|(sq, _, _)| *sq)
            .collect()
    }

    /// Prints a colored representation of the board to the terminal.
    ///
    /// This is designed for human players using a terminal interface.
    #[allow(dead_code)]
    pub fn print(&self) {
        let board = self.core.board();
        let side_to_move = self.core.side_to_move();
        let last_move = self.core.last_move();

        // Header
        println!("      a   b   c   d   e   f   g   h");
        println!("    ┌───┬───┬───┬───┬───┬───┬───┬───┐");

        // Board rows
        for y in 0..8 {
            print!("  {} │", y + 1);

            // Board squares
            for x in 0..8 {
                let sq = Square::from_usize_unchecked(y * 8 + x);
                let piece = board.get_piece_at(sq, side_to_move);
                let is_legal = board.is_legal_move(sq);
                let is_last_move = Some(sq) == last_move;

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
            let (black_count, white_count) = self.core.get_score();
            match y {
                2 => {
                    let player_info = match side_to_move {
                        Piece::Black => "Black's turn (X)".bright_green(),
                        Piece::White => "White's turn (O)".bright_yellow(),
                        _ => unreachable!(),
                    };
                    println!("   {player_info}");
                }
                3 => println!("   Black: {}", format!("{black_count:2}").bright_green()),
                4 => println!("   White: {}", format!("{white_count:2}").bright_yellow()),
                6 => {
                    if self.core.is_game_over() {
                        match black_count.cmp(&white_count) {
                            std::cmp::Ordering::Greater => {
                                println!("   {}", "Black wins!".bright_green())
                            }
                            std::cmp::Ordering::Less => {
                                println!("   {}", "White wins!".bright_yellow())
                            }
                            std::cmp::Ordering::Equal => println!("  {}", "Draw".bright_cyan()),
                        }
                    } else {
                        println!();
                    }
                }
                7 => {
                    if self.core.is_game_over() {
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
}
