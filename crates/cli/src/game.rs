//! Game state management for the Neural Reversi CLI.
//!
//! This module provides the `GameState` struct which wraps the core
//! game state and adds CLI-specific display capabilities.

use std::fmt::Write;

use reversi_core::{board::Board, disc::Disc, game_state, square::Square};

/// Represents the state of a Reversi/Othello game with CLI-specific features.
///
/// This is a thin wrapper around the core `GameState` that adds
/// text-based display functionality for GTP output.
pub struct GameState {
    /// Core game state
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
    /// The initial position has 4 discs in the center (2 black, 2 white)
    /// with Black to move first, following standard Reversi rules.
    pub fn new() -> Self {
        Self {
            core: game_state::GameState::new(),
        }
    }

    /// Returns a reference to the current board position.
    pub fn board(&self) -> &Board {
        self.core.board()
    }

    /// Executes a move and updates the game state.
    ///
    /// If the opponent has no legal moves after this move, an automatic pass
    /// is performed and the turn returns to the current player.
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
    ///
    /// # Panics
    /// Panics if legal moves are available (a pass is only valid when no moves exist)
    pub fn make_pass(&mut self) {
        self.core
            .make_pass()
            .expect("Attempted to pass when legal moves exist");
    }

    /// Returns which player's turn it is to move.
    pub fn side_to_move(&self) -> Disc {
        self.core.side_to_move()
    }

    /// Returns a plain text representation of the board with disc counts
    /// and turn indicator, suitable for GTP output.
    pub fn board_string(&self) -> String {
        let mut result = String::new();
        let board = self.core.board();
        let side_to_move = self.core.side_to_move();
        let (black_count, white_count) = self.core.get_score();

        // Header
        result.push_str("   a b c d e f g h\n");
        result.push_str("  +-+-+-+-+-+-+-+-+\n");

        // Board rows
        for y in 0..8 {
            let _ = write!(result, "{} |", y + 1);

            for x in 0..8 {
                let sq = Square::from_usize_unchecked(y * 8 + x);
                let piece = board.get_disc_at(sq, side_to_move);
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
                        if side_to_move == Disc::Black {
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
        result
    }

    /// Undoes the last move if possible.
    pub fn undo(&mut self) -> bool {
        self.core.undo()
    }

    /// Returns the last move played.
    ///
    /// Returns `None` if the last move was a pass (including automatic passes
    /// after a move) or if no moves have been played yet.
    pub fn last_move(&self) -> Option<Square> {
        self.core.last_move()
    }

    /// Returns the disc count as `(black_count, white_count)`.
    pub fn score(&self) -> (u32, u32) {
        self.core.get_score()
    }

    /// Returns the move history as a list of squares, excluding passes.
    pub fn move_history(&self) -> Vec<Square> {
        self.core
            .move_history()
            .iter()
            .filter_map(|(sq, _, _)| *sq)
            .collect()
    }
}
