//! Game state management for Reversi matches.
//!
//! This module provides the GameState struct which tracks the current state
//! of a Reversi game, including board position, turn order, and move history.

use reversi_core::board::Board;
use reversi_core::piece::Piece;
use reversi_core::square::Square;

/// Represents the current state of a Reversi game.
///
/// GameState tracks all aspects of an ongoing game including the board position,
/// whose turn it is, move history, and game termination conditions.
pub struct GameState {
    board: Board,
    side_to_move: Piece,
    last_move_was_pass: bool,
    move_history: Vec<(Option<Square>, Piece)>,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    /// Create a new game in the standard starting position.
    ///
    /// # Returns
    ///
    /// A new GameState with the initial Reversi setup (black to move).
    pub fn new() -> Self {
        GameState {
            board: Board::new(),
            side_to_move: Piece::Black,
            last_move_was_pass: false,
            move_history: Vec::new(),
        }
    }

    /// Get the player whose turn it is to move.
    ///
    /// # Returns
    ///
    /// The `Piece` (Black or White) representing the current player.
    pub fn side_to_move(&self) -> Piece {
        self.side_to_move
    }

    /// Make a move on the board.
    ///
    /// Attempts to play the specified move for the current player. Handles both
    /// regular moves and pass moves, automatically switching turns and managing
    /// game flow.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to play on, or `None` for a pass move
    ///
    /// # Returns
    ///
    /// `Ok(())` if the move was played successfully.
    ///
    /// # Errors
    ///
    /// Returns an error string if:
    /// - The move is illegal (square not in legal moves)
    /// - Attempting to pass when legal moves are available
    pub fn make_move(&mut self, sq: Option<Square>) -> Result<(), String> {
        match sq {
            Some(square) => {
                if self.board.get_moves() & square.bitboard() == 0 {
                    return Err(format!("Illegal move: {square:?}"));
                }

                self.board = self.board.make_move(square);
                self.move_history.push((Some(square), self.side_to_move));
                self.last_move_was_pass = false;

                self.side_to_move = self.side_to_move.opposite();

                if !self.board.has_legal_moves() {
                    self.handle_pass();
                }
            }
            None => {
                // Pass move
                if self.board.has_legal_moves() {
                    return Err("Cannot pass when legal moves are available".to_string());
                }

                self.handle_pass();
            }
        }

        Ok(())
    }

    fn handle_pass(&mut self) {
        self.move_history.push((None, self.side_to_move));
        self.board = self.board.switch_players();
        self.side_to_move = self.side_to_move.opposite();
        self.last_move_was_pass = true;
    }

    /// Check if the game has ended.
    ///
    /// A game ends when both players pass consecutively or when the board is full.
    ///
    /// # Returns
    ///
    /// `true` if the game is over, `false` otherwise.
    pub fn is_game_over(&self) -> bool {
        if self.last_move_was_pass && !self.board.has_legal_moves() {
            return true;
        }

        self.board.get_empty_count() == 0
    }

    /// Get the current disc count for both players.
    ///
    /// # Returns
    ///
    /// A tuple `(black_count, white_count)` representing the number of
    /// discs each player has on the board.
    pub fn get_score(&self) -> (u32, u32) {
        let black_count;
        let white_count;

        if self.side_to_move == Piece::Black {
            black_count = self.board.get_player_count();
            white_count = self.board.get_opponent_count();
        } else {
            white_count = self.board.get_player_count();
            black_count = self.board.get_opponent_count();
        }

        (black_count, white_count)
    }
}
