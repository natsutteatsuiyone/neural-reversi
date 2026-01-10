//! Game state management for Reversi matches.
//!
//! This module provides the GameState struct which wraps the core
//! game state for match play.

use reversi_core::disc::Disc;
use reversi_core::game_state;
use reversi_core::square::Square;

/// Represents the current state of a Reversi game for match play.
///
/// This is a thin wrapper around the core `GameState` with
/// move history tracking built-in.
pub struct GameState {
    /// Core game state with history support
    core: game_state::GameState,
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
            core: game_state::GameState::new(),
        }
    }

    /// Get the player whose turn it is to move.
    ///
    /// # Returns
    ///
    /// The `Disc` (Black or White) representing the current player.
    pub fn side_to_move(&self) -> Disc {
        self.core.side_to_move()
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
            Some(square) => self.core.make_move(square),
            None => self.core.make_pass(),
        }
    }

    /// Check if the game has ended.
    ///
    /// A game ends when both players pass consecutively or when the board is full.
    ///
    /// # Returns
    ///
    /// `true` if the game is over, `false` otherwise.
    pub fn is_game_over(&self) -> bool {
        self.core.is_game_over()
    }

    /// Get the current disc count for both players.
    ///
    /// # Returns
    ///
    /// A tuple `(black_count, white_count)` representing the number of
    /// discs each player has on the board.
    pub fn get_score(&self) -> (u32, u32) {
        self.core.get_score()
    }
}
