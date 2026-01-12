//! Game state management for Reversi.
//!
//! This module provides the `GameState` struct which maintains the current
//! game position and handles core game logic such as making moves, automatic
//! passing when no legal moves are available, and game termination detection.

use crate::board::Board;
use crate::disc::Disc;
use crate::square::Square;

/// Represents the state of a Reversi game.
///
/// This is a core game state manager that handles move execution,
/// automatic passing, move history tracking, and undo functionality.
#[derive(Clone, Debug)]
pub struct GameState {
    /// The current board position.
    board: Board,
    /// Which player's turn it is to move.
    side_to_move: Disc,
    /// Move history: (move, board_before_move, side_to_move_before).
    /// None for move indicates a pass.
    history: Vec<(Option<Square>, Board, Disc)>,
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
    ///
    /// # Returns
    ///
    /// A new `GameState` in the starting position.
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            side_to_move: Disc::Black,
            history: Vec::new(),
        }
    }

    /// Creates a new game state from an existing board position.
    ///
    /// This is useful for setting up specific positions for analysis
    /// or continuing a game from a saved state.
    ///
    /// # Arguments
    ///
    /// * `board` - The board position to start from
    /// * `side_to_move` - Which player moves next
    ///
    /// # Returns
    ///
    /// A new `GameState` with the specified position.
    pub fn from_board(board: Board, side_to_move: Disc) -> Self {
        Self {
            board,
            side_to_move,
            history: Vec::new(),
        }
    }

    /// Returns a reference to the current board position.
    ///
    /// # Returns
    ///
    /// A reference to the `Board`
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Returns which player's turn it is to move.
    ///
    /// # Returns
    ///
    /// The `Disc` representing the current player (Black or White)
    pub fn side_to_move(&self) -> Disc {
        self.side_to_move
    }

    /// Executes a move and updates the game state.
    ///
    /// This method handles regular moves and automatically manages passing
    /// when the opponent has no legal moves after the move is made.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to place a disc on
    ///
    /// # Returns
    ///
    /// `Ok(())` if the move was successfully executed.
    ///
    /// # Errors
    ///
    /// Returns an error string if the move is not legal on the current board.
    pub fn make_move(&mut self, sq: Square) -> Result<(), String> {
        if self.board.get_moves() & sq.bitboard() == 0 {
            return Err(format!("Illegal move: {sq:?}"));
        }

        // Record history before making the move
        self.history.push((Some(sq), self.board, self.side_to_move));

        self.board = self.board.make_move(sq);
        self.side_to_move = self.side_to_move.opposite();

        // Handle automatic pass if opponent has no legal moves
        if !self.board.has_legal_moves() {
            self.handle_pass();
        }

        Ok(())
    }

    /// Executes a pass move (switching players without placing a disc).
    ///
    /// # Returns
    ///
    /// `Ok(())` if the pass was successfully executed.
    ///
    /// # Errors
    ///
    /// Returns an error string if attempting to pass when legal moves are available.
    pub fn make_pass(&mut self) -> Result<(), String> {
        if self.board.has_legal_moves() {
            return Err("Cannot pass when legal moves are available".to_string());
        }

        self.handle_pass();
        Ok(())
    }

    /// Internal method to handle a pass move.
    fn handle_pass(&mut self) {
        // Record pass in history
        self.history.push((None, self.board, self.side_to_move));

        self.board = self.board.switch_players();
        self.side_to_move = self.side_to_move.opposite();
    }

    /// Checks if the game has ended.
    ///
    /// A game ends when both players pass consecutively (neither player has
    /// legal moves) or when the board is completely filled.
    ///
    /// # Returns
    ///
    /// `true` if the game is over, `false` otherwise
    pub fn is_game_over(&self) -> bool {
        // Check if the last move was a pass and current player has no legal moves
        // (meaning both players passed consecutively)
        if self.history.last().is_some_and(|(sq, _, _)| sq.is_none())
            && !self.board.has_legal_moves()
        {
            return true;
        }

        self.board.get_empty_count() == 0
    }

    /// Returns the disc count for both players.
    ///
    /// # Returns
    ///
    /// A tuple `(black_count, white_count)` representing the number of
    /// discs each player has on the board.
    pub fn get_score(&self) -> (u32, u32) {
        let (black_count, white_count) = if self.side_to_move == Disc::Black {
            (
                self.board.get_player_count(),
                self.board.get_opponent_count(),
            )
        } else {
            (
                self.board.get_opponent_count(),
                self.board.get_player_count(),
            )
        };

        (black_count, white_count)
    }

    /// Returns the last move played.
    ///
    /// # Returns
    ///
    /// `Some(Square)` if a regular move was played, `None` if the last move was a pass
    /// or if no moves have been played yet
    pub fn last_move(&self) -> Option<Square> {
        self.history.last().and_then(|(sq, _, _)| *sq)
    }

    /// Returns a reference to the move history.
    ///
    /// # Returns
    ///
    /// A slice of tuples containing (move, board_before_move, side_to_move_before).
    /// `None` for the move indicates a pass.
    pub fn move_history(&self) -> &[(Option<Square>, Board, Disc)] {
        &self.history
    }

    /// Undoes the last move if possible.
    ///
    /// This restores the game state to what it was before the last move,
    /// including the board position and side to move.
    ///
    /// # Returns
    ///
    /// `true` if a move was successfully undone, `false` if there are no moves to undo
    pub fn undo(&mut self) -> bool {
        match self.history.pop() {
            Some((_, prev_board, prev_side)) => {
                self.board = prev_board;
                self.side_to_move = prev_side;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = GameState::new();
        assert_eq!(game.side_to_move(), Disc::Black);
        assert!(!game.is_game_over());
        assert_eq!(game.get_score(), (2, 2));
    }

    #[test]
    fn test_make_move() {
        let mut game = GameState::new();
        let result = game.make_move(Square::D3);
        assert!(result.is_ok());
        assert_eq!(game.side_to_move(), Disc::White);
    }

    #[test]
    fn test_illegal_move() {
        let mut game = GameState::new();
        let result = game.make_move(Square::A1);
        assert!(result.is_err());
    }

    #[test]
    fn test_game_over() {
        let board = Board::new();
        let mut game = GameState::from_board(board, Disc::Black);

        // Play through a game until it's over
        while !game.is_game_over() {
            if game.board().has_legal_moves() {
                let moves = game.board().get_moves();
                let first_move = crate::bitboard::BitboardIterator::new(moves)
                    .next()
                    .unwrap();
                let _ = game.make_move(first_move);
            } else {
                let _ = game.make_pass();
            }
        }

        assert!(game.is_game_over());
    }

    #[test]
    fn test_undo() {
        let mut game = GameState::new();
        let original_board = *game.board();
        let original_side = game.side_to_move();

        // Make a move
        game.make_move(Square::D3).unwrap();
        assert_ne!(*game.board(), original_board);
        assert_ne!(game.side_to_move(), original_side);

        // Undo the move
        assert!(game.undo());
        assert_eq!(*game.board(), original_board);
        assert_eq!(game.side_to_move(), original_side);
    }

    #[test]
    fn test_undo_multiple() {
        let mut game = GameState::new();

        // Make several moves (use legal moves)
        game.make_move(Square::D3).unwrap();
        game.make_move(Square::C3).unwrap();
        game.make_move(Square::C4).unwrap();

        // Undo all moves
        assert!(game.undo());
        assert!(game.undo());
        assert!(game.undo());

        // Should be back to initial state
        assert_eq!(game.side_to_move(), Disc::Black);
        assert_eq!(game.get_score(), (2, 2));
    }

    #[test]
    fn test_undo_when_empty() {
        let mut game = GameState::new();

        // Cannot undo when no moves have been made
        assert!(!game.undo());
        assert_eq!(game.side_to_move(), Disc::Black);
    }

    #[test]
    fn test_last_move() {
        let mut game = GameState::new();

        // Initially no moves
        assert_eq!(game.last_move(), None);

        // After making a move
        game.make_move(Square::D3).unwrap();
        assert_eq!(game.last_move(), Some(Square::D3));

        // After making another move
        game.make_move(Square::C3).unwrap();
        assert_eq!(game.last_move(), Some(Square::C3));
    }

    #[test]
    fn test_from_board() {
        let board = Board::new();
        let game = GameState::from_board(board, Disc::White);

        assert_eq!(game.side_to_move(), Disc::White);
        assert_eq!(*game.board(), board);
        assert_eq!(game.move_history().len(), 0);
    }

    #[test]
    fn test_history_complete_record() {
        let mut game = GameState::new();

        // Make several moves
        game.make_move(Square::D3).unwrap();
        game.make_move(Square::C3).unwrap();
        game.make_move(Square::C4).unwrap();

        let history = game.move_history();
        assert_eq!(history.len(), 3);

        // Verify first move
        assert_eq!(history[0].0, Some(Square::D3));
        assert_eq!(history[0].2, Disc::Black);

        // Verify second move
        assert_eq!(history[1].0, Some(Square::C3));
        assert_eq!(history[1].2, Disc::White);

        // Verify third move
        assert_eq!(history[2].0, Some(Square::C4));
        assert_eq!(history[2].2, Disc::Black);
    }

    #[test]
    fn test_history_restoration_with_undo() {
        let mut game = GameState::new();
        let initial_board = *game.board();

        // Make a move
        game.make_move(Square::D3).unwrap();
        let board_after_d3 = *game.board();

        // Make another move
        game.make_move(Square::C3).unwrap();

        // Undo - should restore to board_after_d3
        game.undo();
        assert_eq!(*game.board(), board_after_d3);

        // Undo again - should restore to initial_board
        game.undo();
        assert_eq!(*game.board(), initial_board);
    }

    #[test]
    fn test_score_tracking() {
        let mut game = GameState::new();
        let (black, white) = game.get_score();
        assert_eq!(black, 2);
        assert_eq!(white, 2);

        game.make_move(Square::D3).unwrap();
        let (black, white) = game.get_score();
        assert_eq!(black, 4);
        assert_eq!(white, 1);
    }

    #[test]
    fn test_side_to_move_alternates() {
        let mut game = GameState::new();
        assert_eq!(game.side_to_move(), Disc::Black);

        game.make_move(Square::D3).unwrap();
        assert_eq!(game.side_to_move(), Disc::White);

        game.make_move(Square::C3).unwrap();
        assert_eq!(game.side_to_move(), Disc::Black);
    }

    #[test]
    fn test_game_record_black_57_white_7() {
        // Test a specific game record that ends with Black: 57, White: 7
        let mut game = GameState::new();

        let moves_str = "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8b6e8b5g2d8b7a6h2e7d7c8a8a7b8c7h1b2b1";

        // Parse and play each move
        let moves: Vec<&str> = moves_str
            .as_bytes()
            .chunks(2)
            .map(|chunk| std::str::from_utf8(chunk).unwrap())
            .collect();

        for (i, move_str) in moves.iter().enumerate() {
            let square = move_str.parse::<Square>().unwrap_or_else(|_| {
                panic!("Failed to parse move #{}: {}", i + 1, move_str);
            });

            game.make_move(square).unwrap_or_else(|e| {
                panic!("Failed to make move #{} ({}): {}", i + 1, move_str, e);
            });
        }

        // Verify the game is over
        assert!(game.is_game_over(), "Game should be over after all moves");

        // Verify the final score
        let (black_count, white_count) = game.get_score();
        assert_eq!(black_count, 57, "Black should have 57 discs");
        assert_eq!(white_count, 7, "White should have 7 discs");

        // Verify the total disc count
        assert_eq!(black_count + white_count, 64, "Total discs should be 64");

        // Verify history
        let history = game.move_history();
        // The history should contain all moves (60 regular moves + any automatic passes)
        assert!(
            history.len() >= 60,
            "History should contain at least 60 moves, got {}",
            history.len()
        );

        // Verify the first few moves in history
        assert_eq!(history[0].0, Some(Square::E6), "First move should be e6");
        assert_eq!(history[0].2, Disc::Black, "First move by Black");

        assert_eq!(history[1].0, Some(Square::F4), "Second move should be f4");
        assert_eq!(history[1].2, Disc::White, "Second move by White");

        assert_eq!(history[2].0, Some(Square::C3), "Third move should be c3");
        assert_eq!(history[2].2, Disc::Black, "Third move by Black");

        // Verify last_move
        // Note: The last move in history might be a pass (automatic pass after b1)
        // so we check if b1 appears in the history
        let b1_found = history.iter().any(|(sq, _, _)| *sq == Some(Square::B1));
        assert!(b1_found, "b1 should be in the move history");

        // If the last entry is a pass, the previous one should be b1
        if game.last_move().is_none() {
            // Last move was a pass, check the second to last
            let second_to_last = history.iter().rev().nth(1);
            if let Some((sq, _, _)) = second_to_last {
                assert_eq!(*sq, Some(Square::B1), "Second to last move should be b1");
            }
        } else {
            assert_eq!(game.last_move(), Some(Square::B1), "Last move should be b1");
        }

        // Verify complete history matches the game record
        let expected_moves: Vec<Square> =
            moves.iter().map(|s| s.parse::<Square>().unwrap()).collect();

        // Extract non-pass moves from history
        let actual_moves: Vec<Square> = history.iter().filter_map(|(sq, _, _)| *sq).collect();

        // All expected moves should be in the actual moves
        assert_eq!(
            actual_moves.len(),
            expected_moves.len(),
            "Number of non-pass moves should match"
        );

        for (i, (expected, actual)) in expected_moves.iter().zip(actual_moves.iter()).enumerate() {
            assert_eq!(
                actual,
                expected,
                "Move #{} mismatch: expected {:?}, got {:?}",
                i + 1,
                expected,
                actual
            );
        }

        // Verify side_to_move is recorded correctly
        // When there's an automatic pass, the side doesn't change
        // We verify by checking each move in sequence
        for i in 0..history.len().saturating_sub(1) {
            let (sq_current, _, side_current) = history[i];
            let (sq_next, _, side_next) = history[i + 1];

            if sq_current.is_none() {
                // Current is a pass - next move should be by the opposite side
                assert_eq!(
                    side_next,
                    side_current.opposite(),
                    "After pass at #{}, side should switch",
                    i + 1
                );
            } else {
                // Current is a regular move - next should be opposite unless it's a pass
                if sq_next.is_some() {
                    // Next is also a regular move - should be opposite side
                    assert_eq!(
                        side_next,
                        side_current.opposite(),
                        "After regular move at #{}, side should switch",
                        i + 1
                    );
                } else {
                    // Next is a pass - should be the opposite side's pass
                    assert_eq!(
                        side_next,
                        side_current.opposite(),
                        "Pass at #{} should be by opposite side",
                        i + 2
                    );
                }
            }
        }

        // Count passes in history
        let pass_count = history.iter().filter(|(sq, _, _)| sq.is_none()).count();
        println!(
            "Game completed with {} moves and {} automatic passes",
            expected_moves.len(),
            pass_count
        );
    }
}
