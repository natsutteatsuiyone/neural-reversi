//! Game state management for Reversi.
//!
//! This module provides the `GameState` struct which maintains the current
//! game position and handles core game logic such as making moves, automatic
//! passing when no legal moves are available, and game termination detection.

use crate::board::Board;
use crate::disc::Disc;
use crate::square::Square;

/// A single recorded action in a game's history.
///
/// `board` and `side_to_move` capture the state before the action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    /// The move played, or [`None`] for a pass.
    pub mv: Option<Square>,
    /// Board position before the action.
    pub board: Board,
    /// Side to move before the action.
    pub side_to_move: Disc,
    /// Whether this pass was inserted automatically by [`GameState::make_move`].
    pub auto_pass: bool,
}

/// The state of a Reversi game.
///
/// Handles move execution, automatic passing, move history tracking,
/// and undo functionality.
#[derive(Clone, Debug)]
pub struct GameState {
    /// The current board position.
    board: Board,
    /// Which player's turn it is to move.
    side_to_move: Disc,
    /// Move history, one [`HistoryEntry`] per action.
    history: Vec<HistoryEntry>,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    /// Creates a new game in the standard initial position with Black to move.
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            side_to_move: Disc::Black,
            history: Vec::new(),
        }
    }

    /// Creates a new game state from an existing [`Board`] position.
    pub fn from_board(board: Board, side_to_move: Disc) -> Self {
        Self {
            board,
            side_to_move,
            history: Vec::new(),
        }
    }

    /// Returns a reference to the current [`Board`] position.
    pub fn board(&self) -> &Board {
        &self.board
    }

    /// Returns the current side to move.
    pub fn side_to_move(&self) -> Disc {
        self.side_to_move
    }

    /// Executes a move and updates the game state.
    ///
    /// Also automatically passes for the opponent if they have no legal moves.
    ///
    /// # Errors
    ///
    /// Returns an error if `sq` is not a legal move on the current board.
    pub fn make_move(&mut self, sq: Square) -> Result<(), String> {
        if !self.board.is_legal_move(sq) {
            return Err(format!("Illegal move: {sq:?}"));
        }

        // Record history before making the move
        self.history.push(HistoryEntry {
            mv: Some(sq),
            board: self.board,
            side_to_move: self.side_to_move,
            auto_pass: false,
        });

        self.board = self.board.make_move(sq);
        self.side_to_move = self.side_to_move.opposite();

        // Handle automatic pass if opponent has no legal moves, but avoid
        // recording a pass after the game has already ended.
        if !self.board.has_legal_moves() && self.board.switch_players().has_legal_moves() {
            self.handle_pass(true);
        }

        Ok(())
    }

    /// Executes a pass move (switches players without placing a disc).
    ///
    /// # Errors
    ///
    /// Returns an error if the current player has legal moves available.
    pub fn make_pass(&mut self) -> Result<(), String> {
        if self.board.is_game_over() {
            return Err("Cannot pass after the game is over".to_string());
        }

        if self.board.has_legal_moves() {
            return Err("Cannot pass when legal moves are available".to_string());
        }

        self.handle_pass(false);
        Ok(())
    }

    /// Records a pass in history and switches the side to move.
    fn handle_pass(&mut self, auto_pass: bool) {
        // Record pass in history
        self.history.push(HistoryEntry {
            mv: None,
            board: self.board,
            side_to_move: self.side_to_move,
            auto_pass,
        });

        self.board = self.board.switch_players();
        self.side_to_move = self.side_to_move.opposite();
    }

    /// Returns whether the game has ended.
    ///
    /// A game ends when both players pass consecutively (neither has
    /// legal moves) or when the board is completely filled.
    pub fn is_game_over(&self) -> bool {
        self.board.is_game_over()
    }

    /// Returns the disc count as `(black_count, white_count)`.
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

    /// Returns the last move played, or [`None`] if the last move was a pass
    /// or no moves have been played yet.
    pub fn last_move(&self) -> Option<Square> {
        self.history.last().and_then(|entry| entry.mv)
    }

    /// Returns a reference to the move history.
    ///
    /// Each entry is a [`HistoryEntry`]; [`None`] for the move indicates a
    /// pass.
    pub fn move_history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Undoes the last action, returning `true` if successful.
    ///
    /// A move and the automatic pass it triggered are undone together;
    /// an explicit pass ([`GameState::make_pass`]) is undone on its own.
    /// Returns `false` if there is nothing to undo.
    pub fn undo(&mut self) -> bool {
        let Some(mut entry) = self.history.pop() else {
            return false;
        };
        if entry.auto_pass {
            entry = self.history.pop().unwrap_or(entry);
        }
        self.board = entry.board;
        self.side_to_move = entry.side_to_move;
        true
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
    fn test_illegal_move() {
        let mut game = GameState::new();
        let result = game.make_move(Square::A1);
        assert!(result.is_err());
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
    fn test_make_pass_rejects_game_over_position() {
        let board = Board::from_bitboards(Square::A1.bitboard(), 0);
        let mut game = GameState::from_board(board, Disc::Black);

        let result = game.make_pass();

        assert!(result.is_err());
        assert_eq!(game.move_history().len(), 0);
        assert_eq!(*game.board(), board);
        assert_eq!(game.side_to_move(), Disc::Black);
    }

    #[test]
    fn test_make_move_does_not_record_pass_after_game_over() {
        let board = Board::from_string(
            "-OXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX",
            Disc::Black,
        )
        .unwrap();
        let mut game = GameState::from_board(board, Disc::Black);

        assert_eq!(game.board().get_moves(), Square::A1.bitboard());
        game.make_move(Square::A1).unwrap();

        assert!(game.is_game_over());
        assert_eq!(game.last_move(), Some(Square::A1));
        assert_eq!(game.move_history().len(), 1);
        assert_eq!(game.side_to_move(), Disc::White);
    }

    #[test]
    fn test_undo_rolls_back_auto_pass_with_move() {
        // Black c1 flips b1; White (b8 only) then has no legal move, Black does.
        let board = Board::from_string(
            "XO------\
             --------\
             --------\
             --------\
             --------\
             --------\
             --------\
             XO------",
            Disc::Black,
        )
        .unwrap();
        let mut game = GameState::from_board(board, Disc::Black);

        game.make_move(Square::C1).unwrap();
        assert_eq!(game.move_history().len(), 2);
        assert_eq!(game.side_to_move(), Disc::Black);

        assert!(game.undo());
        assert_eq!(*game.board(), board);
        assert_eq!(game.side_to_move(), Disc::Black);
        assert!(game.move_history().is_empty());
    }

    #[test]
    fn test_undo_explicit_pass_is_single_step() {
        // White to move with no legal moves (must pass); Black can then play c8.
        let board = Board::from_string(
            "XXX-----\
             --------\
             --------\
             --------\
             --------\
             --------\
             --------\
             XO------",
            Disc::White,
        )
        .unwrap();
        let mut game = GameState::from_board(board, Disc::White);

        game.make_pass().unwrap();
        game.make_move(Square::C8).unwrap();
        assert_eq!(game.move_history().len(), 2);

        assert!(game.undo());
        assert_eq!(game.side_to_move(), Disc::Black);
        assert_eq!(game.move_history().len(), 1);

        assert!(game.undo());
        assert_eq!(*game.board(), board);
        assert_eq!(game.side_to_move(), Disc::White);
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
        assert_eq!(history[0].mv, Some(Square::D3));
        assert_eq!(history[0].side_to_move, Disc::Black);

        // Verify second move
        assert_eq!(history[1].mv, Some(Square::C3));
        assert_eq!(history[1].side_to_move, Disc::White);

        // Verify third move
        assert_eq!(history[2].mv, Some(Square::C4));
        assert_eq!(history[2].side_to_move, Disc::Black);
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

        // Verify history
        let history = game.move_history();

        // Verify the first few moves in history
        assert_eq!(history[0].mv, Some(Square::E6), "First move should be e6");
        assert_eq!(history[0].side_to_move, Disc::Black, "First move by Black");

        assert_eq!(history[1].mv, Some(Square::F4), "Second move should be f4");
        assert_eq!(history[1].side_to_move, Disc::White, "Second move by White");

        assert_eq!(history[2].mv, Some(Square::C3), "Third move should be c3");
        assert_eq!(history[2].side_to_move, Disc::Black, "Third move by Black");

        // Verify last_move
        // Note: The last move in history might be a pass (automatic pass after b1)
        // so we check if b1 appears in the history
        let b1_found = history.iter().any(|entry| entry.mv == Some(Square::B1));
        assert!(b1_found, "b1 should be in the move history");

        // If the last entry is a pass, the previous one should be b1
        if game.last_move().is_none() {
            // Last move was a pass, check the second to last
            let second_to_last = history.iter().rev().nth(1);
            if let Some(entry) = second_to_last {
                assert_eq!(
                    entry.mv,
                    Some(Square::B1),
                    "Second to last move should be b1"
                );
            }
        } else {
            assert_eq!(game.last_move(), Some(Square::B1), "Last move should be b1");
        }

        // Verify complete history matches the game record
        let expected_moves: Vec<Square> =
            moves.iter().map(|s| s.parse::<Square>().unwrap()).collect();

        // Extract non-pass moves from history
        let actual_moves: Vec<Square> = history.iter().filter_map(|entry| entry.mv).collect();

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
            let HistoryEntry {
                mv: sq_current,
                side_to_move: side_current,
                ..
            } = history[i];
            let HistoryEntry {
                mv: sq_next,
                side_to_move: side_next,
                ..
            } = history[i + 1];

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
        let pass_count = history.iter().filter(|entry| entry.mv.is_none()).count();
        println!(
            "Game completed with {} moves and {} automatic passes",
            expected_moves.len(),
            pass_count
        );
    }
}
