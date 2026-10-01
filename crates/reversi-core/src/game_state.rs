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

    /// Creates a game state by replaying a sequence of moves from the
    /// initial position.
    ///
    /// Passes are inserted automatically; `moves` must contain only actual
    /// moves.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first move that is illegal when reached.
    pub fn from_moves(moves: &[Square]) -> Result<Self, String> {
        let mut state = Self::new();
        for (i, &sq) in moves.iter().enumerate() {
            state
                .make_move(sq)
                .map_err(|_| format!("Illegal move at position {}: {sq}", i + 1))?;
        }
        Ok(state)
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
    fn test_from_moves_reports_offending_position() {
        let err = GameState::from_moves(&[Square::D3, Square::D3]).unwrap_err();
        assert!(err.contains("position 2"), "unexpected error: {err}");
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
        let moves = Square::parse_sequence(
            "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8b6e8b5g2d8b7a6h2e7d7c8a8a7b8c7h1b2b1",
        )
        .unwrap();
        let game = GameState::from_moves(&moves).unwrap();

        assert!(game.is_game_over());
        assert_eq!(game.get_score(), (57, 7));

        let history = game.move_history();
        let played: Vec<Square> = history.iter().filter_map(|entry| entry.mv).collect();
        assert_eq!(played, moves);
        assert_eq!(history[0].side_to_move, Disc::Black);
        assert!(
            history
                .windows(2)
                .all(|w| w[1].side_to_move == w[0].side_to_move.opposite())
        );
    }
}
