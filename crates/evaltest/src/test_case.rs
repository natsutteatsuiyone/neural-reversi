//! Test case definitions and utilities

use reversi_core::{board::Board, disc::Disc};
use std::fmt;

/// A single test case containing position and expected results
#[derive(Debug, Clone)]
pub struct TestCase {
    pub line_number: usize,
    board_str: String,
    side_to_move: Disc,
    pub expected_score: i32,
    best_moves: Vec<String>,
    second_best_moves: Vec<String>,
    third_best_moves: Vec<String>,
}

impl fmt::Display for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{} ({} to move, score: {})",
            self.line_number,
            if self.side_to_move == Disc::Black {
                "Black"
            } else {
                "White"
            },
            self.expected_score
        )
    }
}

impl TestCase {
    /// Create a new test case from parsed OBF data
    pub fn new(
        line_number: usize,
        board_str: String,
        side_to_move: Disc,
        expected_score: i32,
        best_moves: Vec<String>,
        second_best_moves: Vec<String>,
        third_best_moves: Vec<String>,
    ) -> Self {
        Self {
            line_number,
            board_str,
            side_to_move,
            expected_score,
            best_moves,
            second_best_moves,
            third_best_moves,
        }
    }

    /// Whether this is a pass position (no legal moves for the side to move)
    pub fn is_pass(&self) -> bool {
        self.best_moves.is_empty()
    }

    /// Convert the test case into a playable Board instance
    pub fn get_board(&self) -> Board {
        Board::from_string(&self.board_str, self.side_to_move)
            .expect("Test case should have valid board string")
    }

    /// Check if the given move is one of the best moves
    pub fn is_best_move(&self, move_str: &str) -> bool {
        self.best_moves.iter().any(|m| m == move_str)
    }

    /// Check if the given move is one of the second-best moves
    pub fn is_second_best_move(&self, move_str: &str) -> bool {
        self.second_best_moves.iter().any(|m| m == move_str)
    }

    /// Check if the given move is one of the third-best moves
    pub fn is_third_best_move(&self, move_str: &str) -> bool {
        self.third_best_moves.iter().any(|m| m == move_str)
    }

    /// Get the best moves as a comma-separated string
    pub fn get_best_moves_str(&self) -> String {
        self.best_moves.join(",")
    }

    /// Returns the side to move for this test case.
    pub fn side_to_move(&self) -> Disc {
        self.side_to_move
    }
}
