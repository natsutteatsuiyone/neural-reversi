//! FFO test case definitions and utilities

use reversi_core::{board::Board, piece::Piece};
use std::fmt;

/// A single FFO test case containing position and expected results
#[derive(Debug, Clone)]
pub struct TestCase {
    pub no: usize,
    board_str: &'static str,
    side_to_move: Piece,
    pub expected_score: i32,
    best_moves: Vec<&'static str>,
    second_best_moves: Vec<&'static str>,
    third_best_moves: Vec<&'static str>,
}

impl fmt::Display for TestCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FFO#{:02} ({} to move, score: {})",
            self.no,
            if self.side_to_move == Piece::Black { "Black" } else { "White" },
            self.expected_score
        )
    }
}

impl TestCase {
    /// Create a new test case
    ///
    /// # Arguments
    /// * `no` - Test case number (1-79)
    /// * `board_str` - 64-character board representation (X=Black, O=White, -=Empty)
    /// * `side_to_move` - "X" for Black, "O" for White
    /// * `expected_score` - Optimal score from the mover's perspective
    /// * `best_moves` - Comma-separated list of optimal moves
    /// * `second_best_moves` - Comma-separated list of second-best moves
    /// * `third_best_moves` - Comma-separated list of third-best moves
    pub fn new(
        no: usize,
        board_str: &'static str,
        side_to_move: &'static str,
        expected_score: i32,
        best_moves: &'static str,
        second_best_moves: &'static str,
        third_best_moves: &'static str,
    ) -> Self {
        let stm = match side_to_move {
            "X" => Piece::Black,
            "O" => Piece::White,
            _ => panic!("Invalid side to move: {}", side_to_move),
        };

        Self {
            no,
            board_str,
            side_to_move: stm,
            expected_score,
            best_moves: Self::parse_moves(best_moves),
            second_best_moves: Self::parse_moves(second_best_moves),
            third_best_moves: Self::parse_moves(third_best_moves),
        }
    }

    /// Parse comma-separated moves, filtering out empty strings
    fn parse_moves(moves_str: &str) -> Vec<&str> {
        moves_str
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Convert the test case into a playable Board instance
    pub fn get_board(&self) -> Board {
        Board::from_string(self.board_str, self.side_to_move)
    }

    /// Check if the given move is one of the best moves
    pub fn is_best_move(&self, move_str: &str) -> bool {
        self.best_moves.contains(&move_str)
    }

    /// Check if the given move is one of the second-best moves
    pub fn is_second_best_move(&self, move_str: &str) -> bool {
        self.second_best_moves.contains(&move_str)
    }

    /// Check if the given move is one of the third-best moves
    pub fn is_third_best_move(&self, move_str: &str) -> bool {
        self.third_best_moves.contains(&move_str)
    }

    /// Get the best moves as a comma-separated string
    pub fn get_best_moves_str(&self) -> String {
        self.best_moves.join(",")
    }

}

/// Validate all test cases for data integrity
#[allow(dead_code)]
fn validate_test_cases(cases: &[TestCase]) -> Result<(), String> {
    // Check for duplicate test numbers
    let mut seen = std::collections::HashSet::new();
    for case in cases {
        if !seen.insert(case.no) {
            return Err(format!("Duplicate test case number: {}", case.no));
        }
    }

    // Basic validation
    for case in cases {
        // Check board string length
        if case.board_str.len() != 64 {
            return Err(format!(
                "Test case {}: Invalid board string length: {} (expected 64)",
                case.no,
                case.board_str.len()
            ));
        }

        // Check that best moves are provided
        if case.best_moves.is_empty() {
            return Err(format!("Test case {}: No best moves specified", case.no));
        }
    }

    Ok(())
}

/// Get all 79 FFO test cases
pub fn get_test_cases() -> Vec<TestCase> {
    #[rustfmt::skip]
    let cases = vec![
        TestCase::new(1, "--XXXXX--OOOXX-O-OOOXXOX-OXOXOXXOXXXOXXX--XOXOXX-XXXOOO--OOOOO--", "X", 18, "G8", "H1", "H7,A2"),
        TestCase::new(2, "-XXXXXX---XOOOO--XOXXOOX-OOOOOOOOOOOXXOOOOOXXOOX--XXOO----XXXXX-", "X", 10, "A4", "B2", "A3"),
        TestCase::new(3, "----OX----OOXX---OOOXX-XOOXXOOOOOXXOXXOOOXXXOOOOOXXXXOXO--OOOOOX", "X", 2, "D1", "G3", "B8"),
        TestCase::new(4, "-XXXXXX-X-XXXOO-XOXXXOOXXXOXOOOX-OXOOXXX--OOOXXX--OOXX----XOXXO-", "X", 0, "H8,A5", "B6,B7", "A6"),
        TestCase::new(5, "-OOOOO----OXXO-XXXOXOXX-XXOXOXXOXXOOXOOOXXXXOO-OX-XOOO---XXXXX--", "X", 32, "G8", "G2", "B2"),
        TestCase::new(6, "--OXXX--OOOXXX--OOOXOXO-OOXOOOX-OOXXXXXXXOOXXOX--OOOOX---XXXXXX-", "X", 14, "A1,H3", "A8", "H2,G2"),
        TestCase::new(7, "--OXXO--XOXXXX--XOOOXXXXXOOXXXXXXOOOOXXX-XXXXXXX--XXOOO----XXOO-", "X", 8, "A6", "G1", "A1"),
        TestCase::new(8, "---X-X--X-XXXX--XXXXOXXXXXXOOOOOXXOXXXO-XOXXXXO-XOOXXX--XOOXXO--", "O", 8, "E1", "H2,G2,B2,G7", "B1"),
        TestCase::new(9, "--XOXX--O-OOXXXX-OOOXXXX-XOXXXOXXXOXOOOXOXXOXOXX--OXOO----OOOO--", "O", -8, "G7,A4", "B1,A7", "B7"),
        TestCase::new(10, "-XXXX-----OXXX--XOXOXOXXOXOXXOXXOXXOXOOOXXXOXOOX--OXXO---OOOOO--", "O", 10, "B2", "B7", "F1"),
        TestCase::new(11, "---O-XOX----XXOX---XXOOXO-XXOXOXXXXOOXOX-XOOXXXXXOOOXX-XOOOOOOO-", "O", 30, "B3", "C2", "A6"),
        TestCase::new(12, "--O--O--X-OOOOX-XXOOOXOOXXOXOXOOXXOXXOOOXXXXOOOO--OXXX---XXXXX--", "O", -8, "B7", "A7", "G7,G8"),
        TestCase::new(13, "--XXXXX--OOOXX---OOOXXXX-OXOXOXXOXXXOXXX--XOXOXX--OXOOO--OOOOO--", "X", 14, "B7", "A4", "A3"),
        TestCase::new(14, "--XXXXX---OOOX---XOOXXXX-OOOOOOOOOOXXXOOOOOXXOOX--XXOO----XXXXX-", "X", 18, "A3", "A4", "B1"),
        TestCase::new(15, "----O------OOX---OOOXX-XOOOXOOOOOXXOXXOOOXXXOOOOOXXXOOXO--OOOOOX", "X", 4, "G3,B8", "F1,C1", "C2"),
        TestCase::new(16, "-XXXXXX-X-XXXOO-XOXXXOOXXOOXXXOX-OOOXXXX--OOXXXX---OOO----XOX-O-", "X", 24, "F8", "C7", "A5,H1"),
        TestCase::new(17, "-OOOOO----OXXO-XXXOOOXX-XXOXOXXOXXOOXOOOXXXXOO-OX-XOO----XXXX---", "X", 8, "F8", "G2", "G6"),
        TestCase::new(18, "-XXX------OOOX--XOOOOOXXOXOXOOXXOXXOOOOOXXXOXOOX--OXXO---OOOOO--", "X", -2, "G2", "B7", "F1"),
        TestCase::new(19, "--OXXO--XOXXXX--XOOOOXXXXOOOXXXXX-OOOXXX--OOOOXX--XXOOO----XXOO-", "X", 8, "B6", "H8", "B7"),
        TestCase::new(20, "XXXOXXXXOXXXXXXXOOXXXXXXOOOXXXXXOOOXXOO-OOOOO---OOOOOOO-OOOOOOO-", "X", 6, "H5", "G6", "F6"),
        TestCase::new(21, "OOOOOOOOXOOXXX--XXOOXOO-XOXOOO--XOOOOX--XOOXOO--XOOOOO--XXXX----", "O", 0, "G5", "G2", "G4"),
        TestCase::new(22, "--OOOO--X-OOOOO-XXOOXOXXXOXOXXXXXXXOXXXX-XXOXOXX--OXXX-X----X---", "O", 2, "G8", "A6", "F8,A7,H2"),
        TestCase::new(23, "--O-------OOX---OOOXXXO-OOOOXOXXXXXOOXOXXXXXXOOXX-XXXXOX--XXXX--", "X", 4, "A2", "D1,H3", "B1,G2,E1"),
        TestCase::new(24, "--O--O-----OOOX--X-XOXOO--XXXOOOXXXXOOOOXXXOXXOOXXXXXX--XOXX-O--", "O", 0, "C3", "B4", "C2"),
        TestCase::new(25, "----X------XXXO--OOOXXXXXOOOOXXO-XXOOXXOOOXOXXXXOOOXX---X-XXXX--", "O", 0, "G1,A5", "F1", "D1"),
        TestCase::new(26, "-OOOOO----OXXO---OOOOXXO-OOOXOXX-OOXOOXX-XOXXOXX--O-XXXX--O----O", "X", 0, "D8", "A6", "A4,B7"),
        TestCase::new(27, "--XO-O----OOOO--OOXOXXO-OOOOXXOOOOOXXOX-OXOXXXXX--XXXX----X-O-X-", "X", -2, "B7", "E1", "B1"),
        TestCase::new(28, "--O-------OOO--X-XOOOOXXXXXXOXOX-XXOXOOXXXOXOOXX-OOOOO-X---OOO--", "X", 0, "F1,B2,E1", "B1", "F2,G7"),
        TestCase::new(29, "-OXXXX----OXXO--XXOOXOOOXXXOOXOOXXOOXOOOXXXXOO-XX-XXO-----------", "X", 10, "G2", "A1", "G6"),
        TestCase::new(30, "-XXX----X-XOO---XXOXOO--XOXOXO--XOOXOXXXXOOXXOX---OOOOO--XXXXX--", "X", 0, "G3", "G2", "E1"),
        TestCase::new(31, "-OOOOO----OOOO--OXXOOO---XXXOO--XXXXXXO-XXXOOO-OX-OOOO---OOOOO--", "X", -2, "G6", "G3", "G4"),
        TestCase::new(32, "--XX----O-XXOX--OOXOO---OXOXOOO-OOXXOOOXOOXXXOOX--XXXXOX--X--X-X", "X", -4, "G3", "B7", "E1"),
        TestCase::new(33, "-XXXXXXX--XOOO----OXOOXX-OOXXOXX-OOOOOXX-X-XOOXX---O-X-X--OOOO--", "X", -8, "E7,A3", "A6,B2,G7,G2", "A4"),
        TestCase::new(34, "-------------O-O-OOOOOOOOOOOOXOOOXXOOOXO-XXXOXOO--XXXOXO--OXXXXO", "X", -2, "C2", "D2,E2", "A3,A2"),
        TestCase::new(35, "--XXX-----XXXX-OOOXXOOOOOOOOOOXO-OOXXXXO-OOOXXXO---XOXX---X-----", "O", 0, "C7", "D8,H8", "B2"),
        TestCase::new(36, "---X-O----XXXO-XXXXXXXXXXOOXXOOXXOXOOOXXXXOOOO-XX--OOOO---------", "O", 0, "B7", "B1", "E1"),
        TestCase::new(37, "--OOOO--O-OOOO--OXXXOOO-OXXOXO--OOXXOXX-OOXXXX--O-XXX-----XX-O--", "X", -20, "G2", "G4,B7,H3", "G1"),
        TestCase::new(38, "--OOOO----OOOO---XOXXOOXOOXOOOOX-OOOOOXXXOOXXXXX--X-X-----------", "X", 4, "B2", "A5", "H2"),
        TestCase::new(39, "O-OOOO--XOXXOX--XOOOXXX-XOOOXX--XOOXOX--XOXXX---X-XX------------", "O", 64, "A8,B1,G1,G5,G6,C8,H3,E8,H4", "F7,D8,E7,H2,B8", "G2,G4"),
        TestCase::new(40, "O--OOOOX-OOOOOOXOOXXOOOXOOXOOOXXOOOOOOXX---OOOOX----O--X--------", "X", 38, "A2", "C7", "D8"),
        TestCase::new(41, "-OOOOO----OOOOX--OOOOOO-XXXXXOO--XXOOX--OOXOXX----OXXO---OOO--O-", "X", 0, "H4", "H3,F8", "G5"),
        TestCase::new(42, "--OOO-------XX-OOOOOOXOO-OOOOXOOX-OOOXXO---OOXOO---OOOXO--OOOO--", "X", 6, "G2", "A4", "C6"),
        TestCase::new(43, "--XXXXX---XXXX---OOOXX---OOXXXX--OOXXXO-OOOOXOO----XOX----XXXXX-", "O", -12, "G3,C7", "H4", "G7"),
        TestCase::new(44, "--O-X-O---O-XO-O-OOXXXOOOOOOXXXOOOOOXX--XXOOXO----XXXX-----XXX--", "O", -14, "D2,B8", "G2,G6", "F1"),
        TestCase::new(45, "---XXXX-X-XXXO--XXOXOO--XXXOXO--XXOXXO---OXXXOO-O-OOOO------OO--", "X", 6, "B2", "G5", "H6"),
        TestCase::new(46, "---XXX----OOOX----OOOXX--OOOOXXX--OOOOXX--OXOXXX--XXOO---XXXX-O-", "X", -8, "B3", "B7", "A3"),
        TestCase::new(47, "-OOOOO----OOOO---OOOOX--XXXXXX---OXOOX--OOOXOX----OOXX----XXXX--", "O", 4, "G2", "G6", "G4"),
        TestCase::new(48, "-----X--X-XXX---XXXXOO--XOXOOXX-XOOXXX--XOOXX-----OOOX---XXXXXX-", "O", 28, "F6", "G5", "G6"),
        TestCase::new(49, "--OX-O----XXOO--OOOOOXX-OOOOOX--OOOXOXX-OOOOXX-----OOX----X-O---", "X", 16, "E1", "B1", "B2"),
        TestCase::new(50, "----X-----XXX----OOOXOOO-OOOXOOO-OXOXOXO-OOXXOOO--OOXO----O--O--", "X", 10, "D8", "H7", "A4,B2"),
        TestCase::new(51, "----O-X------X-----XXXO-OXXXXXOO-XXOOXOOXXOXXXOO--OOOO-O----OO--", "O", 6, "E2,A3", "F1", "G7"),
        TestCase::new(52, "---X-------OX--X--XOOXXXXXXOXXXXXXXOOXXXXXXOOOXX--XO---X--------", "O", 0, "A3", "E1,B3", "F2"),
        TestCase::new(53, "----OO-----OOO---XXXXOOO--XXOOXO-XXXXXOO--OOOXOO--X-OX-O-----X--", "X", -2, "D8", "C1", "E8"),
        TestCase::new(54, "--OOO---XXOO----XXXXOOOOXXXXOX--XXXOXX--XXOOO------OOO-----O----", "X", -2, "C7", "F8,F2,C8", "F6"),
        TestCase::new(55, "--------X-X------XXXXOOOOOXOXX--OOOXXXX-OOXXXX--O-OOOX-----OO---", "O", 0, "G6,B7,E2,G4", "F2,H4,D2", "C1,H5"),
        TestCase::new(56, "--XXXXX---XXXX---OOOXX---OOXOX---OXXXXX-OOOOOXO----OXX----------", "O", 2, "H5", "F8", "G3"),
        TestCase::new(57, "-------------------XXOOO--XXXOOO--XXOXOO-OOOXXXO--OXOO-O-OOOOO--", "X", -10, "A6", "F2", "B5"),
        TestCase::new(58, "--XOOO----OOO----OOOXOO--OOOOXO--OXOXXX-OOXXXX----X-XX----------", "X", 4, "G1", "A5", "B2"),
        TestCase::new(59, "-----------------------O--OOOOO---OOOOOXOOOOXXXX--XXOOXX--XX-O-X", "X", 64, "H4,G8,E8", "A5", "B3"),
        TestCase::new(60, "---OOOO----OOO----XOXOXX--XOOXXX--XOOXXX--XOOOXX--OXXX-X--XXXX--", "X", 20, "C2", "B8", "C1"),
        TestCase::new(61, "-XXXX---X-XXOX--XXXXOXX-XOOXOOOOXOOOOOO-XXOOOO--X---O-----------", "O", -14, "H3,G1", "F1", "B7"),
        TestCase::new(62, "--OOOO----OOXX----OXXXXXXXOXXOOO-XXXXOO-OXXXXXXO----X-----------", "O", 28, "E8", "D7,H7", "F7"),
        TestCase::new(63, "--X-------X-X----OXXXX---OXXXXO-OOXOXOOOOOOOOXO---XOXX-----XXXX-", "O", -2, "F2", "B8", "D2"),
        TestCase::new(64, "--O--X----O--X-O-OOXXXOO--XXXXXO--XXOOOO-XXXXXX---XXX----X-OOO--", "O", 20, "B4", "E1,E2,B5", "D2"),
        TestCase::new(65, "----OO----OOOOX---OXXXX-O-OXXXX--OOXXOX-XXOXXXX---OOOO-------O--", "X", 10, "G1", "C8,A5", "B7"),
        TestCase::new(66, "-OOO----X-OXX---XXOXXOO-XOXXOO--XXOOOO--XXOOOO----OOO-----O-----", "X", 30, "H3", "G4", "F8"),
        TestCase::new(67, "-XXXXX----XOXX--OOOXOXO--OOOXOOO-OOOXXO---OOOX-O---OX-----------", "X", 22, "H3", "C8,A6", "B6,D8"),
        TestCase::new(68, "---OOO----OOOO----OXXOOX-OOXXOX--OOXXXX--XOOXX----OOO--------O--", "X", 28, "E8", "A5", "A6,A4"),
        TestCase::new(69, "--OOOO-----OOO---OOOOO--XXOXXOO--OXOXOO-OXXXXXX---X-X-----------", "X", 0, "H3", "H5,A2", "A5,G2"),
        TestCase::new(70, "---X----X-XXX---XXXX----XXXOOO--XXXXOO--XXOOXXX-X-OOXX----O-----", "X", -24, "E3", "E8,D8,G5", "F3"),
        TestCase::new(71, "------------------XXXXX--XXXXXO--OXXXOOX--OXOXXX--OOXX-X---XXXX-", "O", 20, "D2", "F2", "B3"),
        TestCase::new(72, "---O------OOXX---XXOXXX-XXXXOOXX-XXXXOO---XXXOO----XX-------X---", "O", 24, "E1", "A3,A6", "C8"),
        TestCase::new(73, "--X--X----XXX---OOXXXX---OOXXX---OXOXXO-OOOXXXX-O--OXO----------", "O", -4, "G4", "D8,H5", "H6,F8"),
        TestCase::new(74, "----X-----OXXO-X--OXOOXX-OOXXOXX--OXXO-X--XXOO----XOOO-------O--", "O", -30, "F1", "B5,C1", "B6"),
        TestCase::new(75, "----O-------OO----XXOX-O-XXXOXOO--OOOOO---OOOXOX--OOOOX------O--", "X", 14, "D2", "H5", "D1"),
        TestCase::new(76, "---O------OO-O-----OOOX-OOOOOOX--XXXXOXX--OOOOOO--OOO-------O---", "X", 32, "A3", "F7,E1,C1", "H7,F8"),
        TestCase::new(77, "--O-OX--X-OOO---XXOOO---XXOXOOOO-OOOOO--O-X-O-----OX------------", "X", 34, "B7", "C8", "B6"),
        TestCase::new(78, "----O-----OOOO---OOOX-X-OOXOXXXX-XOOX---XOOO-X----OO-------O----", "X", 8, "F1", "A7", "C8"),
        TestCase::new(79, "--------------X-----O-XX---OOOX-OOOOXOXX--OOOOOO--O-OO-O----OO--", "X", 64, "D7", "D8", "H8"),
    ];

    // Validate test cases in debug builds
    #[cfg(debug_assertions)]
    if let Err(e) = validate_test_cases(&cases) {
        panic!("Test case validation failed: {}", e);
    }

    cases
}
