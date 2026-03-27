//! Reversi board representation using bitboards.

use std::cmp::Ordering;
use std::fmt;
use std::hash::Hash;

use crate::bitboard::Bitboard;
use crate::constants::SCORE_MAX;
use crate::disc::Disc;
use crate::flip;
use crate::square::Square;
use crate::types::{ScaledScore, Score};

/// Represents a Reversi board using player/opponent [`Bitboard`] pairs.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    /// Bitboard representing the player's discs.
    pub player: Bitboard,
    /// Bitboard representing the opponent's discs.
    pub opponent: Bitboard,
}

impl Default for Board {
    /// Creates a board with the standard Reversi starting position.
    ///
    /// The initial position has:
    /// - Black discs on D5 and E4
    /// - White discs on D4 and E5
    fn default() -> Self {
        Board {
            player: Square::D5.bitboard() | Square::E4.bitboard(),
            opponent: Square::D4.bitboard() | Square::E5.bitboard(),
        }
    }
}

impl Board {
    /// Creates a new [`Board`] with the standard initial position.
    pub fn new() -> Board {
        Default::default()
    }

    /// Creates a [`Board`] from given bitboards.
    ///
    /// # Panics
    ///
    /// In debug builds only, panics if `player` and `opponent` overlap.
    /// In release builds, overlapping bitboards produce an invalid board state without panicking.
    pub fn from_bitboards(player: impl Into<Bitboard>, opponent: impl Into<Bitboard>) -> Board {
        let player = player.into();
        let opponent = opponent.into();
        debug_assert!(
            (player & opponent).is_empty(),
            "player and opponent bitboards must not overlap"
        );
        Board { player, opponent }
    }

    /// Parses a [`Board`] from a 64-character string (`'X'`/`'O'`/`'-'`).
    ///
    /// `current_player` determines which character maps to `player`:
    /// if [`Disc::Black`], `'X'` → player and `'O'` → opponent;
    /// if [`Disc::White`], `'O'` → player and `'X'` → opponent.
    ///
    /// # Errors
    ///
    /// - [`BoardError::InvalidPlayer`] if `current_player` is [`Disc::Empty`].
    /// - [`BoardError::TooShort`] if the string has fewer than 64 characters.
    /// - [`BoardError::TooLong`] if the string has more than 64 characters.
    /// - [`BoardError::InvalidChar`] if the string contains an invalid character.
    pub fn from_string(board_string: &str, current_player: Disc) -> Result<Board, BoardError> {
        if current_player == Disc::Empty {
            return Err(BoardError::InvalidPlayer);
        }

        let char_count = board_string.chars().count();

        if char_count < 64 {
            return Err(BoardError::TooShort {
                expected: 64,
                actual: char_count,
            });
        }
        if char_count > 64 {
            return Err(BoardError::TooLong {
                expected: 64,
                actual: char_count,
            });
        }

        let player_char = current_player.to_char();
        let opponent_char = current_player.opposite().to_char();

        let mut player = Bitboard::new(0);
        let mut opponent = Bitboard::new(0);

        for (square, c) in Square::iter().zip(board_string.chars()) {
            if c == player_char {
                player = player.set(square);
            } else if c == opponent_char {
                opponent = opponent.set(square);
            } else if c != '-' {
                return Err(BoardError::InvalidChar {
                    char: c,
                    position: square.index(),
                });
            }
        }

        Ok(Board { player, opponent })
    }

    /// Returns the disc at `sq` from the perspective of `side_to_move`.
    #[inline]
    pub fn get_disc_at(&self, sq: Square, side_to_move: Disc) -> Disc {
        if self.player.contains(sq) {
            side_to_move
        } else if self.opponent.contains(sq) {
            side_to_move.opposite()
        } else {
            Disc::Empty
        }
    }

    /// Returns whether the game is over (neither player can move).
    #[inline]
    pub fn is_game_over(&self) -> bool {
        if self.has_legal_moves() {
            return false;
        }

        let switched = self.switch_players();
        !switched.has_legal_moves()
    }

    /// Returns a [`Bitboard`] of empty squares.
    #[inline(always)]
    pub fn get_empty(&self) -> Bitboard {
        !(self.player | self.opponent)
    }

    /// Returns the number of discs the current player has on the board.
    #[inline(always)]
    pub fn get_player_count(&self) -> u32 {
        self.player.count()
    }

    /// Returns the number of discs the opponent has on the board.
    #[inline(always)]
    pub fn get_opponent_count(&self) -> u32 {
        self.opponent.count()
    }

    /// Returns the number of empty squares on the board.
    #[inline(always)]
    pub fn get_empty_count(&self) -> u32 {
        self.get_empty().count()
    }

    /// Returns the disc-difference score without any assertions.
    #[inline(always)]
    fn disc_score(&self) -> Score {
        self.get_player_count() as Score * 2 - SCORE_MAX
    }

    /// Returns the final score (disc difference) from the current player's perspective.
    ///
    /// # Panics
    ///
    /// In debug builds only, panics if the board has empty squares.
    /// In release builds, empty squares produce an incorrect score without panicking.
    /// Use [`solve`](Self::solve) if the board may have empty squares.
    #[inline(always)]
    pub fn final_score(&self) -> Score {
        debug_assert!(
            self.get_empty().is_empty(),
            "final_score requires no empty squares; use solve() for boards with empties"
        );
        self.disc_score()
    }

    /// Returns the final score as a [`ScaledScore`].
    ///
    /// This is equivalent to `ScaledScore::from_disc_diff(self.final_score())`.
    #[inline(always)]
    pub fn final_score_scaled(&self) -> ScaledScore {
        ScaledScore::from_disc_diff(self.final_score())
    }

    /// Calculates the final score when both players have passed.
    ///
    /// Unlike [`final_score`](Self::final_score), handles positions with remaining
    /// empties by awarding them to the player with more discs.
    /// The caller must pass the actual empty count; no validation is performed.
    #[inline(always)]
    pub fn solve(&self, n_empties: u32) -> Score {
        debug_assert!(
            n_empties == self.get_empty_count(),
            "n_empties ({n_empties}) does not match actual empty count ({})",
            self.get_empty_count()
        );
        let score = self.disc_score();
        let diff = score + n_empties as Score;

        match diff.cmp(&0) {
            Ordering::Equal => diff,
            Ordering::Greater => diff + n_empties as Score,
            Ordering::Less => score,
        }
    }

    /// Calculates the final score as a [`ScaledScore`] when both players have passed.
    ///
    /// This is equivalent to `ScaledScore::from_disc_diff(self.solve(n_empties))`.
    #[inline(always)]
    pub fn solve_scaled(&self, n_empties: u32) -> ScaledScore {
        ScaledScore::from_disc_diff(self.solve(n_empties))
    }

    /// Returns a new [`Board`] with the player and opponent swapped.
    #[inline(always)]
    pub fn switch_players(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
        }
    }

    /// Attempts to make a move at `sq`, returning [`None`] if the move is invalid.
    #[inline(always)]
    pub fn try_make_move(&self, sq: Square) -> Option<Board> {
        if !self.opponent.has_adjacent_bit(sq) {
            return None;
        }

        let flipped = flip::flip(sq, self.player, self.opponent);
        if flipped.is_empty() {
            return None;
        }

        Some(Board {
            player: self.opponent.apply_flip(flipped),
            opponent: self.player.apply_move(flipped, sq),
        })
    }

    /// Makes a move at `sq` for the current player.
    ///
    /// # Panics
    ///
    /// In debug builds only, panics if `sq` is not a valid move.
    /// In release builds, invalid moves produce an incorrect board state without panicking.
    /// Use [`is_legal_move`](Self::is_legal_move) or [`try_make_move`](Self::try_make_move)
    /// if validity is uncertain.
    #[inline(always)]
    pub fn make_move(&self, sq: Square) -> Board {
        let flipped = flip::flip(sq, self.player, self.opponent);
        debug_assert!(!flipped.is_empty());
        Board {
            player: self.opponent.apply_flip(flipped),
            opponent: self.player.apply_move(flipped, sq),
        }
    }

    /// Makes a move at `sq` using pre-computed `flipped` discs.
    ///
    /// # Panics
    ///
    /// In debug builds only, panics if:
    /// - `flipped` is empty
    /// - `sq` is not empty
    /// - `flipped` contains discs not belonging to the opponent
    ///
    /// In release builds, invalid inputs produce an incorrect board state without panicking.
    #[inline(always)]
    pub fn make_move_with_flipped(&self, flipped: Bitboard, sq: Square) -> Board {
        debug_assert!(!flipped.is_empty(), "flipped must not be empty");
        debug_assert!(self.get_empty().contains(sq), "sq must be an empty square");
        debug_assert!(
            (flipped & !self.opponent).is_empty(),
            "flipped must be a subset of opponent's discs"
        );
        Board {
            player: self.opponent.apply_flip(flipped),
            opponent: self.player.apply_move(flipped, sq),
        }
    }

    /// Returns a [`Bitboard`] of legal moves for the current player.
    #[inline(always)]
    pub fn get_moves(&self) -> Bitboard {
        self.player.get_moves(self.opponent)
    }

    /// Returns whether the current player has any legal moves.
    #[inline(always)]
    pub fn has_legal_moves(&self) -> bool {
        !self.get_moves().is_empty()
    }

    /// Returns whether placing a disc at `sq` is a legal move.
    #[inline(always)]
    pub fn is_legal_move(&self, sq: Square) -> bool {
        self.get_moves().contains(sq)
    }

    /// Returns the number of stable discs for the current player.
    ///
    /// Stable discs cannot be flipped for the remainder of the game (e.g. corners,
    /// discs protected by filled edges or other stable discs).
    #[inline]
    pub fn get_stability(&self) -> u32 {
        crate::stability::get_stable_discs(self.player, self.opponent).count()
    }

    /// Returns a [`Bitboard`] of potential moves for the current player.
    #[inline(always)]
    pub fn get_potential_moves(&self) -> Bitboard {
        self.player.get_potential_moves(self.opponent)
    }

    /// Returns `(legal_moves, potential_moves)` for the current player.
    #[inline(always)]
    pub fn get_moves_and_potential(&self) -> (Bitboard, Bitboard) {
        self.player.get_moves_and_potential(self.opponent)
    }

    /// Returns whether `sq` is empty.
    #[inline]
    pub fn is_square_empty(&self, sq: Square) -> bool {
        self.get_empty().contains(sq)
    }

    /// Calculates a hash of the current board position.
    #[inline]
    pub fn hash(&self) -> u64 {
        use rapidhash::v3;
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&self.player.bits().to_ne_bytes());
        bytes[8..].copy_from_slice(&self.opponent.bits().to_ne_bytes());
        v3::rapidhash_v3_nano_inline::<true, false>(&bytes, &v3::DEFAULT_RAPID_SECRETS)
    }

    /// Applies a [`Bitboard`] transformation to both player and opponent.
    #[inline(always)]
    fn transform(&self, f: fn(Bitboard) -> Bitboard) -> Board {
        Board {
            player: f(self.player),
            opponent: f(self.opponent),
        }
    }

    /// Rotates the board 90 degrees clockwise.
    #[inline]
    pub fn rotate_90_clockwise(&self) -> Board {
        self.transform(Bitboard::rotate_90_clockwise)
    }

    /// Rotates the board 180 degrees.
    #[inline]
    pub fn rotate_180_clockwise(&self) -> Board {
        self.transform(Bitboard::rotate_180_clockwise)
    }

    /// Rotates the board 270 degrees clockwise.
    #[inline]
    pub fn rotate_270_clockwise(&self) -> Board {
        self.transform(Bitboard::rotate_270_clockwise)
    }

    /// Flips the board vertically (top to bottom).
    #[inline]
    pub fn flip_vertical(&self) -> Board {
        self.transform(Bitboard::flip_vertical)
    }

    /// Flips the board horizontally (left to right).
    #[inline]
    pub fn flip_horizontal(&self) -> Board {
        self.transform(Bitboard::flip_horizontal)
    }

    /// Flips the board along the main diagonal (A1-H8).
    #[inline]
    pub fn flip_diag_a1h8(&self) -> Board {
        self.transform(Bitboard::flip_diag_a1h8)
    }

    /// Flips the board along the anti-diagonal (A8-H1).
    #[inline]
    pub fn flip_diag_a8h1(&self) -> Board {
        self.transform(Bitboard::flip_diag_a8h1)
    }

    /// Returns the canonical (unique) form of this board.
    ///
    /// The canonical form is the [`Board`] with the lexicographically smallest
    /// `(player.bits(), opponent.bits())` tuple among all 8 symmetric variants
    /// (original, 3 rotations, and 4 reflections).
    #[inline]
    pub fn unique(&self) -> Board {
        let candidates = [
            self.rotate_90_clockwise(),
            self.rotate_180_clockwise(),
            self.rotate_270_clockwise(),
            self.flip_horizontal(),
            self.flip_vertical(),
            self.flip_diag_a1h8(),
            self.flip_diag_a8h1(),
        ];

        let mut result = *self;
        for candidate in candidates {
            if (candidate.player.bits(), candidate.opponent.bits())
                < (result.player.bits(), result.opponent.bits())
            {
                result = candidate;
            }
        }

        result
    }

    /// Converts the board to an 8x8 string representation (`'X'`/`'O'`/`'-'`).
    pub fn to_string_as_board(&self, current_player: Disc) -> String {
        let mut s = String::with_capacity(64 + 8);
        for (i, sq) in Square::iter().enumerate() {
            if i > 0 && i % 8 == 0 {
                s.push('\n');
            }
            if self.player.contains(sq) {
                s.push(current_player.to_char());
            } else if self.opponent.contains(sq) {
                s.push(current_player.opposite().to_char());
            } else {
                s.push(Disc::Empty.to_char());
            }
        }
        s
    }
}

impl fmt::Display for Board {
    /// Formats the board for display, showing Black as the current player.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_as_board(Disc::Black))
    }
}

/// Error type for [`Board::from_string`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoardError {
    /// Current player is [`Disc::Empty`].
    InvalidPlayer,
    /// String is too short (less than 64 characters).
    TooShort {
        /// Expected number of characters.
        expected: usize,
        /// Actual number of characters.
        actual: usize,
    },
    /// String is too long (more than 64 characters).
    TooLong {
        /// Expected number of characters.
        expected: usize,
        /// Actual number of characters.
        actual: usize,
    },
    /// Invalid character at position.
    InvalidChar {
        /// The invalid character.
        char: char,
        /// Position in the string (0-indexed).
        position: usize,
    },
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoardError::InvalidPlayer => {
                write!(f, "Invalid player: current_player must be Black or White")
            }
            BoardError::TooShort { expected, actual } => {
                write!(
                    f,
                    "Board string too short: expected {expected} characters, got {actual}"
                )
            }
            BoardError::TooLong { expected, actual } => {
                write!(
                    f,
                    "Board string too long: expected {expected} characters, got {actual}"
                )
            }
            BoardError::InvalidChar { char, position } => {
                write!(
                    f,
                    "Invalid character '{char}' at position {position}: must be 'X', 'O', or '-'"
                )
            }
        }
    }
}

impl std::error::Error for BoardError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_board() {
        let board = Board::default();
        assert_eq!(board.get_player_count(), 2);
        assert_eq!(board.get_opponent_count(), 2);
        assert_eq!(board.get_empty_count(), 60);
    }

    #[test]
    fn test_new_board() {
        let board = Board::new();
        // Should be same as default
        assert_eq!(board, Board::default());
    }

    #[test]
    fn test_from_bitboards() {
        let player = Square::A1.bitboard();
        let opponent = Square::H8.bitboard();
        let board = Board::from_bitboards(player, opponent);
        assert!(board.player.contains(Square::A1));
        assert!(board.opponent.contains(Square::H8));
    }

    #[test]
    fn test_from_string() {
        let board_string = "--------\
                            --------\
                            --------\
                            ---OX---\
                            ---XO---\
                            --------\
                            --------\
                            --------";
        let board = Board::from_string(board_string, Disc::Black).unwrap();
        assert!(board.player.contains(Square::D5));
        assert!(board.player.contains(Square::E4));
        assert!(board.opponent.contains(Square::D4));
        assert!(board.opponent.contains(Square::E5));
    }

    #[test]
    fn test_from_string_white_perspective() {
        let board_string = "--------\
                            --------\
                            --------\
                            ---XO---\
                            ---OX---\
                            --------\
                            --------\
                            --------";
        let board = Board::from_string(board_string, Disc::White).unwrap();
        // From White's perspective, O discs are the player
        assert!(board.player.contains(Square::E4));
        assert!(board.player.contains(Square::D5));
        assert!(board.opponent.contains(Square::D4));
        assert!(board.opponent.contains(Square::E5));
    }

    #[test]
    fn test_get_disc_at() {
        let board = Board::new();

        // Check discs from Black's perspective (Black is the first player)
        assert_eq!(board.get_disc_at(Square::D5, Disc::Black), Disc::Black);
        assert_eq!(board.get_disc_at(Square::E4, Disc::Black), Disc::Black);
        assert_eq!(board.get_disc_at(Square::D4, Disc::Black), Disc::White);
        assert_eq!(board.get_disc_at(Square::E5, Disc::Black), Disc::White);
        assert_eq!(board.get_disc_at(Square::A1, Disc::Black), Disc::Empty);

        // Switch to White's perspective
        let white_board = board.switch_players();
        assert_eq!(
            white_board.get_disc_at(Square::D5, Disc::White),
            Disc::Black
        );
        assert_eq!(
            white_board.get_disc_at(Square::E4, Disc::White),
            Disc::Black
        );
        assert_eq!(
            white_board.get_disc_at(Square::D4, Disc::White),
            Disc::White
        );
        assert_eq!(
            white_board.get_disc_at(Square::E5, Disc::White),
            Disc::White
        );
    }

    #[test]
    fn test_is_game_over() {
        // Initial position - not over
        let board = Board::new();
        assert!(!board.is_game_over());

        // Full board - game over
        let full_board = Board::from_bitboards(u64::MAX, 0);
        assert!(full_board.is_game_over());

        // Empty board - game over (no moves)
        let empty_board = Board::from_bitboards(0, 0);
        assert!(empty_board.is_game_over());
    }

    #[test]
    fn test_get_empty() {
        let board = Board::new();
        let empty = board.get_empty();

        assert!(!empty.contains(Square::D4));
        assert!(!empty.contains(Square::E5));
        assert!(!empty.contains(Square::D5));
        assert!(!empty.contains(Square::E4));
        assert!(empty.contains(Square::A1));
    }

    #[test]
    fn test_counts() {
        let board = Board::new();
        assert_eq!(board.get_player_count(), 2);
        assert_eq!(board.get_opponent_count(), 2);
        assert_eq!(board.get_empty_count(), 60);

        // Custom board
        let custom = Board::from_bitboards(
            Square::A1.bitboard() | Square::A2.bitboard() | Square::A3.bitboard(),
            Square::H8.bitboard(),
        );
        assert_eq!(custom.get_player_count(), 3);
        assert_eq!(custom.get_opponent_count(), 1);
        assert_eq!(custom.get_empty_count(), 60);
    }

    #[test]
    fn test_switch_players() {
        let board = Board::new();
        let switched_board = board.switch_players();

        assert!(switched_board.player.contains(Square::D4));
        assert!(switched_board.player.contains(Square::E5));
        assert!(switched_board.opponent.contains(Square::D5));
        assert!(switched_board.opponent.contains(Square::E4));

        // Double switch should return to original
        let double_switched = switched_board.switch_players();
        assert_eq!(board, double_switched);
    }

    #[test]
    fn test_try_make_move() {
        let board = Board::new();

        // Valid move
        let result = board.try_make_move(Square::D3);
        assert!(result.is_some());
        let new_board = result.unwrap();
        assert!(new_board.opponent.contains(Square::D3));
        assert!(new_board.opponent.contains(Square::D4));

        // Invalid move - no adjacent opponent discs
        let result = board.try_make_move(Square::A1);
        assert!(result.is_none());

        // Invalid move - occupied square
        let result = board.try_make_move(Square::D4);
        assert!(result.is_none());
    }

    #[test]
    fn test_make_move() {
        let board = Board::new();

        // Make a valid move
        let new_board = board.make_move(Square::D3);
        assert!(new_board.opponent.contains(Square::D3));
        assert!(new_board.opponent.contains(Square::D4));
        assert_eq!(new_board.get_opponent_count(), 4); // 2 original + 1 new + 1 flipped
        assert_eq!(new_board.get_player_count(), 1); // 2 original - 1 flipped
    }

    #[test]
    fn test_make_move_with_flipped() {
        let board = Board::new();
        let flipped = Square::D4.bitboard();
        let new_board = board.make_move_with_flipped(flipped, Square::D3);

        assert!(new_board.opponent.contains(Square::D3));
        assert!(new_board.opponent.contains(Square::D4));
    }

    #[test]
    fn test_get_moves() {
        let board = Board::new();
        let moves = board.get_moves();

        // Initial position has 4 legal moves
        assert_eq!(moves.count(), 4);
        assert!(moves.contains(Square::D3));
        assert!(moves.contains(Square::C4));
        assert!(moves.contains(Square::F5));
        assert!(moves.contains(Square::E6));
    }

    #[test]
    fn test_has_legal_moves() {
        let board = Board::new();
        assert!(board.has_legal_moves());

        // Board with no moves
        let no_moves = Board::from_bitboards(0, u64::MAX);
        assert!(!no_moves.has_legal_moves());
    }

    #[test]
    fn test_is_legal_move() {
        let board = Board::new();

        // Legal moves
        assert!(board.is_legal_move(Square::D3));
        assert!(board.is_legal_move(Square::C4));
        assert!(board.is_legal_move(Square::F5));
        assert!(board.is_legal_move(Square::E6));

        // Illegal moves
        assert!(!board.is_legal_move(Square::A1));
        assert!(!board.is_legal_move(Square::D4)); // Occupied
        assert!(!board.is_legal_move(Square::D2)); // No flip
    }

    #[test]
    fn test_is_square_empty() {
        let board = Board::new();

        assert!(board.is_square_empty(Square::A1));
        assert!(board.is_square_empty(Square::H8));
        assert!(!board.is_square_empty(Square::D4));
        assert!(!board.is_square_empty(Square::E5));
        assert!(!board.is_square_empty(Square::D5));
        assert!(!board.is_square_empty(Square::E4));
    }

    #[test]
    fn test_hash() {
        let board1 = Board::new();
        let board2 = Board::new();
        let board3 = Board::from_bitboards(1, 2);

        // Same boards should have same hash
        assert_eq!(board1.hash(), board2.hash());

        // Different boards should (likely) have different hashes
        assert_ne!(board1.hash(), board3.hash());

        // Switched boards should have different hashes
        let switched = board1.switch_players();
        assert_ne!(board1.hash(), switched.hash());
    }

    #[test]
    fn test_rotate_90_clockwise() {
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());
        let rotated = board.rotate_90_clockwise();

        // A1 -> H1, H8 -> A8
        assert!(rotated.player.contains(Square::H1));
        assert!(rotated.opponent.contains(Square::A8));

        // Four rotations should return to original
        let rotated4 = board
            .rotate_90_clockwise()
            .rotate_90_clockwise()
            .rotate_90_clockwise()
            .rotate_90_clockwise();
        assert_eq!(board, rotated4);
    }

    #[test]
    fn test_rotate_180_clockwise() {
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());
        let rotated = board.rotate_180_clockwise();

        // A1 -> H8, H8 -> A1
        assert!(rotated.player.contains(Square::H8));
        assert!(rotated.opponent.contains(Square::A1));

        // Two rotations should return to original
        let rotated2 = board.rotate_180_clockwise().rotate_180_clockwise();
        assert_eq!(board, rotated2);
    }

    #[test]
    fn test_rotate_270_clockwise() {
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());
        let rotated = board.rotate_270_clockwise();

        // A1 -> A8, H8 -> H1
        assert!(rotated.player.contains(Square::A8));
        assert!(rotated.opponent.contains(Square::H1));

        // Four rotations should return to original
        let rotated4 = board
            .rotate_270_clockwise()
            .rotate_270_clockwise()
            .rotate_270_clockwise()
            .rotate_270_clockwise();
        assert_eq!(board, rotated4);
    }

    #[test]
    fn test_flip_vertical() {
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());
        let flipped = board.flip_vertical();

        // A1 -> A8, H8 -> H1
        assert!(flipped.player.contains(Square::A8));
        assert!(flipped.opponent.contains(Square::H1));

        // Double flip returns to original
        let double_flipped = flipped.flip_vertical();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_flip_horizontal() {
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());
        let flipped = board.flip_horizontal();

        // A1 -> H1, H8 -> A8
        assert!(flipped.player.contains(Square::H1));
        assert!(flipped.opponent.contains(Square::A8));

        // Double flip returns to original
        let double_flipped = flipped.flip_horizontal();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_flip_diag_a1h8() {
        let board = Board::from_bitboards(Square::A2.bitboard(), Square::B1.bitboard());
        let flipped = board.flip_diag_a1h8();

        // A2 -> B1, B1 -> A2
        assert!(flipped.player.contains(Square::B1));
        assert!(flipped.opponent.contains(Square::A2));

        // Double flip returns to original
        let double_flipped = flipped.flip_diag_a1h8();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_flip_diag_a8h1() {
        let board = Board::from_bitboards(Square::A7.bitboard(), Square::B8.bitboard());
        let flipped = board.flip_diag_a8h1();

        // A7 -> B8, B8 -> A7
        assert!(flipped.player.contains(Square::B8));
        assert!(flipped.opponent.contains(Square::A7));

        // Double flip returns to original
        let double_flipped = flipped.flip_diag_a8h1();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_to_string_as_board() {
        let board = Board::new();

        // As Black
        let black_str = board.to_string_as_board(Disc::Black);
        assert!(black_str.contains("OX"));
        assert!(black_str.contains("XO"));

        // As White
        let white_str = board.to_string_as_board(Disc::White);
        assert!(white_str.contains("XO"));
        assert!(white_str.contains("OX"));

        // Check length and newlines
        let lines: Vec<&str> = black_str.split('\n').collect();
        assert_eq!(lines.len(), 8);
        assert_eq!(lines[0].len(), 8);
    }

    #[test]
    fn test_display() {
        let board = Board::new();
        let board_display = format!("{board}");
        let expected_display = "--------\n\
                                      --------\n\
                                      --------\n\
                                      ---OX---\n\
                                      ---XO---\n\
                                      --------\n\
                                      --------\n\
                                      --------";
        assert_eq!(board_display, expected_display);
    }

    #[test]
    fn test_board_equality() {
        let board1 = Board::new();
        let board2 = Board::new();
        let board3 = Board::from_bitboards(1, 2);

        assert_eq!(board1, board2);
        assert_ne!(board1, board3);
    }

    #[test]
    fn test_complex_game_sequence() {
        let mut board = Board::new();

        // Play a few moves
        board = board.make_move(Square::D3); // Black
        board = board.make_move(Square::C3); // White
        board = board.make_move(Square::C4); // Black
        board = board.make_move(Square::C5); // White

        // Verify board state
        assert!(board.get_player_count() > 0);
        assert!(board.get_opponent_count() > 0);
        assert!(!board.is_game_over());
        assert!(board.has_legal_moves());
    }

    #[test]
    fn test_final_score() {
        // Player wins 64-0
        let board = Board::from_bitboards(u64::MAX, 0);
        assert_eq!(board.final_score(), 64);

        // Opponent wins 0-64
        let board = Board::from_bitboards(0, u64::MAX);
        assert_eq!(board.final_score(), -64);

        // Draw 32-32
        let board = Board::from_bitboards(0x00000000FFFFFFFF, 0xFFFFFFFF00000000);
        assert_eq!(board.final_score(), 0);
    }

    #[test]
    fn test_solve_player_ahead() {
        // Player has 40 discs (bits 0-39), opponent has 20 discs (bits 40-59), 4 empties (bits 60-63)
        // Score = 40 * 2 - 64 = 16
        // diff = 16 + 4 = 20 > 0, so player gets all empties
        // Result = 20 + 4 = 24
        let board = Board::from_bitboards(0x000000FFFFFFFFFFu64, 0x0FFFFF0000000000u64);
        let player_count = board.get_player_count();
        let opponent_count = board.get_opponent_count();
        assert_eq!(player_count, 40);
        assert_eq!(opponent_count, 20);
        let n_empties = 64 - player_count - opponent_count;
        assert_eq!(n_empties, 4);
        let score = board.solve(n_empties);
        // Player ahead: gets all empty squares
        assert!(score > 0);
    }

    #[test]
    fn test_solve_opponent_ahead() {
        // Player has 20 discs (bits 0-19), opponent has 40 discs (bits 20-59), 4 empties (bits 60-63)
        // Score = 20 * 2 - 64 = -24
        // diff = -24 + 4 = -20 < 0, so player gets no empties
        // Result = -24
        let board = Board::from_bitboards(0x00000000000FFFFFu64, 0x0FFFFFFFFFF00000u64);
        let player_count = board.get_player_count();
        let opponent_count = board.get_opponent_count();
        assert_eq!(player_count, 20);
        assert_eq!(opponent_count, 40);
        let n_empties = 64 - player_count - opponent_count;
        assert_eq!(n_empties, 4);
        let score = board.solve(n_empties);
        // Opponent ahead: player gets no empty squares
        assert!(score < 0);
    }

    #[test]
    fn test_solve_tied() {
        // For diff == 0, we need: score + n_empties == 0
        // score = player_count * 2 - 64
        // So: player_count * 2 - 64 + n_empties == 0
        // Example: player_count = 30, n_empties = 4
        // 30 * 2 - 64 + 4 = 0

        // Create board with 30 player discs, 30 opponent discs, 4 empties
        // Player: bits 0-29 (30 bits)
        let player = 0x000000003FFFFFFFu64;
        // Opponent: bits 30-59 (30 bits)
        let opponent = 0x0FFFFFFFC0000000u64;

        let board = Board::from_bitboards(player, opponent);
        assert_eq!(board.get_player_count(), 30);
        assert_eq!(board.get_opponent_count(), 30);

        let n_empties = 64 - 30 - 30;
        assert_eq!(n_empties, 4);

        // When diff == 0, empties are split, score is 0
        let score = board.solve(n_empties);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_solve_no_empties() {
        // Full board, no empties
        // Player wins 64-0
        let board = Board::from_bitboards(u64::MAX, 0);
        assert_eq!(board.solve(0), 64);

        // Opponent wins 0-64
        let board = Board::from_bitboards(0, u64::MAX);
        assert_eq!(board.solve(0), -64);

        // Draw 32-32
        let board = Board::from_bitboards(0x00000000FFFFFFFF, 0xFFFFFFFF00000000);
        assert_eq!(board.solve(0), 0);
    }

    #[test]
    fn test_from_string_too_short() {
        let result = Board::from_string("XXXXXXXX", Disc::Black);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoardError::TooShort { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 8);
            }
            _ => panic!("Expected TooShort error"),
        }
    }

    #[test]
    fn test_from_string_too_long() {
        let board_string = "-".repeat(65);
        let result = Board::from_string(&board_string, Disc::Black);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoardError::TooLong { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 65);
            }
            _ => panic!("Expected TooLong error"),
        }
    }

    #[test]
    fn test_from_string_invalid_char() {
        // Valid start but invalid character 'Z' at position 10
        let board_string = "----------Z".to_string() + &"-".repeat(53);
        let result = Board::from_string(&board_string, Disc::Black);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoardError::InvalidChar { char, position } => {
                assert_eq!(char, 'Z');
                assert_eq!(position, 10);
            }
            _ => panic!("Expected InvalidChar error"),
        }
    }

    #[test]
    fn test_from_string_empty() {
        let result = Board::from_string("", Disc::Black);
        assert!(result.is_err());
        match result.unwrap_err() {
            BoardError::TooShort { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 0);
            }
            _ => panic!("Expected TooShort error"),
        }
    }

    #[test]
    fn test_board_error_display() {
        assert_eq!(
            BoardError::TooShort {
                expected: 64,
                actual: 10
            }
            .to_string(),
            "Board string too short: expected 64 characters, got 10"
        );
        assert_eq!(
            BoardError::TooLong {
                expected: 64,
                actual: 100
            }
            .to_string(),
            "Board string too long: expected 64 characters, got 100"
        );
        assert_eq!(
            BoardError::InvalidChar {
                char: 'Z',
                position: 5
            }
            .to_string(),
            "Invalid character 'Z' at position 5: must be 'X', 'O', or '-'"
        );
    }

    #[test]
    fn test_unique_identity() {
        // A board that is already canonical should return itself
        let board = Board::from_bitboards(1u64, 2u64);
        let unique = board.unique();
        // The unique board should be one of the 8 symmetric variants
        assert!(
            unique == board
                || unique == board.rotate_90_clockwise()
                || unique == board.rotate_180_clockwise()
                || unique == board.rotate_270_clockwise()
                || unique == board.flip_horizontal()
                || unique == board.flip_vertical()
                || unique == board.flip_diag_a1h8()
                || unique == board.flip_diag_a8h1()
        );
    }

    #[test]
    fn test_unique_symmetric_boards_same_result() {
        // All symmetric variants should produce the same unique board
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());

        let unique_original = board.unique();
        let unique_rot90 = board.rotate_90_clockwise().unique();
        let unique_rot180 = board.rotate_180_clockwise().unique();
        let unique_rot270 = board.rotate_270_clockwise().unique();
        let unique_flip_h = board.flip_horizontal().unique();
        let unique_flip_v = board.flip_vertical().unique();
        let unique_diag1 = board.flip_diag_a1h8().unique();
        let unique_diag2 = board.flip_diag_a8h1().unique();

        assert_eq!(unique_original, unique_rot90);
        assert_eq!(unique_original, unique_rot180);
        assert_eq!(unique_original, unique_rot270);
        assert_eq!(unique_original, unique_flip_h);
        assert_eq!(unique_original, unique_flip_v);
        assert_eq!(unique_original, unique_diag1);
        assert_eq!(unique_original, unique_diag2);
    }

    #[test]
    fn test_unique_selects_smallest() {
        // Create a board where we can verify the smallest is selected
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::B2.bitboard());

        let unique = board.unique();

        // Verify that unique has the smallest (player, opponent) tuple
        let candidates = [
            board,
            board.rotate_90_clockwise(),
            board.rotate_180_clockwise(),
            board.rotate_270_clockwise(),
            board.flip_horizontal(),
            board.flip_vertical(),
            board.flip_diag_a1h8(),
            board.flip_diag_a8h1(),
        ];

        for candidate in candidates {
            assert!(
                (unique.player.bits(), unique.opponent.bits())
                    <= (candidate.player.bits(), candidate.opponent.bits())
            );
        }
    }

    #[test]
    fn test_unique_initial_position() {
        // Initial position is symmetric, so unique should return itself or equivalent
        let board = Board::new();
        let unique = board.unique();

        // All 8 variants should give the same unique result
        assert_eq!(unique, board.rotate_90_clockwise().unique());
        assert_eq!(unique, board.flip_horizontal().unique());
    }

    #[test]
    fn test_unique_idempotent() {
        // Applying unique twice should give the same result
        // Use non-overlapping bitboards
        let board = Board::from_bitboards(0x00000000FFFFFFFF, 0xFFFFFFFF00000000);
        let unique1 = board.unique();
        let unique2 = unique1.unique();
        assert_eq!(unique1, unique2);
    }
}
