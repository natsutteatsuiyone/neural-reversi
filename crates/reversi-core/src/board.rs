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

/// Represents a Reversi board with bitboards for the player and opponent.
///
/// The `Board` struct contains two 64-bit integers representing the positions of the player's
/// and opponent's discs on the board. Each bit in the integer corresponds to a square on the
/// board.
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
    /// Creates a new `Board` with the initial Reversi setup.
    ///
    /// # Returns
    ///
    /// A new `Board` instance.
    pub fn new() -> Board {
        Default::default()
    }

    /// Creates a `Board` from given bitboards.
    ///
    /// # Arguments
    ///
    /// * `player` - Bitboard representing the player's discs.
    /// * `opponent` - Bitboard representing the opponent's discs.
    ///
    /// # Returns
    ///
    /// A new `Board` instance.
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

    /// Creates a `Board` from a string representation.
    ///
    /// The string must contain exactly 64 characters representing the board squares from A1 to H8.
    /// Characters are interpreted as:
    /// - The current player's disc character (e.g., 'X' for Black)
    /// - The opponent's disc character (e.g., 'O' for White)
    /// - '-' for empty squares
    ///
    /// # Arguments
    ///
    /// * `board_string` - A string representing the board.
    /// * `current_player` - The current player.
    ///
    /// # Returns
    ///
    /// `Ok(Board)` if the string is valid, `Err(BoardError)` otherwise.
    ///
    /// # Errors
    /// - [`BoardError::InvalidPlayer`] if `current_player` is [`Disc::Empty`].
    /// - [`BoardError::TooShort`] if the string has fewer than 64 characters.
    /// - [`BoardError::TooLong`] if the string has more than 64 characters.
    /// - [`BoardError::InvalidChar`] if the string contains an invalid character.
    pub fn from_string(board_string: &str, current_player: Disc) -> Result<Board, BoardError> {
        if current_player == Disc::Empty {
            return Err(BoardError::InvalidPlayer);
        }

        let chars: Vec<char> = board_string.chars().collect();

        if chars.len() < 64 {
            return Err(BoardError::TooShort {
                expected: 64,
                actual: chars.len(),
            });
        }
        if chars.len() > 64 {
            return Err(BoardError::TooLong {
                expected: 64,
                actual: chars.len(),
            });
        }

        let player_char = current_player.to_char();
        let opponent_char = current_player.opposite().to_char();

        let mut player = Bitboard::new(0);
        let mut opponent = Bitboard::new(0);

        for (sq, &c) in chars.iter().enumerate() {
            // Note: sq is guaranteed to be < 64 because chars.len() == 64
            let square = Square::from_usize_unchecked(sq);
            if c == player_char {
                player = player.set(square);
            } else if c == opponent_char {
                opponent = opponent.set(square);
            } else if c != '-' {
                return Err(BoardError::InvalidChar {
                    char: c,
                    position: sq,
                });
            }
        }

        Ok(Board { player, opponent })
    }

    /// Gets the disc at a specific square from the perspective of the current player.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to check.
    /// * `side_to_move` - The current player's disc.
    ///
    /// # Returns
    ///
    /// The disc at the specified square (current player's disc, opponent's disc, or empty).
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

    /// Checks if the game is over (neither player can make a move).
    ///
    /// # Returns
    ///
    /// `true` if the game is over, `false` otherwise.
    #[inline]
    pub fn is_game_over(&self) -> bool {
        if self.has_legal_moves() {
            return false;
        }

        let switched = self.switch_players();
        !switched.has_legal_moves()
    }

    /// Gets the empty squares.
    ///
    /// # Returns
    ///
    /// A `Bitboard` value representing the empty squares.
    #[inline(always)]
    pub fn get_empty(&self) -> Bitboard {
        !(self.player | self.opponent)
    }

    /// Returns the number of discs the current player has on the board.
    ///
    /// # Returns
    ///
    /// The number of discs the current player has on the board.
    #[inline(always)]
    pub fn get_player_count(&self) -> u32 {
        self.player.count()
    }

    /// Returns the number of discs the opponent has on the board.
    ///
    /// # Returns
    ///
    /// The number of discs the opponent has on the board.
    #[inline(always)]
    pub fn get_opponent_count(&self) -> u32 {
        self.opponent.count()
    }

    /// Returns the number of empty squares on the board.
    ///
    /// # Returns
    ///
    /// The number of empty squares on the board.
    #[inline(always)]
    pub fn get_empty_count(&self) -> u32 {
        self.get_empty().count()
    }

    /// Returns the final score from the current player's perspective.
    ///
    /// # Returns
    ///
    /// Disc difference (positive if player has more discs).
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
        self.get_player_count() as Score * 2 - SCORE_MAX
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
    /// Unlike [`final_score`](Self::final_score) which assumes no empty squares,
    /// this method handles positions with remaining empties by awarding them
    /// to the player with more discs (standard Reversi endgame rules).
    ///
    /// # Scoring Rules
    ///
    /// Let `P` = player's disc count, `O` = opponent's disc count:
    /// - `P > O`: Player gets all empties → returns `(P - O) + n_empties`
    /// - `P < O`: Opponent gets all empties → returns `(P - O) - n_empties`
    /// - `P == O`: Empties split evenly → returns `0`
    ///
    /// # Arguments
    ///
    /// * `n_empties` - Number of empty squares on the board.
    ///
    /// # Returns
    ///
    /// Final score as disc difference (positive = player wins).
    #[inline(always)]
    pub fn solve(&self, n_empties: u32) -> Score {
        let score = self.get_player_count() as Score * 2 - SCORE_MAX;
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

    /// Switches the players.
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the players switched.
    #[inline(always)]
    pub fn switch_players(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
        }
    }

    /// Attempts to make a move for the current player.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square where the player is attempting to place a disc.
    ///
    /// # Returns
    ///
    /// `Some(Board)` with the updated board if the move is valid, `None` otherwise.
    #[inline]
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

    /// Makes a move for the current player.
    ///
    /// # Arguments
    /// * `sq` - The square where the player is placing a disc.
    ///
    /// # Returns
    /// A new `Board` instance with the updated board state after the move.
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

    /// Makes a move for the current player, given the already calculated flipped discs.
    ///
    /// # Arguments
    ///
    /// * `flipped` - The bitboard representing the discs flipped by the move.
    /// * `sq` - The square where the player is placing a disc.
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the updated board state after the move.
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

    /// Returns a bitboard representing the valid moves for the current player.
    ///
    /// # Returns
    ///
    /// A bitboard where each set bit represents a valid move for the current player.
    #[inline(always)]
    pub fn get_moves(&self) -> Bitboard {
        self.player.get_moves(self.opponent)
    }

    /// Checks if the current player has any legal moves.
    ///
    /// # Returns
    ///
    /// `true` if the current player has legal moves, `false` otherwise.
    #[inline(always)]
    pub fn has_legal_moves(&self) -> bool {
        !self.get_moves().is_empty()
    }

    /// Checks if a move to a specific square is legal for the current player.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to check.
    ///
    /// # Returns
    ///
    /// `true` if the move is legal, `false` otherwise.
    #[inline(always)]
    pub fn is_legal_move(&self, sq: Square) -> bool {
        self.get_moves().contains(sq)
    }

    /// Gets the number of stable discs for the current player.
    ///
    /// Stable discs are discs that cannot be flipped for the remainder of the game.
    /// This includes corner discs and discs protected by filled edges or other stable discs.
    ///
    /// # Returns
    ///
    /// The number of stable discs for the current player.
    #[inline]
    pub fn get_stability(&self) -> u32 {
        crate::stability::get_stable_discs(self.player, self.opponent).count()
    }

    /// Gets the potential moves for the current player.
    ///
    /// # Returns
    ///
    /// A `Bitboard` value representing the potential moves for the current player.
    #[inline(always)]
    pub fn get_potential_moves(&self) -> Bitboard {
        self.player.get_potential_moves(self.opponent)
    }

    /// Gets both the legal moves and potential moves for the current player.
    ///
    /// # Returns
    ///
    /// A tuple containing two `Bitboard` values:
    /// - The first value represents the legal moves.
    /// - The second value represents the potential moves.
    #[inline(always)]
    pub fn get_moves_and_potential(&self) -> (Bitboard, Bitboard) {
        self.player.get_moves_and_potential(self.opponent)
    }

    /// Checks if a given square is empty.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square to check.
    ///
    /// # Returns
    ///
    /// `true` if the square is empty, `false` otherwise.
    #[inline]
    pub fn is_square_empty(&self, sq: Square) -> bool {
        self.get_empty().contains(sq)
    }

    /// Calculates a hash of the current board position.
    ///
    /// # Returns
    ///
    /// A 64-bit hash value representing the current board position.
    #[inline]
    pub fn hash(&self) -> u64 {
        use rapidhash::v3;
        let words = [self.player.bits(), self.opponent.bits()];
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, 16) };
        v3::rapidhash_v3_nano_inline::<true, false>(bytes, &v3::DEFAULT_RAPID_SECRETS)
    }

    /// Rotates the board 90 degrees clockwise.
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the board rotated 90 degrees clockwise.
    #[inline]
    pub fn rotate_90_clockwise(&self) -> Board {
        Board {
            player: self.player.rotate_90_clockwise(),
            opponent: self.opponent.rotate_90_clockwise(),
        }
    }

    /// Rotates the board 180 degrees.
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the board rotated 180 degrees.
    #[inline]
    pub fn rotate_180_clockwise(&self) -> Board {
        Board {
            player: self.player.rotate_180_clockwise(),
            opponent: self.opponent.rotate_180_clockwise(),
        }
    }

    /// Rotates the board 270 degrees clockwise.
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the board rotated 270 degrees clockwise.
    #[inline]
    pub fn rotate_270_clockwise(&self) -> Board {
        Board {
            player: self.player.rotate_270_clockwise(),
            opponent: self.opponent.rotate_270_clockwise(),
        }
    }

    /// Flips the board vertically (top to bottom).
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the board flipped vertically.
    #[inline]
    pub fn flip_vertical(&self) -> Board {
        Board {
            player: self.player.flip_vertical(),
            opponent: self.opponent.flip_vertical(),
        }
    }

    /// Flips the board horizontally (left to right).
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the board flipped horizontally.
    #[inline]
    pub fn flip_horizontal(&self) -> Board {
        Board {
            player: self.player.flip_horizontal(),
            opponent: self.opponent.flip_horizontal(),
        }
    }

    /// Flips the board along the main diagonal (A1-H8).
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the board flipped along the main diagonal.
    #[inline]
    pub fn flip_diag_a1h8(&self) -> Board {
        Board {
            player: self.player.flip_diag_a1h8(),
            opponent: self.opponent.flip_diag_a1h8(),
        }
    }

    /// Flips the board along the anti-diagonal (A8-H1).
    ///
    /// # Returns
    ///
    /// A new `Board` instance with the board flipped along the anti-diagonal.
    #[inline]
    pub fn flip_diag_a8h1(&self) -> Board {
        Board {
            player: self.player.flip_diag_a8h1(),
            opponent: self.opponent.flip_diag_a8h1(),
        }
    }

    /// Returns the canonical (unique) form of this board.
    ///
    /// The canonical form is the board with the lexicographically smallest
    /// `(player.bits(), opponent.bits())` tuple among all 8 symmetric variants
    /// (original, 3 rotations, and 4 reflections).
    ///
    /// # Returns
    ///
    /// The canonical `Board` among all 8 symmetric variants.
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

    /// Converts the board to a string representation.
    ///
    /// The output format shows the board as an 8x8 grid with:
    /// - 'X' for Black discs
    /// - 'O' for White discs
    /// - '-' for empty squares
    ///
    /// # Arguments
    ///
    /// * `current_player` - The current player (determines which discs are shown as 'X' or 'O').
    ///
    /// # Returns
    ///
    /// A string representation of the board with newlines between rows.
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

/// Error type for board parsing operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoardError {
    /// Current player is `Disc::Empty`
    InvalidPlayer,
    /// String is too short (less than 64 characters)
    TooShort {
        /// Expected number of characters
        expected: usize,
        /// Actual number of characters
        actual: usize,
    },
    /// String is too long (more than 64 characters)
    TooLong {
        /// Expected number of characters
        expected: usize,
        /// Actual number of characters
        actual: usize,
    },
    /// Invalid character at position
    InvalidChar {
        /// The invalid character
        char: char,
        /// Position in the string (0-indexed)
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
