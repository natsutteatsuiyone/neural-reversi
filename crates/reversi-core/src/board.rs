//! Reversi board representation using bitboards.

use std::fmt;
use std::hash::Hash;

use crate::bitboard::Bitboard;
use crate::disc::Disc;
use crate::flip;
use crate::square::Square;

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
    /// A new `Board` instance.
    pub fn new() -> Board {
        Default::default()
    }

    /// Creates a `Board` from given bitboards.
    ///
    /// # Arguments
    /// * `player` - Bitboard representing the player's discs.
    /// * `opponent` - Bitboard representing the opponent's discs.
    ///
    /// # Returns
    /// A new `Board` instance.
    pub fn from_bitboards(player: impl Into<Bitboard>, opponent: impl Into<Bitboard>) -> Board {
        Board {
            player: player.into(),
            opponent: opponent.into(),
        }
    }

    /// Creates a `Board` from a string representation.
    ///
    /// The string should contain 64 characters representing the board squares from A1 to H8.
    /// Characters are interpreted as:
    /// - The current player's disc character (e.g., 'X' for Black)
    /// - The opponent's disc character (e.g., 'O' for White)
    /// - '-' for empty squares
    ///
    /// # Arguments
    /// * `board_string` - A string representing the board.
    /// * `current_player` - The current player.
    ///
    /// # Returns
    /// A new `Board` instance.
    pub fn from_string(board_string: &str, current_player: Disc) -> Board {
        let mut player = Bitboard::new(0);
        let mut opponent = Bitboard::new(0);
        for (sq, c) in board_string.chars().enumerate() {
            let square = Square::from_usize_unchecked(sq);
            if c == current_player.to_char() {
                player = player.set(square);
            } else if c != '-' {
                opponent = opponent.set(square);
            }
        }
        Board { player, opponent }
    }

    /// Gets the disc at a specific square from the perspective of the current player.
    ///
    /// # Arguments
    /// * `sq` - The square to check.
    /// * `side_to_move` - The current player's disc.
    ///
    /// # Returns
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

    /// Switches the players.
    ///
    /// # Returns
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
    /// * `sq` - The square where the player is attempting to place a disc.
    ///
    /// # Returns
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
    /// Panics if the move is invalid.
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
    /// * `flipped` - The bitboard representing the discs flipped by the move.
    /// * `sq` - The square where the player is placing a disc.
    ///
    /// # Returns
    /// A new `Board` instance with the updated board state after the move.
    #[inline(always)]
    pub fn make_move_with_flipped(&self, flipped: Bitboard, sq: Square) -> Board {
        Board {
            player: self.opponent.apply_flip(flipped),
            opponent: self.player.apply_move(flipped, sq),
        }
    }

    /// Returns a bitboard representing the valid moves for the current player.
    ///
    /// # Returns
    /// A bitboard where each set bit represents a valid move for the current player.
    #[inline(always)]
    pub fn get_moves(&self) -> Bitboard {
        self.player.get_moves(self.opponent)
    }

    /// Checks if the current player has any legal moves.
    ///
    /// # Returns
    /// `true` if the current player has legal moves, `false` otherwise.
    #[inline(always)]
    pub fn has_legal_moves(&self) -> bool {
        !self.get_moves().is_empty()
    }

    /// Checks if a move to a specific square is legal for the current player.
    ///
    /// # Arguments
    /// * `sq` - The square to check.
    ///
    /// # Returns
    /// `true` if the move is legal, `false` otherwise.
    #[inline(always)]
    pub fn is_legal_move(&self, sq: Square) -> bool {
        self.get_moves().contains(sq)
    }

    /// Gets the number of stable discs for the current player.
    ///
    /// # Returns
    /// The number of stable discs for the current player.
    #[inline]
    pub fn get_stability(&self) -> u32 {
        crate::stability::get_stable_discs(self.player, self.opponent).count()
    }

    /// Gets the potential moves for the current player.
    ///
    /// # Returns
    /// A `Bitboard` value representing the potential moves for the current player.
    #[inline(always)]
    pub fn get_potential_moves(&self) -> Bitboard {
        self.player.get_potential_moves(self.opponent)
    }

    /// Gets both the legal moves and potential moves for the current player.
    ///
    /// # Returns
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
    /// * `sq` - The square to check.
    ///
    /// # Returns
    /// `true` if the square is empty, `false` otherwise.
    #[inline]
    pub fn is_square_empty(&self, sq: Square) -> bool {
        self.get_empty().contains(sq)
    }

    /// Calculates a hash of the current board position.
    ///
    /// # Returns
    /// A 64-bit hash value representing the current board position.
    #[inline]
    pub fn hash(&self) -> u64 {
        use rapidhash::v3;
        let words = [self.player.0, self.opponent.0];
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, 16) };
        v3::rapidhash_v3_nano_inline::<true, false>(bytes, &v3::DEFAULT_RAPID_SECRETS)
    }

    /// Rotates the board 90 degrees clockwise.
    ///
    /// # Returns
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
    /// A new `Board` instance with the board flipped vertically.
    pub fn flip_vertical(&self) -> Board {
        Board {
            player: self.player.flip_vertical(),
            opponent: self.opponent.flip_vertical(),
        }
    }

    /// Flips the board horizontally (left to right).
    ///
    /// # Returns
    /// A new `Board` instance with the board flipped horizontally.
    pub fn flip_horizontal(&self) -> Board {
        Board {
            player: self.player.flip_horizontal(),
            opponent: self.opponent.flip_horizontal(),
        }
    }

    /// Flips the board along the main diagonal (A1-H8).
    ///
    /// # Returns
    /// A new `Board` instance with the board flipped along the main diagonal.
    pub fn flip_diag_a1h8(&self) -> Board {
        Board {
            player: self.player.flip_diag_a1h8(),
            opponent: self.opponent.flip_diag_a1h8(),
        }
    }

    /// Flips the board along the anti-diagonal (A8-H1).
    ///
    /// # Returns
    /// A new `Board` instance with the board flipped along the anti-diagonal.
    pub fn flip_diag_a8h1(&self) -> Board {
        Board {
            player: self.player.flip_diag_a8h1(),
            opponent: self.opponent.flip_diag_a8h1(),
        }
    }

    /// Converts the board to a string representation.
    ///
    /// The output format shows the board as an 8x8 grid with:
    /// - 'X' for Black discs
    /// - 'O' for White discs
    /// - '-' for empty squares
    ///
    /// # Arguments
    /// * `current_player` - The current player (determines which discs are shown as 'X' or 'O').
    ///
    /// # Returns
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
        let board = Board::from_string(board_string, Disc::Black);
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
        let board = Board::from_string(board_string, Disc::White);
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
}
