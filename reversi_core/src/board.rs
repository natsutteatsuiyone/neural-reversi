use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;

use crate::bit;
use crate::bitboard;
use crate::flip;
use crate::piece::Piece;
use crate::square::Square;

/// Represents a Reversi board with bitboards for the player and opponent.
///
/// The `Board` struct contains two 64-bit integers representing the positions of the player's
/// and opponent's pieces on the board. Each bit in the integer corresponds to a square on the
/// board.
///
/// # Fields
/// * `player` - Bitboard representing the player's pieces.
/// * `opponent` - Bitboard representing the opponent's pieces.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Board {
    pub player: u64,
    pub opponent: u64,
}

impl Default for Board {
    /// Creates a board with the standard Reversi starting position.
    ///
    /// The initial position has:
    /// - Black pieces on D5 and E4
    /// - White pieces on D4 and E5
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
    /// * `player` - Bitboard representing the player's pieces.
    /// * `opponent` - Bitboard representing the opponent's pieces.
    ///
    /// # Returns
    /// A new `Board` instance.
    pub fn from_bitboards(player: u64, opponent: u64) -> Board {
        Board { player, opponent }
    }

    /// Creates a `Board` from a string representation.
    ///
    /// The string should contain 64 characters representing the board squares from A1 to H8.
    /// Characters are interpreted as:
    /// - The current player's piece character (e.g., 'X' for Black)
    /// - The opponent's piece character (e.g., 'O' for White)
    /// - '-' for empty squares
    ///
    /// # Arguments
    /// * `board_string` - A string representing the board.
    /// * `current_player` - The current player.
    ///
    /// # Returns
    /// A new `Board` instance.
    pub fn from_string(board_string: &str, current_player: Piece) -> Board {
        let mut player: u64 = 0;
        let mut opponent: u64 = 0;
        for (sq, c) in board_string.chars().enumerate() {
            if c == current_player.to_char() {
                player = bitboard::set(player, Square::from_usize_unchecked(sq));
            } else if c != '-' {
                opponent = bitboard::set(opponent, Square::from_usize_unchecked(sq));
            }
        }
        Board::from_bitboards(player, opponent)
    }

    /// Gets the piece at a specific square from the perspective of the current player.
    ///
    /// # Arguments
    /// * `sq` - The square to check.
    /// * `side_to_move` - The current player's color.
    ///
    /// # Returns
    /// The piece at the specified square (current player's piece, opponent's piece, or empty).
    #[inline]
    pub fn get_piece_at(&self, sq: Square, side_to_move: Piece) -> Piece {
        if bitboard::is_set(self.player, sq) {
            side_to_move
        } else if bitboard::is_set(self.opponent, sq) {
            side_to_move.opposite()
        } else {
            Piece::Empty
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
    /// A `u64` value representing the empty squares.
    #[inline]
    pub fn get_empty(&self) -> u64 {
        bitboard::empty_board(self.player, self.opponent)
    }

    /// Returns the number of pieces the current player has on the board.
    ///
    /// # Returns
    ///
    /// The number of pieces the current player has on the board.
    #[inline]
    pub fn get_player_count(&self) -> u32 {
        self.player.count_ones()
    }

    /// Returns the number of pieces the opponent has on the board.
    ///
    /// # Returns
    ///
    /// The number of pieces the opponent has on the board.
    #[inline]
    pub fn get_opponent_count(&self) -> u32 {
        self.opponent.count_ones()
    }

    /// Returns the number of empty squares on the board.
    ///
    /// # Returns
    ///
    /// The number of empty squares on the board.
    #[inline]
    pub fn get_empty_count(&self) -> u32 {
        self.get_empty().count_ones()
    }

    /// Switches the players.
    ///
    /// # Returns
    /// A new `Board` instance with the players switched.
    #[inline]
    pub fn switch_players(&self) -> Board {
        Board {
            player: self.opponent,
            opponent: self.player,
        }
    }

    /// Attempts to make a move for the current player.
    ///
    /// # Arguments
    /// * `sq` - The square where the player is attempting to place a piece.
    ///
    /// # Returns
    /// `Some(Board)` with the updated board if the move is valid, `None` otherwise.
    #[inline]
    pub fn try_make_move(&self, sq: Square) -> Option<Board> {
        if !bitboard::has_adjacent_bit(self.opponent, sq) {
            return None;
        }

        let flipped = flip::flip(sq, self.player, self.opponent);
        if flipped == 0 {
            return None;
        }

        Some(Board {
            player: bitboard::opponent_flip(self.opponent, flipped),
            opponent: bitboard::player_flip(self.player, flipped, sq),
        })
    }

    /// Makes a move for the current player.
    ///
    /// # Arguments
    /// * `sq` - The square where the player is placing a piece.
    ///
    /// # Returns
    /// A new `Board` instance with the updated board state after the move.
    ///
    /// # Panics
    ///
    /// Panics if the move is invalid.
    #[inline]
    pub fn make_move(&self, sq: Square) -> Board {
        let flipped = flip::flip(sq, self.player, self.opponent);
        debug_assert!(flipped != 0);
        Board {
            player: bitboard::opponent_flip(self.opponent, flipped),
            opponent: bitboard::player_flip(self.player, flipped, sq),
        }
    }

    /// Makes a move for the current player, given the already calculated flipped pieces.
    ///
    /// # Arguments
    /// * `flipped` - The bitboard representing the pieces flipped by the move.
    /// * `sq` - The square where the player is placing a piece.
    ///
    /// # Returns
    /// A new `Board` instance with the updated board state after the move.
    #[inline]
    pub fn make_move_with_flipped(&self, flipped: u64, sq: Square) -> Board {
        Board {
            player: bitboard::opponent_flip(self.opponent, flipped),
            opponent: bitboard::player_flip(self.player, flipped, sq),
        }
    }

    /// Returns a bitboard representing the valid moves for the current player.
    ///
    /// # Returns
    /// A bitboard where each set bit represents a valid move for the current player.
    #[inline]
    pub fn get_moves(&self) -> u64 {
        bitboard::get_moves(self.player, self.opponent)
    }

    /// Checks if the current player has any legal moves.
    ///
    /// # Returns
    /// `true` if the current player has legal moves, `false` otherwise.
    #[inline]
    pub fn has_legal_moves(&self) -> bool {
        self.get_moves() != 0
    }

    /// Checks if a move to a specific square is legal for the current player.
    ///
    /// # Arguments
    /// * `sq` - The square to check.
    ///
    /// # Returns
    /// `true` if the move is legal, `false` otherwise.
    pub fn is_legal_move(&self, sq: Square) -> bool {
        self.get_moves() & sq.bitboard() != 0
    }

    /// Gets the number of stable discs for the current player.
    ///
    /// # Returns
    /// The number of stable discs for the current player.
    #[inline]
    pub fn get_stability(&self) -> i32 {
        crate::stability::get_stable_discs(self.player, self.opponent).count_ones() as i32
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
        bitboard::is_set(self.get_empty(), sq)
    }

    /// Calculates a hash of the current board position.
    ///
    /// # Returns
    /// A 64-bit hash value representing the current board position.
    #[inline]
    pub fn hash(&self) -> u64 {
        let mut hasher = rapidhash::fast::RapidHasher::default();
        hasher.write_u64(self.player);
        hasher.write_u64(self.opponent);
        hasher.finish()
    }

    /// Rotates the board 90 degrees clockwise.
    ///
    /// # Returns
    /// A new `Board` instance with the board rotated 90 degrees clockwise.
    #[inline]
    pub fn rotate_90_clockwise(&self) -> Board {
        Board::from_bitboards(
            bit::rotate_90_clockwise(self.player),
            bit::rotate_90_clockwise(self.opponent),
        )
    }

    /// Rotates the board 180 degrees.
    ///
    /// # Returns
    /// A new `Board` instance with the board rotated 180 degrees.
    #[inline]
    pub fn rotate_180_clockwise(&self) -> Board {
        Board::from_bitboards(
            bit::rotate_180_clockwise(self.player),
            bit::rotate_180_clockwise(self.opponent),
        )
    }

    /// Rotates the board 270 degrees clockwise.
    ///
    /// # Returns
    /// A new `Board` instance with the board rotated 270 degrees clockwise.
    #[inline]
    pub fn rotate_270_clockwise(&self) -> Board {
        Board::from_bitboards(
            bit::rotate_270_clockwise(self.player),
            bit::rotate_270_clockwise(self.opponent),
        )
    }

    /// Flips the board vertically (top to bottom).
    ///
    /// # Returns
    /// A new `Board` instance with the board flipped vertically.
    pub fn flip_vertical(&self) -> Board {
        Board::from_bitboards(
            bit::flip_vertical(self.player),
            bit::flip_vertical(self.opponent),
        )
    }

    /// Flips the board horizontally (left to right).
    ///
    /// # Returns
    /// A new `Board` instance with the board flipped horizontally.
    pub fn flip_horizontal(&self) -> Board {
        Board::from_bitboards(
            bit::flip_horizontal(self.player),
            bit::flip_horizontal(self.opponent),
        )
    }

    /// Flips the board along the main diagonal (A1-H8).
    ///
    /// # Returns
    /// A new `Board` instance with the board flipped along the main diagonal.
    pub fn flip_diag_a1h8(&self) -> Board {
        Board::from_bitboards(
            bit::flip_diag_a1h8(self.player),
            bit::flip_diag_a1h8(self.opponent),
        )
    }

    /// Flips the board along the anti-diagonal (A8-H1).
    ///
    /// # Returns
    /// A new `Board` instance with the board flipped along the anti-diagonal.
    pub fn flip_diag_a8h1(&self) -> Board {
        Board::from_bitboards(
            bit::flip_diag_a8h1(self.player),
            bit::flip_diag_a8h1(self.opponent),
        )
    }

    /// Converts the board to a string representation.
    ///
    /// The output format shows the board as an 8x8 grid with:
    /// - 'X' for Black pieces
    /// - 'O' for White pieces
    /// - '-' for empty squares
    ///
    /// # Arguments
    /// * `current_player` - The current player (determines which pieces are shown as 'X' or 'O').
    ///
    /// # Returns
    /// A string representation of the board with newlines between rows.
    pub fn to_string_as_board(&self, current_player: Piece) -> String {
        let mut s = String::with_capacity(64 + 8);
        for (i, sq) in Square::iter().enumerate() {
            if i > 0 && i % 8 == 0 {
                s.push('\n');
            }
            if bitboard::is_set(self.player, sq) {
                s.push(current_player.to_char());
            } else if bitboard::is_set(self.opponent, sq) {
                s.push(current_player.opposite().to_char());
            } else {
                s.push(Piece::Empty.to_char());
            }
        }
        s
    }
}

impl fmt::Display for Board {
    /// Formats the board for display, showing Black as the current player.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_as_board(Piece::Black))
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
        assert!(bitboard::is_set(board.player, Square::A1));
        assert!(bitboard::is_set(board.opponent, Square::H8));
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
        let board = Board::from_string(board_string, Piece::Black);
        assert!(bitboard::is_set(board.player, Square::D5));
        assert!(bitboard::is_set(board.player, Square::E4));
        assert!(bitboard::is_set(board.opponent, Square::D4));
        assert!(bitboard::is_set(board.opponent, Square::E5));
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
        let board = Board::from_string(board_string, Piece::White);
        // From White's perspective, O pieces are the player
        assert!(bitboard::is_set(board.player, Square::E4));
        assert!(bitboard::is_set(board.player, Square::D5));
        assert!(bitboard::is_set(board.opponent, Square::D4));
        assert!(bitboard::is_set(board.opponent, Square::E5));
    }

    #[test]
    fn test_get_piece_at() {
        let board = Board::new();

        // Check pieces from Black's perspective (Black is the first player)
        assert_eq!(board.get_piece_at(Square::D5, Piece::Black), Piece::Black);
        assert_eq!(board.get_piece_at(Square::E4, Piece::Black), Piece::Black);
        assert_eq!(board.get_piece_at(Square::D4, Piece::Black), Piece::White);
        assert_eq!(board.get_piece_at(Square::E5, Piece::Black), Piece::White);
        assert_eq!(board.get_piece_at(Square::A1, Piece::Black), Piece::Empty);

        // Switch to White's perspective
        let white_board = board.switch_players();
        assert_eq!(
            white_board.get_piece_at(Square::D5, Piece::White),
            Piece::Black
        );
        assert_eq!(
            white_board.get_piece_at(Square::E4, Piece::White),
            Piece::Black
        );
        assert_eq!(
            white_board.get_piece_at(Square::D4, Piece::White),
            Piece::White
        );
        assert_eq!(
            white_board.get_piece_at(Square::E5, Piece::White),
            Piece::White
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

        assert!(!bitboard::is_set(empty, Square::D4));
        assert!(!bitboard::is_set(empty, Square::E5));
        assert!(!bitboard::is_set(empty, Square::D5));
        assert!(!bitboard::is_set(empty, Square::E4));
        assert!(bitboard::is_set(empty, Square::A1));
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

        assert!(bitboard::is_set(switched_board.player, Square::D4));
        assert!(bitboard::is_set(switched_board.player, Square::E5));
        assert!(bitboard::is_set(switched_board.opponent, Square::D5));
        assert!(bitboard::is_set(switched_board.opponent, Square::E4));

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
        assert!(bitboard::is_set(new_board.opponent, Square::D3));
        assert!(bitboard::is_set(new_board.opponent, Square::D4));

        // Invalid move - no adjacent opponent pieces
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
        assert!(bitboard::is_set(new_board.opponent, Square::D3));
        assert!(bitboard::is_set(new_board.opponent, Square::D4));
        assert_eq!(new_board.get_opponent_count(), 4); // 2 original + 1 new + 1 flipped
        assert_eq!(new_board.get_player_count(), 1); // 2 original - 1 flipped
    }

    #[test]
    fn test_make_move_with_flipped() {
        let board = Board::new();
        let flipped = Square::D4.bitboard();
        let new_board = board.make_move_with_flipped(flipped, Square::D3);

        assert!(bitboard::is_set(new_board.opponent, Square::D3));
        assert!(bitboard::is_set(new_board.opponent, Square::D4));
    }

    #[test]
    fn test_get_moves() {
        let board = Board::new();
        let moves = board.get_moves();

        // Initial position has 4 legal moves
        assert_eq!(moves.count_ones(), 4);
        assert!(bitboard::is_set(moves, Square::D3));
        assert!(bitboard::is_set(moves, Square::C4));
        assert!(bitboard::is_set(moves, Square::F5));
        assert!(bitboard::is_set(moves, Square::E6));
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
        assert!(bitboard::is_set(rotated.player, Square::H1));
        assert!(bitboard::is_set(rotated.opponent, Square::A8));

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
        assert!(bitboard::is_set(rotated.player, Square::H8));
        assert!(bitboard::is_set(rotated.opponent, Square::A1));

        // Two rotations should return to original
        let rotated2 = board.rotate_180_clockwise().rotate_180_clockwise();
        assert_eq!(board, rotated2);
    }

    #[test]
    fn test_rotate_270_clockwise() {
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());
        let rotated = board.rotate_270_clockwise();

        // A1 -> A8, H8 -> H1
        assert!(bitboard::is_set(rotated.player, Square::A8));
        assert!(bitboard::is_set(rotated.opponent, Square::H1));

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
        assert!(bitboard::is_set(flipped.player, Square::A8));
        assert!(bitboard::is_set(flipped.opponent, Square::H1));

        // Double flip returns to original
        let double_flipped = flipped.flip_vertical();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_flip_horizontal() {
        let board = Board::from_bitboards(Square::A1.bitboard(), Square::H8.bitboard());
        let flipped = board.flip_horizontal();

        // A1 -> H1, H8 -> A8
        assert!(bitboard::is_set(flipped.player, Square::H1));
        assert!(bitboard::is_set(flipped.opponent, Square::A8));

        // Double flip returns to original
        let double_flipped = flipped.flip_horizontal();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_flip_diag_a1h8() {
        let board = Board::from_bitboards(Square::A2.bitboard(), Square::B1.bitboard());
        let flipped = board.flip_diag_a1h8();

        // A2 -> B1, B1 -> A2
        assert!(bitboard::is_set(flipped.player, Square::B1));
        assert!(bitboard::is_set(flipped.opponent, Square::A2));

        // Double flip returns to original
        let double_flipped = flipped.flip_diag_a1h8();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_flip_diag_a8h1() {
        let board = Board::from_bitboards(Square::A7.bitboard(), Square::B8.bitboard());
        let flipped = board.flip_diag_a8h1();

        // A7 -> B8, B8 -> A7
        assert!(bitboard::is_set(flipped.player, Square::B8));
        assert!(bitboard::is_set(flipped.opponent, Square::A7));

        // Double flip returns to original
        let double_flipped = flipped.flip_diag_a8h1();
        assert_eq!(board, double_flipped);
    }

    #[test]
    fn test_to_string_as_board() {
        let board = Board::new();

        // As Black
        let black_str = board.to_string_as_board(Piece::Black);
        assert!(black_str.contains("OX"));
        assert!(black_str.contains("XO"));

        // As White
        let white_str = board.to_string_as_board(Piece::White);
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
