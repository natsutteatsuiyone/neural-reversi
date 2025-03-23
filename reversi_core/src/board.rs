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
        let mut hasher = rapidhash::RapidInlineHasher::default();
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
            bit::rotate_90_clockwise(self.opponent))
    }

    pub fn flip_vertical(&self) -> Board {
        Board::from_bitboards(
            bit::flip_vertical(self.player),
            bit::flip_vertical(self.opponent))
    }

    pub fn flip_horizontal(&self) -> Board {
        Board::from_bitboards(
            bit::flip_horizontal(self.player),
            bit::flip_horizontal(self.opponent))
    }

    pub fn flip_diag_a1h8(&self) -> Board {
        Board::from_bitboards(
            bit::flip_diag_a1h8(self.player),
            bit::flip_diag_a1h8(self.opponent))
    }

    pub fn flip_diag_a8h1(&self) -> Board {
        Board::from_bitboards(
            bit::flip_diag_a8h1(self.player),
            bit::flip_diag_a8h1(self.opponent))
    }

    /// Converts the board to a string representation.
    ///
    /// # Arguments
    /// * `current_player` - The current player.
    ///
    /// # Returns
    /// A string representation of the board.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_as_board(Piece::Black))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_switch_players() {
        let board = Board::new();
        let switched_board = board.switch_players();

        assert!(bitboard::is_set(switched_board.player, Square::D4));
        assert!(bitboard::is_set(switched_board.player, Square::E5));
        assert!(bitboard::is_set(switched_board.opponent, Square::D5));
        assert!(bitboard::is_set(switched_board.opponent, Square::E4));
    }

    #[test]
    fn test_display() {
        let board = Board::new();
        let board_display = format!("{}", board);
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
}
