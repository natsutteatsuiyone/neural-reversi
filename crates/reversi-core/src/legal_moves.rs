//! Legal move sets.

use std::ops::BitAnd;

use crate::bitboard::{Bitboard, BitboardIterator};
use crate::board::Board;
use crate::square::Square;

/// The legal moves of a position, or a subset of them.
///
/// Holds at most [`LegalMoves::MAX_COUNT`] squares, which unsafe code may
/// rely on. A non-empty value comes only from a [`Board`], and no method adds
/// a square. Which board the moves belong to is not tracked.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct LegalMoves(Bitboard);

impl LegalMoves {
    /// Largest number of legal moves of any position, including set-up
    /// positions unreachable from the start.
    pub const MAX_COUNT: usize = 35;

    /// The empty set.
    pub const EMPTY: LegalMoves = LegalMoves(Bitboard::new(0));

    /// Returns the legal moves for the current player of `board`.
    #[inline(always)]
    pub fn new(board: &Board) -> LegalMoves {
        let moves = board.player().legal_moves(board.opponent());
        debug_assert!(moves.count() as usize <= Self::MAX_COUNT);
        LegalMoves(moves)
    }

    /// Checks whether there is no move.
    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    /// Checks whether `sq` is one of the moves.
    #[inline(always)]
    pub fn contains(self, sq: Square) -> bool {
        self.0.contains(sq)
    }

    /// Returns the move if there is exactly one.
    #[inline(always)]
    pub fn only_move(self) -> Option<Square> {
        if self.0.is_empty() || !self.0.has_single_bit_nonzero() {
            return None;
        }
        self.0.lsb_square()
    }

    /// Returns the number of moves.
    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count()
    }

    /// Returns the number of moves with corner squares weighted double.
    #[inline(always)]
    pub fn corner_weighted_count(self) -> u32 {
        self.0.corner_weighted_count()
    }

    /// Returns an iterator over the moves from the lowest square up.
    #[inline(always)]
    pub fn iter(self) -> BitboardIterator {
        self.0.iter()
    }

    /// Returns the moves that are not in `mask`.
    #[inline(always)]
    pub fn and_not(self, mask: Bitboard) -> LegalMoves {
        LegalMoves(self.0 & !mask)
    }

    /// Returns the moves on corner squares.
    #[inline(always)]
    pub const fn corners(self) -> LegalMoves {
        LegalMoves(self.0.corners())
    }

    /// Returns the moves on non-corner squares.
    #[inline(always)]
    pub const fn non_corners(self) -> LegalMoves {
        LegalMoves(self.0.non_corners())
    }

    /// Returns the moves as a [`Bitboard`].
    #[inline(always)]
    pub const fn bitboard(self) -> Bitboard {
        self.0
    }

    /// Returns the moves as raw bits.
    #[inline(always)]
    pub const fn bits(self) -> u64 {
        self.0.bits()
    }
}

impl BitAnd<Bitboard> for LegalMoves {
    type Output = LegalMoves;

    #[inline(always)]
    fn bitand(self, rhs: Bitboard) -> LegalMoves {
        LegalMoves(self.0 & rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_move_returns_some_iff_exactly_one_move() {
        assert_eq!(LegalMoves::EMPTY.only_move(), None);

        let initial = LegalMoves::new(&Board::new());
        assert_eq!(initial.only_move(), None);

        let single_board = Board::from_bitboards(Square::A1.bitboard(), Square::B1.bitboard());
        let single = LegalMoves::new(&single_board);
        assert_eq!(single.only_move(), Some(Square::C1));
    }

    #[test]
    fn corner_weighted_count_matches_underlying_bitboard() {
        let initial = LegalMoves::new(&Board::new());
        assert_eq!(initial.corner_weighted_count(), 4);

        let corner_board = Board::from_bitboards(Square::C1.bitboard(), Square::B1.bitboard());
        let corner_moves = LegalMoves::new(&corner_board);
        assert_eq!(corner_moves.only_move(), Some(Square::A1));
        assert_eq!(corner_moves.corner_weighted_count(), 2);
    }
}
