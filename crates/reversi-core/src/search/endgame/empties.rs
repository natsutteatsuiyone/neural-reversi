//! Empty squares of an endgame position.

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::square::Square;

/// Strategic ordering of squares for optimal move generation.
#[rustfmt::skip]
const PRESORTED: [Square; 64] = [
    Square::A1, Square::H1, Square::A8, Square::H8,

    Square::C1, Square::F1, Square::A3, Square::H3,
    Square::A6, Square::H6, Square::C8, Square::F8,

    Square::C3, Square::F3, Square::C6, Square::F6,
    Square::D1, Square::E1, Square::A4, Square::H4,

    Square::A5, Square::H5, Square::D8, Square::E8,
    Square::D3, Square::E3, Square::C4, Square::F4,
    Square::C5, Square::F5, Square::D6, Square::E6,
    Square::D2, Square::E2,

    Square::B4, Square::G4, Square::B5, Square::G5,
    Square::D7, Square::E7, Square::C2, Square::F2,
    Square::B3, Square::G3, Square::B6, Square::G6,
    Square::C7, Square::F7,

    Square::B1, Square::G1, Square::A2, Square::H2,
    Square::A7, Square::H7, Square::B8, Square::G8,
    Square::B2, Square::G2, Square::B7, Square::G7,
    Square::D4, Square::E4, Square::D5, Square::E5,
];

/// Position of each square in [`PRESORTED`].
const PRESORTED_INDEX: [u8; 64] = {
    let mut index = [0; 64];
    let mut i = 0;
    while i < 64 {
        index[PRESORTED[i] as usize] = i as u8;
        i += 1;
    }
    index
};

/// Empty squares of a position, passed by value down the endgame search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Empties {
    /// Bit `i` is set when `PRESORTED[i]` is empty.
    presorted: u64,
    count: u32,
}

impl Empties {
    /// Collects the empty squares of `board`.
    pub fn new(board: &Board) -> Self {
        let empties = board.empty_squares();
        let mut presorted = 0;
        for sq in empties.iter() {
            presorted |= 1 << PRESORTED_INDEX[sq.index()];
        }
        Self {
            presorted,
            count: empties.count(),
        }
    }

    /// Returns the number of empty squares.
    #[inline(always)]
    pub fn count(self) -> u32 {
        self.count
    }

    /// Returns the empty squares left after a move to `sq`.
    ///
    /// `sq` must be empty.
    #[inline(always)]
    pub fn without(self, sq: Square) -> Self {
        let bit = 1 << PRESORTED_INDEX[sq.index()];
        debug_assert!(self.presorted & bit != 0, "Square {sq:?} is not empty");
        Self {
            presorted: self.presorted ^ bit,
            count: self.count - 1,
        }
    }

    /// Returns the first `N` empty squares in presorted order.
    ///
    /// # Panics
    ///
    /// Panics if fewer than `N` squares are empty.
    #[inline(always)]
    pub fn first<const N: usize>(self) -> [Square; N] {
        let mut presorted = self.presorted;
        std::array::from_fn(|_| {
            let sq = PRESORTED[presorted.trailing_zeros() as usize];
            presorted &= presorted.wrapping_sub(1);
            sq
        })
    }
}

/// Squares of the quadrant A1-D4.
const QUADRANT_A1_D4: u64 = 0x0F0F_0F0F;

/// Returns the squares of the quadrants that hold an odd number of `empties`.
///
/// The board is divided into the quadrants A1-D4, E1-H4, A5-D8 and E5-H8.
#[inline(always)]
pub fn odd_quadrants(empties: Bitboard) -> Bitboard {
    const QUADRANT_ORIGINS: u64 = 0x0000_0011_0000_0011;
    let bits = empties.bits();
    let rank_pair_parity = bits ^ (bits >> 8);
    let file_parity = rank_pair_parity ^ (rank_pair_parity >> 16);
    let file_pair_parity = file_parity ^ (file_parity >> 2);
    let quadrant_parity = file_pair_parity ^ (file_pair_parity >> 1);
    Bitboard::new((quadrant_parity & QUADRANT_ORIGINS).wrapping_mul(QUADRANT_A1_D4))
}

/// Returns the squares of the quadrant that contains `sq`.
#[inline(always)]
pub fn quadrant(sq: Square) -> Bitboard {
    const FILES_E_TO_H: u32 = 1 << 2;
    const RANKS_5_TO_8: u32 = 1 << 5;
    Bitboard::new(QUADRANT_A1_D4 << (sq as u32 & (FILES_E_TO_H | RANKS_5_TO_8)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empties_of(squares: &[Square]) -> Bitboard {
        squares
            .iter()
            .fold(Bitboard::new(0), |empties, &sq| empties.set(sq))
    }

    #[test]
    fn presorted_lists_every_square_once() {
        let mut seen = [false; 64];
        for sq in PRESORTED {
            assert!(!seen[sq.index()], "Duplicate square in PRESORTED: {sq:?}");
            seen[sq.index()] = true;
        }
    }

    #[test]
    fn first_follows_the_presorted_order() {
        let board = Board::new();
        let empties = Empties::new(&board);
        assert_eq!(empties.count(), 60);
        assert_eq!(empties.first::<60>()[..], PRESORTED[..60]);
    }

    #[test]
    fn without_removes_one_square_and_keeps_the_order() {
        let empties = Empties::new(&Board::new());
        let rest = empties.without(Square::A1).without(Square::C3);
        assert_eq!(rest.count(), 58);
        let expected: Vec<Square> = PRESORTED[..60]
            .iter()
            .copied()
            .filter(|&sq| sq != Square::A1 && sq != Square::C3)
            .collect();
        assert_eq!(rest.first::<58>()[..], expected[..]);
    }

    #[test]
    fn without_agrees_with_the_empties_of_the_child_board() {
        let board = Board::new();
        for sq in board.legal_moves().iter() {
            assert_eq!(
                Empties::new(&board).without(sq),
                Empties::new(&board.make_move(sq))
            );
        }
    }

    #[test]
    #[should_panic]
    fn first_panics_when_too_few_squares_are_empty() {
        let board = Board::from_bitboards(u64::MAX ^ 1, 0_u64);
        Empties::new(&board).first::<2>();
    }

    #[test]
    fn odd_quadrants_marks_each_quadrant_by_its_parity() {
        assert_eq!(odd_quadrants(Bitboard::new(0)).bits(), 0);
        assert_eq!(odd_quadrants(Bitboard::new(u64::MAX)).bits(), 0);
        for sq in Square::iter() {
            assert_eq!(odd_quadrants(sq.bitboard()), quadrant(sq), "{sq:?}");
        }

        let empties = [Square::A1, Square::D4, Square::H1, Square::A8];
        assert_eq!(
            odd_quadrants(empties_of(&empties)),
            quadrant(Square::H1) | quadrant(Square::A8)
        );
        let mut odd = Bitboard::new(0);
        let mut bits = Bitboard::new(0);
        for sq in Square::iter() {
            bits = bits.set(sq);
            odd ^= quadrant(sq);
            assert_eq!(odd_quadrants(bits), odd, "{sq:?}");
        }
    }

    #[test]
    fn quadrant_splits_the_board_into_four_blocks() {
        assert_eq!(quadrant(Square::A1).bits(), 0x0000_0000_0F0F_0F0F);
        assert_eq!(quadrant(Square::D4).bits(), 0x0000_0000_0F0F_0F0F);
        assert_eq!(quadrant(Square::E1).bits(), 0x0000_0000_F0F0_F0F0);
        assert_eq!(quadrant(Square::A5).bits(), 0x0F0F_0F0F_0000_0000);
        assert_eq!(quadrant(Square::H8).bits(), 0xF0F0_F0F0_0000_0000);
        for sq in Square::iter() {
            assert!(quadrant(sq).contains(sq));
        }
    }
}
