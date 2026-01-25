//! Disc flip calculation for move execution.

use crate::bitboard::Bitboard;
use crate::square::Square;
use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl"))] {
        mod flip_avx512;
    } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
        mod flip_avx2;
    } else if #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))] {
        mod flip_bmi2;
    } else {
        mod flip_kindergarten;
    }
}

/// Calculates which opponent discs would be flipped by placing a disc at the given square.
///
/// # Arguments
///
/// * `sq` - The square where the disc is being placed
/// * `p` - Bitboard representing the current player's discs
/// * `o` - Bitboard representing the opponent's discs
///
/// # Returns
///
/// A bitboard representing all opponent discs that would be flipped by this move.
/// Returns empty bitboard if no discs would be flipped (invalid move).
#[inline(always)]
pub fn flip(sq: Square, p: Bitboard, o: Bitboard) -> Bitboard {
    cfg_if! {
        if #[cfg(all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl"))] {
            Bitboard::new(unsafe { flip_avx512::flip(sq, p.bits(), o.bits()) })
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            Bitboard::new(unsafe { flip_avx2::flip(sq, p.bits(), o.bits()) })
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))] {
            Bitboard::new(flip_bmi2::flip(sq, p.bits(), o.bits()))
        } else {
            Bitboard::new(flip_kindergarten::flip(sq, p.bits(), o.bits()))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;

    use super::*;

    #[test]
    fn test_flip() {
        let p = Bitboard::from_square(Square::D5) | Bitboard::from_square(Square::E4);
        let o = Bitboard::from_square(Square::D4) | Bitboard::from_square(Square::E5);
        let flipped_c4_d4 = flip(Square::C4, p, o);
        let flipped_d3_d4 = flip(Square::D3, p, o);
        let flipped_e6_e5 = flip(Square::E6, p, o);
        let flipped_f5_e5 = flip(Square::F5, p, o);
        assert_eq!(flipped_c4_d4, Bitboard::from_square(Square::D4));
        assert_eq!(flipped_d3_d4, Bitboard::from_square(Square::D4));
        assert_eq!(flipped_e6_e5, Bitboard::from_square(Square::E5));
        assert_eq!(flipped_f5_e5, Bitboard::from_square(Square::E5));
    }

    #[test]
    fn test_flip_2() {
        let board = Board::from_string(
            "XXXXXXXOXOOXXXXOXOXXXOXOXOOXOXXOXOXOOOXOXOOOOOXOXOOOXXXO-X-OXOOO",
            crate::disc::Disc::Black,
        )
        .unwrap();
        let flipped = flip(Square::A8, board.player, board.opponent);
        let expected = Bitboard::from_square(Square::B7)
            | Bitboard::from_square(Square::C6)
            | Bitboard::from_square(Square::D5)
            | Bitboard::from_square(Square::E4)
            | Bitboard::from_square(Square::F3);
        assert_eq!(flipped, expected);
    }
}
