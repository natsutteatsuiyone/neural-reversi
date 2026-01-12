//! Disc flip calculation for move execution.

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
/// Returns 0 if no discs would be flipped (invalid move).
#[inline(always)]
pub fn flip(sq: Square, p: u64, o: u64) -> u64 {
    cfg_if! {
        if #[cfg(all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl"))] {
            unsafe { flip_avx512::flip(sq, p, o) }
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
            unsafe { flip_avx2::flip(sq, p, o) }
        } else if #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))] {
            flip_bmi2::flip(sq, p, o)
        } else {
            flip_kindergarten::flip(sq, p, o)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;

    use super::*;

    #[test]
    fn test_flip() {
        let p = Square::D5.bitboard() | Square::E4.bitboard();
        let o = Square::D4.bitboard() | Square::E5.bitboard();
        let flipped_c4_d4 = flip(Square::C4, p, o);
        let flipped_d3_d4 = flip(Square::D3, p, o);
        let flipped_e6_e5 = flip(Square::E6, p, o);
        let flipped_f5_e5 = flip(Square::F5, p, o);
        assert_eq!(flipped_c4_d4, Square::D4.bitboard());
        assert_eq!(flipped_d3_d4, Square::D4.bitboard());
        assert_eq!(flipped_e6_e5, Square::E5.bitboard());
        assert_eq!(flipped_f5_e5, Square::E5.bitboard());
    }

    #[test]
    fn test_flip_2() {
        let board = Board::from_string(
            "XXXXXXXOXOOXXXXOXOXXXOXOXOOXOXXOXOXOOOXOXOOOOOXOXOOOXXXO-X-OXOOO",
            crate::disc::Disc::Black,
        );
        let flipped = flip(Square::A8, board.player, board.opponent);
        let expected = Square::B7.bitboard()
            | Square::C6.bitboard()
            | Square::D5.bitboard()
            | Square::E4.bitboard()
            | Square::F3.bitboard();
        assert_eq!(flipped, expected);
    }
}
