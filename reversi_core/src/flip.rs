use crate::flip_avx;
use crate::flip_bmi2;
use crate::square::Square;

#[inline]
pub fn flip(sq: Square, p: u64, o: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { flip_avx::flip(sq, p, o) };
        }
    }

    flip_bmi2::flip(sq, p, o)
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
            crate::piece::Piece::Black,
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
