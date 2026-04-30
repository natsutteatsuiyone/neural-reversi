//! Disc flip calculation for move execution.

use crate::bitboard::Bitboard;
use crate::square::Square;

// SIMD variants are gated by their own target features but a higher-priority
// dispatch (e.g. AVX-512 over AVX2) may shadow them at runtime. `allow(dead_code)`
// keeps the build quiet without having to mirror the dispatcher cfgs here.
// Portable is always compiled: on non-SIMD targets it's the active dispatch;
// on SIMD targets it remains reachable from `#[cfg(test)]` cross-checks.
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod flip_avx2;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512cd",
    target_feature = "avx512vl"
))]
mod flip_avx512;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod flip_neon;
#[allow(dead_code)]
mod flip_portable;
mod lrmask;

/// Calculates which opponent discs would be flipped by placing a disc at `sq`.
///
/// Dispatches to a platform-specific implementation (AVX-512, AVX2, NEON, or
/// portable scalar bitboard).
#[inline(always)]
pub fn flip(sq: Square, p: Bitboard, o: Bitboard) -> Bitboard {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl") => {
            Bitboard::new(unsafe { flip_avx512::flip(sq, p.bits(), o.bits()) })
        }
        all(target_arch = "x86_64", target_feature = "avx2") => {
            Bitboard::new(unsafe { flip_avx2::flip(sq, p.bits(), o.bits()) })
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            Bitboard::new(unsafe { flip_neon::flip(sq, p.bits(), o.bits()) })
        }
        _ => {
            Bitboard::new(flip_portable::flip(sq, p.bits(), o.bits()))
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

    #[test]
    fn test_portable_flip_matches_reference() {
        let mut seed = 0x6eed_0e11_d15c_a11du64;

        for _ in 0..2048 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p = seed;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let o = seed & !p;

            for sq_idx in 0..64 {
                let sq = unsafe { Square::from_u32_unchecked(sq_idx) };
                if ((p | o) & (1u64 << sq_idx)) != 0 {
                    continue;
                }

                assert_eq!(
                    flip_portable::flip(sq, p, o),
                    reference_flip(sq, p, o),
                    "sq={sq:?} p={p:#018x} o={o:#018x}",
                );
            }
        }
    }

    fn reference_flip(sq: Square, p: u64, o: u64) -> u64 {
        const DIRECTIONS: [(i32, i32); 8] = [
            (1, 0),
            (0, 1),
            (1, 1),
            (-1, 1),
            (-1, 0),
            (0, -1),
            (-1, -1),
            (1, -1),
        ];

        let sq_idx = sq.index() as i32;
        let x = sq_idx & 7;
        let y = sq_idx >> 3;
        let mut flipped = 0;

        for (dx, dy) in DIRECTIONS {
            let mut cx = x + dx;
            let mut cy = y + dy;
            let mut line = 0;

            while (0..8).contains(&cx) && (0..8).contains(&cy) {
                let bit = 1u64 << (cy * 8 + cx);
                if (o & bit) != 0 {
                    line |= bit;
                } else {
                    if line != 0 && (p & bit) != 0 {
                        flipped |= line;
                    }
                    break;
                }

                cx += dx;
                cy += dy;
            }
        }

        flipped
    }
}
