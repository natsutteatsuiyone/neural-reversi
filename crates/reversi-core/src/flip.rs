//! Disc flip calculation for move execution.

use crate::bitboard::Bitboard;
use crate::square::Square;

// SIMD variants are gated by their own target features, but the dispatcher
// prefers wider backends first (AVX-512 over AVX2). `allow(dead_code)`
// keeps the build quiet without having to mirror that dispatch order here.
// Portable is always compiled: on non-SIMD targets it's the active dispatch;
// on SIMD targets it remains reachable from `#[cfg(test)]` cross-checks.
#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod flip_avx2;
#[allow(dead_code)]
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
            Bitboard::new(flip_avx512::flip(sq, p.bits(), o.bits()))
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

/// Computes flips for two squares sharing the same `(player, opponent)` board.
///
/// Equivalent to `(flip(sq1, p, o), flip(sq2, p, o))`; SIMD backends reuse
/// shared board broadcasts when that is profitable.
#[inline(always)]
pub fn flip2(sq1: Square, sq2: Square, p: Bitboard, o: Bitboard) -> (Bitboard, Bitboard) {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl") => {
            let ctx = flip_avx512::BoardCtx::new(p.bits(), o.bits());
            let (f0, f1) = ctx.flip2(sq1.index(), sq2.index());
            (Bitboard::new(f0), Bitboard::new(f1))
        }
        all(target_arch = "x86_64", target_feature = "avx2") => {
            let ctx = flip_avx2::BoardCtx::new(p.bits(), o.bits());
            let (f0, f1) = ctx.flip2(sq1.index(), sq2.index());
            (Bitboard::new(f0), Bitboard::new(f1))
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            let ctx = unsafe { flip_neon::BoardCtx::new(p.bits(), o.bits()) };
            let (f0, f1) = unsafe { ctx.flip2(sq1.index(), sq2.index()) };
            (Bitboard::new(f0), Bitboard::new(f1))
        }
        _ => {
            (flip(sq1, p, o), flip(sq2, p, o))
        }
    }
}

/// Computes flips for three squares sharing the same `(player, opponent)` board.
///
/// Equivalent to `(flip(sq1, p, o), flip(sq2, p, o), flip(sq3, p, o))`;
/// SIMD backends reuse shared board broadcasts when that is profitable.
#[inline(always)]
pub fn flip3(
    sq1: Square,
    sq2: Square,
    sq3: Square,
    p: Bitboard,
    o: Bitboard,
) -> (Bitboard, Bitboard, Bitboard) {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl") => {
            let ctx = flip_avx512::BoardCtx::new(p.bits(), o.bits());
            let (f0, f1, f2) = ctx.flip3(sq1.index(), sq2.index(), sq3.index());
            (Bitboard::new(f0), Bitboard::new(f1), Bitboard::new(f2))
        }
        all(target_arch = "x86_64", target_feature = "avx2") => {
            let ctx = flip_avx2::BoardCtx::new(p.bits(), o.bits());
            let (f0, f1, f2) = ctx.flip3(sq1.index(), sq2.index(), sq3.index());
            (Bitboard::new(f0), Bitboard::new(f1), Bitboard::new(f2))
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            let ctx = unsafe { flip_neon::BoardCtx::new(p.bits(), o.bits()) };
            let (f0, f1, f2) = unsafe { ctx.flip3(sq1.index(), sq2.index(), sq3.index()) };
            (Bitboard::new(f0), Bitboard::new(f1), Bitboard::new(f2))
        }
        _ => {
            (flip(sq1, p, o), flip(sq2, p, o), flip(sq3, p, o))
        }
    }
}

/// Computes flips for four squares sharing the same `(player, opponent)` board.
///
/// Equivalent to applying [`flip`] to each square; SIMD backends reuse shared
/// board broadcasts when that is profitable.
#[inline(always)]
pub fn flip4(
    sq1: Square,
    sq2: Square,
    sq3: Square,
    sq4: Square,
    p: Bitboard,
    o: Bitboard,
) -> (Bitboard, Bitboard, Bitboard, Bitboard) {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl") => {
            let ctx = flip_avx512::BoardCtx::new(p.bits(), o.bits());
            let (f0, f1, f2, f3) = ctx.flip4(sq1.index(), sq2.index(), sq3.index(), sq4.index());
            (Bitboard::new(f0), Bitboard::new(f1), Bitboard::new(f2), Bitboard::new(f3))
        }
        all(target_arch = "x86_64", target_feature = "avx2") => {
            let ctx = flip_avx2::BoardCtx::new(p.bits(), o.bits());
            let (f0, f1, f2, f3) = ctx.flip4(sq1.index(), sq2.index(), sq3.index(), sq4.index());
            (Bitboard::new(f0), Bitboard::new(f1), Bitboard::new(f2), Bitboard::new(f3))
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            let ctx = unsafe { flip_neon::BoardCtx::new(p.bits(), o.bits()) };
            let (f0, f1, f2, f3) =
                unsafe { ctx.flip4(sq1.index(), sq2.index(), sq3.index(), sq4.index()) };
            (Bitboard::new(f0), Bitboard::new(f1), Bitboard::new(f2), Bitboard::new(f3))
        }
        _ => {
            (flip(sq1, p, o), flip(sq2, p, o), flip(sq3, p, o), flip(sq4, p, o))
        }
    }
}

/// Crate-private AVX-512 shared-board context for move-list construction.
///
/// Only available on builds that compile the AVX-512 backend; callers must
/// mirror the same `cfg` gate.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512cd",
    target_feature = "avx512vl"
))]
pub(crate) use flip_avx512::BoardCtx as Avx512BoardCtx;

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
        let flipped = flip(Square::A8, board.player(), board.opponent());
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

    #[test]
    fn flip2_matches_portable() {
        let mut seed = 0x1234_5678_9abc_def0u64;

        for _ in 0..4096 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p = seed;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let o = seed & !p;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq1 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq2 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };

            let (a, b) = flip2(sq1, sq2, Bitboard::new(p), Bitboard::new(o));
            assert_eq!(
                a,
                Bitboard::new(flip_portable::flip(sq1, p, o)),
                "sq1={sq1:?} p={p:#018x} o={o:#018x}",
            );
            assert_eq!(
                b,
                Bitboard::new(flip_portable::flip(sq2, p, o)),
                "sq2={sq2:?} p={p:#018x} o={o:#018x}",
            );
        }
    }

    #[test]
    fn flip3_matches_portable() {
        let mut seed = 0x0f1e_2d3c_4b5a_6978u64;

        for _ in 0..4096 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p = seed;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let o = seed & !p;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq1 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq2 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq3 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };

            let (a, b, c) = flip3(sq1, sq2, sq3, Bitboard::new(p), Bitboard::new(o));
            assert_eq!(
                a,
                Bitboard::new(flip_portable::flip(sq1, p, o)),
                "sq1={sq1:?} p={p:#018x} o={o:#018x}",
            );
            assert_eq!(
                b,
                Bitboard::new(flip_portable::flip(sq2, p, o)),
                "sq2={sq2:?} p={p:#018x} o={o:#018x}",
            );
            assert_eq!(
                c,
                Bitboard::new(flip_portable::flip(sq3, p, o)),
                "sq3={sq3:?} p={p:#018x} o={o:#018x}",
            );
        }
    }

    #[test]
    fn flip4_matches_portable() {
        let mut seed = 0xc0ff_ee15_d15e_a5edu64;

        for _ in 0..4096 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p = seed;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let o = seed & !p;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq1 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq2 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq3 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sq4 = unsafe { Square::from_u32_unchecked((seed % 64) as u32) };

            let (a, b, c, d) = flip4(sq1, sq2, sq3, sq4, Bitboard::new(p), Bitboard::new(o));
            assert_eq!(
                a,
                Bitboard::new(flip_portable::flip(sq1, p, o)),
                "sq1={sq1:?} p={p:#018x} o={o:#018x}",
            );
            assert_eq!(
                b,
                Bitboard::new(flip_portable::flip(sq2, p, o)),
                "sq2={sq2:?} p={p:#018x} o={o:#018x}",
            );
            assert_eq!(
                c,
                Bitboard::new(flip_portable::flip(sq3, p, o)),
                "sq3={sq3:?} p={p:#018x} o={o:#018x}",
            );
            assert_eq!(
                d,
                Bitboard::new(flip_portable::flip(sq4, p, o)),
                "sq4={sq4:?} p={p:#018x} o={o:#018x}",
            );
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
