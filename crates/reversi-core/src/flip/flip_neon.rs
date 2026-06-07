//! NEON variant of flip function.
//!
//! NEON has no 256-bit lane, so the eight direction masks are processed as
//! four `uint64x2_t` pairs. The kernel uses carry propagation for the
//! LSB-to-MSB directions; MSB-to-LSB directions are bit-reversed and run
//! through the same carry-propagation primitive.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/flip_avx_ppseq.c>

use super::lrmask::LRMASK;
use crate::square::Square;
use std::arch::aarch64::*;

#[repr(align(64))]
#[derive(Copy, Clone)]
struct NeonMaskEntry([u64; 8]);

static NEON_MASK: [NeonMaskEntry; 66] = build_neon_masks();

const fn build_neon_masks() -> [NeonMaskEntry; 66] {
    let mut out = [NeonMaskEntry([0; 8]); 66];
    let mut i = 0;
    while i < 66 {
        let mut j = 0;
        while j < 4 {
            out[i].0[j] = LRMASK[i].0[j];
            // Store right-side masks in bit-reversed form so they can reuse
            // the same lowest-outflank primitive as left-side masks.
            out[i].0[j + 4] = LRMASK[i].0[j + 4].reverse_bits();
            j += 1;
        }
        i += 1;
    }
    out
}

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
///
/// # Safety
///
/// `sq.index()` must be in `0..64`. NEON intrinsics require `target_feature = "neon"`,
/// which is implied by aarch64-apple-darwin and any v8a baseline.
#[target_feature(enable = "neon")]
#[inline]
pub fn flip(sq: Square, player: u64, opponent: u64) -> u64 {
    BoardCtx::new(player, opponent).flip1(sq.index())
}

/// SIMD board context for runtime squares that share the same `(player,
/// opponent)` board.
#[derive(Copy, Clone)]
pub(super) struct BoardCtx {
    pp: uint64x2_t,
    no: uint64x2_t,
    pp_rev: uint64x2_t,
    no_rev: uint64x2_t,
    zero: uint64x2_t,
    one: uint64x2_t,
}

impl BoardCtx {
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn new(player: u64, opponent: u64) -> Self {
        let not_opponent = !opponent;
        let pp = vdupq_n_u64(player);
        let no = vdupq_n_u64(not_opponent);
        Self {
            pp,
            no,
            pp_rev: vdupq_n_u64(player.reverse_bits()),
            no_rev: vdupq_n_u64(not_opponent.reverse_bits()),
            zero: vdupq_n_u64(0),
            one: vdupq_n_u64(1),
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip1(&self, pos: usize) -> u64 {
        unsafe {
            flip_index_prepared(
                pos,
                self.pp,
                self.no,
                self.pp_rev,
                self.no_rev,
                self.zero,
                self.one,
            )
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip2(&self, x0: usize, x1: usize) -> (u64, u64) {
        (self.flip1(x0), self.flip1(x1))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip3(&self, x0: usize, x1: usize, x2: usize) -> (u64, u64, u64) {
        (self.flip1(x0), self.flip1(x1), self.flip1(x2))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip4(&self, x0: usize, x1: usize, x2: usize, x3: usize) -> (u64, u64, u64, u64) {
        (
            self.flip1(x0),
            self.flip1(x1),
            self.flip1(x2),
            self.flip1(x3),
        )
    }
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn flip_index_prepared(
    pos: usize,
    pp: uint64x2_t,
    no: uint64x2_t,
    pp_rev: uint64x2_t,
    no_rev: uint64x2_t,
    zero: uint64x2_t,
    one: uint64x2_t,
) -> u64 {
    let mask_ptr = unsafe { NEON_MASK.get_unchecked(pos).0.as_ptr() };

    let mask_l_a = unsafe { vld1q_u64(mask_ptr) };
    let mask_l_b = unsafe { vld1q_u64(mask_ptr.add(2)) };
    let flip_l_a = unsafe { flip_left_pair(mask_l_a, pp, no, zero, one) };
    let flip_l_b = unsafe { flip_left_pair(mask_l_b, pp, no, zero, one) };

    let mask_rr_a = unsafe { vld1q_u64(mask_ptr.add(4)) };
    let mask_rr_b = unsafe { vld1q_u64(mask_ptr.add(6)) };
    let flip_rr_a = unsafe { flip_left_pair(mask_rr_a, pp_rev, no_rev, zero, one) };
    let flip_rr_b = unsafe { flip_left_pair(mask_rr_b, pp_rev, no_rev, zero, one) };

    let flip_l = vorrq_u64(flip_l_a, flip_l_b);
    let flip_rr = vorrq_u64(flip_rr_a, flip_rr_b);
    unsafe { fold_or_pair(flip_l) | fold_or_pair(flip_rr).reverse_bits() }
}

/// LEFT side masks: E, S, SE, SW. The closest square is the least significant
/// bit in each mask.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn flip_left_pair(
    mask: uint64x2_t,
    pp: uint64x2_t,
    no: uint64x2_t,
    zero: uint64x2_t,
    one: uint64x2_t,
) -> uint64x2_t {
    let non_opponent = vandq_u64(mask, no);
    let outflank = vandq_u64(vandq_u64(non_opponent, vsubq_u64(zero, non_opponent)), pp);
    vandq_u64(mask, vqsubq_u64(outflank, one))
}

/// The two lanes in each pair represent distinct rays from the origin square,
/// so their bitboards never overlap. A horizontal add therefore matches OR
/// and compiles to a cheaper reduction on AArch64.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn fold_or_pair(x: uint64x2_t) -> u64 {
    let summed = vadd_u64(vget_low_u64(x), vget_high_u64(x));
    vget_lane_u64::<0>(summed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flip::flip_portable;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// Cross-check NEON flip against the kindergarten oracle for every square
    /// across many randomly generated player/opponent pairs.
    #[test]
    fn neon_matches_kindergarten_on_random_positions() {
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let mut checked = 0usize;
        for _ in 0..2048 {
            let p: u64 = rng.random();
            let o_raw: u64 = rng.random();
            let o = o_raw & !p;
            for sq_idx in 0..64u8 {
                let bit = 1u64 << sq_idx;
                if (p | o) & bit != 0 {
                    continue;
                }
                let sq = Square::from_u8(sq_idx).unwrap();
                let expected = flip_portable::flip(sq, p, o);
                let got = unsafe { flip(sq, p, o) };
                assert_eq!(
                    got, expected,
                    "mismatch at sq={:?} p={:#x} o={:#x}: got={:#x} expected={:#x}",
                    sq, p, o, got, expected,
                );
                checked += 1;
            }
        }
        assert!(checked > 10_000, "too few checks: {checked}");
    }

    /// Edge cases the random sweep is unlikely to hit: an empty board, a
    /// classic D3-on-starting-position flip, and a corner-anchored long diagonal.
    #[test]
    fn neon_specific_cases() {
        let cases: &[(Square, u64, u64)] = &[
            (Square::D5, 0, 0),
            (
                Square::C4,
                (1u64 << 27) | (1u64 << 36),
                (1u64 << 28) | (1u64 << 35),
            ),
            (
                Square::D3,
                (1u64 << 27) | (1u64 << 36),
                (1u64 << 28) | (1u64 << 35),
            ),
            // corner A1 with player on H8 and opponents on the diagonal between.
            (
                Square::A1,
                1u64 << 63,
                (1u64 << 9)
                    | (1u64 << 18)
                    | (1u64 << 27)
                    | (1u64 << 36)
                    | (1u64 << 45)
                    | (1u64 << 54),
            ),
        ];
        for &(sq, p, o) in cases {
            if (p | o) & (1u64 << sq.index()) != 0 {
                continue;
            }
            let expected = flip_portable::flip(sq, p, o);
            let got = unsafe { flip(sq, p, o) };
            assert_eq!(got, expected, "sq={:?} p={:#x} o={:#x}", sq, p, o);
        }
    }

    #[test]
    fn fold_pair_matches_or_for_disjoint_lanes() {
        let pair = unsafe {
            vsetq_lane_u64::<0>(0x0000_0000_0000_0015, vdupq_n_u64(0x0240_0000_0000_0000))
        };
        let expected = 0x0240_0000_0000_0015u64;
        let got = unsafe { fold_or_pair(pair) };
        assert_eq!(got, expected);
    }
}
