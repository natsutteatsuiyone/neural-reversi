//! NEON variant of flip function.
//!
//! NEON has no 256-bit lane, so each 4-direction group from the AVX2
//! reference is split into two 128-bit `uint64x2_t` pairs running the
//! same parallel-prefix scan.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/flip_avx_ppseq.c>

use super::lrmask::LRMASK;
use crate::square::Square;
use std::arch::aarch64::*;

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
///
/// # Safety
///
/// `sq.index()` must be in `0..64`. NEON intrinsics require `target_feature = "neon"`,
/// which is implied by aarch64-apple-darwin and any v8a baseline.
#[target_feature(enable = "neon")]
#[inline]
pub fn flip(sq: Square, player: u64, opponent: u64) -> u64 {
    unsafe {
        let pos = sq.index();
        let pp = vdupq_n_u64(player);
        let oo = vdupq_n_u64(opponent);
        let mask_ptr = LRMASK.get_unchecked(pos).0.as_ptr();

        // === LEFT side: lanes 0..3 are E, S, SE, SW (shifts toward higher bits).
        // Uses the BLSMSK identity: ((x - 1) ^ x) & mask isolates the lowest set bit
        // and below; combined with a non-opponent mask it picks out a contiguous
        // run of opponents bracketed by a player on the far side.
        let mask_l_a = vld1q_u64(mask_ptr);
        let mask_l_b = vld1q_u64(mask_ptr.add(2));

        // lo = ~oo & mask  (vbicq(a, b) = a & ~b)
        let lo_a = vbicq_u64(mask_l_a, oo);
        let lo_b = vbicq_u64(mask_l_b, oo);

        let neg_one = vdupq_n_u64(u64::MAX);
        // ((lo + (-1)) ^ lo) & mask
        let blsm_a = vandq_u64(veorq_u64(vaddq_u64(lo_a, neg_one), lo_a), mask_l_a);
        let blsm_b = vandq_u64(veorq_u64(vaddq_u64(lo_b, neg_one), lo_b), mask_l_b);
        // lf = ~pp & blsm
        let lf_a = vbicq_u64(blsm_a, pp);
        let lf_b = vbicq_u64(blsm_b, pp);
        // flip_left = lf if lf != blsm, else 0  →  lf & ~(lf == blsm)
        let eq_a = vceqq_u64(lf_a, blsm_a);
        let eq_b = vceqq_u64(lf_b, blsm_b);
        let flip_l_a = vbicq_u64(lf_a, eq_a);
        let flip_l_b = vbicq_u64(lf_b, eq_b);

        // === RIGHT side: lanes 0..3 are W (1), N (8), NW (9), NE (7) — right shifts.
        // Pair A holds shifts 1, 8 (orthogonal). Pair B holds shifts 9, 7 (diagonal).
        let mask_r_a = vld1q_u64(mask_ptr.add(4));
        let mask_r_b = vld1q_u64(mask_ptr.add(6));

        let rp_a = vandq_u64(pp, mask_r_a);
        let rp_b = vandq_u64(pp, mask_r_b);

        // Per-lane right shifts via vshlq with negative counts.
        let sh1_a = vld1q_s64([-1_i64, -8].as_ptr());
        let sh1_b = vld1q_s64([-9_i64, -7].as_ptr());
        let sh2_a = vld1q_s64([-2_i64, -16].as_ptr());
        let sh2_b = vld1q_s64([-18_i64, -14].as_ptr());
        let sh4_a = vld1q_s64([-4_i64, -32].as_ptr());
        let sh4_b = vld1q_s64([-36_i64, -28].as_ptr());

        let mut rs_a = vorrq_u64(rp_a, vshlq_u64(rp_a, sh1_a));
        let mut rs_b = vorrq_u64(rp_b, vshlq_u64(rp_b, sh1_b));
        rs_a = vorrq_u64(rs_a, vshlq_u64(rs_a, sh2_a));
        rs_b = vorrq_u64(rs_b, vshlq_u64(rs_b, sh2_b));
        rs_a = vorrq_u64(rs_a, vshlq_u64(rs_a, sh4_a));
        rs_b = vorrq_u64(rs_b, vshlq_u64(rs_b, sh4_b));

        let re_a = veorq_u64(vbicq_u64(mask_r_a, oo), rp_a);
        let re_b = veorq_u64(vbicq_u64(mask_r_b, oo), rp_b);

        // vcgtq_s64 is signed; rp/re are bitboards but the compare is correct
        // because the BLSMSK and prefix-OR steps preserve the leading-bit ordering.
        let cmp_a = vcgtq_s64(vreinterpretq_s64_u64(rp_a), vreinterpretq_s64_u64(re_a));
        let cmp_b = vcgtq_s64(vreinterpretq_s64_u64(rp_b), vreinterpretq_s64_u64(re_b));
        let flip_r_a = vandq_u64(vbicq_u64(mask_r_a, rs_a), cmp_a);
        let flip_r_b = vandq_u64(vbicq_u64(mask_r_b, rs_b), cmp_b);

        let flip_a = vorrq_u64(flip_l_a, flip_r_a);
        let flip_b = vorrq_u64(flip_l_b, flip_r_b);
        let flip = vorrq_u64(flip_a, flip_b);
        vgetq_lane_u64::<0>(flip) | vgetq_lane_u64::<1>(flip)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flip::flip_kindergarten;
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
                let expected = flip_kindergarten::flip(sq, p, o);
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
            let expected = flip_kindergarten::flip(sq, p, o);
            let got = unsafe { flip(sq, p, o) };
            assert_eq!(got, expected, "sq={:?} p={:#x} o={:#x}", sq, p, o);
        }
    }
}
