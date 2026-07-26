//! NEON flip backend for single-square and shared-board batch paths.
//!
//! The kernel uses carry propagation for the LSB-to-MSB directions;
//! MSB-to-LSB directions are bit-reversed and run through the same
//! carry-propagation primitive.
//!
//! Per square, [`NEON_MASK`] holds the eight ray masks as four pairs
//! (right-side masks bit-reversed, the second pair of each side
//! complemented), [`BoardCtx::flip_pairs`] computes the spans and merges them
//! with the ray masks, and [`fold_addp`] reduces the two sides to one bitboard.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/flip_neon_rbit.c>

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
        // The second pair of each side (slots 2-3 and 6-7) is stored
        // COMPLEMENTED: the kernel re-derives `x & mask` as `BIC(x, !mask)`
        // at the same cost. The complemented mask also lets SHA3 builds merge
        // with one `BCAX` instead of an AND plus an OR; non-SHA3 builds use it
        // directly as the `BSL` selector.
        out[i].0[2] = !out[i].0[2];
        out[i].0[3] = !out[i].0[3];
        out[i].0[6] = !out[i].0[6];
        out[i].0[7] = !out[i].0[7];
        i += 1;
    }
    out
}

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
///
/// # Target features
///
/// Requires NEON. The parent module only selects this backend under the
/// matching `cfg` gate. (`Square::index()` is always a valid mask index.)
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
    oo: uint64x2_t,
    pp_rev: uint64x2_t,
    oo_rev: uint64x2_t,
    one: uint64x2_t,
}

impl BoardCtx {
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn new(player: u64, opponent: u64) -> Self {
        let pp = vdupq_n_u64(player);
        let oo = vdupq_n_u64(opponent);
        Self {
            pp,
            oo,
            pp_rev: vdupq_n_u64(player.reverse_bits()),
            oo_rev: vdupq_n_u64(opponent.reverse_bits()),
            one: vdupq_n_u64(1),
        }
    }

    /// Computes the merged flips for `pos` into a left pair (normal bit-space)
    /// and a right pair (bit-reversed space). Each returned lane is the OR of
    /// disjoint rays from the same side. All four mask pairs are loaded before
    /// the span kernels so the normal and bit-reversed sides can expose
    /// independent arithmetic chains to the scheduler.
    ///
    /// # Safety
    ///
    /// `pos` must be a valid [`NEON_MASK`] index (a square index, or one of
    /// the two trailing pass placeholders: `0..66`).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn flip_pairs(&self, pos: usize) -> (uint64x2_t, uint64x2_t) {
        let mask_ptr = unsafe { NEON_MASK.get_unchecked(pos).0.as_ptr() };
        let mask_l_a = unsafe { vld1q_u64(mask_ptr) };
        let cmask_l_b = unsafe { vld1q_u64(mask_ptr.add(2)) };
        let mask_rr_a = unsafe { vld1q_u64(mask_ptr.add(4)) };
        let cmask_rr_b = unsafe { vld1q_u64(mask_ptr.add(6)) };

        let w_l_a = flip_span_pair(mask_l_a, self.pp, self.oo, self.one);
        let w_rr_a = flip_span_pair(mask_rr_a, self.pp_rev, self.oo_rev, self.one);
        let w_l_b = flip_span_pair_inv(cmask_l_b, self.pp, self.oo, self.one);
        let w_rr_b = flip_span_pair_inv(cmask_rr_b, self.pp_rev, self.oo_rev, self.one);

        // SAFETY: the SHA3 variant of `merge_spans` only exists when the build
        // statically enables `sha3`.
        #[cfg(target_feature = "sha3")]
        let flip_l = unsafe { merge_spans(mask_l_a, w_l_a, cmask_l_b, w_l_b) };
        #[cfg(target_feature = "sha3")]
        let flip_rr = unsafe { merge_spans(mask_rr_a, w_rr_a, cmask_rr_b, w_rr_b) };
        #[cfg(not(target_feature = "sha3"))]
        let flip_l = merge_spans(mask_l_a, w_l_a, cmask_l_b, w_l_b);
        #[cfg(not(target_feature = "sha3"))]
        let flip_rr = merge_spans(mask_rr_a, w_rr_a, cmask_rr_b, w_rr_b);
        (flip_l, flip_rr)
    }

    /// One flip, for every arity. Folding both sides with a single pairwise add
    /// ([`fold_addp`]) costs one vector op less than folding each side on its
    /// own, and that holds even for the batched helpers below: the endgame
    /// leaves consume each flip immediately (`is_empty`, `apply_flip`), so the
    /// batched call sites are latency-bound too rather than saturating the
    /// NEON pipes.
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip1(&self, pos: usize) -> u64 {
        let (flip_l, flip_rr) = unsafe { self.flip_pairs(pos) };
        fold_addp(flip_l, flip_rr)
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

/// Folds both 2-lane results with a single pairwise add (ADDP). Lane 0 holds
/// the OR of `flip_l`'s two lanes, lane 1 the OR of `flip_rr`'s two lanes. The
/// lanes contain disjoint ray groups, so add matches OR.
///
/// Swapping the operands so the `RBIT` tail hangs off the low-lane `fmov`
/// instead of the high-lane `umov` measured 0.35% slower, so keep this order.
#[target_feature(enable = "neon")]
#[inline]
fn fold_addp(flip_l: uint64x2_t, flip_rr: uint64x2_t) -> u64 {
    let folded = vpaddq_u64(flip_l, flip_rr);
    let left = vgetq_lane_u64::<0>(folded);
    let right = vgetq_lane_u64::<1>(folded);
    left | right.reverse_bits()
}

/// Merges the two pairs of one side: `(mask_a & w_a) | (mask_b & w_b)`,
/// where the `b` mask arrives complemented (`cmask_b == !mask_b`).
///
/// The rays are pairwise disjoint, so the OR can be an XOR, and with SHA3
/// `BCAX` (`x ^ (y & !z)`) the `b`-side AND and the combine fuse into one
/// op: two vector ops per side instead of three.
#[cfg(target_feature = "sha3")]
#[inline]
#[target_feature(enable = "neon,sha3")]
fn merge_spans(
    mask_a: uint64x2_t,
    w_a: uint64x2_t,
    cmask_b: uint64x2_t,
    w_b: uint64x2_t,
) -> uint64x2_t {
    vbcaxq_u64(vandq_u64(mask_a, w_a), w_b, cmask_b)
}

/// Merges the two pairs of one side: `(mask_a & w_a) | (mask_b & w_b)`,
/// where the `b` mask arrives complemented (`cmask_b == !mask_b`).
#[cfg(not(target_feature = "sha3"))]
#[inline]
#[target_feature(enable = "neon")]
fn merge_spans(
    mask_a: uint64x2_t,
    w_a: uint64x2_t,
    cmask_b: uint64x2_t,
    w_b: uint64x2_t,
) -> uint64x2_t {
    vbslq_u64(cmask_b, vandq_u64(mask_a, w_a), w_b)
}

/// Computes the *unmasked* flip span for a pair of LSB-first rays: all bits
/// strictly below the outflank disc (the caller still ANDs with the ray
/// mask, fused into the pair merge in [`merge_spans`]).
///
/// The opponent board stays uncomplemented in [`BoardCtx`]. BIC forms
/// `mask & !oo` without the scalar `MVN` that lengthens context setup.
#[inline]
#[target_feature(enable = "neon")]
fn flip_span_pair(mask: uint64x2_t, pp: uint64x2_t, oo: uint64x2_t, one: uint64x2_t) -> uint64x2_t {
    let non_opponent = vbicq_u64(mask, oo);
    let player_on_ray = vandq_u64(mask, pp);
    let outflank = vandq_u64(neg_u64(non_opponent), player_on_ray);
    vqsubq_u64(outflank, one)
}

/// [`flip_span_pair`] for a pair whose ray mask arrives complemented from the
/// table. The carry seed becomes `oo | cmask`, while `pp & mask` is a BIC.
#[inline]
#[target_feature(enable = "neon")]
fn flip_span_pair_inv(
    cmask: uint64x2_t,
    pp: uint64x2_t,
    oo: uint64x2_t,
    one: uint64x2_t,
) -> uint64x2_t {
    let carry = vaddq_u64(vorrq_u64(oo, cmask), one);
    let player_on_ray = vbicq_u64(pp, cmask);
    let outflank = vandq_u64(carry, player_on_ray);
    vqsubq_u64(outflank, one)
}

#[inline]
#[target_feature(enable = "neon")]
fn neg_u64(x: uint64x2_t) -> uint64x2_t {
    vreinterpretq_u64_s64(vnegq_s64(vreinterpretq_s64_u64(x)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flip::flip_scalar;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// Cross-check NEON flip against the scalar oracle for every square
    /// across many randomly generated player/opponent pairs.
    #[test]
    fn neon_matches_scalar_on_random_positions() {
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
                let expected = flip_scalar::flip(sq, p, o);
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

    /// Edge cases the random sweep is unlikely to hit: an empty board,
    /// classic C4/D3-on-starting-position flips, and a corner-anchored long
    /// diagonal.
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
            let expected = flip_scalar::flip(sq, p, o);
            let got = unsafe { flip(sq, p, o) };
            assert_eq!(got, expected, "sq={:?} p={:#x} o={:#x}", sq, p, o);
        }
    }

    /// `fold_addp` may only substitute add for OR because the lanes it folds
    /// hold disjoint ray groups; it also has to bit-reverse the right side.
    #[test]
    fn fold_addp_ors_disjoint_lanes_and_reverses_the_right_side() {
        let left = unsafe {
            vsetq_lane_u64::<0>(0x0000_0000_0000_0015, vdupq_n_u64(0x0240_0000_0000_0000))
        };
        let right = unsafe { vsetq_lane_u64::<0>(0x0000_0000_0000_0900, vdupq_n_u64(0x0004_0000)) };
        let expected =
            0x0240_0000_0000_0015u64 | (0x0000_0000_0000_0900u64 | 0x0004_0000).reverse_bits();
        let got = unsafe { fold_addp(left, right) };
        assert_eq!(got, expected);
    }
}
