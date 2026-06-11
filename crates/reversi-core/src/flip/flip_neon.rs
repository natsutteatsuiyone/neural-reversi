//! NEON variant of flip function.
//!
//! The kernel uses carry propagation for the LSB-to-MSB directions;
//! MSB-to-LSB directions are bit-reversed and run through the same
//! carry-propagation primitive.
//!
//! Per square, [`NEON_MASK`] holds the eight ray masks as four pairs
//! (right-side masks bit-reversed, the second pair of each side
//! complemented), [`BoardCtx::flip_pairs`] computes the masked spans for all
//! eight rays, and one of two reductions folds them: [`fold_addp`] for
//! latency-bound single flips, [`fold_dual`] for throughput-bound batched
//! flips.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/flip_neon_rbit.c>
//! (same lowest-outflank + bit-reversal algorithm; the complemented-mask
//! merge and the two reductions are local changes).

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
        // at the same cost, and having `!mask_b` in a register lets the pair
        // merge in `merge_spans` use one `BCAX` instead of an AND plus an OR.
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
/// # Safety
///
/// Requires NEON, which is mandatory on aarch64 and guaranteed by this
/// module's `cfg` gate. (`Square::index()` is always a valid mask index.)
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

    /// Computes the masked flip spans for one side's two mask pairs, given
    /// that side's broadcast player / non-opponent boards.
    ///
    /// The per-pair kernel returns the *unmasked* span below the outflank;
    /// the ray masks are applied while merging the two pairs in
    /// [`merge_spans`] (the `b` pair's mask arrives complemented from the
    /// table).
    ///
    /// # Safety
    ///
    /// `mask_ptr` must point at four consecutive in-bounds `u64` mask words.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn flip_side(&self, mask_ptr: *const u64, pp: uint64x2_t, no: uint64x2_t) -> uint64x2_t {
        let mask_a = unsafe { vld1q_u64(mask_ptr) };
        let cmask_b = unsafe { vld1q_u64(mask_ptr.add(2)) };
        let w_a = flip_span_pair(mask_a, pp, no, self.zero, self.one);
        let w_b = flip_span_pair_inv(cmask_b, pp, no, self.zero, self.one);
        // SAFETY: the SHA3 variant of `merge_spans` only exists when the build
        // statically enables `sha3`.
        unsafe { merge_spans(mask_a, w_a, cmask_b, w_b) }
    }

    /// Computes the per-ray flips for `pos` into a left pair (normal
    /// bit-space) and a right pair (bit-reversed space). Returns
    /// `(flip_l, flip_rr)` for a reduction to fold.
    ///
    /// # Safety
    ///
    /// `pos` must be a valid [`NEON_MASK`] index (a square index, or one of
    /// the two trailing pass placeholders: `0..66`).
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn flip_pairs(&self, pos: usize) -> (uint64x2_t, uint64x2_t) {
        let mask_ptr = unsafe { NEON_MASK.get_unchecked(pos).0.as_ptr() };
        let flip_l = unsafe { self.flip_side(mask_ptr, self.pp, self.no) };
        let flip_rr = unsafe { self.flip_side(mask_ptr.add(4), self.pp_rev, self.no_rev) };
        (flip_l, flip_rr)
    }

    /// One flip in isolation, or separated from the next flip by other work
    /// (the move-list generation loop interleaves a bitscan, store and compare
    /// between flips). That regime is latency-bound, so it uses the ADDP
    /// reduction ([`fold_addp`]): folding both sides with one pairwise add
    /// shortens the critical path (~9% faster than the dual fold here).
    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip1(&self, pos: usize) -> u64 {
        let (flip_l, flip_rr) = unsafe { self.flip_pairs(pos) };
        fold_addp(flip_l, flip_rr)
    }

    /// Flip used by the batched `flipN` helpers, which compute several
    /// independent flips back-to-back (e.g. the endgame leaf solver). With no
    /// work between flips the NEON pipes saturate and the kernel is
    /// throughput-bound; there the dual-fold reduction ([`fold_dual`]) wins,
    /// because its two side-folds stay independent and both extracts come from
    /// the low lane (no high-lane `umov`). The single ADDP would instead
    /// serialize the folds and add a high-lane extract (~3% slower there).
    #[target_feature(enable = "neon")]
    #[inline]
    fn flip1_batched(&self, pos: usize) -> u64 {
        let (flip_l, flip_rr) = unsafe { self.flip_pairs(pos) };
        fold_dual(flip_l, flip_rr)
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip2(&self, x0: usize, x1: usize) -> (u64, u64) {
        (self.flip1_batched(x0), self.flip1_batched(x1))
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip3(&self, x0: usize, x1: usize, x2: usize) -> (u64, u64, u64) {
        (
            self.flip1_batched(x0),
            self.flip1_batched(x1),
            self.flip1_batched(x2),
        )
    }

    #[target_feature(enable = "neon")]
    #[inline]
    pub fn flip4(&self, x0: usize, x1: usize, x2: usize, x3: usize) -> (u64, u64, u64, u64) {
        (
            self.flip1_batched(x0),
            self.flip1_batched(x1),
            self.flip1_batched(x2),
            self.flip1_batched(x3),
        )
    }
}

/// Latency-optimized reduction: fold both 2-lane results with a single pairwise
/// add (ADDP). Lane 0 holds the OR of `flip_l`'s two rays, lane 1 the OR of
/// `flip_rr`'s two rays (the rays are disjoint, so add matches or).
#[target_feature(enable = "neon")]
#[inline]
fn fold_addp(flip_l: uint64x2_t, flip_rr: uint64x2_t) -> u64 {
    let folded = vpaddq_u64(flip_l, flip_rr);
    let left = vgetq_lane_u64::<0>(folded);
    let right = vgetq_lane_u64::<1>(folded);
    left | right.reverse_bits()
}

/// Throughput-optimized reduction: fold each side independently. The two folds
/// have no mutual dependency and both extract from the low lane (`fmov`, no
/// high-lane `umov`), which the NEON/GPR ports sustain better when several
/// flips are computed back-to-back.
#[target_feature(enable = "neon")]
#[inline]
fn fold_dual(flip_l: uint64x2_t, flip_rr: uint64x2_t) -> u64 {
    fold_or_pair(flip_l) | fold_or_pair(flip_rr).reverse_bits()
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
    vorrq_u64(vandq_u64(mask_a, w_a), vbicq_u64(w_b, cmask_b))
}

/// Computes the *unmasked* flip span for a pair of LSB-first rays: all bits
/// strictly below the outflank disc (the caller still ANDs with the ray
/// mask, fused into the pair merge in [`merge_spans`]).
///
/// The textbook outflank is `t & -t & pp` with `t = mask & no` (lowest
/// non-opponent square on the ray, kept if it is a player disc). Because the
/// player and opponent boards are disjoint, `pp` is a subset of `no`, so
/// `t & pp == mask & pp` and the outflank reassociates to
/// `-t & (mask & pp)`. Rewriting the canonical `t & -t` form away also keeps
/// instcombine from "canonicalizing" the kernel into a longer sequence.
#[inline]
#[target_feature(enable = "neon")]
fn flip_span_pair(
    mask: uint64x2_t,
    pp: uint64x2_t,
    no: uint64x2_t,
    zero: uint64x2_t,
    one: uint64x2_t,
) -> uint64x2_t {
    let non_opponent = vandq_u64(mask, no);
    let player_on_ray = vandq_u64(mask, pp);
    let outflank = vandq_u64(vsubq_u64(zero, non_opponent), player_on_ray);
    vqsubq_u64(outflank, one)
}

/// [`flip_span_pair`] for a pair whose ray mask arrives complemented from the
/// table: `x & mask` becomes `BIC(x, cmask)` at identical cost.
#[inline]
#[target_feature(enable = "neon")]
fn flip_span_pair_inv(
    cmask: uint64x2_t,
    pp: uint64x2_t,
    no: uint64x2_t,
    zero: uint64x2_t,
    one: uint64x2_t,
) -> uint64x2_t {
    let non_opponent = vbicq_u64(no, cmask);
    let player_on_ray = vbicq_u64(pp, cmask);
    let outflank = vandq_u64(vsubq_u64(zero, non_opponent), player_on_ray);
    vqsubq_u64(outflank, one)
}

/// The two lanes in each pair represent distinct rays from the origin square,
/// so their bitboards never overlap. A horizontal add therefore matches OR
/// and compiles to a cheaper reduction on AArch64.
#[inline]
#[target_feature(enable = "neon")]
fn fold_or_pair(x: uint64x2_t) -> u64 {
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
