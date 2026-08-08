use super::{DIAGONAL_MASK, HORIZONTAL_MASK, VERTICAL_MASK};

/// Returns the legal moves for the player.
///
/// Dispatches to the best available implementation at compile time.
///
/// Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L822>
#[inline(always)]
pub(super) fn get_moves(player: u64, opponent: u64) -> u64 {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx512vl") => {
            unsafe { get_moves_avx512(player, opponent) }
        }
        all(target_arch = "x86_64", target_feature = "avx2") => {
            unsafe { get_moves_avx2(player, opponent) }
        }
        all(target_arch = "aarch64", target_feature = "neon", target_feature = "sha3") => {
            unsafe { get_moves_neon_sha3(player, opponent) }
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            unsafe { get_moves_neon(player, opponent) }
        }
        _ => {
            get_moves_portable(player, opponent)
        }
    }
}

/// Portable scalar implementation of `get_moves`.
///
/// Direction-specific scalar implementation used when no faster `get_moves`
/// SIMD path is selected.
#[inline(always)]
#[allow(dead_code)]
pub(super) fn get_moves_portable(player: u64, opponent: u64) -> u64 {
    let h_opp = opponent & HORIZONTAL_MASK;

    let mut flip7 = h_opp & (player << 7);
    let mut flip9 = h_opp & (player << 9);
    let mut flip8 = opponent & (player << 8);
    let mut flip1 = h_opp & (player << 1);

    flip7 |= h_opp & (flip7 << 7);
    flip9 |= h_opp & (flip9 << 9);
    flip8 |= opponent & (flip8 << 8);
    let mut moves = h_opp.wrapping_add(flip1);

    let mut pre7 = h_opp & (h_opp << 7);
    let mut pre9 = h_opp & (h_opp << 9);
    let mut pre8 = opponent & (opponent << 8);

    flip7 |= pre7 & (flip7 << 14);
    flip9 |= pre9 & (flip9 << 18);
    flip8 |= pre8 & (flip8 << 16);
    flip7 |= pre7 & (flip7 << 14);
    flip9 |= pre9 & (flip9 << 18);
    flip8 |= pre8 & (flip8 << 16);

    moves |= (flip7 << 7) | (flip9 << 9) | (flip8 << 8);

    flip7 = h_opp & (player >> 7);
    flip9 = h_opp & (player >> 9);
    flip8 = opponent & (player >> 8);
    flip1 = h_opp & (player >> 1);

    flip7 |= h_opp & (flip7 >> 7);
    flip9 |= h_opp & (flip9 >> 9);
    flip8 |= opponent & (flip8 >> 8);
    flip1 |= h_opp & (flip1 >> 1);

    pre7 >>= 7;
    pre9 >>= 9;
    pre8 >>= 8;
    let pre1 = h_opp & (h_opp >> 1);

    flip7 |= pre7 & (flip7 >> 14);
    flip9 |= pre9 & (flip9 >> 18);
    flip8 |= pre8 & (flip8 >> 16);
    flip1 |= pre1 & (flip1 >> 2);
    flip7 |= pre7 & (flip7 >> 14);
    flip9 |= pre9 & (flip9 >> 18);
    flip8 |= pre8 & (flip8 >> 16);
    flip1 |= pre1 & (flip1 >> 2);

    moves |= (flip7 >> 7) | (flip9 >> 9) | (flip8 >> 8) | (flip1 >> 1);
    moves & !(player | opponent)
}

/// Reduces a 256-bit vector to a single `u64` by OR-ing all four 64-bit lanes.
#[cfg(target_arch = "x86_64")]
macro_rules! horizontal_or_u64 {
    ($mm:expr) => {{
        let m128 = _mm_or_si128(
            _mm256_castsi256_si128($mm),
            _mm256_extracti128_si256($mm, 1),
        );
        _mm_cvtsi128_si64(_mm_or_si128(m128, _mm_shuffle_epi32(m128, 0x4e))) as u64
    }};
}

/// Reduces two 256-bit vectors in parallel.
///
/// Returns `(or(a lanes), or(b lanes))`. `get_moves_and_potential_*` already
/// has both vectors live; interleaving their lanes avoids doing two independent
/// horizontal reductions on the hot combined path.
#[cfg(target_arch = "x86_64")]
macro_rules! horizontal_or2_u64 {
    ($a:expr, $b:expr) => {{
        let lo = _mm256_unpacklo_epi64($a, $b);
        let hi = _mm256_unpackhi_epi64($a, $b);
        let pairs = _mm256_or_si256(lo, hi);
        let m128 = _mm_or_si128(
            _mm256_castsi256_si128(pairs),
            _mm256_extracti128_si256(pairs, 1),
        );
        (
            _mm_cvtsi128_si64(m128) as u64,
            _mm_cvtsi128_si64(_mm_shuffle_epi32(m128, 0x4e)) as u64,
        )
    }};
}

/// Reduces two 512-bit vectors in parallel.
///
/// Returns `(or(a lanes), or(b lanes))`. Interleaving the two vectors first
/// keeps both reductions in one instruction chain.
#[cfg(target_arch = "x86_64")]
macro_rules! horizontal_or2_zmm_u64 {
    ($a:expr, $b:expr) => {{
        let lo = _mm512_unpacklo_epi64($a, $b);
        let hi = _mm512_unpackhi_epi64($a, $b);
        let pairs = _mm512_or_si512(lo, hi);
        let m256 = _mm256_or_si256(
            _mm512_castsi512_si256(pairs),
            _mm512_extracti64x4_epi64::<1>(pairs),
        );
        let m128 = _mm_or_si128(
            _mm256_castsi256_si128(m256),
            _mm256_extracti128_si256(m256, 1),
        );
        (
            _mm_cvtsi128_si64(m128) as u64,
            _mm_cvtsi128_si64(_mm_shuffle_epi32(m128, 0x4e)) as u64,
        )
    }};
}

/// AVX-512-optimized implementation of `get_moves`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vl")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_avx512(player: u64, opponent: u64) -> u64 {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        DIAGONAL_MASK as i64,
        DIAGONAL_MASK as i64,
        VERTICAL_MASK as i64,
        HORIZONTAL_MASK as i64,
    );

    let empty = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);

    let masked_oo = _mm256_and_si256(oo, masks);

    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh));
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh));

    fl = _mm256_ternarylogic_epi64(fl, masked_oo, _mm256_sllv_epi64(fl, sh), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, masked_oo, _mm256_srlv_epi64(fr, sh), 0xF8);

    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));
    let pre_r = _mm256_srlv_epi64(pre_l, sh);

    let sh2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_ternarylogic_epi64(fl, pre_l, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, pre_r, _mm256_srlv_epi64(fr, sh2), 0xF8);

    fl = _mm256_ternarylogic_epi64(fl, pre_l, _mm256_sllv_epi64(fl, sh2), 0xF8);
    fr = _mm256_ternarylogic_epi64(fr, pre_r, _mm256_srlv_epi64(fr, sh2), 0xF8);

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));
    let moves = horizontal_or_u64!(mm);

    moves & empty
}

/// AVX2-optimized implementation of `get_moves`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_avx2(player: u64, opponent: u64) -> u64 {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        DIAGONAL_MASK as i64,
        DIAGONAL_MASK as i64,
        VERTICAL_MASK as i64,
        HORIZONTAL_MASK as i64,
    );

    let empty = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);
    let masked_oo = _mm256_and_si256(oo, masks);

    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh));
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh));

    fl = _mm256_or_si256(fl, _mm256_and_si256(masked_oo, _mm256_sllv_epi64(fl, sh)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(masked_oo, _mm256_srlv_epi64(fr, sh)));

    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));
    let pre_r = _mm256_srlv_epi64(pre_l, sh);

    let shift2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));
    let moves = horizontal_or_u64!(mm);

    moves & empty
}

/// Finishes the AArch64 NEON `get_moves` paths after diagonal move generation.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn finish_get_moves_neon(player: u64, opponent: u64, h_opp: u64, diag: u64) -> u64 {
    let empty = !(player | opponent);

    // --- Horizontal (shift 1): additive carry trick, both directions. ---
    let rp = player.reverse_bits();
    let rh = h_opp.reverse_bits();
    let mut moves = h_opp.wrapping_add(h_opp & (player << 1));
    moves |= rh.wrapping_add(rh & (rp << 1)).reverse_bits();

    // --- Vertical (shift 8): scalar parallel-prefix fill, both directions. ---
    let mut flip8 = opponent & (player << 8);
    flip8 |= opponent & (flip8 << 8);
    let mut pre8 = opponent & (opponent << 8);
    flip8 |= pre8 & (flip8 << 16);
    flip8 |= pre8 & (flip8 << 16);
    moves |= flip8 << 8;

    flip8 = opponent & (player >> 8);
    flip8 |= opponent & (flip8 >> 8);
    pre8 >>= 8;
    flip8 |= pre8 & (flip8 >> 16);
    flip8 |= pre8 & (flip8 >> 16);
    moves |= flip8 >> 8;

    (moves | diag) & empty
}

/// AArch64 NEON implementation of `get_moves` without SHA3 instructions.
///
/// Uses NEON for the two diagonal directions and scalar code for horizontal and
/// vertical scans.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_neon(player: u64, opponent: u64) -> u64 {
    use std::arch::aarch64::*;

    let h_opp = opponent & HORIZONTAL_MASK;

    // --- Diagonals (shifts 7 and 9) on NEON. ---
    // Lane 0 = shift 7, lane 1 = shift 9; both share the horizontal mask, which
    // blocks column-edge wraparound. `pre_neg == m & (m >> s)` because
    // `(m & (m << s)) >> s == m & (m >> s)`.
    let pv = vdupq_n_u64(player);
    let mv = vdupq_n_u64(h_opp);
    let sh = vcombine_s64(vdup_n_s64(7), vdup_n_s64(9));
    let sh_neg = vcombine_s64(vdup_n_s64(-7), vdup_n_s64(-9));
    let sh2 = vcombine_s64(vdup_n_s64(14), vdup_n_s64(18));
    let sh2_neg = vcombine_s64(vdup_n_s64(-14), vdup_n_s64(-18));

    let pre = vandq_u64(mv, vshlq_u64(mv, sh));
    let pre_neg = vshlq_u64(pre, sh_neg);

    let mut fl = vandq_u64(mv, vshlq_u64(pv, sh));
    let mut fr = vandq_u64(mv, vshlq_u64(pv, sh_neg));

    fl = vorrq_u64(fl, vandq_u64(mv, vshlq_u64(fl, sh)));
    fr = vorrq_u64(fr, vandq_u64(mv, vshlq_u64(fr, sh_neg)));

    // Two doublings reach the maximum diagonal run length of 6. These are not
    // disjoint, so they stay as OR.
    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));

    let md = vorrq_u64(vshlq_u64(fl, sh), vshlq_u64(fr, sh_neg));
    let diag = vgetq_lane_u64::<0>(md) | vgetq_lane_u64::<1>(md);

    finish_get_moves_neon(player, opponent, h_opp, diag)
}

/// AArch64 NEON implementation of `get_moves` using SHA3 `bcax`.
///
/// Uses NEON for the two diagonal directions and scalar code for horizontal and
/// vertical scans.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha3"
))]
#[target_feature(enable = "neon,sha3")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_neon_sha3(player: u64, opponent: u64) -> u64 {
    use std::arch::aarch64::*;

    let h_opp = opponent & HORIZONTAL_MASK;

    // --- Diagonals (shifts 7 and 9) on NEON. ---
    // Lane 0 = shift 7, lane 1 = shift 9; both share the horizontal mask, which
    // blocks column-edge wraparound. `pre_neg == m & (m >> s)` because
    // `(m & (m << s)) >> s == m & (m >> s)`.
    let pv = vdupq_n_u64(player);
    let mv = vdupq_n_u64(h_opp);
    let sh = vcombine_s64(vdup_n_s64(7), vdup_n_s64(9));
    let sh_neg = vcombine_s64(vdup_n_s64(-7), vdup_n_s64(-9));
    let sh2 = vcombine_s64(vdup_n_s64(14), vdup_n_s64(18));
    let sh2_neg = vcombine_s64(vdup_n_s64(-14), vdup_n_s64(-18));

    let pre = vandq_u64(mv, vshlq_u64(mv, sh));
    let pre_neg = vshlq_u64(pre, sh_neg);

    let mut fl = vandq_u64(mv, vshlq_u64(pv, sh));
    let mut fr = vandq_u64(mv, vshlq_u64(pv, sh_neg));

    // The squares added here are disjoint from `f`, so the OR is an XOR and can
    // use one `bcax` (`a ^ (b & ~c)`) with SHA3.
    let nmv = vdupq_n_u64(!h_opp);
    fl = vbcaxq_u64(fl, vshlq_u64(fl, sh), nmv);
    fr = vbcaxq_u64(fr, vshlq_u64(fr, sh_neg), nmv);

    // Two doublings reach the maximum diagonal run length of 6. These are not
    // disjoint, so they stay as OR.
    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));

    let md = vorrq_u64(vshlq_u64(fl, sh), vshlq_u64(fr, sh_neg));
    let diag = vgetq_lane_u64::<0>(md) | vgetq_lane_u64::<1>(md);

    finish_get_moves_neon(player, opponent, h_opp, diag)
}

/// Returns both legal and potential moves for the current player.
///
/// Dispatches to the best available implementation at compile time.
#[inline(always)]
pub(super) fn get_moves_and_potential(player: u64, opponent: u64) -> (u64, u64) {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx512vl") => {
            unsafe { get_moves_and_potential_avx512(player, opponent) }
        }
        all(target_arch = "x86_64", target_feature = "avx2") => {
            unsafe { get_moves_and_potential_avx2(player, opponent) }
        }
        all(target_arch = "aarch64", target_feature = "neon", target_feature = "sha3") => {
            unsafe { get_moves_and_potential_neon_sha3(player, opponent) }
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            unsafe { get_moves_and_potential_neon(player, opponent) }
        }
        _ => {
            get_moves_and_potential_portable(player, opponent)
        }
    }
}

/// Portable scalar implementation of `get_moves_and_potential`.
#[inline(always)]
#[allow(dead_code)]
pub(super) fn get_moves_and_potential_portable(player: u64, opponent: u64) -> (u64, u64) {
    let empty = !(player | opponent);
    let h_opp = opponent & HORIZONTAL_MASK;

    let mut flip7 = h_opp & (player << 7);
    let mut flip9 = h_opp & (player << 9);
    let mut flip8 = opponent & (player << 8);
    let mut flip1 = h_opp & (player << 1);

    flip7 |= h_opp & (flip7 << 7);
    flip9 |= h_opp & (flip9 << 9);
    flip8 |= opponent & (flip8 << 8);
    let mut moves = h_opp.wrapping_add(flip1);

    let mut pre7 = h_opp & (h_opp << 7);
    let mut pre9 = h_opp & (h_opp << 9);
    let mut pre8 = opponent & (opponent << 8);

    flip7 |= pre7 & (flip7 << 14);
    flip9 |= pre9 & (flip9 << 18);
    flip8 |= pre8 & (flip8 << 16);
    flip7 |= pre7 & (flip7 << 14);
    flip9 |= pre9 & (flip9 << 18);
    flip8 |= pre8 & (flip8 << 16);

    moves |= (flip7 << 7) | (flip9 << 9) | (flip8 << 8);

    flip7 = h_opp & (player >> 7);
    flip9 = h_opp & (player >> 9);
    flip8 = opponent & (player >> 8);
    flip1 = h_opp & (player >> 1);

    flip7 |= h_opp & (flip7 >> 7);
    flip9 |= h_opp & (flip9 >> 9);
    flip8 |= opponent & (flip8 >> 8);
    flip1 |= h_opp & (flip1 >> 1);

    pre7 >>= 7;
    pre9 >>= 9;
    pre8 >>= 8;
    let pre1 = h_opp & (h_opp >> 1);

    flip7 |= pre7 & (flip7 >> 14);
    flip9 |= pre9 & (flip9 >> 18);
    flip8 |= pre8 & (flip8 >> 16);
    flip1 |= pre1 & (flip1 >> 2);
    flip7 |= pre7 & (flip7 >> 14);
    flip9 |= pre9 & (flip9 >> 18);
    flip8 |= pre8 & (flip8 >> 16);
    flip1 |= pre1 & (flip1 >> 2);

    moves |= (flip7 >> 7) | (flip9 >> 9) | (flip8 >> 8) | (flip1 >> 1);

    let v_opp = opponent & VERTICAL_MASK;
    let d_opp = opponent & DIAGONAL_MASK;
    let potential = ((h_opp << 1)
        | (h_opp >> 1)
        | (v_opp << 8)
        | (v_opp >> 8)
        | (d_opp << 7)
        | (d_opp >> 7)
        | (d_opp << 9)
        | (d_opp >> 9))
        & empty;

    (moves & empty, potential)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_and_potential_avx512(player: u64, opponent: u64) -> (u64, u64) {
    use std::arch::x86_64::*;

    // Positive counts in the low half and their modulo-64 negatives in the
    // high half let one rotate propagate both directions. The edge masks make
    // the wrapped rotate bits equivalent to zeros from logical shifts.
    let sh = _mm512_set_epi64(57, 55, 56, 63, 7, 9, 8, 1);
    let masks = _mm512_set_epi64(
        DIAGONAL_MASK as i64,
        DIAGONAL_MASK as i64,
        VERTICAL_MASK as i64,
        HORIZONTAL_MASK as i64,
        DIAGONAL_MASK as i64,
        DIAGONAL_MASK as i64,
        VERTICAL_MASK as i64,
        HORIZONTAL_MASK as i64,
    );
    let empty = !(player | opponent);

    let pp = _mm512_set1_epi64(player as i64);
    let oo = _mm512_set1_epi64(opponent as i64);

    let masked_oo = _mm512_and_si512(oo, masks);
    let potential_mm = _mm512_rolv_epi64(masked_oo, sh);

    // Moves calculation
    let mut flips = _mm512_and_si512(masked_oo, _mm512_rolv_epi64(pp, sh));
    flips = _mm512_ternarylogic_epi64(flips, masked_oo, _mm512_rolv_epi64(flips, sh), 0xF8);

    let pre = _mm512_and_si512(masked_oo, potential_mm);
    let sh2 = _mm512_add_epi64(sh, sh);
    flips = _mm512_ternarylogic_epi64(flips, pre, _mm512_rolv_epi64(flips, sh2), 0xF8);
    flips = _mm512_ternarylogic_epi64(flips, pre, _mm512_rolv_epi64(flips, sh2), 0xF8);

    let moves_mm = _mm512_rolv_epi64(flips, sh);
    let (moves, potential) = horizontal_or2_zmm_u64!(moves_mm, potential_mm);

    (moves & empty, potential & empty)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_and_potential_avx2(player: u64, opponent: u64) -> (u64, u64) {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        DIAGONAL_MASK as i64,
        DIAGONAL_MASK as i64,
        VERTICAL_MASK as i64,
        HORIZONTAL_MASK as i64,
    );
    let empty = !(player | opponent);

    let pp = _mm256_set1_epi64x(player as i64);
    let oo = _mm256_set1_epi64x(opponent as i64);
    let masked_oo = _mm256_and_si256(oo, masks);
    let potential_mm = _mm256_or_si256(
        _mm256_sllv_epi64(masked_oo, sh),
        _mm256_srlv_epi64(masked_oo, sh),
    );

    // Moves calculation
    let mut fl = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(pp, sh));
    let mut fr = _mm256_and_si256(masked_oo, _mm256_srlv_epi64(pp, sh));

    fl = _mm256_or_si256(fl, _mm256_and_si256(masked_oo, _mm256_sllv_epi64(fl, sh)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(masked_oo, _mm256_srlv_epi64(fr, sh)));

    let pre_l = _mm256_and_si256(masked_oo, _mm256_sllv_epi64(masked_oo, sh));
    let pre_r = _mm256_srlv_epi64(pre_l, sh);

    let shift2 = _mm256_add_epi64(sh, sh);

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    fl = _mm256_or_si256(fl, _mm256_and_si256(pre_l, _mm256_sllv_epi64(fl, shift2)));
    fr = _mm256_or_si256(fr, _mm256_and_si256(pre_r, _mm256_srlv_epi64(fr, shift2)));

    let mm = _mm256_or_si256(_mm256_sllv_epi64(fl, sh), _mm256_srlv_epi64(fr, sh));
    let (moves, potential) = horizontal_or2_u64!(mm, potential_mm);

    (moves & empty, potential & empty)
}

/// Finishes the AArch64 NEON `get_moves_and_potential` paths.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
fn finish_get_moves_and_potential_neon(
    player: u64,
    opponent: u64,
    h_opp: u64,
    diag: u64,
    pot_diag: u64,
) -> (u64, u64) {
    let empty = !(player | opponent);

    // --- Horizontal (shift 1) moves: additive carry trick, both directions. ---
    let rp = player.reverse_bits();
    let rh = h_opp.reverse_bits();
    let mut moves = h_opp.wrapping_add(h_opp & (player << 1));
    moves |= rh.wrapping_add(rh & (rp << 1)).reverse_bits();

    // --- Vertical (shift 8) moves: scalar parallel-prefix fill, both directions. ---
    let mut flip8 = opponent & (player << 8);
    flip8 |= opponent & (flip8 << 8);
    let mut pre8 = opponent & (opponent << 8);
    flip8 |= pre8 & (flip8 << 16);
    flip8 |= pre8 & (flip8 << 16);
    moves |= flip8 << 8;

    flip8 = opponent & (player >> 8);
    flip8 |= opponent & (flip8 >> 8);
    pre8 >>= 8;
    flip8 |= pre8 & (flip8 >> 16);
    flip8 |= pre8 & (flip8 >> 16);
    moves |= flip8 >> 8;

    // --- Horizontal + vertical potential (one-step opponent neighbors). ---
    let v_opp = opponent & VERTICAL_MASK;
    let pot_hv = (h_opp << 1) | (h_opp >> 1) | (v_opp << 8) | (v_opp >> 8);

    ((moves | diag) & empty, (pot_hv | pot_diag) & empty)
}

/// AArch64 NEON implementation of `get_moves_and_potential` without SHA3.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_and_potential_neon(player: u64, opponent: u64) -> (u64, u64) {
    use std::arch::aarch64::*;

    let h_opp = opponent & HORIZONTAL_MASK;
    let d_opp = opponent & DIAGONAL_MASK;

    let pv = vdupq_n_u64(player);
    let mv = vdupq_n_u64(h_opp);
    let dd = vdupq_n_u64(d_opp);
    let sh = vcombine_s64(vdup_n_s64(7), vdup_n_s64(9));
    let sh_neg = vcombine_s64(vdup_n_s64(-7), vdup_n_s64(-9));
    let sh2 = vcombine_s64(vdup_n_s64(14), vdup_n_s64(18));
    let sh2_neg = vcombine_s64(vdup_n_s64(-14), vdup_n_s64(-18));

    let pre = vandq_u64(mv, vshlq_u64(mv, sh));
    let pre_neg = vshlq_u64(pre, sh_neg);

    let mut fl = vandq_u64(mv, vshlq_u64(pv, sh));
    let mut fr = vandq_u64(mv, vshlq_u64(pv, sh_neg));

    fl = vorrq_u64(fl, vandq_u64(mv, vshlq_u64(fl, sh)));
    fr = vorrq_u64(fr, vandq_u64(mv, vshlq_u64(fr, sh_neg)));

    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));

    let md = vorrq_u64(vshlq_u64(fl, sh), vshlq_u64(fr, sh_neg));
    // Diagonal potential: lane 0 = shift 7, lane 1 = shift 9 of the masked
    // opponent — a one-step dilation in both directions.
    let pd = vorrq_u64(vshlq_u64(dd, sh), vshlq_u64(dd, sh_neg));

    // Interleave so lane 0 carries both diagonal-move halves and lane 1 carries
    // both diagonal-potential halves, reducing both with one OR and two extracts.
    let comb = vorrq_u64(vzip1q_u64(md, pd), vzip2q_u64(md, pd));
    let diag = vgetq_lane_u64::<0>(comb);
    let pot_diag = vgetq_lane_u64::<1>(comb);

    finish_get_moves_and_potential_neon(player, opponent, h_opp, diag, pot_diag)
}

/// AArch64 NEON implementation of `get_moves_and_potential` using SHA3 `bcax`.
#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "sha3"
))]
#[target_feature(enable = "neon,sha3")]
#[inline]
#[allow(dead_code)]
pub(super) fn get_moves_and_potential_neon_sha3(player: u64, opponent: u64) -> (u64, u64) {
    use std::arch::aarch64::*;

    let h_opp = opponent & HORIZONTAL_MASK;
    let d_opp = opponent & DIAGONAL_MASK;

    let pv = vdupq_n_u64(player);
    let mv = vdupq_n_u64(h_opp);
    let dd = vdupq_n_u64(d_opp);
    let sh = vcombine_s64(vdup_n_s64(7), vdup_n_s64(9));
    let sh_neg = vcombine_s64(vdup_n_s64(-7), vdup_n_s64(-9));
    let sh2 = vcombine_s64(vdup_n_s64(14), vdup_n_s64(18));
    let sh2_neg = vcombine_s64(vdup_n_s64(-14), vdup_n_s64(-18));

    let pre = vandq_u64(mv, vshlq_u64(mv, sh));
    let pre_neg = vshlq_u64(pre, sh_neg);

    let mut fl = vandq_u64(mv, vshlq_u64(pv, sh));
    let mut fr = vandq_u64(mv, vshlq_u64(pv, sh_neg));

    // The squares added here are disjoint from `f`, so the OR is an XOR and can
    // use one `bcax` (`a ^ (b & ~c)`) with SHA3.
    let nmv = vdupq_n_u64(!h_opp);
    fl = vbcaxq_u64(fl, vshlq_u64(fl, sh), nmv);
    fr = vbcaxq_u64(fr, vshlq_u64(fr, sh_neg), nmv);

    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fl = vorrq_u64(fl, vandq_u64(pre, vshlq_u64(fl, sh2)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));
    fr = vorrq_u64(fr, vandq_u64(pre_neg, vshlq_u64(fr, sh2_neg)));

    let md = vorrq_u64(vshlq_u64(fl, sh), vshlq_u64(fr, sh_neg));
    // Diagonal potential: lane 0 = shift 7, lane 1 = shift 9 of the masked
    // opponent — a one-step dilation in both directions.
    let pd = vorrq_u64(vshlq_u64(dd, sh), vshlq_u64(dd, sh_neg));

    // Interleave so lane 0 carries both diagonal-move halves and lane 1 carries
    // both diagonal-potential halves, reducing both with one OR and two extracts.
    let comb = vorrq_u64(vzip1q_u64(md, pd), vzip2q_u64(md, pd));
    let diag = vgetq_lane_u64::<0>(comb);
    let pot_diag = vgetq_lane_u64::<1>(comb);

    finish_get_moves_and_potential_neon(player, opponent, h_opp, diag, pot_diag)
}
