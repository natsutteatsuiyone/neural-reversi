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
        _mm_cvtsi128_si64(_mm_or_si128(m128, _mm_srli_si128(m128, 8))) as u64
    }};
}

/// AVX-512-optimized implementation of `get_moves`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vl")]
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

/// Returns the potential moves for the player.
///
/// Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c#L944>
#[inline(always)]
pub(super) fn get_potential_moves(p: u64, o: u64) -> u64 {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx2") => {
            unsafe { get_potential_moves_avx2(p, o) }
        }
        _ => {
            get_potential_moves_portable(p, o)
        }
    }
}

#[inline(always)]
#[allow(dead_code)]
fn get_potential_moves_portable(p: u64, o: u64) -> u64 {
    let h_opp = o & HORIZONTAL_MASK;
    let v_opp = o & VERTICAL_MASK;
    let d_opp = o & DIAGONAL_MASK;

    let h = (h_opp << 1) | (h_opp >> 1);
    let v = (v_opp << 8) | (v_opp >> 8);
    let d1 = (d_opp << 7) | (d_opp >> 7);
    let d2 = (d_opp << 9) | (d_opp >> 9);

    (h | v | d1 | d2) & !(p | o)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
fn get_potential_moves_avx2(p: u64, o: u64) -> u64 {
    use std::arch::x86_64::*;

    let sh = _mm256_set_epi64x(7, 9, 8, 1);
    let masks = _mm256_set_epi64x(
        DIAGONAL_MASK as i64,
        DIAGONAL_MASK as i64,
        VERTICAL_MASK as i64,
        HORIZONTAL_MASK as i64,
    );
    let oo = _mm256_set1_epi64x(o as i64);
    let masked_oo = _mm256_and_si256(oo, masks);

    horizontal_or_u64!(_mm256_or_si256(
        _mm256_sllv_epi64(masked_oo, sh),
        _mm256_srlv_epi64(masked_oo, sh),
    )) & !(p | o)
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
#[target_feature(enable = "avx512vl")]
#[allow(dead_code)]
pub(super) fn get_moves_and_potential_avx512(player: u64, opponent: u64) -> (u64, u64) {
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
    let potential = horizontal_or_u64!(_mm256_or_si256(
        _mm256_sllv_epi64(masked_oo, sh),
        _mm256_srlv_epi64(masked_oo, sh),
    )) & empty;

    // Moves calculation
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

    (moves & empty, potential)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
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
    let potential = horizontal_or_u64!(_mm256_or_si256(
        _mm256_sllv_epi64(masked_oo, sh),
        _mm256_srlv_epi64(masked_oo, sh),
    )) & empty;

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
    let moves = horizontal_or_u64!(mm);

    (moves & empty, potential)
}

/// AArch64 implementation of `get_moves_and_potential`.
///
/// Legal moves use the portable scalar shape. Potential moves use NEON lanes so
/// they do not add to the scalar move-generation dependency chain.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn get_moves_and_potential_neon(player: u64, opponent: u64) -> (u64, u64) {
    use std::arch::aarch64::*;
    let empty = !(player | opponent);
    let h_opp = opponent & HORIZONTAL_MASK;

    // Lane 0 = horizontal (shift 1), lane 1 = vertical (shift 8).
    let mask_hv = vcombine_u64(vdup_n_u64(h_opp), vdup_n_u64(opponent & VERTICAL_MASK));
    let sh_hv = vcombine_s64(vdup_n_s64(1), vdup_n_s64(8));
    let sh_hv_neg = vcombine_s64(vdup_n_s64(-1), vdup_n_s64(-8));

    // Lane 0 = anti-diagonal (shift 7), lane 1 = main-diagonal (shift 9); shared mask.
    let mask_d = vdupq_n_u64(opponent & DIAGONAL_MASK);
    let sh_d = vcombine_s64(vdup_n_s64(7), vdup_n_s64(9));
    let sh_d_neg = vcombine_s64(vdup_n_s64(-7), vdup_n_s64(-9));

    // Potential: shift the masked opponent both ways and combine.
    let pot_hv = vorrq_u64(vshlq_u64(mask_hv, sh_hv), vshlq_u64(mask_hv, sh_hv_neg));
    let pot_d = vorrq_u64(vshlq_u64(mask_d, sh_d), vshlq_u64(mask_d, sh_d_neg));
    let pot_all = vorrq_u64(pot_hv, pot_d);
    let potential = vgetq_lane_u64::<0>(pot_all) | vgetq_lane_u64::<1>(pot_all);

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

    (moves & empty, potential & empty)
}
