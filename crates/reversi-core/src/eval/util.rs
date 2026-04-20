//! Utility functions for neural network evaluation.

use crate::{
    eval::pattern_feature::{NUM_PATTERN_FEATURES, PATTERN_FEATURE_OFFSETS, PatternFeature},
    util::align::Align64,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::*;

/// Returns the smallest multiple of `base` that is greater than or equal to `n`.
///
/// Used to pad neural network layer dimensions to SIMD-friendly boundaries.
pub const fn ceil_to_multiple(n: usize, base: usize) -> usize {
    n.div_ceil(base) * base
}

/// Clones a bias vector into a 64-byte-aligned array.
///
/// # Safety
///
/// `biases` must have at least `N` elements. The function uses
/// `copy_nonoverlapping` without bounds checking.
#[inline(always)]
pub fn clone_biases<T: Copy, const N: usize>(biases: &[T]) -> Align64<[T; N]> {
    let mut acc = std::mem::MaybeUninit::<Align64<[T; N]>>::uninit();
    unsafe {
        std::ptr::copy_nonoverlapping(
            biases.as_ptr(),
            (*acc.as_mut_ptr()).as_mut_ptr() as *mut T,
            N,
        );
        acc.assume_init()
    }
}

/// Computes the feature offset for a given pattern feature index.
///
/// # Safety
///
/// `idx` must be less than [`NUM_PATTERN_FEATURES`]. The function uses
/// `get_unchecked` without bounds checking (debug builds assert this).
#[inline(always)]
pub fn feature_offset(pattern_feature: &PatternFeature, idx: usize) -> usize {
    debug_assert!(
        idx < NUM_PATTERN_FEATURES,
        "feature index {} out of bounds (max {})",
        idx,
        NUM_PATTERN_FEATURES - 1
    );
    *unsafe { PATTERN_FEATURE_OFFSETS.get_unchecked(idx) }
        + unsafe { pattern_feature.get_unchecked(idx) } as usize
}

/// Multiplies signed 16-bit lanes by signed 16-bit lanes and accumulates into 32-bit results.
/// Matches the semantics of `VPDPWSSD`, using a portable fallback when VNNI is unavailable.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512bw")]
#[inline]
#[allow(dead_code)]
pub fn mm512_dpwssd_epi32<const USE_VNNI: bool>(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    if USE_VNNI {
        unsafe { _mm512_dpwssd_epi32(src, a, b) }
    } else {
        let products = _mm512_madd_epi16(a, b);
        _mm512_add_epi32(src, products)
    }
}

/// Multiplies signed 16-bit lanes by signed 16-bit lanes and accumulates into 32-bit results.
/// Matches the semantics of `VPDPWSSD`, using a portable fallback when VNNI is unavailable.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub fn mm256_dpwssd_epi32<const USE_VNNI: bool>(src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    if USE_VNNI {
        unsafe { _mm256_dpwssd_avx_epi32(src, a, b) }
    } else {
        let products = _mm256_madd_epi16(a, b);
        _mm256_add_epi32(src, products)
    }
}

/// Multiplies unsigned 8-bit lanes by signed 8-bit lanes and accumulates into 32-bit results.
/// Emulates `VPDPBUSD`, expanding to a VNNI-free sequence when the instruction is missing.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[target_feature(enable = "avx512bw")]
#[inline]
#[allow(dead_code)]
pub fn mm512_dpbusd_epi32<const USE_VNNI: bool>(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    if USE_VNNI {
        unsafe { _mm512_dpbusd_epi32(src, a, b) }
    } else {
        let highest_bit = _mm512_set1_epi8(0x80u8 as i8);
        let ones16 = _mm512_set1_epi16(1);
        let a_low7 = _mm512_andnot_si512(highest_bit, a);
        let a_msb = _mm512_and_si512(a, highest_bit);
        let low7_i16 = _mm512_maddubs_epi16(a_low7, b);
        let msb_i16 = _mm512_maddubs_epi16(a_msb, b);
        let low7_i32 = _mm512_madd_epi16(low7_i16, ones16);
        let msb_i32 = _mm512_madd_epi16(msb_i16, ones16);
        _mm512_add_epi32(src, _mm512_add_epi32(low7_i32, msb_i32))
    }
}

/// Multiplies unsigned 8-bit lanes by signed 8-bit lanes and accumulates into 32-bit results.
/// Emulates `VPDPBUSD`, expanding to a VNNI-free sequence when the instruction is missing.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub fn mm256_dpbusd_epi32<const USE_VNNI: bool>(src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    if USE_VNNI {
        unsafe { _mm256_dpbusd_avx_epi32(src, a, b) }
    } else {
        let highest_bit = _mm256_set1_epi8(0x80u8 as i8);
        let ones16 = _mm256_set1_epi16(1);
        let a_low7 = _mm256_andnot_si256(highest_bit, a);
        let a_msb = _mm256_and_si256(a, highest_bit);
        let low7_i16 = _mm256_maddubs_epi16(a_low7, b);
        let msb_i16 = _mm256_maddubs_epi16(a_msb, b);
        let low7_i32 = _mm256_madd_epi16(low7_i16, ones16);
        let msb_i32 = _mm256_madd_epi16(msb_i16, ones16);
        _mm256_add_epi32(src, _mm256_add_epi32(low7_i32, msb_i32))
    }
}

/// Multiplies unsigned 8-bit lanes by signed 8-bit lanes and accumulates into 32-bit results.
/// Widen-and-reduce emulation of the `VUSDOT` operation; the hardware instruction is
/// gated behind an unstable Rust stdarch feature flag and thus cannot be used on stable.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
#[allow(dead_code)]
pub fn neon_dpbusd_s32(
    src: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::uint8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    let a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a)));
    let a_hi = vreinterpretq_s16_u16(vmovl_high_u8(a));
    let b_lo = vmovl_s8(vget_low_s8(b));
    let b_hi = vmovl_high_s8(b);

    let p0 = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
    let p1 = vmull_high_s16(a_lo, b_lo);
    let p2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
    let p3 = vmull_high_s16(a_hi, b_hi);

    let s01 = vpaddq_s32(p0, p1);
    let s23 = vpaddq_s32(p2, p3);
    let delta = vpaddq_s32(s01, s23);

    vaddq_s32(src, delta)
}

/// Multiplies unsigned 8-bit lanes by signed 8-bit lanes and accumulates into 32-bit results.
/// Maps directly to the `USDOT` instruction on nightly builds.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon,i8mm")]
#[inline]
#[allow(dead_code)]
pub fn neon_dpbusd_s32_i8mm(
    src: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::uint8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    vusdotq_s32(src, a, b)
}

/// Horizontally adds all 32-bit lanes in a 256-bit vector.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
pub fn m256_hadd(sum_vec: __m256i) -> i32 {
    let mut sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum_vec),
        _mm256_extracti128_si256(sum_vec, 1),
    );
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    _mm_cvtsi128_si32(sum128)
}
