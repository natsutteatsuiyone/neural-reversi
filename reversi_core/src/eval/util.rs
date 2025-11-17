use crate::{
    eval::{constants::PATTERN_FEATURE_OFFSETS, pattern_feature::PatternFeature},
    util::align::Align64,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Clone an evaluation bias vector into a 64-byte aligned array.
///
/// # Type Parameters
/// - `T`: bias element type (e.g., `i16`, `f32`).
/// - `N`: number of bias elements to copy.
///
/// # Arguments
///
/// - `biases`: slice of bias elements to copy from. Must have at least `N` elements.
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

/// Compute the feature offset for a given pattern feature index.
///
/// # Arguments
///
/// * `pattern_feature` - Extracted pattern features from the board
/// * `idx` - Index of the pattern feature
///
/// # Returns
///
/// The computed feature offset as usize.
#[inline(always)]
pub fn feature_offset(pattern_feature: &PatternFeature, idx: usize) -> usize {
    *unsafe { PATTERN_FEATURE_OFFSETS.get_unchecked(idx) }
        + unsafe { pattern_feature.get_unchecked(idx) } as usize
}

/// Multiply signed 16-bit lanes by signed 16-bit lanes and accumulate into 32-bit results.
/// Matches the semantics of `VPMDPWSSD`, using a portable fallback when VNNI is unavailable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vl,avx512vnni")]
#[inline]
pub fn mm512_dpwssd_epi32<const USE_VNNI: bool>(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    if USE_VNNI {
        _mm512_dpwssd_epi32(src, a, b)
    } else {
        let products = _mm512_madd_epi16(a, b);
        _mm512_add_epi32(src, products)
    }
}

/// Multiply signed 16-bit lanes by signed 16-bit lanes and accumulate into 32-bit results.
/// Matches the semantics of `VPMDPWSSD`, using a portable fallback when VNNI is unavailable.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub fn mm256_dpwssd_epi32<const USE_VNNI: bool>(src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    if USE_VNNI {
        _mm256_dpwssd_avx_epi32(src, a, b)
    } else {
        let products = _mm256_madd_epi16(a, b);
        _mm256_add_epi32(src, products)
    }
}

/// Multiply unsigned 8-bit lanes by signed 8-bit lanes and accumulate into 32-bit results.
/// Emulates `VPDPBUSD`, expanding to a VNNI-free sequence when the instruction is missing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vl,avx512vnni")]
#[inline]
pub fn mm512_dpbusd_epi32<const USE_VNNI: bool>(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    if USE_VNNI {
        _mm512_dpbusd_epi32(src, a, b)
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

/// Multiply unsigned 8-bit lanes by signed 8-bit lanes and accumulate into 32-bit results.
/// Emulates `VPDPBUSD`, expanding to a VNNI-free sequence when the instruction is missing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
#[inline]
pub fn mm256_dpbusd_epi32<const USE_VNNI: bool>(src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    if USE_VNNI {
        _mm256_dpbusd_avx_epi32(src, a, b)
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub fn m256_hadd(sum_vec: __m256i) -> i32 {
    let mut sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum_vec),
        _mm256_extracti128_si256(sum_vec, 1),
    );
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    _mm_cvtsi128_si32(sum128)
}
