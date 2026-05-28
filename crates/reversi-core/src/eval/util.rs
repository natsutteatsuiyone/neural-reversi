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
///
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
///
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
///
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
///
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
///
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
///
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

/// Multiplies unsigned 8-bit lanes by signed 8-bit lanes and accumulates into 32-bit results.
///
/// Emulates `USDOT` on dotprod-only hardware (no FEAT_I8MM) by splitting the
/// unsigned input into its low 7 bits and the sign bit, then issuing two
/// `SDOT`s. The sign-bit contribution is subtracted to recover the +128
/// magnitude that the i8 reinterpret negates.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon,dotprod")]
#[inline]
#[allow(dead_code)]
pub fn neon_dpbusd_s32_dotprod(
    src: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::uint8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;

    let high_bit = vdupq_n_u8(0x80);
    let a_low7_s8 = vreinterpretq_s8_u8(vbicq_u8(a, high_bit));
    let a_msb_i8 = vreinterpretq_s8_u8(vandq_u8(a, high_bit));
    let with_low = vdotq_s32(src, a_low7_s8, b);
    let neg_high = vdotq_s32(vdupq_n_s32(0), a_msb_i8, b);
    vsubq_s32(with_low, neg_high)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::pattern_feature::calc_pattern_size;

    #[test]
    fn ceil_to_multiple_handles_const_zero_exact_and_rounded_values() {
        const ZERO: usize = ceil_to_multiple(0, 8);
        const EXACT: usize = ceil_to_multiple(64, 32);
        const ROUNDED: usize = ceil_to_multiple(65, 32);
        const BASE_ONE: usize = ceil_to_multiple(17, 1);

        assert_eq!(ZERO, 0);
        assert_eq!(EXACT, 64);
        assert_eq!(ROUNDED, 96);
        assert_eq!(BASE_ONE, 17);
    }

    #[test]
    fn clone_biases_copies_the_prefix_into_a_64_byte_aligned_array() {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        struct Bias {
            value: i16,
            phase: u8,
        }

        let source = [
            Bias {
                value: -7,
                phase: 0,
            },
            Bias {
                value: 13,
                phase: 1,
            },
            Bias {
                value: i16::MIN,
                phase: 2,
            },
            Bias {
                value: i16::MAX,
                phase: 3,
            },
            Bias {
                value: 99,
                phase: 4,
            },
        ];

        let cloned: Align64<[Bias; 4]> = clone_biases(&source);

        assert_eq!(cloned.as_slice(), &source[..4]);
        assert_eq!((cloned.as_ptr() as usize) % 64, 0);
    }

    #[test]
    fn clone_biases_supports_empty_outputs() {
        let cloned: Align64<[u32; 0]> = clone_biases(&[]);

        assert!(cloned.as_slice().is_empty());
        assert_eq!((cloned.as_ptr() as usize) % 64, 0);
    }

    #[test]
    fn feature_offset_adds_each_pattern_value_to_its_table_offset() {
        let mut pattern_feature = PatternFeature::new();

        for idx in 0..NUM_PATTERN_FEATURES {
            let value = ((idx * 4099 + 17) % calc_pattern_size(idx)) as u16;
            pattern_feature[idx] = value;

            assert_eq!(
                feature_offset(&pattern_feature, idx),
                PATTERN_FEATURE_OFFSETS[idx] + usize::from(value),
                "feature {idx}"
            );
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "feature index")]
    fn feature_offset_rejects_out_of_bounds_indices_in_debug_builds() {
        let pattern_feature = PatternFeature::new();

        let _ = feature_offset(&pattern_feature, NUM_PATTERN_FEATURES);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn scalar_dpbusd(src: [i32; 4], a: [u8; 16], b: [i8; 16]) -> [i32; 4] {
        let mut out = src;
        for (lane, out_lane) in out.iter_mut().enumerate() {
            let base = lane * 4;
            *out_lane += i32::from(a[base]) * i32::from(b[base])
                + i32::from(a[base + 1]) * i32::from(b[base + 1])
                + i32::from(a[base + 2]) * i32::from(b[base + 2])
                + i32::from(a[base + 3]) * i32::from(b[base + 3]);
        }
        out
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn generated_dpbusd_case(mut state: u32) -> ([i32; 4], [u8; 16], [i8; 16]) {
        fn next_u32(state: &mut u32) -> u32 {
            *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *state
        }

        let mut src = [0i32; 4];
        let mut a = [0u8; 16];
        let mut b = [0i8; 16];

        for v in &mut src {
            *v = (next_u32(&mut state) & 0x3fff) as i32 - 8192;
        }
        for v in &mut a {
            *v = (next_u32(&mut state) >> 24) as u8;
        }
        for v in &mut b {
            *v = (next_u32(&mut state) >> 24) as u8 as i8;
        }

        (src, a, b)
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    macro_rules! assert_dpbusd_kernel_matches_scalar {
        ($kernel:path, $src:expr, $a:expr, $b:expr) => {{
            let src = $src;
            let a = $a;
            let b = $b;
            let mut actual = [0i32; 4];
            unsafe {
                let actual_vec = $kernel(
                    std::arch::aarch64::vld1q_s32(src.as_ptr()),
                    std::arch::aarch64::vld1q_u8(a.as_ptr()),
                    std::arch::aarch64::vld1q_s8(b.as_ptr()),
                );
                std::arch::aarch64::vst1q_s32(actual.as_mut_ptr(), actual_vec);
            }

            assert_eq!(actual, scalar_dpbusd(src, a, b));
        }};
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    const DPBUSD_EDGE_CASES: [([i32; 4], [u8; 16], [i8; 16]); 3] = [
        (
            [0, 1, -2, 12_345],
            [0, 1, 127, 128, 255, 200, 129, 64, 32, 16, 8, 4, 3, 2, 1, 0],
            [
                -128, -1, 0, 1, 127, -127, 64, -64, 32, -32, 16, -16, 8, -8, 4, -4,
            ],
        ),
        (
            [i32::MIN / 4, -1024, 1024, i32::MAX / 4],
            [
                255, 255, 255, 255, 128, 128, 128, 128, 127, 127, 127, 127, 1, 2, 3, 4,
            ],
            [
                127, -128, 1, -1, 127, -128, 1, -1, 127, -128, 1, -1, -1, 1, -2, 2,
            ],
        ),
        (
            [-31, 0, 31, 1024],
            [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53],
            [
                53, -47, 43, -41, 37, -31, 29, -23, 19, -17, 13, -11, 7, -5, 3, -1,
            ],
        ),
    ];

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_dpbusd_s32_matches_scalar_reference() {
        for &(src, a, b) in &DPBUSD_EDGE_CASES {
            assert_dpbusd_kernel_matches_scalar!(neon_dpbusd_s32, src, a, b);
        }
        for seed in 0..32 {
            let (src, a, b) = generated_dpbusd_case(seed);
            assert_dpbusd_kernel_matches_scalar!(neon_dpbusd_s32, src, a, b);
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_dpbusd_s32_dotprod_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        for &(src, a, b) in &DPBUSD_EDGE_CASES {
            assert_dpbusd_kernel_matches_scalar!(neon_dpbusd_s32_dotprod, src, a, b);
        }
        for seed in 32..64 {
            let (src, a, b) = generated_dpbusd_case(seed);
            assert_dpbusd_kernel_matches_scalar!(neon_dpbusd_s32_dotprod, src, a, b);
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_dpbusd_s32_i8mm_matches_scalar_reference() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        for &(src, a, b) in &DPBUSD_EDGE_CASES {
            assert_dpbusd_kernel_matches_scalar!(neon_dpbusd_s32_i8mm, src, a, b);
        }
        for seed in 64..96 {
            let (src, a, b) = generated_dpbusd_case(seed);
            assert_dpbusd_kernel_matches_scalar!(neon_dpbusd_s32_i8mm, src, a, b);
        }
    }
}
