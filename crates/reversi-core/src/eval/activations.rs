//! Activation functions for neural network evaluation.
//!
//! Reference:
//! - [Clipped ReLU](https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/clipped_relu.h)
//! - [Squared Clipped ReLU](https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/sqr_clipped_relu.h)

cfg_select! {
    all(target_arch = "x86_64", target_feature = "avx2") => {
        use std::arch::x86_64::*;
        const AVX2_SIMD_WIDTH: usize = std::mem::size_of::<__m256i>() / std::mem::size_of::<u8>();
        const SSE2_SIMD_WIDTH: usize = std::mem::size_of::<__m128i>() / std::mem::size_of::<u8>();
    }
    all(target_arch = "aarch64", target_feature = "neon") => {
        const NEON_SIMD_WIDTH: usize = 16;
    }
    _ => {}
}

const HIDDEN_WEIGHT_SCALE_BITS: i32 = 6;

/// Applies a clipped ReLU activation function to `input`.
///
/// Values are right-shifted by `HIDDEN_WEIGHT_SCALE_BITS` (6) and clamped to [0, 255].
///
/// # Safety
///
/// On x86-64 with AVX2, both `input` and `output` must be 32-byte aligned
/// (or 16-byte aligned when `SIZE` is not a multiple of 32).
#[inline(always)]
pub fn clipped_relu<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx2") => {
            unsafe { clipped_relu_avx2::<SIZE>(input, output) };
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            unsafe { clipped_relu_neon::<SIZE>(input, output) };
        }
        _ => {
            clipped_relu_fallback(input, output, 0);
        }
    }
}

/// Clipped ReLU with AVX2 SIMD optimization.
///
/// # Safety
///
/// Both `input` and `output` must be 32-byte aligned for AVX2 loads/stores.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn clipped_relu_avx2<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    unsafe {
        if SIZE.is_multiple_of(AVX2_SIMD_WIDTH) {
            let num_chunks = SIZE / AVX2_SIMD_WIDTH;
            let shuffle: __m256i = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
            let input_ptr = input.as_ptr() as *const __m256i;
            let output_ptr = output.as_mut_ptr() as *mut __m256i;
            for i in 0..num_chunks {
                let packed0 = _mm256_packus_epi32(
                    _mm256_load_si256(input_ptr.add(i * 4)),
                    _mm256_load_si256(input_ptr.add(i * 4 + 1)),
                );
                let packed1 = _mm256_packus_epi32(
                    _mm256_load_si256(input_ptr.add(i * 4 + 2)),
                    _mm256_load_si256(input_ptr.add(i * 4 + 3)),
                );

                let words0 = _mm256_srli_epi16(packed0, HIDDEN_WEIGHT_SCALE_BITS);
                let words1 = _mm256_srli_epi16(packed1, HIDDEN_WEIGHT_SCALE_BITS);

                _mm256_store_si256(
                    output_ptr.add(i),
                    _mm256_permutevar8x32_epi32(_mm256_packus_epi16(words0, words1), shuffle),
                );
            }
            return;
        }

        let num_chunks = SIZE / SSE2_SIMD_WIDTH;
        let input_ptr = input.as_ptr() as *const __m128i;
        let output_ptr = output.as_mut_ptr() as *mut __m128i;
        for i in 0..num_chunks {
            let words0 = _mm_srli_epi16(
                _mm_packus_epi32(
                    _mm_load_si128(input_ptr.add(i * 4)),
                    _mm_load_si128(input_ptr.add(i * 4 + 1)),
                ),
                HIDDEN_WEIGHT_SCALE_BITS,
            );

            let words1 = _mm_srli_epi16(
                _mm_packus_epi32(
                    _mm_load_si128(input_ptr.add(i * 4 + 2)),
                    _mm_load_si128(input_ptr.add(i * 4 + 3)),
                ),
                HIDDEN_WEIGHT_SCALE_BITS,
            );

            _mm_store_si128(output_ptr.add(i), _mm_packus_epi16(words0, words1));
        }
    }

    let start_idx = SIZE / SSE2_SIMD_WIDTH * SSE2_SIMD_WIDTH;
    clipped_relu_fallback(input, output, start_idx);
}

/// Clipped ReLU with ARM NEON SIMD optimization.
///
/// Processes 16 `i32` elements per iteration into a single `uint8x16_t` store,
/// then hands the remainder off to `clipped_relu_fallback`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn clipped_relu_neon<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    use std::arch::aarch64::*;
    let num_chunks = SIZE / NEON_SIMD_WIDTH;
    unsafe {
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        for i in 0..num_chunks {
            let base = i * NEON_SIMD_WIDTH;
            let v0 = vld1q_s32(input_ptr.add(base));
            let v1 = vld1q_s32(input_ptr.add(base + 4));
            let v2 = vld1q_s32(input_ptr.add(base + 8));
            let v3 = vld1q_s32(input_ptr.add(base + 12));

            let w0 = vcombine_u16(vqmovun_s32(v0), vqmovun_s32(v1));
            let w1 = vcombine_u16(vqmovun_s32(v2), vqmovun_s32(v3));

            let w0 = vshrq_n_u16::<HIDDEN_WEIGHT_SCALE_BITS>(w0);
            let w1 = vshrq_n_u16::<HIDDEN_WEIGHT_SCALE_BITS>(w1);

            let bytes = vcombine_u8(vqmovn_u16(w0), vqmovn_u16(w1));
            vst1q_u8(output_ptr.add(base), bytes);
        }
    }
    let start_idx = num_chunks * NEON_SIMD_WIDTH;
    clipped_relu_fallback(input, output, start_idx);
}

/// Clipped ReLU scalar fallback implementation.
#[inline(always)]
fn clipped_relu_fallback(input: &[i32], output: &mut [u8], start_idx: usize) {
    for i in start_idx..input.len() {
        let val = input[i] >> HIDDEN_WEIGHT_SCALE_BITS;
        output[i] = val.clamp(0, 255) as u8;
    }
}

/// Applies the Stockfish-style square-clipped activation to `input`.
///
/// Negative inputs are squared just like positive ones (no rectification) before scaling and clipping.
/// Output = min((input² >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8)), 255)
///
/// # Safety
///
/// On x86-64 with AVX2, both `input` and `output` must be 16-byte aligned
/// for SSE2 loads/stores.
#[inline(always)]
pub fn sqr_clipped_relu<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx2") => {
            unsafe { sqr_clipped_relu_avx2::<SIZE>(input, output) };
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            unsafe { sqr_clipped_relu_neon::<SIZE>(input, output) };
        }
        _ => {
            sqr_clipped_relu_fallback(input, output, 0);
        }
    }
}

/// Square-clipped activation with AVX2 SIMD optimization.
///
/// Uses SSE2 instructions (128-bit) for processing.
///
/// # Safety
///
/// Both `input` and `output` must be 16-byte aligned for SSE2 loads/stores.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn sqr_clipped_relu_avx2<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    let num_chunks = SIZE / SSE2_SIMD_WIDTH;

    unsafe {
        let input_ptr = input.as_ptr() as *const __m128i;
        let output_ptr = output.as_mut_ptr() as *mut __m128i;

        for i in 0..num_chunks {
            let mut words0 = _mm_packs_epi32(
                _mm_load_si128(input_ptr.add(i * 4)),
                _mm_load_si128(input_ptr.add(i * 4 + 1)),
            );
            let mut words1 = _mm_packs_epi32(
                _mm_load_si128(input_ptr.add(i * 4 + 2)),
                _mm_load_si128(input_ptr.add(i * 4 + 3)),
            );

            const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8 - 16;
            words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), SHIFT);
            words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), SHIFT);
            _mm_store_si128(output_ptr.add(i), _mm_packus_epi16(words0, words1));
        }
    }

    let start_idx = num_chunks * SSE2_SIMD_WIDTH;
    sqr_clipped_relu_fallback(input, output, start_idx);
}

/// Square-clipped activation with ARM NEON SIMD optimization.
///
/// Processes 16 `i32` elements per iteration. Signed-saturating pack to i16,
/// full-width signed square (`vmull_s16` / `vmull_high_s16`), arithmetic shift
/// right by `2 * HIDDEN_WEIGHT_SCALE_BITS + 8`, then saturating narrow to u8.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn sqr_clipped_relu_neon<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    use std::arch::aarch64::*;
    let num_chunks = SIZE / NEON_SIMD_WIDTH;
    const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8;
    unsafe {
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        for i in 0..num_chunks {
            let base = i * NEON_SIMD_WIDTH;
            let v0 = vld1q_s32(input_ptr.add(base));
            let v1 = vld1q_s32(input_ptr.add(base + 4));
            let v2 = vld1q_s32(input_ptr.add(base + 8));
            let v3 = vld1q_s32(input_ptr.add(base + 12));

            let s0 = vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1));
            let s1 = vcombine_s16(vqmovn_s32(v2), vqmovn_s32(v3));

            let p0_lo = vmull_s16(vget_low_s16(s0), vget_low_s16(s0));
            let p0_hi = vmull_high_s16(s0, s0);
            let p1_lo = vmull_s16(vget_low_s16(s1), vget_low_s16(s1));
            let p1_hi = vmull_high_s16(s1, s1);

            // Squares are non-negative, so arithmetic >> equals logical >>.
            let q0_lo = vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p0_lo));
            let q0_hi = vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p0_hi));
            let q1_lo = vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p1_lo));
            let q1_hi = vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p1_hi));

            let w0 = vcombine_u16(vqmovn_u32(q0_lo), vqmovn_u32(q0_hi));
            let w1 = vcombine_u16(vqmovn_u32(q1_lo), vqmovn_u32(q1_hi));
            let bytes = vcombine_u8(vqmovn_u16(w0), vqmovn_u16(w1));
            vst1q_u8(output_ptr.add(base), bytes);
        }
    }
    let start_idx = num_chunks * NEON_SIMD_WIDTH;
    sqr_clipped_relu_fallback(input, output, start_idx);
}

/// Square-clipped activation scalar fallback implementation.
#[inline(always)]
fn sqr_clipped_relu_fallback(input: &[i32], output: &mut [u8], start_idx: usize) {
    for i in start_idx..input.len() {
        let val = ((input[i] * input[i]) as u64 >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8)).min(255);
        output[i] = val as u8;
    }
}

/// Applies the Squared Clipped ReLU (SCReLU) activation function to `input`.
///
/// Clamps input to [0, 255 << HIDDEN_WEIGHT_SCALE_BITS], squares, then scales down.
/// Output = (clamp(input, 0, max)² >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8))
///
/// # Safety
///
/// On x86-64 with AVX2, both `input` and `output` must be 32-byte aligned
/// (or 16-byte aligned when `SIZE` is not a multiple of 32).
#[inline(always)]
pub fn screlu<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx2") => {
            unsafe { screlu_avx2::<SIZE>(input, output) };
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            unsafe { screlu_neon::<SIZE>(input, output) };
        }
        _ => {
            screlu_fallback(input, output, 0);
        }
    }
}

/// Squared Clipped ReLU with AVX2 SIMD optimization.
///
/// # Safety
///
/// Both `input` and `output` must be 32-byte aligned for AVX2 loads/stores
/// (or 16-byte aligned when `SIZE` is not a multiple of 32).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
fn screlu_avx2<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    unsafe {
        if SIZE.is_multiple_of(AVX2_SIMD_WIDTH) {
            let num_chunks = SIZE / AVX2_SIMD_WIDTH;
            let shuffle: __m256i = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
            let input_ptr = input.as_ptr() as *const __m256i;
            let output_ptr = output.as_mut_ptr() as *mut __m256i;
            let max_val = _mm256_set1_epi16(255 << HIDDEN_WEIGHT_SCALE_BITS);

            for i in 0..num_chunks {
                let mut words0 = _mm256_packus_epi32(
                    _mm256_load_si256(input_ptr.add(i * 4)),
                    _mm256_load_si256(input_ptr.add(i * 4 + 1)),
                );
                let mut words1 = _mm256_packus_epi32(
                    _mm256_load_si256(input_ptr.add(i * 4 + 2)),
                    _mm256_load_si256(input_ptr.add(i * 4 + 3)),
                );

                words0 = _mm256_min_epi16(words0, max_val);
                words1 = _mm256_min_epi16(words1, max_val);

                const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8 - 16;
                words0 = _mm256_srli_epi16(_mm256_mulhi_epu16(words0, words0), SHIFT);
                words1 = _mm256_srli_epi16(_mm256_mulhi_epu16(words1, words1), SHIFT);

                _mm256_store_si256(
                    output_ptr.add(i),
                    _mm256_permutevar8x32_epi32(_mm256_packus_epi16(words0, words1), shuffle),
                );
            }
            return;
        }

        let num_chunks = SIZE / SSE2_SIMD_WIDTH;
        let input_ptr = input.as_ptr() as *const __m128i;
        let output_ptr = output.as_mut_ptr() as *mut __m128i;
        let max_val = _mm_set1_epi16(255 << HIDDEN_WEIGHT_SCALE_BITS);

        for i in 0..num_chunks {
            let mut words0 = _mm_packus_epi32(
                _mm_load_si128(input_ptr.add(i * 4)),
                _mm_load_si128(input_ptr.add(i * 4 + 1)),
            );
            let mut words1 = _mm_packus_epi32(
                _mm_load_si128(input_ptr.add(i * 4 + 2)),
                _mm_load_si128(input_ptr.add(i * 4 + 3)),
            );

            words0 = _mm_min_epi16(words0, max_val);
            words1 = _mm_min_epi16(words1, max_val);

            const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8 - 16;
            words0 = _mm_srli_epi16(_mm_mulhi_epu16(words0, words0), SHIFT);
            words1 = _mm_srli_epi16(_mm_mulhi_epu16(words1, words1), SHIFT);

            _mm_store_si128(output_ptr.add(i), _mm_packus_epi16(words0, words1));
        }
    }

    let start_idx = SIZE / SSE2_SIMD_WIDTH * SSE2_SIMD_WIDTH;
    screlu_fallback(input, output, start_idx);
}

/// Squared Clipped ReLU with ARM NEON SIMD optimization.
///
/// Processes 16 `i32` elements per iteration. Unsigned-saturating pack to u16,
/// unsigned clamp to `[0, 255 << HIDDEN_WEIGHT_SCALE_BITS]`, full-width unsigned
/// square, logical shift right by `2 * HIDDEN_WEIGHT_SCALE_BITS + 8`, then
/// narrow to u8.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn screlu_neon<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
    use std::arch::aarch64::*;
    let num_chunks = SIZE / NEON_SIMD_WIDTH;
    const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8;
    unsafe {
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        let max_val = vdupq_n_u16((255 << HIDDEN_WEIGHT_SCALE_BITS) as u16);
        for i in 0..num_chunks {
            let base = i * NEON_SIMD_WIDTH;
            let v0 = vld1q_s32(input_ptr.add(base));
            let v1 = vld1q_s32(input_ptr.add(base + 4));
            let v2 = vld1q_s32(input_ptr.add(base + 8));
            let v3 = vld1q_s32(input_ptr.add(base + 12));

            let w0 = vminq_u16(vcombine_u16(vqmovun_s32(v0), vqmovun_s32(v1)), max_val);
            let w1 = vminq_u16(vcombine_u16(vqmovun_s32(v2), vqmovun_s32(v3)), max_val);

            let p0_lo = vmull_u16(vget_low_u16(w0), vget_low_u16(w0));
            let p0_hi = vmull_high_u16(w0, w0);
            let p1_lo = vmull_u16(vget_low_u16(w1), vget_low_u16(w1));
            let p1_hi = vmull_high_u16(w1, w1);

            let n0 = vcombine_u16(
                vqmovn_u32(vshrq_n_u32::<SHIFT>(p0_lo)),
                vqmovn_u32(vshrq_n_u32::<SHIFT>(p0_hi)),
            );
            let n1 = vcombine_u16(
                vqmovn_u32(vshrq_n_u32::<SHIFT>(p1_lo)),
                vqmovn_u32(vshrq_n_u32::<SHIFT>(p1_hi)),
            );
            let bytes = vcombine_u8(vqmovn_u16(n0), vqmovn_u16(n1));
            vst1q_u8(output_ptr.add(base), bytes);
        }
    }
    let start_idx = num_chunks * NEON_SIMD_WIDTH;
    screlu_fallback(input, output, start_idx);
}

/// Squared Clipped ReLU scalar fallback implementation.
#[inline(always)]
fn screlu_fallback(input: &[i32], output: &mut [u8], start_idx: usize) {
    for i in start_idx..input.len() {
        let clamped = input[i].clamp(0, 255 << HIDDEN_WEIGHT_SCALE_BITS) as u64;
        const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8;
        let val = (clamped * clamped) >> SHIFT;
        output[i] = val as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::align::Align64;

    fn expected_screlu(val: i32) -> u8 {
        let clamped = val.clamp(0, 255 << HIDDEN_WEIGHT_SCALE_BITS) as u64;
        let shift = (HIDDEN_WEIGHT_SCALE_BITS * 2 + 8) as u32;
        ((clamped * clamped) >> shift) as u8
    }

    #[test]
    fn test_clipped_relu_fallback() {
        const SIZE: usize = 6;

        let input_data = [100, -50, 200, i32::MAX, i32::MIN, 0];
        let input = Align64(input_data);
        let mut output = Align64([0; SIZE]);

        clipped_relu_fallback(input.as_slice(), output.as_mut_slice(), 0);

        let expected = [1, 0, 3, 255, 0, 0];
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_clipped_relu() {
        const SIZE: usize = 32;

        let mut input_data = [0i32; SIZE];
        let mut expected = [0u8; SIZE];

        input_data[0] = 100;
        expected[0] = 1;

        input_data[1] = 0;
        expected[1] = 0;

        input_data[2] = 1000;
        expected[2] = 15;

        input_data[3] = i32::MAX;
        expected[3] = 255;

        input_data[4] = i32::MIN;
        expected[4] = 0;

        input_data[5] = 0;
        expected[5] = 0;

        input_data[6] = 255 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[6] = 255;

        input_data[7] = (255 << HIDDEN_WEIGHT_SCALE_BITS) + 1;
        expected[7] = 255;

        let input = Align64(input_data);
        let mut output = Align64::<[u8; SIZE]>::default();
        clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_sqr_clipped_relu_fallback() {
        const SIZE: usize = 6;

        let input_data = [10, -5, 20, 5000, -5000, 256 << HIDDEN_WEIGHT_SCALE_BITS];
        let input = Align64(input_data);
        let mut output = Align64([0; SIZE]);

        sqr_clipped_relu_fallback(input.as_slice(), output.as_mut_slice(), 0);

        let expected = [0, 0, 0, 23, 23, 255];
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_sqr_clipped_relu() {
        const SIZE: usize = 16;

        let mut input_data = [0i32; SIZE];
        let mut expected = [0u8; SIZE];

        input_data[0] = 10;
        expected[0] = 0;

        input_data[1] = -5;
        expected[1] = 0;

        input_data[2] = 5000;
        expected[2] = 23;

        input_data[3] = -5000;
        expected[3] = 23;

        input_data[4] = 1000;
        expected[4] = 0;

        input_data[5] = -1000;
        expected[5] = 0;

        input_data[6] = 255 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[6] = 254;

        input_data[7] = 256 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[7] = 255;

        let input = Align64(input_data);
        let mut output = Align64([0; SIZE]);
        sqr_clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_screlu_fallback() {
        const SIZE: usize = 8;

        let input_data = [
            -5000,
            -1,
            0,
            (1 << HIDDEN_WEIGHT_SCALE_BITS) - 1,
            1 << (HIDDEN_WEIGHT_SCALE_BITS + 4),
            2048,
            8192,
            (255 << HIDDEN_WEIGHT_SCALE_BITS) + 1024,
        ];
        let input = Align64(input_data);
        let mut output = Align64([0u8; SIZE]);

        screlu_fallback(input.as_slice(), output.as_mut_slice(), 0);

        let mut expected = [0u8; SIZE];
        for (idx, value) in input_data.iter().enumerate() {
            expected[idx] = expected_screlu(*value);
        }

        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_screlu() {
        // Size divisible by AVX2 width
        {
            const SIZE: usize = 32;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32 - 10) * 1024;
                expected[i] = expected_screlu(input_data[i]);
            }

            let input = Align64(input_data);
            let mut output = Align64([0u8; SIZE]);
            screlu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }

        // Size not divisible by AVX2 width
        {
            const SIZE: usize = 37;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32 * 1500) - 20000;
                expected[i] = expected_screlu(input_data[i]);
            }

            let input = Align64(input_data);
            let mut output = Align64([0u8; SIZE]);
            screlu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }
    }

    #[test]
    fn test_screlu_consistency() {
        const SIZE: usize = 96;
        let mut input_data = [0i32; SIZE];

        for (idx, value) in input_data.iter_mut().enumerate() {
            let raw = ((idx * 9876 + 4321) % (400 << HIDDEN_WEIGHT_SCALE_BITS)) as i32;
            *value = raw - (200 << HIDDEN_WEIGHT_SCALE_BITS);
        }

        let input = Align64(input_data);
        let mut output_main = Align64([0u8; SIZE]);
        let mut output_fallback = Align64([0u8; SIZE]);

        screlu::<SIZE>(input.as_slice(), output_main.as_mut_slice());
        screlu_fallback(input.as_slice(), output_fallback.as_mut_slice(), 0);

        assert_eq!(output_main.as_ref(), output_fallback.as_ref());
    }

    #[test]
    fn test_clipped_relu_edge_cases() {
        const SIZE: usize = 16;
        let mut input_data = [0i32; SIZE];
        let mut expected = [0u8; SIZE];

        // Test negative values get clipped to 0
        input_data[0] = -1;
        expected[0] = 0;

        input_data[1] = -1000;
        expected[1] = 0;

        // Test exact boundary values
        input_data[2] = 255 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[2] = 255;

        input_data[3] = 256 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[3] = 255; // Should be clipped to 255

        // Test values just below the boundary
        input_data[4] = (255 << HIDDEN_WEIGHT_SCALE_BITS) - 1;
        expected[4] = 254;

        // Test very large positive values
        input_data[5] = 1000000;
        expected[5] = 255; // Should be clipped to 255

        // Test scaling edge cases
        input_data[6] = (1 << HIDDEN_WEIGHT_SCALE_BITS) - 1;
        expected[6] = 0; // Just below 1 after shifting

        input_data[7] = 1 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[7] = 1; // Exactly 1 after shifting

        let input = Align64(input_data);
        let mut output = Align64([0; SIZE]);
        clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_sqr_clipped_relu_edge_cases() {
        const SIZE: usize = 16;
        let mut input_data = [0i32; SIZE];
        let mut expected = [0u8; SIZE];

        // Test that negative values squared become positive
        input_data[0] = -100;
        input_data[1] = 100;
        // Both should produce same result after squaring
        expected[0] = 0;
        expected[1] = 0;

        // Test maximum safe value that won't overflow
        // sqrt(255 * 2^20) ≈ 16352
        input_data[2] = 255 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[2] = 254;

        input_data[3] = -(255 << HIDDEN_WEIGHT_SCALE_BITS);
        expected[3] = 254;

        // Test values that will saturate to 255
        input_data[4] = 256 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[4] = 255;

        input_data[5] = -(256 << HIDDEN_WEIGHT_SCALE_BITS);
        expected[5] = 255;

        // Test boundary for producing 1
        // sqrt(1 * 2^20) = 1024; smaller magnitudes like 724 still round to 0
        input_data[6] = 724;
        expected[6] = 0; // Actually produces 0 with this implementation

        input_data[7] = -724;
        expected[7] = 0; // Actually produces 0 with this implementation

        let input = Align64(input_data);
        let mut output = Align64([0; SIZE]);
        sqr_clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_clipped_relu_different_sizes() {
        // Test size 8 (smaller than AVX2 width)
        {
            const SIZE: usize = 8;
            let input_data = [
                100,
                -50,
                8192,
                16384,
                -8192,
                0,
                127 << HIDDEN_WEIGHT_SCALE_BITS,
                128 << HIDDEN_WEIGHT_SCALE_BITS,
            ];
            let expected = [1, 0, 128, 255, 0, 0, 127, 128];

            let input = Align64(input_data);
            let mut output = Align64([0; SIZE]);
            clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }

        // Test size 64 (multiple of AVX2 width)
        {
            const SIZE: usize = 64;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            // Set some test values
            for i in 0..SIZE {
                input_data[i] = (i as i32 - 32) * 100;
                let val = input_data[i] >> HIDDEN_WEIGHT_SCALE_BITS;
                expected[i] = val.clamp(0, 255) as u8;
            }

            let input = Align64(input_data);
            let mut output = Align64([0; SIZE]);
            clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }

        // Test odd size (not divisible by AVX2 width)
        {
            const SIZE: usize = 37;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32) * 200 - 2000;
                let val = input_data[i] >> HIDDEN_WEIGHT_SCALE_BITS;
                expected[i] = val.clamp(0, 255) as u8;
            }

            let input = Align64(input_data);
            let mut output = Align64([0; SIZE]);
            clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }
    }

    #[test]
    fn test_sqr_clipped_relu_different_sizes() {
        // Test size 16 (exactly one chunk)
        {
            const SIZE: usize = 16;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32 - 8) * 500;
                let val = ((input_data[i] as i64 * input_data[i] as i64)
                    >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8))
                    .min(255);
                expected[i] = val as u8;
            }

            let input = Align64(input_data);
            let mut output = Align64([0; SIZE]);
            sqr_clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }

        // Test size 48 (multiple chunks)
        {
            const SIZE: usize = 48;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32 - 24) * 300;
                let val = ((input_data[i] as i64 * input_data[i] as i64)
                    >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8))
                    .min(255);
                expected[i] = val as u8;
            }

            let input = Align64(input_data);
            let mut output = Align64([0; SIZE]);
            sqr_clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }

        // Test odd size
        {
            const SIZE: usize = 23;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32) * 400 - 4000;
                let val = ((input_data[i] as i64 * input_data[i] as i64)
                    >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8))
                    .min(255);
                expected[i] = val as u8;
            }

            let input = Align64(input_data);
            let mut output = Align64([0; SIZE]);
            sqr_clipped_relu::<SIZE>(input.as_slice(), output.as_mut_slice());
            assert_eq!(output.as_ref(), &expected);
        }
    }

    #[test]
    fn test_clipped_relu_consistency() {
        // Test that AVX2 and fallback produce same results
        const SIZE: usize = 128;
        let mut input_data = [0i32; SIZE];

        for (i, val) in input_data.iter_mut().enumerate() {
            *val = ((i * 12345 + 6789) % 20000) as i32 - 10000;
        }

        let input = Align64(input_data);
        let mut output_main = Align64([0; SIZE]);
        let mut output_fallback = Align64([0; SIZE]);

        clipped_relu::<SIZE>(input.as_slice(), output_main.as_mut_slice());
        clipped_relu_fallback(input.as_slice(), output_fallback.as_mut_slice(), 0);

        assert_eq!(output_main.as_ref(), output_fallback.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn clipped_relu_neon_matches_fallback() {
        fn run<const SIZE: usize>(seed: usize) {
            let mut input_data = [0i32; SIZE];
            for (i, v) in input_data.iter_mut().enumerate() {
                *v = ((i * 13 + seed) as i32 * 257).wrapping_sub(12_000);
            }
            let input = Align64(input_data);
            let mut out_neon = Align64([0u8; SIZE]);
            let mut out_fb = Align64([0u8; SIZE]);
            // SAFETY: NEON is the aarch64 baseline, asserted by the cfg above.
            unsafe { clipped_relu_neon::<SIZE>(input.as_slice(), out_neon.as_mut_slice()) };
            clipped_relu_fallback(input.as_slice(), out_fb.as_mut_slice(), 0);
            assert_eq!(out_neon.as_ref(), out_fb.as_ref());
        }
        run::<32>(1);
        run::<80>(7);
        run::<37>(11);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn sqr_clipped_relu_neon_matches_fallback() {
        // Keep inputs under sqrt(i32::MAX) so the fallback's i32*i32 multiplication
        // does not wrap (NEON pre-saturates to i16, producing a different result).
        fn run<const SIZE: usize>(seed: usize) {
            let mut input_data = [0i32; SIZE];
            for (i, v) in input_data.iter_mut().enumerate() {
                *v = ((i * 9 + seed) as i32 * 37).wrapping_sub(2_000);
            }
            let input = Align64(input_data);
            let mut out_neon = Align64([0u8; SIZE]);
            let mut out_fb = Align64([0u8; SIZE]);
            // SAFETY: NEON is the aarch64 baseline, asserted by the cfg above.
            unsafe { sqr_clipped_relu_neon::<SIZE>(input.as_slice(), out_neon.as_mut_slice()) };
            sqr_clipped_relu_fallback(input.as_slice(), out_fb.as_mut_slice(), 0);
            assert_eq!(out_neon.as_ref(), out_fb.as_ref());
        }
        run::<16>(2);
        run::<48>(5);
        run::<23>(9);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn screlu_neon_matches_fallback() {
        fn run<const SIZE: usize>(seed: usize) {
            let mut input_data = [0i32; SIZE];
            for (i, v) in input_data.iter_mut().enumerate() {
                let raw = ((i * 9876 + seed) % (400 << HIDDEN_WEIGHT_SCALE_BITS)) as i32;
                *v = raw - (200 << HIDDEN_WEIGHT_SCALE_BITS);
            }
            let input = Align64(input_data);
            let mut out_neon = Align64([0u8; SIZE]);
            let mut out_fb = Align64([0u8; SIZE]);
            // SAFETY: NEON is the aarch64 baseline, asserted by the cfg above.
            unsafe { screlu_neon::<SIZE>(input.as_slice(), out_neon.as_mut_slice()) };
            screlu_fallback(input.as_slice(), out_fb.as_mut_slice(), 0);
            assert_eq!(out_neon.as_ref(), out_fb.as_ref());
        }
        run::<32>(4321);
        run::<96>(1234);
        run::<37>(777);
    }

    #[test]
    fn test_sqr_clipped_relu_consistency() {
        // Test that AVX2 and fallback produce same results
        const SIZE: usize = 80;
        let mut input_data = [0i32; SIZE];

        for (i, val) in input_data.iter_mut().enumerate() {
            *val = ((i * 9876 + 1000) % 10000) as i32 - 3000;
        }

        let input = Align64(input_data);
        let mut output_main = Align64([0; SIZE]);
        let mut output_fallback = Align64([0; SIZE]);

        sqr_clipped_relu::<SIZE>(input.as_slice(), output_main.as_mut_slice());
        sqr_clipped_relu_fallback(input.as_slice(), output_fallback.as_mut_slice(), 0);

        assert_eq!(output_main.as_ref(), output_fallback.as_ref());
    }
}
