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
/// Values are right-shifted by `HIDDEN_WEIGHT_SCALE_BITS` (6) and clamped to `[0, 255]`.
///
/// On x86-64 with AVX2, both `input` and `output` must be 32-byte aligned (or
/// 16-byte aligned when `SIZE` is not a multiple of 32). Misaligned buffers
/// cause undefined behavior in the SIMD loads.
#[inline(always)]
#[allow(dead_code)]
fn clipped_relu<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
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

/// Computes clipped ReLU using AVX2 SIMD.
///
/// # Safety
///
/// Both `input` and `output` must be 32-byte aligned for AVX2 loads/stores.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
unsafe fn clipped_relu_avx2<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
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

/// Computes clipped ReLU using ARM NEON SIMD.
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

            let w0 = vcombine_u16(
                vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v0),
                vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v1),
            );
            let w1 = vcombine_u16(
                vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v2),
                vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v3),
            );

            let bytes = vcombine_u8(vqmovn_u16(w0), vqmovn_u16(w1));
            vst1q_u8(output_ptr.add(base), bytes);
        }
    }
    let start_idx = num_chunks * NEON_SIMD_WIDTH;
    clipped_relu_fallback(input, output, start_idx);
}

/// Computes clipped ReLU using the scalar fallback.
#[inline(always)]
#[allow(dead_code)]
fn clipped_relu_fallback(input: &[i32], output: &mut [u8], start_idx: usize) {
    for i in start_idx..input.len() {
        let val = input[i] >> HIDDEN_WEIGHT_SCALE_BITS;
        output[i] = val.clamp(0, 255) as u8;
    }
}

/// Applies the Stockfish-style square-clipped activation to `input`.
///
/// Negative inputs are squared just like positive ones (no rectification) before scaling and clipping.
/// Output = min((input² >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8)), 255).
///
/// On x86-64 with AVX2, both `input` and `output` must be 16-byte aligned for
/// the SSE2 loads/stores. Misaligned buffers cause undefined behavior.
#[inline(always)]
#[allow(dead_code)]
fn sqr_clipped_relu<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
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

/// Applies [`sqr_clipped_relu::<16>`] followed by [`clipped_relu::<16>`] into a
/// single 32-byte output buffer.
///
/// The first 16 bytes receive the square-clipped output, and the next 16 bytes
/// receive the clipped-ReLU output.
///
/// On x86-64 with AVX2, both `input` and `output` must be 16-byte aligned for
/// the SSE2 loads/stores. Misaligned buffers cause undefined behavior.
///
/// [`sqr_clipped_relu::<16>`]: sqr_clipped_relu
/// [`clipped_relu::<16>`]: clipped_relu
#[inline(always)]
pub fn sqr_clipped_and_clipped_relu_16(input: &[i32], output: &mut [u8]) {
    debug_assert!(input.len() >= 16);
    debug_assert!(output.len() >= 32);

    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx2") => {
            unsafe { sqr_clipped_and_clipped_relu_16_avx2(input, output) };
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            unsafe { sqr_clipped_and_clipped_relu_16_neon(input, output) };
        }
        _ => {
            let (sqr_out, relu_out) = output[..32].split_at_mut(16);
            sqr_clipped_relu::<16>(input, sqr_out);
            clipped_relu::<16>(input, relu_out);
        }
    }
}

/// # Safety
///
/// `input` must contain at least 16 `i32` values and be 16-byte aligned.
/// `output` must contain at least 32 bytes and be 16-byte aligned.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sqr_clipped_and_clipped_relu_16_avx2(input: &[i32], output: &mut [u8]) {
    unsafe {
        let input_ptr = input.as_ptr() as *const __m128i;
        let output_ptr = output.as_mut_ptr() as *mut __m128i;

        let in0 = _mm_load_si128(input_ptr);
        let in1 = _mm_load_si128(input_ptr.add(1));
        let in2 = _mm_load_si128(input_ptr.add(2));
        let in3 = _mm_load_si128(input_ptr.add(3));

        let sqr_words0 = _mm_packs_epi32(in0, in1);
        let sqr_words1 = _mm_packs_epi32(in2, in3);

        const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8 - 16;
        let sqr_words0 = _mm_srli_epi16(_mm_mulhi_epi16(sqr_words0, sqr_words0), SHIFT);
        let sqr_words1 = _mm_srli_epi16(_mm_mulhi_epi16(sqr_words1, sqr_words1), SHIFT);
        _mm_store_si128(output_ptr, _mm_packus_epi16(sqr_words0, sqr_words1));

        let relu_words0 = _mm_srli_epi16(_mm_packus_epi32(in0, in1), HIDDEN_WEIGHT_SCALE_BITS);
        let relu_words1 = _mm_srli_epi16(_mm_packus_epi32(in2, in3), HIDDEN_WEIGHT_SCALE_BITS);
        _mm_store_si128(
            output_ptr.add(1),
            _mm_packus_epi16(relu_words0, relu_words1),
        );
    }
}

/// Applies the fused L1 activation using ARM NEON.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
fn sqr_clipped_and_clipped_relu_16_neon(input: &[i32], output: &mut [u8]) {
    use std::arch::aarch64::*;

    unsafe {
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();

        let v0 = vld1q_s32(input_ptr);
        let v1 = vld1q_s32(input_ptr.add(4));
        let v2 = vld1q_s32(input_ptr.add(8));
        let v3 = vld1q_s32(input_ptr.add(12));

        let relu_w0 = vcombine_u16(
            vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v0),
            vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v1),
        );
        let relu_w1 = vcombine_u16(
            vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v2),
            vqshrun_n_s32::<HIDDEN_WEIGHT_SCALE_BITS>(v3),
        );
        vst1q_u8(
            output_ptr.add(16),
            vcombine_u8(vqmovn_u16(relu_w0), vqmovn_u16(relu_w1)),
        );

        let s0 = vcombine_s16(vqmovn_s32(v0), vqmovn_s32(v1));
        let s1 = vcombine_s16(vqmovn_s32(v2), vqmovn_s32(v3));

        let p0_lo = vmull_s16(vget_low_s16(s0), vget_low_s16(s0));
        let p0_hi = vmull_high_s16(s0, s0);
        let p1_lo = vmull_s16(vget_low_s16(s1), vget_low_s16(s1));
        let p1_hi = vmull_high_s16(s1, s1);

        const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8;
        let sqr_w0 = vcombine_u16(
            vqmovn_u32(vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p0_lo))),
            vqmovn_u32(vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p0_hi))),
        );
        let sqr_w1 = vcombine_u16(
            vqmovn_u32(vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p1_lo))),
            vqmovn_u32(vreinterpretq_u32_s32(vshrq_n_s32::<SHIFT>(p1_hi))),
        );
        vst1q_u8(
            output_ptr,
            vcombine_u8(vqmovn_u16(sqr_w0), vqmovn_u16(sqr_w1)),
        );
    }
}

/// Computes the square-clipped activation using AVX2 SIMD.
///
/// Uses SSE2 instructions (128-bit) for processing.
///
/// # Safety
///
/// Both `input` and `output` must be 16-byte aligned for SSE2 loads/stores.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(dead_code)]
unsafe fn sqr_clipped_relu_avx2<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
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

/// Computes the square-clipped activation using ARM NEON SIMD.
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

/// Computes the square-clipped activation using the scalar fallback.
#[inline(always)]
#[allow(dead_code)]
fn sqr_clipped_relu_fallback(input: &[i32], output: &mut [u8], start_idx: usize) {
    for i in start_idx..input.len() {
        let saturated = i64::from(input[i].clamp(i16::MIN as i32, i16::MAX as i32));
        let val = ((saturated * saturated) as u64 >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8)).min(255);
        output[i] = val as u8;
    }
}

/// Applies the Squared Clipped ReLU (SCReLU) activation function to `input`.
///
/// Clamps input to `[0, 255 << HIDDEN_WEIGHT_SCALE_BITS]`, squares, then scales down.
/// Output = (clamp(input, 0, max)² >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 8)).
///
/// On x86-64 with AVX2, both `input` and `output` must be 32-byte aligned (or
/// 16-byte aligned when `SIZE` is not a multiple of 32). Misaligned buffers
/// cause undefined behavior in the SIMD loads.
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

/// Computes Squared Clipped ReLU using AVX2 SIMD.
///
/// # Safety
///
/// Both `input` and `output` must be 32-byte aligned for AVX2 loads/stores
/// (or 16-byte aligned when `SIZE` is not a multiple of 32).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn screlu_avx2<const SIZE: usize>(input: &[i32], output: &mut [u8]) {
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

                words0 = _mm256_min_epu16(words0, max_val);
                words1 = _mm256_min_epu16(words1, max_val);

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

            words0 = _mm_min_epu16(words0, max_val);
            words1 = _mm_min_epu16(words1, max_val);

            const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 8 - 16;
            words0 = _mm_srli_epi16(_mm_mulhi_epu16(words0, words0), SHIFT);
            words1 = _mm_srli_epi16(_mm_mulhi_epu16(words1, words1), SHIFT);

            _mm_store_si128(output_ptr.add(i), _mm_packus_epi16(words0, words1));
        }
    }

    let start_idx = SIZE / SSE2_SIMD_WIDTH * SSE2_SIMD_WIDTH;
    screlu_fallback(input, output, start_idx);
}

/// Computes Squared Clipped ReLU using ARM NEON SIMD.
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

/// Computes Squared Clipped ReLU using the scalar fallback.
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

    const ACTIVATION_CEILING: i32 = 255 << HIDDEN_WEIGHT_SCALE_BITS;
    const SQR_SHIFT: u32 = (HIDDEN_WEIGHT_SCALE_BITS * 2 + 8) as u32;

    fn reference_clipped_relu(value: i32) -> u8 {
        (value >> HIDDEN_WEIGHT_SCALE_BITS).clamp(0, 255) as u8
    }

    fn reference_sqr_clipped_relu(value: i32) -> u8 {
        let squared = i64::from(value) * i64::from(value);
        ((squared as u64 >> SQR_SHIFT).min(255)) as u8
    }

    fn reference_screlu(value: i32) -> u8 {
        let clamped = value.clamp(0, ACTIVATION_CEILING) as u64;
        ((clamped * clamped) >> SQR_SHIFT) as u8
    }

    fn patterned_input<const SIZE: usize>(seed: i32, step: i32, bias: i32) -> [i32; SIZE] {
        let mut input = [0; SIZE];
        for (idx, value) in input.iter_mut().enumerate() {
            *value = ((idx as i32 * step + seed).rem_euclid(50_000)) - bias;
        }
        input
    }

    fn assert_clipped_relu_matches_reference<const SIZE: usize>(input: [i32; SIZE]) {
        let input = Align64(input);
        let mut actual = Align64([0xA5; SIZE]);
        let mut expected = [0; SIZE];

        clipped_relu::<SIZE>(input.as_slice(), actual.as_mut_slice());
        for (out, &value) in expected.iter_mut().zip(input.iter()) {
            *out = reference_clipped_relu(value);
        }

        assert_eq!(actual.as_ref(), &expected);
    }

    fn assert_sqr_clipped_relu_matches_reference<const SIZE: usize>(input: [i32; SIZE]) {
        let input = Align64(input);
        let mut actual = Align64([0xA5; SIZE]);
        let mut expected = [0; SIZE];

        sqr_clipped_relu::<SIZE>(input.as_slice(), actual.as_mut_slice());
        for (out, &value) in expected.iter_mut().zip(input.iter()) {
            *out = reference_sqr_clipped_relu(value);
        }

        assert_eq!(actual.as_ref(), &expected);
    }

    fn assert_screlu_matches_reference<const SIZE: usize>(input: [i32; SIZE]) {
        let input = Align64(input);
        let mut actual = Align64([0xA5; SIZE]);
        let mut expected = [0; SIZE];

        screlu::<SIZE>(input.as_slice(), actual.as_mut_slice());
        for (out, &value) in expected.iter_mut().zip(input.iter()) {
            *out = reference_screlu(value);
        }

        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn clipped_relu_matches_shift_and_saturation_boundaries() {
        assert_clipped_relu_matches_reference([
            i32::MIN,
            -65,
            -64,
            -1,
            0,
            1,
            (1 << HIDDEN_WEIGHT_SCALE_BITS) - 1,
            1 << HIDDEN_WEIGHT_SCALE_BITS,
            (2 << HIDDEN_WEIGHT_SCALE_BITS) - 1,
            2 << HIDDEN_WEIGHT_SCALE_BITS,
            (254 << HIDDEN_WEIGHT_SCALE_BITS) + 63,
            255 << HIDDEN_WEIGHT_SCALE_BITS,
            (255 << HIDDEN_WEIGHT_SCALE_BITS) + 1,
            i32::MAX,
            1234,
            -1234,
        ]);
    }

    #[test]
    fn clipped_relu_matches_reference_for_vector_chunks_and_tail() {
        assert_clipped_relu_matches_reference(patterned_input::<32>(7, 7919, 25_000));
        assert_clipped_relu_matches_reference(patterned_input::<37>(17, 3571, 12_000));
    }

    #[test]
    fn sqr_clipped_relu_squares_signed_inputs_and_matches_tail_reference() {
        assert_sqr_clipped_relu_matches_reference([
            -32_768, -16_321, -16_320, -4096, -1024, -1, 0, 1, 1023, 1024, 4096, 16_319, 16_320,
            16_321, 32_767, 257, -257, 8191, -8191, 12_345, -12_345, 2222, -3333,
        ]);
    }

    #[test]
    fn screlu_rectifies_before_squaring_and_saturates_at_255() {
        assert_screlu_matches_reference([
            i32::MIN,
            -1,
            0,
            1,
            (1 << HIDDEN_WEIGHT_SCALE_BITS) - 1,
            1 << HIDDEN_WEIGHT_SCALE_BITS,
            4096,
            8192,
            ACTIVATION_CEILING - 1,
            ACTIVATION_CEILING,
            ACTIVATION_CEILING + 1,
            i32::MAX,
            17_000,
            31,
            63,
            64,
            patterned_input::<1>(123, 1, 0)[0],
        ]);
        assert_screlu_matches_reference(patterned_input::<32>(19, 3011, 18_000));
        assert_screlu_matches_reference(patterned_input::<37>(23, 4099, 20_000));
    }

    #[test]
    fn fused_l1_activation_writes_square_then_clipped_outputs_only() {
        let input_values = [
            -16_321, -16_320, -4096, -1024, -1, 0, 1, 1024, 4096, 8192, 16_319, 16_320, 16_321,
            1234, -5678, 9999,
        ];
        let input = Align64(input_values);
        let mut output = Align64([0xCC; 40]);
        let mut expected_prefix = [0; 32];

        sqr_clipped_and_clipped_relu_16(input.as_slice(), output.as_mut_slice());
        for (out, &value) in expected_prefix[..16].iter_mut().zip(input.iter()) {
            *out = reference_sqr_clipped_relu(value);
        }
        for (out, &value) in expected_prefix[16..].iter_mut().zip(input.iter()) {
            *out = reference_clipped_relu(value);
        }

        assert_eq!(&output.as_ref()[..32], &expected_prefix);
        assert_eq!(&output.as_ref()[32..], &[0xCC; 8]);
    }

    #[test]
    fn fallback_helpers_resume_at_start_index_without_touching_prefix() {
        let input_values = [-2048, -64, -1, 0, 63, 64, 4096, 16_320];
        let input = Align64(input_values);

        let mut clipped = Align64([0xEE; 8]);
        clipped_relu_fallback(input.as_slice(), clipped.as_mut_slice(), 3);
        assert_eq!(&clipped.as_ref()[..3], &[0xEE; 3]);
        for (idx, &value) in input.iter().enumerate().skip(3) {
            assert_eq!(clipped[idx], reference_clipped_relu(value), "clipped {idx}");
        }

        let mut sqr = Align64([0xEE; 8]);
        sqr_clipped_relu_fallback(input.as_slice(), sqr.as_mut_slice(), 3);
        assert_eq!(&sqr.as_ref()[..3], &[0xEE; 3]);
        for (idx, &value) in input.iter().enumerate().skip(3) {
            assert_eq!(sqr[idx], reference_sqr_clipped_relu(value), "sqr {idx}");
        }

        let mut screlu_out = Align64([0xEE; 8]);
        screlu_fallback(input.as_slice(), screlu_out.as_mut_slice(), 3);
        assert_eq!(&screlu_out.as_ref()[..3], &[0xEE; 3]);
        for (idx, &value) in input.iter().enumerate().skip(3) {
            assert_eq!(screlu_out[idx], reference_screlu(value), "screlu {idx}");
        }
    }

    #[test]
    fn sqr_clipped_relu_fallback_matches_hand_computed_values_across_the_i32_range() {
        // The scalar fallback computes `(clamp_i16(x))^2 >> SQR_SHIFT`, capped at
        // 255, widening to i64 so the square cannot overflow. The i16 clamp mirrors
        // the SIMD backends' signed-saturating pack (e.g. AVX2 `_mm_packs_epi32`).
        // Any |x| at or above 16_352 already pins the u8 output to 255, so the
        // clamp never changes the result for out-of-i16 inputs; its job is to keep
        // the square from overflowing the way the previous `x * x` (i32) did for
        // |x| > 46_340. Expected values are computed by hand (SQR_SHIFT = 20, i.e.
        // `(x^2 >> 20).min(255)`) so the test pins the arithmetic instead of just
        // restating the implementation against a same-formula reference.
        let cases: [(i32, u8); 11] = [
            (0, 0),
            (4_096, 16),
            (-4_096, 16),
            (8_192, 64),
            (16_000, 244),
            (16_352, 255), // x^2 >> 20 first reaches the 255 ceiling here
            (32_767, 255), // i16::MAX, still within range
            (46_341, 255), // beyond i16: the old `x * x` overflowed i32 here
            (-100_000, 255),
            (i32::MAX, 255),
            (i32::MIN, 255),
        ];

        let input: [i32; 11] = cases.map(|(value, _)| value);
        let mut actual = [0u8; 11];

        sqr_clipped_relu_fallback(&input, &mut actual, 0);

        for (out, &(value, expected)) in actual.iter().zip(cases.iter()) {
            assert_eq!(*out, expected, "value={value}");
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_activation_kernels_match_references_for_chunks_and_tails() {
        fn run_clipped<const SIZE: usize>(input: [i32; SIZE]) {
            let input = Align64(input);
            let mut actual = Align64([0; SIZE]);
            let mut expected = [0; SIZE];

            unsafe { clipped_relu_neon::<SIZE>(input.as_slice(), actual.as_mut_slice()) };
            for (out, &value) in expected.iter_mut().zip(input.iter()) {
                *out = reference_clipped_relu(value);
            }

            assert_eq!(actual.as_ref(), &expected);
        }

        fn run_sqr<const SIZE: usize>(input: [i32; SIZE]) {
            let input = Align64(input);
            let mut actual = Align64([0; SIZE]);
            let mut expected = [0; SIZE];

            unsafe { sqr_clipped_relu_neon::<SIZE>(input.as_slice(), actual.as_mut_slice()) };
            for (out, &value) in expected.iter_mut().zip(input.iter()) {
                *out = reference_sqr_clipped_relu(value);
            }

            assert_eq!(actual.as_ref(), &expected);
        }

        fn run_screlu<const SIZE: usize>(input: [i32; SIZE]) {
            let input = Align64(input);
            let mut actual = Align64([0; SIZE]);
            let mut expected = [0; SIZE];

            unsafe { screlu_neon::<SIZE>(input.as_slice(), actual.as_mut_slice()) };
            for (out, &value) in expected.iter_mut().zip(input.iter()) {
                *out = reference_screlu(value);
            }

            assert_eq!(actual.as_ref(), &expected);
        }

        run_clipped(patterned_input::<32>(1, 257, 4096));
        run_clipped(patterned_input::<37>(3, 509, 8192));
        run_sqr(patterned_input::<48>(5, 97, 5000));
        run_sqr(patterned_input::<23>(7, 211, 7000));
        run_screlu(patterned_input::<32>(11, 1234, 20_000));
        run_screlu(patterned_input::<37>(13, 1777, 25_000));
    }
}
