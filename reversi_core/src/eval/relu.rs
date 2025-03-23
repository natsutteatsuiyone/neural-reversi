//! https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/clipped_relu.h
//! https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/sqr_clipped_relu.h
use std::arch::x86_64::*;

use crate::eval::AVX2_SIMD_WIDTH;

/// Applies a clipped ReLU activation function to a slice of 32-bit integers,
/// writing the results as 8-bit integers to the output slice.
///
/// The clipped ReLU function returns the input value if it is positive, and zero otherwise.
/// The result is then clamped to the maximum value of an 8-bit signed integer (`i8::MAX`).
///
/// # Arguments
///
/// * `input` - A slice containing the 32-bit integer input values.
/// * `output` - A mutable slice where the 8-bit integer results will be written.
///
/// # Safety
///
/// This function has the following safety requirements:
///
/// * The `input` and `output` slices must have the same length.
/// * Both `input` and `output` slices must be aligned to 32-byte boundaries. This is crucial for the AVX2 implementation to function correctly.
/// * The lengths of both `input` and `output` slices must be multiples of 32. This ensures that the vectorized operations process complete chunks of data.
pub fn clipped_relu<const WEIGHT_SCALE_BITS: i32>(input: &[i32], output: &mut [u8]) {
    debug_assert!(input.len() == output.len());

    if is_x86_feature_detected!("avx2") {
        unsafe { return clipped_relu_avx2::<WEIGHT_SCALE_BITS>(input, output) }
    }

    clipped_relu_fallback::<WEIGHT_SCALE_BITS>(input, output, 0);
}

/// Applies a clipped ReLU activation function to a slice of 32-bit integers using AVX2 intrinsics,
/// writing the results as 8-bit integers to the output slice.
///
/// This is an optimized version of `clipped_relu` that leverages AVX2 vector instructions for increased performance.
///
/// # Arguments
///
/// * `input` - A slice containing the 32-bit integer input values.
/// * `output` - A mutable slice where the 8-bit integer results will be written.
unsafe fn clipped_relu_avx2<const WEIGHT_SCALE_BITS: i32>(input: &[i32], output: &mut [u8]) {
    if input.len() % AVX2_SIMD_WIDTH == 0 {
        let num_chunks = input.len() / AVX2_SIMD_WIDTH;
        let offsets: __m256i = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
        let input_ptr = input.as_ptr() as *const __m256i;
        let output_ptr = output.as_mut_ptr() as *mut __m256i;
        for i in 0..num_chunks {
            let words0 = _mm256_srli_epi16(
                _mm256_packus_epi32(
                    _mm256_load_si256(input_ptr.add(i * 4)),
                    _mm256_load_si256(input_ptr.add(i * 4 + 1)),
                ),
                WEIGHT_SCALE_BITS,
            );
            let words1 = _mm256_srli_epi16(
                _mm256_packus_epi32(
                    _mm256_load_si256(input_ptr.add(i * 4 + 2)),
                    _mm256_load_si256(input_ptr.add(i * 4 + 3)),
                ),
                WEIGHT_SCALE_BITS,
            );
            _mm256_store_si256(
                output_ptr.add(i),
                _mm256_permutevar8x32_epi32(_mm256_packs_epi16(words0, words1), offsets),
            );
        }
    } else {
        let num_chunks = input.len() / (AVX2_SIMD_WIDTH / 2);
        let input_ptr = input.as_ptr() as *const __m128i;
        let output_ptr = output.as_mut_ptr() as *mut __m128i;
        for i in 0..num_chunks {
            let words0 = _mm_srli_epi16(
                _mm_packus_epi32(
                    _mm_load_si128(input_ptr.add(i * 4)),
                    _mm_load_si128(input_ptr.add(i * 4 + 1)),
                ),
                WEIGHT_SCALE_BITS,
            );

            let words1 = _mm_srli_epi16(
                _mm_packus_epi32(
                    _mm_load_si128(input_ptr.add(i * 4 + 2)),
                    _mm_load_si128(input_ptr.add(i * 4 + 3)),
                ),
                WEIGHT_SCALE_BITS,
            );

            _mm_store_si128(output_ptr.add(i), _mm_packs_epi16(words0, words1));
        }
    }

    let start = if input.len() % AVX2_SIMD_WIDTH == 0 {
        input.len() / AVX2_SIMD_WIDTH * AVX2_SIMD_WIDTH
    } else {
        input.len() / (AVX2_SIMD_WIDTH / 2) * (AVX2_SIMD_WIDTH / 2)
    };

    clipped_relu_fallback::<WEIGHT_SCALE_BITS>(input, output, start);
}

/// Applies a clipped ReLU activation function to a slice of 32-bit integers using a scalar fallback implementation,
///
/// # Arguments
///
/// * `input` - A slice containing the 32-bit integer input values.
/// * `output` - A mutable slice where the 8-bit integer results will be written.
/// * `start_idx` - The index at which to start processing the input slice.
fn clipped_relu_fallback<const WEIGHT_SCALE_BITS: i32>(
    input: &[i32],
    output: &mut [u8],
    start_idx: usize,
) {
    for i in start_idx..input.len() {
        let val = input[i] >> WEIGHT_SCALE_BITS;
        output[i] = val.clamp(0, 127) as u8;
    }
}

/// Applies a squared clipped ReLU activation function to a slice of 32-bit integers,
///
/// # Arguments
///
/// * `input` - A slice containing the 32-bit integer input values.
/// * `output` - A mutable slice where the 8-bit integer results will be written.
pub fn sqr_clipped_relu<const WEIGHT_SCALE_BITS: i32>(input: &[i32], output: &mut [u8]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { return sqr_clipped_relu_avx2::<WEIGHT_SCALE_BITS>(input, output) }
    }

    sqr_clipped_relu_fallback::<WEIGHT_SCALE_BITS>(input, output, 0);
}

/// Applies a squared clipped ReLU activation function to a slice of 32-bit integers using AVX2 intrinsics,
///
/// # Arguments
///
/// * `input` - A slice containing the 32-bit integer input values.
/// * `output` - A mutable slice where the 8-bit integer results will be written.
unsafe fn sqr_clipped_relu_avx2<const WEIGHT_SCALE_BITS: i32>(input: &[i32], output: &mut [u8]) {
    let num_chunks = input.len() / 16;
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

        // We shift by WeightScaleBits * 2 = 12 and divide by 128
        // which is an additional shift-right of 7, meaning 19 in total.
        // MulHi strips the lower 16 bits so we need to shift out 3 more to match.
        words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
        words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);
        _mm_store_si128(output_ptr.add(i), _mm_packs_epi16(words0, words1));
    }

    let start_idx = num_chunks * 16;
    sqr_clipped_relu_fallback::<WEIGHT_SCALE_BITS>(input, output, start_idx);
}

/// Applies a squared clipped ReLU activation function to a slice of 32-bit integers using a scalar fallback implementation,
///
/// # Arguments
///
/// * `input` - A slice containing the 32-bit integer input values.
/// * `output` - A mutable slice where the 8-bit integer results will be written.
/// * `start_idx` - The index at which to start processing the input slice.
#[inline(always)]
fn sqr_clipped_relu_fallback<const WEIGHT_SCALE_BITS: i32>(
    input: &[i32],
    output: &mut [u8],
    start_idx: usize,
) {
    for i in start_idx..input.len() {
        let val = ((input[i] * input[i]) as u64 >> (2 * WEIGHT_SCALE_BITS + 7)).min(127);
        output[i] = val as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::CACHE_LINE_SIZE;
    use aligned_vec::{avec, AVec, ConstAlign};

    #[test]
    fn test_clipped_relu_fallback() {
        let input: AVec<i32, ConstAlign<CACHE_LINE_SIZE>> =
            avec![[CACHE_LINE_SIZE] | 100, -50, 200, i32::MAX, i32::MIN, 0];
        let mut output = avec![[CACHE_LINE_SIZE]|0; input.len()];

        clipped_relu_fallback::<2>(&input, &mut output, 0);

        let expected = avec![[CACHE_LINE_SIZE] | 25, 0, 50, 127, 0, 0];
        assert_eq!(output, expected);
    }
}
