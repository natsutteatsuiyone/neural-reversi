//! - [Clipped ReLU](https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/clipped_relu.h)
//! - [Squared Clipped ReLU](https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/sqr_clipped_relu.h)

use std::arch::x86_64::*;

use aligned::{Aligned, A64};

use crate::eval::AVX2_SIMD_WIDTH;

use super::constants::HIDDEN_WEIGHT_SCALE_BITS;

/// Applies a clipped ReLU activation function to `input`.
///
/// This function performs the following operations:
/// 1. Right-shifts values by `HIDDEN_WEIGHT_SCALE_BITS` for scaling
/// 2. Clamps the results to the range `[0, 127]`
/// 3. Stores as 8-bit unsigned integers
///
/// # Arguments
///
/// * `input` - An aligned slice of `SIZE` 32-bit integers representing pre-activation values
/// * `output` - An aligned mutable slice for `SIZE` 8-bit integer results
pub fn clipped_relu<const SIZE: usize>(
    input: &Aligned<A64, [i32; SIZE]>,
    output: &mut Aligned<A64, [u8; SIZE]>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { clipped_relu_avx2::<SIZE>(input, output) };
            return;
        }
    }

    clipped_relu_fallback::<SIZE>(input, output, 0);
}

/// Clipped ReLU with AVX2 SIMD optimization.
///
/// # Arguments
///
/// * `input` - An aligned slice of `SIZE` 32-bit integers
/// * `output` - An aligned mutable slice for `SIZE` 8-bit integer results
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
#[inline]
fn clipped_relu_avx2<const SIZE: usize>(
    input: &Aligned<A64, [i32; SIZE]>,
    output: &mut Aligned<A64, [u8; SIZE]>,
) {
    unsafe {
        if SIZE % AVX2_SIMD_WIDTH == 0 {
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
                    _mm256_permutevar8x32_epi32(_mm256_packs_epi16(words0, words1), shuffle),
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
                    HIDDEN_WEIGHT_SCALE_BITS,
                );

                let words1 = _mm_srli_epi16(
                    _mm_packus_epi32(
                        _mm_load_si128(input_ptr.add(i * 4 + 2)),
                        _mm_load_si128(input_ptr.add(i * 4 + 3)),
                    ),
                    HIDDEN_WEIGHT_SCALE_BITS,
                );

                _mm_store_si128(output_ptr.add(i), _mm_packs_epi16(words0, words1));
            }
        }
    }

    let start = if SIZE % AVX2_SIMD_WIDTH == 0 {
        SIZE / AVX2_SIMD_WIDTH * AVX2_SIMD_WIDTH
    } else {
        SIZE / (AVX2_SIMD_WIDTH / 2) * (AVX2_SIMD_WIDTH / 2)
    };

    clipped_relu_fallback::<SIZE>(input, output, start);
}

/// Clipped ReLU scalar fallback implementation.
///
/// # Arguments
///
/// * `input` - An aligned slice of `SIZE` 32-bit integers
/// * `output` - An aligned mutable slice for `SIZE` 8-bit integer results
/// * `start_idx` - Start index for processing (allows partial processing)
#[inline(always)]
fn clipped_relu_fallback<const SIZE: usize>(
    input: &Aligned<A64, [i32; SIZE]>,
    output: &mut Aligned<A64, [u8; SIZE]>,
    start_idx: usize,
) {
    for i in start_idx..input.len() {
        let val = input[i] >> HIDDEN_WEIGHT_SCALE_BITS;
        output[i] = val.clamp(0, 127) as u8;
    }
}

/// Applies a sqr clipped ReLU activation function to `input`.
///
/// This function performs the following operations:
/// 1. Squares the input values
/// 2. Right-shifts by `(2 * HIDDEN_WEIGHT_SCALE_BITS + 7)` for scaling
/// 3. Clamps the results to the range `[0, 127]`
/// 4. Stores as 8-bit unsigned integers
///
/// # Arguments
///
/// * `input` - An aligned slice of `SIZE` 32-bit integers representing pre-activation values
/// * `output` - An aligned mutable slice for `SIZE` 8-bit integer results
pub fn sqr_clipped_relu<const SIZE: usize>(
    input: &Aligned<A64, [i32; SIZE]>,
    output: &mut Aligned<A64, [u8; SIZE]>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sqr_clipped_relu_avx2::<SIZE>(input, output) };
            return;
        }
    }

    sqr_clipped_relu_fallback::<SIZE>(input, output, 0);
}

/// Sqr clipped ReLU with AVX2 SIMD optimization.
///
/// # Arguments
///
/// * `input` - An aligned slice of `SIZE` 32-bit integers
/// * `output` - An aligned mutable slice for `SIZE` 8-bit integer results
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
#[inline]
fn sqr_clipped_relu_avx2<const SIZE: usize>(
    input: &Aligned<A64, [i32; SIZE]>,
    output: &mut Aligned<A64, [u8; SIZE]>,
) {
    let num_chunks = SIZE / 16;
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

            // We shift by WeightScaleBits * 2 = 12 and divide by 128
            // which is an additional shift-right of 7, meaning 19 in total.
            // MulHi strips the lower 16 bits so we need to shift out 3 more to match.
            const SHIFT: i32 = HIDDEN_WEIGHT_SCALE_BITS * 2 + 7 - 16;
            words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), SHIFT);
            words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), SHIFT);
            _mm_store_si128(output_ptr.add(i), _mm_packs_epi16(words0, words1));
        }
    }

    let start_idx = num_chunks * 16;
    sqr_clipped_relu_fallback::<SIZE>(input, output, start_idx);
}

/// Sqr clipped ReLU scalar fallback implementation.
///
/// # Arguments
///
/// * `input` - An aligned slice of `SIZE` 32-bit integers
/// * `output` - An aligned mutable slice for `SIZE` 8-bit integer results
/// * `start_idx` - Start index for processing (allows partial processing)
#[inline(always)]
fn sqr_clipped_relu_fallback<const SIZE: usize>(
    input: &Aligned<A64, [i32; SIZE]>,
    output: &mut Aligned<A64, [u8; SIZE]>,
    start_idx: usize,
) {
    for i in start_idx..input.len() {
        let val = ((input[i] * input[i]) as u64 >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 7)).min(127);
        output[i] = val as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aligned::{Aligned, A64};

    #[test]
    fn test_clipped_relu_fallback() {
        const SIZE: usize = 6;

        let input_data = [100, -50, 200, i32::MAX, i32::MIN, 0];
        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);

        clipped_relu_fallback::<SIZE>(&input, &mut output, 0);

        let expected = [1, 0, 3, 127, 0, 0];
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
        expected[3] = 127;

        input_data[4] = i32::MIN;
        expected[4] = 0;

        input_data[5] = 0;
        expected[5] = 0;

        input_data[6] = 127 << 6;
        expected[6] = 127;

        input_data[7] = (127 << 6) + 1;
        expected[7] = 127;

        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output = Aligned::<A64, [u8; SIZE]>::default();
        clipped_relu::<SIZE>(&input, &mut output);
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_sqr_clipped_relu_fallback() {
        const SIZE: usize = 6;

        let input_data = [10, -5, 20, 5000, -5000, 127 << HIDDEN_WEIGHT_SCALE_BITS];
        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);

        sqr_clipped_relu_fallback::<SIZE>(&input, &mut output, 0);

        let expected = [0, 0, 0, 47, 47, 126];
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
        expected[2] = 47;

        input_data[3] = -5000;
        expected[3] = 47;

        input_data[4] = 1000;
        expected[4] = 1;

        input_data[5] = -1000;
        expected[5] = 1;

        input_data[6] = 0;
        expected[6] = 0;

        input_data[7] = 127 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[7] = 126;

        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
        sqr_clipped_relu::<SIZE>(&input, &mut output);
        assert_eq!(output.as_ref(), &expected);
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
        input_data[2] = 127 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[2] = 127;

        input_data[3] = 128 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[3] = 127;  // Should be clipped to 127

        // Test values just below the boundary
        input_data[4] = (127 << HIDDEN_WEIGHT_SCALE_BITS) - 1;
        expected[4] = 126;

        // Test very large positive values
        input_data[5] = 1000000;
        expected[5] = 127;  // Should be clipped to 127

        // Test scaling edge cases
        input_data[6] = (1 << HIDDEN_WEIGHT_SCALE_BITS) - 1;
        expected[6] = 0;  // Just below 1 after shifting

        input_data[7] = 1 << HIDDEN_WEIGHT_SCALE_BITS;
        expected[7] = 1;  // Exactly 1 after shifting

        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
        clipped_relu::<SIZE>(&input, &mut output);
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
        // sqrt(127 * 2^19) ≈ 8128
        input_data[2] = 8128;
        expected[2] = 126;

        input_data[3] = -8128;
        expected[3] = 126;

        // Test values that will saturate to 127
        input_data[4] = 10000;
        expected[4] = 127;

        input_data[5] = -10000;
        expected[5] = 127;

        // Test boundary for producing 1
        // sqrt(1 * 2^19) ≈ 724, but calculation differs
        input_data[6] = 724;
        expected[6] = 0;  // Actually produces 0 with this implementation

        input_data[7] = -724;
        expected[7] = 0;  // Actually produces 0 with this implementation

        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
        sqr_clipped_relu::<SIZE>(&input, &mut output);
        assert_eq!(output.as_ref(), &expected);
    }

    #[test]
    fn test_clipped_relu_different_sizes() {
        // Test size 8 (smaller than AVX2 width)
        {
            const SIZE: usize = 8;
            let input_data = [100, -50, 8192, 16384, -8192, 0, 127 << 6, 128 << 6];
            let expected = [1, 0, 127, 127, 0, 0, 127, 127];

            let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
            let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
            clipped_relu::<SIZE>(&input, &mut output);
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
                expected[i] = val.clamp(0, 127) as u8;
            }

            let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
            let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
            clipped_relu::<SIZE>(&input, &mut output);
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
                expected[i] = val.clamp(0, 127) as u8;
            }

            let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
            let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
            clipped_relu::<SIZE>(&input, &mut output);
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
                let val = ((input_data[i] as i64 * input_data[i] as i64) >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 7)).min(127);
                expected[i] = val as u8;
            }

            let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
            let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
            sqr_clipped_relu::<SIZE>(&input, &mut output);
            assert_eq!(output.as_ref(), &expected);
        }

        // Test size 48 (multiple chunks)
        {
            const SIZE: usize = 48;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32 - 24) * 300;
                let val = ((input_data[i] as i64 * input_data[i] as i64) >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 7)).min(127);
                expected[i] = val as u8;
            }

            let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
            let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
            sqr_clipped_relu::<SIZE>(&input, &mut output);
            assert_eq!(output.as_ref(), &expected);
        }

        // Test odd size
        {
            const SIZE: usize = 23;
            let mut input_data = [0i32; SIZE];
            let mut expected = [0u8; SIZE];

            for i in 0..SIZE {
                input_data[i] = (i as i32) * 400 - 4000;
                let val = ((input_data[i] as i64 * input_data[i] as i64) >> (2 * HIDDEN_WEIGHT_SCALE_BITS + 7)).min(127);
                expected[i] = val as u8;
            }

            let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
            let mut output: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
            sqr_clipped_relu::<SIZE>(&input, &mut output);
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

        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output_main: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
        let mut output_fallback: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);

        clipped_relu::<SIZE>(&input, &mut output_main);
        clipped_relu_fallback::<SIZE>(&input, &mut output_fallback, 0);

        assert_eq!(output_main.as_ref(), output_fallback.as_ref());
    }

    #[test]
    fn test_sqr_clipped_relu_consistency() {
        // Test that AVX2 and fallback produce same results
        const SIZE: usize = 80;
        let mut input_data = [0i32; SIZE];

        for (i, val) in input_data.iter_mut().enumerate() {
            *val = ((i * 9876 + 1000) % 10000) as i32 - 3000;
        }

        let input: Aligned<A64, [i32; SIZE]> = Aligned(input_data);
        let mut output_main: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);
        let mut output_fallback: Aligned<A64, [u8; SIZE]> = Aligned([0; SIZE]);

        sqr_clipped_relu::<SIZE>(&input, &mut output_main);
        sqr_clipped_relu_fallback::<SIZE>(&input, &mut output_fallback, 0);

        assert_eq!(output_main.as_ref(), output_fallback.as_ref());
    }
}
