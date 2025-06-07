//! Reference: https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/affine_transform.h
use std::{
    arch::x86_64::*,
    io::{self, Read},
    is_x86_feature_detected,
};

use aligned::{Aligned, A64};
use aligned_vec::{avec, AVec, ConstAlign};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::{eval::CACHE_LINE_SIZE, misc::ceil_to_multiple};

/// Linear transformation layer with AVX2-optimized weight layout.
///
/// # Type Parameters
///
/// * `INPUT_DIMS` - Actual number of input features
/// * `OUTPUT_DIMS` - Actual number of output neurons
/// * `PADDED_INPUT_DIMS` - Input dimensions padded to SIMD width (must be ≥ INPUT_DIMS)
/// * `PADDED_OUTPUT_DIMS` - Output dimensions padded to SIMD width (must be ≥ OUTPUT_DIMS)
#[derive(Debug)]
pub struct LinearLayer<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const PADDED_INPUT_DIMS: usize,
    const PADDED_OUTPUT_DIMS: usize,
> {
    biases: AVec<i32, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i8, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<
        const INPUT_DIMS: usize,
        const OUTPUT_DIMS: usize,
        const PADDED_INPUT_DIMS: usize,
        const PADDED_OUTPUT_DIMS: usize,
    > LinearLayer<INPUT_DIMS, OUTPUT_DIMS, PADDED_INPUT_DIMS, PADDED_OUTPUT_DIMS>
{
    /// Loads weights and biases from binary format.
    ///
    /// Format: biases (i32 LE) followed by weights (i8) in packed layout.
    ///
    /// # Arguments
    /// * `reader` - Input stream to read from
    ///
    /// # Returns
    /// * `Ok(LinearLayer)` on success
    /// * `Err(io::Error)` on failure
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i32; PADDED_OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i8; PADDED_INPUT_DIMS * PADDED_OUTPUT_DIMS];

        for i in 0..OUTPUT_DIMS {
            biases[i] = reader.read_i32::<LittleEndian>()?;
        }

        let num_weights_to_read = PADDED_INPUT_DIMS * OUTPUT_DIMS;
        for i in 0..num_weights_to_read {
            let idx = Self::get_weight_index(i, PADDED_INPUT_DIMS, OUTPUT_DIMS);
            weights[idx] = reader.read_i8()?;
        }

        Ok(LinearLayer { biases, weights })
    }

    /// Converts matrix index to packed format for SIMD efficiency.
    #[inline(always)]
    fn get_weight_index(i: usize, input_size: usize, output_size: usize) -> usize {
        const STRIDE_MULTIPLIER: usize = 4;
        let output_stride = output_size * STRIDE_MULTIPLIER;
        (i / 4) % (input_size / 4) * output_stride + i / input_size * STRIDE_MULTIPLIER + i % 4
    }

    /// Gets packed index for weight at (input_idx, output_idx).
    #[inline(always)]
    fn get_packed_weight_index(&self, input_idx: usize, output_idx: usize) -> usize {
        let conceptual_index = output_idx * PADDED_INPUT_DIMS + input_idx;
        Self::get_weight_index(conceptual_index, PADDED_INPUT_DIMS, OUTPUT_DIMS)
    }

    /// Performs the forward pass of the linear layer.
    ///
    /// # Arguments
    ///
    /// * `input` - Aligned input activations (u8 values)
    /// * `output` - Aligned output buffer for results (i32 accumulators)
    pub fn forward(&self, input: &Aligned<A64, [u8; PADDED_INPUT_DIMS]>, output: &mut Aligned<A64, [i32; PADDED_OUTPUT_DIMS]>) {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.forward_avx2(input, output);
            }
        } else {
            self.forward_fallback(input, output);
        }
    }

    /// AVX2-accelerated forward pass.
    unsafe fn forward_avx2(
        &self,
        input: &Aligned<A64, [u8; PADDED_INPUT_DIMS]>,
        output: &mut Aligned<A64, [i32; PADDED_OUTPUT_DIMS]>,
    ) {
        if OUTPUT_DIMS > 1 {
            let mut acc: Aligned<A64, [i32; OUTPUT_DIMS]> = std::mem::zeroed();
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;

            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = OUTPUT_DIMS / 8;

            std::ptr::copy_nonoverlapping(
                self.biases.as_ptr() as *const __m256i,
                acc_ptr,
                num_regs,
            );

            let input32 = input.as_ptr() as *const i32;
            for i in 0..num_chunks {
                let in0 = _mm256_set1_epi32(*input32.add(i));
                let col0 = self.weights.as_ptr().add(i * OUTPUT_DIMS * 4) as *const __m256i;

                for j in 0..num_regs {
                    let a = acc_ptr.add(j);
                    *a = mm256_dpbusd_epi32(*a, in0, *col0.add(j));
                }
            }

            std::ptr::copy_nonoverlapping(acc_ptr, output.as_ptr() as *mut __m256i, num_regs);
        } else {
            const INPUT_SIMD_WIDTH: usize =
                std::mem::size_of::<__m256i>() / std::mem::size_of::<u8>();

            debug_assert_eq!(INPUT_DIMS % INPUT_SIMD_WIDTH, 0);

            let num_chunks: usize = PADDED_INPUT_DIMS / INPUT_SIMD_WIDTH;
            let mut sum0 = _mm256_setzero_si256();
            let row0 = self.weights.as_ptr() as *const __m256i;
            let input_vector = input.as_ptr() as *const __m256i;

            for j in 0..num_chunks {
                let in_vec = *input_vector.add(j);
                sum0 = mm256_dpbusd_epi32(sum0, in_vec, *row0.add(j));
            }

            output[0] = m256_hadd(sum0, self.biases[0]);
        }
    }

    /// Portable forward pass.
    fn forward_fallback(&self, input: &Aligned<A64, [u8; PADDED_INPUT_DIMS]>, output: &mut Aligned<A64, [i32; PADDED_OUTPUT_DIMS]>) {
        if OUTPUT_DIMS > 1 {
            output[..OUTPUT_DIMS].copy_from_slice(&self.biases[..OUTPUT_DIMS]);

            for (i, &input_byte) in input.iter().take(INPUT_DIMS).enumerate() {
                let input_val = input_byte as i32;
                if input_val == 0 {
                    continue;
                }

                for (k, out) in output.iter_mut().take(OUTPUT_DIMS).enumerate() {
                    let weight_idx = self.get_packed_weight_index(i, k);
                    let weight_val = self.weights[weight_idx] as i32;
                    *out += input_val * weight_val;
                }
            }
        } else {
            let mut acc: i32 = self.biases[0];
            for (i, &input_byte) in input.iter().take(INPUT_DIMS).enumerate() {
                let input_val = input_byte as i32;
                if input_val == 0 {
                    continue;
                }

                let weight_idx = self.get_packed_weight_index(i, 0); // output_idx is 0
                let weight_val = self.weights[weight_idx] as i32;
                acc += input_val * weight_val;
            }
            output[0] = acc;
        }
    }
}

/// Emulates the VPDPBUSD instruction for dot product of unsigned and signed bytes.
///
/// Computes: `acc += sum(a[i] * b[i])` for i in 0..32
/// where `a` contains unsigned bytes and `b` contains signed bytes.
#[inline(always)]
unsafe fn mm256_dpbusd_epi32(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    let product0 = _mm256_maddubs_epi16(a, b);
    let product1 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    _mm256_add_epi32(acc, product1)
}

/// Horizontal sum of all 32-bit integers in a 256-bit vector plus bias.
///
/// Efficiently reduces 8×32-bit values to a single sum using shuffle operations.
#[inline(always)]
unsafe fn m256_hadd(sum_vec: __m256i, bias: i32) -> i32 {
    // Add upper and lower 128-bit lanes
    let mut sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum_vec),
        _mm256_extracti128_si256(sum_vec, 1),
    );
    // Horizontal add within 128-bit lane
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    _mm_cvtsi128_si32(sum128) + bias
}
