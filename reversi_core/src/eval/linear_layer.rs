//! https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/affine_transform.h
use std::{
    arch::x86_64::*,
    io::{self, Read},
    is_x86_feature_detected,
};

use aligned_vec::{avec, AVec, ConstAlign};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::{eval::CACHE_LINE_SIZE, misc::ceil_to_multiple};

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

    #[inline(always)]
    fn get_weight_index(i: usize, input_size: usize, output_size: usize) -> usize {
        const STRIDE_MULTIPLIER: usize = 4;
        let output_stride = output_size * STRIDE_MULTIPLIER;
        (i / 4) % (input_size / 4) * output_stride + i / input_size * STRIDE_MULTIPLIER + i % 4
    }

    #[inline(always)]
    fn get_packed_weight_index(&self, input_idx: usize, output_idx: usize) -> usize {
        let conceptual_index = output_idx * PADDED_INPUT_DIMS + input_idx;
        Self::get_weight_index(conceptual_index, PADDED_INPUT_DIMS, OUTPUT_DIMS)
    }

    pub fn forward(&self, input: &[u8], output: &mut [i32]) {
        debug_assert!(input.len() >= INPUT_DIMS, "Input slice too short");
        debug_assert!(output.len() >= OUTPUT_DIMS, "Output slice too short");

        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.forward_avx2(input, output);
            }
        } else {
            self.forward_fallback(input, output);
        }
    }

    unsafe fn forward_avx2(&self, input: &[u8], output: &mut [i32]) {
        if OUTPUT_DIMS > 1 {
            let mut acc: [i32; OUTPUT_DIMS] = std::mem::zeroed();
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

            std::ptr::copy_nonoverlapping(
                acc_ptr,
                output.as_ptr() as *mut __m256i,
                num_regs,
            );
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

    fn forward_fallback(&self, input: &[u8], output: &mut [i32]) {
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
                let weight_idx = self.get_packed_weight_index(i, 0); // output_idx is 0
                let weight_val = self.weights[weight_idx] as i32;
                acc += input_val * weight_val;
            }
            output[0] = acc;
        }
    }
}

#[inline(always)]
unsafe fn mm256_dpbusd_epi32(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    let product0 = _mm256_maddubs_epi16(a, b);
    let product1 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    _mm256_add_epi32(acc, product1)
}

#[inline(always)]
unsafe fn m256_hadd(sum_vec: __m256i, bias: i32) -> i32 {
    let mut sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum_vec),
        _mm256_extracti128_si256(sum_vec, 1),
    );
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01_00_11_10));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b10_11_00_01));
    _mm_cvtsi128_si32(sum128) + bias
}
