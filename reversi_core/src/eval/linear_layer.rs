//! https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/affine_transform.h
use std::{
    arch::x86_64::*,
    io::{self, Read},
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
    const NUM_REGS: usize,
> {
    biases: AVec<i32, ConstAlign<CACHE_LINE_SIZE>>,
    pub weights: AVec<i8, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<
        const INPUT_DIMS: usize,
        const OUTPUT_DIMS: usize,
        const PADDED_INPUT_DIMS: usize,
        const PADDED_OUTPUT_DIMS: usize,
        const NUM_REGS: usize,
    > LinearLayer<INPUT_DIMS, OUTPUT_DIMS, PADDED_INPUT_DIMS, PADDED_OUTPUT_DIMS, NUM_REGS>
{
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i32; OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i8; PADDED_INPUT_DIMS * PADDED_OUTPUT_DIMS];

        for i in 0..OUTPUT_DIMS {
            biases[i] = reader.read_i32::<LittleEndian>()?;
        }

        for i in 0..(PADDED_INPUT_DIMS * OUTPUT_DIMS) {
            let idx = Self::get_weight_index(i, PADDED_INPUT_DIMS, OUTPUT_DIMS);
            weights[idx] = reader.read_i8()?;
        }

        Ok(LinearLayer { biases, weights })
    }

    fn get_weight_index(i: usize, input_size: usize, output_size: usize) -> usize {
        (i / 4) % (input_size / 4) * output_size * 4 + i / input_size * 4 + i % 4
    }

    pub fn forward(&self, input: &[u8], output: &mut [i32]) {
        self.forward_avx2(input, output)
    }

    fn forward_avx2(&self, input: &[u8], output: &mut [i32]) {
        unsafe {
            use std::arch::x86_64::*;

            if OUTPUT_DIMS > 1 {
                let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;

                let input32 = input.as_ptr() as *const i32;
                let biasvec = self.biases.as_ptr() as *const __m256i;
                let mut acc: [__m256i; NUM_REGS] = std::mem::zeroed();
                for (k, acc_item) in acc.iter_mut().enumerate() {
                    *acc_item = *biasvec.add(k);
                }

                for i in 0..num_chunks {
                    let in0 = _mm256_set1_epi32(*input32.add(i));
                    let col0 = self.weights.as_ptr().add(i * OUTPUT_DIMS * 4) as *const __m256i;

                    for (k, acc_item) in acc.iter_mut().enumerate() {
                        *acc_item = mm256_dpbusd_epi32(*acc_item, in0, *col0.add(k));
                    }
                }

                let outptr = output.as_mut_ptr() as *mut __m256i;
                for (k, acc_item) in acc.iter().enumerate() {
                    _mm256_store_si256(outptr.add(k), *acc_item);
                }
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
    }
}

#[inline(always)]
unsafe fn mm256_dpbusd_epi32(acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    // if is_x86_feature_detected!("avx512vnni") {
    //     return core::arch::x86_64::_mm256_dpbusd_epi32(acc, a, b);
    // }

    let mut product0 = _mm256_maddubs_epi16(a, b);
    product0 = _mm256_madd_epi16(product0, _mm256_set1_epi16(1));
    _mm256_add_epi32(acc, product0)
}

#[inline(always)]
unsafe fn m256_hadd(sum: __m256i, bias: i32) -> i32 {
    let mut sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum),
        _mm256_extracti128_si256(sum, 1),
    );
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0x4E));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0xB1));
    _mm_cvtsi128_si32(sum128) + bias
}
