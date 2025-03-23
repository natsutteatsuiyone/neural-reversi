//! https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h#L21
use std::io::{self, Read};
use std::mem::size_of;

use aligned_vec::{avec, AVec, ConstAlign};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::CACHE_LINE_SIZE;

use super::constants::{UNIV_INPUT_OUTPUT_DIMS, INPUT_FEATURE_DIMS};

const OUTPUT_SIZE: usize = UNIV_INPUT_OUTPUT_DIMS;
const HIDDEN_LAYER_SIZE: usize = OUTPUT_SIZE * 2;
const NUM_REGS: usize = HIDDEN_LAYER_SIZE / 16;

#[derive(Debug)]
pub struct UniversalInput {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl UniversalInput
{
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; HIDDEN_LAYER_SIZE];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_FEATURE_DIMS * HIDDEN_LAYER_SIZE];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        for i in 0..HIDDEN_LAYER_SIZE {
            biases[i] *= 2;
        }

        for i in 0..INPUT_FEATURE_DIMS * HIDDEN_LAYER_SIZE {
            weights[i] *= 2;
        }

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // permute weights and biases for AVX2
                let num_chunks = (HIDDEN_LAYER_SIZE * size_of::<i16>()) / size_of::<u64>();
                let bias_slice: &mut [u64] =
                    std::slice::from_raw_parts_mut(biases.as_mut_ptr() as *mut u64, num_chunks);

                for i in 0..num_chunks/8 {
                    let base = i * 8;
                    bias_slice.swap(base + 2, base + 4);
                    bias_slice.swap(base + 3, base + 5);
                }

                for i in 0..INPUT_FEATURE_DIMS {
                    let ptr = weights.as_mut_ptr().add(i * HIDDEN_LAYER_SIZE) as *mut u64;
                    let weight_slice: &mut [u64] = std::slice::from_raw_parts_mut(ptr, num_chunks);

                    for j in 0..num_chunks/8 {
                        let base = j * 8;
                        weight_slice.swap(base + 2, base + 4);
                        weight_slice.swap(base + 3, base + 5);
                    }
                }
            }
        }

        Ok(UniversalInput { biases, weights })
    }

    pub fn forward(&self, feature_indices: &[u16], output: &mut [u8]) {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.forward_avx2(feature_indices, output);
            }
            return;
        }

        let mut acc = [0; HIDDEN_LAYER_SIZE];
        for fi in feature_indices {
            let weights = &self.weights[(*fi as usize) * HIDDEN_LAYER_SIZE..][..HIDDEN_LAYER_SIZE];
            for (a, w) in acc.iter_mut().zip(weights.iter()) {
                *a += *w;
            }
        }

        for i in 0..OUTPUT_SIZE {
            let sum0 = acc[i] + self.biases[i];
            let sum1 = acc[i + OUTPUT_SIZE] + self.biases[i + OUTPUT_SIZE];
            let sum0 = sum0.clamp(0, 127 * 2) as u32;
            let sum1 = sum1.clamp(0, 127 * 2) as u32;
            output[i] = ((sum0 * sum1) / 512) as u8;
        }
    }

    unsafe fn forward_avx2(&self, feature_indices: &[u16], output: &mut [u8]) {
        use std::arch::x86_64::*;
        let mut acc: [__m256i; NUM_REGS] = std::mem::zeroed();

        // バイアスを一度にロード
        std::ptr::copy_nonoverlapping(
            self.biases.as_ptr() as *const __m256i,
            acc.as_mut_ptr(),
            NUM_REGS,
        );

        let weight_ptr = self.weights.as_ptr();
        let len = feature_indices.len();

        for i in 0..len {
            let idx = *feature_indices.get_unchecked(i) as usize;
            let weight_ptr = weight_ptr.add(idx * HIDDEN_LAYER_SIZE) as *const __m256i;

            for j in 0..NUM_REGS {
                *acc.get_unchecked_mut(j) = _mm256_add_epi16(
                    *acc.get_unchecked(j),
                    _mm256_loadu_si256(weight_ptr.add(j))
                );
            }
        }

        let output_ptr = output.as_mut_ptr() as *mut __m256i;
        let one = _mm256_set1_epi16(127 * 2);
        let zero = _mm256_setzero_si256();
        let in0 = &acc[0..acc.len() / 2];
        let in1 = &acc[(acc.len() / 2)..];
        for j in 0..(acc.len() / 4) {
            let sum0a = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(in0[j * 2], one), zero), 7);
            let sum0b = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(in0[j * 2 + 1], one), zero), 7);
            let sum1a = _mm256_min_epi16(in1[j * 2], one);
            let sum1b = _mm256_min_epi16(in1[j * 2 + 1], one);
            let pa = _mm256_mulhi_epi16(sum0a, sum1a);
            let pb = _mm256_mulhi_epi16(sum0b, sum1b);
            *output_ptr.add(j) = _mm256_packus_epi16(pa, pb);
        }
    }
}
