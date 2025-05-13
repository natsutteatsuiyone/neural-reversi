//! https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h#L21
use std::io::{self, Read};
use std::mem::size_of;

use aligned::{Aligned, A64};
use aligned_vec::{avec, AVec, ConstAlign};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::CACHE_LINE_SIZE;

#[derive(Debug)]
pub struct BaseInput<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize, const HIDDEN_DIMS: usize> {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize, const HIDDEN_DIMS: usize>
    BaseInput<INPUT_DIMS, OUTPUT_DIMS, HIDDEN_DIMS>
{
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; HIDDEN_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_DIMS * HIDDEN_DIMS];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        for i in 0..HIDDEN_DIMS {
            biases[i] *= 2;
        }

        for i in 0..INPUT_DIMS * HIDDEN_DIMS {
            weights[i] *= 2;
        }

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // permute weights and biases for AVX2
                let num_chunks = (HIDDEN_DIMS * size_of::<i16>()) / size_of::<u64>();
                let bias_slice: &mut [u64] =
                    std::slice::from_raw_parts_mut(biases.as_mut_ptr() as *mut u64, num_chunks);

                for i in 0..num_chunks / 8 {
                    let base = i * 8;
                    bias_slice.swap(base + 2, base + 4);
                    bias_slice.swap(base + 3, base + 5);
                }

                for i in 0..INPUT_DIMS {
                    let ptr = weights.as_mut_ptr().add(i * HIDDEN_DIMS) as *mut u64;
                    let weight_slice: &mut [u64] = std::slice::from_raw_parts_mut(ptr, num_chunks);

                    for j in 0..num_chunks / 8 {
                        let base = j * 8;
                        weight_slice.swap(base + 2, base + 4);
                        weight_slice.swap(base + 3, base + 5);
                    }
                }
            }
        }

        Ok(BaseInput { biases, weights })
    }

    pub fn forward(&self, feature_indices: &[usize], output: &mut [u8]) {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.forward_avx2(feature_indices, output);
            }
            return;
        }

        let mut acc = [0; HIDDEN_DIMS];
        for fi in feature_indices {
            let weights = &self.weights[*fi * HIDDEN_DIMS..][..HIDDEN_DIMS];
            for (a, w) in acc.iter_mut().zip(weights.iter()) {
                *a += *w;
            }
        }

        let half_len = acc.len() / 2;
        let (acc0, acc1) = acc.split_at(half_len);
        let (bias0, bias1) = self.biases.split_at(half_len);
        for i in 0..half_len {
            let sum = (acc0[i] + bias0[i]).clamp(0, 127 * 2) as i32;
            let hs = ((acc1[i] + bias1[i]) >> 2) + 127;
            let hs = hs.clamp(0, 127 * 2) as i32;
            output[i] = ((sum * hs) / 512) as u8;
        }
    }

    unsafe fn forward_avx2(&self, feature_indices: &[usize], output: &mut [u8]) {
        use std::arch::x86_64::*;
        let mut acc: Aligned::<A64, [i16; HIDDEN_DIMS]> = std::mem::zeroed();
        let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
        let num_regs = HIDDEN_DIMS / 16;

        std::ptr::copy_nonoverlapping(
            self.biases.as_ptr() as *const __m256i,
            acc_ptr,
            num_regs,
        );

        let weight_ptr = self.weights.as_ptr();
        let len = feature_indices.len();

        for i in 0..len {
            let idx = *feature_indices.get_unchecked(i);
            let weight_ptr = weight_ptr.add(idx * HIDDEN_DIMS) as *const __m256i;

            for j in 0..num_regs {
                *acc_ptr.add(j) = _mm256_add_epi16(*acc_ptr.add(j), _mm256_loadu_si256(weight_ptr.add(j)));
            }
        }

        let output_ptr = output.as_mut_ptr() as *mut __m256i;
        let one = _mm256_set1_epi16(127 * 2);
        let zero = _mm256_setzero_si256();
        let offset = _mm256_set1_epi16(127); // 63.5 * 2
        let in0_ptr = acc_ptr;
        let in1_ptr = acc_ptr.add(num_regs / 2);
        for j in 0..(num_regs / 4) {
            let in00 = *in0_ptr.add(j * 2);
            let in01 = *in0_ptr.add(j * 2 + 1);
            let sum0 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(in00, one), zero), 7);
            let sum1 = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(in01, one), zero), 7);

            let in10 = *in1_ptr.add(j * 2);
            let in11 = *in1_ptr.add(j * 2 + 1);

            // Hard Sigmoid x * 0.25 + 0.5
            let mut hs0 = _mm256_add_epi16(_mm256_srai_epi16(in10, 2), offset);
            let mut hs1 = _mm256_add_epi16(_mm256_srai_epi16(in11, 2), offset);
            hs0 = _mm256_min_epi16(hs0, one);
            hs1 = _mm256_min_epi16(hs1, one);

            let pa = _mm256_mulhi_epi16(sum0, hs0);
            let pb = _mm256_mulhi_epi16(sum1, hs1);

            *output_ptr.add(j) = _mm256_packus_epi16(pa, pb);
        }
    }
}
