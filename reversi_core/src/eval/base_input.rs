//! Neural Network Base Input Layer
//!
//! Reference: https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h
use std::io::{self, Read};
use std::mem::size_of;

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::CACHE_LINE_SIZE;
use crate::util::align::Align64;

/// Neural network base input layer
///
/// # Type Parameters
/// - `INPUT_DIMS`: Number of input features (sparse)
/// - `OUTPUT_DIMS`: Number of output dimensions (dense)
/// - `HIDDEN_DIMS`: Number of hidden units (must be 2 * OUTPUT_DIMS)
#[derive(Debug)]
pub struct BaseInput<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize, const HIDDEN_DIMS: usize> {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize, const HIDDEN_DIMS: usize>
    BaseInput<INPUT_DIMS, OUTPUT_DIMS, HIDDEN_DIMS>
{
    /// Loads network weights and biases from a binary reader.
    ///
    /// # Arguments
    /// * `reader` - Binary data reader containing network parameters
    ///
    /// # Returns
    /// * `Ok(BaseInput)` - Successfully loaded network layer
    /// * `Err(io::Error)` - I/O error during loading
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    // Permute weights and biases for optimal AVX2 access patterns.
                    // This reorders data to match the SIMD lane requirements.
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
                        let weight_slice: &mut [u64] =
                            std::slice::from_raw_parts_mut(ptr, num_chunks);

                        for j in 0..num_chunks / 8 {
                            let base = j * 8;
                            weight_slice.swap(base + 2, base + 4);
                            weight_slice.swap(base + 3, base + 5);
                        }
                    }
                }
            }
        }

        Ok(BaseInput { biases, weights })
    }

    /// Performs forward pass through the base input layer.
    ///
    /// # Arguments
    /// * `feature_indices` - Sparse indices of active features to accumulate
    /// * `output` - Output buffer to write results (length must be OUTPUT_DIMS)
    pub fn forward(&self, feature_indices: &[usize], output: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.forward_avx2(feature_indices, output) };
                return;
            }
        }

        self.forward_fallback(feature_indices, output)
    }

    /// AVX2-optimized forward pass implementation.
    #[target_feature(enable = "avx2")]
    #[cfg(target_arch = "x86_64")]
    unsafe fn forward_avx2(&self, feature_indices: &[usize], output: &mut [u8]) {
        const {
            assert!(HIDDEN_DIMS.is_multiple_of(64), "HIDDEN_DIMS must be a multiple of 64");
            assert!(OUTPUT_DIMS * 2 == HIDDEN_DIMS, "OUTPUT_DIMS must be half of HIDDEN_DIMS");
        }

        unsafe {
            use std::arch::x86_64::*;

            let mut acc: Align64<[i16; HIDDEN_DIMS]> = std::mem::zeroed();
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
            let num_regs = HIDDEN_DIMS / 16;

            // Initialize accumulator with bias values
            std::ptr::copy_nonoverlapping(
                self.biases.as_ptr() as *const __m256i,
                acc_ptr,
                num_regs,
            );

            let weight_ptr = self.weights.as_ptr();
            let len = feature_indices.len();

            // Accumulate weights for each active feature
            for i in 0..len {
                let idx = *feature_indices.get_unchecked(i);
                let weight_ptr = weight_ptr.add(idx * HIDDEN_DIMS) as *const __m256i;

                // Add weights for this feature to accumulator using SIMD
                for j in 0..num_regs {
                    let weight = _mm256_loadu_si256(weight_ptr.add(j));
                    *acc_ptr.add(j) = _mm256_add_epi16(*acc_ptr.add(j), weight);
                }
            }

            let output_ptr = output.as_mut_ptr() as *mut __m256i;
            let one = _mm256_set1_epi16(127 * 2);
            let zero = _mm256_setzero_si256();
            let in0_ptr = acc_ptr;
            let in1_ptr = acc_ptr.add(num_regs / 2);
            for j in 0..(num_regs / 4) {
                let in00 = *in0_ptr.add(j * 2);
                let in01 = *in0_ptr.add(j * 2 + 1);
                let in10 = *in1_ptr.add(j * 2);
                let in11 = *in1_ptr.add(j * 2 + 1);
                let sum0a = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(in00, one), zero), 7);
                let sum0b = _mm256_slli_epi16(_mm256_max_epi16(_mm256_min_epi16(in01, one), zero), 7);
                let sum1a = _mm256_min_epi16(in10, one);
                let sum1b = _mm256_min_epi16(in11, one);
                let pa = _mm256_mulhi_epi16(sum0a, sum1a);
                let pb = _mm256_mulhi_epi16(sum0b, sum1b);
                *output_ptr.add(j) = _mm256_packus_epi16(pa, pb);
            }
        }
    }

    /// Fallback scalar implementation.
    fn forward_fallback(&self, feature_indices: &[usize], output: &mut [u8]) {
        let mut acc = [0; HIDDEN_DIMS];

        // Accumulate weights for all active features
        for fi in feature_indices {
            let weights = &self.weights[*fi * HIDDEN_DIMS..][..HIDDEN_DIMS];
            for (a, w) in acc.iter_mut().zip(weights.iter()) {
                *a += *w;
            }
        }

        for i in 0..OUTPUT_DIMS {
            let sum0 = acc[i] + self.biases[i];
            let sum1 = acc[i + OUTPUT_DIMS] + self.biases[i + OUTPUT_DIMS];
            let sum0 = sum0.clamp(0, 127 * 2) as u32;
            let sum1 = sum1.clamp(0, 127 * 2) as u32;
            output[i] = ((sum0 * sum1) / 512) as u8;
        }
    }
}
