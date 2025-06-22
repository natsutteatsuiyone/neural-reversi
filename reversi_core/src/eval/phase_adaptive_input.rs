//! Phase-adaptive input layer implementation.

use std::io::{self, Read};
use std::mem::size_of;

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::CACHE_LINE_SIZE;
use crate::util::align::Align64;

/// Phase-adaptive input layer.
///
/// # Type Parameters
///
/// - `INPUT_DIMS`: Number of input features (sparse)
/// - `OUTPUT_DIMS`: Number of output dimensions (dense)
#[derive(Debug)]
pub struct PhaseAdaptiveInput<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize> {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize>
    PhaseAdaptiveInput<INPUT_DIMS, OUTPUT_DIMS>
{
    /// Loads network weights and biases from a binary reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Input stream containing serialized network parameters
    ///
    /// # Returns
    ///
    /// * `Ok(PhaseAdaptiveInput)` - Successfully loaded network layer
    /// * `Err(io::Error)` - If reading from the stream fails
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_DIMS * OUTPUT_DIMS];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    // Permute weights and biases for AVX2 SIMD optimization.
                    // This permutation rearranges data to match the SIMD access pattern,
                    // avoiding the need for permutation during inference.
                    let num_chunks = (OUTPUT_DIMS * size_of::<i16>()) / size_of::<u64>();
                    let bias_slice: &mut [u64] =
                        std::slice::from_raw_parts_mut(biases.as_mut_ptr() as *mut u64, num_chunks);

                    for i in 0..num_chunks / 8 {
                        let base = i * 8;
                        bias_slice.swap(base + 2, base + 4);
                        bias_slice.swap(base + 3, base + 5);
                    }

                    for i in 0..INPUT_DIMS {
                        let ptr = weights.as_mut_ptr().add(i * OUTPUT_DIMS) as *mut u64;
                        let weight_slice: &mut [u64] = std::slice::from_raw_parts_mut(ptr, num_chunks);

                        for j in 0..num_chunks / 8 {
                            let base = j * 8;
                            weight_slice.swap(base + 2, base + 4);
                            weight_slice.swap(base + 3, base + 5);
                        }
                    }
                }
            }
        }

        Ok(PhaseAdaptiveInput { biases, weights })
    }

    /// Performs forward pass through the phase-adaptive input layer.
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

        self.forward_fallback(feature_indices, output);
    }

    /// AVX2-optimized forward pass implementation.
    #[target_feature(enable = "avx2")]
    #[cfg(target_arch = "x86_64")]
    unsafe fn forward_avx2(&self, feature_indices: &[usize], output: &mut [u8]) {
        const {
            assert!(OUTPUT_DIMS % 32 == 0, "HIDDEN_DIMS must be a multiple of 64");
        }

        unsafe {
            use std::arch::x86_64::*;
            let mut acc: Align64<[i16; OUTPUT_DIMS]> = std::mem::zeroed();
            let mut acc_ptr = acc.as_mut_ptr() as *mut __m256i;
            let num_regs = OUTPUT_DIMS / 16;

            std::ptr::copy_nonoverlapping(
                self.biases.as_ptr() as *const __m256i,
                acc_ptr,
                num_regs,
            );

            let weight_ptr = self.weights.as_ptr();
            let len = feature_indices.len();

            for i in 0..len {
                let idx = *feature_indices.get_unchecked(i);
                let weight_ptr = weight_ptr.add(idx * OUTPUT_DIMS) as *const __m256i;

                for j in 0..num_regs {
                    *acc_ptr.add(j) = _mm256_add_epi16(*acc_ptr.add(j), _mm256_loadu_si256(weight_ptr.add(j)));
                }
            }

            let mut output_ptr = output.as_mut_ptr() as *mut __m256i;
            let bias8 = _mm256_set1_epi8(16);
            let zero8 = _mm256_set1_epi8(0);

            let iterations = num_regs / 2;
            for _ in 0..iterations {
                let a = _mm256_load_si256(acc_ptr);
                let b = _mm256_load_si256(acc_ptr.add(1));
                acc_ptr = acc_ptr.add(2);

                // Apply LeakyReLU activation: result = max(x, x >> 3)
                // This preserves positive values while attenuating negative values by 1/8
                let sa = _mm256_max_epi16(a, _mm256_srai_epi16(a, 3));
                let sb = _mm256_max_epi16(b, _mm256_srai_epi16(b, 3));

                // Pack 16-bit values to 8-bit with saturation.
                // Values > 127 are clamped to 127, values < -128 are clamped to -128.
                let packed = _mm256_packs_epi16(sa, sb);

                // Add bias to shift the range for unsigned output.
                // Input range (packed): [-128, 127]
                // Intermediate range: [-128+16, 127+16] -> [-112, 143]
                // Output range (added): [-112, 127] (saturated addition)
                let added = _mm256_adds_epi8(packed, bias8);

                // Clamp to non-negative range for final output.
                // Input range (added): [-112, 127]
                // Output range (result): [0, 127]
                let result = _mm256_max_epi8(added, zero8);

                _mm256_store_si256(output_ptr, result);
                output_ptr = output_ptr.add(1);
            }
        }
    }

    /// Fallback scalar implementation.
    fn forward_fallback(&self, feature_indices: &[usize], output: &mut [u8]) {
        let mut acc = [0; OUTPUT_DIMS];

        for fi in feature_indices {
            let weights = &self.weights[*fi * OUTPUT_DIMS..][..OUTPUT_DIMS];
            for (a, w) in acc.iter_mut().zip(weights.iter()) {
                *a += *w;
            }
        }

        for i in 0..OUTPUT_DIMS {
            let v = acc[i] + self.biases[i];
            // Apply LeakyReLU activation and clamp
            let v = if v >= 0 {
                v.min(111)
            } else {
                (v >> 3).max(-16)
            };

            // Add bias and convert to unsigned 8-bit
            output[i] = (v + 16) as u8;
        }
    }
}
