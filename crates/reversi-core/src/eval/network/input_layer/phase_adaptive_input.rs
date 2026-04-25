//! Phase-adaptive input layer for neural network evaluation.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
#[allow(unused_imports)]
use crate::eval::pattern_feature::NUM_FEATURES;
use crate::eval::pattern_feature::{INPUT_FEATURE_DIMS, PatternFeature};
use crate::eval::util::clone_biases;
#[allow(unused_imports)]
use crate::eval::util::feature_offset;
use crate::util::align::Align64;

use super::accumulate_scalar;

const ACTIVATION_MAX: i16 = 255 * 2;
const ACTIVATION_SHIFT: u32 = 10;
pub(in crate::eval::network) const OUTPUT_DIMS: usize = 128;
const NUM_PA_INPUTS: usize = 6;
const PA_INPUT_BUCKET_SIZE: usize = 60 / NUM_PA_INPUTS;

/// Phase-adaptive input layer.
#[derive(Debug)]
pub struct PhaseAdaptiveInputLayer {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl PhaseAdaptiveInputLayer {
    /// Loads network weights and biases from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_FEATURE_DIMS * OUTPUT_DIMS];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // Permute weights and biases for optimal SIMD access patterns.
            use super::simd_layout::permute_rows;
            permute_rows(biases.as_mut_slice(), OUTPUT_DIMS);
            permute_rows(weights.as_mut_slice(), OUTPUT_DIMS);
        }

        Ok(PhaseAdaptiveInputLayer { biases, weights })
    }

    /// Performs a forward pass through the phase-adaptive input layer.
    #[inline(always)]
    pub fn forward(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx512bw") => {
                unsafe { self.forward_avx512(pattern_feature, output) };
            }
            all(target_arch = "x86_64", target_feature = "avx2") => {
                unsafe { self.forward_avx2(pattern_feature, output) };
            }
            all(target_arch = "aarch64", target_feature = "neon") => {
                unsafe { self.forward_neon(pattern_feature, output) };
            }
            _ => {
                self.forward_fallback(pattern_feature, output);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    #[allow(dead_code)]
    fn forward_avx512(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        use std::arch::x86_64::*;

        const NUM_REGS: usize = 4;

        unsafe {
            let weights_ptr = self.weights.as_ptr() as *const __m512i;
            let bias_ptr = self.biases.as_ptr() as *const __m512i;

            let mut acc0 = _mm512_load_si512(bias_ptr);
            let mut acc1 = _mm512_load_si512(bias_ptr.add(1));
            let mut acc2 = _mm512_load_si512(bias_ptr.add(2));
            let mut acc3 = _mm512_load_si512(bias_ptr.add(3));

            macro_rules! accumulate_feature {
                ($feature_idx:expr) => {{
                    let idx = feature_offset(pattern_feature, $feature_idx) * NUM_REGS;
                    acc0 = _mm512_add_epi16(acc0, _mm512_load_si512(weights_ptr.add(idx)));
                    acc1 = _mm512_add_epi16(acc1, _mm512_load_si512(weights_ptr.add(idx + 1)));
                    acc2 = _mm512_add_epi16(acc2, _mm512_load_si512(weights_ptr.add(idx + 2)));
                    acc3 = _mm512_add_epi16(acc3, _mm512_load_si512(weights_ptr.add(idx + 3)));
                }};
            }

            let mut feature_idx = 0;
            while feature_idx < NUM_FEATURES {
                accumulate_feature!(feature_idx);
                feature_idx += 1;
            }

            let one = _mm512_set1_epi16(ACTIVATION_MAX);
            let zero = _mm512_setzero_si512();
            let output_ptr = output.as_mut_ptr() as *mut __m512i;

            macro_rules! activate_pair {
                ($a:ident, $b:ident, $dst:expr) => {{
                    let a = _mm512_max_epi16(_mm512_min_epi16($a, one), zero);
                    let b = _mm512_max_epi16(_mm512_min_epi16($b, one), zero);
                    let a = _mm512_mulhi_epu16(_mm512_slli_epi16(a, 6), a);
                    let b = _mm512_mulhi_epu16(_mm512_slli_epi16(b, 6), b);
                    _mm512_store_si512(output_ptr.add($dst), _mm512_packus_epi16(a, b));
                }};
            }

            activate_pair!(acc0, acc1, 0);
            activate_pair!(acc2, acc3, 1);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        use std::arch::x86_64::*;

        const NUM_REGS: usize = 8;

        unsafe {
            let weights_ptr = self.weights.as_ptr() as *const __m256i;
            let bias_ptr = self.biases.as_ptr() as *const __m256i;

            let mut acc0 = _mm256_load_si256(bias_ptr);
            let mut acc1 = _mm256_load_si256(bias_ptr.add(1));
            let mut acc2 = _mm256_load_si256(bias_ptr.add(2));
            let mut acc3 = _mm256_load_si256(bias_ptr.add(3));
            let mut acc4 = _mm256_load_si256(bias_ptr.add(4));
            let mut acc5 = _mm256_load_si256(bias_ptr.add(5));
            let mut acc6 = _mm256_load_si256(bias_ptr.add(6));
            let mut acc7 = _mm256_load_si256(bias_ptr.add(7));

            macro_rules! accumulate_feature {
                ($feature_idx:expr) => {{
                    let idx = feature_offset(pattern_feature, $feature_idx) * NUM_REGS;
                    acc0 = _mm256_add_epi16(acc0, _mm256_load_si256(weights_ptr.add(idx)));
                    acc1 = _mm256_add_epi16(acc1, _mm256_load_si256(weights_ptr.add(idx + 1)));
                    acc2 = _mm256_add_epi16(acc2, _mm256_load_si256(weights_ptr.add(idx + 2)));
                    acc3 = _mm256_add_epi16(acc3, _mm256_load_si256(weights_ptr.add(idx + 3)));
                    acc4 = _mm256_add_epi16(acc4, _mm256_load_si256(weights_ptr.add(idx + 4)));
                    acc5 = _mm256_add_epi16(acc5, _mm256_load_si256(weights_ptr.add(idx + 5)));
                    acc6 = _mm256_add_epi16(acc6, _mm256_load_si256(weights_ptr.add(idx + 6)));
                    acc7 = _mm256_add_epi16(acc7, _mm256_load_si256(weights_ptr.add(idx + 7)));
                }};
            }

            let mut feature_idx = 0;
            while feature_idx < NUM_FEATURES {
                accumulate_feature!(feature_idx);
                feature_idx += 1;
            }

            let one = _mm256_set1_epi16(ACTIVATION_MAX);
            let zero = _mm256_setzero_si256();
            let output_ptr = output.as_mut_ptr() as *mut __m256i;

            macro_rules! activate_pair {
                ($a:ident, $b:ident, $dst:expr) => {{
                    let a = _mm256_max_epi16(_mm256_min_epi16($a, one), zero);
                    let b = _mm256_max_epi16(_mm256_min_epi16($b, one), zero);
                    let a = _mm256_mulhi_epu16(_mm256_slli_epi16(a, 6), a);
                    let b = _mm256_mulhi_epu16(_mm256_slli_epi16(b, 6), b);
                    _mm256_store_si256(output_ptr.add($dst), _mm256_packus_epi16(a, b));
                }};
            }

            activate_pair!(acc0, acc1, 0);
            activate_pair!(acc2, acc3, 1);
            activate_pair!(acc4, acc5, 2);
            activate_pair!(acc6, acc7, 3);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[allow(dead_code)]
    fn forward_neon(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        use std::arch::aarch64::*;

        unsafe {
            let weights_ptr = self.weights.as_ptr();
            let bias_ptr = self.biases.as_ptr();

            let mut acc0 = vld1q_s16(bias_ptr);
            let mut acc1 = vld1q_s16(bias_ptr.add(8));
            let mut acc2 = vld1q_s16(bias_ptr.add(16));
            let mut acc3 = vld1q_s16(bias_ptr.add(24));
            let mut acc4 = vld1q_s16(bias_ptr.add(32));
            let mut acc5 = vld1q_s16(bias_ptr.add(40));
            let mut acc6 = vld1q_s16(bias_ptr.add(48));
            let mut acc7 = vld1q_s16(bias_ptr.add(56));
            let mut acc8 = vld1q_s16(bias_ptr.add(64));
            let mut acc9 = vld1q_s16(bias_ptr.add(72));
            let mut acc10 = vld1q_s16(bias_ptr.add(80));
            let mut acc11 = vld1q_s16(bias_ptr.add(88));
            let mut acc12 = vld1q_s16(bias_ptr.add(96));
            let mut acc13 = vld1q_s16(bias_ptr.add(104));
            let mut acc14 = vld1q_s16(bias_ptr.add(112));
            let mut acc15 = vld1q_s16(bias_ptr.add(120));

            macro_rules! accumulate_feature {
                ($feature_idx:expr) => {{
                    let idx = feature_offset(pattern_feature, $feature_idx) * OUTPUT_DIMS;
                    let base = weights_ptr.add(idx);
                    acc0 = vaddq_s16(acc0, vld1q_s16(base));
                    acc1 = vaddq_s16(acc1, vld1q_s16(base.add(8)));
                    acc2 = vaddq_s16(acc2, vld1q_s16(base.add(16)));
                    acc3 = vaddq_s16(acc3, vld1q_s16(base.add(24)));
                    acc4 = vaddq_s16(acc4, vld1q_s16(base.add(32)));
                    acc5 = vaddq_s16(acc5, vld1q_s16(base.add(40)));
                    acc6 = vaddq_s16(acc6, vld1q_s16(base.add(48)));
                    acc7 = vaddq_s16(acc7, vld1q_s16(base.add(56)));
                    acc8 = vaddq_s16(acc8, vld1q_s16(base.add(64)));
                    acc9 = vaddq_s16(acc9, vld1q_s16(base.add(72)));
                    acc10 = vaddq_s16(acc10, vld1q_s16(base.add(80)));
                    acc11 = vaddq_s16(acc11, vld1q_s16(base.add(88)));
                    acc12 = vaddq_s16(acc12, vld1q_s16(base.add(96)));
                    acc13 = vaddq_s16(acc13, vld1q_s16(base.add(104)));
                    acc14 = vaddq_s16(acc14, vld1q_s16(base.add(112)));
                    acc15 = vaddq_s16(acc15, vld1q_s16(base.add(120)));
                }};
            }

            let mut feature_idx = 0;
            while feature_idx < NUM_FEATURES {
                accumulate_feature!(feature_idx);
                feature_idx += 1;
            }

            let one = vdupq_n_s16(ACTIVATION_MAX);
            let zero = vdupq_n_s16(0);
            let output_ptr = output.as_mut_ptr();

            macro_rules! activate_pair {
                ($a:ident, $b:ident, $dst:expr) => {{
                    let a = vmaxq_s16(vminq_s16($a, one), zero);
                    let b = vmaxq_s16(vminq_s16($b, one), zero);
                    let a = vqdmulhq_s16(vshlq_n_s16::<5>(a), a);
                    let b = vqdmulhq_s16(vshlq_n_s16::<5>(b), b);
                    vst1q_u8(
                        output_ptr.add($dst * 16),
                        vcombine_u8(vqmovun_s16(a), vqmovun_s16(b)),
                    );
                }};
            }

            activate_pair!(acc0, acc1, 0);
            activate_pair!(acc2, acc3, 1);
            activate_pair!(acc4, acc5, 2);
            activate_pair!(acc6, acc7, 3);
            activate_pair!(acc8, acc9, 4);
            activate_pair!(acc10, acc11, 5);
            activate_pair!(acc12, acc13, 6);
            activate_pair!(acc14, acc15, 7);
        }
    }

    /// Fallback scalar implementation.
    #[allow(dead_code)]
    fn forward_fallback(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        let mut acc: Align64<[i16; OUTPUT_DIMS]> = clone_biases(&self.biases);
        accumulate_scalar::<OUTPUT_DIMS>(pattern_feature, &self.weights, &mut acc);

        for (out, &acc_v) in output[..OUTPUT_DIMS].iter_mut().zip(acc.0.iter()) {
            let v = acc_v.clamp(0, ACTIVATION_MAX) as u32;
            *out = ((v * v) >> ACTIVATION_SHIFT) as u8;
        }
    }
}

/// A set of phase-adaptive input layers for different game phases.
///
/// Encapsulates the phase selection logic, choosing the appropriate
/// input layer based on the current ply.
#[derive(Debug)]
pub struct PhaseAdaptiveInput {
    inputs: Vec<PhaseAdaptiveInputLayer>,
}

impl PhaseAdaptiveInput {
    /// Loads all phase-adaptive input layers from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let inputs = (0..NUM_PA_INPUTS)
            .map(|_| PhaseAdaptiveInputLayer::load(reader))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(PhaseAdaptiveInput { inputs })
    }

    /// Performs a forward pass, selecting the input layer based on the current ply.
    #[inline(always)]
    pub fn forward(&self, pattern_feature: &PatternFeature, ply: usize, output: &mut [u8]) {
        debug_assert!(ply < 60, "ply {} out of valid range 0-59", ply);
        let pa_index = ply / PA_INPUT_BUCKET_SIZE;
        let pa_input = &self.inputs[pa_index];
        pa_input.forward(pattern_feature, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aligned_vec::avec;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use super::super::simd_layout::permute_rows;

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn forward_neon_128_matches_fallback() {
        let mut pattern_feature = PatternFeature::new();
        for idx in 0..crate::eval::pattern_feature::NUM_FEATURES {
            pattern_feature[idx] = ((idx * 17 + 5) % 113) as u16;
        }

        let mut layer = PhaseAdaptiveInputLayer {
            biases: avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS],
            weights: avec![[CACHE_LINE_SIZE]|0i16; INPUT_FEATURE_DIMS * OUTPUT_DIMS],
        };

        for (idx, bias) in layer.biases.iter_mut().enumerate() {
            *bias = ((idx as i32 * 19 + 7) % 997 - 498) as i16;
        }

        for (idx, weight) in layer.weights.iter_mut().enumerate() {
            *weight = ((idx as i32 * 23 + 11) % 31 - 15) as i16;
        }

        let mut expected = Align64([0u8; OUTPUT_DIMS]);
        let mut actual = Align64([0u8; OUTPUT_DIMS]);
        layer.forward_fallback(&pattern_feature, expected.as_mut_slice());
        // SAFETY: NEON is the aarch64 baseline, asserted by the cfg above.
        unsafe { layer.forward_neon(&pattern_feature, actual.as_mut_slice()) };

        assert_eq!(actual.as_ref(), expected.as_ref());
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512bw"),
    ))]
    fn forward_avx2_128_matches_fallback() {
        let mut pattern_feature = PatternFeature::new();
        for idx in 0..crate::eval::pattern_feature::NUM_FEATURES {
            pattern_feature[idx] = ((idx * 17 + 5) % 113) as u16;
        }

        let mut natural = PhaseAdaptiveInputLayer {
            biases: avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS],
            weights: avec![[CACHE_LINE_SIZE]|0i16; INPUT_FEATURE_DIMS * OUTPUT_DIMS],
        };

        for (idx, bias) in natural.biases.iter_mut().enumerate() {
            *bias = ((idx as i32 * 19 + 7) % 997 - 498) as i16;
        }

        for (idx, weight) in natural.weights.iter_mut().enumerate() {
            *weight = ((idx as i32 * 23 + 11) % 31 - 15) as i16;
        }

        let mut simd = PhaseAdaptiveInputLayer {
            biases: natural.biases.clone(),
            weights: natural.weights.clone(),
        };
        permute_rows(simd.biases.as_mut_slice(), OUTPUT_DIMS);
        permute_rows(simd.weights.as_mut_slice(), OUTPUT_DIMS);

        let mut expected = Align64([0u8; OUTPUT_DIMS]);
        let mut actual = Align64([0u8; OUTPUT_DIMS]);
        natural.forward_fallback(&pattern_feature, expected.as_mut_slice());
        // SAFETY: AVX2 is asserted by the cfg above; output is 64-byte aligned.
        unsafe { simd.forward_avx2(&pattern_feature, actual.as_mut_slice()) };

        assert_eq!(actual.as_ref(), expected.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn forward_avx512_128_matches_fallback() {
        let mut pattern_feature = PatternFeature::new();
        for idx in 0..crate::eval::pattern_feature::NUM_FEATURES {
            pattern_feature[idx] = ((idx * 17 + 5) % 113) as u16;
        }

        let mut natural = PhaseAdaptiveInputLayer {
            biases: avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS],
            weights: avec![[CACHE_LINE_SIZE]|0i16; INPUT_FEATURE_DIMS * OUTPUT_DIMS],
        };

        for (idx, bias) in natural.biases.iter_mut().enumerate() {
            *bias = ((idx as i32 * 19 + 7) % 997 - 498) as i16;
        }

        for (idx, weight) in natural.weights.iter_mut().enumerate() {
            *weight = ((idx as i32 * 23 + 11) % 31 - 15) as i16;
        }

        let mut simd = PhaseAdaptiveInputLayer {
            biases: natural.biases.clone(),
            weights: natural.weights.clone(),
        };
        permute_rows(simd.biases.as_mut_slice(), OUTPUT_DIMS);
        permute_rows(simd.weights.as_mut_slice(), OUTPUT_DIMS);

        let mut expected = Align64([0u8; OUTPUT_DIMS]);
        let mut actual = Align64([0u8; OUTPUT_DIMS]);
        natural.forward_fallback(&pattern_feature, expected.as_mut_slice());
        // SAFETY: AVX-512BW is asserted by the cfg above; output is 64-byte aligned.
        unsafe { simd.forward_avx512(&pattern_feature, actual.as_mut_slice()) };

        assert_eq!(actual.as_ref(), expected.as_ref());
    }
}
