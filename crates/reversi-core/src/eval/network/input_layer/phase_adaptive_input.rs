//! Phase-adaptive input layer for neural network evaluation.

use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
#[allow(unused_imports)]
use crate::eval::pattern_feature::NUM_FEATURES;
use crate::eval::pattern_feature::{INPUT_FEATURE_DIMS, PatternFeature};
use crate::eval::util::clone_biases;
#[allow(unused_imports)]
use crate::eval::util::feature_offset;
use crate::util::align::Align64;
use crate::util::aligned_buffer::AlignedBuffer;

use super::accumulate_scalar;

const ACTIVATION_MAX: i16 = 255 * 2;
const ACTIVATION_SHIFT: u32 = 10;
pub(in crate::eval::network) const OUTPUT_DIMS: usize = 128;
const NUM_PA_INPUTS: usize = 6;
const PA_INPUT_BUCKET_SIZE: usize = 60 / NUM_PA_INPUTS;

/// Phase-adaptive input layer.
#[derive(Debug)]
pub struct PhaseAdaptiveInputLayer {
    biases: AlignedBuffer<i16, CACHE_LINE_SIZE>,
    weights: AlignedBuffer<i16, CACHE_LINE_SIZE>,
}

impl PhaseAdaptiveInputLayer {
    /// Loads network weights and biases from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = AlignedBuffer::<i16, CACHE_LINE_SIZE>::from_elem(0, OUTPUT_DIMS);
        let mut weights =
            AlignedBuffer::<i16, CACHE_LINE_SIZE>::from_elem(0, INPUT_FEATURE_DIMS * OUTPUT_DIMS);

        reader.read_i16_into::<LittleEndian>(biases.as_mut_slice())?;
        reader.read_i16_into::<LittleEndian>(weights.as_mut_slice())?;

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
            let output_ptr = output.as_mut_ptr();

            // SQSHLU fuses clamp-to-non-negative with the <<5 step. After
            // `vminq_s16(_, one)` the post-shift max is 510<<5 = 16320 (within
            // u16); negatives saturate to 0, zeroing the squared product.
            macro_rules! activate_pair {
                ($a:ident, $b:ident, $dst:expr) => {{
                    let a_min = vminq_s16($a, one);
                    let b_min = vminq_s16($b, one);
                    let a = vqdmulhq_s16(vreinterpretq_s16_u16(vqshluq_n_s16::<5>(a_min)), a_min);
                    let b = vqdmulhq_s16(vreinterpretq_s16_u16(vqshluq_n_s16::<5>(b_min)), b_min);
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

    /// Computes the forward pass using the scalar fallback.
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
    use crate::eval::pattern_feature::{PATTERN_FEATURE_OFFSETS, calc_pattern_size};

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use super::super::simd_layout::permute_rows;

    fn valid_pattern_feature(seed: usize) -> PatternFeature {
        let mut pattern_feature = PatternFeature::new();
        for idx in 0..NUM_FEATURES {
            pattern_feature[idx] = ((idx * seed + 17) % calc_pattern_size(idx)) as u16;
        }
        pattern_feature
    }

    fn build_layer(pattern_feature: &PatternFeature, seed: i32) -> PhaseAdaptiveInputLayer {
        let mut layer = PhaseAdaptiveInputLayer {
            biases: AlignedBuffer::from_elem(0, OUTPUT_DIMS),
            weights: AlignedBuffer::from_elem(0, INPUT_FEATURE_DIMS * OUTPUT_DIMS),
        };

        for (idx, bias) in layer.biases.iter_mut().enumerate() {
            *bias = ((idx as i32 * 31 + seed).rem_euclid(1100) - 350) as i16;
        }

        for feature_idx in 0..NUM_FEATURES {
            let row =
                PATTERN_FEATURE_OFFSETS[feature_idx] + usize::from(pattern_feature[feature_idx]);
            let start = row * OUTPUT_DIMS;
            for (dim, weight) in layer.weights[start..start + OUTPUT_DIMS]
                .iter_mut()
                .enumerate()
            {
                *weight =
                    ((feature_idx as i32 * 19 + dim as i32 * 5 + seed).rem_euclid(37) - 18) as i16;
            }
        }

        layer
    }

    fn constant_layer(value: i16) -> PhaseAdaptiveInputLayer {
        PhaseAdaptiveInputLayer {
            biases: AlignedBuffer::from_elem(value, OUTPUT_DIMS),
            weights: AlignedBuffer::from_elem(0, INPUT_FEATURE_DIMS * OUTPUT_DIMS),
        }
    }

    fn dispatch_ready_layer(natural: &PhaseAdaptiveInputLayer) -> PhaseAdaptiveInputLayer {
        let layer = PhaseAdaptiveInputLayer {
            biases: natural.biases.clone(),
            weights: natural.weights.clone(),
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let mut layer = layer;
            permute_rows(layer.biases.as_mut_slice(), OUTPUT_DIMS);
            permute_rows(layer.weights.as_mut_slice(), OUTPUT_DIMS);
            layer
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            layer
        }
    }

    fn reference_forward(
        layer: &PhaseAdaptiveInputLayer,
        pattern_feature: &PatternFeature,
    ) -> [u8; OUTPUT_DIMS] {
        let mut acc = [0i16; OUTPUT_DIMS];
        acc.copy_from_slice(&layer.biases[..OUTPUT_DIMS]);

        for feature_idx in 0..NUM_FEATURES {
            let row =
                PATTERN_FEATURE_OFFSETS[feature_idx] + usize::from(pattern_feature[feature_idx]);
            let start = row * OUTPUT_DIMS;
            for (acc_value, &weight) in acc
                .iter_mut()
                .zip(&layer.weights[start..start + OUTPUT_DIMS])
            {
                *acc_value += weight;
            }
        }

        let mut output = [0; OUTPUT_DIMS];
        for (out, &value) in output.iter_mut().zip(acc.iter()) {
            let value = value.clamp(0, ACTIVATION_MAX) as u32;
            *out = ((value * value) >> ACTIVATION_SHIFT) as u8;
        }
        output
    }

    #[test]
    fn layer_forward_fallback_matches_independent_accumulate_and_screlu_reference() {
        let pattern_feature = valid_pattern_feature(4099);
        let layer = build_layer(&pattern_feature, 29);
        let expected = reference_forward(&layer, &pattern_feature);
        let mut actual = Align64([0xCC; OUTPUT_DIMS + 4]);

        layer.forward_fallback(&pattern_feature, actual.as_mut_slice());

        assert_eq!(&actual.as_ref()[..OUTPUT_DIMS], &expected);
        assert_eq!(&actual.as_ref()[OUTPUT_DIMS..], &[0xCC; 4]);
    }

    #[test]
    fn phase_adaptive_input_selects_the_expected_ten_ply_bucket() {
        let pattern_feature = PatternFeature::new();
        let input = PhaseAdaptiveInput {
            inputs: (0..NUM_PA_INPUTS)
                .map(|bucket| constant_layer((bucket as i16) * 64))
                .collect(),
        };
        let cases = [
            (0, 0),
            (9, 0),
            (10, 1),
            (19, 1),
            (20, 2),
            (29, 2),
            (30, 3),
            (39, 3),
            (40, 4),
            (49, 4),
            (50, 5),
            (59, 5),
        ];

        for (ply, bucket) in cases {
            let mut output = Align64([0xFF; OUTPUT_DIMS]);
            input.forward(&pattern_feature, ply, output.as_mut_slice());
            let value = (bucket as u32) * 64;
            let expected = ((value * value) >> ACTIVATION_SHIFT) as u8;

            assert_eq!(output.as_ref(), &[expected; OUTPUT_DIMS], "ply {ply}");
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "ply 60 out of valid range 0-59")]
    fn phase_adaptive_input_rejects_ply_60_in_debug_builds() {
        let pattern_feature = PatternFeature::new();
        let input = PhaseAdaptiveInput {
            inputs: (0..NUM_PA_INPUTS).map(|_| constant_layer(0)).collect(),
        };
        let mut output = Align64([0; OUTPUT_DIMS]);

        input.forward(&pattern_feature, 60, output.as_mut_slice());
    }

    #[test]
    fn layer_forward_dispatch_matches_fallback_for_the_runtime_layout() {
        let pattern_feature = valid_pattern_feature(2053);
        let natural = build_layer(&pattern_feature, 83);
        let dispatch = dispatch_ready_layer(&natural);
        let expected = reference_forward(&natural, &pattern_feature);
        let mut actual = Align64([0; OUTPUT_DIMS]);

        dispatch.forward(&pattern_feature, actual.as_mut_slice());

        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn forward_neon_matches_fallback_for_natural_layout() {
        let pattern_feature = valid_pattern_feature(1237);
        let layer = build_layer(&pattern_feature, 101);
        let expected = reference_forward(&layer, &pattern_feature);
        let mut actual = Align64([0; OUTPUT_DIMS]);

        unsafe { layer.forward_neon(&pattern_feature, actual.as_mut_slice()) };

        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512bw"),
    ))]
    fn forward_avx2_matches_fallback_for_permuted_layout() {
        let pattern_feature = valid_pattern_feature(1237);
        let natural = build_layer(&pattern_feature, 101);
        let simd = dispatch_ready_layer(&natural);
        let expected = reference_forward(&natural, &pattern_feature);
        let mut actual = Align64([0; OUTPUT_DIMS]);

        unsafe { simd.forward_avx2(&pattern_feature, actual.as_mut_slice()) };

        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn forward_avx512_matches_fallback_for_permuted_layout() {
        let pattern_feature = valid_pattern_feature(1237);
        let natural = build_layer(&pattern_feature, 101);
        let simd = dispatch_ready_layer(&natural);
        let expected = reference_forward(&natural, &pattern_feature);
        let mut actual = Align64([0; OUTPUT_DIMS]);

        unsafe { simd.forward_avx512(&pattern_feature, actual.as_mut_slice()) };

        assert_eq!(actual.as_ref(), &expected);
    }
}
