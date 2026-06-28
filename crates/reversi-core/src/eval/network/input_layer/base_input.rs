//! Base input layer for neural network evaluation.

use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
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
const HIDDEN_DIMS: usize = OUTPUT_DIMS * 2;
const _: () = assert!(NUM_FEATURES == 32);

/// Neural network base input layer.
///
/// Reference: <https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h>
pub struct BaseInput {
    biases: AlignedBuffer<i16, CACHE_LINE_SIZE>,
    weights: AlignedBuffer<i16, CACHE_LINE_SIZE>,
}

impl BaseInput {
    /// Loads network weights and biases from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = AlignedBuffer::<i16, CACHE_LINE_SIZE>::from_elem(0, HIDDEN_DIMS);
        let mut weights =
            AlignedBuffer::<i16, CACHE_LINE_SIZE>::from_elem(0, INPUT_FEATURE_DIMS * HIDDEN_DIMS);

        reader.read_i16_into::<LittleEndian>(biases.as_mut_slice())?;
        reader.read_i16_into::<LittleEndian>(weights.as_mut_slice())?;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // Permute weights and biases for optimal SIMD access patterns.
            use super::simd_layout::permute_rows;
            permute_rows(biases.as_mut_slice(), HIDDEN_DIMS);
            permute_rows(weights.as_mut_slice(), HIDDEN_DIMS);
        }

        Ok(BaseInput { biases, weights })
    }

    /// Performs a forward pass through the base input layer.
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
                self.forward_scalar(pattern_feature, output);
            }
        }
    }

    /// Computes the forward pass using the scalar fallback.
    #[allow(dead_code)]
    fn forward_scalar(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        let mut acc: Align64<[i16; HIDDEN_DIMS]> = clone_biases(&self.biases);

        accumulate_scalar::<HIDDEN_DIMS>(pattern_feature, &self.weights, &mut acc);

        let (lo, hi) = acc.0.split_at(OUTPUT_DIMS);
        for ((out, &v0), &v1) in output[..OUTPUT_DIMS].iter_mut().zip(lo).zip(hi) {
            let sum0 = v0.clamp(0, ACTIVATION_MAX) as u32;
            let sum1 = v1.clamp(0, ACTIVATION_MAX) as u32;
            *out = ((sum0 * sum1) >> ACTIVATION_SHIFT) as u8;
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    #[allow(dead_code)]
    fn forward_avx512(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        use std::arch::x86_64::*;

        const NUM_REGS: usize = 8;

        unsafe {
            let weights_ptr = self.weights.as_ptr() as *const __m512i;
            let bias_ptr = self.biases.as_ptr() as *const __m512i;

            let mut acc0 = _mm512_load_si512(bias_ptr);
            let mut acc1 = _mm512_load_si512(bias_ptr.add(1));
            let mut acc2 = _mm512_load_si512(bias_ptr.add(2));
            let mut acc3 = _mm512_load_si512(bias_ptr.add(3));
            let mut acc4 = _mm512_load_si512(bias_ptr.add(4));
            let mut acc5 = _mm512_load_si512(bias_ptr.add(5));
            let mut acc6 = _mm512_load_si512(bias_ptr.add(6));
            let mut acc7 = _mm512_load_si512(bias_ptr.add(7));

            macro_rules! accumulate_reg {
                ($idx0:ident, $idx1:ident, $idx2:ident, $idx3:ident, $j:expr, $acc:ident) => {{
                    let w0 = _mm512_load_si512(weights_ptr.add($idx0 + $j));
                    let w1 = _mm512_load_si512(weights_ptr.add($idx1 + $j));
                    let w2 = _mm512_load_si512(weights_ptr.add($idx2 + $j));
                    let w3 = _mm512_load_si512(weights_ptr.add($idx3 + $j));
                    let sum01 = _mm512_add_epi16(w0, w1);
                    let sum23 = _mm512_add_epi16(w2, w3);
                    $acc = _mm512_add_epi16($acc, _mm512_add_epi16(sum01, sum23));
                }};
            }

            macro_rules! accumulate_feature_group {
                ($base:expr) => {{
                    let idx0 = feature_offset(pattern_feature, $base) * NUM_REGS;
                    let idx1 = feature_offset(pattern_feature, $base + 1) * NUM_REGS;
                    let idx2 = feature_offset(pattern_feature, $base + 2) * NUM_REGS;
                    let idx3 = feature_offset(pattern_feature, $base + 3) * NUM_REGS;
                    accumulate_reg!(idx0, idx1, idx2, idx3, 0, acc0);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 1, acc1);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 2, acc2);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 3, acc3);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 4, acc4);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 5, acc5);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 6, acc6);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 7, acc7);
                }};
            }

            accumulate_feature_group!(0);
            accumulate_feature_group!(4);
            accumulate_feature_group!(8);
            accumulate_feature_group!(12);
            accumulate_feature_group!(16);
            accumulate_feature_group!(20);
            accumulate_feature_group!(24);
            accumulate_feature_group!(28);

            let one = _mm512_set1_epi16(ACTIVATION_MAX);
            let zero = _mm512_setzero_si512();
            let output_ptr = output.as_mut_ptr() as *mut __m512i;

            macro_rules! activate_pair {
                ($lo0:ident, $lo1:ident, $hi0:ident, $hi1:ident, $dst:expr) => {{
                    let lo0 = _mm512_max_epi16(_mm512_min_epi16($lo0, one), zero);
                    let lo1 = _mm512_max_epi16(_mm512_min_epi16($lo1, one), zero);
                    let hi0 = _mm512_min_epi16($hi0, one);
                    let hi1 = _mm512_min_epi16($hi1, one);
                    let out0 = _mm512_mulhi_epi16(_mm512_slli_epi16(lo0, 6), hi0);
                    let out1 = _mm512_mulhi_epi16(_mm512_slli_epi16(lo1, 6), hi1);
                    _mm512_store_si512(output_ptr.add($dst), _mm512_packus_epi16(out0, out1));
                }};
            }

            activate_pair!(acc0, acc1, acc4, acc5, 0);
            activate_pair!(acc2, acc3, acc6, acc7, 1);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        use std::arch::x86_64::*;

        const NUM_REGS: usize = 16;
        const HALF_REGS: usize = NUM_REGS / 2;

        unsafe {
            let weights_ptr = self.weights.as_ptr() as *const __m256i;
            let bias_ptr = self.biases.as_ptr() as *const __m256i;
            let mut lo_acc = Align64([0i16; OUTPUT_DIMS]);
            let lo_acc_ptr = lo_acc.as_mut_ptr() as *mut __m256i;

            let mut acc0 = _mm256_load_si256(bias_ptr);
            let mut acc1 = _mm256_load_si256(bias_ptr.add(1));
            let mut acc2 = _mm256_load_si256(bias_ptr.add(2));
            let mut acc3 = _mm256_load_si256(bias_ptr.add(3));
            let mut acc4 = _mm256_load_si256(bias_ptr.add(4));
            let mut acc5 = _mm256_load_si256(bias_ptr.add(5));
            let mut acc6 = _mm256_load_si256(bias_ptr.add(6));
            let mut acc7 = _mm256_load_si256(bias_ptr.add(7));

            macro_rules! accumulate_reg {
                ($idx0:ident, $idx1:ident, $idx2:ident, $idx3:ident, $j:expr, $acc:ident) => {{
                    let w0 = _mm256_load_si256(weights_ptr.add($idx0 + $j));
                    let w1 = _mm256_load_si256(weights_ptr.add($idx1 + $j));
                    let w2 = _mm256_load_si256(weights_ptr.add($idx2 + $j));
                    let w3 = _mm256_load_si256(weights_ptr.add($idx3 + $j));
                    let sum01 = _mm256_add_epi16(w0, w1);
                    let sum23 = _mm256_add_epi16(w2, w3);
                    $acc = _mm256_add_epi16($acc, _mm256_add_epi16(sum01, sum23));
                }};
            }

            macro_rules! accumulate_feature_group {
                ($base:expr) => {{
                    let idx0 = feature_offset(pattern_feature, $base) * NUM_REGS;
                    let idx1 = feature_offset(pattern_feature, $base + 1) * NUM_REGS;
                    let idx2 = feature_offset(pattern_feature, $base + 2) * NUM_REGS;
                    let idx3 = feature_offset(pattern_feature, $base + 3) * NUM_REGS;
                    accumulate_reg!(idx0, idx1, idx2, idx3, 0, acc0);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 1, acc1);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 2, acc2);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 3, acc3);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 4, acc4);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 5, acc5);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 6, acc6);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 7, acc7);
                }};
            }

            accumulate_feature_group!(0);
            accumulate_feature_group!(4);
            accumulate_feature_group!(8);
            accumulate_feature_group!(12);
            accumulate_feature_group!(16);
            accumulate_feature_group!(20);
            accumulate_feature_group!(24);
            accumulate_feature_group!(28);

            _mm256_store_si256(lo_acc_ptr, acc0);
            _mm256_store_si256(lo_acc_ptr.add(1), acc1);
            _mm256_store_si256(lo_acc_ptr.add(2), acc2);
            _mm256_store_si256(lo_acc_ptr.add(3), acc3);
            _mm256_store_si256(lo_acc_ptr.add(4), acc4);
            _mm256_store_si256(lo_acc_ptr.add(5), acc5);
            _mm256_store_si256(lo_acc_ptr.add(6), acc6);
            _mm256_store_si256(lo_acc_ptr.add(7), acc7);

            let mut acc0 = _mm256_load_si256(bias_ptr.add(HALF_REGS));
            let mut acc1 = _mm256_load_si256(bias_ptr.add(HALF_REGS + 1));
            let mut acc2 = _mm256_load_si256(bias_ptr.add(HALF_REGS + 2));
            let mut acc3 = _mm256_load_si256(bias_ptr.add(HALF_REGS + 3));
            let mut acc4 = _mm256_load_si256(bias_ptr.add(HALF_REGS + 4));
            let mut acc5 = _mm256_load_si256(bias_ptr.add(HALF_REGS + 5));
            let mut acc6 = _mm256_load_si256(bias_ptr.add(HALF_REGS + 6));
            let mut acc7 = _mm256_load_si256(bias_ptr.add(HALF_REGS + 7));

            macro_rules! accumulate_feature_group_hi {
                ($base:expr) => {{
                    let idx0 = feature_offset(pattern_feature, $base) * NUM_REGS + HALF_REGS;
                    let idx1 = feature_offset(pattern_feature, $base + 1) * NUM_REGS + HALF_REGS;
                    let idx2 = feature_offset(pattern_feature, $base + 2) * NUM_REGS + HALF_REGS;
                    let idx3 = feature_offset(pattern_feature, $base + 3) * NUM_REGS + HALF_REGS;
                    accumulate_reg!(idx0, idx1, idx2, idx3, 0, acc0);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 1, acc1);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 2, acc2);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 3, acc3);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 4, acc4);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 5, acc5);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 6, acc6);
                    accumulate_reg!(idx0, idx1, idx2, idx3, 7, acc7);
                }};
            }

            accumulate_feature_group_hi!(0);
            accumulate_feature_group_hi!(4);
            accumulate_feature_group_hi!(8);
            accumulate_feature_group_hi!(12);
            accumulate_feature_group_hi!(16);
            accumulate_feature_group_hi!(20);
            accumulate_feature_group_hi!(24);
            accumulate_feature_group_hi!(28);

            let one = _mm256_set1_epi16(ACTIVATION_MAX);
            let zero = _mm256_setzero_si256();
            let output_ptr = output.as_mut_ptr() as *mut __m256i;

            macro_rules! activate_pair {
                ($lo_idx:expr, $hi0:ident, $hi1:ident, $dst:expr) => {{
                    let lo0 = _mm256_max_epi16(
                        _mm256_min_epi16(_mm256_load_si256(lo_acc_ptr.add($lo_idx)), one),
                        zero,
                    );
                    let lo1 = _mm256_max_epi16(
                        _mm256_min_epi16(_mm256_load_si256(lo_acc_ptr.add($lo_idx + 1)), one),
                        zero,
                    );
                    let hi0 = _mm256_min_epi16($hi0, one);
                    let hi1 = _mm256_min_epi16($hi1, one);
                    let out0 = _mm256_mulhi_epi16(_mm256_slli_epi16(lo0, 6), hi0);
                    let out1 = _mm256_mulhi_epi16(_mm256_slli_epi16(lo1, 6), hi1);
                    _mm256_store_si256(output_ptr.add($dst), _mm256_packus_epi16(out0, out1));
                }};
            }

            activate_pair!(0, acc0, acc1, 0);
            activate_pair!(2, acc2, acc3, 1);
            activate_pair!(4, acc4, acc5, 2);
            activate_pair!(6, acc6, acc7, 3);
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
            let mut lo_acc = Align64([0i16; OUTPUT_DIMS]);
            let lo_acc_ptr = lo_acc.as_mut_ptr() as *mut i16;

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

            macro_rules! accumulate_feature_lo {
                ($feature_idx:expr) => {{
                    let idx = feature_offset(pattern_feature, $feature_idx) * HIDDEN_DIMS;
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
                accumulate_feature_lo!(feature_idx);
                feature_idx += 1;
            }

            vst1q_s16(lo_acc_ptr, acc0);
            vst1q_s16(lo_acc_ptr.add(8), acc1);
            vst1q_s16(lo_acc_ptr.add(16), acc2);
            vst1q_s16(lo_acc_ptr.add(24), acc3);
            vst1q_s16(lo_acc_ptr.add(32), acc4);
            vst1q_s16(lo_acc_ptr.add(40), acc5);
            vst1q_s16(lo_acc_ptr.add(48), acc6);
            vst1q_s16(lo_acc_ptr.add(56), acc7);
            vst1q_s16(lo_acc_ptr.add(64), acc8);
            vst1q_s16(lo_acc_ptr.add(72), acc9);
            vst1q_s16(lo_acc_ptr.add(80), acc10);
            vst1q_s16(lo_acc_ptr.add(88), acc11);
            vst1q_s16(lo_acc_ptr.add(96), acc12);
            vst1q_s16(lo_acc_ptr.add(104), acc13);
            vst1q_s16(lo_acc_ptr.add(112), acc14);
            vst1q_s16(lo_acc_ptr.add(120), acc15);

            let mut acc0 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS));
            let mut acc1 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 8));
            let mut acc2 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 16));
            let mut acc3 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 24));
            let mut acc4 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 32));
            let mut acc5 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 40));
            let mut acc6 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 48));
            let mut acc7 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 56));
            let mut acc8 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 64));
            let mut acc9 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 72));
            let mut acc10 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 80));
            let mut acc11 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 88));
            let mut acc12 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 96));
            let mut acc13 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 104));
            let mut acc14 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 112));
            let mut acc15 = vld1q_s16(bias_ptr.add(OUTPUT_DIMS + 120));

            macro_rules! accumulate_feature_hi {
                ($feature_idx:expr) => {{
                    let idx =
                        feature_offset(pattern_feature, $feature_idx) * HIDDEN_DIMS + OUTPUT_DIMS;
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

            feature_idx = 0;
            while feature_idx < NUM_FEATURES {
                accumulate_feature_hi!(feature_idx);
                feature_idx += 1;
            }

            let one = vdupq_n_s16(ACTIVATION_MAX);
            let output_ptr = output.as_mut_ptr();

            // SQSHLU fuses clamp-to-non-negative with the <<5 step:
            // negative inputs saturate to 0, and after `vminq_s16(_, one)`
            // the post-shift max is 510<<5 = 16320, well within u16.
            macro_rules! activate_pair {
                ($lo_idx:expr, $hi0:ident, $hi1:ident, $dst:expr) => {{
                    let lo0 = vreinterpretq_s16_u16(vqshluq_n_s16::<5>(vminq_s16(
                        vld1q_s16(lo_acc_ptr.add($lo_idx)),
                        one,
                    )));
                    let lo1 = vreinterpretq_s16_u16(vqshluq_n_s16::<5>(vminq_s16(
                        vld1q_s16(lo_acc_ptr.add($lo_idx + 8)),
                        one,
                    )));
                    let hi0 = vminq_s16($hi0, one);
                    let hi1 = vminq_s16($hi1, one);
                    let out0 = vqdmulhq_s16(lo0, hi0);
                    let out1 = vqdmulhq_s16(lo1, hi1);
                    vst1q_u8(
                        output_ptr.add($dst * 16),
                        vcombine_u8(vqmovun_s16(out0), vqmovun_s16(out1)),
                    );
                }};
            }

            activate_pair!(0, acc0, acc1, 0);
            activate_pair!(16, acc2, acc3, 1);
            activate_pair!(32, acc4, acc5, 2);
            activate_pair!(48, acc6, acc7, 3);
            activate_pair!(64, acc8, acc9, 4);
            activate_pair!(80, acc10, acc11, 5);
            activate_pair!(96, acc12, acc13, 6);
            activate_pair!(112, acc14, acc15, 7);
        }
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

    fn build_layer(pattern_feature: &PatternFeature, seed: i32) -> BaseInput {
        let mut layer = BaseInput {
            biases: AlignedBuffer::from_elem(0, HIDDEN_DIMS),
            weights: AlignedBuffer::from_elem(0, INPUT_FEATURE_DIMS * HIDDEN_DIMS),
        };

        for (idx, bias) in layer.biases.iter_mut().enumerate() {
            *bias = ((idx as i32 * 37 + seed).rem_euclid(1400) - 450) as i16;
        }

        for feature_idx in 0..NUM_FEATURES {
            let row =
                PATTERN_FEATURE_OFFSETS[feature_idx] + usize::from(pattern_feature[feature_idx]);
            let start = row * HIDDEN_DIMS;
            for (dim, weight) in layer.weights[start..start + HIDDEN_DIMS]
                .iter_mut()
                .enumerate()
            {
                *weight =
                    ((feature_idx as i32 * 17 + dim as i32 * 7 + seed).rem_euclid(41) - 20) as i16;
            }
        }

        layer
    }

    fn reference_forward(layer: &BaseInput, pattern_feature: &PatternFeature) -> [u8; OUTPUT_DIMS] {
        let mut acc = [0i16; HIDDEN_DIMS];
        acc.copy_from_slice(&layer.biases[..HIDDEN_DIMS]);

        for feature_idx in 0..NUM_FEATURES {
            let row =
                PATTERN_FEATURE_OFFSETS[feature_idx] + usize::from(pattern_feature[feature_idx]);
            let start = row * HIDDEN_DIMS;
            for (acc_value, &weight) in acc
                .iter_mut()
                .zip(&layer.weights[start..start + HIDDEN_DIMS])
            {
                *acc_value += weight;
            }
        }

        let mut output = [0; OUTPUT_DIMS];
        for idx in 0..OUTPUT_DIMS {
            let lo = acc[idx].clamp(0, ACTIVATION_MAX) as u32;
            let hi = acc[idx + OUTPUT_DIMS].clamp(0, ACTIVATION_MAX) as u32;
            output[idx] = ((lo * hi) >> ACTIVATION_SHIFT) as u8;
        }
        output
    }

    fn dispatch_ready_layer(natural: &BaseInput) -> BaseInput {
        let layer = BaseInput {
            biases: natural.biases.clone(),
            weights: natural.weights.clone(),
        };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let mut layer = layer;
            permute_rows(layer.biases.as_mut_slice(), HIDDEN_DIMS);
            permute_rows(layer.weights.as_mut_slice(), HIDDEN_DIMS);
            layer
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            layer
        }
    }

    #[test]
    fn forward_scalar_matches_independent_accumulate_and_activation_reference() {
        let pattern_feature = valid_pattern_feature(4099);
        let layer = build_layer(&pattern_feature, 23);
        let expected = reference_forward(&layer, &pattern_feature);
        let mut actual = Align64([0xCC; OUTPUT_DIMS + 4]);

        layer.forward_scalar(&pattern_feature, actual.as_mut_slice());

        assert_eq!(&actual.as_ref()[..OUTPUT_DIMS], &expected);
        assert_eq!(&actual.as_ref()[OUTPUT_DIMS..], &[0xCC; 4]);
    }

    #[test]
    fn forward_dispatch_matches_scalar_for_the_runtime_layout() {
        let pattern_feature = valid_pattern_feature(2053);
        let natural = build_layer(&pattern_feature, 71);
        let dispatch = dispatch_ready_layer(&natural);
        let expected = reference_forward(&natural, &pattern_feature);
        let mut actual = Align64([0; OUTPUT_DIMS]);

        dispatch.forward(&pattern_feature, actual.as_mut_slice());

        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn forward_neon_matches_scalar_for_natural_layout() {
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
    fn forward_avx2_matches_scalar_for_permuted_layout() {
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
    fn forward_avx512_matches_scalar_for_permuted_layout() {
        let pattern_feature = valid_pattern_feature(1237);
        let natural = build_layer(&pattern_feature, 101);
        let simd = dispatch_ready_layer(&natural);
        let expected = reference_forward(&natural, &pattern_feature);
        let mut actual = Align64([0; OUTPUT_DIMS]);

        unsafe { simd.forward_avx512(&pattern_feature, actual.as_mut_slice()) };

        assert_eq!(actual.as_ref(), &expected);
    }
}
