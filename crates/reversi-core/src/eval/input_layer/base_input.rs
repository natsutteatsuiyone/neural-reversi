//! Base input layer for neural network evaluation.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};
use cfg_if::cfg_if;

use crate::constants::CACHE_LINE_SIZE;
use crate::eval::pattern_feature::PatternFeature;
use crate::eval::util::clone_biases;
use crate::util::align::Align64;

use super::{ACTIVATION_MAX, accumulate_scalar, apply_base_activation_scalar};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use super::accumulate_avx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::accumulate_avx2;

#[allow(unused_macros)]
macro_rules! impl_base_input_apply_activation {
    (
        $fn_name:ident,
        $target_feature:literal,
        $lane_ty:ty,
        set1 = $set1:path,
        setzero = $setzero:path,
        max = $max:path,
        min = $min:path,
        slli = $slli:path,
        mulhi = $mulhi:path,
        packus = $packus:path
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $target_feature)]
        #[inline]
        fn $fn_name(&self, acc: &[i16; HIDDEN_DIMS], output: &mut [u8]) {
            use std::arch::x86_64::*;
            use std::mem::size_of;
            unsafe {
                let acc_ptr = acc.as_ptr() as *mut $lane_ty;
                let mut output_ptr = output.as_mut_ptr() as *mut $lane_ty;
                const LANES_PER_REG: usize = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = HIDDEN_DIMS / LANES_PER_REG;
                let one = $set1(ACTIVATION_MAX);
                let zero = $setzero();
                let mut in0_ptr = acc_ptr;
                let mut in1_ptr = acc_ptr.add(num_regs / 2);
                let iterations = num_regs / 4;

                for _ in 0..iterations {
                    let in00 = *in0_ptr;
                    let in01 = *in0_ptr.add(1);
                    in0_ptr = in0_ptr.add(2);
                    let in10 = *in1_ptr;
                    let in11 = *in1_ptr.add(1);
                    in1_ptr = in1_ptr.add(2);
                    let clamp0a = $max($min(in00, one), zero);
                    let clamp0b = $max($min(in01, one), zero);
                    let sum0a = $slli(clamp0a, 6);
                    let sum0b = $slli(clamp0b, 6);
                    let sum1a = $min(in10, one);
                    let sum1b = $min(in11, one);
                    let pa = $mulhi(sum0a, sum1a);
                    let pb = $mulhi(sum0b, sum1b);
                    *output_ptr = $packus(pa, pb);
                    output_ptr = output_ptr.add(1);
                }
            }
        }
    };
}

/// Neural network base input layer.
///
/// # Reference
///
/// - <https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h>
///
/// # Type Parameters
///
/// * `INPUT_DIMS` - Number of input features (sparse).
/// * `OUTPUT_DIMS` - Number of output dimensions (dense).
/// * `HIDDEN_DIMS` - Number of hidden units (must be 2 * OUTPUT_DIMS).
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
    ///
    /// * `reader` - Binary data reader containing network parameters.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; HIDDEN_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_DIMS * HIDDEN_DIMS];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // Permute weights and biases for optimal SIMD access patterns.
            use super::simd_layout::permute_rows;
            permute_rows(biases.as_mut_slice(), HIDDEN_DIMS);
            permute_rows(weights.as_mut_slice(), HIDDEN_DIMS);
        }

        Ok(BaseInput { biases, weights })
    }

    /// Performs forward pass through the base input layer.
    ///
    /// # Arguments
    ///
    /// * `pattern_feature` - Sparse feature values encoded by pattern index.
    /// * `output` - Output buffer to write results.
    #[inline(always)]
    pub fn forward(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))] {
                unsafe { self.forward_avx512(pattern_feature, output) };
            } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]  {
                unsafe { self.forward_avx2(pattern_feature, output) };
            } else {
                self.forward_fallback(pattern_feature, output);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    #[allow(dead_code)]
    fn forward_avx512(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        const {
            assert!(HIDDEN_DIMS.is_multiple_of(128));
            assert!(OUTPUT_DIMS * 2 == HIDDEN_DIMS);
        }

        let mut acc: Align64<[i16; HIDDEN_DIMS]> = clone_biases(&self.biases);
        accumulate_avx512::<HIDDEN_DIMS>(pattern_feature, &self.weights, &mut acc);
        self.apply_activation_avx512(&acc, output);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        const {
            assert!(HIDDEN_DIMS.is_multiple_of(64));
            assert!(OUTPUT_DIMS * 2 == HIDDEN_DIMS);
        }

        let mut acc: Align64<[i16; HIDDEN_DIMS]> = clone_biases(&self.biases);
        accumulate_avx2::<HIDDEN_DIMS>(pattern_feature, &self.weights, &mut acc);
        self.apply_activation_avx2(&acc, output);
    }

    // AVX-512-optimized activation function.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    impl_base_input_apply_activation!(
        apply_activation_avx512,
        "avx512bw",
        __m512i,
        set1 = _mm512_set1_epi16,
        setzero = _mm512_setzero_si512,
        max = _mm512_max_epi16,
        min = _mm512_min_epi16,
        slli = _mm512_slli_epi16,
        mulhi = _mm512_mulhi_epi16,
        packus = _mm512_packus_epi16
    );

    // AVX2-optimized activation function.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    impl_base_input_apply_activation!(
        apply_activation_avx2,
        "avx2",
        __m256i,
        set1 = _mm256_set1_epi16,
        setzero = _mm256_setzero_si256,
        max = _mm256_max_epi16,
        min = _mm256_min_epi16,
        slli = _mm256_slli_epi16,
        mulhi = _mm256_mulhi_epi16,
        packus = _mm256_packus_epi16
    );

    /// Fallback scalar implementation.
    #[allow(dead_code)]
    fn forward_fallback(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        let mut acc: Align64<[i16; HIDDEN_DIMS]> = clone_biases(&self.biases);

        accumulate_scalar::<HIDDEN_DIMS>(pattern_feature, &self.weights, &mut acc);
        apply_base_activation_scalar::<OUTPUT_DIMS, HIDDEN_DIMS>(&acc, output);
    }
}
