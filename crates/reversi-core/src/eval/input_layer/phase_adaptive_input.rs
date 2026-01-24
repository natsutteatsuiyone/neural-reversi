//! Phase-adaptive input layer for neural network evaluation.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};
use cfg_if::cfg_if;

use crate::constants::CACHE_LINE_SIZE;
use crate::eval::pattern_feature::PatternFeature;
use crate::eval::util::clone_biases;
use crate::util::align::Align64;

use super::{accumulate_scalar, apply_phase_activation_scalar};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use super::accumulate_avx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::{ACTIVATION_MAX, accumulate_avx2};

const NUM_PA_INPUTS: usize = 6;
const PA_INPUT_BUCKET_SIZE: usize = 60 / NUM_PA_INPUTS;

#[allow(unused_macros)]
macro_rules! impl_phase_input_apply_activation {
    (
        $fn_name:ident,
        $target_feature:literal,
        $lane_ty:ty,
        load = $load:path,
        store = $store:path,
        set1_epi16 = $set1:path,
        setzero = $setzero:path,
        min_epi16 = $min:path,
        max_epi16 = $max:path,
        slli_epi16 = $slli:path,
        mulhi_epu16 = $mulhi:path,
        packus_epi16 = $packus:path
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $target_feature)]
        #[inline]
        fn $fn_name(&self, acc: &[i16; OUTPUT_DIMS], output: &mut [u8]) {
            use std::arch::x86_64::*;
            use std::mem::size_of;
            unsafe {
                let mut output_ptr = output.as_mut_ptr() as *mut $lane_ty;
                let mut acc_ptr = acc.as_ptr() as *mut $lane_ty;
                const LANES_PER_REG: usize = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = OUTPUT_DIMS / LANES_PER_REG;
                let one = $set1(ACTIVATION_MAX);
                let zero = $setzero();
                let iterations = num_regs / 2;

                for _ in 0..iterations {
                    let a1 = $load(acc_ptr);
                    let a2 = $load(acc_ptr.add(1));

                    let b1 = $max($min(a1, one), zero);
                    let b2 = $max($min(a2, one), zero);

                    let c1 = $mulhi($slli(b1, 6), b1);
                    let c2 = $mulhi($slli(b2, 6), b2);

                    $store(output_ptr, $packus(c1, c2));

                    acc_ptr = acc_ptr.add(2);
                    output_ptr = output_ptr.add(1);
                }
            }
        }
    };
}

/// Phase-adaptive input layer.
///
/// # Type Parameters
///
/// * `INPUT_DIMS` - Number of input features (sparse).
/// * `OUTPUT_DIMS` - Number of output dimensions (dense).
#[derive(Debug)]
pub struct PhaseAdaptiveInputLayer<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize> {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize>
    PhaseAdaptiveInputLayer<INPUT_DIMS, OUTPUT_DIMS>
{
    /// Loads network weights and biases from a binary reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Binary data reader containing network parameters.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_DIMS * OUTPUT_DIMS];

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

    /// Performs forward pass through the phase-adaptive input layer.
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
            assert!(OUTPUT_DIMS.is_multiple_of(128));
        }

        let mut acc: Align64<[i16; OUTPUT_DIMS]> = clone_biases(&self.biases);
        accumulate_avx512::<OUTPUT_DIMS>(pattern_feature, &self.weights, &mut acc);
        self.apply_activation_avx512(&acc, output);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        const {
            assert!(OUTPUT_DIMS.is_multiple_of(64));
        }

        let mut acc: Align64<[i16; OUTPUT_DIMS]> = clone_biases(&self.biases);
        accumulate_avx2::<OUTPUT_DIMS>(pattern_feature, &self.weights, &mut acc);
        self.apply_activation_avx2(&acc, output);
    }

    // AVX-512-optimized activation function.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    impl_phase_input_apply_activation!(
        apply_activation_avx512,
        "avx512bw",
        __m512i,
        load = _mm512_load_si512,
        store = _mm512_store_si512,
        set1_epi16 = _mm512_set1_epi16,
        setzero = _mm512_setzero_si512,
        min_epi16 = _mm512_min_epi16,
        max_epi16 = _mm512_max_epi16,
        slli_epi16 = _mm512_slli_epi16,
        mulhi_epu16 = _mm512_mulhi_epu16,
        packus_epi16 = _mm512_packus_epi16
    );

    // AVX2-optimized activation function.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    impl_phase_input_apply_activation!(
        apply_activation_avx2,
        "avx2",
        __m256i,
        load = _mm256_load_si256,
        store = _mm256_store_si256,
        set1_epi16 = _mm256_set1_epi16,
        setzero = _mm256_setzero_si256,
        min_epi16 = _mm256_min_epi16,
        max_epi16 = _mm256_max_epi16,
        slli_epi16 = _mm256_slli_epi16,
        mulhi_epu16 = _mm256_mulhi_epu16,
        packus_epi16 = _mm256_packus_epi16
    );

    /// Fallback scalar implementation.
    #[allow(dead_code)]
    fn forward_fallback(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        let mut acc: Align64<[i16; OUTPUT_DIMS]> = clone_biases(&self.biases);
        accumulate_scalar::<OUTPUT_DIMS>(pattern_feature, &self.weights, &mut acc);
        apply_phase_activation_scalar::<OUTPUT_DIMS>(&acc, output);
    }
}

/// A set of phase-adaptive input layers for different game phases.
///
/// This struct encapsulates the phase selection logic, choosing the appropriate
/// input layer based on the current ply.
///
/// # Type Parameters
///
/// * `INPUT_DIMS` - Number of input features (sparse).
/// * `OUTPUT_DIMS` - Number of output dimensions (dense).
#[derive(Debug)]
pub struct PhaseAdaptiveInput<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize> {
    inputs: Vec<PhaseAdaptiveInputLayer<INPUT_DIMS, OUTPUT_DIMS>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize>
    PhaseAdaptiveInput<INPUT_DIMS, OUTPUT_DIMS>
{
    /// Loads all phase-adaptive input layers from a binary reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Binary data reader containing network parameters.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let inputs = (0..NUM_PA_INPUTS)
            .map(|_| PhaseAdaptiveInputLayer::<INPUT_DIMS, OUTPUT_DIMS>::load(reader))
            .collect::<io::Result<Vec<_>>>()?;
        Ok(PhaseAdaptiveInput { inputs })
    }

    /// Performs forward pass through the appropriate phase-adaptive input layer.
    ///
    /// Selects the input layer based on the current ply and performs the forward pass.
    ///
    /// # Arguments
    ///
    /// * `pattern_feature` - Sparse feature values encoded by pattern index.
    /// * `ply` - Current game ply (must be < 60).
    /// * `output` - Output buffer to write results.
    #[inline(always)]
    pub fn forward(&self, pattern_feature: &PatternFeature, ply: usize, output: &mut [u8]) {
        debug_assert!(ply < 60, "ply {} out of valid range 0-59", ply);
        let pa_index = ply / PA_INPUT_BUCKET_SIZE;
        let pa_input = &self.inputs[pa_index];
        pa_input.forward(pattern_feature, output);
    }
}
