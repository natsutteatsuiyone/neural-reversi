//! Phase-adaptive input layer for neural network evaluation.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
use crate::eval::pattern_feature::PatternFeature;
use crate::eval::util::clone_biases;
use crate::util::align::Align64;

use super::accumulate_scalar;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
use super::accumulate_avx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::accumulate_avx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::accumulate_neon;

const ACTIVATION_MAX: i16 = 255 * 2;
const ACTIVATION_SHIFT: u32 = 10;

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
                let mut acc_ptr = acc.as_ptr() as *const $lane_ty;
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
#[derive(Debug)]
pub struct PhaseAdaptiveInputLayer<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize> {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize>
    PhaseAdaptiveInputLayer<INPUT_DIMS, OUTPUT_DIMS>
{
    /// Loads network weights and biases from a binary reader.
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

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[allow(dead_code)]
    fn forward_neon(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        const {
            assert!(OUTPUT_DIMS.is_multiple_of(16));
        }

        let mut acc: Align64<[i16; OUTPUT_DIMS]> = clone_biases(&self.biases);
        accumulate_neon::<OUTPUT_DIMS>(pattern_feature, &self.weights, &mut acc);
        self.apply_activation_neon(&acc, output);
    }

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

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[inline]
    fn apply_activation_neon(&self, acc: &[i16; OUTPUT_DIMS], output: &mut [u8]) {
        use std::arch::aarch64::*;

        const LANES_PER_REG: usize = 8;
        let num_regs = OUTPUT_DIMS / LANES_PER_REG;
        let iterations = num_regs / 2;

        unsafe {
            let mut acc_ptr = acc.as_ptr();
            let mut output_ptr = output.as_mut_ptr();
            let one = vdupq_n_s16(ACTIVATION_MAX);
            let zero = vdupq_n_s16(0);

            for _ in 0..iterations {
                let a1 = vld1q_s16(acc_ptr);
                let a2 = vld1q_s16(acc_ptr.add(LANES_PER_REG));
                acc_ptr = acc_ptr.add(2 * LANES_PER_REG);

                let b1 = vmaxq_s16(vminq_s16(a1, one), zero);
                let b2 = vmaxq_s16(vminq_s16(a2, one), zero);

                let s1 = vshlq_n_s16::<6>(b1);
                let s2 = vshlq_n_s16::<6>(b2);

                // Emulate mulhi: full 32-bit signed product, take high 16 bits.
                let prod_lo_1 = vmull_s16(vget_low_s16(s1), vget_low_s16(b1));
                let prod_hi_1 = vmull_high_s16(s1, b1);
                let c1 = vcombine_s16(vshrn_n_s32::<16>(prod_lo_1), vshrn_n_s32::<16>(prod_hi_1));

                let prod_lo_2 = vmull_s16(vget_low_s16(s2), vget_low_s16(b2));
                let prod_hi_2 = vmull_high_s16(s2, b2);
                let c2 = vcombine_s16(vshrn_n_s32::<16>(prod_lo_2), vshrn_n_s32::<16>(prod_hi_2));

                let packed = vcombine_u8(vqmovun_s16(c1), vqmovun_s16(c2));
                vst1q_u8(output_ptr, packed);
                output_ptr = output_ptr.add(2 * LANES_PER_REG);
            }
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
pub struct PhaseAdaptiveInput<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize> {
    inputs: Vec<PhaseAdaptiveInputLayer<INPUT_DIMS, OUTPUT_DIMS>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize>
    PhaseAdaptiveInput<INPUT_DIMS, OUTPUT_DIMS>
{
    /// Loads all phase-adaptive input layers from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let inputs = (0..NUM_PA_INPUTS)
            .map(|_| PhaseAdaptiveInputLayer::<INPUT_DIMS, OUTPUT_DIMS>::load(reader))
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

    /// Squared activation on the natural-layout `acc`: clamp `acc[i]` to
    /// `[0, ACTIVATION_MAX]`, square, and shift right by `ACTIVATION_SHIFT` —
    /// the algebraic specification all SIMD paths and the scalar fallback compute.
    #[allow(dead_code)]
    fn scalar_reference<const DIMS: usize>(acc: &[i16; DIMS], output: &mut [u8; DIMS]) {
        for (o, &v) in output.iter_mut().zip(acc.iter()) {
            let clamped = v.clamp(0, ACTIVATION_MAX) as u32;
            *o = ((clamped * clamped) >> ACTIVATION_SHIFT) as u8;
        }
    }

    /// Builds an `acc` covering negatives, zero, mid-range, > ACTIVATION_MAX,
    /// and (for large `DIMS`) wraparound values near `i16::MIN` so the
    /// negative-saturation path of `packus`/`vqmovun_s16` is exercised.
    #[allow(dead_code)]
    fn make_acc<const DIMS: usize>(seed: i32) -> Align64<[i16; DIMS]> {
        let mut acc = Align64([0i16; DIMS]);
        for (i, v) in acc.0.iter_mut().enumerate() {
            *v = ((i as i32) * 83 + seed * 11 - 700) as i16;
        }
        acc
    }

    /// `apply_activation_*` only reads `acc`; bias/weight contents are irrelevant
    /// for these tests.
    #[allow(dead_code)]
    fn make_layer<const DIMS: usize>() -> PhaseAdaptiveInputLayer<1, DIMS> {
        PhaseAdaptiveInputLayer {
            biases: avec![[CACHE_LINE_SIZE]|0i16; DIMS],
            weights: avec![[CACHE_LINE_SIZE]|0i16; DIMS],
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn apply_activation_neon_matches_scalar() {
        fn run<const DIMS: usize>(seed: i32) {
            let layer: PhaseAdaptiveInputLayer<1, DIMS> = make_layer();
            let acc = make_acc::<DIMS>(seed);

            let mut neon_out = [0u8; DIMS];
            let mut expected = [0u8; DIMS];
            // SAFETY: NEON is the aarch64 baseline, asserted by the cfg above.
            unsafe { layer.apply_activation_neon(&acc.0, &mut neon_out) };
            scalar_reference::<DIMS>(&acc.0, &mut expected);
            assert_eq!(neon_out, expected);
        }
        // OUTPUT_DIMS must be a multiple of 16 so num_regs/2 iterates at least once.
        run::<32>(1);
        run::<64>(7);
        run::<128>(13);
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512bw"),
    ))]
    fn apply_activation_avx2_matches_scalar() {
        fn run<const DIMS: usize>(seed: i32) {
            let layer: PhaseAdaptiveInputLayer<1, DIMS> = make_layer();
            let acc_natural = make_acc::<DIMS>(seed);

            // The AVX2 path reads `acc` in pre-permuted layout — match the
            // bias/weight permutation that `PhaseAdaptiveInputLayer::load` applies,
            // so the SIMD output recovers the natural ordering.
            let mut acc_permuted = acc_natural;
            permute_rows(&mut acc_permuted.0, DIMS);

            let mut simd_out = Align64([0u8; DIMS]);
            let mut expected = [0u8; DIMS];
            // SAFETY: AVX2 is asserted by the cfg above; output is 64-byte aligned.
            unsafe { layer.apply_activation_avx2(&acc_permuted.0, &mut simd_out.0) };
            scalar_reference::<DIMS>(&acc_natural.0, &mut expected);
            assert_eq!(simd_out.0, expected);
        }
        // OUTPUT_DIMS must be a multiple of 64 (AVX2 forward-pass requirement).
        run::<64>(1);
        run::<128>(7);
        run::<256>(13);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn apply_activation_avx512_matches_scalar() {
        fn run<const DIMS: usize>(seed: i32) {
            let layer: PhaseAdaptiveInputLayer<1, DIMS> = make_layer();
            let acc_natural = make_acc::<DIMS>(seed);

            // Same pre-permute trick as the AVX2 test, but `permute_rows`
            // automatically picks the AVX-512 ordering when avx512bw is on.
            let mut acc_permuted = acc_natural;
            permute_rows(&mut acc_permuted.0, DIMS);

            let mut simd_out = Align64([0u8; DIMS]);
            let mut expected = [0u8; DIMS];
            // SAFETY: AVX-512BW is asserted by the cfg above; output is 64-byte aligned.
            unsafe { layer.apply_activation_avx512(&acc_permuted.0, &mut simd_out.0) };
            scalar_reference::<DIMS>(&acc_natural.0, &mut expected);
            assert_eq!(simd_out.0, expected);
        }
        // OUTPUT_DIMS must be a multiple of 128 (AVX-512 forward-pass requirement).
        run::<128>(1);
        run::<256>(7);
        run::<512>(13);
    }
}
