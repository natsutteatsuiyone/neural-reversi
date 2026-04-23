//! Base input layer for neural network evaluation.

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

#[allow(unused_macros)]
macro_rules! impl_base_input_apply_activation {
    (
        $fn_name:ident,
        $target_feature:literal,
        $lane_ty:ty,
        load = $load:path,
        store = $store:path,
        set1 = $set1:path,
        setzero = $setzero:path,
        max = $max:path,
        min = $min:path,
        slli = $slli:path,
        mulhi = $mulhi:path,
        packus = $packus:path
    ) => {
        /// # Safety
        ///
        /// `output` must be aligned to the SIMD lane width (32 bytes for AVX2,
        /// 64 bytes for AVX-512) for aligned SIMD stores.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $target_feature)]
        #[inline]
        fn $fn_name(&self, acc: &[i16; HIDDEN_DIMS], output: &mut [u8]) {
            use std::arch::x86_64::*;
            use std::mem::{align_of, size_of};
            debug_assert!(
                (output.as_ptr() as usize).is_multiple_of(align_of::<$lane_ty>()),
                "output must be {}-byte aligned",
                align_of::<$lane_ty>(),
            );
            unsafe {
                let acc_ptr = acc.as_ptr() as *const $lane_ty;
                let mut output_ptr = output.as_mut_ptr() as *mut $lane_ty;
                const LANES_PER_REG: usize = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = HIDDEN_DIMS / LANES_PER_REG;
                let one = $set1(ACTIVATION_MAX);
                let zero = $setzero();
                let mut in0_ptr = acc_ptr;
                let mut in1_ptr = acc_ptr.add(num_regs / 2);
                let iterations = num_regs / 4;

                for _ in 0..iterations {
                    let in00 = $load(in0_ptr);
                    let in01 = $load(in0_ptr.add(1));
                    in0_ptr = in0_ptr.add(2);
                    let in10 = $load(in1_ptr);
                    let in11 = $load(in1_ptr.add(1));
                    in1_ptr = in1_ptr.add(2);
                    let clamp0a = $max($min(in00, one), zero);
                    let clamp0b = $max($min(in01, one), zero);
                    let sum0a = $slli(clamp0a, 6);
                    let sum0b = $slli(clamp0b, 6);
                    let sum1a = $min(in10, one);
                    let sum1b = $min(in11, one);
                    let pa = $mulhi(sum0a, sum1a);
                    let pb = $mulhi(sum0b, sum1b);
                    $store(output_ptr, $packus(pa, pb));
                    output_ptr = output_ptr.add(1);
                }
            }
        }
    };
}

/// Neural network base input layer.
///
/// Reference: <https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h>
pub struct BaseInput<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize, const HIDDEN_DIMS: usize> {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<const INPUT_DIMS: usize, const OUTPUT_DIMS: usize, const HIDDEN_DIMS: usize>
    BaseInput<INPUT_DIMS, OUTPUT_DIMS, HIDDEN_DIMS>
{
    /// Loads network weights and biases from a binary reader.
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

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[allow(dead_code)]
    fn forward_neon(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        const {
            assert!(HIDDEN_DIMS.is_multiple_of(32));
            assert!(OUTPUT_DIMS * 2 == HIDDEN_DIMS);
        }

        let mut acc: Align64<[i16; HIDDEN_DIMS]> = clone_biases(&self.biases);
        accumulate_neon::<HIDDEN_DIMS>(pattern_feature, &self.weights, &mut acc);
        self.apply_activation_neon(&acc, output);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    impl_base_input_apply_activation!(
        apply_activation_avx512,
        "avx512bw",
        __m512i,
        load = _mm512_load_si512,
        store = _mm512_store_si512,
        set1 = _mm512_set1_epi16,
        setzero = _mm512_setzero_si512,
        max = _mm512_max_epi16,
        min = _mm512_min_epi16,
        slli = _mm512_slli_epi16,
        mulhi = _mm512_mulhi_epi16,
        packus = _mm512_packus_epi16
    );

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    impl_base_input_apply_activation!(
        apply_activation_avx2,
        "avx2",
        __m256i,
        load = _mm256_load_si256,
        store = _mm256_store_si256,
        set1 = _mm256_set1_epi16,
        setzero = _mm256_setzero_si256,
        max = _mm256_max_epi16,
        min = _mm256_min_epi16,
        slli = _mm256_slli_epi16,
        mulhi = _mm256_mulhi_epi16,
        packus = _mm256_packus_epi16
    );

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[inline]
    fn apply_activation_neon(&self, acc: &[i16; HIDDEN_DIMS], output: &mut [u8]) {
        use std::arch::aarch64::*;

        const LANES_PER_REG: usize = 8;
        let num_regs = HIDDEN_DIMS / LANES_PER_REG;
        let iterations = num_regs / 4;

        unsafe {
            let acc_ptr = acc.as_ptr();
            let mut output_ptr = output.as_mut_ptr();
            let one = vdupq_n_s16(ACTIVATION_MAX);
            let zero = vdupq_n_s16(0);
            let mut in0_ptr = acc_ptr;
            let mut in1_ptr = acc_ptr.add((num_regs / 2) * LANES_PER_REG);

            for _ in 0..iterations {
                let in00 = vld1q_s16(in0_ptr);
                let in01 = vld1q_s16(in0_ptr.add(LANES_PER_REG));
                in0_ptr = in0_ptr.add(2 * LANES_PER_REG);
                let in10 = vld1q_s16(in1_ptr);
                let in11 = vld1q_s16(in1_ptr.add(LANES_PER_REG));
                in1_ptr = in1_ptr.add(2 * LANES_PER_REG);

                // SQDMULH computes sat((x*y*2) >> 16), so pre-shift sum0 by 5 (not 6)
                // to match the x86 mulhi(sum0<<6, sum1) semantics. Inputs stay well
                // within saturation bounds (|sum0*sum1*2| < 2^31, |>>16| < 2^15).
                let sum0a = vshlq_n_s16::<5>(vmaxq_s16(vminq_s16(in00, one), zero));
                let sum0b = vshlq_n_s16::<5>(vmaxq_s16(vminq_s16(in01, one), zero));
                let sum1a = vminq_s16(in10, one);
                let sum1b = vminq_s16(in11, one);

                let pa = vqdmulhq_s16(sum0a, sum1a);
                let pb = vqdmulhq_s16(sum0b, sum1b);

                let packed = vcombine_u8(vqmovun_s16(pa), vqmovun_s16(pb));
                vst1q_u8(output_ptr, packed);
                output_ptr = output_ptr.add(2 * LANES_PER_REG);
            }
        }
    }

    /// Fallback scalar implementation.
    #[allow(dead_code)]
    fn forward_fallback(&self, pattern_feature: &PatternFeature, output: &mut [u8]) {
        let mut acc: Align64<[i16; HIDDEN_DIMS]> = clone_biases(&self.biases);

        accumulate_scalar::<HIDDEN_DIMS>(pattern_feature, &self.weights, &mut acc);

        const { assert!(OUTPUT_DIMS * 2 == HIDDEN_DIMS) }
        let (lo, hi) = acc.0.split_at(OUTPUT_DIMS);
        for ((out, &v0), &v1) in output[..OUTPUT_DIMS].iter_mut().zip(lo).zip(hi) {
            let sum0 = v0.clamp(0, ACTIVATION_MAX) as u32;
            let sum1 = v1.clamp(0, ACTIVATION_MAX) as u32;
            *out = ((sum0 * sum1) >> ACTIVATION_SHIFT) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aligned_vec::avec;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use super::super::simd_layout::permute_rows;

    /// Pair-wise multiplication on the natural-layout `acc`: clamp `acc[i]` and
    /// `acc[OUT + i]` to `[0, ACTIVATION_MAX]`, multiply, and shift right by
    /// `ACTIVATION_SHIFT` — the algebraic specification all SIMD paths and the
    /// scalar fallback compute.
    #[allow(dead_code)]
    fn scalar_reference<const HIDDEN: usize, const OUT: usize>(
        acc: &[i16; HIDDEN],
        output: &mut [u8; OUT],
    ) {
        assert_eq!(OUT * 2, HIDDEN);
        let (lo, hi) = acc.split_at(OUT);
        for ((o, &v0), &v1) in output.iter_mut().zip(lo).zip(hi) {
            let sum0 = v0.clamp(0, ACTIVATION_MAX) as u32;
            let sum1 = v1.clamp(0, ACTIVATION_MAX) as u32;
            *o = ((sum0 * sum1) >> ACTIVATION_SHIFT) as u8;
        }
    }

    /// Builds an `acc` covering negatives, zero, mid-range, > ACTIVATION_MAX,
    /// and (for large `HIDDEN`) wraparound values near `i16::MIN` so that the
    /// negative-saturation path of `packus`/`vqmovun_s16` is exercised.
    #[allow(dead_code)]
    fn make_acc<const HIDDEN: usize>(seed: i32) -> Align64<[i16; HIDDEN]> {
        let mut acc = Align64([0i16; HIDDEN]);
        for (i, v) in acc.0.iter_mut().enumerate() {
            *v = ((i as i32) * 91 + seed * 11 - 1500) as i16;
        }
        acc
    }

    /// `apply_activation_*` only reads `acc`; bias/weight contents are irrelevant
    /// for these tests.
    #[allow(dead_code)]
    fn make_layer<const HIDDEN: usize, const OUT: usize>() -> BaseInput<1, OUT, HIDDEN> {
        BaseInput {
            biases: avec![[CACHE_LINE_SIZE]|0i16; HIDDEN],
            weights: avec![[CACHE_LINE_SIZE]|0i16; HIDDEN],
        }
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn apply_activation_neon_matches_scalar() {
        fn run<const HIDDEN: usize, const OUT: usize>(seed: i32) {
            const { assert!(HIDDEN == OUT * 2) };
            let layer: BaseInput<1, OUT, HIDDEN> = make_layer();
            let acc = make_acc::<HIDDEN>(seed);

            let mut neon_out = [0u8; OUT];
            let mut expected = [0u8; OUT];
            // SAFETY: NEON is the aarch64 baseline, asserted by the cfg above.
            unsafe { layer.apply_activation_neon(&acc.0, &mut neon_out) };
            scalar_reference::<HIDDEN, OUT>(&acc.0, &mut expected);
            assert_eq!(neon_out, expected);
        }
        // HIDDEN must be a multiple of 32 so num_regs/4 iterates at least once.
        run::<64, 32>(1);
        run::<128, 64>(7);
        run::<256, 128>(13);
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512bw"),
    ))]
    fn apply_activation_avx2_matches_scalar() {
        fn run<const HIDDEN: usize, const OUT: usize>(seed: i32) {
            const { assert!(HIDDEN == OUT * 2) };
            let layer: BaseInput<1, OUT, HIDDEN> = make_layer();
            let acc_natural = make_acc::<HIDDEN>(seed);

            // The AVX2 path reads `acc` in pre-permuted layout — match the
            // bias/weight permutation that `BaseInput::load` applies, so the
            // SIMD output recovers the natural pairing.
            let mut acc_permuted = acc_natural;
            permute_rows(&mut acc_permuted.0, HIDDEN);

            let mut simd_out = Align64([0u8; OUT]);
            let mut expected = [0u8; OUT];
            // SAFETY: AVX2 is asserted by the cfg above; output is 64-byte aligned.
            unsafe { layer.apply_activation_avx2(&acc_permuted.0, &mut simd_out.0) };
            scalar_reference::<HIDDEN, OUT>(&acc_natural.0, &mut expected);
            assert_eq!(simd_out.0, expected);
        }
        // HIDDEN must be a multiple of 64 (AVX2 forward-pass requirement).
        run::<64, 32>(1);
        run::<128, 64>(7);
        run::<256, 128>(13);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn apply_activation_avx512_matches_scalar() {
        fn run<const HIDDEN: usize, const OUT: usize>(seed: i32) {
            const { assert!(HIDDEN == OUT * 2) };
            let layer: BaseInput<1, OUT, HIDDEN> = make_layer();
            let acc_natural = make_acc::<HIDDEN>(seed);

            // Same pre-permute trick as the AVX2 test, but `permute_rows`
            // automatically picks the AVX-512 ordering when avx512bw is on.
            let mut acc_permuted = acc_natural;
            permute_rows(&mut acc_permuted.0, HIDDEN);

            let mut simd_out = Align64([0u8; OUT]);
            let mut expected = [0u8; OUT];
            // SAFETY: AVX-512BW is asserted by the cfg above; output is 64-byte aligned.
            unsafe { layer.apply_activation_avx512(&acc_permuted.0, &mut simd_out.0) };
            scalar_reference::<HIDDEN, OUT>(&acc_natural.0, &mut expected);
            assert_eq!(simd_out.0, expected);
        }
        // HIDDEN must be a multiple of 128 (AVX-512 forward-pass requirement).
        run::<128, 64>(1);
        run::<256, 128>(7);
        run::<512, 256>(13);
    }
}
