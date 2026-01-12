//! Input layers for the neural evaluation network.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};
use cfg_if::cfg_if;

use crate::constants::CACHE_LINE_SIZE;
use crate::eval::pattern_feature::{NUM_FEATURES, PatternFeature};
use crate::eval::util::{clone_biases, feature_offset};
use crate::util::align::Align64;

const ACTIVATION_MAX: i16 = 255 * 2;
const ACTIVATION_SHIFT: u32 = 10;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod simd_layout {
    use std::mem::size_of;

    use cfg_if::cfg_if;

    const PERMUTE_WIDTH: usize = 8;

    unsafe fn permute_tile<T, const ORDER_LEN: usize>(tile: &mut [T], order: &[usize; ORDER_LEN])
    where
        T: Copy + Default,
    {
        let mut temp = [T::default(); ORDER_LEN];
        for (dst, &src_idx) in temp.iter_mut().zip(order.iter()) {
            *dst = tile[src_idx];
        }
        tile.copy_from_slice(&temp);
    }

    unsafe fn permute_rows_impl<T, const ORDER_LEN: usize>(
        data: &mut [i16],
        row_len: usize,
        order: &[usize; ORDER_LEN],
    ) where
        T: Copy + Default,
    {
        if row_len == 0 {
            return;
        }

        let row_count = data.len() / row_len;

        let tiles_per_row = (row_len * size_of::<i16>()) / size_of::<T>();
        if tiles_per_row == 0 {
            return;
        }

        let total_tiles = tiles_per_row * row_count;
        let tile_ptr = data.as_mut_ptr() as *mut T;
        let tile_slice = unsafe { std::slice::from_raw_parts_mut(tile_ptr, total_tiles) };

        let mut rows = tile_slice.chunks_exact_mut(tiles_per_row);
        for row in &mut rows {
            let mut tiles = row.chunks_exact_mut(ORDER_LEN);
            for tile in &mut tiles {
                unsafe { permute_tile::<T, ORDER_LEN>(tile, order) };
            }
        }
    }

    cfg_if! {
        if #[cfg(target_feature = "avx512bw")] {
            const AVX512_ORDER: [usize; PERMUTE_WIDTH] = [0, 2, 4, 6, 1, 3, 5, 7];
            pub fn permute_rows(data: &mut [i16], row_len: usize) {
                unsafe { permute_rows_impl::<u128, PERMUTE_WIDTH>(data, row_len, &AVX512_ORDER) };
            }
        } else if #[cfg(target_feature = "avx2")]  {
            const AVX2_ORDER: [usize; PERMUTE_WIDTH] = [0, 1, 4, 5, 2, 3, 6, 7];
            pub fn permute_rows(data: &mut [i16], row_len: usize) {
                unsafe { permute_rows_impl::<u64, PERMUTE_WIDTH>(data, row_len, &AVX2_ORDER) };
            }
        }
    }
}

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

        // Permute weights and biases for optimal SIMD access patterns.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // Permute weights and biases for optimal SIMD access patterns.
            use crate::eval::input_layer::simd_layout::permute_rows;
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
                unsafe { self.forward_avx512(pattern_feature, output) }
            } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]  {
                unsafe { self.forward_avx2(pattern_feature, output) }
            } else {
                self.forward_fallback(pattern_feature, output)
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
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

/// Phase-adaptive input layer.
///
/// # Type Parameters
///
/// * `INPUT_DIMS` - Number of input features (sparse).
/// * `OUTPUT_DIMS` - Number of output dimensions (dense).
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
    /// * `reader` - Binary data reader containing network parameters.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_DIMS * OUTPUT_DIMS];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // Permute weights and biases for optimal SIMD access patterns.
            use crate::eval::input_layer::simd_layout::permute_rows;
            permute_rows(biases.as_mut_slice(), OUTPUT_DIMS);
            permute_rows(weights.as_mut_slice(), OUTPUT_DIMS);
        }

        Ok(PhaseAdaptiveInput { biases, weights })
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

#[allow(unused_macros)]
macro_rules! impl_accumulate {
    (
        $fn_name:ident,
        $target_feature:literal,
        $lane_ty:ty,
        load = $load:path,
        store = $store:path,
        add = $add:path
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $target_feature)]
        #[inline]
        fn $fn_name<const DIMS: usize>(
            pattern_feature: &PatternFeature,
            weights: &[i16],
            acc: &mut [i16; DIMS],
        ) {
            use std::arch::x86_64::*;
            use std::mem::size_of;

            unsafe {
                let acc_ptr = acc.as_mut_ptr() as *mut $lane_ty;
                const LANES_PER_REG: usize = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = DIMS / LANES_PER_REG;
                debug_assert!(DIMS.is_multiple_of(LANES_PER_REG));
                let weights_ptr = weights.as_ptr() as *const $lane_ty;

                let mut base = 0;
                while base + 8 <= NUM_FEATURES {
                    let idx0 = feature_offset(pattern_feature, base);
                    let idx1 = feature_offset(pattern_feature, base + 1);
                    let idx2 = feature_offset(pattern_feature, base + 2);
                    let idx3 = feature_offset(pattern_feature, base + 3);
                    let idx4 = feature_offset(pattern_feature, base + 4);
                    let idx5 = feature_offset(pattern_feature, base + 5);
                    let idx6 = feature_offset(pattern_feature, base + 6);
                    let idx7 = feature_offset(pattern_feature, base + 7);

                    let fw0 = weights_ptr.add(idx0 * num_regs);
                    let fw1 = weights_ptr.add(idx1 * num_regs);
                    let fw2 = weights_ptr.add(idx2 * num_regs);
                    let fw3 = weights_ptr.add(idx3 * num_regs);
                    let fw4 = weights_ptr.add(idx4 * num_regs);
                    let fw5 = weights_ptr.add(idx5 * num_regs);
                    let fw6 = weights_ptr.add(idx6 * num_regs);
                    let fw7 = weights_ptr.add(idx7 * num_regs);

                    for j in 0..num_regs {
                        let w0 = $load(fw0.add(j));
                        let w1 = $load(fw1.add(j));
                        let w2 = $load(fw2.add(j));
                        let w3 = $load(fw3.add(j));
                        let w4 = $load(fw4.add(j));
                        let w5 = $load(fw5.add(j));
                        let w6 = $load(fw6.add(j));
                        let w7 = $load(fw7.add(j));

                        let sum01 = $add(w0, w1);
                        let sum23 = $add(w2, w3);
                        let sum45 = $add(w4, w5);
                        let sum67 = $add(w6, w7);

                        let sum0123 = $add(sum01, sum23);
                        let sum4567 = $add(sum45, sum67);

                        let sum_all = $add(sum0123, sum4567);

                        let acc_val = $load(acc_ptr.add(j));
                        $store(acc_ptr.add(j), $add(acc_val, sum_all));
                    }

                    base += 8;
                }

                while base + 4 <= NUM_FEATURES {
                    let idx0 = feature_offset(pattern_feature, base);
                    let idx1 = feature_offset(pattern_feature, base + 1);
                    let idx2 = feature_offset(pattern_feature, base + 2);
                    let idx3 = feature_offset(pattern_feature, base + 3);

                    let fw0 = weights_ptr.add(idx0 * num_regs);
                    let fw1 = weights_ptr.add(idx1 * num_regs);
                    let fw2 = weights_ptr.add(idx2 * num_regs);
                    let fw3 = weights_ptr.add(idx3 * num_regs);

                    for j in 0..num_regs {
                        let w0 = $load(fw0.add(j));
                        let w1 = $load(fw1.add(j));
                        let w2 = $load(fw2.add(j));
                        let w3 = $load(fw3.add(j));

                        let sum = $add($add(w0, w1), $add(w2, w3));

                        let acc_val = $load(acc_ptr.add(j));
                        $store(acc_ptr.add(j), $add(acc_val, sum));
                    }

                    base += 4;
                }

                while base < NUM_FEATURES {
                    let idx = feature_offset(pattern_feature, base);
                    let feature_weights = weights_ptr.add(idx * num_regs);
                    for j in 0..num_regs {
                        let weight = $load(feature_weights.add(j));
                        let acc_val = $load(acc_ptr.add(j));
                        $store(acc_ptr.add(j), $add(acc_val, weight));
                    }
                    base += 1;
                }
            }
        }
    };
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl_accumulate!(
    accumulate_avx512,
    "avx512bw",
    __m512i,
    load = _mm512_load_si512,
    store = _mm512_store_si512,
    add = _mm512_add_epi16
);

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl_accumulate!(
    accumulate_avx2,
    "avx2",
    __m256i,
    load = _mm256_load_si256,
    store = _mm256_store_si256,
    add = _mm256_add_epi16
);

/// Accumulates weights of active sparse features into accumulator (scalar fallback).
///
/// # Arguments
///
/// * `pattern_feature` - Sparse feature values indexed by pattern identifier.
/// * `weights` - Flat weight matrix in feature-major order.
/// * `acc` - Dense accumulation buffer.
#[inline(always)]
fn accumulate_scalar<const DIMS: usize>(
    pattern_feature: &PatternFeature,
    weights: &[i16],
    acc: &mut [i16; DIMS],
) {
    let acc_ptr = acc.as_mut_ptr();
    let weights_ptr = weights.as_ptr();
    for idx in 0..NUM_FEATURES {
        let offset = feature_offset(pattern_feature, idx);
        let row_ptr = unsafe { weights_ptr.add(offset * DIMS) };
        let mut j = 0;
        unsafe {
            while j + 4 <= DIMS {
                let acc_j = acc_ptr.add(j);
                let row_j = row_ptr.add(j);
                *acc_j += *row_j;
                *acc_j.add(1) = *acc_j.add(1) + *row_j.add(1);
                *acc_j.add(2) = *acc_j.add(2) + *row_j.add(2);
                *acc_j.add(3) = *acc_j.add(3) + *row_j.add(3);
                j += 4;
            }

            while j < DIMS {
                let acc_j = acc_ptr.add(j);
                *acc_j += *row_ptr.add(j);
                j += 1;
            }
        }
    }
}

#[inline(always)]
fn clamp_activation(value: i16) -> u16 {
    if value <= 0 {
        0
    } else if value >= ACTIVATION_MAX {
        ACTIVATION_MAX as u16
    } else {
        value as u16
    }
}

#[inline(always)]
fn apply_base_activation_scalar<const OUTPUT_DIMS: usize, const HIDDEN_DIMS: usize>(
    acc: &[i16; HIDDEN_DIMS],
    output: &mut [u8],
) {
    debug_assert!(OUTPUT_DIMS * 2 == HIDDEN_DIMS);
    debug_assert!(output.len() >= OUTPUT_DIMS);
    let acc_ptr = acc.as_ptr();
    let out_ptr = output.as_mut_ptr();
    unsafe {
        for i in 0..OUTPUT_DIMS {
            let sum0 = clamp_activation(*acc_ptr.add(i)) as u32;
            let sum1 = clamp_activation(*acc_ptr.add(i + OUTPUT_DIMS)) as u32;
            *out_ptr.add(i) = ((sum0 * sum1) >> ACTIVATION_SHIFT) as u8;
        }
    }
}

#[inline(always)]
fn apply_phase_activation_scalar<const OUTPUT_DIMS: usize>(
    acc: &[i16; OUTPUT_DIMS],
    output: &mut [u8],
) {
    debug_assert!(output.len() >= OUTPUT_DIMS);
    let acc_ptr = acc.as_ptr();
    let out_ptr = output.as_mut_ptr();
    unsafe {
        for i in 0..OUTPUT_DIMS {
            let v = clamp_activation(*acc_ptr.add(i)) as u32;
            *out_ptr.add(i) = ((v * v) >> ACTIVATION_SHIFT) as u8;
        }
    }
}
