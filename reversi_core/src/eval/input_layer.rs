//! Input layers for the neural evaluation network.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};
use cfg_if::cfg_if;

use crate::constants::CACHE_LINE_SIZE;
use crate::eval::pattern_feature::{NUM_FEATURES, PatternFeature};
use crate::eval::util::{clone_biases, feature_offset};
use crate::util::align::Align64;

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
        fn $fn_name(&self, acc: &[i16; HIDDEN_DIMS], output: &mut [u8]) {
            use std::arch::x86_64::*;
            use std::mem::size_of;
            unsafe {
                let acc_ptr = acc.as_ptr() as *mut $lane_ty;
                let mut output_ptr = output.as_mut_ptr() as *mut $lane_ty;
                const LANES_PER_REG: usize = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = HIDDEN_DIMS / LANES_PER_REG;
                let one = $set1(255 * 2);
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
        fn $fn_name(&self, acc: &[i16; OUTPUT_DIMS], output: &mut [u8]) {
            use std::arch::x86_64::*;
            use std::mem::size_of;
            unsafe {
                let mut output_ptr = output.as_mut_ptr() as *mut $lane_ty;
                let mut acc_ptr = acc.as_ptr() as *mut $lane_ty;
                const LANES_PER_REG: usize = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = OUTPUT_DIMS / LANES_PER_REG;
                let one = $set1(255 * 2);
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

/// Neural network base input layer
///
/// Reference: https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h
///
/// # Type Parameters
/// - `INPUT_DIMS`: Number of input features (sparse)
/// - `OUTPUT_DIMS`: Number of output dimensions (dense)
/// - `HIDDEN_DIMS`: Number of hidden units (must be 2 * OUTPUT_DIMS)
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
    /// * `pattern_feature` - Sparse feature values encoded by pattern index
    /// * `output` - Output buffer to write results (length must be OUTPUT_DIMS)
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
        let mut acc = [0; HIDDEN_DIMS];

        accumulate_scalar::<HIDDEN_DIMS>(pattern_feature, &self.weights, &mut acc);

        for i in 0..OUTPUT_DIMS {
            let sum0 = acc[i] + self.biases[i];
            let sum1 = acc[i + OUTPUT_DIMS] + self.biases[i + OUTPUT_DIMS];
            let sum0 = sum0.clamp(0, 255 * 2) as u32;
            let sum1 = sum1.clamp(0, 255 * 2) as u32;
            output[i] = ((sum0 * sum1) / 1024) as u8;
        }
    }
}

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
    /// * `pattern_feature` - Sparse feature values encoded by pattern index
    /// * `output` - Output buffer to write results (length must be OUTPUT_DIMS)
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
        self.apply_activation_avx512(&*acc, output);
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
        self.apply_activation_avx2(&*acc, output);
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
        for i in 0..OUTPUT_DIMS {
            let v = acc[i] as i32;
            let v = v.clamp(0, 255 * 2).pow(2) / 1024;
            output[i] = v as u8;
        }
    }
}

#[allow(unused_macros)]
macro_rules! impl_accumulate {
    (
        $fn_name:ident,
        $target_feature:literal,
        $lane_ty:ty,
        load_u = $loadu:path,
        load = $load:path,
        store = $store:path,
        add = $add:path
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $target_feature)]
        fn $fn_name<const DIMS: usize>(
            pattern_feature: &PatternFeature,
            weights: &[i16],
            acc: &mut [i16; DIMS],
        ) {
            use std::arch::x86_64::*;

            unsafe {
                let acc_ptr = acc.as_mut_ptr() as *mut $lane_ty;
                let lanes_per_reg = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = DIMS / lanes_per_reg;
                let weights_ptr = weights.as_ptr();

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

                    let fw0 = weights_ptr.add(idx0 * DIMS) as *const $lane_ty;
                    let fw1 = weights_ptr.add(idx1 * DIMS) as *const $lane_ty;
                    let fw2 = weights_ptr.add(idx2 * DIMS) as *const $lane_ty;
                    let fw3 = weights_ptr.add(idx3 * DIMS) as *const $lane_ty;
                    let fw4 = weights_ptr.add(idx4 * DIMS) as *const $lane_ty;
                    let fw5 = weights_ptr.add(idx5 * DIMS) as *const $lane_ty;
                    let fw6 = weights_ptr.add(idx6 * DIMS) as *const $lane_ty;
                    let fw7 = weights_ptr.add(idx7 * DIMS) as *const $lane_ty;

                    for j in 0..num_regs {
                        let w0 = $loadu(fw0.add(j));
                        let w1 = $loadu(fw1.add(j));
                        let w2 = $loadu(fw2.add(j));
                        let w3 = $loadu(fw3.add(j));
                        let w4 = $loadu(fw4.add(j));
                        let w5 = $loadu(fw5.add(j));
                        let w6 = $loadu(fw6.add(j));
                        let w7 = $loadu(fw7.add(j));

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

                    let fw0 = weights_ptr.add(idx0 * DIMS) as *const $lane_ty;
                    let fw1 = weights_ptr.add(idx1 * DIMS) as *const $lane_ty;
                    let fw2 = weights_ptr.add(idx2 * DIMS) as *const $lane_ty;
                    let fw3 = weights_ptr.add(idx3 * DIMS) as *const $lane_ty;

                    for j in 0..num_regs {
                        let w0 = $loadu(fw0.add(j));
                        let w1 = $loadu(fw1.add(j));
                        let w2 = $loadu(fw2.add(j));
                        let w3 = $loadu(fw3.add(j));

                        let sum = $add($add(w0, w1), $add(w2, w3));

                        let acc_val = $load(acc_ptr.add(j));
                        $store(acc_ptr.add(j), $add(acc_val, sum));
                    }

                    base += 4;
                }

                while base < NUM_FEATURES {
                    let idx = feature_offset(pattern_feature, base);
                    let feature_weights = weights_ptr.add(idx * DIMS) as *const $lane_ty;
                    for j in 0..num_regs {
                        let weight = $loadu(feature_weights.add(j));
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
    load_u = _mm512_loadu_si512,
    load = _mm512_load_si512,
    store = _mm512_store_si512,
    add = _mm512_add_epi16
);

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl_accumulate!(
    accumulate_avx2,
    "avx2",
    __m256i,
    load_u = _mm256_loadu_si256,
    load = _mm256_load_si256,
    store = _mm256_store_si256,
    add = _mm256_add_epi16
);

/// Accumulates the weights of all active sparse features into a dense accumulator.
///
/// This scalar variant is used by the fallback code paths where SIMD is unavailable.
///
/// # Parameters
/// * `pattern_feature` - Sparse feature values indexed by pattern identifier.
/// * `weights` - Flat weight matrix laid out in feature-major order.
/// * `acc` - Dense accumulation buffer that will be incremented in-place.
fn accumulate_scalar<const DIMS: usize>(
    pattern_feature: &PatternFeature,
    weights: &[i16],
    acc: &mut [i16; DIMS],
) {
    for idx in 0..NUM_FEATURES {
        let offset = feature_offset(pattern_feature, idx);
        let weights_row = &weights[offset * DIMS..][..DIMS];
        for (a, w) in acc.iter_mut().zip(weights_row.iter()) {
            *a += *w;
        }
    }
}
