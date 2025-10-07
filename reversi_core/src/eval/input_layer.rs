//! Input layers for the neural evaluation network.
//!
//! Combines base input, phase-adaptive input, and SIMD layout helpers.

use std::io::{self, Read};
use std::mem::zeroed;
use std::ptr::copy_nonoverlapping;

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::CACHE_LINE_SIZE;
use crate::util::align::Align64;

const PREFETCH_DISTANCE: usize = 8;
const AVX512_ACTIVATION_TILE: usize = 128;

#[cfg(target_arch = "x86_64")]
mod simd_layout {
    use std::mem::size_of;

    const PERMUTE_WIDTH: usize = 8;
    const AVX512_ORDER: [usize; PERMUTE_WIDTH] = [0, 2, 4, 6, 1, 3, 5, 7];
    const AVX2_ORDER: [usize; PERMUTE_WIDTH] = [0, 1, 4, 5, 2, 3, 6, 7];

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

    pub fn permute_rows_avx512(data: &mut [i16], row_len: usize) {
        unsafe { permute_rows_impl::<u128, PERMUTE_WIDTH>(data, row_len, &AVX512_ORDER) };
    }

    pub fn permute_rows_avx2(data: &mut [i16], row_len: usize) {
        unsafe { permute_rows_impl::<u64, PERMUTE_WIDTH>(data, row_len, &AVX2_ORDER) };
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod simd_layout {
    pub fn permute_rows_avx512(_data: &mut [i16], _row_len: usize) {}

    pub fn permute_rows_avx2(_data: &mut [i16], _row_len: usize) {}
}

use self::simd_layout::{permute_rows_avx2, permute_rows_avx512};

/// Neural network base input layer
///
/// Reference: https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/nnue_feature_transformer.h
///
/// # Type Parameters
/// - `INPUT_DIMS`: Number of input features (sparse)
/// - `OUTPUT_DIMS`: Number of output dimensions (dense)
/// - `HIDDEN_DIMS`: Number of hidden units (must be 2 * OUTPUT_DIMS)
#[derive(Debug)]
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

        for i in 0..HIDDEN_DIMS {
            biases[i] *= 2;
        }

        for i in 0..INPUT_DIMS * HIDDEN_DIMS {
            weights[i] *= 2;
        }

        #[cfg(target_arch = "x86_64")]
        {
            // Permute weights and biases for optimal SIMD access patterns.
            if cfg!(target_feature = "avx512bw")
                && HIDDEN_DIMS.is_multiple_of(AVX512_ACTIVATION_TILE)
            {
                permute_rows_avx512(biases.as_mut_slice(), HIDDEN_DIMS);
                permute_rows_avx512(weights.as_mut_slice(), HIDDEN_DIMS);
            } else if cfg!(target_feature = "avx2") {
                permute_rows_avx2(biases.as_mut_slice(), HIDDEN_DIMS);
                permute_rows_avx2(weights.as_mut_slice(), HIDDEN_DIMS);
            }
        }

        Ok(BaseInput { biases, weights })
    }

    /// Performs forward pass through the base input layer.
    ///
    /// # Arguments
    /// * `feature_indices` - Sparse indices of active features to accumulate
    /// * `output` - Output buffer to write results (length must be OUTPUT_DIMS)
    #[inline(always)]
    pub fn forward(&self, feature_indices: &[usize], output: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx512bw") {
                unsafe { self.forward_avx512(feature_indices, output) };
                return;
            } else if cfg!(target_feature = "avx2") {
                unsafe { self.forward_avx2(feature_indices, output) };
                return;
            }
        }

        self.forward_fallback(feature_indices, output)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512(&self, feature_indices: &[usize], output: &mut [u8]) {
        const {
            assert!(
                HIDDEN_DIMS.is_multiple_of(64),
                "HIDDEN_DIMS must be a multiple of 64"
            );
            assert!(
                OUTPUT_DIMS * 2 == HIDDEN_DIMS,
                "OUTPUT_DIMS must be half of HIDDEN_DIMS"
            );
        }

        unsafe {
            use std::arch::x86_64::*;

            let mut acc: Align64<[i16; HIDDEN_DIMS]> = zeroed();
            let acc_ptr = acc.as_mut_ptr() as *mut __m512i;
            let num_regs = HIDDEN_DIMS / (512 / 16);

            copy_nonoverlapping(self.biases.as_ptr() as *const __m512i, acc_ptr, num_regs);
            accumulate_sparse_avx512::<HIDDEN_DIMS>(feature_indices, &self.weights, &mut acc);

            if HIDDEN_DIMS.is_multiple_of(AVX512_ACTIVATION_TILE) {
                let output_ptr = output.as_mut_ptr() as *mut __m512i;
                let one = _mm512_set1_epi16(127 * 2);
                let zero = _mm512_setzero_si512();
                let in0_ptr = acc_ptr;
                let in1_ptr = acc_ptr.add(num_regs / 2);
                for j in 0..(num_regs / 4) {
                    let in00 = *in0_ptr.add(j * 2);
                    let in01 = *in0_ptr.add(j * 2 + 1);
                    let in10 = *in1_ptr.add(j * 2);
                    let in11 = *in1_ptr.add(j * 2 + 1);
                    let clamp0a = _mm512_max_epi16(_mm512_min_epi16(in00, one), zero);
                    let clamp0b = _mm512_max_epi16(_mm512_min_epi16(in01, one), zero);
                    let sum0a = _mm512_slli_epi16(clamp0a, 7);
                    let sum0b = _mm512_slli_epi16(clamp0b, 7);
                    let sum1a = _mm512_min_epi16(in10, one);
                    let sum1b = _mm512_min_epi16(in11, one);
                    let pa = _mm512_mulhi_epi16(sum0a, sum1a);
                    let pb = _mm512_mulhi_epi16(sum0b, sum1b);
                    *output_ptr.add(j) = _mm512_packus_epi16(pa, pb);
                }
            } else {
                self.apply_activation_avx2(&acc, output);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn forward_avx2(&self, feature_indices: &[usize], output: &mut [u8]) {
        const {
            assert!(
                HIDDEN_DIMS.is_multiple_of(64),
                "HIDDEN_DIMS must be a multiple of 64"
            );
            assert!(
                OUTPUT_DIMS * 2 == HIDDEN_DIMS,
                "OUTPUT_DIMS must be half of HIDDEN_DIMS"
            );
        }

        unsafe {
            use std::arch::x86_64::*;

            let mut acc: Align64<[i16; HIDDEN_DIMS]> = zeroed();
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
            let num_regs = HIDDEN_DIMS / (256 / 16);

            copy_nonoverlapping(self.biases.as_ptr() as *const __m256i, acc_ptr, num_regs);
            accumulate_sparse_avx2::<HIDDEN_DIMS>(feature_indices, &self.weights, &mut acc);
            self.apply_activation_avx2(&acc, output);
        }
    }

    /// AVX2-optimized activation function.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn apply_activation_avx2(&self, acc: &[i16; HIDDEN_DIMS], output: &mut [u8]) {
        unsafe {
            use std::arch::x86_64::*;

            let acc_ptr = acc.as_ptr() as *mut __m256i;
            let output_ptr = output.as_mut_ptr() as *mut __m256i;
            let num_regs = HIDDEN_DIMS / (256 / 16);
            let one = _mm256_set1_epi16(127 * 2);
            let zero = _mm256_setzero_si256();
            let in0_ptr = acc_ptr;
            let in1_ptr = acc_ptr.add(num_regs / 2);
            for j in 0..(num_regs / 4) {
                let in00 = *in0_ptr.add(j * 2);
                let in01 = *in0_ptr.add(j * 2 + 1);
                let in10 = *in1_ptr.add(j * 2);
                let in11 = *in1_ptr.add(j * 2 + 1);
                let clamp0a = _mm256_max_epi16(_mm256_min_epi16(in00, one), zero);
                let clamp0b = _mm256_max_epi16(_mm256_min_epi16(in01, one), zero);
                let sum0a = _mm256_slli_epi16(clamp0a, 7);
                let sum0b = _mm256_slli_epi16(clamp0b, 7);
                let sum1a = _mm256_min_epi16(in10, one);
                let sum1b = _mm256_min_epi16(in11, one);
                let pa = _mm256_mulhi_epi16(sum0a, sum1a);
                let pb = _mm256_mulhi_epi16(sum0b, sum1b);
                *output_ptr.add(j) = _mm256_packus_epi16(pa, pb);
            }
        }
    }

    /// Fallback scalar implementation.
    fn forward_fallback(&self, feature_indices: &[usize], output: &mut [u8]) {
        let mut acc = [0; HIDDEN_DIMS];

        accumulate_sparse_scalar::<HIDDEN_DIMS>(feature_indices, &self.weights, &mut acc);

        for i in 0..OUTPUT_DIMS {
            let sum0 = acc[i] + self.biases[i];
            let sum1 = acc[i + OUTPUT_DIMS] + self.biases[i + OUTPUT_DIMS];
            let sum0 = sum0.clamp(0, 127 * 2) as u32;
            let sum1 = sum1.clamp(0, 127 * 2) as u32;
            output[i] = ((sum0 * sum1) / 512) as u8;
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

        #[cfg(target_arch = "x86_64")]
        {
            // Permute weights and biases for optimal SIMD access patterns.
            if cfg!(target_feature = "avx512bw")
                && OUTPUT_DIMS.is_multiple_of(AVX512_ACTIVATION_TILE)
            {
                permute_rows_avx512(biases.as_mut_slice(), OUTPUT_DIMS);
                permute_rows_avx512(weights.as_mut_slice(), OUTPUT_DIMS);
            } else if cfg!(target_feature = "avx2") {
                permute_rows_avx2(biases.as_mut_slice(), OUTPUT_DIMS);
                permute_rows_avx2(weights.as_mut_slice(), OUTPUT_DIMS);
            }
        }

        Ok(PhaseAdaptiveInput { biases, weights })
    }

    /// Performs forward pass through the phase-adaptive input layer.
    ///
    /// # Arguments
    /// * `feature_indices` - Sparse indices of active features to accumulate
    /// * `output` - Output buffer to write results (length must be OUTPUT_DIMS)
    #[inline(always)]
    pub fn forward(&self, feature_indices: &[usize], output: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx512bw") {
                unsafe { self.forward_avx512(feature_indices, output) };
                return;
            } else if cfg!(target_feature = "avx2") {
                unsafe { self.forward_avx2(feature_indices, output) };
                return;
            }
        }

        self.forward_fallback(feature_indices, output);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512(&self, feature_indices: &[usize], output: &mut [u8]) {
        const {
            assert!(
                OUTPUT_DIMS.is_multiple_of(32),
                "HIDDEN_DIMS must be a multiple of 32"
            );
        }

        unsafe {
            use std::arch::x86_64::*;

            let mut acc: Align64<[i16; OUTPUT_DIMS]> = zeroed();
            let acc_ptr = acc.as_mut_ptr() as *mut __m512i;
            let num_regs = OUTPUT_DIMS / (512 / 16);

            copy_nonoverlapping(self.biases.as_ptr() as *const __m512i, acc_ptr, num_regs);
            accumulate_sparse_avx512::<OUTPUT_DIMS>(feature_indices, &self.weights, &mut acc);

            if OUTPUT_DIMS.is_multiple_of(AVX512_ACTIVATION_TILE) {
                let mut output_ptr = output.as_mut_ptr() as *mut __m512i;
                let bias8 = _mm512_set1_epi8(16);
                let zero8 = _mm512_set1_epi8(0);

                let mut acc_ptr_base = acc_ptr;
                let iterations = num_regs / 2;
                for _ in 0..iterations {
                    let a = _mm512_load_si512(acc_ptr_base);
                    let b = _mm512_load_si512(acc_ptr_base.add(1));
                    acc_ptr_base = acc_ptr_base.add(2);

                    let sa = _mm512_max_epi16(a, _mm512_srai_epi16(a, 3));
                    let sb = _mm512_max_epi16(b, _mm512_srai_epi16(b, 3));
                    let packed = _mm512_packs_epi16(sa, sb);
                    let added = _mm512_adds_epi8(packed, bias8);
                    let result = _mm512_max_epi8(added, zero8);
                    _mm512_store_si512(output_ptr, result);
                    output_ptr = output_ptr.add(1);
                }
            } else {
                self.apply_activation_avx2(&acc, output);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn forward_avx2(&self, feature_indices: &[usize], output: &mut [u8]) {
        const {
            assert!(
                OUTPUT_DIMS.is_multiple_of(32),
                "HIDDEN_DIMS must be a multiple of 32"
            );
        }

        unsafe {
            use std::arch::x86_64::*;

            let mut acc: Align64<[i16; OUTPUT_DIMS]> = zeroed();
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
            let num_regs = OUTPUT_DIMS / (256 / 16);

            copy_nonoverlapping(self.biases.as_ptr() as *const __m256i, acc_ptr, num_regs);
            accumulate_sparse_avx2::<OUTPUT_DIMS>(feature_indices, &self.weights, &mut acc);
            self.apply_activation_avx2(&acc, output);
        }
    }

    /// AVX2 activation application helper.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn apply_activation_avx2(&self, acc: &[i16; OUTPUT_DIMS], output: &mut [u8]) {
        unsafe {
            use std::arch::x86_64::*;

            let mut output_ptr = output.as_mut_ptr() as *mut __m256i;
            let bias8 = _mm256_set1_epi8(16);
            let zero8 = _mm256_set1_epi8(0);
            let mut acc_ptr = acc.as_ptr() as *mut __m256i;
            let num_regs = OUTPUT_DIMS / (256 / 16);
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

        accumulate_sparse_scalar::<OUTPUT_DIMS>(feature_indices, &self.weights, &mut acc);

        for i in 0..OUTPUT_DIMS {
            let v = acc[i] + self.biases[i];
            let v = if v >= 0 {
                v.min(111)
            } else {
                (v >> 3).max(-16)
            };

            output[i] = (v + 16) as u8;
        }
    }
}

/// Accumulates sparse features into a dense accumulator using AVX-512 instructions.
///
/// # Parameters
/// * `feature_indices` - Sparse list of active features to incorporate.
/// * `weights` - Flat weight matrix stored in feature-major order.
/// * `acc` - Dense accumulator buffer updated in-place via packed `__m512i` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
fn accumulate_sparse_avx512<const DIMS: usize>(
    feature_indices: &[usize],
    weights: &[i16],
    acc: &mut [i16; DIMS],
) {
    use std::arch::x86_64::*;

    unsafe {
        let acc_ptr = acc.as_mut_ptr() as *mut __m512i;
        let num_regs = DIMS / (512 / 16);
        let weights_ptr = weights.as_ptr();

        let len = feature_indices.len();

        // Prefetch initial features
        for i in 0..PREFETCH_DISTANCE.min(len) {
            let idx = *feature_indices.get_unchecked(i);
            let prefetch_ptr = weights_ptr.add(idx * DIMS) as *const i8;
            _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
            _mm_prefetch(prefetch_ptr.add(64), _MM_HINT_T0);
        }

        // Process in chunks of 4 with prefetching
        let mut chunks = feature_indices.chunks_exact(4);
        let mut chunk_idx = 0;

        for chunk in &mut chunks {
            // Prefetch future features
            let prefetch_idx = chunk_idx + PREFETCH_DISTANCE;
            if prefetch_idx < len {
                let idx = *feature_indices.get_unchecked(prefetch_idx);
                let prefetch_ptr = weights_ptr.add(idx * DIMS) as *const i8;
                _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
                _mm_prefetch(prefetch_ptr.add(64), _MM_HINT_T0);
            }

            let idx0 = *chunk.get_unchecked(0);
            let idx1 = *chunk.get_unchecked(1);
            let idx2 = *chunk.get_unchecked(2);
            let idx3 = *chunk.get_unchecked(3);

            let fw0 = weights_ptr.add(idx0 * DIMS) as *const __m512i;
            let fw1 = weights_ptr.add(idx1 * DIMS) as *const __m512i;
            let fw2 = weights_ptr.add(idx2 * DIMS) as *const __m512i;
            let fw3 = weights_ptr.add(idx3 * DIMS) as *const __m512i;

            for j in 0..num_regs {
                let w0 = _mm512_loadu_si512(fw0.add(j));
                let w1 = _mm512_loadu_si512(fw1.add(j));
                let w2 = _mm512_loadu_si512(fw2.add(j));
                let w3 = _mm512_loadu_si512(fw3.add(j));

                let sum01 = _mm512_add_epi16(w0, w1);
                let sum23 = _mm512_add_epi16(w2, w3);
                let sum = _mm512_add_epi16(sum01, sum23);

                let acc_val = _mm512_load_si512(acc_ptr.add(j));
                _mm512_store_si512(acc_ptr.add(j), _mm512_add_epi16(acc_val, sum));
            }

            chunk_idx += 4;
        }

        // Handle remainder with prefetching
        for (i, &idx) in chunks.remainder().iter().enumerate() {
            let prefetch_idx = chunk_idx + i + PREFETCH_DISTANCE;
            if prefetch_idx < len {
                let pidx = *feature_indices.get_unchecked(prefetch_idx);
                let prefetch_ptr = weights_ptr.add(pidx * DIMS) as *const i8;
                _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
            }

            let feature_weights = weights_ptr.add(idx * DIMS) as *const __m512i;
            for j in 0..num_regs {
                let weight = _mm512_loadu_si512(feature_weights.add(j));
                let acc_val = _mm512_load_si512(acc_ptr.add(j));
                _mm512_store_si512(acc_ptr.add(j), _mm512_add_epi16(acc_val, weight));
            }
        }
    }
}

/// Accumulates sparse features into a dense accumulator using AVX2 instructions.
///
/// # Parameters
/// * `feature_indices` - Sparse list of active features to incorporate.
/// * `weights` - Flat weight matrix stored in feature-major order.
/// * `acc` - Dense accumulator buffer updated in-place via packed `__m256i` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn accumulate_sparse_avx2<const DIMS: usize>(
    feature_indices: &[usize],
    weights: &[i16],
    acc: &mut [i16; DIMS],
) {
    use std::arch::x86_64::*;

    unsafe {
        let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
        let num_regs = DIMS / (256 / 16);
        let weights_ptr = weights.as_ptr();

        let len = feature_indices.len();

        // Prefetch initial features
        for i in 0..PREFETCH_DISTANCE.min(len) {
            let idx = *feature_indices.get_unchecked(i);
            let prefetch_ptr = weights_ptr.add(idx * DIMS) as *const i8;
            _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
            _mm_prefetch(prefetch_ptr.add(64), _MM_HINT_T0);
        }

        // Process in chunks of 4 with prefetching
        let mut chunks = feature_indices.chunks_exact(4);
        let mut chunk_idx = 0;

        for chunk in &mut chunks {
            // Prefetch future features
            let prefetch_idx = chunk_idx + PREFETCH_DISTANCE;
            if prefetch_idx < len {
                let idx = *feature_indices.get_unchecked(prefetch_idx);
                let prefetch_ptr = weights_ptr.add(idx * DIMS) as *const i8;
                _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
                _mm_prefetch(prefetch_ptr.add(64), _MM_HINT_T0);
            }

            let idx0 = *chunk.get_unchecked(0);
            let idx1 = *chunk.get_unchecked(1);
            let idx2 = *chunk.get_unchecked(2);
            let idx3 = *chunk.get_unchecked(3);

            let fw0 = weights_ptr.add(idx0 * DIMS) as *const __m256i;
            let fw1 = weights_ptr.add(idx1 * DIMS) as *const __m256i;
            let fw2 = weights_ptr.add(idx2 * DIMS) as *const __m256i;
            let fw3 = weights_ptr.add(idx3 * DIMS) as *const __m256i;

            for j in 0..num_regs {
                let w0 = _mm256_loadu_si256(fw0.add(j));
                let w1 = _mm256_loadu_si256(fw1.add(j));
                let w2 = _mm256_loadu_si256(fw2.add(j));
                let w3 = _mm256_loadu_si256(fw3.add(j));

                let sum01 = _mm256_add_epi16(w0, w1);
                let sum23 = _mm256_add_epi16(w2, w3);
                let sum = _mm256_add_epi16(sum01, sum23);

                let acc_val = _mm256_load_si256(acc_ptr.add(j));
                _mm256_store_si256(acc_ptr.add(j), _mm256_add_epi16(acc_val, sum));
            }

            chunk_idx += 4;
        }

        // Handle remainder with prefetching
        for (i, &idx) in chunks.remainder().iter().enumerate() {
            let prefetch_idx = chunk_idx + i + PREFETCH_DISTANCE;
            if prefetch_idx < len {
                let pidx = *feature_indices.get_unchecked(prefetch_idx);
                let prefetch_ptr = weights_ptr.add(pidx * DIMS) as *const i8;
                _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
            }

            let feature_weights = weights_ptr.add(idx * DIMS) as *const __m256i;
            for j in 0..num_regs {
                let weight = _mm256_loadu_si256(feature_weights.add(j));
                let acc_val = _mm256_load_si256(acc_ptr.add(j));
                _mm256_store_si256(acc_ptr.add(j), _mm256_add_epi16(acc_val, weight));
            }
        }
    }
}

/// Accumulates the weights of all active sparse features into a dense accumulator.
///
/// This scalar variant is used by the fallback code paths where SIMD is unavailable.
///
/// # Parameters
/// * `feature_indices` - Sparse list of active feature indices to accumulate.
/// * `weights` - Flat weight matrix laid out in feature-major order.
/// * `acc` - Dense accumulation buffer that will be incremented in-place.
fn accumulate_sparse_scalar<const DIMS: usize>(
    feature_indices: &[usize],
    weights: &[i16],
    acc: &mut [i16; DIMS],
) {
    for &fi in feature_indices {
        let weights = &weights[fi * DIMS..][..DIMS];
        for (a, w) in acc.iter_mut().zip(weights.iter()) {
            *a += *w;
        }
    }
}
