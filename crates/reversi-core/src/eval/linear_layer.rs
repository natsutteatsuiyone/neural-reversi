//! Linear transformation layer for neural network.
//!
//! Reference: <https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/affine_transform.h>

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
use crate::util::align::Align64;

/// Linear transformation layer.
pub struct LinearLayer<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const PADDED_INPUT_DIMS: usize,
    const PADDED_OUTPUT_DIMS: usize,
> {
    biases: AVec<i32, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i8, ConstAlign<CACHE_LINE_SIZE>>,
    forward_fn: ForwardFn<INPUT_DIMS, OUTPUT_DIMS, PADDED_INPUT_DIMS, PADDED_OUTPUT_DIMS>,
}

/// Type alias for the forward function pointer.
type ForwardFn<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const PADDED_INPUT_DIMS: usize,
    const PADDED_OUTPUT_DIMS: usize,
> = unsafe fn(
    &LinearLayer<INPUT_DIMS, OUTPUT_DIMS, PADDED_INPUT_DIMS, PADDED_OUTPUT_DIMS>,
    &Align64<[u8; PADDED_INPUT_DIMS]>,
    &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
);

impl<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const PADDED_INPUT_DIMS: usize,
    const PADDED_OUTPUT_DIMS: usize,
> LinearLayer<INPUT_DIMS, OUTPUT_DIMS, PADDED_INPUT_DIMS, PADDED_OUTPUT_DIMS>
{
    /// Loads weights and biases from binary format.
    ///
    /// Format: `OUTPUT_DIMS` biases (little-endian `i32`) followed by row-major
    /// signed weights (`i8`) for each output neuron. The raw weights are
    /// repacked into the SIMD-friendly layout while loading.
    ///
    /// `PADDED_INPUT_DIMS` must be a multiple of 4 for correct weight repacking.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i32; PADDED_OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i8; PADDED_INPUT_DIMS * PADDED_OUTPUT_DIMS];

        for i in 0..OUTPUT_DIMS {
            biases[i] = reader.read_i32::<LittleEndian>()?;
        }

        let num_weights_to_read = PADDED_INPUT_DIMS * OUTPUT_DIMS;
        for i in 0..num_weights_to_read {
            let idx = Self::get_weight_index(i, PADDED_INPUT_DIMS, OUTPUT_DIMS);
            weights[idx] = reader.read_i8()?;
        }

        // Select the optimal forward implementation at load time
        let forward_fn = Self::select_forward_fn();

        Ok(LinearLayer {
            biases,
            weights,
            forward_fn,
        })
    }

    /// Selects the optimal forward implementation based on CPU features.
    fn select_forward_fn()
    -> ForwardFn<INPUT_DIMS, OUTPUT_DIMS, PADDED_INPUT_DIMS, PADDED_OUTPUT_DIMS> {
        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx512bw") => {
                use std::arch::x86_64::__m512i;
                use std::arch::is_x86_feature_detected;

                const OUTPUT_SIMD_WIDTH: usize = size_of::<__m512i>() / size_of::<i32>();
                let should_use_avx2 = OUTPUT_DIMS > 1 && OUTPUT_DIMS < OUTPUT_SIMD_WIDTH;

                if should_use_avx2 {
                    if is_x86_feature_detected!("avxvnni") {
                        Self::forward_avx2_vnni
                    } else {
                        Self::forward_avx2_no_vnni
                    }
                } else if is_x86_feature_detected!("avx512vnni") {
                    Self::forward_avx512_vnni
                } else {
                    Self::forward_avx512_no_vnni
                }
            }
            all(target_arch = "x86_64", target_feature = "avx2") => {
                use std::arch::is_x86_feature_detected;
                if is_x86_feature_detected!("avxvnni") {
                    Self::forward_avx2_vnni
                } else {
                    Self::forward_avx2_no_vnni
                }
            }
            all(target_arch = "aarch64", target_feature = "neon") => {
                use std::arch::is_aarch64_feature_detected;

                if is_aarch64_feature_detected!("i8mm") {
                    Self::forward_neon_i8mm_wrapper
                } else if is_aarch64_feature_detected!("dotprod") {
                    Self::forward_neon_dotprod_wrapper
                } else {
                    Self::forward_neon_wrapper
                }
            }
            _ => {
                Self::forward_fallback_wrapper
            }
        }
    }

    /// Converts matrix index to packed format for SIMD efficiency.
    fn get_weight_index(i: usize, input_size: usize, output_size: usize) -> usize {
        const STRIDE_MULTIPLIER: usize = 4;
        let output_stride = output_size * STRIDE_MULTIPLIER;
        (i / 4) % (input_size / 4) * output_stride + i / input_size * STRIDE_MULTIPLIER + i % 4
    }

    /// Returns the packed index for the weight at (input_idx, output_idx).
    #[inline(always)]
    fn get_packed_weight_index(&self, input_idx: usize, output_idx: usize) -> usize {
        let conceptual_index = output_idx * PADDED_INPUT_DIMS + input_idx;
        Self::get_weight_index(conceptual_index, PADDED_INPUT_DIMS, OUTPUT_DIMS)
    }

    /// Performs the forward pass of the linear layer.
    #[inline(always)]
    pub fn forward(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        unsafe { (self.forward_fn)(self, input, output) }
    }

    /// AVX-512 accelerated forward pass with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw,avx512vnni")]
    fn forward_avx512_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx512::<true>(input, output)
    }

    /// AVX-512 accelerated forward pass without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512_no_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx512::<false>(input, output)
    }

    /// AVX2 accelerated forward pass with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,avxvnni")]
    fn forward_avx2_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx2::<true>(input, output)
    }

    /// AVX2 accelerated forward pass without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    fn forward_avx2_no_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx2::<false>(input, output)
    }

    /// ARM NEON forward pass (thin wrapper with the right `target_feature`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    fn forward_neon_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_neon(input, output)
    }

    /// ARM NEON+dotprod forward pass (thin wrapper with the right `target_feature`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,dotprod")]
    fn forward_neon_dotprod_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_neon_dotprod(input, output)
    }

    /// ARM NEON+i8mm forward pass (thin wrapper with the right `target_feature`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,i8mm")]
    fn forward_neon_i8mm_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_neon_i8mm(input, output)
    }

    /// Wrapper for forward_fallback to match the unsafe fn signature.
    #[allow(dead_code)]
    unsafe fn forward_fallback_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_fallback(input, output)
    }

    /// AVX-512 accelerated forward pass optionally using VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    #[inline]
    fn forward_avx512<const USE_VNNI: bool>(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::ceil_to_multiple;
        use crate::eval::util::clone_biases;
        use crate::eval::util::mm512_dpbusd_epi32;
        use std::arch::x86_64::*;
        use std::mem::size_of;
        use std::ptr::copy_nonoverlapping;

        const OUTPUT_SIMD_WIDTH: usize = size_of::<__m512i>() / size_of::<i32>();

        unsafe {
            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut __m512i;

            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = std::cmp::max(OUTPUT_DIMS / OUTPUT_SIMD_WIDTH, 1);

            let input32 = input.as_ptr() as *const i32;
            for i in 0..num_chunks {
                let in0 = _mm512_set1_epi32(*input32.add(i));
                let col0 = self.weights.as_ptr().add(i * OUTPUT_DIMS * 4) as *const __m512i;

                for j in 0..num_regs {
                    let a = acc_ptr.add(j);
                    *a = mm512_dpbusd_epi32::<USE_VNNI>(*a, in0, *col0.add(j));
                }
            }

            copy_nonoverlapping(acc_ptr, output.as_ptr() as *mut __m512i, num_regs);
        }
    }

    /// AVX2 accelerated forward pass optionally using VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[inline]
    fn forward_avx2<const USE_VNNI: bool>(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::ceil_to_multiple;
        use crate::eval::util::clone_biases;
        use crate::eval::util::mm256_dpbusd_epi32;
        use std::arch::x86_64::*;
        use std::mem::size_of;
        use std::ptr::copy_nonoverlapping;

        const OUTPUT_SIMD_WIDTH: usize = size_of::<__m256i>() / size_of::<i32>();

        unsafe {
            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;

            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = OUTPUT_DIMS / OUTPUT_SIMD_WIDTH;

            let input32 = input.as_ptr() as *const i32;
            for i in 0..num_chunks {
                let in0 = _mm256_set1_epi32(*input32.add(i));
                let col0 = self.weights.as_ptr().add(i * OUTPUT_DIMS * 4) as *const __m256i;

                for j in 0..num_regs {
                    let a = acc_ptr.add(j);
                    *a = mm256_dpbusd_epi32::<USE_VNNI>(*a, in0, *col0.add(j));
                }
            }

            copy_nonoverlapping(acc_ptr, output.as_ptr() as *mut __m256i, num_regs);
        }
    }

    /// ARM NEON forward pass.
    ///
    /// Processes one input chunk of 4 `u8` values per outer iteration, broadcasting
    /// it into a `uint8x16_t` and accumulating 4 output neurons per inner `int32x4_t`
    /// register via `neon_dpbusd_s32`. Operates entirely within the pre-packed weight
    /// layout produced by `get_weight_index`, matching the AVX2 kernel.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[inline]
    fn forward_neon(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::ceil_to_multiple;
        use crate::eval::util::clone_biases;
        use crate::eval::util::neon_dpbusd_s32;
        use std::arch::aarch64::*;
        use std::ptr::copy_nonoverlapping;

        const OUTPUT_SIMD_WIDTH: usize = 4;

        unsafe {
            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut i32;

            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = OUTPUT_DIMS / OUTPUT_SIMD_WIDTH;

            let input32 = input.as_ptr() as *const i32;
            for i in 0..num_chunks {
                let packed = *input32.add(i);
                let in0 = vreinterpretq_u8_s32(vdupq_n_s32(packed));
                let col0 = self.weights.as_ptr().add(i * OUTPUT_DIMS * 4);

                for j in 0..num_regs {
                    let a_ptr = acc_ptr.add(j * OUTPUT_SIMD_WIDTH);
                    let a = vld1q_s32(a_ptr);
                    let w = vld1q_s8(col0.add(j * 16));
                    let updated = neon_dpbusd_s32(a, in0, w);
                    vst1q_s32(a_ptr, updated);
                }
            }

            copy_nonoverlapping(
                acc_ptr as *const i32,
                output.as_mut_ptr() as *mut i32,
                OUTPUT_DIMS,
            );
        }
    }

    /// ARM NEON forward pass using `SDOT` (FEAT_DotProd) with sign-correction emulation
    /// of the missing `USDOT`. Picked when i8mm is unavailable but dotprod is present
    /// (e.g. Apple M1, Cortex-A76..A78, Neoverse N1).
    ///
    /// Splits the unsigned input `a` into its low 7 bits and its sign bit: the latter,
    /// reinterpreted as i8, contributes −128·msb·b under SDOT, which the final subtraction
    /// undoes to yield +128·msb·b. The split depends only on `a`, so it is hoisted out
    /// of the inner loop.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,dotprod")]
    #[inline]
    fn forward_neon_dotprod(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::ceil_to_multiple;
        use crate::eval::util::clone_biases;
        use std::arch::aarch64::*;
        use std::ptr::copy_nonoverlapping;

        const OUTPUT_SIMD_WIDTH: usize = 4;

        unsafe {
            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut i32;

            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = OUTPUT_DIMS / OUTPUT_SIMD_WIDTH;

            let high_bit = vdupq_n_u8(0x80);
            let zero = vdupq_n_s32(0);

            let input32 = input.as_ptr() as *const i32;
            for i in 0..num_chunks {
                let packed = *input32.add(i);
                let in0 = vreinterpretq_u8_s32(vdupq_n_s32(packed));
                let a_low7_s8 = vreinterpretq_s8_u8(vbicq_u8(in0, high_bit));
                let a_msb_i8 = vreinterpretq_s8_u8(vandq_u8(in0, high_bit));
                let col0 = self.weights.as_ptr().add(i * OUTPUT_DIMS * 4);

                for j in 0..num_regs {
                    let a_ptr = acc_ptr.add(j * OUTPUT_SIMD_WIDTH);
                    let a = vld1q_s32(a_ptr);
                    let w = vld1q_s8(col0.add(j * 16));
                    let with_low = vdotq_s32(a, a_low7_s8, w);
                    let neg_high = vdotq_s32(zero, a_msb_i8, w);
                    vst1q_s32(a_ptr, vsubq_s32(with_low, neg_high));
                }
            }

            copy_nonoverlapping(
                acc_ptr as *const i32,
                output.as_mut_ptr() as *mut i32,
                OUTPUT_DIMS,
            );
        }
    }

    /// ARM NEON forward pass using the `USDOT` instruction via FEAT_I8MM.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,i8mm")]
    #[inline]
    fn forward_neon_i8mm(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::ceil_to_multiple;
        use crate::eval::util::clone_biases;
        use crate::eval::util::neon_dpbusd_s32_i8mm;
        use std::arch::aarch64::*;
        use std::ptr::copy_nonoverlapping;

        const OUTPUT_SIMD_WIDTH: usize = 4;

        unsafe {
            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut i32;

            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = OUTPUT_DIMS / OUTPUT_SIMD_WIDTH;

            let input32 = input.as_ptr() as *const i32;
            for i in 0..num_chunks {
                let packed = *input32.add(i);
                let in0 = vreinterpretq_u8_s32(vdupq_n_s32(packed));
                let col0 = self.weights.as_ptr().add(i * OUTPUT_DIMS * 4);

                for j in 0..num_regs {
                    let a_ptr = acc_ptr.add(j * OUTPUT_SIMD_WIDTH);
                    let a = vld1q_s32(a_ptr);
                    let w = vld1q_s8(col0.add(j * 16));
                    let updated = neon_dpbusd_s32_i8mm(a, in0, w);
                    vst1q_s32(a_ptr, updated);
                }
            }

            copy_nonoverlapping(
                acc_ptr as *const i32,
                output.as_mut_ptr() as *mut i32,
                OUTPUT_DIMS,
            );
        }
    }

    /// Portable forward pass.
    fn forward_fallback(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        output[..OUTPUT_DIMS].copy_from_slice(&self.biases[..OUTPUT_DIMS]);

        for (i, &input_byte) in input.iter().take(INPUT_DIMS).enumerate() {
            let input_val = input_byte as i32;
            if input_val == 0 {
                continue;
            }

            for (k, out) in output.iter_mut().take(OUTPUT_DIMS).enumerate() {
                let weight_idx = self.get_packed_weight_index(i, k);
                let weight_val = self.weights[weight_idx] as i32;
                *out += input_val * weight_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use aligned_vec::avec;

    use super::*;
    use std::io::Cursor;

    /// Helper to create a test LinearLayer with known weights and biases
    fn create_test_layer<const I: usize, const O: usize, const PI: usize, const PO: usize>()
    -> LinearLayer<I, O, PI, PO> {
        let forward_fn = LinearLayer::<I, O, PI, PO>::select_forward_fn();
        let mut layer = LinearLayer::<I, O, PI, PO> {
            biases: avec![[CACHE_LINE_SIZE]|0i32; PO],
            weights: avec![[CACHE_LINE_SIZE]|0i8; PI * PO],
            forward_fn,
        };

        // Set some test biases
        for i in 0..O {
            layer.biases[i] = (i as i32 + 1) * 100;
        }

        // Set some test weights using the proper packing function
        for output_idx in 0..O {
            for input_idx in 0..I {
                let idx = layer.get_packed_weight_index(input_idx, output_idx);
                // Use a simple pattern that avoids zero
                layer.weights[idx] = ((output_idx + input_idx + 1) % 127) as i8;
            }
        }

        layer
    }

    #[test]
    fn test_weight_index_calculation() {
        // Test the weight index calculation for a small matrix
        const INPUT_SIZE: usize = 8;
        const OUTPUT_SIZE: usize = 4;

        // Test various indices
        let idx0 = LinearLayer::<4, 4, 8, 4>::get_weight_index(0, INPUT_SIZE, OUTPUT_SIZE);
        let idx1 = LinearLayer::<4, 4, 8, 4>::get_weight_index(1, INPUT_SIZE, OUTPUT_SIZE);
        let idx4 = LinearLayer::<4, 4, 8, 4>::get_weight_index(4, INPUT_SIZE, OUTPUT_SIZE);

        // Verify that indices are unique and within bounds
        assert_ne!(idx0, idx1);
        assert_ne!(idx0, idx4);
        assert!(idx0 < INPUT_SIZE * OUTPUT_SIZE);
        assert!(idx1 < INPUT_SIZE * OUTPUT_SIZE);
        assert!(idx4 < INPUT_SIZE * OUTPUT_SIZE);
    }

    #[test]
    fn test_load_from_reader() {
        const I: usize = 4;
        const O: usize = 2;
        const PI: usize = 32; // Padded to SIMD width
        const PO: usize = 32;

        // Create test data
        let mut data = Vec::new();

        // Write biases (2 x i32 little-endian)
        data.extend_from_slice(&100i32.to_le_bytes());
        data.extend_from_slice(&200i32.to_le_bytes());

        // Write weights (PI * O = 32 * 2 = 64 bytes)
        for i in 0..PI * O {
            data.push((i % 127) as u8);
        }

        let mut cursor = Cursor::new(data);
        let layer = LinearLayer::<I, O, PI, PO>::load(&mut cursor).unwrap();

        // Verify biases were loaded correctly
        assert_eq!(layer.biases[0], 100);
        assert_eq!(layer.biases[1], 200);
    }

    #[test]
    fn test_forward_multiple_outputs() {
        const I: usize = 4;
        const O: usize = 8;
        const PI: usize = 32;
        const PO: usize = 32;

        let layer = create_test_layer::<I, O, PI, PO>();

        let mut input = Align64([0; PI]);
        let mut output = Align64([0; PO]);

        // Set input values
        for i in 0..I {
            input[i] = (i + 1) as u8 * 10;
        }

        layer.forward(&input, &mut output);

        // Verify all outputs are computed
        for i in 0..O {
            // Each output should at least have the bias value
            assert!(output[i] >= (i as i32 + 1) * 100);
        }
    }

    #[test]
    fn test_forward_with_zero_input() {
        const I: usize = 8;
        const O: usize = 8; // Must be multiple of 8 for AVX2
        const PI: usize = 32;
        const PO: usize = 32;

        let forward_fn = LinearLayer::<I, O, PI, PO>::select_forward_fn();
        let mut layer = LinearLayer::<I, O, PI, PO> {
            biases: avec![[CACHE_LINE_SIZE]|0i32; PO],
            weights: avec![[CACHE_LINE_SIZE]|0i8; PI * PO],
            forward_fn,
        };

        // Set biases
        for i in 0..O {
            layer.biases[i] = (i as i32 + 1) * 100;
        }

        let input = Align64([0; PI]); // All zeros
        let mut output = Align64([0; PO]);

        layer.forward(&input, &mut output);

        // With zero input, output should equal biases
        for i in 0..O {
            assert_eq!(output[i], (i as i32 + 1) * 100);
        }
    }

    #[test]
    fn test_packed_weight_index_bounds() {
        const I: usize = 64;
        const O: usize = 16;
        const PI: usize = 64;
        const PO: usize = 32;

        let layer = create_test_layer::<I, O, PI, PO>();

        // Test that all valid input/output combinations produce valid indices
        for output_idx in 0..O {
            for input_idx in 0..I {
                let idx = layer.get_packed_weight_index(input_idx, output_idx);
                assert!(
                    idx < PI * PO,
                    "Index {idx} out of bounds for input={input_idx}, output={output_idx}"
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_cross_check<const I: usize, const O: usize, const PI: usize, const PO: usize>() {
        let layer = create_test_layer::<I, O, PI, PO>();
        let mut input = Align64([0u8; PI]);
        for (idx, v) in input.iter_mut().take(I).enumerate() {
            *v = ((idx * 17 + 3) % 251) as u8;
        }

        let mut neon_out = Align64([0i32; PO]);
        let mut fb_out = Align64([0i32; PO]);

        // SAFETY: NEON is the aarch64 baseline, asserted by the cfg above.
        unsafe { layer.forward_neon(&input, &mut neon_out) };
        layer.forward_fallback(&input, &mut fb_out);

        assert_eq!(neon_out.as_ref(), fb_out.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn forward_neon_matches_fallback() {
        // Exercise both num_regs=4 (L1-shaped) and num_regs=16 (L2-shaped) loops.
        neon_cross_check::<32, 16, 32, 32>();
        neon_cross_check::<64, 64, 64, 64>();
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_dotprod_cross_check<
        const I: usize,
        const O: usize,
        const PI: usize,
        const PO: usize,
    >() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }

        let layer = create_test_layer::<I, O, PI, PO>();
        let mut input = Align64([0u8; PI]);
        // Cover the full u8 range so the msb sign-correction path gets exercised.
        for (idx, v) in input.iter_mut().take(I).enumerate() {
            *v = ((idx * 23 + 7) % 251) as u8;
        }

        let mut dp_out = Align64([0i32; PO]);
        let mut fb_out = Align64([0i32; PO]);

        // SAFETY: Guarded by runtime dotprod detection and the cfg above.
        unsafe { layer.forward_neon_dotprod(&input, &mut dp_out) };
        layer.forward_fallback(&input, &mut fb_out);

        assert_eq!(dp_out.as_ref(), fb_out.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn forward_neon_dotprod_matches_fallback() {
        neon_dotprod_cross_check::<32, 16, 32, 32>();
        neon_dotprod_cross_check::<64, 64, 64, 64>();
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_i8mm_cross_check<const I: usize, const O: usize, const PI: usize, const PO: usize>() {
        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        let layer = create_test_layer::<I, O, PI, PO>();
        let mut input = Align64([0u8; PI]);
        for (idx, v) in input.iter_mut().take(I).enumerate() {
            *v = ((idx * 29 + 11) % 251) as u8;
        }

        let mut i8mm_out = Align64([0i32; PO]);
        let mut fb_out = Align64([0i32; PO]);

        // SAFETY: Guarded by runtime i8mm detection and the cfg above.
        unsafe { layer.forward_neon_i8mm(&input, &mut i8mm_out) };
        layer.forward_fallback(&input, &mut fb_out);

        assert_eq!(i8mm_out.as_ref(), fb_out.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn forward_neon_i8mm_matches_fallback() {
        neon_i8mm_cross_check::<32, 16, 32, 32>();
        neon_i8mm_cross_check::<64, 64, 64, 64>();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_no_vnni_cross_check<
        const I: usize,
        const O: usize,
        const PI: usize,
        const PO: usize,
    >() {
        let layer = create_test_layer::<I, O, PI, PO>();
        let mut input = Align64([0u8; PI]);
        for (idx, v) in input.iter_mut().take(I).enumerate() {
            *v = ((idx * 17 + 3) % 251) as u8;
        }

        let mut simd_out = Align64([0i32; PO]);
        let mut fb_out = Align64([0i32; PO]);

        // SAFETY: AVX2 is asserted by the cfg above.
        unsafe { layer.forward_avx2_no_vnni(&input, &mut simd_out) };
        layer.forward_fallback(&input, &mut fb_out);

        assert_eq!(simd_out.as_ref(), fb_out.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn forward_avx2_no_vnni_matches_fallback() {
        avx2_no_vnni_cross_check::<32, 16, 32, 32>();
        avx2_no_vnni_cross_check::<64, 64, 64, 64>();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_vnni_cross_check<const I: usize, const O: usize, const PI: usize, const PO: usize>() {
        if !std::arch::is_x86_feature_detected!("avxvnni") {
            return;
        }

        let layer = create_test_layer::<I, O, PI, PO>();
        let mut input = Align64([0u8; PI]);
        // Cover the full u8 range so the VNNI path's u8×i8 semantics are exercised.
        for (idx, v) in input.iter_mut().take(I).enumerate() {
            *v = ((idx * 23 + 7) % 251) as u8;
        }

        let mut simd_out = Align64([0i32; PO]);
        let mut fb_out = Align64([0i32; PO]);

        // SAFETY: Guarded by runtime avxvnni detection and the cfg above.
        unsafe { layer.forward_avx2_vnni(&input, &mut simd_out) };
        layer.forward_fallback(&input, &mut fb_out);

        assert_eq!(simd_out.as_ref(), fb_out.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn forward_avx2_vnni_matches_fallback() {
        avx2_vnni_cross_check::<32, 16, 32, 32>();
        avx2_vnni_cross_check::<64, 64, 64, 64>();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn avx512_no_vnni_cross_check<
        const I: usize,
        const O: usize,
        const PI: usize,
        const PO: usize,
    >() {
        let layer = create_test_layer::<I, O, PI, PO>();
        let mut input = Align64([0u8; PI]);
        for (idx, v) in input.iter_mut().take(I).enumerate() {
            *v = ((idx * 17 + 3) % 251) as u8;
        }

        let mut simd_out = Align64([0i32; PO]);
        let mut fb_out = Align64([0i32; PO]);

        // SAFETY: AVX-512BW is asserted by the cfg above.
        unsafe { layer.forward_avx512_no_vnni(&input, &mut simd_out) };
        layer.forward_fallback(&input, &mut fb_out);

        assert_eq!(simd_out.as_ref(), fb_out.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn forward_avx512_no_vnni_matches_fallback() {
        avx512_no_vnni_cross_check::<32, 16, 32, 32>();
        avx512_no_vnni_cross_check::<64, 64, 64, 64>();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn avx512_vnni_cross_check<const I: usize, const O: usize, const PI: usize, const PO: usize>() {
        if !std::arch::is_x86_feature_detected!("avx512vnni") {
            return;
        }

        let layer = create_test_layer::<I, O, PI, PO>();
        let mut input = Align64([0u8; PI]);
        for (idx, v) in input.iter_mut().take(I).enumerate() {
            *v = ((idx * 23 + 7) % 251) as u8;
        }

        let mut simd_out = Align64([0i32; PO]);
        let mut fb_out = Align64([0i32; PO]);

        // SAFETY: Guarded by runtime avx512vnni detection and the cfg above.
        unsafe { layer.forward_avx512_vnni(&input, &mut simd_out) };
        layer.forward_fallback(&input, &mut fb_out);

        assert_eq!(simd_out.as_ref(), fb_out.as_ref());
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn forward_avx512_vnni_matches_fallback() {
        avx512_vnni_cross_check::<32, 16, 32, 32>();
        avx512_vnni_cross_check::<64, 64, 64, 64>();
    }

    #[test]
    fn test_forward_sparse_input() {
        // Test the zero-skipping optimization
        const I: usize = 32;
        const O: usize = 8; // Must be multiple of 8 for AVX2
        const PI: usize = 32;
        const PO: usize = 32;

        let forward_fn = LinearLayer::<I, O, PI, PO>::select_forward_fn();
        let mut layer = LinearLayer::<I, O, PI, PO> {
            biases: avec![[CACHE_LINE_SIZE]|0i32; PO],
            weights: avec![[CACHE_LINE_SIZE]|0i8; PI * PO],
            forward_fn,
        };

        // Set positive biases
        for i in 0..O {
            layer.biases[i] = 1000;
        }

        // Set some weights
        for i in 0..O {
            for j in 0..I {
                let idx = layer.get_packed_weight_index(j, i);
                layer.weights[idx] = 1;
            }
        }

        let mut input = Align64([0; PI]);
        let mut output = Align64([0; PO]);

        // Set only a few non-zero values (sparse input)
        input[5] = 100;
        input[15] = 200;

        layer.forward(&input, &mut output);

        // Verify computation completed successfully
        for i in 0..O {
            // Output should be bias (1000) plus contribution from non-zero inputs (300)
            assert_eq!(output[i], 1300);
        }
    }
}
