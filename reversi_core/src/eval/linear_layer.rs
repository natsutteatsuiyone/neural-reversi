//! Reference: https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/affine_transform.h
use std::{
    io::{self, Read},
    mem::size_of,
    ptr::copy_nonoverlapping,
};

#[cfg(target_arch = "x86_64")]
use {std::arch::is_x86_feature_detected, std::arch::x86_64::*};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::constants::CACHE_LINE_SIZE;
use crate::eval::util::clone_biases;
use crate::util::align::Align64;
use crate::util::ceil_to_multiple;

/// Linear transformation layer.
///
/// # Type Parameters
///
/// * `INPUT_DIMS` - Actual number of input features
/// * `OUTPUT_DIMS` - Actual number of output neurons
/// * `PADDED_INPUT_DIMS` - Input dimensions padded to SIMD width (must be ≥ INPUT_DIMS)
/// * `PADDED_OUTPUT_DIMS` - Output dimensions padded to SIMD width (must be ≥ OUTPUT_DIMS)
#[derive(Debug)]
pub struct LinearLayer<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const PADDED_INPUT_DIMS: usize,
    const PADDED_OUTPUT_DIMS: usize,
> {
    biases: AVec<i32, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i8, ConstAlign<CACHE_LINE_SIZE>>,
}

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
    /// repacked into the SIMD-friendly layout while loading, and any padded
    /// slots stay zero-initialised.
    ///
    /// # Arguments
    /// * `reader` - Input stream to read from
    ///
    /// # Returns
    /// * `Ok(LinearLayer)` on success
    /// * `Err(io::Error)` on failure
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

        Ok(LinearLayer { biases, weights })
    }

    /// Converts matrix index to packed format for SIMD efficiency.
    fn get_weight_index(i: usize, input_size: usize, output_size: usize) -> usize {
        const STRIDE_MULTIPLIER: usize = 4;
        let output_stride = output_size * STRIDE_MULTIPLIER;
        (i / 4) % (input_size / 4) * output_stride + i / input_size * STRIDE_MULTIPLIER + i % 4
    }

    /// Gets packed index for weight at (input_idx, output_idx).
    #[inline(always)]
    fn get_packed_weight_index(&self, input_idx: usize, output_idx: usize) -> usize {
        let conceptual_index = output_idx * PADDED_INPUT_DIMS + input_idx;
        Self::get_weight_index(conceptual_index, PADDED_INPUT_DIMS, OUTPUT_DIMS)
    }

    /// Performs the forward pass of the linear layer.
    ///
    /// # Arguments
    ///
    /// * `input` - Aligned activations (`u8`) of length `PADDED_INPUT_DIMS` (only the
    ///   first `INPUT_DIMS` entries are consumed)
    /// * `output` - Aligned accumulator buffer (`i32`) where the first `OUTPUT_DIMS`
    ///   elements are overwritten; padded tail is left untouched
    #[inline(always)]
    pub fn forward(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                if cfg!(target_feature = "avx512bw") {
                    if is_x86_feature_detected!("avx512vnni") {
                        self.forward_avx512::<true>(input, output);
                    } else {
                        self.forward_avx512::<false>(input, output);
                    }
                    return;
                } else if cfg!(target_feature = "avx2") {
                    if is_x86_feature_detected!("avxvnni") {
                        self.forward_avx2::<true>(input, output);
                    } else {
                        self.forward_avx2::<false>(input, output);
                    }
                    return;
                }
            }
        }

        self.forward_fallback(input, output);
    }

    /// AVX-512 accelerated forward pass optionally using VNNI.
    /// Falls back to the AVX2 path when the layer is narrower than a full 512-bit vector.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512bw,avx512vl")]
    fn forward_avx512<const USE_VNNI: bool>(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::mm512_dpbusd_epi32;

        const OUTPUT_SIMD_WIDTH: usize = size_of::<__m512i>() / size_of::<i32>();

        if OUTPUT_DIMS > 1 && OUTPUT_DIMS < OUTPUT_SIMD_WIDTH {
            self.forward_avx2::<USE_VNNI>(input, output);
            return;
        }

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
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn forward_avx2<const USE_VNNI: bool>(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::mm256_dpbusd_epi32;

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
    use super::*;
    use std::io::Cursor;
    use std::is_x86_feature_detected;

    /// Helper to create a test LinearLayer with known weights and biases
    fn create_test_layer<const I: usize, const O: usize, const PI: usize, const PO: usize>()
    -> LinearLayer<I, O, PI, PO> {
        let mut layer = LinearLayer::<I, O, PI, PO> {
            biases: avec![[CACHE_LINE_SIZE]|0i32; PO],
            weights: avec![[CACHE_LINE_SIZE]|0i8; PI * PO],
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

        let mut layer = LinearLayer::<I, O, PI, PO> {
            biases: avec![[CACHE_LINE_SIZE]|0i32; PO],
            weights: avec![[CACHE_LINE_SIZE]|0i8; PI * PO],
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
    fn test_forward_consistency() {
        // Test that forward_avx2 and forward_fallback produce the same results
        const I: usize = 32;
        const O: usize = 8;
        const PI: usize = 32;
        const PO: usize = 32;

        let layer = create_test_layer::<I, O, PI, PO>();

        let mut input = Align64([0; PI]);
        let mut output_avx2 = Align64([0; PO]);
        let mut output_fallback = Align64([0; PO]);

        // Set random-like input values
        for i in 0..I {
            input[i] = ((i * 37 + 13) % 256) as u8;
        }

        // Run both implementations
        if is_x86_feature_detected!("avx2") {
            unsafe {
                layer.forward_avx2::<false>(&input, &mut output_avx2);
            }
        }
        layer.forward_fallback(&input, &mut output_fallback);

        // If AVX2 is available, verify results match
        if is_x86_feature_detected!("avx2") {
            for i in 0..O {
                assert_eq!(
                    output_avx2[i], output_fallback[i],
                    "Mismatch at output index {i}"
                );
            }
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

    #[test]
    fn test_forward_sparse_input() {
        // Test the zero-skipping optimization
        const I: usize = 32;
        const O: usize = 8; // Must be multiple of 8 for AVX2
        const PI: usize = 32;
        const PO: usize = 32;

        let mut layer = LinearLayer::<I, O, PI, PO> {
            biases: avec![[CACHE_LINE_SIZE]|0i32; PO],
            weights: avec![[CACHE_LINE_SIZE]|0i8; PI * PO],
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
