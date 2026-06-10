//! Linear transformation layer for neural network.
//!
//! Reference: <https://github.com/official-stockfish/Stockfish/blob/f3bfce353168b03e4fedce515de1898c691f81ec/src/nnue/layers/affine_transform.h>

use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
use crate::util::align::Align64;
use crate::util::aligned_buffer::AlignedBuffer;

/// Linear transformation layer.
pub struct LinearLayer<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const PADDED_INPUT_DIMS: usize,
    const PADDED_OUTPUT_DIMS: usize,
> {
    biases: AlignedBuffer<i32, CACHE_LINE_SIZE>,
    weights: AlignedBuffer<i8, CACHE_LINE_SIZE>,
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
        let mut biases = AlignedBuffer::<i32, CACHE_LINE_SIZE>::from_elem(0, PADDED_OUTPUT_DIMS);
        let mut weights = AlignedBuffer::<i8, CACHE_LINE_SIZE>::from_elem(
            0,
            PADDED_INPUT_DIMS * PADDED_OUTPUT_DIMS,
        );

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

    /// Runs the AVX-512 accelerated forward pass with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw,avx512vnni")]
    fn forward_avx512_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx512::<true>(input, output)
    }

    /// Runs the AVX-512 accelerated forward pass without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512_no_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx512::<false>(input, output)
    }

    /// Runs the AVX2 accelerated forward pass with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,avxvnni")]
    fn forward_avx2_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx2::<true>(input, output)
    }

    /// Runs the AVX2 accelerated forward pass without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    fn forward_avx2_no_vnni(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_avx2::<false>(input, output)
    }

    /// Runs the ARM NEON forward pass (thin wrapper with the right `target_feature`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    fn forward_neon_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_neon(input, output)
    }

    /// Runs the ARM NEON+dotprod forward pass (thin wrapper with the right `target_feature`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,dotprod")]
    fn forward_neon_dotprod_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_neon_dotprod(input, output)
    }

    /// Runs the ARM NEON+i8mm forward pass (thin wrapper with the right `target_feature`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,i8mm")]
    fn forward_neon_i8mm_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_neon_i8mm(input, output)
    }

    /// Wraps [`forward_fallback`](Self::forward_fallback) to match the unsafe fn signature.
    ///
    /// # Safety
    ///
    /// This wrapper imposes no additional requirements; it forwards directly to
    /// the safe [`forward_fallback`](Self::forward_fallback) and is `unsafe`
    /// only to match the [`ForwardFn`] pointer type.
    #[allow(dead_code)]
    unsafe fn forward_fallback_wrapper(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        self.forward_fallback(input, output)
    }

    /// Runs the AVX-512 accelerated forward pass, optionally using VNNI.
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
            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = (OUTPUT_DIMS / OUTPUT_SIMD_WIDTH).max(1);
            // A single accumulator serializes the chunk loop on `dpbusd`
            // latency; split into `unroll` partial sums and reduce. Integer
            // add is associative, so the result is bit-identical.
            let unroll = (4 / num_regs).max(1);

            let mut p0: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let mut p1: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let mut p2: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let mut p3: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let lanes: [*mut __m512i; 4] = [
                p0.as_mut_ptr() as *mut __m512i,
                p1.as_mut_ptr() as *mut __m512i,
                p2.as_mut_ptr() as *mut __m512i,
                p3.as_mut_ptr() as *mut __m512i,
            ];

            let input32 = input.as_ptr() as *const i32;
            let weights = self.weights.as_ptr();

            let mut i = 0;
            while i + unroll <= num_chunks {
                for (u, &lane) in lanes[..unroll].iter().enumerate() {
                    let in0 = _mm512_set1_epi32(*input32.add(i + u));
                    let col0 = weights.add((i + u) * OUTPUT_DIMS * 4) as *const __m512i;
                    for j in 0..num_regs {
                        *lane.add(j) =
                            mm512_dpbusd_epi32::<USE_VNNI>(*lane.add(j), in0, *col0.add(j));
                    }
                }
                i += unroll;
            }
            while i < num_chunks {
                let in0 = _mm512_set1_epi32(*input32.add(i));
                let col0 = weights.add(i * OUTPUT_DIMS * 4) as *const __m512i;
                let lane = lanes[0];
                for j in 0..num_regs {
                    *lane.add(j) = mm512_dpbusd_epi32::<USE_VNNI>(*lane.add(j), in0, *col0.add(j));
                }
                i += 1;
            }

            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut __m512i;
            for j in 0..num_regs {
                let mut s = *lanes[0].add(j);
                for &lane in &lanes[1..unroll] {
                    s = _mm512_add_epi32(s, *lane.add(j));
                }
                *acc_ptr.add(j) = _mm512_add_epi32(*acc_ptr.add(j), s);
            }

            copy_nonoverlapping(acc_ptr, output.as_ptr() as *mut __m512i, num_regs);
        }
    }

    /// Runs the AVX2 accelerated forward pass, optionally using VNNI.
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
            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = (OUTPUT_DIMS / OUTPUT_SIMD_WIDTH).max(1);
            // A single accumulator serializes the chunk loop on `dpbusd`
            // latency; split into `unroll` partial sums and reduce. Integer
            // add is associative, so the result is bit-identical.
            let unroll = (4 / num_regs).max(1);

            let mut p0: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let mut p1: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let mut p2: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let mut p3: Align64<[i32; OUTPUT_DIMS]> = Align64([0; OUTPUT_DIMS]);
            let lanes: [*mut __m256i; 4] = [
                p0.as_mut_ptr() as *mut __m256i,
                p1.as_mut_ptr() as *mut __m256i,
                p2.as_mut_ptr() as *mut __m256i,
                p3.as_mut_ptr() as *mut __m256i,
            ];

            let input32 = input.as_ptr() as *const i32;
            let weights = self.weights.as_ptr();

            let mut i = 0;
            while i + unroll <= num_chunks {
                for (u, &lane) in lanes[..unroll].iter().enumerate() {
                    let in0 = _mm256_set1_epi32(*input32.add(i + u));
                    let col0 = weights.add((i + u) * OUTPUT_DIMS * 4) as *const __m256i;
                    for j in 0..num_regs {
                        *lane.add(j) =
                            mm256_dpbusd_epi32::<USE_VNNI>(*lane.add(j), in0, *col0.add(j));
                    }
                }
                i += unroll;
            }
            while i < num_chunks {
                let in0 = _mm256_set1_epi32(*input32.add(i));
                let col0 = weights.add(i * OUTPUT_DIMS * 4) as *const __m256i;
                let lane = lanes[0];
                for j in 0..num_regs {
                    *lane.add(j) = mm256_dpbusd_epi32::<USE_VNNI>(*lane.add(j), in0, *col0.add(j));
                }
                i += 1;
            }

            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut __m256i;
            for j in 0..num_regs {
                let mut s = *lanes[0].add(j);
                for &lane in &lanes[1..unroll] {
                    s = _mm256_add_epi32(s, *lane.add(j));
                }
                *acc_ptr.add(j) = _mm256_add_epi32(*acc_ptr.add(j), s);
            }

            copy_nonoverlapping(acc_ptr, output.as_ptr() as *mut __m256i, num_regs);
        }
    }

    /// Runs the ARM NEON forward pass.
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

    /// Runs the ARM NEON forward pass using `SDOT` (FEAT_DotProd) with sign-correction
    /// emulation of the missing `USDOT`.
    ///
    /// Picked when i8mm is unavailable but dotprod is present (e.g. Apple M1,
    /// Cortex-A76..A78, Neoverse N1).
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
        use crate::eval::util::neon_dpbusd_s32_dotprod;
        use std::arch::aarch64::*;
        use std::ptr::copy_nonoverlapping;

        const OUTPUT_SIMD_WIDTH: usize = 4;

        unsafe {
            let mut acc: Align64<[i32; OUTPUT_DIMS]> = clone_biases(&self.biases);
            let acc_ptr = acc.as_mut_ptr() as *mut i32;

            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = OUTPUT_DIMS / OUTPUT_SIMD_WIDTH;

            let high_bit = vdupq_n_u8(0x80);

            let weights_base = self.weights.as_ptr();
            let input_ptr = input.as_ptr() as *const u8;
            let main_end = num_chunks & !3;

            let mut i = 0;
            while i < main_end {
                let input16 = vld1q_u8(input_ptr.add(i * 4));
                let a_low7_s8 = vreinterpretq_s8_u8(vbicq_u8(input16, high_bit));
                let a_msb_i8 = vreinterpretq_s8_u8(vandq_u8(input16, high_bit));

                let col0 = weights_base.add(i * OUTPUT_DIMS * 4);
                let col1 = weights_base.add((i + 1) * OUTPUT_DIMS * 4);
                let col2 = weights_base.add((i + 2) * OUTPUT_DIMS * 4);
                let col3 = weights_base.add((i + 3) * OUTPUT_DIMS * 4);

                for j in 0..num_regs {
                    let a_ptr = acc_ptr.add(j * OUTPUT_SIMD_WIDTH);
                    let mut pos = vld1q_s32(a_ptr);
                    let mut neg = vdupq_n_s32(0);
                    let w0 = vld1q_s8(col0.add(j * 16));
                    let w1 = vld1q_s8(col1.add(j * 16));
                    let w2 = vld1q_s8(col2.add(j * 16));
                    let w3 = vld1q_s8(col3.add(j * 16));
                    pos = vdotq_laneq_s32::<0>(pos, w0, a_low7_s8);
                    pos = vdotq_laneq_s32::<1>(pos, w1, a_low7_s8);
                    pos = vdotq_laneq_s32::<2>(pos, w2, a_low7_s8);
                    pos = vdotq_laneq_s32::<3>(pos, w3, a_low7_s8);
                    neg = vdotq_laneq_s32::<0>(neg, w0, a_msb_i8);
                    neg = vdotq_laneq_s32::<1>(neg, w1, a_msb_i8);
                    neg = vdotq_laneq_s32::<2>(neg, w2, a_msb_i8);
                    neg = vdotq_laneq_s32::<3>(neg, w3, a_msb_i8);
                    vst1q_s32(a_ptr, vsubq_s32(pos, neg));
                }
                i += 4;
            }

            let input32 = input.as_ptr() as *const i32;
            while i < num_chunks {
                let packed = *input32.add(i);
                let in0 = vreinterpretq_u8_s32(vdupq_n_s32(packed));
                let col0 = weights_base.add(i * OUTPUT_DIMS * 4);
                for j in 0..num_regs {
                    let a_ptr = acc_ptr.add(j * OUTPUT_SIMD_WIDTH);
                    let a = vld1q_s32(a_ptr);
                    let w = vld1q_s8(col0.add(j * 16));
                    vst1q_s32(a_ptr, neon_dpbusd_s32_dotprod(a, in0, w));
                }
                i += 1;
            }

            copy_nonoverlapping(
                acc_ptr as *const i32,
                output.as_mut_ptr() as *mut i32,
                OUTPUT_DIMS,
            );
        }
    }

    /// Runs the ARM NEON forward pass using the `USDOT` instruction via FEAT_I8MM.
    ///
    /// Fuses four chunks per `j` register (one 16-byte input load + four laneq
    /// `usdot`s) and keeps accumulators register-resident for the whole pass. For
    /// `num_regs <= 4` the four `usdot`s target independent per-lane partials to
    /// break the latency chain, summed with the bias at the end.
    // Index loops are deliberate: `iter_mut().enumerate()` defeats register
    // promotion here (~2% slower), and the index drives the pointer math.
    #[allow(clippy::needless_range_loop)]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,i8mm")]
    #[inline]
    fn forward_neon_i8mm(
        &self,
        input: &Align64<[u8; PADDED_INPUT_DIMS]>,
        output: &mut Align64<[i32; PADDED_OUTPUT_DIMS]>,
    ) {
        use crate::eval::util::ceil_to_multiple;
        use crate::eval::util::neon_dpbusd_s32_i8mm;
        use std::arch::aarch64::*;

        const OUTPUT_SIMD_WIDTH: usize = 4;
        // Fits both shapes: Shape B uses `num_regs` (<= 16, L2's 64/4); Shape A
        // uses `num_regs * 4` partials but only runs for `num_regs <= 4`.
        const MAX_REGS: usize = 16;

        unsafe {
            let num_chunks: usize = ceil_to_multiple(INPUT_DIMS, 8) / 4;
            let num_regs = OUTPUT_DIMS / OUTPUT_SIMD_WIDTH;
            debug_assert!(num_regs <= MAX_REGS);

            let weights_base = self.weights.as_ptr();
            let input_ptr = input.as_ptr() as *const u8;
            let input32 = input.as_ptr() as *const i32;
            let bias_ptr = self.biases.as_ptr();
            let out_ptr = output.as_mut_ptr() as *mut i32;
            let main_end = num_chunks & !3;

            if num_regs <= 4 {
                // Shape A: four independent per-lane partials per register break
                // the `usdot` latency chain; bias is folded in at the end.
                let mut p = [vdupq_n_s32(0); MAX_REGS];

                let mut i = 0;
                while i < main_end {
                    let input16 = vld1q_u8(input_ptr.add(i * 4));
                    let col0 = weights_base.add(i * OUTPUT_DIMS * 4);
                    let col1 = weights_base.add((i + 1) * OUTPUT_DIMS * 4);
                    let col2 = weights_base.add((i + 2) * OUTPUT_DIMS * 4);
                    let col3 = weights_base.add((i + 3) * OUTPUT_DIMS * 4);
                    for j in 0..num_regs {
                        let w0 = vld1q_s8(col0.add(j * 16));
                        let w1 = vld1q_s8(col1.add(j * 16));
                        let w2 = vld1q_s8(col2.add(j * 16));
                        let w3 = vld1q_s8(col3.add(j * 16));
                        p[j * 4] = vsudotq_laneq_s32::<0>(p[j * 4], w0, input16);
                        p[j * 4 + 1] = vsudotq_laneq_s32::<1>(p[j * 4 + 1], w1, input16);
                        p[j * 4 + 2] = vsudotq_laneq_s32::<2>(p[j * 4 + 2], w2, input16);
                        p[j * 4 + 3] = vsudotq_laneq_s32::<3>(p[j * 4 + 3], w3, input16);
                    }
                    i += 4;
                }

                while i < num_chunks {
                    let in0 = vreinterpretq_u8_s32(vdupq_n_s32(*input32.add(i)));
                    let col0 = weights_base.add(i * OUTPUT_DIMS * 4);
                    for j in 0..num_regs {
                        let w = vld1q_s8(col0.add(j * 16));
                        p[j * 4] = neon_dpbusd_s32_i8mm(p[j * 4], in0, w);
                    }
                    i += 1;
                }

                for j in 0..num_regs {
                    let s = vaddq_s32(
                        vaddq_s32(p[j * 4], p[j * 4 + 1]),
                        vaddq_s32(p[j * 4 + 2], p[j * 4 + 3]),
                    );
                    let b = vld1q_s32(bias_ptr.add(j * 4));
                    vst1q_s32(out_ptr.add(j * 4), vaddq_s32(b, s));
                }
            } else {
                // Shape B: one resident accumulator per register — per-lane
                // partials would need 4x the vectors and spill.
                let mut acc = [vdupq_n_s32(0); MAX_REGS];
                for j in 0..num_regs {
                    acc[j] = vld1q_s32(bias_ptr.add(j * 4));
                }

                let mut i = 0;
                while i < main_end {
                    let input16 = vld1q_u8(input_ptr.add(i * 4));
                    let col0 = weights_base.add(i * OUTPUT_DIMS * 4);
                    let col1 = weights_base.add((i + 1) * OUTPUT_DIMS * 4);
                    let col2 = weights_base.add((i + 2) * OUTPUT_DIMS * 4);
                    let col3 = weights_base.add((i + 3) * OUTPUT_DIMS * 4);
                    for j in 0..num_regs {
                        let w0 = vld1q_s8(col0.add(j * 16));
                        let w1 = vld1q_s8(col1.add(j * 16));
                        let w2 = vld1q_s8(col2.add(j * 16));
                        let w3 = vld1q_s8(col3.add(j * 16));
                        acc[j] = vsudotq_laneq_s32::<0>(acc[j], w0, input16);
                        acc[j] = vsudotq_laneq_s32::<1>(acc[j], w1, input16);
                        acc[j] = vsudotq_laneq_s32::<2>(acc[j], w2, input16);
                        acc[j] = vsudotq_laneq_s32::<3>(acc[j], w3, input16);
                    }
                    i += 4;
                }

                while i < num_chunks {
                    let in0 = vreinterpretq_u8_s32(vdupq_n_s32(*input32.add(i)));
                    let col0 = weights_base.add(i * OUTPUT_DIMS * 4);
                    for j in 0..num_regs {
                        let w = vld1q_s8(col0.add(j * 16));
                        acc[j] = neon_dpbusd_s32_i8mm(acc[j], in0, w);
                    }
                    i += 1;
                }

                for j in 0..num_regs {
                    vst1q_s32(out_ptr.add(j * 4), acc[j]);
                }
            }
        }
    }

    /// Runs the portable forward pass.
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
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Cursor;

    fn build_layer<const I: usize, const O: usize, const PI: usize, const PO: usize>(
        seed: i32,
    ) -> LinearLayer<I, O, PI, PO> {
        let mut layer = LinearLayer {
            biases: AlignedBuffer::from_elem(0, PO),
            weights: AlignedBuffer::from_elem(0, PI * PO),
            forward_fn: LinearLayer::<I, O, PI, PO>::select_forward_fn(),
        };

        for output_idx in 0..O {
            layer.biases[output_idx] = (output_idx as i32 * 101 + seed).rem_euclid(20_000) - 10_000;
            for input_idx in 0..I {
                let weight_idx = layer.get_packed_weight_index(input_idx, output_idx);
                layer.weights[weight_idx] =
                    ((output_idx as i32 * 37 + input_idx as i32 * 19 + seed).rem_euclid(255) - 127)
                        as i8;
            }
        }

        layer
    }

    fn reference_forward<const I: usize, const O: usize, const PI: usize, const PO: usize>(
        layer: &LinearLayer<I, O, PI, PO>,
        input: &Align64<[u8; PI]>,
    ) -> [i32; O] {
        let mut output = [0; O];
        output.copy_from_slice(&layer.biases[..O]);

        for input_idx in 0..I {
            let input_value = i32::from(input[input_idx]);
            if input_value == 0 {
                continue;
            }
            for (output_idx, out) in output.iter_mut().enumerate() {
                let weight_idx = layer.get_packed_weight_index(input_idx, output_idx);
                *out += input_value * i32::from(layer.weights[weight_idx]);
            }
        }

        output
    }

    fn patterned_input<const I: usize, const PI: usize>(seed: usize) -> Align64<[u8; PI]> {
        let mut input = Align64([0; PI]);
        for (idx, value) in input.iter_mut().take(I).enumerate() {
            *value = ((idx * 53 + seed * 17 + 11) & 0xff) as u8;
        }
        for value in input.iter_mut().skip(I) {
            *value = 239;
        }
        input
    }

    fn assert_forward_matches_reference<
        const I: usize,
        const O: usize,
        const PI: usize,
        const PO: usize,
    >(
        seed: i32,
        input_seed: usize,
    ) {
        let layer = build_layer::<I, O, PI, PO>(seed);
        let input = patterned_input::<I, PI>(input_seed);
        let expected = reference_forward(&layer, &input);
        let mut actual = Align64([i32::MIN; PO]);

        layer.forward(&input, &mut actual);

        assert_eq!(&actual.as_ref()[..O], &expected);
    }

    #[test]
    fn weight_index_maps_row_major_matrix_into_unique_packed_slots() {
        const INPUT_SIZE: usize = 8;
        const OUTPUT_SIZE: usize = 3;
        let expected = [
            0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23,
        ];
        let mut seen = [false; INPUT_SIZE * OUTPUT_SIZE];

        for (conceptual_idx, &expected_idx) in expected.iter().enumerate() {
            let packed_idx = LinearLayer::<6, OUTPUT_SIZE, INPUT_SIZE, 8>::get_weight_index(
                conceptual_idx,
                INPUT_SIZE,
                OUTPUT_SIZE,
            );
            assert_eq!(
                packed_idx, expected_idx,
                "conceptual index {conceptual_idx}"
            );
            assert!(!seen[packed_idx], "duplicate packed index {packed_idx}");
            seen[packed_idx] = true;
        }

        assert!(seen.into_iter().all(|hit| hit));
    }

    #[test]
    fn load_reads_biases_and_repackages_row_major_weights() {
        const I: usize = 6;
        const O: usize = 3;
        const PI: usize = 8;
        const PO: usize = 8;
        let biases = [-100, 0, 250];
        let mut data = Vec::new();
        for bias in biases {
            data.write_i32::<LittleEndian>(bias).unwrap();
        }
        for conceptual_idx in 0..PI * O {
            data.write_i8(conceptual_idx as i8 - 40).unwrap();
        }

        let mut cursor = Cursor::new(data);
        let layer = LinearLayer::<I, O, PI, PO>::load(&mut cursor).unwrap();

        assert_eq!(&layer.biases[..O], &biases);
        assert_eq!(&layer.biases[O..PO], &[0; PO - O]);
        for output_idx in 0..O {
            for input_idx in 0..PI {
                let conceptual_idx = output_idx * PI + input_idx;
                let packed_idx = layer.get_packed_weight_index(input_idx, output_idx);
                assert_eq!(
                    layer.weights[packed_idx],
                    conceptual_idx as i8 - 40,
                    "input {input_idx}, output {output_idx}",
                );
            }
        }
    }

    #[test]
    fn load_reports_truncated_biases_or_weights() {
        type Layer = LinearLayer<2, 2, 4, 4>;

        let mut missing_bias = Cursor::new(1i32.to_le_bytes().to_vec());
        assert!(Layer::load(&mut missing_bias).is_err());

        let mut missing_weight = Vec::new();
        missing_weight.write_i32::<LittleEndian>(1).unwrap();
        missing_weight.write_i32::<LittleEndian>(2).unwrap();
        missing_weight.extend([7u8; 7]);
        assert!(Layer::load(&mut Cursor::new(missing_weight)).is_err());
    }

    #[test]
    fn forward_fallback_matches_reference_and_preserves_padded_outputs() {
        const I: usize = 6;
        const O: usize = 5;
        const PI: usize = 8;
        const PO: usize = 8;
        let layer = build_layer::<I, O, PI, PO>(31);
        let input = Align64([0, 255, 7, 0, 31, 128, 200, 201]);
        let expected = reference_forward(&layer, &input);
        let mut actual = Align64([777; PO]);

        layer.forward_fallback(&input, &mut actual);

        assert_eq!(&actual.as_ref()[..O], &expected);
        assert_eq!(&actual.as_ref()[O..], &[777; PO - O]);
    }

    #[test]
    fn forward_dispatch_matches_reference_with_main_chunks_and_padding() {
        assert_forward_matches_reference::<18, 8, 24, 8>(43, 3);
        assert_forward_matches_reference::<64, 16, 64, 16>(97, 11);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_forward_kernels_match_reference_for_main_and_remainder_chunks() {
        fn run<const I: usize, const O: usize, const PI: usize, const PO: usize>(
            seed: i32,
            input_seed: usize,
        ) {
            let layer = build_layer::<I, O, PI, PO>(seed);
            let input = patterned_input::<I, PI>(input_seed);
            let expected = reference_forward(&layer, &input);
            let mut actual = Align64([0; PO]);

            unsafe { layer.forward_neon(&input, &mut actual) };
            assert_eq!(&actual.as_ref()[..O], &expected, "neon");

            if std::arch::is_aarch64_feature_detected!("dotprod") {
                let mut dotprod = Align64([0; PO]);
                unsafe { layer.forward_neon_dotprod(&input, &mut dotprod) };
                assert_eq!(&dotprod.as_ref()[..O], &expected, "dotprod");
            }

            if std::arch::is_aarch64_feature_detected!("i8mm") {
                let mut i8mm = Align64([0; PO]);
                unsafe { layer.forward_neon_i8mm(&input, &mut i8mm) };
                assert_eq!(&i8mm.as_ref()[..O], &expected, "i8mm");
            }
        }

        run::<18, 8, 24, 8>(23, 5);
        run::<64, 16, 64, 16>(71, 13);
        run::<32, 64, 32, 64>(51, 7);
        run::<18, 64, 24, 64>(89, 3);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_forward_kernels_match_reference() {
        let layer = build_layer::<64, 8, 64, 8>(71);
        let input = patterned_input::<64, 64>(13);
        let expected = reference_forward(&layer, &input);
        let mut actual = Align64([0; 8]);

        unsafe { layer.forward_avx2_no_vnni(&input, &mut actual) };
        assert_eq!(actual.as_ref(), &expected, "avx2");

        if std::arch::is_x86_feature_detected!("avxvnni") {
            let mut vnni = Align64([0; 8]);
            unsafe { layer.forward_avx2_vnni(&input, &mut vnni) };
            assert_eq!(vnni.as_ref(), &expected, "avx2 vnni");
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn avx512_forward_kernels_match_reference() {
        let layer = build_layer::<64, 16, 64, 16>(71);
        let input = patterned_input::<64, 64>(13);
        let expected = reference_forward(&layer, &input);
        let mut actual = Align64([0; 16]);

        unsafe { layer.forward_avx512_no_vnni(&input, &mut actual) };
        assert_eq!(actual.as_ref(), &expected, "avx512");

        if std::arch::is_x86_feature_detected!("avx512vnni") {
            let mut vnni = Align64([0; 16]);
            unsafe { layer.forward_avx512_vnni(&input, &mut vnni) };
            assert_eq!(vnni.as_ref(), &expected, "avx512 vnni");
        }
    }
}
