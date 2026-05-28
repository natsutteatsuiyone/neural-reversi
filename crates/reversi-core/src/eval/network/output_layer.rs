//! Output layer with 16-bit weights for neural network evaluation.

use std::io::{self, Read};

use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;
use crate::util::aligned_buffer::AlignedBuffer;

/// Output layer with 16-bit weights for computing a single scalar output.
pub struct OutputLayer<const INPUT_DIMS: usize, const PADDED_INPUT_DIMS: usize> {
    /// Bias term for the output neuron.
    bias: i32,
    /// Weight vector aligned for SIMD access.
    weights: AlignedBuffer<i16, CACHE_LINE_SIZE>,
    /// Weights packed as per-chunk low/high bytes for the AArch64 I8MM path.
    #[cfg(target_arch = "aarch64")]
    i8mm_weights: AlignedBuffer<u8, CACHE_LINE_SIZE>,
    /// Function pointer to the optimal forward implementation, selected at load time
    /// based on detected CPU SIMD capabilities.
    forward_fn: unsafe fn(&Self, [&[u8]; 3]) -> i32,
}

impl<const INPUT_DIMS: usize, const PADDED_INPUT_DIMS: usize>
    OutputLayer<INPUT_DIMS, PADDED_INPUT_DIMS>
{
    /// Loads the output layer from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let bias = reader.read_i32::<LittleEndian>()?;

        let mut weights = AlignedBuffer::<i16, CACHE_LINE_SIZE>::from_elem(0, PADDED_INPUT_DIMS);
        reader.read_i16_into::<LittleEndian>(weights.as_mut_slice())?;

        #[cfg(target_arch = "aarch64")]
        let i8mm_weights = Self::pack_i8mm_weights(weights.as_slice());

        let forward_fn = Self::select_forward_fn();

        Ok(Self {
            bias,
            weights,
            #[cfg(target_arch = "aarch64")]
            i8mm_weights,
            forward_fn,
        })
    }

    #[cfg(target_arch = "aarch64")]
    fn pack_i8mm_weights(weights: &[i16]) -> AlignedBuffer<u8, CACHE_LINE_SIZE> {
        const CHUNK: usize = 16;

        debug_assert!(weights.len() >= PADDED_INPUT_DIMS);

        let packed_len = PADDED_INPUT_DIMS.div_ceil(CHUNK) * CHUNK * 2;
        let mut packed = AlignedBuffer::<u8, CACHE_LINE_SIZE>::from_elem(0, packed_len);
        let mut offset = 0usize;
        while offset < PADDED_INPUT_DIMS {
            let packed_offset = offset * 2;
            let chunk_len = (PADDED_INPUT_DIMS - offset).min(CHUNK);
            for lane in 0..chunk_len {
                let weight = weights[offset + lane];
                packed[packed_offset + lane] = weight as u8;
                packed[packed_offset + CHUNK + lane] = (weight >> 8) as u8;
            }
            offset += CHUNK;
        }

        packed
    }

    /// Selects the optimal forward implementation based on CPU features.
    ///
    /// Selection priority (highest to lowest):
    /// 1. AVX-512 with VNNI (if compiled with `target-feature=+avx512bw` and CPU supports VNNI)
    /// 2. AVX-512 without VNNI (if compiled with `target-feature=+avx512bw`)
    /// 3. AVX2 with VNNI (if compiled with `target-feature=+avx2` and CPU supports `avxvnni`)
    /// 4. AVX2 without VNNI (if compiled with `target-feature=+avx2`)
    /// 5. ARM NEON with I8MM/DotProd (if compiled for `aarch64` with NEON and CPU supports both)
    /// 6. ARM NEON (if compiled for `aarch64` with `target-feature=+neon`)
    /// 7. Scalar fallback (all other cases)
    fn select_forward_fn() -> unsafe fn(&Self, [&[u8]; 3]) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            cfg_select! {
                target_feature = "avx512bw" => {
                    use std::arch::is_x86_feature_detected;
                    if is_x86_feature_detected!("avx512vnni") {
                        Self::forward_avx512_vnni
                    } else {
                        Self::forward_avx512_no_vnni
                    }
                }
                target_feature = "avx2" => {
                    use std::arch::is_x86_feature_detected;
                    if is_x86_feature_detected!("avxvnni") {
                        Self::forward_avx2_vnni
                    } else {
                        Self::forward_avx2_no_vnni
                    }
                }
                _ => {
                    Self::forward_scalar_wrapper
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            cfg_select! {
                target_feature = "neon" => {
                    use std::arch::is_aarch64_feature_detected;

                    if is_aarch64_feature_detected!("i8mm")
                        && is_aarch64_feature_detected!("dotprod")
                    {
                        Self::forward_neon_i8mm
                    } else {
                        Self::forward_neon
                    }
                }
                _ => {
                    Self::forward_scalar_wrapper
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::forward_scalar_wrapper
        }
    }

    /// Computes the weighted sum of 8-bit inputs with 16-bit weights, plus the bias term.
    ///
    /// # Preconditions
    ///
    /// - The sum of all segment lengths must equal `INPUT_DIMS`.
    /// - On AVX-512 builds (`target-feature=+avx512bw`), each non-empty segment must:
    ///   - have length as a multiple of 32, and
    ///   - start at a 32-byte aligned address.
    /// - On AVX2 builds (`target-feature=+avx2`), each non-empty segment must:
    ///   - have length as a multiple of 16, and
    ///   - start at a 16-byte aligned address.
    /// - On NEON builds (`target_arch=aarch64`, `target-feature=+neon`), each non-empty
    ///   segment must have length as a multiple of 16.
    #[inline(always)]
    pub fn forward(&self, segments: [&[u8]; 3]) -> i32 {
        debug_assert_eq!(
            segments.iter().map(|segment| segment.len()).sum::<usize>(),
            INPUT_DIMS
        );

        unsafe { (self.forward_fn)(self, segments) }
    }

    /// Computes the forward pass on AVX-512 with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw,avx512vnni")]
    fn forward_avx512_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx512::<true>(segments)
    }

    /// Computes the forward pass on AVX-512 without VNNI (emulated via `VPMADDWD` + `VPADDD`).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512_no_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx512::<false>(segments)
    }

    /// Computes the forward pass on AVX2 with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,avxvnni")]
    #[allow(dead_code)]
    fn forward_avx2_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx2::<true>(segments)
    }

    /// Computes the forward pass on AVX2 without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2_no_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx2::<false>(segments)
    }

    /// Computes the forward pass on ARM NEON.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    #[inline]
    fn forward_neon(&self, segments: [&[u8]; 3]) -> i32 {
        use std::arch::aarch64::*;

        const CHUNK: usize = 16;
        const UNROLL: usize = 2;

        unsafe {
            // Eight independent i32x4 accumulators expose ILP across SIMD issue ports.
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut acc2 = vdupq_n_s32(0);
            let mut acc3 = vdupq_n_s32(0);
            let mut acc4 = vdupq_n_s32(0);
            let mut acc5 = vdupq_n_s32(0);
            let mut acc6 = vdupq_n_s32(0);
            let mut acc7 = vdupq_n_s32(0);

            let mut weight_offset = 0usize;
            for input in segments {
                let len = input.len();
                debug_assert_eq!(len % CHUNK, 0);

                let num_chunks = len / CHUNK;
                let input_ptr = input.as_ptr();
                let weight_ptr = self.weights.as_ptr().add(weight_offset);
                let mut chunk_idx = 0usize;

                macro_rules! accumulate_chunk {
                    ($input_u8:expr, $w_lo:expr, $w_hi:expr, $a0:ident, $a1:ident, $a2:ident, $a3:ident) => {{
                        let input_u8 = $input_u8;

                        // Zero-extend u8 -> i16 (values in [0, 255] fit in i16).
                        let inp_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(input_u8)));
                        let inp_hi = vreinterpretq_s16_u16(vmovl_high_u8(input_u8));

                        let w_lo = $w_lo;
                        let w_hi = $w_hi;

                        $a0 = vmlal_s16($a0, vget_low_s16(inp_lo), vget_low_s16(w_lo));
                        $a1 = vmlal_high_s16($a1, inp_lo, w_lo);
                        $a2 = vmlal_s16($a2, vget_low_s16(inp_hi), vget_low_s16(w_hi));
                        $a3 = vmlal_high_s16($a3, inp_hi, w_hi);
                    }};
                }

                while chunk_idx + UNROLL <= num_chunks {
                    let offset = chunk_idx * CHUNK;
                    let input_u8 = vld1q_u8_x2(input_ptr.add(offset));
                    let weights = vld1q_s16_x4(weight_ptr.add(offset));

                    accumulate_chunk!(input_u8.0, weights.0, weights.1, acc0, acc1, acc2, acc3);
                    accumulate_chunk!(input_u8.1, weights.2, weights.3, acc4, acc5, acc6, acc7);

                    chunk_idx += UNROLL;
                }

                while chunk_idx < num_chunks {
                    let offset = chunk_idx * CHUNK;
                    let input_u8 = vld1q_u8(input_ptr.add(offset));
                    let weights = vld1q_s16_x2(weight_ptr.add(offset));

                    accumulate_chunk!(input_u8, weights.0, weights.1, acc0, acc1, acc2, acc3);
                    chunk_idx += 1;
                }

                weight_offset += len;
            }

            acc0 = vaddq_s32(acc0, acc4);
            acc1 = vaddq_s32(acc1, acc5);
            acc2 = vaddq_s32(acc2, acc6);
            acc3 = vaddq_s32(acc3, acc7);

            let sum01 = vaddq_s32(acc0, acc1);
            let sum23 = vaddq_s32(acc2, acc3);
            vaddvq_s32(vaddq_s32(sum01, sum23)) + self.bias
        }
    }

    /// Computes the forward pass on ARM NEON using I8MM/DotProd instructions.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon,dotprod,i8mm")]
    #[inline]
    fn forward_neon_i8mm(&self, segments: [&[u8]; 3]) -> i32 {
        use std::arch::aarch64::*;

        const CHUNK: usize = 16;
        const UNROLL: usize = 4;

        unsafe {
            // `i8mm_weights` stores each i16 chunk as low bytes followed by
            // signed high bytes. `udot` handles the low byte, `usdot` handles
            // the high byte, and the two partial dot products are recombined in
            // i32 lanes as low + (high << 8).
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut acc2 = vdupq_n_s32(0);
            let mut acc3 = vdupq_n_s32(0);

            let zero_u32 = vdupq_n_u32(0);
            let zero_s32 = vdupq_n_s32(0);

            let mut weight_offset = 0usize;
            for input in segments {
                let len = input.len();
                debug_assert_eq!(len % CHUNK, 0);

                let num_chunks = len / CHUNK;
                let input_ptr = input.as_ptr();
                let weight_ptr = self.i8mm_weights.as_ptr().add(weight_offset * 2);
                let mut chunk_idx = 0usize;

                macro_rules! accumulate_chunk {
                    ($input_u8:expr, $weight_low:expr, $weight_high:expr, $acc:ident) => {{
                        let input_u8 = $input_u8;
                        let weight_low = $weight_low;
                        let weight_high = vreinterpretq_s8_u8($weight_high);

                        let low = vreinterpretq_s32_u32(vdotq_u32(zero_u32, input_u8, weight_low));
                        let high = vshlq_n_s32::<8>(vusdotq_s32(zero_s32, input_u8, weight_high));
                        $acc = vaddq_s32($acc, vaddq_s32(low, high));
                    }};
                }

                while chunk_idx + UNROLL <= num_chunks {
                    let offset = chunk_idx * CHUNK;
                    let input01_u8 = vld1q_u8_x2(input_ptr.add(offset));
                    let packed01 = vld1q_u8_x4(weight_ptr.add(offset * 2));

                    accumulate_chunk!(input01_u8.0, packed01.0, packed01.1, acc0);
                    accumulate_chunk!(input01_u8.1, packed01.2, packed01.3, acc1);

                    let input23_u8 = vld1q_u8_x2(input_ptr.add(offset + CHUNK * 2));
                    let packed23 = vld1q_u8_x4(weight_ptr.add((offset + CHUNK * 2) * 2));

                    accumulate_chunk!(input23_u8.0, packed23.0, packed23.1, acc2);
                    accumulate_chunk!(input23_u8.1, packed23.2, packed23.3, acc3);

                    chunk_idx += UNROLL;
                }

                while chunk_idx < num_chunks {
                    let offset = chunk_idx * CHUNK;
                    let input_u8 = vld1q_u8(input_ptr.add(offset));
                    let packed = vld1q_u8_x2(weight_ptr.add(offset * 2));

                    accumulate_chunk!(input_u8, packed.0, packed.1, acc0);
                    chunk_idx += 1;
                }

                weight_offset += len;
            }

            acc0 = vaddq_s32(acc0, acc1);
            acc2 = vaddq_s32(acc2, acc3);
            vaddvq_s32(vaddq_s32(acc0, acc2)) + self.bias
        }
    }

    /// Computes the forward pass using the scalar fallback.
    ///
    /// # Safety
    ///
    /// This wrapper has no additional safety requirements; it exists only to
    /// match the `unsafe fn` signature of the SIMD forward implementations.
    #[allow(dead_code)]
    unsafe fn forward_scalar_wrapper(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_scalar(segments)
    }

    /// Computes the forward pass using the AVX-512 implementation.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    #[inline]
    fn forward_avx512<const USE_VNNI: bool>(&self, segments: [&[u8]; 3]) -> i32 {
        use crate::eval::util::mm512_dpwssd_epi32;
        use std::arch::x86_64::*;

        const CHUNK: usize = 32;
        const UNROLL: usize = 4;

        let mut acc0 = _mm512_setzero_si512();
        let mut acc1 = _mm512_setzero_si512();
        let mut acc2 = _mm512_setzero_si512();
        let mut acc3 = _mm512_setzero_si512();

        unsafe {
            let mut weight_offset = 0usize;
            for input in segments {
                let len = input.len();
                debug_assert_eq!(len % CHUNK, 0);

                let num_chunks = len / CHUNK;
                let input_ptr = input.as_ptr() as *const __m256i;
                let weight_ptr = self.weights.as_ptr().add(weight_offset) as *const __m512i;
                let mut chunk_idx = 0usize;

                // Main loop (4-way unrolled)
                while chunk_idx + UNROLL <= num_chunks {
                    let input_u8_0 = _mm256_load_si256(input_ptr.add(chunk_idx));
                    let input_low_0 = _mm512_cvtepu8_epi16(input_u8_0);
                    let weight_0 = _mm512_load_si512(weight_ptr.add(chunk_idx));
                    acc0 = mm512_dpwssd_epi32::<USE_VNNI>(acc0, input_low_0, weight_0);

                    let input_u8_1 = _mm256_load_si256(input_ptr.add(chunk_idx + 1));
                    let input_low_1 = _mm512_cvtepu8_epi16(input_u8_1);
                    let weight_1 = _mm512_load_si512(weight_ptr.add(chunk_idx + 1));
                    acc1 = mm512_dpwssd_epi32::<USE_VNNI>(acc1, input_low_1, weight_1);

                    let input_u8_2 = _mm256_load_si256(input_ptr.add(chunk_idx + 2));
                    let input_low_2 = _mm512_cvtepu8_epi16(input_u8_2);
                    let weight_2 = _mm512_load_si512(weight_ptr.add(chunk_idx + 2));
                    acc2 = mm512_dpwssd_epi32::<USE_VNNI>(acc2, input_low_2, weight_2);

                    let input_u8_3 = _mm256_load_si256(input_ptr.add(chunk_idx + 3));
                    let input_low_3 = _mm512_cvtepu8_epi16(input_u8_3);
                    let weight_3 = _mm512_load_si512(weight_ptr.add(chunk_idx + 3));
                    acc3 = mm512_dpwssd_epi32::<USE_VNNI>(acc3, input_low_3, weight_3);

                    chunk_idx += UNROLL;
                }

                // Remainder
                while chunk_idx < num_chunks {
                    let input_u8 = _mm256_load_si256(input_ptr.add(chunk_idx));
                    let input_i16 = _mm512_cvtepu8_epi16(input_u8);
                    let weight = _mm512_load_si512(weight_ptr.add(chunk_idx));
                    acc0 = mm512_dpwssd_epi32::<USE_VNNI>(acc0, input_i16, weight);

                    chunk_idx += 1;
                }

                weight_offset += len;
            }
        }

        acc0 = _mm512_add_epi32(acc0, acc1);
        acc2 = _mm512_add_epi32(acc2, acc3);
        acc0 = _mm512_add_epi32(acc0, acc2);

        _mm512_reduce_add_epi32(acc0) + self.bias
    }

    /// Computes the forward pass using the AVX2 implementation.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[inline]
    fn forward_avx2<const USE_VNNI: bool>(&self, segments: [&[u8]; 3]) -> i32 {
        use crate::eval::util::{m256_hadd, mm256_dpwssd_epi32};
        use std::arch::x86_64::*;

        const CHUNK: usize = 16;
        const UNROLL: usize = 4;

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();

        unsafe {
            let mut weight_offset = 0usize;
            for input in segments {
                let len = input.len();
                debug_assert_eq!(len % CHUNK, 0);

                let num_chunks = len / CHUNK;
                let input_ptr = input.as_ptr() as *const __m128i;
                let weight_ptr = self.weights.as_ptr().add(weight_offset) as *const __m256i;
                let mut chunk_idx = 0usize;

                // Main loop (4-way unrolled)
                while chunk_idx + UNROLL <= num_chunks {
                    let input_chunk0 = _mm_load_si128(input_ptr.add(chunk_idx));
                    let input_i16_0 = _mm256_cvtepu8_epi16(input_chunk0);
                    let weight_chunk0 = _mm256_load_si256(weight_ptr.add(chunk_idx));
                    acc0 = mm256_dpwssd_epi32::<USE_VNNI>(acc0, input_i16_0, weight_chunk0);

                    let input_chunk1 = _mm_load_si128(input_ptr.add(chunk_idx + 1));
                    let input_i16_1 = _mm256_cvtepu8_epi16(input_chunk1);
                    let weight_chunk1 = _mm256_load_si256(weight_ptr.add(chunk_idx + 1));
                    acc1 = mm256_dpwssd_epi32::<USE_VNNI>(acc1, input_i16_1, weight_chunk1);

                    let input_chunk2 = _mm_load_si128(input_ptr.add(chunk_idx + 2));
                    let input_i16_2 = _mm256_cvtepu8_epi16(input_chunk2);
                    let weight_chunk2 = _mm256_load_si256(weight_ptr.add(chunk_idx + 2));
                    acc2 = mm256_dpwssd_epi32::<USE_VNNI>(acc2, input_i16_2, weight_chunk2);

                    let input_chunk3 = _mm_load_si128(input_ptr.add(chunk_idx + 3));
                    let input_i16_3 = _mm256_cvtepu8_epi16(input_chunk3);
                    let weight_chunk3 = _mm256_load_si256(weight_ptr.add(chunk_idx + 3));
                    acc3 = mm256_dpwssd_epi32::<USE_VNNI>(acc3, input_i16_3, weight_chunk3);

                    chunk_idx += UNROLL;
                }

                // Remainder
                while chunk_idx < num_chunks {
                    let input_chunk = _mm_load_si128(input_ptr.add(chunk_idx));
                    let input_i16 = _mm256_cvtepu8_epi16(input_chunk);
                    let weight_chunk = _mm256_load_si256(weight_ptr.add(chunk_idx));
                    acc0 = mm256_dpwssd_epi32::<USE_VNNI>(acc0, input_i16, weight_chunk);

                    chunk_idx += 1;
                }

                weight_offset += len;
            }
        }

        acc0 = _mm256_add_epi32(acc0, acc1);
        acc2 = _mm256_add_epi32(acc2, acc3);
        acc0 = _mm256_add_epi32(acc0, acc2);

        m256_hadd(acc0) + self.bias
    }

    /// Computes the forward pass using the scalar fallback for non-SIMD architectures or testing.
    fn forward_scalar(&self, segments: [&[u8]; 3]) -> i32 {
        let mut acc = self.bias;

        let mut weight_offset = 0usize;
        for segment in segments {
            for (i, &value) in segment.iter().enumerate() {
                acc += (value as i32) * (self.weights[weight_offset + i] as i32);
            }
            weight_offset += segment.len();
        }

        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::align::Align64;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Cursor;

    fn build_layer<const INPUT: usize, const PADDED: usize>(
        bias: i32,
        seed: i32,
    ) -> OutputLayer<INPUT, PADDED> {
        let mut weights = AlignedBuffer::<i16, CACHE_LINE_SIZE>::from_elem(0, PADDED);
        for (idx, weight) in weights.iter_mut().take(INPUT).enumerate() {
            *weight = ((idx as i32 * 4099 + seed).rem_euclid(60_001) - 30_000) as i16;
        }

        #[cfg(target_arch = "aarch64")]
        let i8mm_weights = OutputLayer::<INPUT, PADDED>::pack_i8mm_weights(weights.as_slice());

        OutputLayer {
            bias,
            weights,
            #[cfg(target_arch = "aarch64")]
            i8mm_weights,
            forward_fn: OutputLayer::<INPUT, PADDED>::select_forward_fn(),
        }
    }

    fn reference_forward<const INPUT: usize, const PADDED: usize>(
        layer: &OutputLayer<INPUT, PADDED>,
        segments: [&[u8]; 3],
    ) -> i32 {
        let mut acc = layer.bias;
        let mut weight_offset = 0;
        for segment in segments {
            for (idx, &value) in segment.iter().enumerate() {
                acc += i32::from(value) * i32::from(layer.weights[weight_offset + idx]);
            }
            weight_offset += segment.len();
        }
        debug_assert_eq!(weight_offset, INPUT);
        acc
    }

    type Chunk16Segments = (Align64<[u8; 16]>, Align64<[u8; 32]>, Align64<[u8; 16]>);
    type Chunk32Segments = (Align64<[u8; 32]>, Align64<[u8; 32]>);

    fn chunk16_segments() -> Chunk16Segments {
        let seg0 = Align64([
            3, 17, 31, 45, 59, 73, 87, 101, 115, 129, 143, 157, 171, 185, 199, 213,
        ]);
        let seg1 = Align64([
            255, 200, 150, 100, 50, 25, 12, 6, 3, 1, 0, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
            29, 31, 33, 35, 37, 39, 41, 43, 45, 47,
        ]);
        let seg2 = Align64([8, 6, 7, 5, 3, 0, 9, 11, 13, 15, 17, 19, 23, 29, 31, 37]);
        (seg0, seg1, seg2)
    }

    #[allow(dead_code)]
    fn chunk32_segments() -> Chunk32Segments {
        let seg0 = Align64([
            3, 17, 31, 45, 59, 73, 87, 101, 115, 129, 143, 157, 171, 185, 199, 213, 227, 241, 255,
            5, 19, 33, 47, 61, 75, 89, 103, 117, 131, 145, 159, 173,
        ]);
        let seg1 = Align64([
            255, 200, 150, 100, 50, 25, 12, 6, 3, 1, 0, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27,
            29, 31, 33, 35, 37, 39, 41, 43, 45, 47,
        ]);
        (seg0, seg1)
    }

    #[test]
    fn load_reads_bias_and_every_padded_weight() {
        const INPUT: usize = 5;
        const PADDED: usize = 8;
        let bias = -123_456;
        let weights = [-30000, -257, -1, 0, 1, 255, 1024, 30_000];
        let mut data = Vec::new();
        data.write_i32::<LittleEndian>(bias).unwrap();
        for weight in weights {
            data.write_i16::<LittleEndian>(weight).unwrap();
        }

        let mut cursor = Cursor::new(data);
        let layer = OutputLayer::<INPUT, PADDED>::load(&mut cursor).unwrap();

        assert_eq!(layer.bias, bias);
        assert_eq!(&layer.weights[..], &weights);
    }

    #[test]
    fn load_reports_truncated_biases_or_weights() {
        type Layer = OutputLayer<4, 8>;

        let mut missing_bias = Cursor::new([0u8; 3]);
        assert!(Layer::load(&mut missing_bias).is_err());

        let mut missing_weight = Vec::new();
        missing_weight.write_i32::<LittleEndian>(1).unwrap();
        for _ in 0..7 {
            missing_weight.write_i16::<LittleEndian>(2).unwrap();
        }
        assert!(Layer::load(&mut Cursor::new(missing_weight)).is_err());
    }

    #[test]
    fn forward_scalar_matches_reference_for_arbitrary_segment_lengths() {
        const INPUT: usize = 7;
        const PADDED: usize = 8;
        let layer = build_layer::<INPUT, PADDED>(1234, 17);
        let seg0 = [255, 0, 7];
        let seg1 = [];
        let seg2 = [1, 2, 3, 4];
        let segments = [&seg0[..], &seg1[..], &seg2[..]];
        let expected = reference_forward(&layer, segments);

        assert_eq!(layer.forward_scalar(segments), expected);
    }

    #[test]
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
    fn forward_dispatch_matches_reference_for_16_byte_chunked_segments() {
        const INPUT: usize = 64;
        const PADDED: usize = 64;
        let layer = build_layer::<INPUT, PADDED>(-6789, 29);
        let (seg0, seg1, seg2) = chunk16_segments();
        let segments = [&seg0.0[..], &seg1.0[..], &seg2.0[..]];
        let expected = reference_forward(&layer, segments);

        assert_eq!(layer.forward(segments), expected);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn forward_dispatch_matches_reference_for_32_byte_chunked_segments() {
        const INPUT: usize = 64;
        const PADDED: usize = 64;
        let layer = build_layer::<INPUT, PADDED>(-6789, 29);
        let (seg0, seg1) = chunk32_segments();
        let segments = [&seg0.0[..], &seg1.0[..], &[][..]];
        let expected = reference_forward(&layer, segments);

        assert_eq!(layer.forward(segments), expected);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn forward_rejects_segment_lengths_that_do_not_sum_to_input_dims() {
        let layer = build_layer::<16, 16>(0, 1);
        let seg0 = Align64([0u8; 16]);
        let extra = [1u8];

        let _ = layer.forward([&seg0.0[..], &extra[..], &[][..]]);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn pack_i8mm_weights_splits_low_and_high_bytes_by_chunk() {
        const INPUT: usize = 20;
        const PADDED: usize = 20;
        let mut weights = [0i16; PADDED];
        weights[0] = 0x1234;
        weights[1] = -2;
        weights[15] = i16::MIN;
        weights[16] = i16::MAX;
        weights[17] = -30_000;

        let packed = OutputLayer::<INPUT, PADDED>::pack_i8mm_weights(&weights);

        assert_eq!(packed.len(), 64);
        assert_eq!(packed[0], 0x34);
        assert_eq!(packed[16], 0x12);
        assert_eq!(packed[1], 0xFE);
        assert_eq!(packed[17], 0xFF);
        assert_eq!(packed[15], 0x00);
        assert_eq!(packed[31], 0x80);
        assert_eq!(packed[32], 0xFF);
        assert_eq!(packed[48], 0x7F);
        assert_eq!(packed[33], weights[17] as u8);
        assert_eq!(packed[49], (weights[17] >> 8) as u8);
        assert!(packed[34..48].iter().all(|&byte| byte == 0));
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn neon_forward_kernels_match_reference_for_chunked_segments() {
        const INPUT: usize = 64;
        const PADDED: usize = 64;
        let layer = build_layer::<INPUT, PADDED>(-6789, 4099);
        let (seg0, seg1, seg2) = chunk16_segments();
        let segments = [&seg0.0[..], &seg1.0[..], &seg2.0[..]];
        let expected = reference_forward(&layer, segments);

        unsafe {
            assert_eq!(layer.forward_neon(segments), expected, "neon");
        }

        if std::arch::is_aarch64_feature_detected!("i8mm")
            && std::arch::is_aarch64_feature_detected!("dotprod")
        {
            unsafe {
                assert_eq!(layer.forward_neon_i8mm(segments), expected, "i8mm");
            }

            let mut single = Align64([0; INPUT]);
            for (idx, value) in single.iter_mut().enumerate() {
                *value = ((idx * 37 + 11) & 0xff) as u8;
            }
            let single_segments = [&single.0[..], &[][..], &[][..]];
            let single_expected = reference_forward(&layer, single_segments);
            unsafe {
                assert_eq!(
                    layer.forward_neon_i8mm(single_segments),
                    single_expected,
                    "i8mm unrolled",
                );
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn avx2_forward_kernels_match_reference_for_chunked_segments() {
        const INPUT: usize = 64;
        const PADDED: usize = 64;
        let layer = build_layer::<INPUT, PADDED>(-6789, 29);
        let (seg0, seg1, seg2) = chunk16_segments();
        let segments = [&seg0.0[..], &seg1.0[..], &seg2.0[..]];
        let expected = reference_forward(&layer, segments);

        unsafe {
            assert_eq!(layer.forward_avx2_no_vnni(segments), expected, "avx2");
        }

        if std::arch::is_x86_feature_detected!("avxvnni") {
            unsafe {
                assert_eq!(layer.forward_avx2_vnni(segments), expected, "avx2 vnni");
            }
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    fn avx512_forward_kernels_match_reference_for_chunked_segments() {
        const INPUT: usize = 64;
        const PADDED: usize = 64;
        let layer = build_layer::<INPUT, PADDED>(-6789, 29);
        let (seg0, seg1) = chunk32_segments();
        let segments = [&seg0.0[..], &seg1.0[..], &[][..]];
        let expected = reference_forward(&layer, segments);

        unsafe {
            assert_eq!(layer.forward_avx512_no_vnni(segments), expected, "avx512");
        }

        if std::arch::is_x86_feature_detected!("avx512vnni") {
            unsafe {
                assert_eq!(layer.forward_avx512_vnni(segments), expected, "avx512 vnni",);
            }
        }
    }
}
