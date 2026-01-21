//! Output layer with 16-bit weights for neural network evaluation.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;

/// Output layer with 16-bit weights for computing a single scalar output.
///
/// # Type Parameters
///
/// * `INPUT_DIMS` - The actual number of input dimensions
/// * `PADDED_INPUT_DIMS` - The padded dimension count for SIMD alignment (must be >= INPUT_DIMS
///   and a multiple of 32)
pub struct OutputLayer<const INPUT_DIMS: usize, const PADDED_INPUT_DIMS: usize> {
    /// Bias term for the output neuron.
    bias: i32,
    /// Weight vector aligned for SIMD access.
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    /// Function pointer to the optimal forward implementation, selected at load time
    /// based on detected CPU SIMD capabilities (AVX512+VNNI > AVX512 > AVX2+VNNI > AVX2 > scalar).
    forward_fn: unsafe fn(&Self, [&[u8]; 3]) -> i32,
}

impl<const INPUT_DIMS: usize, const PADDED_INPUT_DIMS: usize>
    OutputLayer<INPUT_DIMS, PADDED_INPUT_DIMS>
{
    /// Loads the output layer from a binary reader.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let bias = reader.read_i32::<LittleEndian>()?;

        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; PADDED_INPUT_DIMS];
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        let forward_fn = Self::select_forward_fn();

        Ok(Self {
            bias,
            weights,
            forward_fn,
        })
    }

    /// Selects the optimal forward implementation based on CPU features.
    ///
    /// Selection priority (highest to lowest):
    /// 1. AVX-512 with VNNI (if compiled with `target-feature=+avx512bw` and CPU supports VNNI)
    /// 2. AVX-512 without VNNI (if compiled with `target-feature=+avx512bw`)
    /// 3. AVX2 with VNNI (if compiled with `target-feature=+avx2` and CPU supports AVXVNNI)
    /// 4. AVX2 without VNNI (if compiled with `target-feature=+avx2`)
    /// 5. Scalar fallback (all other cases)
    fn select_forward_fn() -> unsafe fn(&Self, [&[u8]; 3]) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            use cfg_if::cfg_if;

            cfg_if! {
                if #[cfg(target_feature = "avx512bw")] {
                    use std::arch::is_x86_feature_detected;
                    if is_x86_feature_detected!("avx512vnni") {
                        Self::forward_avx512_vnni
                    } else {
                        Self::forward_avx512_no_vnni
                    }
                } else if #[cfg(target_feature = "avx2")] {
                    use std::arch::is_x86_feature_detected;
                    if is_x86_feature_detected!("avxvnni") {
                        Self::forward_avx2_vnni
                    } else {
                        Self::forward_avx2_no_vnni
                    }
                } else {
                    Self::forward_scalar_wrapper
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::forward_scalar_wrapper
        }
    }

    /// Computes the weighted sum of 8-bit inputs with 16-bit weights, plus the bias term.
    ///
    /// # Arguments
    ///
    /// * `segments` - Three input segments `[l2_output, base_output, pa_output]`.
    #[inline(always)]
    pub fn forward(&self, segments: [&[u8]; 3]) -> i32 {
        debug_assert_eq!(
            segments.iter().map(|segment| segment.len()).sum::<usize>(),
            INPUT_DIMS
        );

        unsafe { (self.forward_fn)(self, segments) }
    }

    /// AVX-512 forward pass with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw,avx512vnni")]
    fn forward_avx512_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx512::<true>(segments)
    }

    /// AVX-512 forward pass without VNNI (emulated via `VPMADDWD` + `VPADDD`).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512_no_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx512::<false>(segments)
    }

    /// AVX2 forward pass with VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,avxvnni")]
    #[allow(dead_code)]
    fn forward_avx2_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx2::<true>(segments)
    }

    /// AVX2 forward pass without VNNI.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2_no_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx2::<false>(segments)
    }

    /// Scalar fallback.
    #[allow(dead_code)]
    unsafe fn forward_scalar_wrapper(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_scalar(segments)
    }

    /// AVX-512 implementation of the forward pass.
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

    /// AVX2 implementation of the forward pass.
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

    /// Scalar fallback implementation for non-SIMD architectures or testing.
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
