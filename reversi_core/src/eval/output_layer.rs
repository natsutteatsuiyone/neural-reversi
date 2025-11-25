//! Specialized output layer processing 16-bit weights for a single neuron.

use std::io::{self, Read};

use aligned_vec::{AVec, ConstAlign, avec};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::constants::CACHE_LINE_SIZE;

/// Output layer backed by 16-bit weights with a single accumulator.
pub struct OutputLayer<const INPUT_DIMS: usize, const PADDED_INPUT_DIMS: usize> {
    bias: i32,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    forward_fn: unsafe fn(&Self, [&[u8]; 3]) -> i32,
}

impl<const INPUT_DIMS: usize, const PADDED_INPUT_DIMS: usize>
    OutputLayer<INPUT_DIMS, PADDED_INPUT_DIMS>
{
    /// Loads bias and weights (little endian) for the output layer.
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let bias = reader.read_i32::<LittleEndian>()?;

        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; PADDED_INPUT_DIMS];
        for weight in weights.iter_mut().take(PADDED_INPUT_DIMS) {
            *weight = reader.read_i16::<LittleEndian>()?;
        }

        // Select the optimal forward implementation at load time
        let forward_fn = Self::select_forward_fn();

        Ok(Self {
            bias,
            weights,
            forward_fn,
        })
    }

    /// Selects the optimal forward implementation based on CPU features.
    fn select_forward_fn() -> unsafe fn(&Self, [&[u8]; 3]) -> i32 {
        #[cfg(target_arch = "x86_64")]
        {
            use cfg_if::cfg_if;
            use std::arch::is_x86_feature_detected;

            cfg_if! {
                if #[cfg(target_feature = "avx512bw")] {
                    if is_x86_feature_detected!("avx512vnni") {
                        Self::forward_avx512_vnni
                    } else {
                        Self::forward_avx512_no_vnni
                    }
                } else if #[cfg(target_feature = "avx2")] {
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

    /// Computes the dot product between 8-bit inputs and 16-bit weights without
    /// requiring a dedicated contiguous staging buffer.
    #[inline(always)]
    pub fn forward(&self, segments: [&[u8]; 3]) -> i32 {
        debug_assert_eq!(
            segments.iter().map(|segment| segment.len()).sum::<usize>(),
            INPUT_DIMS
        );

        unsafe { (self.forward_fn)(self, segments) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw,avx512vnni")]
    fn forward_avx512_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx512::<true>(segments)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    fn forward_avx512_no_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx512::<false>(segments)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2,avxvnni")]
    #[allow(dead_code)]
    fn forward_avx2_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx2::<true>(segments)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn forward_avx2_no_vnni(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_avx2::<false>(segments)
    }

    /// Wrapper for forward_scalar to match the unsafe fn signature.
    #[allow(dead_code)]
    unsafe fn forward_scalar_wrapper(&self, segments: [&[u8]; 3]) -> i32 {
        self.forward_scalar(segments)
    }

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
