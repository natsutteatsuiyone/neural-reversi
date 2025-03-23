use std::io::{self, Read};
use std::mem::size_of;

use aligned_vec::{avec, AVec, ConstAlign};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::CACHE_LINE_SIZE;

use super::constants::{INPUT_FEATURE_DIMS, L1_PS_INPUT_DIMS};

const OUTPUT_SIZE: usize = L1_PS_INPUT_DIMS;
const NUM_REGS: usize = OUTPUT_SIZE / 16;

#[derive(Debug)]
pub struct PhaseAdaptiveInput {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl PhaseAdaptiveInput
{
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_SIZE];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_FEATURE_DIMS * OUTPUT_SIZE];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // permute weights and biases for AVX2
                let num_chunks = (OUTPUT_SIZE * size_of::<i16>()) / size_of::<u64>();
                let bias_slice: &mut [u64] =
                    std::slice::from_raw_parts_mut(biases.as_mut_ptr() as *mut u64, num_chunks);

                for i in 0..num_chunks/8 {
                    let base = i * 8;
                    bias_slice.swap(base + 2, base + 4);
                    bias_slice.swap(base + 3, base + 5);
                }

                for i in 0..INPUT_FEATURE_DIMS {
                    let ptr = weights.as_mut_ptr().add(i * OUTPUT_SIZE) as *mut u64;
                    let weight_slice: &mut [u64] = std::slice::from_raw_parts_mut(ptr, num_chunks);

                    for j in 0..num_chunks/8 {
                        let base = j * 8;
                        weight_slice.swap(base + 2, base + 4);
                        weight_slice.swap(base + 3, base + 5);
                    }
                }
            }
        }

        Ok(PhaseAdaptiveInput { biases, weights })
    }

    pub fn forward(&self, feature_indices: &[u16], output: &mut [u8]) {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.forward_avx2(feature_indices, output);
            }
            return;
        }

        let mut acc = [0; OUTPUT_SIZE];
        for fi in feature_indices {
            let weights = &self.weights[(*fi as usize) * OUTPUT_SIZE..][..OUTPUT_SIZE];
            for (a, w) in acc.iter_mut().zip(weights.iter()) {
                *a += *w;
            }
        }

        for i in 0..OUTPUT_SIZE {
            output[i] = (acc[i] + self.biases[i]).clamp(0, 127) as u8;
        }
    }

    unsafe fn forward_avx2(&self, feature_indices: &[u16], output: &mut [u8]) {
        use std::arch::x86_64::*;
        let mut acc: [__m256i; NUM_REGS] = std::mem::zeroed();

        std::ptr::copy_nonoverlapping(
            self.biases.as_ptr() as *const __m256i,
            acc.as_mut_ptr(),
            NUM_REGS,
        );

        let weight_ptr = self.weights.as_ptr();
        let len = feature_indices.len();

        for i in 0..len {
            let idx = *feature_indices.get_unchecked(i) as usize;
            let weight_ptr = weight_ptr.add(idx * OUTPUT_SIZE) as *const __m256i;

            for j in 0..NUM_REGS {
                *acc.get_unchecked_mut(j) = _mm256_add_epi16(
                    *acc.get_unchecked(j),
                    _mm256_loadu_si256(weight_ptr.add(j))
                );
            }
        }

        let output_ptr = output.as_mut_ptr() as *mut __m256i;
        let one = _mm256_set1_epi16(127);
        let zero = _mm256_setzero_si256();
        let mut j = 0;
        let mut out_idx = 0;
        while j < NUM_REGS {
            let pa = _mm256_min_epi16(_mm256_max_epi16(acc[j], zero), one);
            let pb = _mm256_min_epi16(_mm256_max_epi16(acc[j + 1], zero), one);
            *output_ptr.add(out_idx) = _mm256_packus_epi16(pa, pb);
            j += 2;
            out_idx += 1;
        }
    }
}
