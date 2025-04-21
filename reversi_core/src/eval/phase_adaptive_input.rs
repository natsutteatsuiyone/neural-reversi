use std::io::{self, Read};
use std::mem::size_of;

use aligned_vec::{avec, AVec, ConstAlign};
use byteorder::{LittleEndian, ReadBytesExt};

use crate::eval::CACHE_LINE_SIZE;


#[derive(Debug)]
pub struct PhaseAdaptiveInput<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const NUM_REGS: usize,
> {
    biases: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
    weights: AVec<i16, ConstAlign<CACHE_LINE_SIZE>>,
}

impl<
    const INPUT_DIMS: usize,
    const OUTPUT_DIMS: usize,
    const NUM_REGS: usize,
> PhaseAdaptiveInput<INPUT_DIMS, OUTPUT_DIMS, NUM_REGS>
{
    pub fn load<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut biases = avec![[CACHE_LINE_SIZE]|0i16; OUTPUT_DIMS];
        let mut weights = avec![[CACHE_LINE_SIZE]|0i16; INPUT_DIMS * OUTPUT_DIMS];

        reader.read_i16_into::<LittleEndian>(&mut biases)?;
        reader.read_i16_into::<LittleEndian>(&mut weights)?;

        if is_x86_feature_detected!("avx2") {
            unsafe {
                // permute weights and biases for AVX2
                let num_chunks = (OUTPUT_DIMS * size_of::<i16>()) / size_of::<u64>();
                let bias_slice: &mut [u64] =
                    std::slice::from_raw_parts_mut(biases.as_mut_ptr() as *mut u64, num_chunks);

                for i in 0..num_chunks/8 {
                    let base = i * 8;
                    bias_slice.swap(base + 2, base + 4);
                    bias_slice.swap(base + 3, base + 5);
                }

                for i in 0..INPUT_DIMS {
                     let ptr = weights.as_mut_ptr().add(i * OUTPUT_DIMS) as *mut u64;
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

    pub fn forward(&self, feature_indices: &[usize], output: &mut [u8]) {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                self.forward_avx2(feature_indices, output);
            }
            return;
        }

        let mut acc = [0; OUTPUT_DIMS];
        for fi in feature_indices {
            let weights = &self.weights[*fi * OUTPUT_DIMS..][..OUTPUT_DIMS];
            for (a, w) in acc.iter_mut().zip(weights.iter()) {
                *a += *w;
            }
        }

        for i in 0..OUTPUT_DIMS {
            output[i] = (acc[i] + self.biases[i]).clamp(0, 127) as u8;
        }
    }

    unsafe fn forward_avx2(&self, feature_indices: &[usize], output: &mut [u8]) {
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
            let idx = *feature_indices.get_unchecked(i);
            let weight_ptr = weight_ptr.add(idx * OUTPUT_DIMS) as *const __m256i;

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
        while j < NUM_REGS {
            let acc_j = *acc.get_unchecked(j);
            let acc_j1 = *acc.get_unchecked(j + 1);
            let clamped_j = _mm256_max_epi16(zero, _mm256_min_epi16(acc_j, one));
            let clamped_j1 = _mm256_max_epi16(zero, _mm256_min_epi16(acc_j1, one));
            _mm256_store_si256(output_ptr.add(j / 2), _mm256_packus_epi16(clamped_j, clamped_j1));
            j += 2;
        }
    }
}
