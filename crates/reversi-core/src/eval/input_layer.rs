//! Input layers for the neural evaluation network.

mod base_input;
mod phase_adaptive_input;

pub use base_input::BaseInput;
pub use phase_adaptive_input::PhaseAdaptiveInput;

use crate::eval::pattern_feature::{NUM_FEATURES, PatternFeature};
use crate::eval::util::feature_offset;

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
macro_rules! impl_accumulate {
    (
        $fn_name:ident,
        $target_feature:literal,
        $lane_ty:ty,
        load = $load:path,
        store = $store:path,
        add = $add:path
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $target_feature)]
        #[inline]
        fn $fn_name<const DIMS: usize>(
            pattern_feature: &PatternFeature,
            weights: &[i16],
            acc: &mut [i16; DIMS],
        ) {
            use std::arch::x86_64::*;
            use std::mem::size_of;

            unsafe {
                let acc_ptr = acc.as_mut_ptr() as *mut $lane_ty;
                const LANES_PER_REG: usize = size_of::<$lane_ty>() / size_of::<i16>();
                let num_regs = DIMS / LANES_PER_REG;
                debug_assert!(DIMS.is_multiple_of(LANES_PER_REG));
                let weights_ptr = weights.as_ptr() as *const $lane_ty;

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

                    let fw0 = weights_ptr.add(idx0 * num_regs);
                    let fw1 = weights_ptr.add(idx1 * num_regs);
                    let fw2 = weights_ptr.add(idx2 * num_regs);
                    let fw3 = weights_ptr.add(idx3 * num_regs);
                    let fw4 = weights_ptr.add(idx4 * num_regs);
                    let fw5 = weights_ptr.add(idx5 * num_regs);
                    let fw6 = weights_ptr.add(idx6 * num_regs);
                    let fw7 = weights_ptr.add(idx7 * num_regs);

                    for j in 0..num_regs {
                        let w0 = $load(fw0.add(j));
                        let w1 = $load(fw1.add(j));
                        let w2 = $load(fw2.add(j));
                        let w3 = $load(fw3.add(j));
                        let w4 = $load(fw4.add(j));
                        let w5 = $load(fw5.add(j));
                        let w6 = $load(fw6.add(j));
                        let w7 = $load(fw7.add(j));

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

                    let fw0 = weights_ptr.add(idx0 * num_regs);
                    let fw1 = weights_ptr.add(idx1 * num_regs);
                    let fw2 = weights_ptr.add(idx2 * num_regs);
                    let fw3 = weights_ptr.add(idx3 * num_regs);

                    for j in 0..num_regs {
                        let w0 = $load(fw0.add(j));
                        let w1 = $load(fw1.add(j));
                        let w2 = $load(fw2.add(j));
                        let w3 = $load(fw3.add(j));

                        let sum = $add($add(w0, w1), $add(w2, w3));

                        let acc_val = $load(acc_ptr.add(j));
                        $store(acc_ptr.add(j), $add(acc_val, sum));
                    }

                    base += 4;
                }

                while base < NUM_FEATURES {
                    let idx = feature_offset(pattern_feature, base);
                    let feature_weights = weights_ptr.add(idx * num_regs);
                    for j in 0..num_regs {
                        let weight = $load(feature_weights.add(j));
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
    load = _mm512_load_si512,
    store = _mm512_store_si512,
    add = _mm512_add_epi16
);

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl_accumulate!(
    accumulate_avx2,
    "avx2",
    __m256i,
    load = _mm256_load_si256,
    store = _mm256_store_si256,
    add = _mm256_add_epi16
);

/// Accumulates weights of active sparse features into accumulator (scalar fallback).
///
/// # Arguments
///
/// * `pattern_feature` - Sparse feature values indexed by pattern identifier.
/// * `weights` - Flat weight matrix in feature-major order.
/// * `acc` - Dense accumulation buffer.
#[inline(always)]
fn accumulate_scalar<const DIMS: usize>(
    pattern_feature: &PatternFeature,
    weights: &[i16],
    acc: &mut [i16; DIMS],
) {
    let acc_ptr = acc.as_mut_ptr();
    let weights_ptr = weights.as_ptr();
    for idx in 0..NUM_FEATURES {
        let offset = feature_offset(pattern_feature, idx);
        let row_ptr = unsafe { weights_ptr.add(offset * DIMS) };
        let mut j = 0;
        unsafe {
            while j + 4 <= DIMS {
                let acc_j = acc_ptr.add(j);
                let row_j = row_ptr.add(j);
                *acc_j += *row_j;
                *acc_j.add(1) = *acc_j.add(1) + *row_j.add(1);
                *acc_j.add(2) = *acc_j.add(2) + *row_j.add(2);
                *acc_j.add(3) = *acc_j.add(3) + *row_j.add(3);
                j += 4;
            }

            while j < DIMS {
                let acc_j = acc_ptr.add(j);
                *acc_j += *row_ptr.add(j);
                j += 1;
            }
        }
    }
}
