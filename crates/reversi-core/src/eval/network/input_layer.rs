//! Input layers for the neural evaluation network.

mod base_input;
mod phase_adaptive_input;

pub use base_input::BaseInput;
pub use phase_adaptive_input::PhaseAdaptiveInput;

pub(in crate::eval::network) use base_input::OUTPUT_DIMS as BASE_OUTPUT_DIMS;
pub(in crate::eval::network) use phase_adaptive_input::OUTPUT_DIMS as PA_OUTPUT_DIMS;

use crate::eval::pattern_feature::{NUM_FEATURES, PatternFeature};
use crate::eval::util::feature_offset;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod simd_layout {
    use std::mem::size_of;

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

    cfg_select! {
        target_feature = "avx512bw" => {
            const AVX512_ORDER: [usize; PERMUTE_WIDTH] = [0, 2, 4, 6, 1, 3, 5, 7];
            pub fn permute_rows(data: &mut [i16], row_len: usize) {
                unsafe { permute_rows_impl::<u128, PERMUTE_WIDTH>(data, row_len, &AVX512_ORDER) };
            }
        }
        target_feature = "avx2" => {
            const AVX2_ORDER: [usize; PERMUTE_WIDTH] = [0, 1, 4, 5, 2, 3, 6, 7];
            pub fn permute_rows(data: &mut [i16], row_len: usize) {
                unsafe { permute_rows_impl::<u64, PERMUTE_WIDTH>(data, row_len, &AVX2_ORDER) };
            }
        }
    }
}

/// Accumulates weights of active sparse features into the accumulator (scalar fallback).
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
