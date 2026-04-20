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

/// Accumulates weights of active sparse features into the accumulator (ARM NEON).
///
/// Uses 128-bit `int16x8_t` lanes with 8-way, 4-way, and 1-way feature unrolling to
/// mirror the x86 macro-generated variants.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
pub(super) fn accumulate_neon<const DIMS: usize>(
    pattern_feature: &PatternFeature,
    weights: &[i16],
    acc: &mut [i16; DIMS],
) {
    use std::arch::aarch64::*;
    const LANES_PER_REG: usize = 8;
    const REGS_PER_STEP: usize = 2;
    debug_assert!(DIMS.is_multiple_of(LANES_PER_REG));
    let num_regs = DIMS / LANES_PER_REG;
    debug_assert!(num_regs.is_multiple_of(REGS_PER_STEP));
    let num_reg_steps = num_regs / REGS_PER_STEP;

    unsafe {
        let acc_ptr = acc.as_mut_ptr();
        let weights_ptr = weights.as_ptr();

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

            let r0 = weights_ptr.add(idx0 * DIMS);
            let r1 = weights_ptr.add(idx1 * DIMS);
            let r2 = weights_ptr.add(idx2 * DIMS);
            let r3 = weights_ptr.add(idx3 * DIMS);
            let r4 = weights_ptr.add(idx4 * DIMS);
            let r5 = weights_ptr.add(idx5 * DIMS);
            let r6 = weights_ptr.add(idx6 * DIMS);
            let r7 = weights_ptr.add(idx7 * DIMS);

            for j in 0..num_reg_steps {
                let off = j * REGS_PER_STEP * LANES_PER_REG;
                let w0 = vld1q_s16_x2(r0.add(off));
                let w1 = vld1q_s16_x2(r1.add(off));
                let w2 = vld1q_s16_x2(r2.add(off));
                let w3 = vld1q_s16_x2(r3.add(off));
                let w4 = vld1q_s16_x2(r4.add(off));
                let w5 = vld1q_s16_x2(r5.add(off));
                let w6 = vld1q_s16_x2(r6.add(off));
                let w7 = vld1q_s16_x2(r7.add(off));

                let sum01_0 = vaddq_s16(w0.0, w1.0);
                let sum01_1 = vaddq_s16(w0.1, w1.1);
                let sum23_0 = vaddq_s16(w2.0, w3.0);
                let sum23_1 = vaddq_s16(w2.1, w3.1);
                let sum45_0 = vaddq_s16(w4.0, w5.0);
                let sum45_1 = vaddq_s16(w4.1, w5.1);
                let sum67_0 = vaddq_s16(w6.0, w7.0);
                let sum67_1 = vaddq_s16(w6.1, w7.1);

                let sum0123_0 = vaddq_s16(sum01_0, sum23_0);
                let sum0123_1 = vaddq_s16(sum01_1, sum23_1);
                let sum4567_0 = vaddq_s16(sum45_0, sum67_0);
                let sum4567_1 = vaddq_s16(sum45_1, sum67_1);

                let a_ptr = acc_ptr.add(off);
                let acc_val = vld1q_s16_x2(a_ptr);
                vst1q_s16_x2(
                    a_ptr,
                    int16x8x2_t(
                        vaddq_s16(acc_val.0, vaddq_s16(sum0123_0, sum4567_0)),
                        vaddq_s16(acc_val.1, vaddq_s16(sum0123_1, sum4567_1)),
                    ),
                );
            }

            base += 8;
        }

        while base + 4 <= NUM_FEATURES {
            let idx0 = feature_offset(pattern_feature, base);
            let idx1 = feature_offset(pattern_feature, base + 1);
            let idx2 = feature_offset(pattern_feature, base + 2);
            let idx3 = feature_offset(pattern_feature, base + 3);

            let r0 = weights_ptr.add(idx0 * DIMS);
            let r1 = weights_ptr.add(idx1 * DIMS);
            let r2 = weights_ptr.add(idx2 * DIMS);
            let r3 = weights_ptr.add(idx3 * DIMS);

            for j in 0..num_reg_steps {
                let off = j * REGS_PER_STEP * LANES_PER_REG;
                let w0 = vld1q_s16_x2(r0.add(off));
                let w1 = vld1q_s16_x2(r1.add(off));
                let w2 = vld1q_s16_x2(r2.add(off));
                let w3 = vld1q_s16_x2(r3.add(off));

                let sum0 = vaddq_s16(vaddq_s16(w0.0, w1.0), vaddq_s16(w2.0, w3.0));
                let sum1 = vaddq_s16(vaddq_s16(w0.1, w1.1), vaddq_s16(w2.1, w3.1));

                let a_ptr = acc_ptr.add(off);
                let acc_val = vld1q_s16_x2(a_ptr);
                vst1q_s16_x2(
                    a_ptr,
                    int16x8x2_t(vaddq_s16(acc_val.0, sum0), vaddq_s16(acc_val.1, sum1)),
                );
            }

            base += 4;
        }

        while base < NUM_FEATURES {
            let idx = feature_offset(pattern_feature, base);
            let row_ptr = weights_ptr.add(idx * DIMS);
            for j in 0..num_reg_steps {
                let off = j * REGS_PER_STEP * LANES_PER_REG;
                let w = vld1q_s16_x2(row_ptr.add(off));
                let a_ptr = acc_ptr.add(off);
                let acc_val = vld1q_s16_x2(a_ptr);
                vst1q_s16_x2(
                    a_ptr,
                    int16x8x2_t(vaddq_s16(acc_val.0, w.0), vaddq_s16(acc_val.1, w.1)),
                );
            }
            base += 1;
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

#[cfg(test)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod tests {
    use super::*;

    fn build_weights<const DIMS: usize>() -> Vec<i16> {
        let mut weights = vec![0i16; crate::eval::pattern_feature::INPUT_FEATURE_DIMS * DIMS];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = ((i as i32 * 17 + 13) % 21 - 10) as i16;
        }
        weights
    }

    fn check_accumulate<const DIMS: usize>() {
        let pattern_feature = PatternFeature::new();
        let weights = build_weights::<DIMS>();
        let mut scalar = [0i16; DIMS];
        let mut neon = [0i16; DIMS];

        for (i, (s, n)) in scalar.iter_mut().zip(neon.iter_mut()).enumerate() {
            let init = ((i as i32 * 7 + 5) % 41 - 20) as i16;
            *s = init;
            *n = init;
        }

        accumulate_scalar::<DIMS>(&pattern_feature, &weights, &mut scalar);
        unsafe { accumulate_neon::<DIMS>(&pattern_feature, &weights, &mut neon) };

        assert_eq!(neon, scalar);
    }

    #[test]
    fn accumulate_neon_matches_scalar_base_dims() {
        check_accumulate::<256>();
    }

    #[test]
    fn accumulate_neon_matches_scalar_phase_adaptive_dims() {
        check_accumulate::<128>();
    }
}
