//! Input layers for the neural evaluation network.

mod base_input;
mod phase_adaptive_input;

pub use base_input::BaseInput;
pub use phase_adaptive_input::PhaseAdaptiveInput;

pub(in crate::eval::network) use base_input::OUTPUT_DIMS as BASE_OUTPUT_DIMS;
pub(in crate::eval::network) use phase_adaptive_input::OUTPUT_DIMS as PA_OUTPUT_DIMS;

use crate::eval::util::accumulate_scalar;

#[cfg(any(test, all(target_arch = "x86_64", target_feature = "avx2")))]
mod simd_layout {
    #[cfg(any(test, target_feature = "avx512bw"))]
    const AVX512_ORDER: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    #[cfg(any(test, not(target_feature = "avx512bw")))]
    const AVX2_ORDER: [usize; 8] = [0, 1, 4, 5, 2, 3, 6, 7];

    fn permute_rows_impl<const TILE_WORDS: usize>(
        data: &mut [i16],
        row_len: usize,
        order: [usize; 8],
    ) {
        if row_len == 0 {
            return;
        }
        for row in data.chunks_exact_mut(row_len) {
            let (tiles, _) = row.as_chunks_mut::<TILE_WORDS>();
            let (blocks, _) = tiles.as_chunks_mut::<8>();
            for block in blocks {
                let source = *block;
                *block = order.map(|index| source[index]);
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub fn permute_rows(data: &mut [i16], row_len: usize) {
        cfg_select! {
            target_feature = "avx512bw" => {
                permute_rows_impl::<8>(data, row_len, AVX512_ORDER);
            }
            _ => {
                permute_rows_impl::<4>(data, row_len, AVX2_ORDER);
            }
        }
    }

    #[cfg(test)]
    #[test]
    fn x86_row_permutations_restore_natural_order_after_lane_packing() {
        fn check<const TILE_WORDS: usize>(order: [usize; 8], width: usize) {
            for row_len in [128, 256] {
                let mut data = (0..2 * row_len as i16 + 2).collect::<Vec<_>>();
                let end = data.len() - 1;
                let guard = data[end];
                permute_rows_impl::<TILE_WORDS>(&mut data[1..end], row_len, order);
                for (row, values) in data[1..end].chunks_exact(row_len).enumerate() {
                    for index in 0..row_len {
                        let lane = index % width;
                        let source = index - lane
                            + (lane / 16) * 8
                            + ((lane / 8) % 2) * (width / 2)
                            + lane % 8;
                        assert_eq!(values[source], (row * row_len + index + 1) as i16);
                    }
                }
                assert_eq!(data[0], 0);
                assert_eq!(data[end], guard);
            }
        }

        check::<4>(AVX2_ORDER, 32);
        check::<8>(AVX512_ORDER, 64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::pattern_feature::{
        INPUT_FEATURE_DIMS, NUM_PATTERN_FEATURES, PatternFeature, calc_pattern_size,
    };

    fn valid_pattern_feature() -> PatternFeature {
        let mut pattern_feature = PatternFeature::new();
        for idx in 0..NUM_PATTERN_FEATURES {
            pattern_feature.set(idx, ((idx * 4099 + 17) % calc_pattern_size(idx)) as u16);
        }
        pattern_feature
    }

    #[test]
    fn accumulate_scalar_adds_the_active_feature_rows_to_existing_accumulator() {
        const DIMS: usize = 6;

        let pattern_feature = valid_pattern_feature();
        let mut weights = vec![0i16; INPUT_FEATURE_DIMS * DIMS];
        let mut expected = [11, -22, 33, -44, 55, -66];
        let mut actual = expected;

        for feature_idx in 0..NUM_PATTERN_FEATURES {
            let row = pattern_feature.row(feature_idx);
            let values = [
                feature_idx as i16 - 17,
                31 - feature_idx as i16,
                (feature_idx as i16 % 7) - 3,
                2 * feature_idx as i16 - 19,
                13 - (feature_idx as i16 % 11),
                (feature_idx as i16 % 5) * 3 - 6,
            ];
            let start = row * DIMS;
            weights[start..start + DIMS].copy_from_slice(&values);
            for (expected_value, value) in expected.iter_mut().zip(values) {
                *expected_value += value;
            }
        }

        accumulate_scalar::<DIMS>(&pattern_feature, &weights, &mut actual);

        assert_eq!(actual, expected);
    }
}
