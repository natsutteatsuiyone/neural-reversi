use super::pattern_feature;

pub const CACHE_LINE_SIZE: usize = 64;
pub const AVX2_SIMD_WIDTH: usize = 32;

pub const INPUT_FEATURE_DIMS: usize = sum_eval_f2x();

pub const NUM_FEATURES: usize = pattern_feature::NUM_PATTERN_FEATURES;
pub const PATTERN_FEATURE_OFFSETS: [usize; pattern_feature::NUM_PATTERN_FEATURES] = calc_feature_offsets();

pub const MOBILITY_SCALE: u8 = 3;
pub const HIDDEN_WEIGHT_SCALE_BITS: i32 = 6;
pub const OUTPUT_WEIGHT_SCALE_BITS: i32 = 4;

const fn calc_pattern_size(pattern_index: usize) -> usize {
    let mut value = 1;
    let mut j = 0;
    while j < pattern_feature::EVAL_F2X[pattern_index].n_square {
        value *= 3;
        j += 1;
    }
    value
}

const fn sum_eval_f2x() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < pattern_feature::NUM_PATTERN_FEATURES {
        total += calc_pattern_size(i);
        i += 1;
    }
    total
}

const fn calc_feature_offsets() -> [usize; pattern_feature::NUM_PATTERN_FEATURES] {
    let mut offsets = [0; pattern_feature::NUM_PATTERN_FEATURES];
    let mut i = 0;
    while i < pattern_feature::NUM_PATTERN_FEATURES {
        offsets[i] = i * calc_pattern_size(i);
        i += 1;
    }
    offsets
}
