use super::pattern_feature;

/// Size of a CPU cache line in bytes.
pub const CACHE_LINE_SIZE: usize = 64;

/// Total number of input feature dimensions for the neural network.
/// This is the sum of all pattern feature sizes (3^n for n-square patterns).
pub const INPUT_FEATURE_DIMS: usize = sum_eval_f2x();

/// Number of distinct pattern features used in the evaluation function.
pub const NUM_FEATURES: usize = pattern_feature::NUM_PATTERN_FEATURES;

/// Precomputed offsets for each pattern feature in the feature vector.
pub const PATTERN_FEATURE_OFFSETS: [usize; pattern_feature::NUM_PATTERN_FEATURES] =
    calc_feature_offsets();

/// Calculates the size of a pattern feature (3^n where n is the number of squares).
/// Each square can have 3 states: empty, black, or white.
pub const fn calc_pattern_size(pattern_index: usize) -> usize {
    let mut value = 1;
    let mut j = 0;
    while j < pattern_feature::EVAL_F2X[pattern_index].n_square {
        value *= 3;
        j += 1;
    }
    value
}

/// Computes the total number of input features across all patterns.
pub const fn sum_eval_f2x() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < pattern_feature::NUM_PATTERN_FEATURES {
        total += calc_pattern_size(i);
        i += 1;
    }
    total
}

/// Precomputes the starting offset for each pattern feature in the feature vector.
pub const fn calc_feature_offsets() -> [usize; pattern_feature::NUM_PATTERN_FEATURES] {
    let mut offsets = [0; pattern_feature::NUM_PATTERN_FEATURES];
    let mut current_offset = 0;
    let mut i = 0;
    while i < pattern_feature::NUM_PATTERN_FEATURES {
        offsets[i] = current_offset;
        current_offset += calc_pattern_size(i);
        i += 1;
    }
    offsets
}
