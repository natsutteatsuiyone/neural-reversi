use crate::misc::ceil_to_multiple;

use super::pattern_feature;

pub const CACHE_LINE_SIZE: usize = 64;
pub const AVX2_SIMD_WIDTH: usize = 32;

pub const INPUT_FEATURE_DIMS: usize = sum_eval_f2x();
pub const UNIV_INPUT_OUTPUT_DIMS: usize = 256;

pub const L1_UNIV_INPUT_DIMS: usize = UNIV_INPUT_OUTPUT_DIMS + 1;
pub const L1_UNIV_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L1_UNIV_INPUT_DIMS, 32);
pub const L1_UNIV_OUTPUT_DIMS: usize = 8;
pub const L1_UNIV_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L1_UNIV_OUTPUT_DIMS, 32);
pub const L1_UNIV_NUM_REGS: usize = 1;

pub const L1_PS_INPUT_DIMS: usize = 128;
pub const L1_PS_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L1_PS_INPUT_DIMS, 32);
pub const L1_PS_OUTPUT_DIMS: usize = 8;
pub const L1_PS_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L1_PS_OUTPUT_DIMS, 32);
pub const L1_PS_NUM_REGS: usize = 1;

pub const L2_INPUT_DIMS: usize = (L1_UNIV_OUTPUT_DIMS + L1_PS_OUTPUT_DIMS) * 2;
pub const L2_PADDED_INPUT_DIMS: usize = ceil_to_multiple(L2_INPUT_DIMS, 32);
pub const L2_OUTPUT_DIMS: usize = 64;
pub const L2_PADDED_OUTPUT_DIMS: usize = ceil_to_multiple(L2_OUTPUT_DIMS, 32);
pub const L2_NUM_REGS: usize = (L2_INPUT_DIMS * 2) / 8 * 2;

pub const LO_INPUT_DIMS: usize = L2_OUTPUT_DIMS;

pub const NUM_FEATURES: usize = pattern_feature::NUM_PATTERN_FEATURES;

pub const HIDDEN_WEIGHT_SCALE_BITS: i32 = 6;
pub const OUTPUT_WEIGHT_SCALE_BITS: i32 = 4;
pub const NUM_LAYER_STACKS: usize = 60;
pub const NUM_PHASE_ADAPTIVE_INPUT: usize = 6;

const fn sum_eval_f2x() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < pattern_feature::NUM_PATTERN_FEATURES {
        let mut value = 1;
        let mut j = 0;
        while j < pattern_feature::EVAL_F2X[i].n_square {
            value *= 3;
            j += 1;
        }
        total += value;
        i += 1;
    }
    total
}
