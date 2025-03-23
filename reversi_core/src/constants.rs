use crate::types::Score;

pub const MAX_PLY: usize = 61;

pub const SCORE_MAX: Score = 64;
pub const SCORE_MIN: Score = -64;
pub const EVAL_SCORE_SCALE_BITS: i32 = 6;
pub const EVAL_SCORE_SCALE: i32 = 1 << EVAL_SCORE_SCALE_BITS;
pub const MID_SCORE_MAX: Score = SCORE_MAX << EVAL_SCORE_SCALE_BITS;
pub const MID_SCORE_MIN: Score = SCORE_MIN << EVAL_SCORE_SCALE_BITS;
pub const SCORE_INF: Score = 10000;
