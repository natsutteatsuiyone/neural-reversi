//! Global constants

use crate::types::{Score, Scoref};

/// Size of a CPU cache line in bytes.
pub const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of plies (moves) in a game
pub const MAX_PLY: usize = 61;

/// Maximum possible score
pub const SCORE_MAX: Score = 64;

/// Minimum possible score
pub const SCORE_MIN: Score = -64;

/// Number of bits used for evaluation score scaling
pub const EVAL_SCORE_SCALE_BITS: i32 = 8;

/// Scale factor for evaluation scores
pub const EVAL_SCORE_SCALE: i32 = 1 << EVAL_SCORE_SCALE_BITS;

/// Maximum scaled score for midgame evaluation
pub const MID_SCORE_MAX: Score = SCORE_MAX << EVAL_SCORE_SCALE_BITS;

/// Minimum scaled score for midgame evaluation
pub const MID_SCORE_MIN: Score = SCORE_MIN << EVAL_SCORE_SCALE_BITS;

/// Infinity score for search algorithms
pub const SCORE_INF: Score = 30000;

/// Maximum number of threads supported
pub const MAX_THREADS: usize = 64;

/// Scales a disc-difference score into the internal midgame representation.
#[inline(always)]
pub const fn scale_score(unscaled_score: Score) -> Score {
    unscaled_score << EVAL_SCORE_SCALE_BITS
}

/// Converts an internal midgame score back to disc-difference units.
#[inline(always)]
pub const fn unscale_score(scaled_score: Score) -> Score {
    scaled_score >> EVAL_SCORE_SCALE_BITS
}

/// Converts an internal midgame score to a floating-point representation.
#[inline(always)]
pub const fn unscale_score_f32(scaled_score: Score) -> Scoref {
    (scaled_score as f32) / (EVAL_SCORE_SCALE as f32)
}
