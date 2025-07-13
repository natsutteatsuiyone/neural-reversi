//! Global constants

use crate::types::Score;

/// Maximum number of plies (moves) in a game
pub const MAX_PLY: usize = 61;

/// Maximum possible score
pub const SCORE_MAX: Score = 64;

/// Minimum possible score
pub const SCORE_MIN: Score = -64;

/// Number of bits used for evaluation score scaling
pub const EVAL_SCORE_SCALE_BITS: i32 = 7;

/// Scale factor for evaluation scores
pub const EVAL_SCORE_SCALE: i32 = 1 << EVAL_SCORE_SCALE_BITS;

/// Maximum scaled score for midgame evaluation
pub const MID_SCORE_MAX: Score = SCORE_MAX << EVAL_SCORE_SCALE_BITS;

/// Minimum scaled score for midgame evaluation
pub const MID_SCORE_MIN: Score = SCORE_MIN << EVAL_SCORE_SCALE_BITS;

/// Infinity score for search algorithms
pub const SCORE_INF: Score = 10000;
