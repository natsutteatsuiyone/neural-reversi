//! Global constants

use crate::types::Score;

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

/// Converts an endgame score to a midgame score
///
/// # Arguments
///
/// * `endgame_score` - The endgame score to convert
///
/// # Returns
///
/// The corresponding midgame score
#[inline(always)]
pub fn to_midgame_score(endgame_score: Score) -> Score {
    endgame_score << EVAL_SCORE_SCALE_BITS
}

/// Converts an midgame score to an endgame score
///
/// # Arguments
///
/// * `midgame_score` - The midgame score to convert
///
/// # Returns
///
/// The corresponding endgame score
#[inline(always)]
pub fn to_endgame_score(midgame_score: Score) -> Score {
    midgame_score >> EVAL_SCORE_SCALE_BITS
}
