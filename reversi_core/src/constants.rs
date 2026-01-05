//! Global constants

use crate::types::Score;

/// Size of a CPU cache line in bytes.
pub const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of plies (moves) in a game
pub const MAX_PLY: usize = 61;

/// Maximum possible score (disc difference)
pub const SCORE_MAX: Score = 64;

/// Minimum possible score (disc difference)
pub const SCORE_MIN: Score = -64;

/// Maximum number of threads supported
pub const MAX_THREADS: usize = 64;

/// Infinity score for search algorithms
pub const SCORE_INF: Score = 30000;
