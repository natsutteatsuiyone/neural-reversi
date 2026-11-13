//! Global constants.

/// Number of squares on the Reversi board.
pub const BOARD_SQUARES: usize = 64;

/// Number of empty squares at game start.
pub const INITIAL_EMPTY_COUNT: usize = 60;

/// Size of a CPU cache line in bytes.
pub const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of plies stored in the search stack.
pub const MAX_PLY: usize = 61;

/// Maximum number of threads supported.
pub const MAX_THREADS: usize = 64;
