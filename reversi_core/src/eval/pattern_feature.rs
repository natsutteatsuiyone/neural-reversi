//! Pattern-based feature extraction for neural network evaluation.
//!
//! Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/eval.c
//!
//! This module implements a pattern feature extraction system that converts board
//! positions into numerical features for the neural network evaluator. It uses
//! various geometric patterns (corners, edges, diagonals) to capture strategic
//! board configurations that are important for position evaluation.
//!
//! # Encoding
//!
//! Each pattern is encoded as a base-3 number where:
//! - 0 = current player's piece
//! - 1 = opponent's piece
//! - 2 = empty square
//!
//! This encoding allows for 3^8 = 6561 possible configurations for 8-square patterns.

use crate::bitboard;
use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::move_list::Move;
use crate::search::search_context::SideToMove;
use crate::square::Square;

/// Number of distinct pattern features used for evaluation.
pub const NUM_PATTERN_FEATURES: usize = 22;

/// Maximum number of features that any single square can participate in.
const MAX_FEATURES_PER_SQUARE: usize = 4;

/// Maximum number of features per square as u32 for loop comparisons.
const MAX_FEATURES_PER_SQUARE_U32: u32 = MAX_FEATURES_PER_SQUARE as u32;

/// Size of the pattern feature vector (padded to 32 for SIMD alignment).
const FEATURE_VECTOR_SIZE: usize = 32;

/// Number of squares on the reversi board.
const BOARD_SQUARES: usize = 64;

/// Base for pattern encoding (ternary: empty=2, opponent=1, player=0).
const PATTERN_BASE: u32 = 3;

/// Type alias for better readability.
type Sq = Square;

/// Maps a feature index to the board squares it covers.
///
/// Each pattern feature examines a specific set of board squares.
/// This struct defines which squares belong to each pattern.
#[derive(Debug, Clone, Copy)]
pub struct FeatureToCoordinate {
    /// Number of squares in this pattern (up to 10).
    pub n_square: usize,
    /// Array of squares that make up this pattern.
    /// Uses Square::None for unused slots.
    pub squares: [Square; 10],
}

/// Macro for concisely creating FeatureToCoordinate instances.
macro_rules! ftc {
    ($n_square:expr, [$($square:expr),* $(,)?]) => {
        FeatureToCoordinate {
            n_square: $n_square,
            squares: [$($square),*],
        }
    };
}

/// Storage for pattern features with SIMD-friendly alignment.
///
/// Uses a union to allow both scalar and SIMD access to the same data.
/// The 32-byte alignment ensures optimal performance for AVX2 operations.
#[derive(Clone, Copy)]
#[repr(align(32))]
pub union PatternFeature {
    /// Scalar view: array of 16-bit pattern indices.
    pub v1: [u16; FEATURE_VECTOR_SIZE],
    /// SIMD view: two 256-bit vectors for AVX2 operations.
    v16: [core::arch::x86_64::__m256i; 2],
}

impl PatternFeature {
    /// Creates a new PatternFeature initialized to zero.
    fn new() -> Self {
        Self { v1: [0; FEATURE_VECTOR_SIZE] }
    }
}

/// Maps a board square to the features it participates in.
///
/// Since each square can be part of multiple patterns, this struct
/// tracks all features affected when a square changes.
#[derive(Debug, Clone, Copy)]
struct CoordinateToFeature {
    /// Number of features this square participates in.
    n_features: u32,
    /// Array of [feature_index, power_of_3] pairs.
    /// power_of_3 indicates the square's position in the pattern.
    features: [[u32; 2]; MAX_FEATURES_PER_SQUARE],
}


/// Board squares that make up each pattern.
#[rustfmt::skip]
pub const EVAL_F2X: [FeatureToCoordinate; NUM_PATTERN_FEATURES] = [
    ftc!(8, [Sq::A1, Sq::B1, Sq::C1, Sq::D1, Sq::A2, Sq::A3, Sq::A4, Sq::B2, Sq::None, Sq::None]),
    ftc!(8, [Sq::H1, Sq::G1, Sq::F1, Sq::E1, Sq::H2, Sq::H3, Sq::H4, Sq::G2, Sq::None, Sq::None]),
    ftc!(8, [Sq::A8, Sq::B8, Sq::C8, Sq::D8, Sq::A7, Sq::A6, Sq::A5, Sq::B7, Sq::None, Sq::None]),
    ftc!(8, [Sq::H8, Sq::G8, Sq::F8, Sq::E8, Sq::H7, Sq::H6, Sq::H5, Sq::G7, Sq::None, Sq::None]),

    ftc!(8, [Sq::C2, Sq::D2, Sq::B3, Sq::C3, Sq::D3, Sq::B4, Sq::C4, Sq::D4, Sq::None, Sq::None]),
    ftc!(8, [Sq::F2, Sq::E2, Sq::G3, Sq::F3, Sq::E3, Sq::G4, Sq::F4, Sq::E4, Sq::None, Sq::None]),
    ftc!(8, [Sq::C7, Sq::D7, Sq::B6, Sq::C6, Sq::D6, Sq::B5, Sq::C5, Sq::D5, Sq::None, Sq::None]),
    ftc!(8, [Sq::F7, Sq::E7, Sq::G6, Sq::F6, Sq::E6, Sq::G5, Sq::F5, Sq::E5, Sq::None, Sq::None]),

    ftc!(8, [Sq::A1, Sq::B2, Sq::C3, Sq::D4, Sq::E5, Sq::F6, Sq::G7, Sq::H8, Sq::None, Sq::None]),
    ftc!(8, [Sq::H1, Sq::G2, Sq::F3, Sq::E4, Sq::D5, Sq::C6, Sq::B7, Sq::A8, Sq::None, Sq::None]),

    ftc!(8, [Sq::A1, Sq::B1, Sq::C1, Sq::D1, Sq::E1, Sq::F1, Sq::G1, Sq::H1, Sq::None, Sq::None]),
    ftc!(8, [Sq::A8, Sq::B8, Sq::C8, Sq::D8, Sq::E8, Sq::F8, Sq::G8, Sq::H8, Sq::None, Sq::None]),
    ftc!(8, [Sq::A1, Sq::A2, Sq::A3, Sq::A4, Sq::A5, Sq::A6, Sq::A7, Sq::A8, Sq::None, Sq::None]),
    ftc!(8, [Sq::H1, Sq::H2, Sq::H3, Sq::H4, Sq::H5, Sq::H6, Sq::H7, Sq::H8, Sq::None, Sq::None]),

    ftc!(8, [Sq::B1, Sq::C1, Sq::D1, Sq::E1, Sq::B2, Sq::C2, Sq::D2, Sq::E2, Sq::None, Sq::None]),
    ftc!(8, [Sq::G1, Sq::F1, Sq::E1, Sq::D1, Sq::G2, Sq::F2, Sq::E2, Sq::D2, Sq::None, Sq::None]),
    ftc!(8, [Sq::B8, Sq::C8, Sq::D8, Sq::E8, Sq::B7, Sq::C7, Sq::D7, Sq::E7, Sq::None, Sq::None]),
    ftc!(8, [Sq::G8, Sq::F8, Sq::E8, Sq::D8, Sq::G7, Sq::F7, Sq::E7, Sq::D7, Sq::None, Sq::None]),

    ftc!(8, [Sq::A2, Sq::A3, Sq::A4, Sq::A5, Sq::B2, Sq::B3, Sq::B4, Sq::B5, Sq::None, Sq::None]),
    ftc!(8, [Sq::A7, Sq::A6, Sq::A5, Sq::A4, Sq::B7, Sq::B6, Sq::B5, Sq::B4, Sq::None, Sq::None]),
    ftc!(8, [Sq::H2, Sq::H3, Sq::H4, Sq::H5, Sq::G2, Sq::G3, Sq::G4, Sq::G5, Sq::None, Sq::None]),
    ftc!(8, [Sq::H7, Sq::H6, Sq::H5, Sq::H4, Sq::G7, Sq::G6, Sq::G5, Sq::G4, Sq::None, Sq::None]),
];

/// Common logic for computing pattern feature indices.
///
/// This function calculates the positional weight (power of 3) for a specific
/// square within a pattern feature, based on the ternary encoding used.
/// 
/// Replaces duplicate logic previously found in both EVAL_FEATURE and EVAL_X2F
/// generation, ensuring consistent computation across both lookup tables.
const fn compute_pattern_feature_index(board: u64, feature: &FeatureToCoordinate) -> u32 {
    let mut multiplier = 0u32;
    let mut feature_index = 0u32;
    let mut i = feature.n_square;
    
    // Process squares in reverse order to match feature encoding
    while i > 0 {
        i -= 1;
        let square = feature.squares[i];
        
        // Skip None squares
        if matches!(square, Square::None) {
            continue;
        }
        
        // Update multiplier for base-3 encoding
        multiplier = if multiplier == 0 { 1 } else { multiplier * PATTERN_BASE };
        
        // If this is the square we're checking, record its position
        if board & (1u64 << (square as u8)) != 0 {
            feature_index = multiplier;
        }
    }
    feature_index
}

/// Generates pattern feature lookup tables at compile time.
///
/// See README.md for detailed explanation of the generated tables.
macro_rules! generate_pattern_tables {
    () => {
        /// Pattern feature weights for each board square.
        #[rustfmt::skip]
        const EVAL_FEATURE: [PatternFeature; BOARD_SQUARES] = {
            let mut result = [PatternFeature { v1: [0; FEATURE_VECTOR_SIZE] }; BOARD_SQUARES];
            let mut square_idx = 0;

            while square_idx < BOARD_SQUARES {
                let board = 1u64 << square_idx;
                let mut feature_values = [0u16; FEATURE_VECTOR_SIZE];

                // Compute feature index for each pattern
                let mut pattern_idx = 0;
                while pattern_idx < NUM_PATTERN_FEATURES {
                    feature_values[pattern_idx] = compute_pattern_feature_index(board, &EVAL_F2X[pattern_idx]) as u16;
                    pattern_idx += 1;
                }

                result[square_idx] = PatternFeature { v1: feature_values };
                square_idx += 1;
            }

            result
        };

        /// Reverse mapping from board squares to pattern features.
        #[rustfmt::skip]
        static EVAL_X2F: [CoordinateToFeature; BOARD_SQUARES] = {
            let mut result = [CoordinateToFeature {
                n_features: 0,
                features: [[0, 0]; MAX_FEATURES_PER_SQUARE]
            }; BOARD_SQUARES];

            let mut square_idx = 0;
            while square_idx < BOARD_SQUARES {
                let board = 1u64 << square_idx;
                let mut n_features = 0u32;
                let mut features = [[0u32, 0u32]; MAX_FEATURES_PER_SQUARE];

                // Find all features that include this square
                let mut feature_idx = 0;
                while feature_idx < NUM_PATTERN_FEATURES && n_features < MAX_FEATURES_PER_SQUARE_U32 {
                    let feature_value = compute_pattern_feature_index(board, &EVAL_F2X[feature_idx]);
                    if feature_value > 0 {
                        features[n_features as usize] = [feature_idx as u32, feature_value];
                        n_features += 1;
                    }
                    feature_idx += 1;
                }

                result[square_idx] = CoordinateToFeature {
                    n_features,
                    features,
                };
                square_idx += 1;
            }

            result
        };
    };
}

// Generate the lookup tables
generate_pattern_tables!();

/// Container for pattern features for both players throughout a game.
///
/// Maintains pattern features for each ply of the game, allowing
/// incremental updates as moves are made.
pub struct PatternFeatures {
    /// Pattern features from the current player's perspective.
    pub p_features: [PatternFeature; 61],
    /// Pattern features from the opponent's perspective.
    pub o_features: [PatternFeature; 61],
}

impl PatternFeatures {
    /// Creates new pattern features from the given board position.
    ///
    /// # Arguments
    ///
    /// * `board` - The current board position
    /// * `ply` - The current ply number
    ///
    /// # Returns
    ///
    /// A new PatternFeatures instance with features computed for both players.
    pub fn new(board: &Board, ply: usize) -> PatternFeatures {
        let mut pattern_features = PatternFeatures {
            p_features: [PatternFeature::new(); 61],
            o_features: [PatternFeature::new(); 61],
        };

        let o_board = board.switch_players();
        let p_feature = &mut pattern_features.p_features[ply];
        let o_feature = &mut pattern_features.o_features[ply];
        for (i, f2x) in EVAL_F2X.iter().enumerate() {
            for j in 0..f2x.n_square {
                let sq = f2x.squares[j];
                unsafe {
                    p_feature.v1[i] = p_feature.v1[i] * 3 + get_square_color(board, sq);
                    o_feature.v1[i] = o_feature.v1[i] * 3 + get_square_color(&o_board, sq);
                }
            }
        }
        pattern_features
    }

    /// Updates pattern features after a move is made.
    ///
    /// This method efficiently updates only the affected patterns rather than
    /// recomputing all features from scratch. It handles both the placed piece
    /// and all flipped pieces.
    ///
    /// # Arguments
    ///
    /// * `mv` - The move that was made
    /// * `ply` - The current ply number
    /// * `side_to_move` - The side that made the move (Player or Opponent)
    #[inline(always)]
    pub fn update(&mut self, mv: &Move, ply: usize, player: SideToMove) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.update_avx2(mv, ply, player) }
                return;
            }
        }

        self.update_fallback(mv, ply, player);
    }

    /// Fallback implementation of pattern feature update for architectures without AVX2 support.
    fn update_fallback(&mut self, mv: &Move, ply: usize, player: SideToMove) {
        let flip = mv.flipped;

        self.p_features.copy_within(ply..ply + 1, ply + 1);
        self.o_features.copy_within(ply..ply + 1, ply + 1);
        let p_out = &mut self.p_features[ply + 1];
        let o_out = &mut self.o_features[ply + 1];
        let s = &EVAL_X2F[mv.sq.index()];

        if player == SideToMove::Player {
            for i in 0..s.n_features {
                let j = s.features[i as usize][0] as usize;
                let x = s.features[i as usize][1] as usize;
                unsafe {
                    p_out.v1[j] -= 2 * x as u16;
                    o_out.v1[j] -= x as u16;
                }
            }

            for x in BitboardIterator::new(flip) {
                let s_bit = &EVAL_X2F[x as usize];
                for i in 0..s_bit.n_features {
                    let j = s_bit.features[i as usize][0] as usize;
                    let x = s_bit.features[i as usize][1] as usize;
                    unsafe {
                        p_out.v1[j] -= x as u16;
                        o_out.v1[j] += x as u16;
                    }
                }
            }
        } else {
            for i in 0..s.n_features {
                let j = s.features[i as usize][0] as usize;
                let x = s.features[i as usize][1] as usize;
                unsafe {
                    p_out.v1[j] -= x as u16;
                    o_out.v1[j] -= 2 * x as u16;
                }
            }

            for x in BitboardIterator::new(flip) {
                let s_bit = &EVAL_X2F[x as usize];
                for i in 0..s_bit.n_features {
                    let j = s_bit.features[i as usize][0] as usize;
                    let x = s_bit.features[i as usize][1] as usize;
                    unsafe {
                        p_out.v1[j] += x as u16;
                        o_out.v1[j] -= x as u16;
                    }
                }
            }
        }
    }

    /// AVX2-optimized implementation of pattern feature update.
    #[target_feature(enable = "avx2")]
    #[cfg(target_arch = "x86_64")]
    fn update_avx2(&mut self, mv: &Move, ply: usize, player: SideToMove) {
        use std::arch::x86_64::*;

        let flip = mv.flipped;
        let p_in = unsafe { self.p_features[ply].v16 };
        let p_out = unsafe { &mut self.p_features[ply + 1].v16 };
        let o_in = unsafe { self.o_features[ply].v16 };
        let o_out = unsafe { &mut self.o_features[ply + 1].v16 };

        let sq_index = mv.sq.index();
        let f = unsafe { &EVAL_FEATURE[sq_index].v16 };

        let (p_scale, o_scale, p_sign, o_sign) = if player == SideToMove::Player {
            (
                _mm256_set1_epi16(2),
                _mm256_set1_epi16(1),
                _mm256_set1_epi16(-1), // p: subtract sum
                _mm256_set1_epi16(1),  // o: add sum
            )
        } else {
            (
                _mm256_set1_epi16(1),
                _mm256_set1_epi16(2),
                _mm256_set1_epi16(1),  // p: add sum
                _mm256_set1_epi16(-1), // o: subtract sum
            )
        };

        p_out[0] = _mm256_sub_epi16(p_in[0], _mm256_mullo_epi16(f[0], p_scale));
        p_out[1] = _mm256_sub_epi16(p_in[1], _mm256_mullo_epi16(f[1], p_scale));
        o_out[0] = _mm256_sub_epi16(o_in[0], _mm256_mullo_epi16(f[0], o_scale));
        o_out[1] = _mm256_sub_epi16(o_in[1], _mm256_mullo_epi16(f[1], o_scale));

        let mut sum0 = _mm256_setzero_si256();
        let mut sum1 = _mm256_setzero_si256();
        for x in BitboardIterator::new(flip) {
            let f = unsafe { &EVAL_FEATURE[x as usize].v16 };
            sum0 = _mm256_add_epi16(sum0, f[0]);
            sum1 = _mm256_add_epi16(sum1, f[1]);
        }

        p_out[0] = _mm256_add_epi16(p_out[0], _mm256_mullo_epi16(sum0, p_sign));
        p_out[1] = _mm256_add_epi16(p_out[1], _mm256_mullo_epi16(sum1, p_sign));
        o_out[0] = _mm256_add_epi16(o_out[0], _mm256_mullo_epi16(sum0, o_sign));
        o_out[1] = _mm256_add_epi16(o_out[1], _mm256_mullo_epi16(sum1, o_sign));
    }
}

/// Computes pattern features for a board position.
///
/// Each pattern is encoded as a base-3 number representing the
/// configuration of pieces in that pattern.
///
/// # Arguments
///
/// * `board` - The board position to extract features from
/// * `patterns` - Output array to store the computed pattern indices
pub fn set_features(board: &Board, patterns: &mut [u16]) {
    for i in 0..NUM_PATTERN_FEATURES {
        let f2x = &EVAL_F2X[i];
        for j in 0..f2x.n_square {
            let sq = f2x.squares[j];
            let c = get_square_color(board, sq);
            patterns[i] = patterns[i] * 3 + c;
        }
    }
}

/// Gets the color/state of a square on the board.
///
/// # Arguments
///
/// * `board` - The board to examine
/// * `sq` - The square to check
///
/// # Returns
///
/// * 0 - Current player's piece
/// * 1 - Opponent's piece
/// * 2 - Empty square
#[inline]
fn get_square_color(board: &Board, sq: Square) -> u16 {
    if bitboard::is_set(board.player, sq) {
        0
    } else if bitboard::is_set(board.opponent, sq) {
        1
    } else {
        2
    }
}
