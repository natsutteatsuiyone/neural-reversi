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
use std::ops::{Index, IndexMut};

use crate::bitboard;
use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::search::search_context::SideToMove;
use crate::square::Square;

/// Number of distinct pattern features used for evaluation.
pub const NUM_PATTERN_FEATURES: usize = 24;

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

/// Storage for pattern features.
#[derive(Clone, Copy)]
#[repr(align(64))]
pub struct PatternFeature {
    data: [u16; FEATURE_VECTOR_SIZE],
}

impl PatternFeature {
    /// Creates a new PatternFeature initialized to zero.
    pub const fn new() -> Self {
        Self {
            data: [0; FEATURE_VECTOR_SIZE],
        }
    }

    /// Builds a PatternFeature from an explicit array.
    pub const fn from_array(data: [u16; FEATURE_VECTOR_SIZE]) -> Self {
        Self { data }
    }

    /// Unsafe getter for internal data without bounds checking.
    pub unsafe fn get_unchecked(&self, idx: usize) -> u16 {
        *unsafe { self.data.get_unchecked(idx) }
    }
}

#[cfg(target_arch = "x86_64")]
impl PatternFeature {
    #[inline(always)]
    unsafe fn as_m256_ptr(&self) -> *const std::arch::x86_64::__m256i {
        self.data.as_ptr() as *const std::arch::x86_64::__m256i
    }

    #[inline(always)]
    unsafe fn as_mut_m256_ptr(&mut self) -> *mut std::arch::x86_64::__m256i {
        self.data.as_mut_ptr() as *mut std::arch::x86_64::__m256i
    }
}

impl Index<usize> for PatternFeature {
    type Output = u16;

    #[inline(always)]
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl IndexMut<usize> for PatternFeature {
    #[inline(always)]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
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

impl CoordinateToFeature {
    #[inline(always)]
    unsafe fn features_slice(&self) -> &[[u32; 2]] {
        unsafe { std::slice::from_raw_parts(self.features.as_ptr(), self.n_features as usize) }
    }
}

/// Board squares that make up each pattern.
#[rustfmt::skip]
pub const EVAL_F2X: [FeatureToCoordinate; NUM_PATTERN_FEATURES] = [
    ftc!(8, [Sq::C2, Sq::D2, Sq::B3, Sq::C3, Sq::D3, Sq::B4, Sq::C4, Sq::D4, Sq::None, Sq::None]),
    ftc!(8, [Sq::F2, Sq::E2, Sq::G3, Sq::F3, Sq::E3, Sq::G4, Sq::F4, Sq::E4, Sq::None, Sq::None]),
    ftc!(8, [Sq::C7, Sq::D7, Sq::B6, Sq::C6, Sq::D6, Sq::B5, Sq::C5, Sq::D5, Sq::None, Sq::None]),
    ftc!(8, [Sq::F7, Sq::E7, Sq::G6, Sq::F6, Sq::E6, Sq::G5, Sq::F5, Sq::E5, Sq::None, Sq::None]),

    ftc!(8, [Sq::A1, Sq::B2, Sq::C3, Sq::D4, Sq::E5, Sq::F6, Sq::G7, Sq::H8, Sq::None, Sq::None]),
    ftc!(8, [Sq::H1, Sq::G2, Sq::F3, Sq::E4, Sq::D5, Sq::C6, Sq::B7, Sq::A8, Sq::None, Sq::None]),
    ftc!(8, [Sq::D3, Sq::E4, Sq::F5, Sq::D4, Sq::E5, Sq::C4, Sq::D5, Sq::E6, Sq::None, Sq::None]),
    ftc!(8, [Sq::E3, Sq::D4, Sq::C5, Sq::E4, Sq::D5, Sq::F4, Sq::E5, Sq::D6, Sq::None, Sq::None]),

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

    ftc!(9, [Sq::A1, Sq::B1, Sq::C1, Sq::A2, Sq::B2, Sq::C2, Sq::A3, Sq::B3, Sq::C3, Sq::None]),
    ftc!(9, [Sq::H1, Sq::G1, Sq::F1, Sq::H2, Sq::G2, Sq::F2, Sq::H3, Sq::G3, Sq::F3, Sq::None]),
    ftc!(9, [Sq::A8, Sq::B8, Sq::C8, Sq::A7, Sq::B7, Sq::C7, Sq::A6, Sq::B6, Sq::C6, Sq::None]),
    ftc!(9, [Sq::H8, Sq::G8, Sq::F8, Sq::H7, Sq::G7, Sq::F7, Sq::H6, Sq::G6, Sq::F6, Sq::None]),
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
        multiplier = if multiplier == 0 {
            1
        } else {
            multiplier * PATTERN_BASE
        };

        // If this is the square we're checking, record its position
        if board & (1u64 << (square as u8)) != 0 {
            feature_index = multiplier;
        }
    }
    feature_index
}

/// Generates the EVAL_FEATURE lookup table at compile time.
const fn generate_eval_feature() -> [PatternFeature; BOARD_SQUARES] {
    let mut result = [PatternFeature::new(); BOARD_SQUARES];
    let mut square_idx = 0;

    while square_idx < BOARD_SQUARES {
        let board = 1u64 << square_idx;
        let mut feature_values = [0u16; FEATURE_VECTOR_SIZE];

        // Compute feature index for each pattern
        let mut pattern_idx = 0;
        while pattern_idx < NUM_PATTERN_FEATURES {
            feature_values[pattern_idx] =
                compute_pattern_feature_index(board, &EVAL_F2X[pattern_idx]) as u16;
            pattern_idx += 1;
        }

        result[square_idx] = PatternFeature::from_array(feature_values);
        square_idx += 1;
    }

    result
}

/// Generates the EVAL_X2F lookup table at compile time.
const fn generate_eval_x2f() -> [CoordinateToFeature; BOARD_SQUARES] {
    let mut result = [CoordinateToFeature {
        n_features: 0,
        features: [[0, 0]; MAX_FEATURES_PER_SQUARE],
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
}

/// Pattern feature weights for each board square.
const EVAL_FEATURE: [PatternFeature; BOARD_SQUARES] = generate_eval_feature();

/// Reverse mapping from board squares to pattern features.
static EVAL_X2F: [CoordinateToFeature; BOARD_SQUARES] = generate_eval_x2f();

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
                p_feature[i] = p_feature[i] * 3 + get_square_color(board, sq);
                o_feature[i] = o_feature[i] * 3 + get_square_color(&o_board, sq);
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
    /// * `sq` - The square where the move was made
    /// * `flipped` - Bitboard of pieces flipped by the move
    /// * `ply` - The current ply number
    /// * `side_to_move` - The side that made the move (Player or Opponent)
    #[inline(always)]
    pub fn update(&mut self, sq: Square, flipped: u64, ply: usize, side_to_move: SideToMove) {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx2") {
                unsafe { self.update_avx2(sq, flipped, ply, side_to_move) }
                return;
            }
        }

        self.update_fallback(sq, flipped, ply, side_to_move);
    }

    /// AVX2-optimized implementation of pattern feature update.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn update_avx2(&mut self, sq: Square, flipped: u64, ply: usize, side_to_move: SideToMove) {
        use std::arch::x86_64::*;

        unsafe {
            let ef = &EVAL_FEATURE;

            let f_ptr = ef.get_unchecked(sq.index()).as_m256_ptr();
            let f0 = _mm256_load_si256(f_ptr);
            let f1 = _mm256_load_si256(f_ptr.add(1));

            let first_idx = flipped.trailing_zeros() as usize;
            let mut bits = _blsr_u64(flipped);

            let first_fp = ef.get_unchecked(first_idx).as_m256_ptr();
            let mut sum0 = _mm256_load_si256(first_fp);
            let mut sum1 = _mm256_load_si256(first_fp.add(1));

            if bits != 0 {
                loop {
                    let idx1 = bits.trailing_zeros() as usize;
                    bits = _blsr_u64(bits);
                    let fp1 = ef.get_unchecked(idx1).as_m256_ptr();
                    sum0 = _mm256_add_epi16(sum0, _mm256_load_si256(fp1));
                    sum1 = _mm256_add_epi16(sum1, _mm256_load_si256(fp1.add(1)));

                    if bits == 0 {
                        break;
                    }

                    let idx2 = bits.trailing_zeros() as usize;
                    bits = _blsr_u64(bits);
                    let fp2 = ef.get_unchecked(idx2).as_m256_ptr();
                    sum0 = _mm256_add_epi16(sum0, _mm256_load_si256(fp2));
                    sum1 = _mm256_add_epi16(sum1, _mm256_load_si256(fp2.add(1)));

                    if bits == 0 {
                        break;
                    }
                }
            }

            let f2_0 = _mm256_slli_epi16(f0, 1);
            let f2_1 = _mm256_slli_epi16(f1, 1);
            let f_minus_sum_0 = _mm256_sub_epi16(f0, sum0);
            let f_minus_sum_1 = _mm256_sub_epi16(f1, sum1);
            let twof_plus_sum_0 = _mm256_add_epi16(f2_0, sum0);
            let twof_plus_sum_1 = _mm256_add_epi16(f2_1, sum1);

            let p_feats = &mut self.p_features;
            let o_feats = &mut self.o_features;
            let p_in_ptr = p_feats.get_unchecked(ply).as_m256_ptr();
            let o_in_ptr = o_feats.get_unchecked(ply).as_m256_ptr();
            let p_in0 = _mm256_load_si256(p_in_ptr);
            let p_in1 = _mm256_load_si256(p_in_ptr.add(1));
            let o_in0 = _mm256_load_si256(o_in_ptr);
            let o_in1 = _mm256_load_si256(o_in_ptr.add(1));

            let p_out_ptr = p_feats.get_unchecked_mut(ply + 1).as_mut_m256_ptr();
            let o_out_ptr = o_feats.get_unchecked_mut(ply + 1).as_mut_m256_ptr();

            match side_to_move {
                SideToMove::Player => {
                    let p_out0 = _mm256_sub_epi16(p_in0, twof_plus_sum_0);
                    let p_out1 = _mm256_sub_epi16(p_in1, twof_plus_sum_1);
                    let o_out0 = _mm256_sub_epi16(o_in0, f_minus_sum_0);
                    let o_out1 = _mm256_sub_epi16(o_in1, f_minus_sum_1);

                    _mm256_store_si256(p_out_ptr, p_out0);
                    _mm256_store_si256(p_out_ptr.add(1), p_out1);
                    _mm256_store_si256(o_out_ptr, o_out0);
                    _mm256_store_si256(o_out_ptr.add(1), o_out1);
                }
                SideToMove::Opponent => {
                    let p_out0 = _mm256_sub_epi16(p_in0, f_minus_sum_0);
                    let p_out1 = _mm256_sub_epi16(p_in1, f_minus_sum_1);
                    let o_out0 = _mm256_sub_epi16(o_in0, twof_plus_sum_0);
                    let o_out1 = _mm256_sub_epi16(o_in1, twof_plus_sum_1);

                    _mm256_store_si256(p_out_ptr, p_out0);
                    _mm256_store_si256(p_out_ptr.add(1), p_out1);
                    _mm256_store_si256(o_out_ptr, o_out0);
                    _mm256_store_si256(o_out_ptr.add(1), o_out1);
                }
            }
        }
    }

    /// Fallback implementation of pattern feature update for architectures without AVX2 support.
    fn update_fallback(&mut self, sq: Square, flipped: u64, ply: usize, side_to_move: SideToMove) {
        self.p_features.copy_within(ply..ply + 1, ply + 1);
        self.o_features.copy_within(ply..ply + 1, ply + 1);
        let p_out = &mut self.p_features[ply + 1];
        let o_out = &mut self.o_features[ply + 1];

        unsafe {
            let s = EVAL_X2F.get_unchecked(sq.index());
            let placed_features = s.features_slice();
            let p_data = &mut p_out.data;
            let o_data = &mut o_out.data;

            if side_to_move == SideToMove::Player {
                for &[feature_idx, power] in placed_features {
                    let idx = feature_idx as usize;
                    let delta = power as u16;
                    *p_data.get_unchecked_mut(idx) -= delta << 1;
                    *o_data.get_unchecked_mut(idx) -= delta;
                }

                for x in BitboardIterator::new(flipped) {
                    let s_bit = EVAL_X2F.get_unchecked(x as usize);
                    for &[feature_idx, power] in s_bit.features_slice() {
                        let idx = feature_idx as usize;
                        let delta = power as u16;
                        *p_data.get_unchecked_mut(idx) -= delta;
                        *o_data.get_unchecked_mut(idx) += delta;
                    }
                }
            } else {
                for &[feature_idx, power] in placed_features {
                    let idx = feature_idx as usize;
                    let delta = power as u16;
                    *p_data.get_unchecked_mut(idx) -= delta;
                    *o_data.get_unchecked_mut(idx) -= delta << 1;
                }

                for x in BitboardIterator::new(flipped) {
                    let s_bit = EVAL_X2F.get_unchecked(x as usize);
                    for &[feature_idx, power] in s_bit.features_slice() {
                        let idx = feature_idx as usize;
                        let delta = power as u16;
                        *p_data.get_unchecked_mut(idx) += delta;
                        *o_data.get_unchecked_mut(idx) -= delta;
                    }
                }
            }
        }
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
