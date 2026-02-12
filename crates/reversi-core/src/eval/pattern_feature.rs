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
//! - 0 = current player's disc
//! - 1 = opponent's disc
//! - 2 = empty square
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

use cfg_if::cfg_if;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::constants::{BOARD_SQUARES, MAX_PLY};
use crate::search::side_to_move::SideToMove;
use crate::square::Square;

/// Number of distinct pattern features used for evaluation.
pub const NUM_PATTERN_FEATURES: usize = 24;

/// Alias for NUM_PATTERN_FEATURES for compatibility.
pub const NUM_FEATURES: usize = NUM_PATTERN_FEATURES;

/// Maximum number of features that any single square can participate in.
const MAX_FEATURES_PER_SQUARE: usize = 4;

/// Maximum number of features per square as u32 for loop comparisons.
const MAX_FEATURES_PER_SQUARE_U32: u32 = MAX_FEATURES_PER_SQUARE as u32;

/// Size of the pattern feature vector (padded to 32 for SIMD alignment).
const FEATURE_VECTOR_SIZE: usize = 32;

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

impl FeatureToCoordinate {
    pub const fn new(n_square: usize, squares: [Square; 10]) -> Self {
        Self { n_square, squares }
    }
}

/// Storage for pattern features.
#[derive(Clone, Copy)]
#[repr(align(64))]
pub struct PatternFeature {
    data: [u16; FEATURE_VECTOR_SIZE],
}

impl Default for PatternFeature {
    fn default() -> Self {
        Self::new()
    }
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
    ///
    /// # Safety
    ///
    /// The caller must ensure that `idx < FEATURE_VECTOR_SIZE`.
    pub unsafe fn get_unchecked(&self, idx: usize) -> u16 {
        *unsafe { self.data.get_unchecked(idx) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl PatternFeature {
    #[inline(always)]
    unsafe fn as_m256_ptr(&self) -> *const std::arch::x86_64::__m256i {
        self.data.as_ptr() as *const std::arch::x86_64::__m256i
    }
}

#[cfg(target_arch = "wasm32")]
impl PatternFeature {
    #[inline(always)]
    unsafe fn as_v128_ptr(&self) -> *const core::arch::wasm32::v128 {
        self.data.as_ptr() as *const core::arch::wasm32::v128
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
    /// Returns a slice of the features this square participates in.
    #[inline(always)]
    fn features(&self) -> &[[u32; 2]] {
        &self.features[..self.n_features as usize]
    }
}

/// Board squares that make up each pattern.
#[rustfmt::skip]
pub const EVAL_F2X: [FeatureToCoordinate; NUM_PATTERN_FEATURES] = [
    FeatureToCoordinate::new(8, [Sq::C2, Sq::D2, Sq::B3, Sq::C3, Sq::D3, Sq::B4, Sq::C4, Sq::D4, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::F2, Sq::E2, Sq::G3, Sq::F3, Sq::E3, Sq::G4, Sq::F4, Sq::E4, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::C7, Sq::D7, Sq::B6, Sq::C6, Sq::D6, Sq::B5, Sq::C5, Sq::D5, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::F7, Sq::E7, Sq::G6, Sq::F6, Sq::E6, Sq::G5, Sq::F5, Sq::E5, Sq::None, Sq::None]),

    FeatureToCoordinate::new(8, [Sq::A1, Sq::B2, Sq::C3, Sq::D4, Sq::E5, Sq::F6, Sq::G7, Sq::H8, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::H1, Sq::G2, Sq::F3, Sq::E4, Sq::D5, Sq::C6, Sq::B7, Sq::A8, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::D3, Sq::E4, Sq::F5, Sq::D4, Sq::E5, Sq::C4, Sq::D5, Sq::E6, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::E3, Sq::D4, Sq::C5, Sq::E4, Sq::D5, Sq::F4, Sq::E5, Sq::D6, Sq::None, Sq::None]),

    FeatureToCoordinate::new(8, [Sq::A1, Sq::B1, Sq::C1, Sq::D1, Sq::E1, Sq::F1, Sq::G1, Sq::H1, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::A8, Sq::B8, Sq::C8, Sq::D8, Sq::E8, Sq::F8, Sq::G8, Sq::H8, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::A1, Sq::A2, Sq::A3, Sq::A4, Sq::A5, Sq::A6, Sq::A7, Sq::A8, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::H1, Sq::H2, Sq::H3, Sq::H4, Sq::H5, Sq::H6, Sq::H7, Sq::H8, Sq::None, Sq::None]),

    FeatureToCoordinate::new(8, [Sq::B1, Sq::C1, Sq::D1, Sq::E1, Sq::B2, Sq::C2, Sq::D2, Sq::E2, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::G1, Sq::F1, Sq::E1, Sq::D1, Sq::G2, Sq::F2, Sq::E2, Sq::D2, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::B8, Sq::C8, Sq::D8, Sq::E8, Sq::B7, Sq::C7, Sq::D7, Sq::E7, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::G8, Sq::F8, Sq::E8, Sq::D8, Sq::G7, Sq::F7, Sq::E7, Sq::D7, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::A2, Sq::A3, Sq::A4, Sq::A5, Sq::B2, Sq::B3, Sq::B4, Sq::B5, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::A7, Sq::A6, Sq::A5, Sq::A4, Sq::B7, Sq::B6, Sq::B5, Sq::B4, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::H2, Sq::H3, Sq::H4, Sq::H5, Sq::G2, Sq::G3, Sq::G4, Sq::G5, Sq::None, Sq::None]),
    FeatureToCoordinate::new(8, [Sq::H7, Sq::H6, Sq::H5, Sq::H4, Sq::G7, Sq::G6, Sq::G5, Sq::G4, Sq::None, Sq::None]),

    FeatureToCoordinate::new(9, [Sq::A1, Sq::B1, Sq::C1, Sq::A2, Sq::B2, Sq::C2, Sq::A3, Sq::B3, Sq::C3, Sq::None]),
    FeatureToCoordinate::new(9, [Sq::H1, Sq::G1, Sq::F1, Sq::H2, Sq::G2, Sq::F2, Sq::H3, Sq::G3, Sq::F3, Sq::None]),
    FeatureToCoordinate::new(9, [Sq::A8, Sq::B8, Sq::C8, Sq::A7, Sq::B7, Sq::C7, Sq::A6, Sq::B6, Sq::C6, Sq::None]),
    FeatureToCoordinate::new(9, [Sq::H8, Sq::G8, Sq::F8, Sq::H7, Sq::G7, Sq::F7, Sq::H6, Sq::G6, Sq::F6, Sq::None]),
];

/// Calculates the size of a pattern feature (3^n where n is the number of squares).
/// Each square can have 3 states: empty, player's disc, or opponent's disc.
pub const fn calc_pattern_size(pattern_index: usize) -> usize {
    let mut value = 1;
    let mut j = 0;
    while j < EVAL_F2X[pattern_index].n_square {
        value *= 3;
        j += 1;
    }
    value
}

/// Computes the total number of input features across all patterns.
pub const fn sum_eval_f2x() -> usize {
    let mut total = 0;
    let mut i = 0;
    while i < NUM_PATTERN_FEATURES {
        total += calc_pattern_size(i);
        i += 1;
    }
    total
}

/// Total number of input feature dimensions for the neural network.
/// This is the sum of all pattern feature sizes (3^n for n-square patterns).
pub const INPUT_FEATURE_DIMS: usize = sum_eval_f2x();

/// Precomputes the starting offset for each pattern feature in the feature vector.
pub const fn calc_feature_offsets() -> [usize; NUM_PATTERN_FEATURES] {
    let mut offsets = [0; NUM_PATTERN_FEATURES];
    let mut current_offset = 0;
    let mut i = 0;
    while i < NUM_PATTERN_FEATURES {
        offsets[i] = current_offset;
        current_offset += calc_pattern_size(i);
        i += 1;
    }
    offsets
}

/// Precomputed offsets for each pattern feature in the feature vector.
pub const PATTERN_FEATURE_OFFSETS: [usize; NUM_PATTERN_FEATURES] = calc_feature_offsets();

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
///
/// # Panics
///
/// Panics at compile time if any square participates in more than
/// `MAX_FEATURES_PER_SQUARE` patterns. This ensures pattern definition
/// errors are caught during compilation rather than causing silent truncation.
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
        while feature_idx < NUM_PATTERN_FEATURES {
            let feature_value = compute_pattern_feature_index(board, &EVAL_F2X[feature_idx]);
            if feature_value > 0 {
                if n_features >= MAX_FEATURES_PER_SQUARE_U32 {
                    panic!(
                        "Square participates in more patterns than MAX_FEATURES_PER_SQUARE allows"
                    );
                }
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
#[allow(dead_code)]
const EVAL_FEATURE: [PatternFeature; BOARD_SQUARES] = generate_eval_feature();

/// Reverse mapping from board squares to pattern features.
static EVAL_X2F: [CoordinateToFeature; BOARD_SQUARES] = generate_eval_x2f();

/// Container for pattern features for both players throughout a game.
///
/// Maintains pattern features for each ply of the game, allowing
/// incremental updates as moves are made.
pub struct PatternFeatures {
    p_features: [MaybeUninit<PatternFeature>; MAX_PLY],
    o_features: [MaybeUninit<PatternFeature>; MAX_PLY],
}

impl PatternFeatures {
    /// Returns a reference to the player's pattern feature at the given ply.
    #[inline(always)]
    pub fn p_feature(&self, ply: usize) -> &PatternFeature {
        debug_assert!(ply < MAX_PLY);
        unsafe { self.p_features.get_unchecked(ply).assume_init_ref() }
    }

    /// Returns a reference to the opponent's pattern feature at the given ply.
    #[inline(always)]
    pub fn o_feature(&self, ply: usize) -> &PatternFeature {
        debug_assert!(ply < MAX_PLY);
        unsafe { self.o_features.get_unchecked(ply).assume_init_ref() }
    }

    /// Creates new pattern features from the given board position.
    pub fn new(board: &Board, ply: usize) -> Self {
        debug_assert!(ply < MAX_PLY);

        let mut pattern_features = PatternFeatures {
            p_features: [const { MaybeUninit::uninit() }; MAX_PLY],
            o_features: [const { MaybeUninit::uninit() }; MAX_PLY],
        };

        pattern_features.p_features[ply] = MaybeUninit::new(PatternFeature::new());
        pattern_features.o_features[ply] = MaybeUninit::new(PatternFeature::new());

        let o_board = board.switch_players();
        let p_feature = unsafe { pattern_features.p_features[ply].assume_init_mut() };
        let o_feature = unsafe { pattern_features.o_features[ply].assume_init_mut() };
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
    /// recomputing all features from scratch. It handles both the placed disc
    /// and all flipped discs.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square where the move was made
    /// * `flipped` - Bitboard of discs flipped by the move
    /// * `ply` - The current ply number
    /// * `side_to_move` - The side that made the move (Player or Opponent)
    #[inline(always)]
    pub fn update(&mut self, sq: Square, flipped: Bitboard, ply: usize, side_to_move: SideToMove) {
        debug_assert!(sq != Square::None);
        debug_assert!(!flipped.is_empty());
        debug_assert!(ply < MAX_PLY - 1);

        cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
                unsafe { self.update_avx2(sq, flipped, ply, side_to_move) }
            } else if #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))] {
                self.update_wasm_simd(sq, flipped, ply, side_to_move)
            } else {
                self.update_fallback(sq, flipped, ply, side_to_move);
            }
        }
    }

    /// AVX2-optimized implementation of pattern feature update.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    fn update_avx2(&mut self, sq: Square, flipped: Bitboard, ply: usize, side_to_move: SideToMove) {
        use std::arch::x86_64::*;

        unsafe {
            let ef = &EVAL_FEATURE;

            let f_ptr = ef.get_unchecked(sq.index()).as_m256_ptr();
            let f0 = _mm256_load_si256(f_ptr);
            let f1 = _mm256_load_si256(f_ptr.add(1));

            let first_idx = flipped.bits().trailing_zeros() as usize;
            let mut bits = _blsr_u64(flipped.bits());

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
            let p_in_ptr = p_feats.get_unchecked(ply).assume_init_ref().as_m256_ptr();
            let o_in_ptr = o_feats.get_unchecked(ply).assume_init_ref().as_m256_ptr();
            let p_in0 = _mm256_load_si256(p_in_ptr);
            let p_in1 = _mm256_load_si256(p_in_ptr.add(1));
            let o_in0 = _mm256_load_si256(o_in_ptr);
            let o_in1 = _mm256_load_si256(o_in_ptr.add(1));

            let p_out_ptr = p_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut __m256i;
            let o_out_ptr = o_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut __m256i;

            let (delta_p0, delta_p1, delta_o0, delta_o1) = if side_to_move == SideToMove::Player {
                (
                    twof_plus_sum_0,
                    twof_plus_sum_1,
                    f_minus_sum_0,
                    f_minus_sum_1,
                )
            } else {
                (
                    f_minus_sum_0,
                    f_minus_sum_1,
                    twof_plus_sum_0,
                    twof_plus_sum_1,
                )
            };

            let p_out0 = _mm256_sub_epi16(p_in0, delta_p0);
            let p_out1 = _mm256_sub_epi16(p_in1, delta_p1);
            let o_out0 = _mm256_sub_epi16(o_in0, delta_o0);
            let o_out1 = _mm256_sub_epi16(o_in1, delta_o1);

            _mm256_store_si256(p_out_ptr, p_out0);
            _mm256_store_si256(p_out_ptr.add(1), p_out1);
            _mm256_store_si256(o_out_ptr, o_out0);
            _mm256_store_si256(o_out_ptr.add(1), o_out1);
        }
    }

    /// WebAssembly SIMD-optimized implementation of pattern feature update.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    fn update_wasm_simd(
        &mut self,
        sq: Square,
        flipped: Bitboard,
        ply: usize,
        side_to_move: SideToMove,
    ) {
        use core::arch::wasm32::*;

        unsafe {
            let ef = &EVAL_FEATURE;
            let f_ptr = ef.get_unchecked(sq.index()).as_v128_ptr();
            let f0 = v128_load(f_ptr);
            let f1 = v128_load(f_ptr.add(1));
            let f2 = v128_load(f_ptr.add(2));
            let f3 = v128_load(f_ptr.add(3));

            let mut sum0 = i16x8_splat(0);
            let mut sum1 = i16x8_splat(0);
            let mut sum2 = i16x8_splat(0);
            let mut sum3 = i16x8_splat(0);

            let first_idx = flipped.bits().trailing_zeros() as usize;
            let mut bits = flipped.bits() & (flipped.bits() - 1);

            if bits != 0 || flipped.bits() != 0 {
                let first_fp = ef.get_unchecked(first_idx).as_v128_ptr();
                sum0 = v128_load(first_fp);
                sum1 = v128_load(first_fp.add(1));
                sum2 = v128_load(first_fp.add(2));
                sum3 = v128_load(first_fp.add(3));

                if bits != 0 {
                    loop {
                        let idx1 = bits.trailing_zeros() as usize;
                        bits = bits & (bits - 1);
                        let fp1 = ef.get_unchecked(idx1).as_v128_ptr();
                        sum0 = i16x8_add(sum0, v128_load(fp1));
                        sum1 = i16x8_add(sum1, v128_load(fp1.add(1)));
                        sum2 = i16x8_add(sum2, v128_load(fp1.add(2)));
                        sum3 = i16x8_add(sum3, v128_load(fp1.add(3)));

                        if bits == 0 {
                            break;
                        }

                        let idx2 = bits.trailing_zeros() as usize;
                        bits = bits & (bits - 1);
                        let fp2 = ef.get_unchecked(idx2).as_v128_ptr();
                        sum0 = i16x8_add(sum0, v128_load(fp2));
                        sum1 = i16x8_add(sum1, v128_load(fp2.add(1)));
                        sum2 = i16x8_add(sum2, v128_load(fp2.add(2)));
                        sum3 = i16x8_add(sum3, v128_load(fp2.add(3)));

                        if bits == 0 {
                            break;
                        }
                    }
                }
            }

            let f2_0 = i16x8_shl(f0, 1);
            let f2_1 = i16x8_shl(f1, 1);
            let f2_2 = i16x8_shl(f2, 1);
            let f2_3 = i16x8_shl(f3, 1);

            let f_minus_sum_0 = i16x8_sub(f0, sum0);
            let f_minus_sum_1 = i16x8_sub(f1, sum1);
            let f_minus_sum_2 = i16x8_sub(f2, sum2);
            let f_minus_sum_3 = i16x8_sub(f3, sum3);

            let twof_plus_sum_0 = i16x8_add(f2_0, sum0);
            let twof_plus_sum_1 = i16x8_add(f2_1, sum1);
            let twof_plus_sum_2 = i16x8_add(f2_2, sum2);
            let twof_plus_sum_3 = i16x8_add(f2_3, sum3);

            let p_feats = &mut self.p_features;
            let o_feats = &mut self.o_features;
            let p_in_ptr = p_feats.get_unchecked(ply).assume_init_ref().as_v128_ptr();
            let o_in_ptr = o_feats.get_unchecked(ply).assume_init_ref().as_v128_ptr();

            let p_in0 = v128_load(p_in_ptr);
            let p_in1 = v128_load(p_in_ptr.add(1));
            let p_in2 = v128_load(p_in_ptr.add(2));
            let p_in3 = v128_load(p_in_ptr.add(3));

            let o_in0 = v128_load(o_in_ptr);
            let o_in1 = v128_load(o_in_ptr.add(1));
            let o_in2 = v128_load(o_in_ptr.add(2));
            let o_in3 = v128_load(o_in_ptr.add(3));

            let p_out_ptr = p_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut v128;
            let o_out_ptr = o_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut v128;

            match side_to_move {
                SideToMove::Player => {
                    let p_out0 = i16x8_sub(p_in0, twof_plus_sum_0);
                    let p_out1 = i16x8_sub(p_in1, twof_plus_sum_1);
                    let p_out2 = i16x8_sub(p_in2, twof_plus_sum_2);
                    let p_out3 = i16x8_sub(p_in3, twof_plus_sum_3);

                    let o_out0 = i16x8_sub(o_in0, f_minus_sum_0);
                    let o_out1 = i16x8_sub(o_in1, f_minus_sum_1);
                    let o_out2 = i16x8_sub(o_in2, f_minus_sum_2);
                    let o_out3 = i16x8_sub(o_in3, f_minus_sum_3);

                    v128_store(p_out_ptr, p_out0);
                    v128_store(p_out_ptr.add(1), p_out1);
                    v128_store(p_out_ptr.add(2), p_out2);
                    v128_store(p_out_ptr.add(3), p_out3);

                    v128_store(o_out_ptr, o_out0);
                    v128_store(o_out_ptr.add(1), o_out1);
                    v128_store(o_out_ptr.add(2), o_out2);
                    v128_store(o_out_ptr.add(3), o_out3);
                }
                SideToMove::Opponent => {
                    let p_out0 = i16x8_sub(p_in0, f_minus_sum_0);
                    let p_out1 = i16x8_sub(p_in1, f_minus_sum_1);
                    let p_out2 = i16x8_sub(p_in2, f_minus_sum_2);
                    let p_out3 = i16x8_sub(p_in3, f_minus_sum_3);

                    let o_out0 = i16x8_sub(o_in0, twof_plus_sum_0);
                    let o_out1 = i16x8_sub(o_in1, twof_plus_sum_1);
                    let o_out2 = i16x8_sub(o_in2, twof_plus_sum_2);
                    let o_out3 = i16x8_sub(o_in3, twof_plus_sum_3);

                    v128_store(p_out_ptr, p_out0);
                    v128_store(p_out_ptr.add(1), p_out1);
                    v128_store(p_out_ptr.add(2), p_out2);
                    v128_store(p_out_ptr.add(3), p_out3);

                    v128_store(o_out_ptr, o_out0);
                    v128_store(o_out_ptr.add(1), o_out1);
                    v128_store(o_out_ptr.add(2), o_out2);
                    v128_store(o_out_ptr.add(3), o_out3);
                }
            }
        }
    }

    /// Fallback implementation of pattern feature update for architectures without AVX2 support.
    #[allow(dead_code)]
    fn update_fallback(
        &mut self,
        sq: Square,
        flipped: Bitboard,
        ply: usize,
        side_to_move: SideToMove,
    ) {
        self.p_features[ply + 1] = self.p_features[ply];
        self.o_features[ply + 1] = self.o_features[ply];
        let p_out = unsafe { self.p_features.get_unchecked_mut(ply + 1).assume_init_mut() };
        let o_out = unsafe { self.o_features.get_unchecked_mut(ply + 1).assume_init_mut() };

        let placed = &EVAL_X2F[sq.index()];

        if side_to_move == SideToMove::Player {
            for &[feature_idx, power] in placed.features() {
                let idx = feature_idx as usize;
                let delta = power as u16;
                p_out[idx] -= delta << 1;
                o_out[idx] -= delta;
            }

            for x in flipped.iter() {
                let flipped_sq = &EVAL_X2F[x as usize];
                for &[feature_idx, power] in flipped_sq.features() {
                    let idx = feature_idx as usize;
                    let delta = power as u16;
                    p_out[idx] -= delta;
                    o_out[idx] += delta;
                }
            }
        } else {
            for &[feature_idx, power] in placed.features() {
                let idx = feature_idx as usize;
                let delta = power as u16;
                p_out[idx] -= delta;
                o_out[idx] -= delta << 1;
            }

            for x in flipped.iter() {
                let flipped_sq = &EVAL_X2F[x as usize];
                for &[feature_idx, power] in flipped_sq.features() {
                    let idx = feature_idx as usize;
                    let delta = power as u16;
                    p_out[idx] += delta;
                    o_out[idx] -= delta;
                }
            }
        }
    }
}

/// Computes pattern features for a board position.
///
/// Each pattern is encoded as a base-3 number representing the
/// configuration of discs in that pattern.
///
/// # Arguments
///
/// * `board` - The board position to extract features from
/// * `patterns` - Output array to store the computed pattern indices
pub fn set_features(board: &Board, patterns: &mut [u16]) {
    patterns.fill(0);
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
/// * 0 - Current player's disc
/// * 1 - Opponent's disc
/// * 2 - Empty square
#[inline]
fn get_square_color(board: &Board, sq: Square) -> u16 {
    if board.player.contains(sq) {
        0
    } else if board.opponent.contains(sq) {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_feature_new() {
        let pf = PatternFeature::new();
        for i in 0..FEATURE_VECTOR_SIZE {
            assert_eq!(pf[i], 0);
        }
    }

    #[test]
    fn test_pattern_feature_from_array() {
        let mut data = [0u16; FEATURE_VECTOR_SIZE];
        data[0] = 100;
        data[5] = 200;
        data[23] = 300;

        let pf = PatternFeature::from_array(data);
        assert_eq!(pf[0], 100);
        assert_eq!(pf[5], 200);
        assert_eq!(pf[23], 300);
    }

    #[test]
    fn test_pattern_feature_index_mut() {
        let mut pf = PatternFeature::new();
        pf[0] = 42;
        pf[15] = 123;
        assert_eq!(pf[0], 42);
        assert_eq!(pf[15], 123);
    }

    #[test]
    fn test_pattern_feature_get_unchecked() {
        let mut data = [0u16; FEATURE_VECTOR_SIZE];
        data[10] = 999;
        let pf = PatternFeature::from_array(data);

        unsafe {
            assert_eq!(pf.get_unchecked(10), 999);
            assert_eq!(pf.get_unchecked(0), 0);
        }
    }

    #[test]
    fn test_calc_pattern_size() {
        // Patterns with 8 squares: 3^8 = 6561
        assert_eq!(calc_pattern_size(0), 6561);
        assert_eq!(calc_pattern_size(4), 6561);
        assert_eq!(calc_pattern_size(8), 6561);

        // Patterns with 9 squares: 3^9 = 19683
        assert_eq!(calc_pattern_size(20), 19683);
        assert_eq!(calc_pattern_size(21), 19683);
        assert_eq!(calc_pattern_size(22), 19683);
        assert_eq!(calc_pattern_size(23), 19683);
    }

    #[test]
    fn test_sum_eval_f2x() {
        // 20 patterns with 8 squares + 4 patterns with 9 squares
        // = 20 * 6561 + 4 * 19683 = 131220 + 78732 = 209952
        let expected = 20 * 6561 + 4 * 19683;
        assert_eq!(sum_eval_f2x(), expected);
        assert_eq!(INPUT_FEATURE_DIMS, expected);
    }

    #[test]
    fn test_calc_feature_offsets() {
        let offsets = calc_feature_offsets();

        // First offset should be 0
        assert_eq!(offsets[0], 0);

        // Each offset should be previous + pattern size
        for i in 1..NUM_PATTERN_FEATURES {
            assert_eq!(offsets[i], offsets[i - 1] + calc_pattern_size(i - 1));
        }

        // Verify PATTERN_FEATURE_OFFSETS matches
        assert_eq!(PATTERN_FEATURE_OFFSETS, offsets);
    }

    #[test]
    fn test_eval_f2x_pattern_definitions() {
        // Verify all patterns have expected number of squares
        for (i, f2x) in EVAL_F2X.iter().enumerate().take(20) {
            assert_eq!(f2x.n_square, 8, "Pattern {} should have 8 squares", i);
        }
        for (i, f2x) in EVAL_F2X.iter().enumerate().skip(20) {
            assert_eq!(f2x.n_square, 9, "Pattern {} should have 9 squares", i);
        }

        // Verify no None squares within n_square range
        for (i, f2x) in EVAL_F2X.iter().enumerate() {
            for j in 0..f2x.n_square {
                assert_ne!(
                    f2x.squares[j],
                    Square::None,
                    "Pattern {} square {} should not be None",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_get_square_color() {
        let board = Board::new(); // Initial position

        // D5 and E4 are player's discs (Black)
        assert_eq!(get_square_color(&board, Square::D5), 0);
        assert_eq!(get_square_color(&board, Square::E4), 0);

        // D4 and E5 are opponent's discs (White)
        assert_eq!(get_square_color(&board, Square::D4), 1);
        assert_eq!(get_square_color(&board, Square::E5), 1);

        // Empty squares
        assert_eq!(get_square_color(&board, Square::A1), 2);
        assert_eq!(get_square_color(&board, Square::H8), 2);
        assert_eq!(get_square_color(&board, Square::C3), 2);
    }

    #[test]
    fn test_set_features_initial_position() {
        let board = Board::new();
        let mut patterns = [0u16; NUM_PATTERN_FEATURES];
        set_features(&board, &mut patterns);

        // Verify values are within valid range
        for (i, &pattern) in patterns.iter().enumerate().take(NUM_PATTERN_FEATURES) {
            let max_val = calc_pattern_size(i) as u16;
            assert!(
                pattern < max_val,
                "Pattern {} = {} >= max {}",
                i,
                pattern,
                max_val
            );
        }

        // Verify not all patterns are at "all empty" value (regression check)
        // Initial position has 4 discs, so patterns covering center must differ from empty
        let all_empty_8 = 6560u16; // 2*(3^8-1)/2
        let all_empty_9 = 19682u16; // 2*(3^9-1)/2

        let mut has_non_empty_pattern = false;
        for (i, &pattern) in patterns.iter().enumerate() {
            let empty_val = if EVAL_F2X[i].n_square == 8 {
                all_empty_8
            } else {
                all_empty_9
            };
            if pattern != empty_val {
                has_non_empty_pattern = true;
                break;
            }
        }
        assert!(
            has_non_empty_pattern,
            "Initial position should have at least one non-empty pattern encoding"
        );
    }

    #[test]
    fn test_set_features_golden_values() {
        // Golden test: verify specific pattern values for initial position
        // These values are derived from the encoding scheme and pattern definitions
        let board = Board::new();
        let mut patterns = [0u16; NUM_PATTERN_FEATURES];
        set_features(&board, &mut patterns);

        // Feature 6: Center pattern [D3, E4, F5, D4, E5, C4, D5, E6]
        // Initial position: D4=opponent(1), D5=player(0), E4=player(0), E5=opponent(1)
        // Others are empty(2)
        // This pattern covers the center, so it should not be all-empty
        assert_ne!(
            patterns[6], 6560,
            "Feature 6 (center) should not be all-empty"
        );

        // Feature 7: Another center pattern [E3, D4, C5, E4, D5, F4, E5, D6]
        assert_ne!(
            patterns[7], 6560,
            "Feature 7 (center) should not be all-empty"
        );

        // Features 8-11 are edge/corner patterns that don't cover center - should be all empty
        assert_eq!(
            patterns[8], 6560,
            "Feature 8 (row 1) should be all-empty initially"
        );
        assert_eq!(
            patterns[9], 6560,
            "Feature 9 (row 8) should be all-empty initially"
        );
    }

    #[test]
    fn test_pattern_features_new() {
        let board = Board::new();
        let ply = 0;
        let pf = PatternFeatures::new(&board, ply);

        // Features should be computed for both players
        for i in 0..NUM_PATTERN_FEATURES {
            // Features should be within valid range
            let p_val = pf.p_feature(ply)[i];
            let o_val = pf.o_feature(ply)[i];
            let max_val = calc_pattern_size(i) as u16;
            assert!(
                p_val < max_val,
                "p_features[{}] = {} >= {}",
                i,
                p_val,
                max_val
            );
            assert!(
                o_val < max_val,
                "o_features[{}] = {} >= {}",
                i,
                o_val,
                max_val
            );
        }
    }

    #[test]
    fn test_pattern_features_symmetry() {
        // For the initial symmetric position, switching players should give
        // consistent feature patterns
        let board = Board::new();
        let switched = board.switch_players();

        let pf1 = PatternFeatures::new(&board, 0);
        let pf2 = PatternFeatures::new(&switched, 0);

        // Player features of board should match opponent features of switched board
        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                pf1.p_feature(0)[i],
                pf2.o_feature(0)[i],
                "Feature {} mismatch: p1={} != o2={}",
                i,
                pf1.p_feature(0)[i],
                pf2.o_feature(0)[i]
            );
        }
    }

    #[test]
    fn test_eval_x2f_coverage() {
        // Verify that EVAL_X2F correctly maps squares to features
        for (sq_idx, x2f) in EVAL_X2F.iter().enumerate() {
            // n_features should be within bounds
            assert!(
                x2f.n_features <= MAX_FEATURES_PER_SQUARE_U32,
                "Square {} has {} features, max is {}",
                sq_idx,
                x2f.n_features,
                MAX_FEATURES_PER_SQUARE
            );

            // Every square should participate in at least one pattern
            assert!(
                x2f.n_features > 0,
                "Square {} has no features - should participate in at least one pattern",
                sq_idx
            );

            // Verify each feature reference is valid and check for duplicates
            let mut seen_features = [false; NUM_PATTERN_FEATURES];
            for i in 0..x2f.n_features as usize {
                let [feature_idx, power] = x2f.features[i];
                assert!(
                    (feature_idx as usize) < NUM_PATTERN_FEATURES,
                    "Square {} feature {} has invalid index {}",
                    sq_idx,
                    i,
                    feature_idx
                );
                assert!(power > 0, "Square {} feature {} has zero power", sq_idx, i);

                // Check for duplicate feature indices
                assert!(
                    !seen_features[feature_idx as usize],
                    "Square {} has duplicate feature index {}",
                    sq_idx, feature_idx
                );
                seen_features[feature_idx as usize] = true;

                // Verify power is a valid power of 3 (1, 3, 9, 27, ...)
                let mut valid_power = false;
                let mut p = 1u32;
                for _ in 0..10 {
                    // max 10 squares per pattern
                    if p == power {
                        valid_power = true;
                        break;
                    }
                    p *= 3;
                }
                assert!(
                    valid_power,
                    "Square {} feature {} has invalid power {} (not a power of 3)",
                    sq_idx, feature_idx, power
                );
            }
        }
    }

    #[test]
    fn test_all_squares_covered_by_patterns() {
        // Verify every square is included in at least one pattern definition
        for sq_idx in 0..BOARD_SQUARES {
            let Some(sq) = Square::from_u8(sq_idx as u8) else {
                continue;
            };
            if matches!(sq, Square::None) {
                continue;
            }

            let mut found_in_pattern = false;
            for f2x in &EVAL_F2X {
                for j in 0..f2x.n_square {
                    if f2x.squares[j] == sq {
                        found_in_pattern = true;
                        break;
                    }
                }
                if found_in_pattern {
                    break;
                }
            }

            assert!(
                found_in_pattern,
                "Square {:?} (index {}) is not covered by any pattern",
                sq, sq_idx
            );
        }
    }

    #[test]
    fn test_eval_feature_consistency() {
        // Verify EVAL_FEATURE values are consistent with EVAL_X2F
        for sq_idx in 0..BOARD_SQUARES {
            let feature = &EVAL_FEATURE[sq_idx];
            let x2f = &EVAL_X2F[sq_idx];

            // For each pattern this square participates in
            for i in 0..x2f.n_features as usize {
                let [feature_idx, power] = x2f.features[i];
                // The power in EVAL_X2F should match the value in EVAL_FEATURE
                assert_eq!(
                    feature[feature_idx as usize] as u32, power,
                    "Square {} feature {}: EVAL_FEATURE={} but EVAL_X2F power={}",
                    sq_idx, feature_idx, feature[feature_idx as usize], power
                );
            }
        }
    }

    #[test]
    fn test_num_features_constant() {
        assert_eq!(NUM_FEATURES, NUM_PATTERN_FEATURES);
        assert_eq!(NUM_PATTERN_FEATURES, 24);
    }

    #[test]
    fn test_feature_vector_size_alignment() {
        // FEATURE_VECTOR_SIZE should be 32 for SIMD alignment
        assert_eq!(FEATURE_VECTOR_SIZE, 32);
        // PatternFeature should have correct alignment
        assert!(std::mem::align_of::<PatternFeature>() >= 64);
    }

    #[test]
    fn test_coordinate_to_feature_features() {
        // Test that features returns correct slice
        for x2f in &EVAL_X2F {
            let slice = x2f.features();
            assert_eq!(slice.len(), x2f.n_features as usize);
        }
    }

    /// Helper to verify Player move update matches fresh computation (both p and o features)
    fn assert_player_update_matches_fresh(
        pf: &PatternFeatures,
        pf_fresh: &PatternFeatures,
        ply: usize,
        context: &str,
    ) {
        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                pf.o_feature(ply + 1)[i],
                pf_fresh.p_feature(ply + 1)[i],
                "{}: o_features[{}] mismatch: {} != {}",
                context,
                i,
                pf.o_feature(ply + 1)[i],
                pf_fresh.p_feature(ply + 1)[i]
            );
            assert_eq!(
                pf.p_feature(ply + 1)[i],
                pf_fresh.o_feature(ply + 1)[i],
                "{}: p_features[{}] mismatch: {} != {}",
                context,
                i,
                pf.p_feature(ply + 1)[i],
                pf_fresh.o_feature(ply + 1)[i]
            );
        }
    }

    /// Helper to verify Opponent move update matches fresh computation (both p and o features)
    fn assert_opponent_update_matches_fresh(
        pf: &PatternFeatures,
        pf_fresh: &PatternFeatures,
        ply: usize,
        context: &str,
    ) {
        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                pf.p_feature(ply + 1)[i],
                pf_fresh.p_feature(ply + 1)[i],
                "{}: p_features[{}] mismatch: {} != {}",
                context,
                i,
                pf.p_feature(ply + 1)[i],
                pf_fresh.p_feature(ply + 1)[i]
            );
            assert_eq!(
                pf.o_feature(ply + 1)[i],
                pf_fresh.o_feature(ply + 1)[i],
                "{}: o_features[{}] mismatch: {} != {}",
                context,
                i,
                pf.o_feature(ply + 1)[i],
                pf_fresh.o_feature(ply + 1)[i]
            );
        }
    }

    #[test]
    fn test_update_multiple_flips_horizontal() {
        // Test flipping multiple discs in a horizontal line
        // Setup: A1(player), B1-D1(opponent) -> move to E1 flips B1, C1, D1
        let player = 1u64 << Square::A1.index();
        let opponent = (1u64 << Square::B1.index())
            | (1u64 << Square::C1.index())
            | (1u64 << Square::D1.index());
        let board = Board::from_bitboards(player, opponent);
        let ply = 10;
        let mut pf = PatternFeatures::new(&board, ply);

        let sq = Square::E1;
        let flipped = Bitboard::new(
            (1u64 << Square::B1.index())
                | (1u64 << Square::C1.index())
                | (1u64 << Square::D1.index()),
        );

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Horizontal flip");
    }

    #[test]
    fn test_update_multiple_flips_diagonal() {
        // Test flipping multiple discs along diagonal
        // Setup: A1(player), B2-D4(opponent) -> move to E5 flips B2, C3, D4
        let player = 1u64 << Square::A1.index();
        let opponent = (1u64 << Square::B2.index())
            | (1u64 << Square::C3.index())
            | (1u64 << Square::D4.index());
        let board = Board::from_bitboards(player, opponent);
        let ply = 8;
        let mut pf = PatternFeatures::new(&board, ply);

        let sq = Square::E5;
        let flipped = Bitboard::new(
            (1u64 << Square::B2.index())
                | (1u64 << Square::C3.index())
                | (1u64 << Square::D4.index()),
        );

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Diagonal flip");
    }

    #[test]
    fn test_update_multiple_directions() {
        // Test flipping in multiple directions simultaneously
        // Setup position where one move flips discs in 2+ directions
        // Player at D1 and D8, opponent at D2-D7 (vertical)
        // Also player at A4, opponent at B4, C4 (horizontal towards D4)
        let player = (1u64 << Square::D1.index())
            | (1u64 << Square::A4.index())
            | (1u64 << Square::D8.index());
        let opponent = (1u64 << Square::D2.index())
            | (1u64 << Square::D3.index())
            | (1u64 << Square::B4.index())
            | (1u64 << Square::C4.index())
            | (1u64 << Square::D5.index())
            | (1u64 << Square::D6.index())
            | (1u64 << Square::D7.index());
        let board = Board::from_bitboards(player, opponent);
        let ply = 15;
        let mut pf = PatternFeatures::new(&board, ply);

        // Move to D4 flips in multiple directions
        let sq = Square::D4;
        let flipped = Bitboard::new(
            (1u64 << Square::D2.index())
                | (1u64 << Square::D3.index())
                | (1u64 << Square::B4.index())
                | (1u64 << Square::C4.index())
                | (1u64 << Square::D5.index())
                | (1u64 << Square::D6.index())
                | (1u64 << Square::D7.index()),
        );

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Multi-direction flip");
    }

    #[test]
    fn test_update_at_ply_zero() {
        // Test update at ply = 0 (first move of the game)
        let board = Board::new();
        let ply = 0;
        let mut pf = PatternFeatures::new(&board, ply);

        let sq = Square::D3;
        let flipped = Bitboard::new(1u64 << Square::D4.index());

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Ply 0");
    }

    #[test]
    fn test_update_at_high_ply() {
        // Test update at a high ply value (near but not at MAX_PLY)
        let board = Board::new();
        let ply = MAX_PLY - 2; // Last valid ply for update
        let mut pf = PatternFeatures::new(&board, ply);

        let sq = Square::D3;
        let flipped = Bitboard::new(1u64 << Square::D4.index());

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "High ply");
    }

    #[test]
    fn test_update_corner_a1() {
        // Test move to corner A1
        // Setup: opponent has B1, A2, B2
        let player = (1u64 << Square::C1.index()) | (1u64 << Square::A3.index());
        let opponent = (1u64 << Square::B1.index())
            | (1u64 << Square::A2.index())
            | (1u64 << Square::B2.index());
        let board = Board::from_bitboards(player, opponent);
        let ply = 10;
        let mut pf = PatternFeatures::new(&board, ply);

        // Move to A1, flipping B1
        let sq = Square::A1;
        let flipped = Bitboard::new(1u64 << Square::B1.index());

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Corner A1");
    }

    #[test]
    fn test_update_corner_h8() {
        // Test move to corner H8
        let player = (1u64 << Square::F8.index()) | (1u64 << Square::H6.index());
        let opponent = (1u64 << Square::G8.index())
            | (1u64 << Square::H7.index())
            | (1u64 << Square::G7.index());
        let board = Board::from_bitboards(player, opponent);
        let ply = 15;
        let mut pf = PatternFeatures::new(&board, ply);

        // Move to H8, flipping G8
        let sq = Square::H8;
        let flipped = Bitboard::new(1u64 << Square::G8.index());

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Corner H8");
    }

    #[test]
    fn test_update_opponent_move() {
        // Test update when opponent makes a move (SideToMove::Opponent)
        // When opponent moves, the roles are swapped: opponent places, player's discs flip
        let board = Board::new();
        let ply = 0;
        let mut pf = PatternFeatures::new(&board, ply);

        // From opponent's perspective: opponent places at D3, flips D5 (player's disc)
        let sq = Square::D3;
        let flipped = Bitboard::new(1u64 << Square::D5.index());

        // Compute expected board state after opponent's move
        // Opponent places disc, player's disc gets flipped
        let mut new_board = board;
        new_board.opponent |= sq.bitboard() | flipped;
        new_board.player &= !flipped;
        // Don't switch players - the update is from current perspective

        pf.update(sq, flipped, ply, SideToMove::Opponent);

        // Verify against fresh computation
        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_opponent_update_matches_fresh(&pf, &pf_fresh, ply, "Opponent move");
    }

    #[test]
    fn test_update_opponent_move_multiple_flips() {
        // Test opponent move with multiple flips
        // Setup: player has B1-D1, opponent has A1
        let player = (1u64 << Square::B1.index())
            | (1u64 << Square::C1.index())
            | (1u64 << Square::D1.index());
        let opponent = 1u64 << Square::A1.index();
        let board = Board::from_bitboards(player, opponent);
        let ply = 5;
        let mut pf = PatternFeatures::new(&board, ply);

        // Opponent moves to E1, flipping B1-D1
        let sq = Square::E1;
        let flipped = Bitboard::new(
            (1u64 << Square::B1.index())
                | (1u64 << Square::C1.index())
                | (1u64 << Square::D1.index()),
        );

        let mut new_board = board;
        new_board.opponent |= sq.bitboard() | flipped;
        new_board.player &= !flipped;

        pf.update(sq, flipped, ply, SideToMove::Opponent);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_opponent_update_matches_fresh(&pf, &pf_fresh, ply, "Opponent multi-flip");
    }

    #[test]
    fn test_update_large_flip_count() {
        // Test with many discs being flipped (stress test for SIMD loop)
        // Create a line of opponent discs that will all be flipped
        let player = 1u64 << Square::A1.index();
        let mut opponent = 0u64;
        // B1 through G1 are opponent's
        for sq in [
            Square::B1,
            Square::C1,
            Square::D1,
            Square::E1,
            Square::F1,
            Square::G1,
        ] {
            opponent |= 1u64 << sq.index();
        }
        let board = Board::from_bitboards(player, opponent);
        let ply = 20;
        let mut pf = PatternFeatures::new(&board, ply);

        // Move to H1, flipping all 6 discs
        let sq = Square::H1;
        let mut flipped_bits = 0u64;
        for sq in [
            Square::B1,
            Square::C1,
            Square::D1,
            Square::E1,
            Square::F1,
            Square::G1,
        ] {
            flipped_bits |= 1u64 << sq.index();
        }
        let flipped = Bitboard::new(flipped_bits);

        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();

        pf.update(sq, flipped, ply, SideToMove::Player);

        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Large flip");
    }

    #[test]
    fn test_fallback_implementation_directly() {
        // Directly test the fallback implementation
        let board = Board::new();
        let ply = 0;
        let mut pf = PatternFeatures::new(&board, ply);

        let sq = Square::D3;
        let flipped = Bitboard::new(1u64 << Square::D4.index());

        // Call fallback directly
        pf.update_fallback(sq, flipped, ply, SideToMove::Player);

        // Compute expected result
        let mut new_board = board;
        new_board.player |= sq.bitboard() | flipped;
        new_board.opponent &= !flipped;
        let new_board = new_board.switch_players();
        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_player_update_matches_fresh(&pf, &pf_fresh, ply, "Fallback Player");
    }

    #[test]
    fn test_fallback_opponent_move() {
        // Test fallback implementation with opponent move
        let board = Board::new();
        let ply = 0;
        let mut pf = PatternFeatures::new(&board, ply);

        let sq = Square::D3;
        let flipped = Bitboard::new(1u64 << Square::D5.index());

        pf.update_fallback(sq, flipped, ply, SideToMove::Opponent);

        let mut new_board = board;
        new_board.opponent |= sq.bitboard() | flipped;
        new_board.player &= !flipped;
        let pf_fresh = PatternFeatures::new(&new_board, ply + 1);

        assert_opponent_update_matches_fresh(&pf, &pf_fresh, ply, "Fallback Opponent");
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_avx2_fallback_equivalence() {
        // Test that AVX2 and fallback implementations produce identical results
        let board = Board::new();
        let ply = 0;

        let sq = Square::D3;
        let flipped = Bitboard::new(1u64 << Square::D4.index());

        // Compute with fallback
        let mut pf_fallback = PatternFeatures::new(&board, ply);
        pf_fallback.update_fallback(sq, flipped, ply, SideToMove::Player);

        // Compute with AVX2
        let mut pf_avx2 = PatternFeatures::new(&board, ply);
        unsafe {
            pf_avx2.update_avx2(sq, flipped, ply, SideToMove::Player);
        }

        // Compare results
        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                pf_fallback.p_feature(ply + 1)[i],
                pf_avx2.p_feature(ply + 1)[i],
                "AVX2 vs fallback p_features[{}] mismatch: {} != {}",
                i,
                pf_fallback.p_feature(ply + 1)[i],
                pf_avx2.p_feature(ply + 1)[i]
            );
            assert_eq!(
                pf_fallback.o_feature(ply + 1)[i],
                pf_avx2.o_feature(ply + 1)[i],
                "AVX2 vs fallback o_features[{}] mismatch: {} != {}",
                i,
                pf_fallback.o_feature(ply + 1)[i],
                pf_avx2.o_feature(ply + 1)[i]
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_avx2_fallback_equivalence_multiple_positions() {
        // Test equivalence across multiple different positions for Player moves
        let test_cases = [
            // (player_squares, opponent_squares, move_square, flipped_squares)
            (
                vec![Square::D5, Square::E4],
                vec![Square::D4, Square::E5],
                Square::C4,
                vec![Square::D4],
            ),
            (
                vec![Square::D5, Square::E4, Square::C4],
                vec![Square::E5, Square::F4],
                Square::G4,
                vec![Square::F4],
            ),
        ];

        for (player_sqs, opponent_sqs, move_sq, flip_sqs) in test_cases {
            let mut player = 0u64;
            for sq in &player_sqs {
                player |= 1u64 << sq.index();
            }
            let mut opponent = 0u64;
            for sq in &opponent_sqs {
                opponent |= 1u64 << sq.index();
            }
            let mut flipped_bits = 0u64;
            for sq in &flip_sqs {
                flipped_bits |= 1u64 << sq.index();
            }

            let board = Board::from_bitboards(player, opponent);
            let ply = 5;
            let flipped = Bitboard::new(flipped_bits);

            let mut pf_fallback = PatternFeatures::new(&board, ply);
            pf_fallback.update_fallback(move_sq, flipped, ply, SideToMove::Player);

            let mut pf_avx2 = PatternFeatures::new(&board, ply);
            unsafe {
                pf_avx2.update_avx2(move_sq, flipped, ply, SideToMove::Player);
            }

            for i in 0..NUM_PATTERN_FEATURES {
                assert_eq!(
                    pf_fallback.p_feature(ply + 1)[i],
                    pf_avx2.p_feature(ply + 1)[i],
                    "Player position {:?} feature {} p mismatch",
                    move_sq,
                    i
                );
                assert_eq!(
                    pf_fallback.o_feature(ply + 1)[i],
                    pf_avx2.o_feature(ply + 1)[i],
                    "Player position {:?} feature {} o mismatch",
                    move_sq,
                    i
                );
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_avx2_fallback_equivalence_opponent_move() {
        // Test AVX2 vs fallback for Opponent moves
        let board = Board::new();
        let ply = 0;

        let sq = Square::D3;
        let flipped = Bitboard::new(1u64 << Square::D5.index());

        // Compute with fallback
        let mut pf_fallback = PatternFeatures::new(&board, ply);
        pf_fallback.update_fallback(sq, flipped, ply, SideToMove::Opponent);

        // Compute with AVX2
        let mut pf_avx2 = PatternFeatures::new(&board, ply);
        unsafe {
            pf_avx2.update_avx2(sq, flipped, ply, SideToMove::Opponent);
        }

        // Compare results
        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                pf_fallback.p_feature(ply + 1)[i],
                pf_avx2.p_feature(ply + 1)[i],
                "AVX2 Opponent vs fallback p_features[{}] mismatch: {} != {}",
                i,
                pf_fallback.p_feature(ply + 1)[i],
                pf_avx2.p_feature(ply + 1)[i]
            );
            assert_eq!(
                pf_fallback.o_feature(ply + 1)[i],
                pf_avx2.o_feature(ply + 1)[i],
                "AVX2 Opponent vs fallback o_features[{}] mismatch: {} != {}",
                i,
                pf_fallback.o_feature(ply + 1)[i],
                pf_avx2.o_feature(ply + 1)[i]
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn test_avx2_fallback_equivalence_opponent_multiple_flips() {
        // Test AVX2 vs fallback for Opponent moves with multiple flips
        let player = (1u64 << Square::B1.index())
            | (1u64 << Square::C1.index())
            | (1u64 << Square::D1.index());
        let opponent = 1u64 << Square::A1.index();
        let board = Board::from_bitboards(player, opponent);
        let ply = 5;

        let sq = Square::E1;
        let flipped = Bitboard::new(
            (1u64 << Square::B1.index())
                | (1u64 << Square::C1.index())
                | (1u64 << Square::D1.index()),
        );

        let mut pf_fallback = PatternFeatures::new(&board, ply);
        pf_fallback.update_fallback(sq, flipped, ply, SideToMove::Opponent);

        let mut pf_avx2 = PatternFeatures::new(&board, ply);
        unsafe {
            pf_avx2.update_avx2(sq, flipped, ply, SideToMove::Opponent);
        }

        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                pf_fallback.p_feature(ply + 1)[i],
                pf_avx2.p_feature(ply + 1)[i],
                "AVX2 Opponent multi-flip p[{}] mismatch",
                i
            );
            assert_eq!(
                pf_fallback.o_feature(ply + 1)[i],
                pf_avx2.o_feature(ply + 1)[i],
                "AVX2 Opponent multi-flip o[{}] mismatch",
                i
            );
        }
    }

    #[test]
    fn test_set_features_clears_buffer() {
        // Test that set_features clears the buffer before computing features
        // This ensures safety regardless of the input buffer's initial state
        let board = Board::new();

        // First, establish baseline with zero-initialized buffer
        let mut patterns = [0u16; NUM_PATTERN_FEATURES];
        set_features(&board, &mut patterns);

        // Now test with a non-zero initial buffer - should get the same results
        // because set_features clears the buffer first
        let mut patterns_with_init = [1u16; NUM_PATTERN_FEATURES];
        set_features(&board, &mut patterns_with_init);

        // Both should produce identical results
        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                patterns_with_init[i], patterns[i],
                "Pattern {} should be identical regardless of initial buffer state: {} != {}",
                i, patterns_with_init[i], patterns[i]
            );
        }

        // Also test with arbitrary garbage values
        let mut patterns_garbage = [0xFFFFu16; NUM_PATTERN_FEATURES];
        set_features(&board, &mut patterns_garbage);

        for i in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                patterns_garbage[i], patterns[i],
                "Pattern {} should be identical with garbage initial values: {} != {}",
                i, patterns_garbage[i], patterns[i]
            );
        }
    }

    #[test]
    fn test_max_features_per_square_not_exceeded() {
        // Verify that no square participates in more features than MAX_FEATURES_PER_SQUARE
        // by counting from EVAL_F2X definitions directly (independent of EVAL_X2F)
        let mut feature_counts = [0usize; BOARD_SQUARES];

        for f2x in &EVAL_F2X {
            for j in 0..f2x.n_square {
                let sq = f2x.squares[j];
                if !matches!(sq, Square::None) {
                    feature_counts[sq.index()] += 1;
                }
            }
        }

        for (sq_idx, &count) in feature_counts.iter().enumerate() {
            assert!(
                count <= MAX_FEATURES_PER_SQUARE,
                "Square {} participates in {} features, exceeds MAX_FEATURES_PER_SQUARE={}. \
                 EVAL_X2F may be silently truncating features!",
                sq_idx,
                count,
                MAX_FEATURES_PER_SQUARE
            );

            // Also verify EVAL_X2F has the same count
            assert_eq!(
                EVAL_X2F[sq_idx].n_features as usize, count,
                "Square {}: EVAL_X2F.n_features={} but EVAL_F2X shows {} features",
                sq_idx, EVAL_X2F[sq_idx].n_features, count
            );
        }
    }

    #[test]
    fn test_compute_pattern_feature_index_corner() {
        // Test that corner squares have correct pattern indices
        // A1 should participate in corner patterns (features 8, 10, 20)
        let board = 1u64 << Square::A1.index();

        // Feature 8: Row 1 [A1, B1, C1, D1, E1, F1, G1, H1]
        // A1 is the first square, so its weight is 3^7 = 2187
        let idx = compute_pattern_feature_index(board, &EVAL_F2X[8]);
        assert_eq!(idx, 2187, "A1 in feature 8 should have weight 3^7 = 2187");

        // Feature 10: Column A [A1, A2, A3, A4, A5, A6, A7, A8]
        // A1 is the first square, so its weight is 3^7 = 2187
        let idx = compute_pattern_feature_index(board, &EVAL_F2X[10]);
        assert_eq!(idx, 2187, "A1 in feature 10 should have weight 3^7 = 2187");

        // Feature 20: 3x3 corner [A1, B1, C1, A2, B2, C2, A3, B3, C3]
        // A1 is the first square, so its weight is 3^8 = 6561
        let idx = compute_pattern_feature_index(board, &EVAL_F2X[20]);
        assert_eq!(idx, 6561, "A1 in feature 20 should have weight 3^8 = 6561");
    }

    #[test]
    fn test_compute_pattern_feature_index_h8() {
        // Test H8 corner
        let board = 1u64 << Square::H8.index();

        // Feature 9: Row 8 [A8, B8, C8, D8, E8, F8, G8, H8]
        // H8 is the last (8th) square, so its weight is 3^0 = 1
        let idx = compute_pattern_feature_index(board, &EVAL_F2X[9]);
        assert_eq!(idx, 1, "H8 in feature 9 should have weight 3^0 = 1");

        // Feature 11: Column H [H1, H2, H3, H4, H5, H6, H7, H8]
        // H8 is the last (8th) square, so its weight is 3^0 = 1
        let idx = compute_pattern_feature_index(board, &EVAL_F2X[11]);
        assert_eq!(idx, 1, "H8 in feature 11 should have weight 3^0 = 1");

        // Feature 23: 3x3 corner H8 [H8, G8, F8, H7, G7, F7, H6, G6, F6]
        // H8 is the first square, so its weight is 3^8 = 6561
        let idx = compute_pattern_feature_index(board, &EVAL_F2X[23]);
        assert_eq!(idx, 6561, "H8 in feature 23 should have weight 3^8 = 6561");
    }

    #[test]
    fn test_compute_pattern_feature_index_center() {
        // Test center squares with golden values
        // Feature 6: [D3, E4, F5, D4, E5, C4, D5, E6]
        // D4 is at position 3 (0-indexed), so weight is 3^(7-3) = 3^4 = 81
        let board_d4 = 1u64 << Square::D4.index();
        let idx = compute_pattern_feature_index(board_d4, &EVAL_F2X[6]);
        assert_eq!(idx, 81, "D4 in feature 6 should have weight 3^4 = 81");

        // E5 is at position 4, so weight is 3^(7-4) = 3^3 = 27
        let board_e5 = 1u64 << Square::E5.index();
        let idx = compute_pattern_feature_index(board_e5, &EVAL_F2X[6]);
        assert_eq!(idx, 27, "E5 in feature 6 should have weight 3^3 = 27");

        // D5 is at position 6, so weight is 3^(7-6) = 3^1 = 3
        let board_d5 = 1u64 << Square::D5.index();
        let idx = compute_pattern_feature_index(board_d5, &EVAL_F2X[6]);
        assert_eq!(idx, 3, "D5 in feature 6 should have weight 3^1 = 3");
    }

    #[test]
    fn test_compute_pattern_feature_index_not_in_pattern() {
        // Test that squares not in a pattern return 0
        // A1 is not in feature 0 (which covers B-D columns, rows 2-4)
        let board_a1 = 1u64 << Square::A1.index();
        let idx = compute_pattern_feature_index(board_a1, &EVAL_F2X[0]);
        assert_eq!(idx, 0, "A1 should not be in feature 0");

        // H8 is not in feature 0
        let board_h8 = 1u64 << Square::H8.index();
        let idx = compute_pattern_feature_index(board_h8, &EVAL_F2X[0]);
        assert_eq!(idx, 0, "H8 should not be in feature 0");
    }

    #[test]
    fn test_compute_pattern_feature_index_consistency_with_eval_feature() {
        // Verify compute_pattern_feature_index matches EVAL_FEATURE lookup
        for (sq_idx, eval_feature) in EVAL_FEATURE.iter().enumerate() {
            let board = 1u64 << sq_idx;
            for pattern_idx in 0..NUM_PATTERN_FEATURES {
                let computed = compute_pattern_feature_index(board, &EVAL_F2X[pattern_idx]);
                let lookup = eval_feature[pattern_idx] as u32;
                assert_eq!(
                    computed, lookup,
                    "Square {} pattern {}: computed {} != lookup {}",
                    sq_idx, pattern_idx, computed, lookup
                );
            }
        }
    }
}
