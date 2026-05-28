//! Pattern-based feature extraction for neural network evaluation.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/eval.c>
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

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::constants::{BOARD_SQUARES, MAX_PLY};
use crate::search::side_to_move::SideToMove;
use crate::square::Square;
#[path = "pattern_table_data.rs"]
mod pattern_table_data;
pub use pattern_table_data::NUM_PATTERN_FEATURES;
use pattern_table_data::{
    EVAL_F2X_RAW, FEATURE_VECTOR_SIZE, SQ_NONE, compute_pattern_feature_index_raw,
};

const _: () = assert!(Square::None as u8 == SQ_NONE);
const _: () = assert!(BOARD_SQUARES == pattern_table_data::BOARD_SQUARES);

// Without this, a typo in `sq::*` would silently miscompile `EVAL_F2X_RAW`
// into an in-range (wrong) `Square` via the `transmute` in `build_eval_f2x`.
macro_rules! assert_sq_matches_square {
    ($($s:ident),* $(,)?) => {
        $(const _: () = assert!(pattern_table_data::sq::$s == Square::$s as u8);)*
    };
}
#[rustfmt::skip]
assert_sq_matches_square!(
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
);

/// Alias for NUM_PATTERN_FEATURES for compatibility.
pub const NUM_FEATURES: usize = NUM_PATTERN_FEATURES;

/// Maximum number of features that any single square can participate in.
const MAX_FEATURES_PER_SQUARE: usize = 5;

/// Maximum number of features per square as `u32` for loop comparisons.
const MAX_FEATURES_PER_SQUARE_U32: u32 = MAX_FEATURES_PER_SQUARE as u32;

#[cfg(any(
    all(
        target_arch = "x86_64",
        any(target_feature = "avx512bw", target_feature = "avx2")
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
use pattern_table_data::{FLIP_U16_BITS, FLIP_U16_TABLES, FLIP_U16_VALUES};
#[cfg(any(
    all(
        target_arch = "x86_64",
        any(target_feature = "avx512bw", target_feature = "avx2")
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
const FLIP_U16_MASK: u64 = (FLIP_U16_VALUES as u64) - 1;

/// Mapping from a feature index to the board squares it covers.
///
/// Each pattern feature examines a specific set of board squares.
/// This struct defines which squares belong to each pattern.
#[derive(Debug, Clone, Copy)]
pub struct FeatureToCoordinate {
    /// Number of squares in this pattern (up to 10).
    pub n_square: usize,
    /// Slots `n_square..` are filled with [`Square::None`].
    pub squares: [Square; 10],
}

impl FeatureToCoordinate {
    /// Creates a new feature-to-coordinate mapping with the given squares.
    pub const fn new(n_square: usize, squares: [Square; 10]) -> Self {
        Self { n_square, squares }
    }
}

/// Storage for pattern features.
#[derive(Debug, Clone, Copy)]
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
    /// Creates a new pattern feature initialized to zero.
    pub const fn new() -> Self {
        Self {
            data: [0; FEATURE_VECTOR_SIZE],
        }
    }

    /// Creates a pattern feature from an explicit array.
    pub const fn from_array(data: [u16; FEATURE_VECTOR_SIZE]) -> Self {
        Self { data }
    }

    /// Returns the feature value at `idx` without bounds checking.
    ///
    /// # Safety
    ///
    /// `idx` must be less than `FEATURE_VECTOR_SIZE`.
    pub unsafe fn get_unchecked(&self, idx: usize) -> u16 {
        *unsafe { self.data.get_unchecked(idx) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
impl PatternFeature {
    /// Returns a pointer to the internal data as a 512-bit SIMD vector.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid for as long as `self` is borrowed,
    /// and the data must be 64-byte aligned for aligned SIMD loads.
    #[inline(always)]
    unsafe fn as_m512_ptr(&self) -> *const std::arch::x86_64::__m512i {
        self.data.as_ptr() as *const std::arch::x86_64::__m512i
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl PatternFeature {
    /// Returns a pointer to the internal data as a 256-bit SIMD vector.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid for as long as `self` is borrowed,
    /// and the data must be 32-byte aligned for aligned SIMD loads.
    #[inline(always)]
    #[allow(dead_code)]
    unsafe fn as_m256_ptr(&self) -> *const std::arch::x86_64::__m256i {
        self.data.as_ptr() as *const std::arch::x86_64::__m256i
    }
}

#[cfg(target_arch = "wasm32")]
impl PatternFeature {
    /// Returns a pointer to the internal data as a 128-bit WASM SIMD vector.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid for as long as `self` is borrowed,
    /// and the data must be 16-byte aligned for aligned SIMD loads.
    #[inline(always)]
    unsafe fn as_v128_ptr(&self) -> *const core::arch::wasm32::v128 {
        self.data.as_ptr() as *const core::arch::wasm32::v128
    }
}

#[cfg(target_arch = "aarch64")]
impl PatternFeature {
    /// Returns a pointer to the internal data as an i16 pointer for NEON loads.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid for as long as `self` is borrowed.
    #[inline(always)]
    unsafe fn as_neon_ptr(&self) -> *const i16 {
        self.data.as_ptr() as *const i16
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

/// A mapping from a board square to the features it participates in.
///
/// Since each square can be part of multiple patterns, this struct
/// tracks all features affected when a square changes.
#[derive(Debug, Clone, Copy)]
struct CoordinateToFeature {
    /// Number of features this square participates in.
    n_features: u32,
    /// Array of `[feature_index, power_of_3]` pairs.
    ///
    /// `power_of_3` indicates the square's position in the pattern.
    features: [[u32; 2]; MAX_FEATURES_PER_SQUARE],
}

impl CoordinateToFeature {
    /// Returns a slice of the features this square participates in.
    #[inline(always)]
    fn features(&self) -> &[[u32; 2]] {
        &self.features[..self.n_features as usize]
    }
}

const fn build_eval_f2x() -> [FeatureToCoordinate; NUM_PATTERN_FEATURES] {
    let mut out = [FeatureToCoordinate {
        n_square: 0,
        squares: [Square::None; 10],
    }; NUM_PATTERN_FEATURES];
    let mut i = 0;
    while i < NUM_PATTERN_FEATURES {
        let (n_square, raw_squares) = EVAL_F2X_RAW[i];
        let mut squares = [Square::None; 10];
        let mut j = 0;
        while j < 10 {
            // SAFETY: `EVAL_F2X_RAW` only stores values in `0..=64`, every one
            // of which is a valid `Square` discriminant (asserted above).
            squares[j] = unsafe { std::mem::transmute::<u8, Square>(raw_squares[j]) };
            j += 1;
        }
        out[i] = FeatureToCoordinate { n_square, squares };
        i += 1;
    }
    out
}

/// Board squares that make up each pattern. Source of truth lives in
/// `pattern_table_data::EVAL_F2X_RAW`; this is the typed wrapper.
pub const EVAL_F2X: [FeatureToCoordinate; NUM_PATTERN_FEATURES] = build_eval_f2x();

/// Calculates the size of a pattern feature (3^n where n is the number of squares).
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
///
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
            let (n_square, squares) = EVAL_F2X_RAW[pattern_idx];
            feature_values[pattern_idx] =
                compute_pattern_feature_index_raw(board, n_square, squares) as u16;
            pattern_idx += 1;
        }

        result[square_idx] = PatternFeature::from_array(feature_values);
        square_idx += 1;
    }

    result
}

/// Type alias for the 16-bit chunk-sum lookup.
///
/// The table itself is generated by `build.rs` and embedded via
/// `include_bytes!`; see [`EVAL_FEATURE_U16_SUM`].
#[cfg(any(
    all(
        target_arch = "x86_64",
        any(target_feature = "avx512bw", target_feature = "avx2")
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
type EvalFeatureU16SumTable = [[PatternFeature; FLIP_U16_VALUES]; FLIP_U16_TABLES];

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
            let (n_square, raw_squares) = EVAL_F2X_RAW[feature_idx];
            let feature_value = compute_pattern_feature_index_raw(board, n_square, raw_squares);
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

/// Per-square base-3 place values for pattern input IDs.
///
/// `EVAL_FEATURE[sq]` is a 32-element vector (padded for SIMD) whose
/// `i`-th entry is the base-3 place value of `sq` within pattern `i`.
/// Entries for patterns that do not include `sq` are zero. Update code uses
/// this as the delta unit when the square's digit changes: player,
/// opponent, and empty squares are encoded as digits 0, 1, and 2, so placing
/// a player disc on an empty square subtracts `2 * power`.
///
/// Example: A1 belongs to pattern 4 (diagonal A1-H8), pattern 8 (row 1),
/// and pattern 10 (column A), each with place value `3^7` because A1 sits
/// at the back of those eight-square patterns, plus pattern 20 (corner A1
/// 3x3) with place value `3^8`:
///
/// ```text
/// EVAL_FEATURE[A1] = [
///     0, 0, 0, 0, 2187, 0, 0, 0,
///     2187, 0, 2187, 0, 0, 0, 0, 0,
///     0, 0, 0, 0, 6561, 0, 0, 0,
///     0, 0, 0, 0, 0, 0, 0, 0,
/// ];
/// ```
#[allow(dead_code)]
const EVAL_FEATURE: [PatternFeature; BOARD_SQUARES] = generate_eval_feature();

/// Chunk-sum lookup: a 64-bit `flipped` bitboard reduces to four
/// `EVAL_FEATURE_U16_SUM[chunk][mask]` loads summed elementwise. Emitted by
/// `build.rs` because const-eval is too slow for the 16 MiB payload.
#[cfg(any(
    all(
        target_arch = "x86_64",
        any(target_feature = "avx512bw", target_feature = "avx2")
    ),
    all(target_arch = "aarch64", target_feature = "neon")
))]
static EVAL_FEATURE_U16_SUM: EvalFeatureU16SumTable = {
    const TABLE_BYTES: usize = std::mem::size_of::<EvalFeatureU16SumTable>();
    const _: () =
        assert!(TABLE_BYTES == FLIP_U16_TABLES * FLIP_U16_VALUES * FEATURE_VECTOR_SIZE * 2);

    // SAFETY: size match is statically asserted above; supported SIMD targets
    // are LE so the on-disk LE bytes are native-order. The transmute is
    // by-value, so the `static`'s 64-byte alignment (from `PatternFeature`'s
    // `#[repr(align(64))]`) is what backs the aligned SIMD loads — not the
    // align-1 of the `include_bytes!` source.
    unsafe {
        std::mem::transmute::<[u8; TABLE_BYTES], EvalFeatureU16SumTable>(*include_bytes!(concat!(
            env!("OUT_DIR"),
            "/eval_feature_u16_sum.bin"
        )))
    }
};

/// Reverse mapping: square -> list of `(pattern_index, power)`.
///
/// `EVAL_X2F[sq]` lists every pattern `sq` participates in, paired with
/// the base-3 place value for `sq`'s position inside that pattern. On a
/// move or flip, the affected pattern input IDs are updated incrementally
/// by walking `EVAL_X2F[sq].features()`; the network input vector is
/// adjusted in place rather than rebuilt.
///
/// Example for A1:
///
/// ```text
/// EVAL_X2F[A1] = CoordinateToFeature {
///     n_features: 4,
///     features: [[4, 2187], [8, 2187], [10, 2187], [20, 6561]],
/// };
/// ```
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
    /// Creates an uninitialized `PatternFeatures` container.
    ///
    /// Callers must initialize the relevant ply slots before reading them.
    fn uninit() -> Self {
        PatternFeatures {
            p_features: [const { MaybeUninit::uninit() }; MAX_PLY],
            o_features: [const { MaybeUninit::uninit() }; MAX_PLY],
        }
    }

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

    /// Creates pattern features by copying pre-computed features at the given ply.
    ///
    /// This avoids the full board scan of [`PatternFeatures::new`] by reusing
    /// features that were already computed (e.g., from the split point owner's context).
    pub fn from_features(
        ply: usize,
        p_feature: &PatternFeature,
        o_feature: &PatternFeature,
    ) -> Self {
        debug_assert!(ply < MAX_PLY);
        let mut pf = Self::uninit();
        pf.p_features[ply] = MaybeUninit::new(*p_feature);
        pf.o_features[ply] = MaybeUninit::new(*o_feature);
        pf
    }

    /// Creates new pattern features from the given board position.
    pub fn new(board: &Board, ply: usize) -> Self {
        debug_assert!(ply < MAX_PLY);

        let mut pattern_features = Self::uninit();

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

    /// Incrementally updates pattern features after a move is made.
    ///
    /// Updates only the affected patterns rather than recomputing all features
    /// from scratch, handling both the placed disc and all flipped discs.
    #[inline(always)]
    pub fn update(&mut self, sq: Square, flipped: Bitboard, ply: usize, side_to_move: SideToMove) {
        debug_assert!(sq != Square::None);
        debug_assert!(!flipped.is_empty());
        debug_assert!(ply < MAX_PLY - 1);

        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx512bw") => {
                unsafe { self.update_avx512(sq, flipped, ply, side_to_move) }
            }
            all(target_arch = "x86_64", target_feature = "avx2") => {
                unsafe { self.update_avx2(sq, flipped, ply, side_to_move) }
            }
            all(target_arch = "aarch64", target_feature = "neon") => {
                unsafe { self.update_neon(sq, flipped, ply, side_to_move) }
            }
            all(target_arch = "wasm32", target_feature = "simd128") => {
                self.update_wasm_simd(sq, flipped, ply, side_to_move)
            }
            _ => {
                self.update_fallback(sq, flipped, ply, side_to_move);
            }
        }
    }

    /// Updates pattern features using the AVX-512BW implementation.
    ///
    /// Processes the whole 32-lane `u16` feature vector as a single 512-bit
    /// register and uses mask blends to avoid a hard-to-predict branch on
    /// `side_to_move`.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[target_feature(enable = "avx512bw")]
    fn update_avx512(
        &mut self,
        sq: Square,
        flipped: Bitboard,
        ply: usize,
        side_to_move: SideToMove,
    ) {
        use std::arch::x86_64::*;

        unsafe {
            let ef = &EVAL_FEATURE;
            let f = _mm512_load_si512(ef.get_unchecked(sq.index()).as_m512_ptr());
            let flipped_bits = flipped.bits();

            let load_u16_sum = |i: usize| {
                _mm512_load_si512(
                    EVAL_FEATURE_U16_SUM
                        .get_unchecked(i)
                        .get_unchecked(
                            ((flipped_bits >> (i * FLIP_U16_BITS)) & FLIP_U16_MASK) as usize,
                        )
                        .as_m512_ptr(),
                )
            };
            let s0 = load_u16_sum(0);
            let s1 = load_u16_sum(1);
            let s2 = load_u16_sum(2);
            let s3 = load_u16_sum(3);

            let sum = _mm512_add_epi16(_mm512_add_epi16(s0, s1), _mm512_add_epi16(s2, s3));

            let f_minus_sum = _mm512_sub_epi16(f, sum);
            let twof_plus_sum = _mm512_add_epi16(_mm512_add_epi16(f, f), sum);

            let p_feats = &mut self.p_features;
            let o_feats = &mut self.o_features;
            let p_in =
                _mm512_load_si512(p_feats.get_unchecked(ply).assume_init_ref().as_m512_ptr());
            let o_in =
                _mm512_load_si512(o_feats.get_unchecked(ply).assume_init_ref().as_m512_ptr());

            let p_out_ptr = p_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut __m512i;
            let o_out_ptr = o_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut __m512i;

            let side_mask: __mmask32 =
                0u32.wrapping_sub((side_to_move != SideToMove::Player) as u32);
            let delta_p = _mm512_mask_blend_epi16(side_mask, twof_plus_sum, f_minus_sum);
            let delta_o = _mm512_mask_blend_epi16(side_mask, f_minus_sum, twof_plus_sum);

            _mm512_store_si512(p_out_ptr, _mm512_sub_epi16(p_in, delta_p));
            _mm512_store_si512(o_out_ptr, _mm512_sub_epi16(o_in, delta_o));
        }
    }

    /// Updates pattern features using the AVX2 implementation.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[allow(dead_code)]
    fn update_avx2(&mut self, sq: Square, flipped: Bitboard, ply: usize, side_to_move: SideToMove) {
        use std::arch::x86_64::*;

        unsafe {
            let ef = &EVAL_FEATURE;

            let f_ptr = ef.get_unchecked(sq.index()).as_m256_ptr();
            let f0 = _mm256_load_si256(f_ptr);
            let f1 = _mm256_load_si256(f_ptr.add(1));

            let flipped_bits = flipped.bits();
            let load_u16_sum = |i: usize| {
                EVAL_FEATURE_U16_SUM
                    .get_unchecked(i)
                    .get_unchecked(((flipped_bits >> (i * FLIP_U16_BITS)) & FLIP_U16_MASK) as usize)
                    .as_m256_ptr()
            };

            let s0 = load_u16_sum(0);
            let s1 = load_u16_sum(1);
            let s2 = load_u16_sum(2);
            let s3 = load_u16_sum(3);

            let sum0 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_load_si256(s0), _mm256_load_si256(s1)),
                _mm256_add_epi16(_mm256_load_si256(s2), _mm256_load_si256(s3)),
            );
            let sum1 = _mm256_add_epi16(
                _mm256_add_epi16(_mm256_load_si256(s0.add(1)), _mm256_load_si256(s1.add(1))),
                _mm256_add_epi16(_mm256_load_si256(s2.add(1)), _mm256_load_si256(s3.add(1))),
            );

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

            let side_mask =
                _mm256_set1_epi16(0i16.wrapping_sub((side_to_move != SideToMove::Player) as i16));
            let delta_p0 = _mm256_blendv_epi8(twof_plus_sum_0, f_minus_sum_0, side_mask);
            let delta_p1 = _mm256_blendv_epi8(twof_plus_sum_1, f_minus_sum_1, side_mask);
            let delta_o0 = _mm256_blendv_epi8(f_minus_sum_0, twof_plus_sum_0, side_mask);
            let delta_o1 = _mm256_blendv_epi8(f_minus_sum_1, twof_plus_sum_1, side_mask);

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

    /// Updates pattern features using the WebAssembly SIMD implementation.
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

            let (delta_p0, delta_p1, delta_p2, delta_p3, delta_o0, delta_o1, delta_o2, delta_o3) =
                if side_to_move == SideToMove::Player {
                    (
                        twof_plus_sum_0,
                        twof_plus_sum_1,
                        twof_plus_sum_2,
                        twof_plus_sum_3,
                        f_minus_sum_0,
                        f_minus_sum_1,
                        f_minus_sum_2,
                        f_minus_sum_3,
                    )
                } else {
                    (
                        f_minus_sum_0,
                        f_minus_sum_1,
                        f_minus_sum_2,
                        f_minus_sum_3,
                        twof_plus_sum_0,
                        twof_plus_sum_1,
                        twof_plus_sum_2,
                        twof_plus_sum_3,
                    )
                };

            let p_out0 = i16x8_sub(p_in0, delta_p0);
            let p_out1 = i16x8_sub(p_in1, delta_p1);
            let p_out2 = i16x8_sub(p_in2, delta_p2);
            let p_out3 = i16x8_sub(p_in3, delta_p3);

            let o_out0 = i16x8_sub(o_in0, delta_o0);
            let o_out1 = i16x8_sub(o_in1, delta_o1);
            let o_out2 = i16x8_sub(o_in2, delta_o2);
            let o_out3 = i16x8_sub(o_in3, delta_o3);

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

    /// Updates pattern features using the ARM NEON implementation.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[target_feature(enable = "neon")]
    fn update_neon(&mut self, sq: Square, flipped: Bitboard, ply: usize, side_to_move: SideToMove) {
        use std::arch::aarch64::*;

        unsafe {
            let ef = &EVAL_FEATURE;
            let f_ptr = ef.get_unchecked(sq.index()).as_neon_ptr();
            let f0 = vld1q_s16(f_ptr);
            let f1 = vld1q_s16(f_ptr.add(8));
            let f2 = vld1q_s16(f_ptr.add(16));
            let f3 = vld1q_s16(f_ptr.add(24));

            let flipped_bits = flipped.bits();
            let load_u16_sum = |i: usize| {
                EVAL_FEATURE_U16_SUM
                    .get_unchecked(i)
                    .get_unchecked(((flipped_bits >> (i * FLIP_U16_BITS)) & FLIP_U16_MASK) as usize)
                    .as_neon_ptr()
            };

            let s0 = load_u16_sum(0);
            let s1 = load_u16_sum(1);
            let s2 = load_u16_sum(2);
            let s3 = load_u16_sum(3);

            let sum0 = vaddq_s16(
                vaddq_s16(vld1q_s16(s0), vld1q_s16(s1)),
                vaddq_s16(vld1q_s16(s2), vld1q_s16(s3)),
            );
            let sum1 = vaddq_s16(
                vaddq_s16(vld1q_s16(s0.add(8)), vld1q_s16(s1.add(8))),
                vaddq_s16(vld1q_s16(s2.add(8)), vld1q_s16(s3.add(8))),
            );
            let sum2 = vaddq_s16(
                vaddq_s16(vld1q_s16(s0.add(16)), vld1q_s16(s1.add(16))),
                vaddq_s16(vld1q_s16(s2.add(16)), vld1q_s16(s3.add(16))),
            );
            let sum3 = vaddq_s16(
                vaddq_s16(vld1q_s16(s0.add(24)), vld1q_s16(s1.add(24))),
                vaddq_s16(vld1q_s16(s2.add(24)), vld1q_s16(s3.add(24))),
            );

            let f2_0 = vshlq_n_s16::<1>(f0);
            let f2_1 = vshlq_n_s16::<1>(f1);
            let f2_2 = vshlq_n_s16::<1>(f2);
            let f2_3 = vshlq_n_s16::<1>(f3);

            let f_minus_sum_0 = vsubq_s16(f0, sum0);
            let f_minus_sum_1 = vsubq_s16(f1, sum1);
            let f_minus_sum_2 = vsubq_s16(f2, sum2);
            let f_minus_sum_3 = vsubq_s16(f3, sum3);

            let twof_plus_sum_0 = vaddq_s16(f2_0, sum0);
            let twof_plus_sum_1 = vaddq_s16(f2_1, sum1);
            let twof_plus_sum_2 = vaddq_s16(f2_2, sum2);
            let twof_plus_sum_3 = vaddq_s16(f2_3, sum3);

            let p_feats = &mut self.p_features;
            let o_feats = &mut self.o_features;
            let p_in_ptr = p_feats.get_unchecked(ply).assume_init_ref().as_neon_ptr();
            let o_in_ptr = o_feats.get_unchecked(ply).assume_init_ref().as_neon_ptr();

            let p_in0 = vld1q_s16(p_in_ptr);
            let p_in1 = vld1q_s16(p_in_ptr.add(8));
            let p_in2 = vld1q_s16(p_in_ptr.add(16));
            let p_in3 = vld1q_s16(p_in_ptr.add(24));

            let o_in0 = vld1q_s16(o_in_ptr);
            let o_in1 = vld1q_s16(o_in_ptr.add(8));
            let o_in2 = vld1q_s16(o_in_ptr.add(16));
            let o_in3 = vld1q_s16(o_in_ptr.add(24));

            let p_out_ptr = p_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut i16;
            let o_out_ptr = o_feats.get_unchecked_mut(ply + 1).as_mut_ptr() as *mut i16;

            if side_to_move == SideToMove::Player {
                vst1q_s16(p_out_ptr, vsubq_s16(p_in0, twof_plus_sum_0));
                vst1q_s16(p_out_ptr.add(8), vsubq_s16(p_in1, twof_plus_sum_1));
                vst1q_s16(p_out_ptr.add(16), vsubq_s16(p_in2, twof_plus_sum_2));
                vst1q_s16(p_out_ptr.add(24), vsubq_s16(p_in3, twof_plus_sum_3));

                vst1q_s16(o_out_ptr, vsubq_s16(o_in0, f_minus_sum_0));
                vst1q_s16(o_out_ptr.add(8), vsubq_s16(o_in1, f_minus_sum_1));
                vst1q_s16(o_out_ptr.add(16), vsubq_s16(o_in2, f_minus_sum_2));
                vst1q_s16(o_out_ptr.add(24), vsubq_s16(o_in3, f_minus_sum_3));
            } else {
                vst1q_s16(p_out_ptr, vsubq_s16(p_in0, f_minus_sum_0));
                vst1q_s16(p_out_ptr.add(8), vsubq_s16(p_in1, f_minus_sum_1));
                vst1q_s16(p_out_ptr.add(16), vsubq_s16(p_in2, f_minus_sum_2));
                vst1q_s16(p_out_ptr.add(24), vsubq_s16(p_in3, f_minus_sum_3));

                vst1q_s16(o_out_ptr, vsubq_s16(o_in0, twof_plus_sum_0));
                vst1q_s16(o_out_ptr.add(8), vsubq_s16(o_in1, twof_plus_sum_1));
                vst1q_s16(o_out_ptr.add(16), vsubq_s16(o_in2, twof_plus_sum_2));
                vst1q_s16(o_out_ptr.add(24), vsubq_s16(o_in3, twof_plus_sum_3));
            }
        }
    }

    /// Updates pattern features using the scalar fallback.
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

/// Computes pattern features for a board position into `patterns`.
///
/// Each pattern is encoded as a base-3 number representing the
/// configuration of discs in that pattern.
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

/// Returns the ternary color of a square: 0 = player, 1 = opponent, 2 = empty.
#[inline]
fn get_square_color(board: &Board, sq: Square) -> u16 {
    if board.player().contains(sq) {
        0
    } else if board.opponent().contains(sq) {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flip;

    #[derive(Clone, Copy)]
    struct MoveCase {
        label: &'static str,
        board: Board,
        sq: Square,
        flipped: Bitboard,
        side_to_move: SideToMove,
        ply: usize,
    }

    fn bitboard(squares: &[Square]) -> Bitboard {
        let mut bits = 0u64;
        for &sq in squares {
            bits |= 1u64 << sq.index();
        }
        Bitboard::new(bits)
    }

    fn board(player: &[Square], opponent: &[Square]) -> Board {
        Board::from_bitboards(bitboard(player), bitboard(opponent))
    }

    fn pow3(exp: usize) -> usize {
        let mut value = 1;
        for _ in 0..exp {
            value *= 3;
        }
        value
    }

    fn reference_color(board: &Board, sq: Square) -> u16 {
        if board.player().contains(sq) {
            0
        } else if board.opponent().contains(sq) {
            1
        } else {
            2
        }
    }

    fn reference_pattern_value(board: &Board, feature: &FeatureToCoordinate) -> u16 {
        let mut value = 0u16;
        for &sq in &feature.squares[..feature.n_square] {
            value = value * 3 + reference_color(board, sq);
        }
        value
    }

    fn expected_square_power(feature: &FeatureToCoordinate, sq: Square) -> Option<u32> {
        for (pos, &pattern_sq) in feature.squares[..feature.n_square].iter().enumerate() {
            if pattern_sq == sq {
                return Some(pow3(feature.n_square - pos - 1) as u32);
            }
        }
        None
    }

    #[track_caller]
    fn assert_pattern_feature_eq(
        label: &str,
        slot: &str,
        expected: &PatternFeature,
        actual: &PatternFeature,
    ) {
        for idx in 0..FEATURE_VECTOR_SIZE {
            assert_eq!(actual[idx], expected[idx], "{label} {slot}[{idx}]");
        }
    }

    #[cfg(any(
        all(
            target_arch = "x86_64",
            any(target_feature = "avx512bw", target_feature = "avx2")
        ),
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[track_caller]
    fn assert_features_match(
        label: &str,
        expected: &PatternFeatures,
        actual: &PatternFeatures,
        ply: usize,
    ) {
        assert_pattern_feature_eq(
            label,
            "player",
            expected.p_feature(ply),
            actual.p_feature(ply),
        );
        assert_pattern_feature_eq(
            label,
            "opponent",
            expected.o_feature(ply),
            actual.o_feature(ply),
        );
    }

    fn rebuilt_board_after(case: MoveCase) -> Board {
        match case.side_to_move {
            SideToMove::Player => case.board.make_move_with_flipped(case.flipped, case.sq),
            SideToMove::Opponent => Board::from_bitboards(
                case.board.player() & !case.flipped,
                case.board.opponent() | case.flipped | case.sq.bitboard(),
            ),
        }
    }

    #[track_caller]
    fn assert_case_matches_full_rebuild(updated: &PatternFeatures, case: MoveCase) {
        let rebuilt = rebuilt_board_after(case);
        let fresh = PatternFeatures::new(&rebuilt, case.ply + 1);

        match case.side_to_move {
            SideToMove::Player => {
                assert_pattern_feature_eq(
                    case.label,
                    "player",
                    fresh.o_feature(case.ply + 1),
                    updated.p_feature(case.ply + 1),
                );
                assert_pattern_feature_eq(
                    case.label,
                    "opponent",
                    fresh.p_feature(case.ply + 1),
                    updated.o_feature(case.ply + 1),
                );
            }
            SideToMove::Opponent => {
                assert_pattern_feature_eq(
                    case.label,
                    "player",
                    fresh.p_feature(case.ply + 1),
                    updated.p_feature(case.ply + 1),
                );
                assert_pattern_feature_eq(
                    case.label,
                    "opponent",
                    fresh.o_feature(case.ply + 1),
                    updated.o_feature(case.ply + 1),
                );
            }
        }
    }

    #[track_caller]
    fn assert_case_flips_match_move_generator(case: MoveCase) {
        let generated = match case.side_to_move {
            SideToMove::Player => flip::flip(case.sq, case.board.player(), case.board.opponent()),
            SideToMove::Opponent => flip::flip(case.sq, case.board.opponent(), case.board.player()),
        };

        assert_eq!(generated, case.flipped, "{} flipped discs", case.label);
    }

    fn update_cases() -> [MoveCase; 6] {
        [
            MoveCase {
                label: "opening player move",
                board: Board::new(),
                sq: Square::D3,
                flipped: Square::D4.bitboard(),
                side_to_move: SideToMove::Player,
                ply: 0,
            },
            MoveCase {
                label: "opening opponent move",
                board: Board::new(),
                sq: Square::C5,
                flipped: Square::D5.bitboard(),
                side_to_move: SideToMove::Opponent,
                ply: 1,
            },
            MoveCase {
                label: "edge line flip",
                board: board(&[Square::A1], &[Square::B1, Square::C1, Square::D1]),
                sq: Square::E1,
                flipped: bitboard(&[Square::B1, Square::C1, Square::D1]),
                side_to_move: SideToMove::Player,
                ply: 10,
            },
            MoveCase {
                label: "multi direction flip",
                board: board(
                    &[Square::D1, Square::A4, Square::D8],
                    &[
                        Square::D2,
                        Square::D3,
                        Square::B4,
                        Square::C4,
                        Square::D5,
                        Square::D6,
                        Square::D7,
                    ],
                ),
                sq: Square::D4,
                flipped: bitboard(&[
                    Square::D2,
                    Square::D3,
                    Square::B4,
                    Square::C4,
                    Square::D5,
                    Square::D6,
                    Square::D7,
                ]),
                side_to_move: SideToMove::Player,
                ply: 15,
            },
            MoveCase {
                label: "diagonal flip across every u16 chunk",
                board: board(
                    &[Square::A1],
                    &[
                        Square::B2,
                        Square::C3,
                        Square::D4,
                        Square::E5,
                        Square::F6,
                        Square::G7,
                    ],
                ),
                sq: Square::H8,
                flipped: bitboard(&[
                    Square::B2,
                    Square::C3,
                    Square::D4,
                    Square::E5,
                    Square::F6,
                    Square::G7,
                ]),
                side_to_move: SideToMove::Player,
                ply: 3,
            },
            MoveCase {
                label: "high ply opponent corner move",
                board: board(&[Square::B1], &[Square::C1]),
                sq: Square::A1,
                flipped: Square::B1.bitboard(),
                side_to_move: SideToMove::Opponent,
                ply: MAX_PLY - 2,
            },
        ]
    }

    #[test]
    fn pattern_feature_storage_preserves_values_and_alignment() {
        let zeros = PatternFeature::new();
        assert_eq!(zeros.data, [0; FEATURE_VECTOR_SIZE]);
        assert_eq!(std::mem::align_of::<PatternFeature>(), 64);
        assert_eq!(
            std::mem::size_of::<PatternFeature>(),
            FEATURE_VECTOR_SIZE * std::mem::size_of::<u16>()
        );

        let data = std::array::from_fn(|idx| (idx as u16).wrapping_mul(17).wrapping_add(3));
        let mut feature = PatternFeature::from_array(data);

        for idx in 0..FEATURE_VECTOR_SIZE {
            assert_eq!(feature[idx], data[idx], "idx {idx}");
        }

        feature[7] = 0xBEEF;
        assert_eq!(feature[7], 0xBEEF);
        unsafe {
            assert_eq!(feature.get_unchecked(7), 0xBEEF);
        }
    }

    #[test]
    fn pattern_definitions_have_expected_shape_padding_and_unique_squares() {
        assert_eq!(NUM_FEATURES, NUM_PATTERN_FEATURES);
        assert_eq!(NUM_PATTERN_FEATURES, 32);
        assert_eq!(FEATURE_VECTOR_SIZE, 32);

        for (idx, feature) in EVAL_F2X.iter().enumerate() {
            let expected_len = match idx {
                0..=19 => 8,
                20..=27 => 9,
                28..=31 => 7,
                _ => unreachable!(),
            };
            assert_eq!(feature.n_square, expected_len, "pattern {idx}");

            let mut seen = [false; BOARD_SQUARES];
            for pos in 0..feature.n_square {
                let sq = feature.squares[pos];
                assert_ne!(sq, Square::None, "pattern {idx} active slot {pos}");
                assert!(
                    !seen[sq.index()],
                    "pattern {idx} contains {sq:?} more than once"
                );
                seen[sq.index()] = true;
            }

            for pos in feature.n_square..feature.squares.len() {
                assert_eq!(
                    feature.squares[pos],
                    Square::None,
                    "pattern {idx} tail {pos}"
                );
            }
        }
    }

    #[test]
    fn feature_dimensions_are_derived_from_pattern_lengths() {
        let mut expected_offsets = [0usize; NUM_PATTERN_FEATURES];
        let mut running_total = 0usize;

        for (idx, feature) in EVAL_F2X.iter().enumerate() {
            expected_offsets[idx] = running_total;
            let expected_size = pow3(feature.n_square);
            assert_eq!(calc_pattern_size(idx), expected_size, "pattern {idx}");
            running_total += expected_size;
        }

        assert_eq!(calc_feature_offsets(), expected_offsets);
        assert_eq!(PATTERN_FEATURE_OFFSETS, expected_offsets);
        assert_eq!(sum_eval_f2x(), running_total);
        assert_eq!(INPUT_FEATURE_DIMS, running_total);
    }

    #[test]
    fn square_color_uses_player_perspective() {
        let board = Board::new();

        assert_eq!(get_square_color(&board, Square::D5), 0);
        assert_eq!(get_square_color(&board, Square::E4), 0);
        assert_eq!(get_square_color(&board, Square::D4), 1);
        assert_eq!(get_square_color(&board, Square::E5), 1);
        assert_eq!(get_square_color(&board, Square::A1), 2);

        let switched = board.switch_players();
        assert_eq!(get_square_color(&switched, Square::D5), 1);
        assert_eq!(get_square_color(&switched, Square::D4), 0);
    }

    #[test]
    fn set_features_encodes_pattern_digits_in_declared_order() {
        let board = board(
            &[Square::A1, Square::D1, Square::D5, Square::E4],
            &[Square::B1, Square::H1, Square::D4, Square::E5],
        );
        let mut patterns = [0xFFFF; NUM_PATTERN_FEATURES];

        set_features(&board, &mut patterns);

        let row_1_expected = 729 + 2 * 243 + 2 * 27 + 2 * 9 + 2 * 3 + 1;
        assert_eq!(patterns[8], row_1_expected);

        for (idx, feature) in EVAL_F2X.iter().enumerate() {
            let expected = reference_pattern_value(&board, feature);
            assert_eq!(patterns[idx], expected, "pattern {idx}");
            assert!(
                usize::from(patterns[idx]) < calc_pattern_size(idx),
                "pattern {idx} value {} outside ternary range",
                patterns[idx]
            );
        }
    }

    #[test]
    fn set_features_overwrites_existing_buffer_contents() {
        let board = Board::new();
        let mut zeroed = [0; NUM_PATTERN_FEATURES];
        let mut filled = [0xFFFF; NUM_PATTERN_FEATURES];

        set_features(&board, &mut zeroed);
        set_features(&board, &mut filled);

        assert_eq!(filled, zeroed);
    }

    #[test]
    fn pattern_features_new_matches_set_features_for_both_perspectives() {
        let board = board(
            &[Square::A1, Square::D4, Square::G7],
            &[Square::B1, Square::C3, Square::H8],
        );
        let ply = 7;
        let features = PatternFeatures::new(&board, ply);
        let mut expected_player = [0; NUM_PATTERN_FEATURES];
        let mut expected_opponent = [0; NUM_PATTERN_FEATURES];

        set_features(&board, &mut expected_player);
        set_features(&board.switch_players(), &mut expected_opponent);

        for idx in 0..NUM_PATTERN_FEATURES {
            assert_eq!(
                features.p_feature(ply)[idx],
                expected_player[idx],
                "player {idx}"
            );
            assert_eq!(
                features.o_feature(ply)[idx],
                expected_opponent[idx],
                "opponent {idx}"
            );
        }
    }

    #[test]
    fn pattern_features_from_features_copies_the_requested_ply() {
        let board = board(
            &[Square::A1, Square::B2, Square::C3],
            &[Square::H8, Square::G7],
        );
        let source_ply = 4;
        let target_ply = 11;
        let source = PatternFeatures::new(&board, source_ply);

        let copied = PatternFeatures::from_features(
            target_ply,
            source.p_feature(source_ply),
            source.o_feature(source_ply),
        );

        assert_pattern_feature_eq(
            "from_features",
            "player",
            source.p_feature(source_ply),
            copied.p_feature(target_ply),
        );
        assert_pattern_feature_eq(
            "from_features",
            "opponent",
            source.o_feature(source_ply),
            copied.o_feature(target_ply),
        );
    }

    #[test]
    fn forward_and_reverse_square_maps_are_exact_inverses() {
        for (sq_idx, sq) in Square::iter().enumerate() {
            let mut expected_feature = [0u16; FEATURE_VECTOR_SIZE];
            let mut expected_pairs = [[0u32; 2]; MAX_FEATURES_PER_SQUARE];
            let mut expected_count = 0usize;

            for (feature_idx, feature) in EVAL_F2X.iter().enumerate() {
                if let Some(power) = expected_square_power(feature, sq) {
                    expected_feature[feature_idx] = power as u16;
                    assert!(
                        expected_count < MAX_FEATURES_PER_SQUARE,
                        "{sq:?} participates in more than {MAX_FEATURES_PER_SQUARE} patterns"
                    );
                    expected_pairs[expected_count] = [feature_idx as u32, power];
                    expected_count += 1;
                }
            }

            assert!(expected_count > 0, "{sq:?} is not covered by any pattern");
            assert_eq!(EVAL_FEATURE[sq_idx].data, expected_feature, "{sq:?}");

            let x2f = &EVAL_X2F[sq_idx];
            assert_eq!(x2f.n_features as usize, expected_count, "{sq:?}");
            assert_eq!(x2f.features(), &expected_pairs[..expected_count], "{sq:?}");
        }
    }

    #[test]
    fn single_square_raw_encoder_matches_eval_feature_table() {
        for (sq_idx, eval_feature) in EVAL_FEATURE.iter().enumerate() {
            let board = 1u64 << sq_idx;

            for (pattern_idx, feature) in EVAL_F2X.iter().enumerate() {
                let raw_squares = feature.squares.map(|sq| sq as u8);
                let encoded =
                    compute_pattern_feature_index_raw(board, feature.n_square, raw_squares);

                assert_eq!(
                    encoded,
                    u32::from(eval_feature[pattern_idx]),
                    "square {sq_idx}, pattern {pattern_idx}"
                );
            }
        }
    }

    #[test]
    fn incremental_update_matches_full_rebuild_for_movegen_verified_cases() {
        for case in update_cases() {
            assert_case_flips_match_move_generator(case);

            let mut updated = PatternFeatures::new(&case.board, case.ply);
            updated.update(case.sq, case.flipped, case.ply, case.side_to_move);

            assert_case_matches_full_rebuild(&updated, case);
        }
    }

    #[test]
    fn scalar_fallback_update_matches_full_rebuild_for_movegen_verified_cases() {
        for case in update_cases() {
            assert_case_flips_match_move_generator(case);

            let mut updated = PatternFeatures::new(&case.board, case.ply);
            updated.update_fallback(case.sq, case.flipped, case.ply, case.side_to_move);

            assert_case_matches_full_rebuild(&updated, case);
        }
    }

    #[cfg(any(
        all(
            target_arch = "x86_64",
            any(target_feature = "avx512bw", target_feature = "avx2")
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    ))]
    #[test]
    fn eval_feature_u16_sum_matches_scalar_square_sums_for_every_mask() {
        for (chunk_idx, chunk) in EVAL_FEATURE_U16_SUM.iter().enumerate() {
            for (mask, entry) in chunk.iter().enumerate() {
                let mut expected = [0u16; FEATURE_VECTOR_SIZE];

                for bit in 0..FLIP_U16_BITS {
                    if (mask & (1usize << bit)) == 0 {
                        continue;
                    }

                    let square_idx = chunk_idx * FLIP_U16_BITS + bit;
                    for (feature_idx, expected_value) in expected.iter_mut().enumerate() {
                        *expected_value += EVAL_FEATURE[square_idx][feature_idx];
                    }
                }

                assert_eq!(entry.data, expected, "chunk {chunk_idx}, mask {mask:#06x}");
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    #[test]
    fn avx512_update_matches_fallback_for_move_cases() {
        for case in update_cases() {
            let mut expected = PatternFeatures::new(&case.board, case.ply);
            expected.update_fallback(case.sq, case.flipped, case.ply, case.side_to_move);

            let mut actual = PatternFeatures::new(&case.board, case.ply);
            unsafe {
                actual.update_avx512(case.sq, case.flipped, case.ply, case.side_to_move);
            }

            assert_features_match(case.label, &expected, &actual, case.ply + 1);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[test]
    fn avx2_update_matches_fallback_for_move_cases() {
        for case in update_cases() {
            let mut expected = PatternFeatures::new(&case.board, case.ply);
            expected.update_fallback(case.sq, case.flipped, case.ply, case.side_to_move);

            let mut actual = PatternFeatures::new(&case.board, case.ply);
            unsafe {
                actual.update_avx2(case.sq, case.flipped, case.ply, case.side_to_move);
            }

            assert_features_match(case.label, &expected, &actual, case.ply + 1);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn neon_update_matches_fallback_for_move_cases() {
        for case in update_cases() {
            let mut expected = PatternFeatures::new(&case.board, case.ply);
            expected.update_fallback(case.sq, case.flipped, case.ply, case.side_to_move);

            let mut actual = PatternFeatures::new(&case.board, case.ply);
            unsafe {
                actual.update_neon(case.sq, case.flipped, case.ply, case.side_to_move);
            }

            assert_features_match(case.label, &expected, &actual, case.ply + 1);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[test]
    fn wasm_simd_update_matches_fallback_for_move_cases() {
        for case in update_cases() {
            let mut expected = PatternFeatures::new(&case.board, case.ply);
            expected.update_fallback(case.sq, case.flipped, case.ply, case.side_to_move);

            let mut actual = PatternFeatures::new(&case.board, case.ply);
            actual.update_wasm_simd(case.sq, case.flipped, case.ply, case.side_to_move);

            assert_features_match(case.label, &expected, &actual, case.ply + 1);
        }
    }
}
