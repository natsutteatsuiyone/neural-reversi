// Pattern table data shared between `build.rs` and the runtime module.
//
// Single source of truth for `EVAL_F2X`, the chunk-sum helper, and the
// constants needed to generate the AVX2/AVX-512 chunk-sum lookup. Included
// from `build.rs` via `include!` and from `pattern_feature.rs` via
// `#[path]`, so this file must only use primitive types and must contain no
// `use` statements that depend on the rest of the crate.

pub const NUM_PATTERN_FEATURES: usize = 32;
/// Padded for SIMD alignment.
pub const FEATURE_VECTOR_SIZE: usize = 32;
pub const BOARD_SQUARES: usize = 64;
/// Base for pattern encoding (ternary: empty=2, opponent=1, player=0).
pub const PATTERN_BASE: u32 = 3;
/// Sentinel for unused slots in `EVAL_F2X_RAW.squares`.
pub const SQ_NONE: u8 = 64;

#[allow(dead_code)]
pub const FLIP_U16_BITS: usize = 16;
#[allow(dead_code)]
pub const FLIP_U16_TABLES: usize = BOARD_SQUARES / FLIP_U16_BITS;
#[allow(dead_code)]
pub const FLIP_U16_VALUES: usize = 1 << FLIP_U16_BITS;

/// Mirrors `Square` discriminants (A1=0..H8=63); pinned by
/// `assert_sq_matches_square!` in `pattern_feature.rs`.
#[rustfmt::skip]
#[allow(dead_code)]
pub mod sq {
    pub const A1: u8 =  0; pub const B1: u8 =  1; pub const C1: u8 =  2; pub const D1: u8 =  3; pub const E1: u8 =  4; pub const F1: u8 =  5; pub const G1: u8 =  6; pub const H1: u8 =  7;
    pub const A2: u8 =  8; pub const B2: u8 =  9; pub const C2: u8 = 10; pub const D2: u8 = 11; pub const E2: u8 = 12; pub const F2: u8 = 13; pub const G2: u8 = 14; pub const H2: u8 = 15;
    pub const A3: u8 = 16; pub const B3: u8 = 17; pub const C3: u8 = 18; pub const D3: u8 = 19; pub const E3: u8 = 20; pub const F3: u8 = 21; pub const G3: u8 = 22; pub const H3: u8 = 23;
    pub const A4: u8 = 24; pub const B4: u8 = 25; pub const C4: u8 = 26; pub const D4: u8 = 27; pub const E4: u8 = 28; pub const F4: u8 = 29; pub const G4: u8 = 30; pub const H4: u8 = 31;
    pub const A5: u8 = 32; pub const B5: u8 = 33; pub const C5: u8 = 34; pub const D5: u8 = 35; pub const E5: u8 = 36; pub const F5: u8 = 37; pub const G5: u8 = 38; pub const H5: u8 = 39;
    pub const A6: u8 = 40; pub const B6: u8 = 41; pub const C6: u8 = 42; pub const D6: u8 = 43; pub const E6: u8 = 44; pub const F6: u8 = 45; pub const G6: u8 = 46; pub const H6: u8 = 47;
    pub const A7: u8 = 48; pub const B7: u8 = 49; pub const C7: u8 = 50; pub const D7: u8 = 51; pub const E7: u8 = 52; pub const F7: u8 = 53; pub const G7: u8 = 54; pub const H7: u8 = 55;
    pub const A8: u8 = 56; pub const B8: u8 = 57; pub const C8: u8 = 58; pub const D8: u8 = 59; pub const E8: u8 = 60; pub const F8: u8 = 61; pub const G8: u8 = 62; pub const H8: u8 = 63;
}
use sq::*;

/// Board squares that make up each pattern. Each entry is
/// `(n_square, squares)`; unused slots use `SQ_NONE`.
#[rustfmt::skip]
pub const EVAL_F2X_RAW: [(usize, [u8; 10]); NUM_PATTERN_FEATURES] = [
    (8, [C2, D2, E2, F2, C3, D3, E3, F3, SQ_NONE, SQ_NONE]),  // 0: inner top
    (8, [C7, D7, E7, F7, C6, D6, E6, F6, SQ_NONE, SQ_NONE]),  // 1: inner bottom
    (8, [B3, B4, B5, B6, C3, C4, C5, C6, SQ_NONE, SQ_NONE]),  // 2: inner left
    (8, [G3, G4, G5, G6, F3, F4, F5, F6, SQ_NONE, SQ_NONE]),  // 3: inner right

    (8, [A1, B2, C3, D4, E5, F6, G7, H8, SQ_NONE, SQ_NONE]),  // 4: diagonal A1-H8
    (8, [H1, G2, F3, E4, D5, C6, B7, A8, SQ_NONE, SQ_NONE]),  // 5: diagonal H1-A8

    (8, [C4, D4, E4, F4, C5, D5, E5, F5, SQ_NONE, SQ_NONE]),  // 6: center 2x4 horizontal
    (8, [D3, E3, D4, E4, D5, E5, D6, E6, SQ_NONE, SQ_NONE]),  // 7: center 2x4 vertical

    (8, [A1, B1, C1, D1, E1, F1, G1, H1, SQ_NONE, SQ_NONE]),  // 8: row 1
    (8, [A8, B8, C8, D8, E8, F8, G8, H8, SQ_NONE, SQ_NONE]),  // 9: row 8
    (8, [A1, A2, A3, A4, A5, A6, A7, A8, SQ_NONE, SQ_NONE]),  // 10: column A
    (8, [H1, H2, H3, H4, H5, H6, H7, H8, SQ_NONE, SQ_NONE]),  // 11: column H

    (8, [B1, C1, D1, E1, B2, C2, D2, E2, SQ_NONE, SQ_NONE]),  // 12: top edge 2x4
    (8, [G1, F1, E1, D1, G2, F2, E2, D2, SQ_NONE, SQ_NONE]),  // 13: top edge 2x4 (mirrored)
    (8, [B8, C8, D8, E8, B7, C7, D7, E7, SQ_NONE, SQ_NONE]),  // 14: bottom edge 2x4
    (8, [G8, F8, E8, D8, G7, F7, E7, D7, SQ_NONE, SQ_NONE]),  // 15: bottom edge 2x4 (mirrored)
    (8, [A2, A3, A4, A5, B2, B3, B4, B5, SQ_NONE, SQ_NONE]),  // 16: left edge 2x4
    (8, [A7, A6, A5, A4, B7, B6, B5, B4, SQ_NONE, SQ_NONE]),  // 17: left edge 2x4 (mirrored)
    (8, [H2, H3, H4, H5, G2, G3, G4, G5, SQ_NONE, SQ_NONE]),  // 18: right edge 2x4
    (8, [H7, H6, H5, H4, G7, G6, G5, G4, SQ_NONE, SQ_NONE]),  // 19: right edge 2x4 (mirrored)

    (9, [A1, B1, C1, A2, B2, C2, A3, B3, C3, SQ_NONE]),  // 20: corner A1 3x3
    (9, [H1, G1, F1, H2, G2, F2, H3, G3, F3, SQ_NONE]),  // 21: corner H1 3x3
    (9, [A8, B8, C8, A7, B7, C7, A6, B6, C6, SQ_NONE]),  // 22: corner A8 3x3
    (9, [H8, G8, F8, H7, G7, F7, H6, G6, F6, SQ_NONE]),  // 23: corner H8 3x3

    (9, [B2, C2, D2, B3, C3, D3, B4, C4, D4, SQ_NONE]),  // 24: center 3x3 NW
    (9, [G2, F2, E2, G3, F3, E3, G4, F4, E4, SQ_NONE]),  // 25: center 3x3 NE
    (9, [B7, C7, D7, B6, C6, D6, B5, C5, D5, SQ_NONE]),  // 26: center 3x3 SW
    (9, [G7, F7, E7, G6, F6, E6, G5, F5, E5, SQ_NONE]),  // 27: center 3x3 SE

    (7, [B1, C2, D3, E4, F5, G6, H7, SQ_NONE, SQ_NONE, SQ_NONE]),  // 28: adjacent diagonal B1-H7
    (7, [A2, B3, C4, D5, E6, F7, G8, SQ_NONE, SQ_NONE, SQ_NONE]),  // 29: adjacent diagonal A2-G8
    (7, [G1, F2, E3, D4, C5, B6, A7, SQ_NONE, SQ_NONE, SQ_NONE]),  // 30: adjacent diagonal G1-A7
    (7, [H2, G3, F4, E5, D6, C7, B8, SQ_NONE, SQ_NONE, SQ_NONE]),  // 31: adjacent diagonal H2-B8
];

/// Returns the base-3 place value of the single set bit in `board` within
/// `squares[..n_square]`, or `0` if that bit lies outside the pattern.
///
/// `board` must have at most one bit set.
pub const fn compute_pattern_feature_index_raw(
    board: u64,
    n_square: usize,
    squares: [u8; 10],
) -> u32 {
    let mut multiplier = 0u32;
    let mut feature_index = 0u32;
    let mut i = n_square;
    while i > 0 {
        i -= 1;
        let s = squares[i];
        if s == SQ_NONE {
            continue;
        }
        multiplier = if multiplier == 0 {
            1
        } else {
            multiplier * PATTERN_BASE
        };
        if board & (1u64 << s) != 0 {
            feature_index = multiplier;
        }
    }
    feature_index
}
