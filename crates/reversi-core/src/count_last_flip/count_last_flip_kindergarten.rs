//! Based on count_last_flip_kindergarten.c from edax-reversi.
//! Reference: https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/count_last_flip_kindergarten.c

use crate::square::Square;
use crate::uget;

/// precomputed count flip array
#[rustfmt::skip]
pub static COUNT_FLIP: [[i8; 256]; 8] = [
    [
         0,  0,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
        10, 10,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
        12, 12,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
        10, 10,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
    ],
    [
         0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         8,  8,  8,  8,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
        10, 10, 10, 10,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         8,  8,  8,  8,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
    ],
    [
         0,  2,  0,  0,  0,  2,  0,  0,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         6,  8,  6,  6,  6,  8,  6,  6,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         8, 10,  8,  8,  8, 10,  8,  8,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         6,  8,  6,  6,  6,  8,  6,  6,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
    ],
    [
         0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         4,  8,  6,  6,  4,  4,  4,  4,  4,  8,  6,  6,  4,  4,  4,  4,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         6, 10,  8,  8,  6,  6,  6,  6,  6, 10,  8,  8,  6,  6,  6,  6,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         4,  8,  6,  6,  4,  4,  4,  4,  4,  8,  6,  6,  4,  4,  4,  4,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
    ],
    [
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         4, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  4, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
    [
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         2, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         2, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
    [
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
    [
         0, 12, 10, 10,  8,  8,  8,  8,  6,  6,  6,  6,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 12, 10, 10,  8,  8,  8,  8,  6,  6,  6,  6,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
];

#[inline(always)]
fn lookup(table: usize, idx: usize) -> i32 {
    *uget!(COUNT_FLIP; table, idx) as i32
}

/// Counts last flipped discs when playing on square A1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a1(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(0, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b1(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = (((p & 0x0080402010080402u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(1, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c1(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = ((p & 0x0000804020110a04u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square D1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d1(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = ((p & 0x0000008041221408u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square E1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e1(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = ((p & 0x0000000182442810u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square F1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f1(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = ((p & 0x0000010204885020u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square G1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g1(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = (((p & 0x0001020408102040u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(6, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H1.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h1(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = (p & 0xff) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(0, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(7, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square A2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a2(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = (((p & 0x4020100804020100u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(0, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b2(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(1, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c2(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = ((p & 0x00804020110a0400u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square D2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d2(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = ((p & 0x0000804122140800u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square E2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e2(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = ((p & 0x0000018244281000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square F2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f2(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = ((p & 0x0001020488502000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square G2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g2(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(6, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H2.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h2(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = ((p >> 8) & 0xff) as usize;
    let idx2 = (((p & 0x0204081020408000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(1, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(7, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square A3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a3(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = ((((p & 0x2010080402010204u64) + 0x6070787c7e7f7e7cu64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b3(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = ((((p & 0x4020100804020408u64) + 0x406070787c7e7c78u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c3(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = (((p & 0x0000000102040810u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped += lookup(2, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square D3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d3(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = (((p & 0x0000010204081020u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x0080402010080402u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped += lookup(3, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square E3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e3(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = (((p & 0x0001020408102040u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x0000804020100804u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped += lookup(4, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square F3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f3(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x0000008040201008u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped += lookup(5, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square G3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g3(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = ((((p & 0x0204081020402010u64) + 0x7e7c787060406070u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H3.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h3(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = ((p >> 16) & 0xff) as usize;
    let idx2 = ((((p & 0x0408102040804020u64) + 0x7c78706040004060u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(2, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square A4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a4(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = ((((p & 0x1008040201020408u64) + 0x70787c7e7f7e7c78u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b4(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = ((((p & 0x2010080402040810u64) + 0x6070787c7e7c7870u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c4(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = (((p & 0x0000010204081020u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x4020100804020100u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped += lookup(2, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square D4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d4(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = (((p & 0x0001020408102040u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped += lookup(3, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square E4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e4(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x0080402010080402u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped += lookup(4, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square F4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f4(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = (((p & 0x0204081020408000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x0000804020100804u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped += lookup(5, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square G4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g4(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = ((((p & 0x0408102040201008u64) + 0x7c78706040607078u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H4.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h4(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = ((p >> 24) & 0xff) as usize;
    let idx2 = ((((p & 0x0810204080402010u64) + 0x7870604000406070u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(3, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square A5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a5(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = ((((p & 0x0804020102040810u64) + 0x787c7e7f7e7c7870u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b5(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = ((((p & 0x1008040204081020u64) + 0x70787c7e7c787060u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c5(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = (((p & 0x0001020408102040u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x2010080402010000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped += lookup(2, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square D5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d5(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x4020100804020100u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped += lookup(3, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square E5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e5(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = (((p & 0x0204081020408000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped += lookup(4, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square F5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f5(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = (((p & 0x0408102040800000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x0080402010080402u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped += lookup(5, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square G5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g5(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = ((((p & 0x0810204020100804u64) + 0x787060406070787cu64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H5.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h5(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = ((p >> 32) & 0xff) as usize;
    let idx2 = ((((p & 0x1020408040201008u64) + 0x7060400040607078u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(4, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square A6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a6(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = ((((p & 0x0402010204081020u64) + 0x7c7e7f7e7c787060u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b6(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = ((((p & 0x0804020408102040u64) + 0x787c7e7c78706040u64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c6(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x1008040201000000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped += lookup(2, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square D6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d6(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = (((p & 0x0204081020408000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x2010080402010000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped += lookup(3, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square E6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e6(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = (((p & 0x0408102040800000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x4020100804020100u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped += lookup(4, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square F6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f6(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = (((p & 0x0810204080000000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let idx3 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped += lookup(5, idx3);
    n_flipped
}

/// Counts last flipped discs when playing on square G6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g6(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = ((((p & 0x1020402010080402u64) + 0x7060406070787c7eu64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H6.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h6(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = ((p >> 40) & 0xff) as usize;
    let idx2 = ((((p & 0x2040804020100804u64) + 0x604000406070787cu64) & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64) >> 56) as usize;
    let mut n_flipped = lookup(5, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square A7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a7(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = (((p & 0x0001020408102040u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(0, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b7(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(1, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c7(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = ((p & 0x00040a1120408000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square D7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d7(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = ((p & 0x0008142241800000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square E7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e7(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = ((p & 0x0010284482010000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square F7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f7(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = ((p & 0x0020508804020100u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square G7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g7(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(6, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H7.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h7(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = ((p >> 48) & 0xff) as usize;
    let idx2 = (((p & 0x0080402010080402u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(6, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(7, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square A8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_a8(p: u64) -> i32 {
    let idx0 = (((p & 0x0101010101010101u64).wrapping_mul(0x0102040810204080u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = (((p & 0x0102040810204080u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(0, idx1);
    n_flipped += lookup(0, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square B8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_b8(p: u64) -> i32 {
    let idx0 = (((p & 0x0202020202020202u64).wrapping_mul(0x0081020408102040u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = (((p & 0x0204081020408000u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(1, idx1);
    n_flipped += lookup(1, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square C8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_c8(p: u64) -> i32 {
    let idx0 = (((p & 0x0404040404040404u64).wrapping_mul(0x0040810204081020u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = ((p & 0x040a112040800000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(2, idx1);
    n_flipped += lookup(2, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square D8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_d8(p: u64) -> i32 {
    let idx0 = (((p & 0x0808080808080808u64).wrapping_mul(0x0020408102040810u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = ((p & 0x0814224180000000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(3, idx1);
    n_flipped += lookup(3, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square E8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_e8(p: u64) -> i32 {
    let idx0 = (((p & 0x1010101010101010u64).wrapping_mul(0x0010204081020408u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = ((p & 0x1028448201000000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(4, idx1);
    n_flipped += lookup(4, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square F8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_f8(p: u64) -> i32 {
    let idx0 = (((p & 0x2020202020202020u64).wrapping_mul(0x0008102040810204u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = ((p & 0x2050880402010000u64).wrapping_mul(0x0101010101010101u64) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(5, idx1);
    n_flipped += lookup(5, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square G8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_g8(p: u64) -> i32 {
    let idx0 = (((p & 0x4040404040404040u64).wrapping_mul(0x0004081020408102u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = (((p & 0x4020100804020100u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(6, idx1);
    n_flipped += lookup(6, idx2);
    n_flipped
}

/// Counts last flipped discs when playing on square H8.
///
/// # Arguments
///
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[rustfmt::skip]
pub fn count_last_flip_h8(p: u64) -> i32 {
    let idx0 = (((p & 0x8080808080808080u64).wrapping_mul(0x0002040810204081u64)) >> 56) as usize;
    let idx1 = (p >> 56) as usize;
    let idx2 = (((p & 0x8040201008040201u64).wrapping_mul(0x0101010101010101u64)) >> 56) as usize;
    let mut n_flipped = lookup(7, idx0);
    n_flipped += lookup(7, idx1);
    n_flipped += lookup(7, idx2);
    n_flipped
}

pub type CountLastFlipFn = fn(u64) -> i32;

/** Array of functions to count flipped discs of the last move */
#[rustfmt::skip]
pub static LAST_FLIP: [CountLastFlipFn; 64] = [
    count_last_flip_a1, count_last_flip_b1, count_last_flip_c1, count_last_flip_d1, count_last_flip_e1, count_last_flip_f1, count_last_flip_g1, count_last_flip_h1,
    count_last_flip_a2, count_last_flip_b2, count_last_flip_c2, count_last_flip_d2, count_last_flip_e2, count_last_flip_f2, count_last_flip_g2, count_last_flip_h2,
    count_last_flip_a3, count_last_flip_b3, count_last_flip_c3, count_last_flip_d3, count_last_flip_e3, count_last_flip_f3, count_last_flip_g3, count_last_flip_h3,
    count_last_flip_a4, count_last_flip_b4, count_last_flip_c4, count_last_flip_d4, count_last_flip_e4, count_last_flip_f4, count_last_flip_g4, count_last_flip_h4,
    count_last_flip_a5, count_last_flip_b5, count_last_flip_c5, count_last_flip_d5, count_last_flip_e5, count_last_flip_f5, count_last_flip_g5, count_last_flip_h5,
    count_last_flip_a6, count_last_flip_b6, count_last_flip_c6, count_last_flip_d6, count_last_flip_e6, count_last_flip_f6, count_last_flip_g6, count_last_flip_h6,
    count_last_flip_a7, count_last_flip_b7, count_last_flip_c7, count_last_flip_d7, count_last_flip_e7, count_last_flip_f7, count_last_flip_g7, count_last_flip_h7,
    count_last_flip_a8, count_last_flip_b8, count_last_flip_c8, count_last_flip_d8, count_last_flip_e8, count_last_flip_f8, count_last_flip_g8, count_last_flip_h8,
];

/// Counts last flipped discs.
///
/// # Arguments
///
/// * `sq` - The square where the last move was played.
/// * `p` - Player's disc pattern.
///
/// # Returns
///
/// Flipped disc count.
#[inline(always)]
pub fn count_last_flip(p: u64, sq: Square) -> i32 {
    (unsafe { LAST_FLIP.get_unchecked(sq.index()) })(p)
}
