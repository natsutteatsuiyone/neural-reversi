//! Portable scalar variant of flip function.
//!
//! This uses the same per-square left/right direction masks as the SIMD
//! backends, but computes each lane with scalar bit operations.

use crate::square::Square;

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
#[inline(always)]
pub fn flip(sq: Square, p: u64, o: u64) -> u64 {
    let masks = unsafe { &super::lrmask::LRMASK.get_unchecked(sq.index()).0 };
    let not_o = !o;

    flip_left(masks[0], p, not_o)
        | flip_left(masks[1], p, not_o)
        | flip_left(masks[2], p, not_o)
        | flip_left(masks[3], p, not_o)
        | flip_right(masks[4], p, not_o)
        | flip_right(masks[5], p, not_o)
        | flip_right(masks[6], p, not_o)
        | flip_right(masks[7], p, not_o)
}

/// LEFT side masks: E, S, SE, SW. The closest square is the least significant
/// bit in each mask.
#[inline(always)]
fn flip_left(mask: u64, p: u64, not_o: u64) -> u64 {
    let non_opponent = not_o & mask;
    let flank = non_opponent & non_opponent.wrapping_neg();
    if (flank & p) != 0 {
        mask & flank.wrapping_sub(1)
    } else {
        0
    }
}

/// RIGHT side masks: W, N, NW, NE. The closest square is the most significant
/// bit in each mask.
#[inline(always)]
fn flip_right(mask: u64, p: u64, not_o: u64) -> u64 {
    let non_opponent = not_o & mask;
    if non_opponent == 0 {
        return 0;
    }

    let flank = 1u64 << (u64::BITS - 1 - non_opponent.leading_zeros());
    if (flank & p) != 0 {
        mask & !(flank | flank.wrapping_sub(1))
    } else {
        0
    }
}
