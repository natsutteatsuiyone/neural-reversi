//! Scalar flip backend.
//!
//! This uses the shared per-square left/right direction-mask table, but
//! computes each lane with scalar bit operations.

use crate::square::Square;

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
#[inline(always)]
pub fn flip(sq: Square, player: u64, opponent: u64) -> u64 {
    let masks = unsafe { &super::lrmask::LRMASK.get_unchecked(sq.index()).0 };
    let not_opponent = !opponent;

    flip_left(masks[0], player, not_opponent)
        | flip_left(masks[1], player, not_opponent)
        | flip_left(masks[2], player, not_opponent)
        | flip_left(masks[3], player, not_opponent)
        | flip_right(masks[4], player, not_opponent)
        | flip_right(masks[5], player, not_opponent)
        | flip_right(masks[6], player, not_opponent)
        | flip_right(masks[7], player, not_opponent)
}

/// LEFT side masks: E, S, SE, SW. The closest square is the least significant
/// bit in each mask.
#[inline(always)]
fn flip_left(mask: u64, player: u64, not_opponent: u64) -> u64 {
    let non_opponent = not_opponent & mask;
    let flank = non_opponent.isolate_lowest_one();
    if (flank & player) != 0 {
        mask & flank.wrapping_sub(1)
    } else {
        0
    }
}

/// RIGHT side masks: W, N, NW, NE. The closest square is the most significant
/// bit in each mask.
#[inline(always)]
fn flip_right(mask: u64, player: u64, not_opponent: u64) -> u64 {
    let non_opponent = not_opponent & mask;
    if non_opponent == 0 {
        return 0;
    }

    let flank = 1u64 << (u64::BITS - 1 - non_opponent.leading_zeros());
    if (flank & player) != 0 {
        mask & !(flank | flank.wrapping_sub(1))
    } else {
        0
    }
}
