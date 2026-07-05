//! AArch64-specialized last-empty-square scoring for the shared tables.

use crate::constants::SCORE_MAX;
use crate::square::Square;
use crate::types::Score;

use super::solve1_tables::{COUNT_FLIP, MULT_D9, PARAMS};

#[inline(always)]
pub(super) fn solve1(player: u64, alpha: Score, sq: Square) -> Score {
    let pp = unsafe { PARAMS.get_unchecked(sq.index()) };

    let idx0 = ((player & pp.mask_v).wrapping_mul(pp.mult_v) >> 56) as usize;
    let idx1 = ((player >> pp.row_shift) & 0xff) as usize;
    // Non-addend squares have addend7 = 0 and post_mask7 = !0.
    let idx2 = (((player & pp.mask_d7).wrapping_add(pp.addend7) & pp.post_mask7)
        .wrapping_mul(pp.mult_d7)
        >> 56) as usize;
    let count_flip = &COUNT_FLIP.0;
    let mut nf = unsafe {
        *count_flip.get_unchecked(pp.t0 as usize).get_unchecked(idx0) as i32
            + *count_flip.get_unchecked(pp.t1 as usize).get_unchecked(idx1) as i32
            + *count_flip.get_unchecked(pp.t2 as usize).get_unchecked(idx2) as i32
    };
    if pp.mask_d9 != 0 {
        let idx3 = ((player & pp.mask_d9).wrapping_mul(MULT_D9) >> 56) as usize;
        nf += unsafe { *count_flip.get_unchecked(pp.t3 as usize).get_unchecked(idx3) as i32 };
    }

    let score_base = 2 * player.count_ones() as Score - SCORE_MAX + 2;
    let sip = if score_base > 0 {
        score_base
    } else {
        score_base - 2
    };
    let o0 = idx0 ^ 0xff;
    let o1 = idx1 ^ 0xff;
    let od7 = (((!player & pp.mask_d7).wrapping_add(pp.addend7) & pp.post_mask7)
        .wrapping_mul(pp.mult_d7)
        >> 56) as usize;
    let mut nf2 = unsafe {
        *count_flip.get_unchecked(pp.t0 as usize).get_unchecked(o0) as i32
            + *count_flip.get_unchecked(pp.t1 as usize).get_unchecked(o1) as i32
            + *count_flip.get_unchecked(pp.t2 as usize).get_unchecked(od7) as i32
    };
    if pp.mask_d9 != 0 {
        let od9 = ((!player & pp.mask_d9).wrapping_mul(MULT_D9) >> 56) as usize;
        nf2 += unsafe { *count_flip.get_unchecked(pp.t3 as usize).get_unchecked(od9) as i32 };
    }

    // Use `&` so the second predicate stays non-short-circuiting.
    let zero_path = if (sip > alpha) & (nf2 > 0) {
        score_base - 2 - nf2
    } else {
        sip
    };

    if nf != 0 { score_base + nf } else { zero_path }
}
