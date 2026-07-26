//! AArch64-specialized last-empty-square scoring for the shared tables.

use crate::constants::SCORE_MAX;
use crate::square::Square;
use crate::types::Score;

use super::solve1_tables::{COUNT_PAIR, META_HAS_D9, MULT_D9, NEON_PARAMS};

#[inline(always)]
pub(super) fn solve1(player: u64, alpha: Score, sq: Square) -> Score {
    let pp = unsafe { NEON_PARAMS.get_unchecked(sq.index()) };
    let meta = pp.meta;
    let c0 = ((meta >> 8) & 0xff) as usize;
    let c1 = ((meta >> 16) & 0xff) as usize;
    let c2 = ((meta >> 24) & 0xff) as usize;

    let idx0 = ((player & pp.mask_v).wrapping_mul(pp.mult_v) >> 56) as usize;
    let idx1 = ((player >> (meta as u32 & 0xff)) & 0xff) as usize;
    // Non-addend squares have addend7 = 0 and post_mask7 = !0.
    let idx2 = (((player & pp.mask_d7).wrapping_add(pp.addend7) & pp.post_mask7)
        .wrapping_mul(pp.mult_d7)
        >> 56) as usize;

    // Each entry holds the mover's line count in its low byte and the
    // opponent's in its high byte, so both sides are summed in one pass.
    let pair = &COUNT_PAIR.0;
    let mut packed = unsafe {
        *pair.get_unchecked(c0).get_unchecked(idx0) as u32
            + *pair.get_unchecked(c1).get_unchecked(idx1) as u32
            + *pair.get_unchecked(c2).get_unchecked(idx2) as u32
    };
    if meta & META_HAS_D9 != 0 {
        let c3 = ((meta >> 32) & 0xff) as usize;
        let idx3 = ((player & pp.mask_d9).wrapping_mul(MULT_D9) >> 56) as usize;
        packed += unsafe { *pair.get_unchecked(c3).get_unchecked(idx3) as u32 };
    }
    let nf = (packed & 0xff) as Score;
    let nf2 = (packed >> 8) as Score;

    let score_base = 2 * player.count_ones() as Score - SCORE_MAX + 2;
    let sip = if score_base > 0 {
        score_base
    } else {
        score_base - 2
    };

    // Use `&` so the second predicate stays non-short-circuiting.
    let zero_path = if (sip > alpha) & (nf2 > 0) {
        score_base - 2 - nf2
    } else {
        sip
    };

    if nf != 0 { score_base + nf } else { zero_path }
}
