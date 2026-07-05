//! Portable scalar last-empty-square scoring.

use crate::constants::SCORE_MAX;
use crate::square::Square;
use crate::types::Score;

use super::solve1_tables::{COUNT_FLIP, MULT_D9, PARAMS, SqParams};

#[inline(always)]
pub(super) fn solve1(player: u64, alpha: Score, sq: Square) -> Score {
    let pp = unsafe { PARAMS.get_unchecked(sq.index()) };
    let mut n_flipped = count_one(player, pp);
    let mut score = 2 * player.count_ones() as Score - SCORE_MAX + 2 + n_flipped;

    if n_flipped == 0 {
        let score_if_opp_passes = if score > 0 { score } else { score - 2 };
        if score_if_opp_passes > alpha {
            n_flipped = count_one(!player, pp);
            score = if n_flipped > 0 {
                score - 2 - n_flipped
            } else {
                score_if_opp_passes
            };
        } else {
            score = score_if_opp_passes;
        }
    }

    score
}

/// Sums the four COUNT_FLIP lookups for one bitboard, given pre-loaded params.
#[inline(always)]
fn count_one(p: u64, pp: &SqParams) -> i32 {
    let idx0 = ((p & pp.mask_v).wrapping_mul(pp.mult_v) >> 56) as usize;
    let idx1 = ((p >> pp.row_shift) & 0xff) as usize;
    // Non-addend squares have addend7 = 0 and post_mask7 = !0, so this is the
    // simple diagonal extractor for them.
    let idx2 = (((p & pp.mask_d7).wrapping_add(pp.addend7) & pp.post_mask7)
        .wrapping_mul(pp.mult_d7)
        >> 56) as usize;

    let count_flip = &COUNT_FLIP.0;
    let count = unsafe {
        *count_flip.get_unchecked(pp.t0 as usize).get_unchecked(idx0) as i32
            + *count_flip.get_unchecked(pp.t1 as usize).get_unchecked(idx1) as i32
            + *count_flip.get_unchecked(pp.t2 as usize).get_unchecked(idx2) as i32
    };

    if pp.mask_d9 == 0 {
        count
    } else {
        let idx3 = ((p & pp.mask_d9).wrapping_mul(MULT_D9) >> 56) as usize;
        count + unsafe { *count_flip.get_unchecked(pp.t3 as usize).get_unchecked(idx3) as i32 }
    }
}
