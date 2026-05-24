//! Last move flip counting optimized for endgame.

use crate::bitboard::Bitboard;
use crate::square::Square;
use crate::types::Score;

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
mod count_last_flip_bmi2;
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
use count_last_flip_bmi2 as imp;

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
mod count_last_flip_portable;
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
use count_last_flip_portable as imp;

/// Counts the number of discs that would be flipped by the last move.
///
/// Returns twice the actual flip count for optimization purposes.
#[inline(always)]
pub fn count_last_flip(player: Bitboard, sq: Square) -> i32 {
    imp::count_last_flip(player.bits(), sq)
}

/// Scores a position with exactly one empty square.
#[doc(hidden)]
#[inline(always)]
pub fn solve1(player: Bitboard, alpha: Score, sq: Square) -> Score {
    imp::solve1(player.bits(), alpha, sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{SCORE_MAX, SCORE_MIN};
    use crate::flip::flip;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// `count_last_flip(p, sq)` is defined as twice the number of opponent discs
    /// flipped if the player places a disc at `sq` on a "last move" board (one
    /// empty square left, opponent fills the rest). Cross-check it against the
    /// independently-verified `flip()` function: for any `p` and empty `sq`,
    /// `count_last_flip(p, sq) == 2 * popcount(flip(sq, p, !p ^ (1<<sq)))`.
    #[test]
    fn matches_flip_popcount_on_random_positions() {
        let mut rng = StdRng::seed_from_u64(0xfeed_face);
        for _ in 0..2048 {
            let p: u64 = rng.random();
            for sq_idx in 0..64u8 {
                let sq_bit = 1u64 << sq_idx;
                if p & sq_bit != 0 {
                    continue;
                }
                let sq = Square::from_u8(sq_idx).unwrap();
                let opponent = !p & !sq_bit;
                let flipped = flip(sq, Bitboard::new(p), Bitboard::new(opponent));
                let expected = 2 * flipped.bits().count_ones() as i32;
                let got = count_last_flip(Bitboard::new(p), sq);
                assert_eq!(
                    got, expected,
                    "mismatch at sq={:?} p={:#x}: got={} expected={}",
                    sq, p, got, expected,
                );
            }
        }
    }

    #[test]
    fn solve1_matches_direct_counts() {
        let mut rng = StdRng::seed_from_u64(0x51d3_f00d);
        for _ in 0..2048 {
            let p: u64 = rng.random();
            for sq_idx in 0..64u8 {
                let sq_bit = 1u64 << sq_idx;
                if p & sq_bit != 0 {
                    continue;
                }

                let sq = Square::from_u8(sq_idx).unwrap();
                let player = Bitboard::new(p);
                let alpha = rng.random_range(SCORE_MIN..=SCORE_MAX);
                let mut n_flipped = count_last_flip(player, sq);
                let mut expected = 2 * player.count() as Score - SCORE_MAX + 2 + n_flipped;

                if n_flipped == 0 {
                    let score_if_opp_passes = if expected > 0 { expected } else { expected - 2 };
                    if score_if_opp_passes > alpha {
                        n_flipped = count_last_flip(!player, sq);
                        expected = if n_flipped > 0 {
                            expected - 2 - n_flipped
                        } else {
                            score_if_opp_passes
                        };
                    } else {
                        expected = score_if_opp_passes;
                    }
                }

                assert_eq!(solve1(player, alpha, sq), expected);
            }
        }
    }
}
