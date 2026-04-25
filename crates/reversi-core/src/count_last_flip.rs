//! Last move flip counting optimized for endgame.

use crate::bitboard::Bitboard;
use crate::square::Square;

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
mod count_last_flip_bmi2;
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
mod count_last_flip_portable;

/// Counts the number of discs that would be flipped by the last move.
///
/// Returns twice the actual flip count for optimization purposes.
#[inline(always)]
pub fn count_last_flip(player: Bitboard, sq: Square) -> i32 {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "bmi2") => {
            unsafe { count_last_flip_bmi2::count_last_flip(player.bits(), sq) }
        }
        _ => {
            count_last_flip_portable::count_last_flip(player.bits(), sq)
        }
    }
}

/// Counts flipped discs for both players simultaneously (BMI2 only).
///
/// **Precondition**: callers must satisfy the "last move" invariant — `sq` is
/// empty and every other square is occupied by either player.
///
/// Returns `(player_flipped, opponent_flipped)` where values are twice the actual flip count.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub fn count_last_flip_double(player: Bitboard, sq: Square) -> (i32, i32) {
    unsafe { count_last_flip_bmi2::count_last_flip_double(player.bits(), sq) }
}

#[cfg(test)]
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// Cross-checks the BMI2 `count_last_flip_double` complementarity-XOR shortcut
    /// against two independent `count_last_flip` calls.
    #[test]
    fn double_matches_single_under_complementarity() {
        let mut rng = StdRng::seed_from_u64(0xb1da_1100);
        let mut checked = 0usize;
        for _ in 0..1024 {
            let p: u64 = rng.random();
            for sq_idx in 0..64u8 {
                let sq_bit = 1u64 << sq_idx;
                if p & sq_bit != 0 {
                    continue;
                }
                let sq = Square::from_u8(sq_idx).unwrap();
                let opp = !p & !sq_bit;
                let player_bb = Bitboard::new(p);
                let opp_bb = Bitboard::new(opp);
                let expected = (count_last_flip(player_bb, sq), count_last_flip(opp_bb, sq));
                let got = count_last_flip_double(player_bb, sq);
                assert_eq!(
                    got, expected,
                    "mismatch at sq={:?} p={:#x}: got={:?} expected={:?}",
                    sq, p, got, expected,
                );
                checked += 1;
            }
        }
        assert!(checked > 10_000, "too few checks: {checked}");
    }
}
