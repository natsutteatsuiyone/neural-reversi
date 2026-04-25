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

/// Counts flipped discs for both players simultaneously.
///
/// Returns `(player_flipped, opponent_flipped)` where values are twice the actual flip count.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub fn count_last_flip_double(player: Bitboard, opponent: Bitboard, sq: Square) -> (i32, i32) {
    unsafe { count_last_flip_bmi2::count_last_flip_double(player.bits(), opponent.bits(), sq) }
}
