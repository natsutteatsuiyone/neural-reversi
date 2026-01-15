//! Last move flip counting optimized for endgame.

use cfg_if::cfg_if;

use crate::bitboard::Bitboard;
use crate::square::Square;

cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))] {
        mod count_last_flip_bmi2;
    } else {
        mod count_last_flip_kindergarten;
    }
}

/// Counts the number of discs that would be flipped by the last move.
///
/// # Arguments
///
/// * `player` - Current player's bitboard.
/// * `sq` - Square where the last move is played.
///
/// # Returns
///
/// Returns twice the actual number of flipped discs for optimization purposes.
#[inline(always)]
pub fn count_last_flip(player: Bitboard, sq: Square) -> i32 {
    cfg_if! {
        if #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))] {
            unsafe { count_last_flip_bmi2::count_last_flip(player.0, sq) }
        } else {
            count_last_flip_kindergarten::count_last_flip(player.0, sq)
        }
    }
}

/// Counts flipped discs for both players simultaneously.
///
/// # Arguments
///
/// * `player` - Current player's bitboard.
/// * `opponent` - Opponent's bitboard.
/// * `sq` - Square where the last move is played.
///
/// # Returns
///
/// A tuple of `(player_flipped, opponent_flipped)` where values are twice the actual flip count.
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub fn count_last_flip_double(player: Bitboard, opponent: Bitboard, sq: Square) -> (i32, i32) {
    unsafe { count_last_flip_bmi2::count_last_flip_double(player.0, opponent.0, sq) }
}
