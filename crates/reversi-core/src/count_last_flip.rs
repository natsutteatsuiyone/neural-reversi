cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))] {
        mod count_last_flip_bmi2;
    } else {
        mod count_last_flip_kindergarten;
    }
}

use crate::square::Square;
use cfg_if::cfg_if;

/// Counts the number of pieces that would be flipped by the last move.
///
/// # Arguments
///
/// * `player` - Current player's bitboard
/// * `sq` - Square where the last move is played
///
/// # Returns
///
/// Returns twice the actual number of flipped pieces for optimization purposes.
#[inline(always)]
pub fn count_last_flip(player: u64, sq: Square) -> i32 {
    cfg_if! {
        if #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))] {
            unsafe { count_last_flip_bmi2::count_last_flip(player, sq) }
        } else {
            count_last_flip_kindergarten::count_last_flip(player, sq)
        }
    }
}

/// Counts flipped pieces for both players simultaneously.
///
/// # Arguments
///
/// * `player` - Current player's bitboard
/// * `opponent` - Opponent's bitboard
/// * `sq` - Square where the last move is played
///
/// # Returns
///
/// Tuple of (player_flipped, opponent_flipped) where values are 2x actual flip count
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
pub fn count_last_flip_double(player: u64, opponent: u64, sq: Square) -> (i32, i32) {
    unsafe { count_last_flip_bmi2::count_last_flip_double(player, opponent, sq) }
}
