mod count_last_flip_bmi2;
mod count_last_flip_kindergarten;

use crate::square::Square;

/// Counts the number of pieces that would be flipped by the last move.
///
/// # Arguments
///
/// * `player` - Bitboard representing the player's pieces
/// * `sq` - The square where the move would be played
///
/// # Returns
///
/// Returns twice the actual number of flipped pieces for optimization purposes.
#[inline(always)]
pub fn count_last_flip(player: u64, sq: Square) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if cfg!(target_feature = "bmi2") {
            return count_last_flip_bmi2::count_last_flip(player, sq);
        }
    }

    count_last_flip_kindergarten::count_last_flip(sq, player)
}
