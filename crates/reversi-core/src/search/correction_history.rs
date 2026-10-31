//! Corrections to the main network's evaluation, learned during midgame search.

use std::sync::Mutex;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::transposition_table::Bound;
use crate::types::{Depth, ScaledScore};

/// Index bits of [`CorrectionHistory`].
const CORRECTION_BITS: u32 = 14;

/// Squares whose discs key [`CorrectionHistory`]: the outer ring of the board
/// and the X-squares.
///
/// ```text
///   A B C D E F G H
/// 1 # # # # # # # #
/// 2 # # . . . . # #
/// 3 # . . . . . . #
/// 4 # . . . . . . #
/// 5 # . . . . . . #
/// 6 # . . . . . . #
/// 7 # # . . . . # #
/// 8 # # # # # # # #
/// ```
const CORRECTION_KEY_MASK: Bitboard = Bitboard::new(0xFFC3_8181_8181_C3FF);

/// Largest correction, and largest gap between a search score and the
/// corrected evaluation that one update takes in.
const CORRECTION_LIMIT: ScaledScore = ScaledScore::from_disc_diff(8);

/// Search depth from which an update takes the full step.
const CORRECTION_FULL_STEP_DEPTH: Depth = 16;

/// Divisor of the update step; a full step moves an entry by
/// `CORRECTION_FULL_STEP_DEPTH / CORRECTION_STEP_DIV` of the gap.
const CORRECTION_STEP_DIV: i32 = 64;

/// Corrections to the main network's evaluation, learned by one search thread
/// from how far search scores land from it.
pub(in crate::search) struct CorrectionHistory([i16; 1 << CORRECTION_BITS]);

/// The [`CorrectionHistory`] of every search thread, kept across the searches
/// of one game.
#[derive(Default)]
pub(crate) struct CorrectionHistories(Mutex<Vec<Option<Box<CorrectionHistory>>>>);

impl CorrectionHistories {
    /// Takes thread `idx`'s history for a search, or a new one if it has none.
    pub(in crate::search) fn take(&self, idx: usize) -> Box<CorrectionHistory> {
        self.0
            .lock()
            .unwrap()
            .get_mut(idx)
            .and_then(Option::take)
            .unwrap_or_else(|| Box::new(CorrectionHistory([0; 1 << CORRECTION_BITS])))
    }

    /// Keeps thread `idx`'s history for its next search.
    pub(in crate::search) fn put(&self, idx: usize, history: Box<CorrectionHistory>) {
        let mut slots = self.0.lock().unwrap();
        if slots.len() <= idx {
            slots.resize_with(idx + 1, || None);
        }
        slots[idx] = Some(history);
    }

    /// Drops every history.
    pub(crate) fn clear(&self) {
        self.0.lock().unwrap().clear();
    }
}

impl CorrectionHistory {
    #[inline(always)]
    fn index(board: &Board) -> usize {
        let player = (board.player() & CORRECTION_KEY_MASK).bits();
        let opponent = (board.opponent() & CORRECTION_KEY_MASK).bits();
        let product = u128::from(player ^ 0x9E37_79B9_7F4A_7C15)
            * u128::from(opponent ^ 0xC2B2_AE3D_27D4_EB4F);
        let hash = product as u64 ^ (product >> 64) as u64;
        (hash >> (u64::BITS - CORRECTION_BITS)) as usize
    }

    /// Returns `eval` corrected for positions sharing `board`'s key squares.
    #[inline(always)]
    pub(in crate::search) fn correct(&self, board: &Board, eval: ScaledScore) -> ScaledScore {
        (eval + i32::from(self.0[Self::index(board)]))
            .clamp(ScaledScore::MIN + 1, ScaledScore::MAX - 1)
    }

    /// Moves the correction for `board`'s key squares by part of the gap between a
    /// search `score` and the corrected static `eval`.
    ///
    /// A bound counts only when it shows the gap's sign.
    pub(in crate::search) fn update(
        &mut self,
        board: &Board,
        eval: ScaledScore,
        score: ScaledScore,
        bound: Bound,
        depth: Depth,
    ) {
        let shows_gap = match bound {
            Bound::Lower => score > eval,
            Bound::Upper => score < eval,
            Bound::Exact => true,
            Bound::None => false,
        };
        if !shows_gap {
            return;
        }

        let limit = CORRECTION_LIMIT.value();
        let gap = (score - eval).value().clamp(-limit, limit);
        let step = depth.min(CORRECTION_FULL_STEP_DEPTH) as i32;
        let entry = &mut self.0[Self::index(board)];
        *entry = (i32::from(*entry) + gap * step / CORRECTION_STEP_DIV).clamp(-limit, limit) as i16;
    }
}
