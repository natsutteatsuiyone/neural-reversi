//! Move generation, evaluation, and ordering for Reversi positions.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/move.c>

mod iterator;
mod move_array;
mod ordering;

use std::mem::MaybeUninit;
use std::slice;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::flip;
use crate::search::search_context::SearchContext;
use crate::square::Square;

pub use iterator::{BestFirstMoveIterator, ConcurrentMoveIterator};
use move_array::MoveArray;

/// Maximum number of moves possible in a Reversi position.
const MAX_MOVES: usize = 34;

/// A single move.
///
/// Field order is chosen to keep each entry 16 bytes in the inline move buffer.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Move {
    /// Bitboard representing all opponent discs flipped by this move.
    pub flipped: Bitboard,
    /// Evaluation score for move ordering (higher = better).
    pub value: i32,
    /// The square where the disc is placed.
    pub sq: Square,
}

impl Move {
    /// Creates a new move with the specified square and flipped discs.
    #[inline]
    pub fn new(sq: Square, flipped: Bitboard) -> Move {
        debug_assert!(sq != Square::None, "Move cannot have Square::None");
        debug_assert!(!flipped.is_empty(), "Move must flip at least one disc");

        Move {
            flipped,
            value: 0,
            sq,
        }
    }
}

const _: () = assert!(std::mem::size_of::<Move>() == 16);

/// Container for all legal moves in a position with evaluation and ordering capabilities.
#[derive(Clone, Debug)]
pub struct MoveList {
    moves: MoveArray,
    wipeout_move: Option<Square>,
}

impl MoveList {
    /// Generates all legal moves for the current player.
    #[inline]
    pub fn new(board: &Board) -> MoveList {
        Self::with_moves(board, board.get_moves())
    }

    /// Creates a [`MoveList`] from a precomputed legal moves [`Bitboard`].
    #[inline(always)]
    pub fn with_moves(board: &Board, moves_bb: Bitboard) -> MoveList {
        let mut result: MaybeUninit<MoveList> = MaybeUninit::uninit();
        // SAFETY: `fill_in_place` writes `len`, `wipeout_move`, and the
        // first `len` slots of `data` on every path.
        unsafe {
            Self::fill_in_place::<false, false>(board, moves_bb, result.as_mut_ptr());
            result.assume_init()
        }
    }

    /// Creates a [`MoveList`] when the caller has already ruled out the
    /// no-move (pass) case but the position may still have only one legal
    /// move. Skips the `moves_bb == 0` dispatch but still dispatches a
    /// scalar single-square path when there is exactly one bit set.
    #[inline(always)]
    pub(crate) fn with_at_least_one_move(board: &Board, moves_bb: Bitboard) -> MoveList {
        let mut result: MaybeUninit<MoveList> = MaybeUninit::uninit();
        // SAFETY: caller guarantees `moves_bb != 0`.
        unsafe {
            Self::fill_in_place::<true, false>(board, moves_bb, result.as_mut_ptr());
            result.assume_init()
        }
    }

    /// Creates a [`MoveList`] when the caller has already handled empty and
    /// single-move positions.
    #[inline(always)]
    pub(crate) fn with_at_least_two_moves(board: &Board, moves_bb: Bitboard) -> MoveList {
        let mut result: MaybeUninit<MoveList> = MaybeUninit::uninit();
        // SAFETY: caller guarantees >= 2 set bits.
        unsafe {
            Self::fill_in_place::<true, true>(board, moves_bb, result.as_mut_ptr());
            result.assume_init()
        }
    }

    /// Writes a fully-initialised [`MoveList`] into `out`.
    ///
    /// Kept out-of-line so the [`Self::with_moves`] /
    /// [`Self::with_at_least_one_move`] / [`Self::with_at_least_two_moves`]
    /// wrappers stay small enough to inline at call sites; that lets `*out`
    /// writes hit the caller's sret slot directly instead of staging the
    /// struct on this frame and memcpy'ing it back on return.
    ///
    /// The two const generics control which dispatch shortcuts are elided
    /// based on what the caller has already proven about `moves_bb`:
    ///
    /// | `SKIP_EMPTY` | `SKIP_SINGLE` | Caller precondition          |
    /// | ------------ | ------------- | ---------------------------- |
    /// | `false`      | `false`       | none (may be empty)          |
    /// | `true`       | `false`       | `moves_bb != 0`              |
    /// | `true`       | `true`        | `moves_bb.count_ones() >= 2` |
    ///
    /// `SKIP_EMPTY = false` with `SKIP_SINGLE = true` is unused - there is
    /// no caller that proves "not single" without first proving "not empty".
    ///
    /// # Safety
    /// `out` must be a valid, properly-aligned pointer to writable memory
    /// large enough to hold a [`MoveList`]. The const generics must match
    /// what the caller has guaranteed about `moves_bb`.
    #[inline(never)]
    unsafe fn fill_in_place<const SKIP_EMPTY: bool, const SKIP_SINGLE: bool>(
        board: &Board,
        moves_bb: Bitboard,
        out: *mut MoveList,
    ) {
        let moves_array_ptr = unsafe { &raw mut (*out).moves };
        let data_ptr = unsafe { MoveArray::data_ptr_from(moves_array_ptr) };
        let len_ptr = unsafe { MoveArray::len_ptr_from(moves_array_ptr) };
        let wipeout_ptr = unsafe { &raw mut (*out).wipeout_move };

        let opponent = board.opponent;
        let mut bb = moves_bb.bits();

        if !SKIP_EMPTY && bb == 0 {
            unsafe {
                len_ptr.write(0);
                wipeout_ptr.write(None);
            }
            return;
        }

        if !SKIP_SINGLE && bb & (bb - 1) == 0 {
            let x = bb.trailing_zeros() as u8;
            // SAFETY: `x` is a bit position of a legal-move bitboard (0..=63).
            let sq = unsafe { Square::from_u8_unchecked(x) };
            let flipped = flip::flip(sq, board.player, opponent);
            unsafe {
                data_ptr.write(Move::new(sq, flipped));
                len_ptr.write(1);
                wipeout_ptr.write(if flipped == opponent { Some(sq) } else { None });
            }
            return;
        }
        debug_assert!(bb.count_ones() >= 2);

        let mut wipeout_move = None;
        let mut len = 0usize;

        cfg_select! {
            all(target_arch = "x86_64", target_feature = "avx512cd", target_feature = "avx512vl") => {
                let ctx = flip::Avx512BoardCtx::new(board.player.bits(), opponent.bits());
                let opponent_bits = opponent.bits();
                let pair_count = bb.count_ones() as usize / 2;
                for _ in 0..pair_count {
                    let x0 = bb.trailing_zeros() as u8;
                    bb &= bb - 1;

                    let x1 = bb.trailing_zeros() as u8;
                    bb &= bb - 1;

                    let (f0, f1) = ctx.flip2(x0 as usize, x1 as usize);
                    let flipped0 = Bitboard::new(f0);
                    let flipped1 = Bitboard::new(f1);
                    // SAFETY: `x0`, `x1` are bit positions from a legal-move bitboard (0..=63).
                    let sq0 = unsafe { Square::from_u8_unchecked(x0) };
                    let sq1 = unsafe { Square::from_u8_unchecked(x1) };
                    debug_assert!(len + 2 <= MAX_MOVES);
                    // SAFETY: at most MAX_MOVES (34) legal moves per Reversi position.
                    unsafe {
                        data_ptr.add(len).write(Move::new(sq0, flipped0));
                        data_ptr.add(len + 1).write(Move::new(sq1, flipped1));
                    }
                    len += 2;
                    if f0 == opponent_bits {
                        wipeout_move = Some(sq0);
                    }
                    if f1 == opponent_bits {
                        wipeout_move = Some(sq1);
                    }
                }

                if bb != 0 {
                    let x = bb.trailing_zeros() as u8;
                    let flipped_bits = ctx.flip1(x as usize);
                    let flipped = Bitboard::new(flipped_bits);
                    // SAFETY: `x` is a bit position from a legal-move bitboard (0..=63).
                    let sq = unsafe { Square::from_u8_unchecked(x) };
                    debug_assert!(len < MAX_MOVES);
                    // SAFETY: at most MAX_MOVES (34) legal moves per Reversi position.
                    unsafe { data_ptr.add(len).write(Move::new(sq, flipped)) };
                    len += 1;
                    if flipped_bits == opponent_bits {
                        wipeout_move = Some(sq);
                    }
                }
            }
            _ => {
                let player = board.player;
                while bb != 0 {
                    let x = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    // SAFETY: `x` is a bit position from a legal-move bitboard (0..=63).
                    let sq = unsafe { Square::from_u8_unchecked(x) };
                    let flipped = flip::flip(sq, player, opponent);
                    debug_assert!(len < MAX_MOVES);
                    // SAFETY: at most MAX_MOVES (34) legal moves per Reversi position.
                    unsafe { data_ptr.add(len).write(Move::new(sq, flipped)) };
                    len += 1;
                    if flipped == opponent {
                        wipeout_move = Some(sq);
                    }
                }
            }
        }

        unsafe {
            len_ptr.write(len);
            wipeout_ptr.write(wipeout_move);
        }
    }

    /// Returns the wipeout move square if one exists.
    #[inline(always)]
    pub fn wipeout_move(&self) -> Option<Square> {
        self.wipeout_move
    }

    /// Returns the number of legal moves in this position.
    #[inline]
    pub fn count(&self) -> usize {
        self.moves.len()
    }

    /// Returns the first move in the list, if any.
    #[inline]
    pub fn first(&self) -> Option<&Move> {
        if self.moves.is_empty() {
            None
        } else {
            // SAFETY: list is non-empty.
            Some(unsafe { self.moves.get_unchecked_ref(0) })
        }
    }

    /// Swaps two moves in the list by index.
    #[inline(always)]
    pub fn swap_moves(&mut self, a: usize, b: usize) {
        let len = self.moves.len();
        assert!(
            a < len && b < len,
            "move index out of bounds: len is {len}, indexes are {a} and {b}"
        );
        // SAFETY: bounds checked above.
        unsafe { self.moves.swap_unchecked(a, b) };
    }

    /// Returns the move at the given index by value.
    #[inline(always)]
    pub fn get_move(&self, index: usize) -> Move {
        debug_assert!(index < self.moves.len());
        // SAFETY: caller ensures `index < self.count()`.
        unsafe { self.moves.get_unchecked(index) }
    }

    /// Returns an iterator over all moves in the list.
    ///
    /// The moves are returned in their current order, which may be the generation
    /// order or sorted order depending on previous operations.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, Move> {
        self.moves.iter()
    }

    /// Returns a mutable iterator over all moves in the list.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, Move> {
        self.moves.iter_mut()
    }

    /// Excludes moves that were selected as best moves for earlier PV lines in Multi-PV search.
    ///
    /// In Multi-PV mode, each PV line explores a different best move at the root. This method
    /// retains only moves that appear in `root_moves` from `pv_idx` onwards, excluding moves
    /// that were already selected as the best move for earlier PV lines (indices 0..pv_idx).
    pub fn exclude_earlier_pv_moves(&mut self, ctx: &SearchContext, board: &Board) {
        if ctx.pv_idx() == 0 {
            return;
        }

        self.moves
            .retain(|mv| ctx.root_moves.contains_from_pv_idx(mv.sq));

        // Multiple wipeout moves are possible; `with_moves` caches only the
        // last, so rebuild after retain in case the cached one was excluded.
        self.wipeout_move = self
            .moves
            .iter()
            .find(|mv| mv.flipped == board.opponent)
            .map(|mv| mv.sq);
    }
}

#[cfg(test)]
mod tests;
