//! Move list for [`null_window_search_with_ec`](super::null_window_search_with_ec).

use std::iter;
use std::mem::MaybeUninit;
use std::slice;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::flip;
use crate::legal_moves::LegalMoves;
use crate::move_list::fast_move_value;
use crate::search::context::SearchContext;
use crate::square::Square;

/// A move and the position it leads to.
#[derive(Clone, Copy)]
pub(super) struct SmallMove {
    /// Position after the move.
    pub(super) next: Board,
    /// Legal moves of `next`, set by [`SmallMoveList::evaluate_moves`].
    pub(super) next_moves: LegalMoves,
    /// Ordering value, set by [`SmallMoveList::evaluate_moves`]; higher is
    /// searched first.
    value: i32,
    pub(super) sq: Square,
}

impl SmallMove {
    #[inline(always)]
    fn new(board: &Board, sq: Square, flipped: Bitboard) -> Self {
        SmallMove {
            next: board.make_move_with_flipped(flipped, sq),
            next_moves: LegalMoves::EMPTY,
            value: 0,
            sq,
        }
    }
}

/// Move list that keeps each child position and its legal moves.
///
/// The enhanced transposition cutoff and the child search reuse them instead
/// of computing them again.
pub(super) struct SmallMoveList {
    moves: [MaybeUninit<SmallMove>; Self::CAPACITY],
    len: usize,
}

impl SmallMoveList {
    /// One legal move at most per empty square.
    const CAPACITY: usize = super::DEPTH_TO_NWS as usize;

    #[inline(always)]
    pub(super) fn new() -> Self {
        SmallMoveList {
            moves: [MaybeUninit::uninit(); Self::CAPACITY],
            len: 0,
        }
    }

    /// Adds `moves`, the legal moves of `board`, to an empty list in square
    /// order.
    ///
    /// Returns `false` as soon as a move captures every opponent disc.
    ///
    /// # Panics
    ///
    /// Panics if `moves` has more than [`Self::CAPACITY`] squares.
    #[inline(always)]
    pub(super) fn generate(&mut self, board: &Board, moves: LegalMoves) -> bool {
        debug_assert_eq!(self.len, 0);
        assert!(moves.count() as usize <= Self::CAPACITY);
        let opponent = board.opponent();

        cfg_select! {
            all(
                target_arch = "x86_64",
                target_feature = "avx512cd",
                target_feature = "avx512vl"
            ) => {
                let flip_ctx = flip::Avx512BoardCtx::new(board.player().bits(), opponent.bits());
                let mut bb = moves.bits();
                while bb & bb.wrapping_sub(1) != 0 {
                    let x0 = bb.trailing_zeros();
                    bb &= bb - 1;
                    let x1 = bb.trailing_zeros();
                    bb &= bb - 1;
                    let (f0, f1) = flip_ctx.flip2_wide_load(x0 as usize, x1 as usize);
                    if f0 == opponent.bits() || f1 == opponent.bits() {
                        return false;
                    }
                    // SAFETY: `x0` and `x1` are bit positions of a `u64`.
                    let (sq0, sq1) = unsafe {
                        (
                            Square::from_u32_unchecked(x0),
                            Square::from_u32_unchecked(x1),
                        )
                    };
                    self.push(SmallMove::new(board, sq0, Bitboard::new(f0)));
                    self.push(SmallMove::new(board, sq1, Bitboard::new(f1)));
                }
                if bb != 0 {
                    let x = bb.trailing_zeros();
                    let flipped = Bitboard::new(flip_ctx.flip1(x as usize));
                    if flipped == opponent {
                        return false;
                    }
                    // SAFETY: `x` is a bit position of a non-zero `u64`.
                    let sq = unsafe { Square::from_u32_unchecked(x) };
                    self.push(SmallMove::new(board, sq, flipped));
                }
            }
            _ => {
                for sq in moves.iter() {
                    let flipped = flip::flip(sq, board.player(), opponent);
                    if flipped == opponent {
                        return false;
                    }
                    self.push(SmallMove::new(board, sq, flipped));
                }
            }
        }

        true
    }

    #[inline(always)]
    fn push(&mut self, mv: SmallMove) {
        debug_assert!(self.len < Self::CAPACITY);
        // SAFETY: `generate` pushes at most `CAPACITY` moves into an empty list.
        unsafe { self.moves.get_unchecked_mut(self.len).write(mv) };
        self.len += 1;
    }

    #[inline(always)]
    pub(super) fn iter(&self) -> slice::Iter<'_, SmallMove> {
        // SAFETY: `push` initialized `0..self.len`.
        unsafe { slice::from_raw_parts(self.moves.as_ptr().cast(), self.len) }.iter()
    }

    #[inline(always)]
    fn iter_mut(&mut self) -> slice::IterMut<'_, SmallMove> {
        // SAFETY: `push` initialized `0..self.len`.
        unsafe { slice::from_raw_parts_mut(self.moves.as_mut_ptr().cast(), self.len) }.iter_mut()
    }

    /// Computes the legal moves and the [`fast_move_value`] of every child.
    #[inline(always)]
    pub(super) fn evaluate_moves(&mut self, ctx: &mut SearchContext) {
        for mv in self.iter_mut() {
            ctx.increment_nodes();
            mv.next_moves = mv.next.legal_moves();
            mv.value = fast_move_value(mv.sq, &mv.next, mv.next_moves);
        }
    }

    /// Returns an iterator over the moves from the highest value down.
    ///
    /// Ties are broken as in [`crate::move_list::BestFirstMoveIterator`].
    #[inline(always)]
    pub(super) fn best_first_iter(&mut self) -> impl Iterator<Item = SmallMove> + '_ {
        let mut from = 0;
        iter::from_fn(move || {
            if from == self.len {
                return None;
            }
            let mv = self.take_best(from);
            from += 1;
            Some(mv)
        })
    }

    /// Removes the highest-valued move in `from..` and returns it; the move at
    /// `from` takes its slot.
    #[inline(always)]
    fn take_best(&mut self, from: usize) -> SmallMove {
        debug_assert!(from < self.len);
        // SAFETY: `best_first_iter` calls this only with `from < self.len`, so
        // every pointer stays within the initialized `0..self.len`.
        unsafe {
            let first: *mut SmallMove = self.moves.as_mut_ptr().add(from).cast();
            let end = first.add(self.len - from);
            let mut best = first;
            let mut best_value = (*first).value;
            let mut ptr = first.add(1);
            while ptr < end {
                let value = (*ptr).value;
                if value > best_value {
                    best_value = value;
                    best = ptr;
                }
                ptr = ptr.add(1);
            }
            let mv = *best;
            *best = *first;
            mv
        }
    }
}
