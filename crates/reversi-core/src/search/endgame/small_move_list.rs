//! Move list for [`null_window_search`](super::null_window_search).

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_initializes_every_legal_child_at_capacity_and_with_an_odd_tail() {
        let opponent = 0x0000_2222_2222_2222_u64;
        for empties in [0x0000_1111_1111_1111_u64, 0x0000_1111_1111_1110] {
            let board = Board::from_bitboards(!(empties | opponent), opponent);
            let moves = board.legal_moves();
            assert_eq!(moves.bits(), empties);

            let mut list = SmallMoveList::new();
            assert!(list.generate(&board, moves));
            assert!(list.iter().map(|mv| mv.sq).eq(moves.iter()));
            for mv in list.iter() {
                assert_eq!(mv.next, board.make_move(mv.sq));
                assert_eq!(mv.next_moves, LegalMoves::EMPTY);
                assert_eq!(mv.value, 0);
            }
        }
    }

    #[test]
    fn generate_stops_before_writing_the_wipeout_batch() {
        let empties = 0x7_0002_u64;
        let opponent = 0x600_u64;
        let board = Board::from_bitboards(!(empties | opponent), opponent);
        let moves = board.legal_moves();
        assert_eq!(moves.bits(), empties);
        let mut list = SmallMoveList::new();
        assert!(!list.generate(&board, moves));

        let paired = cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512cd",
            target_feature = "avx512vl"
        ));
        let written = if paired { 2 } else { 3 };
        assert!(list.iter().map(|mv| mv.sq).eq(moves.iter().take(written)));
    }

    #[test]
    fn best_first_iter_yields_moves_in_descending_value_order_with_tie_breaking() {
        let board = Board::new();
        let squares: Vec<_> = board.legal_moves().iter().collect();
        for (values, order) in [
            ([10, 30, 30, 20], [1, 2, 3, 0]),
            ([i32::MIN, i32::MAX, i32::MIN, i32::MAX], [1, 3, 2, 0]),
        ] {
            let mut list = SmallMoveList::new();
            for (&sq, value) in squares.iter().zip(values) {
                let next = board.make_move(sq);
                list.push(SmallMove {
                    sq,
                    next,
                    next_moves: next.legal_moves(),
                    value,
                });
            }

            let mut moves = list.best_first_iter();
            for i in order {
                let mv = moves.next().unwrap();
                let next = board.make_move(squares[i]);
                assert_eq!((mv.sq, mv.value), (squares[i], values[i]));
                assert_eq!(mv.next, next);
                assert_eq!(mv.next_moves, next.legal_moves());
            }
            assert!(moves.next().is_none());
        }
    }

    #[test]
    fn evaluates_and_orders_a_twelve_empty_position() {
        use crate::eval::Eval;
        use crate::probcut::Selectivity;
        use crate::search::root_move::RootMoves;
        use crate::transposition_table::TranspositionTable;

        let board = Board::from_bitboards(0x4022_0607_3700_0080, 0x14d9_b8f8_c87e_fc7f);
        let eval = Eval::shared_for_tests();
        let tt = TranspositionTable::new(0);
        let roots = RootMoves::new(&board);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
        let mut list = SmallMoveList::new();
        assert!(list.generate(&board, board.legal_moves()));
        list.evaluate_moves(&mut ctx);

        let records: Vec<_> = list
            .best_first_iter()
            .map(|mv| {
                assert_eq!(mv.next, board.make_move(mv.sq));
                (mv.sq.index(), mv.value, mv.next_moves.bits())
            })
            .collect();
        assert_eq!(
            records,
            [
                (50, 477_952, 0x2b00_4100_0000_0000),
                (8, 477_696, 0x2104_4100_0001_0000),
                (23, 464_896, 0x2304_4100_0001_0000),
                (61, 446_464, 0x8b04_4100_0000_0000),
                (46, 445_184, 0xa304_0100_0001_0000),
                (9, 444_672, 0x2304_4100_0001_0100),
            ]
        );
    }
}
