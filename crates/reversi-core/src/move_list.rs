//! Move generation, evaluation, and ordering for Reversi positions.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/move.c>

use arrayvec::ArrayVec;
use std::slice;
use std::sync::atomic;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::flip;
use crate::probcut;
use crate::search::midgame;
use crate::search::search_context::SearchContext;
use crate::search::search_strategy::SearchStrategy;
use crate::square::Square;
use crate::types::{Depth, ScaledScore};

/// Maximum number of moves possible in a Reversi position.
const MAX_MOVES: usize = 34;

/// Value assigned to wipeout moves (capturing all opponent discs).
const WIPEOUT_VALUE: i32 = 1 << 30;

/// Value assigned to moves suggested by the transposition table.
const TT_MOVE_VALUE: i32 = 1 << 20;

/// Represents a single move.
#[derive(Clone, Copy, Debug, Default)]
pub struct Move {
    /// The square where the disc is placed.
    pub sq: Square,
    /// Bitboard representing all opponent discs flipped by this move.
    pub flipped: Bitboard,
    /// Evaluation score for move ordering (higher = better).
    pub value: i32,
    /// Suggested depth reduction for this move in search.
    pub reduction_depth: Depth,
}

impl Move {
    /// Creates a new move with the specified square and flipped discs.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square where the disc is placed
    /// * `flipped` - Bitboard of opponent discs flipped by this move
    ///
    /// # Returns
    ///
    /// A new Move instance with default evaluation data.
    #[inline]
    pub fn new(sq: Square, flipped: Bitboard) -> Move {
        debug_assert!(sq != Square::None, "Move cannot have Square::None");
        debug_assert!(!flipped.is_empty(), "Move must flip at least one disc");

        Move {
            sq,
            flipped,
            value: 0,
            reduction_depth: 0,
        }
    }
}

/// Container for all legal moves in a position with evaluation and ordering capabilities.
#[derive(Clone, Debug)]
pub struct MoveList {
    /// List of moves.
    moves: ArrayVec<Move, MAX_MOVES>,
    /// The square of the wipeout move, if found.
    wipeout_move: Option<Square>,
}

impl MoveList {
    /// Generates all legal moves for the current player.
    ///
    /// # Arguments
    ///
    /// * `board` - The current game state
    ///
    /// # Returns
    ///
    /// A new MoveList containing all legal moves for the current player.
    #[inline]
    pub fn new(board: &Board) -> MoveList {
        Self::with_moves(board, board.get_moves())
    }

    /// Creates a MoveList from precomputed legal moves bitboard.
    ///
    /// # Arguments
    ///
    /// * `board` - Current game state.
    /// * `moves_bb` - Precomputed bitboard of legal moves (from `board.get_moves()`).
    ///
    /// # Returns
    ///
    /// A new MoveList containing all legal moves.
    #[inline]
    pub fn with_moves(board: &Board, moves_bb: Bitboard) -> MoveList {
        let mut moves = ArrayVec::new();
        let mut wipeout_move = None;
        for sq in moves_bb.iter() {
            let flipped = flip::flip(sq, board.player, board.opponent);
            let mut mv = Move::new(sq, flipped);
            mv.value = i32::MIN;

            debug_assert!(moves.len() < moves.capacity());
            unsafe { moves.push_unchecked(mv) };

            if flipped == board.opponent {
                wipeout_move = Some(sq);
            }
        }

        MoveList {
            moves,
            wipeout_move,
        }
    }

    /// Returns the wipeout move square if one exists.
    ///
    /// # Returns
    ///
    /// - `Some(sq)` if a wipeout move exists
    /// - `None` if no wipeout move exists
    #[inline(always)]
    pub fn wipeout_move(&self) -> Option<Square> {
        self.wipeout_move
    }

    /// Returns the number of legal moves in this position.
    ///
    /// # Returns
    ///
    /// The count of legal moves.
    #[inline]
    pub fn count(&self) -> usize {
        self.moves.len()
    }

    /// Returns the first move in the list, if any exists.
    ///
    /// # Returns
    ///
    /// Reference to the first move, or None if no legal moves exist.
    #[inline]
    pub fn first(&self) -> Option<&Move> {
        self.moves.first()
    }

    /// Returns an iterator over all moves in the list.
    ///
    /// The moves are returned in their current order, which may be the generation
    /// order or sorted order depending on previous operations.
    ///
    /// # Returns
    ///
    /// Iterator over moves in the list.
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, Move> {
        self.moves.iter()
    }

    /// Returns a mutable iterator over all moves in the list.
    ///
    /// # Returns
    ///
    /// Mutable iterator over moves in the list.
    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<'_, Move> {
        self.moves.iter_mut()
    }

    /// Returns an iterator that yields moves in order of decreasing value.
    ///
    /// # Returns
    ///
    /// Iterator that yields moves in best-first order.
    #[inline]
    pub fn into_best_first_iter(self) -> BestFirstMoveIterator {
        BestFirstMoveIterator::new(self.moves)
    }

    /// Evaluates all moves to assign ordering values and reduction depths.
    ///
    /// # Type Parameters
    ///
    /// * `SS` - Search strategy determining midgame vs endgame evaluation
    ///
    /// # Arguments
    ///
    /// * `ctx` - Search context with transposition table and statistics
    /// * `board` - Current position before making any move
    /// * `depth` - Remaining search depth at this node
    /// * `tt_move` - Best move from transposition table (if any)
    pub fn evaluate_moves<SS: SearchStrategy>(
        &mut self,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        tt_move: Square,
    ) {
        // Minimum depth required for shallow search evaluation based on empty squares
        // When depth is below this threshold, use fast heuristic evaluation instead
        #[rustfmt::skip]
        const MIN_DEPTH: [u32; 64] = [
            19, 18, 18, 18, 17, 17, 17, 16,  // 0-7 empty squares
            16, 16, 15, 15, 15, 14, 14, 14,  // 8-15 empty squares
            13, 13, 13, 12, 12, 12, 11, 11,  // 16-23 empty squares
            11, 10, 10, 10, 9,  9,  9,  9,   // 24-31 empty squares
            9,  9,  9,  9,  9,  9,  9,  9,   // 32-39 empty squares
            9,  9,  9,  9,  9,  9,  9,  9,   // 40-47 empty squares
            9,  9,  9,  9,  9,  9,  9,  9,   // 48-55 empty squares
            9,  9,  9,  9,  9,  9,  9,  9    // 56-63 empty squares
        ];

        if depth <= MIN_DEPTH[ctx.empty_list.count as usize] {
            self.evaluate_moves_fast(ctx, board, tt_move);
            return;
        }

        if SS::IS_ENDGAME {
            self.evaluate_moves_endgame(ctx, board, depth, tt_move);
        } else {
            self.evaluate_moves_midgame(ctx, board, depth, tt_move);
        }
    }

    /// Evaluates moves specifically for midgame positions.
    fn evaluate_moves_midgame(
        &mut self,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        tt_move: Square,
    ) {
        const MAX_SORT_DEPTH: i32 = 2;
        let mut sort_depth = (depth as i32 - 15) / 3;
        sort_depth = sort_depth.clamp(0, MAX_SORT_DEPTH);

        let mut best_sort_value = i32::MIN;

        for mv in self.iter_mut() {
            if mv.flipped == board.opponent {
                // Wipeout move
                mv.value = WIPEOUT_VALUE;
            } else if mv.sq == tt_move {
                // Transposition table move
                mv.value = TT_MOVE_VALUE;
            } else {
                // Evaluate using shallow search
                let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                ctx.update(mv.sq, mv.flipped);

                let score = match sort_depth {
                    0 => -midgame::evaluate(ctx, &next),
                    1 => -midgame::evaluate_depth1(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                    2 => -midgame::evaluate_depth2(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                    _ => unreachable!(),
                };
                mv.value = score.value();

                ctx.undo(mv.sq);
                best_sort_value = best_sort_value.max(mv.value);
            };
        }

        if best_sort_value == i32::MIN || !ctx.selectivity.is_enabled() {
            return;
        }

        // Score-Based Reduction: reduce depth for poor moves
        // This implements a form of late move reduction based on evaluation scores,
        // using the same statistical error model as ProbCut.
        let sigma = probcut::get_sigma(ctx.ply(), sort_depth as Depth, depth);
        let sbr_margin = (ctx.selectivity.t_value() * sigma).ceil() as i32;
        if sbr_margin == 0 {
            return;
        }

        // best_lower_bound = best_sort_value - sbr_margin
        // other_upper_bound = mv.value + sbr_margin
        // Condition: best_lower_bound > other_upper_bound
        //         => best_sort_value - sbr_margin > mv.value + sbr_margin
        //         => mv.value < best_sort_value - 2 * sbr_margin
        let reduction_threshold = best_sort_value - 2 * sbr_margin;
        for mv in self.iter_mut() {
            if mv.value < reduction_threshold {
                mv.reduction_depth = 1;
            }
        }
    }

    /// Evaluates moves specifically for endgame positions.
    fn evaluate_moves_endgame(
        &mut self,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        tt_move: Square,
    ) {
        let sort_depth = match depth {
            0..=17 => 0,
            18..=25 => 1,
            _ => 2,
        };

        for mv in self.iter_mut() {
            if mv.flipped == board.opponent {
                // Wipeout move
                mv.value = WIPEOUT_VALUE;
            } else if mv.sq == tt_move {
                // Transposition table move
                mv.value = TT_MOVE_VALUE;
            } else {
                // Evaluate using shallow search
                let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                ctx.update(mv.sq, mv.flipped);

                let score = match sort_depth {
                    0 => -midgame::evaluate(ctx, &next),
                    1 => -midgame::evaluate_depth1(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                    2 => -midgame::evaluate_depth2(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                    _ => unreachable!(),
                };
                mv.value = score.value();

                const MOBILITY_SCALE: i32 = ScaledScore::SCALE * 2;
                const POTENTIAL_MOBILITY_SCALE: i32 = ScaledScore::SCALE;

                let (moves, potential) = next.get_moves_and_potential();
                let mobility = moves.corner_weighted_count() as i32;
                let potential_mobility = potential.corner_weighted_count() as i32;
                mv.value -= mobility * MOBILITY_SCALE;
                mv.value -= potential_mobility * POTENTIAL_MOBILITY_SCALE;

                ctx.undo(mv.sq);
            };
        }
    }

    /// Evaluates moves using heuristics.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Search context for node counting
    /// * `board` - Current board position
    /// * `tt_move` - Move suggested by transposition table (if any)
    pub fn evaluate_moves_fast(&mut self, ctx: &mut SearchContext, board: &Board, tt_move: Square) {
        /// Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/move.c#L30
        #[rustfmt::skip]
        const SQUARE_VALUE: [i32; 64] = [
            18,  4, 16, 12, 12, 16,  4, 18,
             4,  2,  6,  8,  8,  6,  2,  4,
            16,  6, 14, 10, 10, 14,  6, 16,
            12,  8, 10,  0,  0, 10,  8, 12,
            12,  8, 10,  0,  0, 10,  8, 12,
            16,  6, 14, 10, 10, 14,  6, 16,
             4,  2,  6,  8,  8,  6,  2,  4,
            18,  4, 16, 12, 12, 16,  4, 18,
        ];

        const SQUARE_VALUE_WEIGHT: i32 = 1 << 8;
        const CORNER_STABILITY_WEIGHT: i32 = 1 << 12;
        const POTENTIAL_MOBILITY_WEIGHT: i32 = 1 << 10;
        const MOBILITY_WEIGHT: i32 = 1 << 14;

        for mv in self.iter_mut() {
            mv.value = if mv.flipped == board.opponent {
                // Wipeout move (capture all opponent discs)
                WIPEOUT_VALUE
            } else if mv.sq == tt_move {
                // Transposition table move
                TT_MOVE_VALUE
            } else {
                ctx.increment_nodes();
                let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                let (moves, potential) = next.get_moves_and_potential();
                let potential_mobility = potential.corner_weighted_count() as i32;
                let corner_stability = next.opponent.corner_stability() as i32;
                let weighted_mobility = moves.corner_weighted_count() as i32;
                let mut value = SQUARE_VALUE[mv.sq.index()] * SQUARE_VALUE_WEIGHT;
                value += corner_stability * CORNER_STABILITY_WEIGHT;
                value += (36 - potential_mobility) * POTENTIAL_MOBILITY_WEIGHT;
                value += (36 - weighted_mobility) * MOBILITY_WEIGHT;
                value
            }
        }
    }

    /// Sorts all moves in descending order of their evaluation values.
    #[inline]
    pub fn sort(&mut self) {
        let len = self.moves.len();
        match len {
            0 | 1 => {}
            2 => sort2(&mut self.moves),
            3 => sort3(&mut self.moves),
            _ => self.moves.sort_unstable_by_key(|m| -m.value),
        }
    }

    /// Excludes moves that were selected as best moves for earlier PV lines in Multi-PV search.
    ///
    /// In Multi-PV mode, each PV line explores a different best move at the root. This method
    /// retains only moves that appear in `root_moves` from `pv_idx` onwards, excluding moves
    /// that were already selected as the best move for earlier PV lines (indices 0..pv_idx).
    ///
    /// # Arguments
    ///
    /// * `ctx` - Search context containing root_moves and pv_idx
    pub fn exclude_earlier_pv_moves(&mut self, ctx: &SearchContext) {
        if ctx.pv_idx() == 0 {
            return; // No filtering needed for first PV line
        }

        self.moves
            .retain(|mv| ctx.root_moves.contains_from_pv_idx(mv.sq));
    }
}

#[inline(always)]
fn cas(moves: &mut [Move], i: usize, j: usize) {
    debug_assert!(i < moves.len());
    debug_assert!(j < moves.len());

    unsafe {
        let base_ptr = moves.as_mut_ptr();

        let ptr_i = base_ptr.add(i);
        let ptr_j = base_ptr.add(j);

        if (*ptr_i).value < (*ptr_j).value {
            std::ptr::swap(ptr_i, ptr_j);
        }
    }
}

/// 2 elements: 1 comparison (optimal).
#[inline]
fn sort2(m: &mut [Move]) {
    cas(m, 0, 1);
}

/// 3 elements: 3 comparisons (optimal).
#[inline]
fn sort3(m: &mut [Move]) {
    cas(m, 0, 1);
    cas(m, 1, 2);
    cas(m, 0, 1);
}

/// Thread-safe iterator for distributing moves across multiple search threads.
pub struct ConcurrentMoveIterator {
    /// The move list being iterated over.
    move_list: MoveList,
    /// Atomic counter tracking the next move index to return.
    current: atomic::AtomicUsize,
}

impl ConcurrentMoveIterator {
    /// Creates a new concurrent iterator from a move list.
    ///
    /// # Arguments
    ///
    /// * `move_list` - The move list to iterate over
    ///
    /// # Returns
    ///
    /// A new concurrent iterator starting at the first move.
    pub fn new(move_list: MoveList) -> ConcurrentMoveIterator {
        ConcurrentMoveIterator {
            move_list,
            current: atomic::AtomicUsize::new(0),
        }
    }

    /// Retrieves the next move and its index.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - Reference to the next move
    /// - 1-based index of the move in the original list
    ///
    /// Returns None when all moves have been consumed.
    pub fn next(&self) -> Option<(&Move, usize)> {
        let current = self.current.fetch_add(1, atomic::Ordering::Relaxed);
        if current < self.move_list.moves.len() {
            Some((&self.move_list.moves[current], current + 1))
        } else {
            None
        }
    }

    /// Returns the total number of moves available in this iterator.
    ///
    /// # Returns
    ///
    /// Total count of moves (does not change as moves are consumed).
    #[inline]
    pub fn count(&self) -> usize {
        self.move_list.count()
    }

    /// Returns the number of moves remaining to be consumed.
    ///
    /// # Returns
    ///
    /// Remaining move count.
    #[inline]
    pub fn remaining(&self) -> usize {
        let current = self.current.load(atomic::Ordering::Relaxed);
        self.move_list.count().saturating_sub(current)
    }
}

/// Lazy-sorting iterator that yields moves in order of decreasing evaluation value.
///
/// This iterator implements a selection-sort approach, finding the best remaining
/// move on each call to next(). This is more efficient than full sorting when
/// only the first few moves are needed, which is common in alpha-beta search
/// due to early cutoffs.
pub struct BestFirstMoveIterator {
    /// Owned moves array, partially sorted as iteration progresses.
    moves: ArrayVec<Move, MAX_MOVES>,
    /// Current position in the iteration.
    current: usize,
}

impl BestFirstMoveIterator {
    /// Creates a new owning best-first iterator from a moves array.
    ///
    /// # Arguments
    ///
    /// * `moves` - The moves array to take ownership of
    #[inline]
    pub fn new(moves: ArrayVec<Move, MAX_MOVES>) -> Self {
        BestFirstMoveIterator { moves, current: 0 }
    }

    /// Returns the number of remaining moves.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.moves.len() - self.current
    }
}

impl Iterator for BestFirstMoveIterator {
    type Item = Move;

    /// Returns the next best move, consuming it.
    ///
    /// This performs a partial selection sort: finds the maximum among
    /// remaining elements, swaps it to the current position, and returns it.
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let len = self.moves.len();
        let current = self.current;
        if current >= len {
            return None;
        }

        unsafe {
            let base_ptr = self.moves.as_mut_ptr();
            let current_ptr = base_ptr.add(current);

            let mut max_ptr = current_ptr;
            let mut max_val = (*current_ptr).value;

            let mut ptr = current_ptr.add(1);
            let end_ptr = base_ptr.add(len);

            while ptr < end_ptr {
                let val = (*ptr).value;
                if val > max_val {
                    max_val = val;
                    max_ptr = ptr;
                }
                ptr = ptr.add(1);
            }

            // Swap if needed
            if max_ptr != current_ptr {
                std::ptr::swap(current_ptr, max_ptr);
            }

            let result = *current_ptr;
            self.current = current + 1;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.moves.len() - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BestFirstMoveIterator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disc::Disc;
    use std::collections::HashSet;

    /// Tests move generation for the starting position.
    #[test]
    fn test_move_list_new() {
        let board = Board::new();
        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count(), 4);

        // Verify moves are at correct positions
        let moves: Vec<Square> = move_list.iter().map(|m| m.sq).collect();
        assert!(moves.contains(&Square::D3));
        assert!(moves.contains(&Square::C4));
        assert!(moves.contains(&Square::F5));
        assert!(moves.contains(&Square::E6));
    }

    /// Tests move generation with more complex position.
    #[test]
    fn test_move_list_generation_complex() {
        // Create a position with known moves - use standard reversi position
        let board = Board::from_string(
            "--------\
             --------\
             ---OX---\
             --OXX---\
             --XXX---\
             --------\
             --------\
             --------",
            Disc::Black,
        )
        .unwrap();

        let move_list = MoveList::new(&board);
        assert!(move_list.count() > 0);

        // Verify all moves have valid flipped discs
        for mv in move_list.iter() {
            assert!(!mv.flipped.is_empty());
            assert_eq!(mv.value, i32::MIN); // Initial value
            assert_eq!(mv.reduction_depth, 0);
        }
    }

    /// Tests move generation when no moves are available.
    #[test]
    fn test_move_list_no_moves() {
        // Create position with no legal moves
        let board = Board::from_bitboards(u64::MAX, 0);
        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count(), 0);
        assert!(move_list.first().is_none());
    }

    /// Tests Move struct creation and properties.
    #[test]
    fn test_move_new() {
        let sq = Square::E4;
        let flipped = Bitboard(0x0000001000000000u64);
        let mv = Move::new(sq, flipped);

        assert_eq!(mv.sq, sq);
        assert_eq!(mv.flipped, flipped);
        assert_eq!(mv.value, 0);
        assert_eq!(mv.reduction_depth, 0);
    }

    /// Tests first() method.
    #[test]
    fn test_first() {
        let board = Board::new();
        let move_list = MoveList::new(&board);

        let first = move_list.first().unwrap();
        let first_iter = move_list.iter().next().unwrap();
        assert_eq!(first.sq, first_iter.sq);
    }

    /// Tests iterator methods.
    #[test]
    fn test_iterators() {
        let board = Board::new();
        let move_list = MoveList::new(&board);

        // Test iter()
        let count_iter = move_list.iter().count();
        assert_eq!(count_iter, move_list.count());

        // Test that iterator returns moves in order
        let squares: Vec<Square> = move_list.iter().map(|m| m.sq).collect();
        assert_eq!(squares.len(), 4);
    }

    /// Tests the sort method.
    #[test]
    fn test_sort() {
        let board = Board::new();
        let mut move_list = MoveList::new(&board);

        // Set values in non-sorted order
        move_list.moves[0].value = 10;
        move_list.moves[1].value = 30;
        move_list.moves[2].value = 20;
        move_list.moves[3].value = 40;

        move_list.sort();

        // Verify sorted in descending order
        let values: Vec<i32> = move_list.iter().map(|m| m.value).collect();
        assert_eq!(values, vec![40, 30, 20, 10]);
    }

    /// Tests the best-first iterator with manually set move values.
    #[test]
    fn test_best_first_iter() {
        let board = Board::new();
        let mut move_list = MoveList::new(&board);
        move_list.moves[0].value = 10;
        move_list.moves[1].value = 5;
        move_list.moves[2].value = 15;
        move_list.moves[3].value = 2;
        // move_list.count = 4; // Already 4 from new()

        let mut iter = move_list.into_best_first_iter();
        assert_eq!(iter.next().unwrap().value, 15);
        assert_eq!(iter.next().unwrap().value, 10);
        assert_eq!(iter.next().unwrap().value, 5);
        assert_eq!(iter.next().unwrap().value, 2);
        assert!(iter.next().is_none());
    }

    /// Tests best-first iterator with equal values.
    #[test]
    fn test_best_first_iter_equal_values() {
        let board = Board::new();
        let mut move_list = MoveList::new(&board);

        // Set all values equal
        for i in 0..move_list.count() {
            move_list.moves[i].value = 100;
        }

        let total = move_list.count();
        let iter = move_list.into_best_first_iter();
        let mut count = 0;
        for mv in iter {
            assert_eq!(mv.value, 100);
            count += 1;
        }
        assert_eq!(count, total);
    }

    /// Tests best-first iterator behavior with no legal moves.
    #[test]
    fn test_best_first_iter_empty_list() {
        let board = Board::from_bitboards(u64::MAX, 0);
        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count(), 0);

        let mut iter = move_list.into_best_first_iter();
        assert!(iter.next().is_none());
    }

    /// Tests best-first iterator with single move.
    #[test]
    fn test_best_first_iter_single_move() {
        // Create position with only one legal move
        let board = Board::from_string(
            "XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXX\
             XXXXXXXO\
             XXXXXXO-",
            Disc::Black,
        )
        .unwrap();

        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count(), 1);

        let mut iter = move_list.into_best_first_iter();
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    /// Tests best-first iterator preserves all moves.
    #[test]
    fn test_best_first_iter_completeness() {
        let board = Board::new();
        let mut move_list = MoveList::new(&board);

        // Set unique values
        for i in 0..move_list.count() {
            move_list.moves[i].value = (i * 10) as i32;
        }

        let count = move_list.count();
        let iter = move_list.into_best_first_iter();
        let mut seen_values = HashSet::new();

        for mv in iter {
            assert!(seen_values.insert(mv.value));
        }

        assert_eq!(seen_values.len(), count);
    }

    /// Tests concurrent move iterator.
    #[test]
    fn test_concurrent_move_iterator() {
        let board = Board::new();
        let move_list = MoveList::new(&board);
        let concurrent_iter = ConcurrentMoveIterator::new(move_list);

        assert_eq!(concurrent_iter.count(), 4);
        assert_eq!(concurrent_iter.remaining(), 4);

        // Get all moves
        let mut moves = Vec::new();
        while let Some((mv, idx)) = concurrent_iter.next() {
            moves.push((mv.sq, idx));
        }

        assert_eq!(moves.len(), 4);
        assert_eq!(concurrent_iter.remaining(), 0);
        // Verify indices are 1-based and sequential
        for (i, (_, idx)) in moves.iter().enumerate() {
            assert_eq!(*idx, i + 1);
        }

        // Verify no more moves
        assert!(concurrent_iter.next().is_none());
    }
}
