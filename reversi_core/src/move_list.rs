//! Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/move.c

use std::slice;
use std::sync::atomic;

use crate::bitboard::{BitboardIterator, corner_weighted_count, get_corner_stability};
use crate::board::Board;
use crate::constants::{EVAL_SCORE_SCALE_BITS, SCORE_INF};
use crate::flip;
use crate::search::midgame;
use crate::search::node_type::{NodeType, PV};
use crate::search::search_context::{GamePhase, SearchContext};
use crate::square::Square;
use crate::types::Depth;

/// Maximum number of moves possible in a Reversi position.
const MAX_MOVES: usize = 34;

/// Value assigned to wipeout moves (capturing all opponent pieces).
const WIPEOUT_VALUE: i32 = 1 << 30;

/// Value assigned to moves suggested by the transposition table.
const TT_MOVE_VALUE: i32 = 1 << 20;

/// Weight factor for potential mobility evaluation.
const POTENTIAL_MOBILITY_WEIGHT: i32 = 1 << 5;

/// Weight factor for mobility evaluation.
const MOBILITY_WEIGHT: i32 = 1 << 14;

/// Weight factor for corner stability evaluation.
const CORNER_STABILITY_WEIGHT: i32 = 1 << 11;

/// Value assigned to moves that have already been searched in root node.
const SEARCHED_MOVE_VALUE: i32 = -(1 << 20);

/// Represents a single move.
#[derive(Clone, Copy, Debug, Default)]
pub struct Move {
    /// The square where the piece is placed.
    pub sq: Square,
    /// Bitboard representing all opponent pieces flipped by this move.
    pub flipped: u64,
    /// Evaluation score for move ordering (higher = better).
    pub value: i32,
    /// Suggested depth reduction for this move in search.
    pub reduction_depth: Depth,
}

impl Move {
    /// Creates a new move with the specified square and flipped pieces.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square where the piece is placed
    /// * `flipped` - Bitboard of opponent pieces flipped by this move
    ///
    /// # Returns
    ///
    /// A new Move instance with default evaluation data
    #[inline]
    pub fn new(sq: Square, flipped: u64) -> Move {
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
    /// Fixed-size buffer storing all moves in the position.
    move_buffer: [Move; MAX_MOVES],
    /// Number of legal moves in this position.
    count: usize,
    /// The square of the wipeout move, if found.
    pub wipeout_move: Option<Square>,
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
    /// A new MoveList containing all legal moves for the current player
    pub fn new(board: &Board) -> MoveList {
        let mut move_buffer = [Move::default(); MAX_MOVES];
        let mut count = 0;
        let mut wipeout_move = None;
        for sq in BitboardIterator::new(board.get_moves()) {
            let flipped = flip::flip(sq, board.player, board.opponent);
            move_buffer[count].sq = sq;
            move_buffer[count].flipped = flipped;
            move_buffer[count].value = i32::MIN;
            count += 1;

            if flipped == board.opponent {
                wipeout_move = Some(sq);
            }
        }

        MoveList {
            move_buffer,
            count,
            wipeout_move,
        }
    }

    /// Returns the number of legal moves in this position.
    ///
    /// # Returns
    ///
    /// The count of legal moves
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the first move in the list, if any exists.
    ///
    /// # Returns
    ///
    /// Reference to the first move, or None if no legal moves exist
    #[inline]
    pub fn first(&self) -> Option<&Move> {
        if self.count == 0 {
            None
        } else {
            Some(&self.move_buffer[0])
        }
    }

    /// Returns an iterator over all moves in the list.
    ///
    /// The moves are returned in their current order, which may be the generation
    /// order or sorted order depending on previous operations.
    ///
    /// # Returns
    ///
    /// Iterator over moves in the list
    #[inline]
    pub fn iter(&self) -> slice::Iter<'_, Move> {
        self.move_buffer[..self.count].iter()
    }

    /// Returns a mutable iterator over all moves in the list.
    ///
    /// # Returns
    ///
    /// Mutable iterator over moves in the list
    #[inline]
    fn iter_mut(&mut self) -> slice::IterMut<'_, Move> {
        self.move_buffer[..self.count].iter_mut()
    }

    /// Returns an iterator that yields moves in order of decreasing value.
    ///
    /// # Returns
    ///
    /// Iterator that yields moves in best-first order
    #[inline]
    pub fn best_first_iter(&self) -> BestFirstMoveIterator<'_> {
        BestFirstMoveIterator::new(self)
    }

    /// Evaluates all moves to assign ordering values and reduction depths.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Search context with transposition table and statistics
    /// * `board` - Current position before making any move
    /// * `depth` - Remaining search depth at this node
    /// * `tt_move` - Best move from transposition table (if any)
    ///
    /// # Type Parameters
    ///
    /// * `NT` - Node type affecting evaluation depth and strategy
    pub fn evaluate_moves<NT: NodeType>(
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
            self.evaluate_moves_fast(board, tt_move);
            return;
        }

        const MAX_SORT_DEPTH: i32 = 3;
        let mut sort_depth = (depth as i32 - 15) / 3;
        sort_depth = sort_depth.clamp(1, MAX_SORT_DEPTH);

        let mut max_evaluated_value = -SCORE_INF;

        for mv in self.iter_mut() {
            if NT::ROOT_NODE && ctx.is_move_searched(mv.sq) {
                // Already searched in previous iteration
                mv.value = SEARCHED_MOVE_VALUE;
            } else if mv.flipped == board.opponent {
                // Wipeout move
                mv.value = WIPEOUT_VALUE;
            } else if mv.sq == tt_move {
                // Transposition table move
                mv.value = TT_MOVE_VALUE;
            } else {
                // Evaluate using shallow search
                let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                ctx.update(mv);

                mv.value = match sort_depth {
                    0 => -midgame::evaluate(ctx, &next),
                    1 => -midgame::evaluate_depth1(ctx, &next, -SCORE_INF, SCORE_INF),
                    2 => -midgame::evaluate_depth2(ctx, &next, -SCORE_INF, SCORE_INF),
                    _ => -midgame::shallow_search::<PV>(
                        ctx,
                        &next,
                        sort_depth as Depth,
                        -SCORE_INF,
                        SCORE_INF,
                    ),
                };

                ctx.undo(mv);
                max_evaluated_value = max_evaluated_value.max(mv.value);
            };
        }

        if ctx.game_phase == GamePhase::MidGame {
            // Score-Based Reduction: reduce depth for poor moves
            // This implements a form of late move reduction based on evaluation scores
            let sbr_margin: i32 = (9 + (MAX_SORT_DEPTH - sort_depth) * 2) << EVAL_SCORE_SCALE_BITS;
            let reduction_threshold = max_evaluated_value - sbr_margin;

            for mv in self.iter_mut() {
                if mv.value < reduction_threshold && mv.value != SEARCHED_MOVE_VALUE {
                    // Calculate reduction based on how much worse this move is
                    let diff = max_evaluated_value - mv.value;
                    let step = sbr_margin * 2;
                    mv.reduction_depth = ((diff + sbr_margin) / step) as Depth;
                }
            }
        }
    }

    /// Evaluates moves using heuristics
    ///
    /// # Arguments
    ///
    /// * `board` - Current board position
    /// * `tt_move` - Move suggested by transposition table (if any)
    pub fn evaluate_moves_fast(&mut self, board: &Board, tt_move: Square) {
        for mv in self.iter_mut() {
            mv.value = if mv.flipped == board.opponent {
                // Wipeout move (capture all opponent pieces)
                WIPEOUT_VALUE
            } else if mv.sq == tt_move {
                // Transposition table move
                TT_MOVE_VALUE
            } else {
                let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                let potential_mobility = corner_weighted_count(next.get_potential_moves()) as i32;
                let corner_stability = get_corner_stability(next.opponent) as i32;
                let weighted_mobility = corner_weighted_count(next.get_moves()) as i32;
                let mut value = corner_stability * CORNER_STABILITY_WEIGHT;
                value += (36 - potential_mobility) * POTENTIAL_MOBILITY_WEIGHT;
                value += (36 - weighted_mobility) * MOBILITY_WEIGHT;
                value
            }
        }
    }

    /// Sorts all moves in descending order of their evaluation values.
    #[inline]
    pub fn sort(&mut self) {
        self.move_buffer[..self.count].sort_unstable_by_key(|m| -m.value);
    }
}

/// Thread-safe iterator for distributing moves across multiple search threads.
pub struct ConcurrentMoveIterator {
    /// The move list being iterated over
    move_list: MoveList,
    /// Atomic counter tracking the next move index to return
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
    /// A new concurrent iterator starting at the first move
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
        if current < self.move_list.count {
            Some((&self.move_list.move_buffer[current], current + 1))
        } else {
            None
        }
    }

    /// Returns the total number of moves available in this iterator.
    ///
    /// # Returns
    ///
    /// Total count of moves (does not change as moves are consumed)
    #[inline]
    pub fn count(&self) -> usize {
        self.move_list.count
    }
}

/// Lazy-sorting iterator that yields moves in order of decreasing evaluation value.
///
/// This iterator implements a selection-sort approach, finding the best remaining
/// move on each call to next(). This is more efficient than full sorting when
/// only the first few moves are needed, which is common in alpha-beta search
/// due to early cutoffs.
pub struct BestFirstMoveIterator<'a> {
    /// Reference to the move list being iterated
    move_list: &'a MoveList,
    /// Indices into the move list, rearranged as moves are selected
    indices: [usize; MAX_MOVES],
    /// Current position in the iteration (number of moves already returned)
    current: usize,
}

impl BestFirstMoveIterator<'_> {
    /// Creates a new best-first iterator over the given move list.
    ///
    /// # Arguments
    ///
    /// * `move_list` - Reference to the move list to iterate over
    ///
    /// # Returns
    ///
    /// A new iterator ready to yield moves in best-first order
    pub fn new(move_list: &MoveList) -> BestFirstMoveIterator<'_> {
        let indices = BestFirstMoveIterator::create_indices();

        BestFirstMoveIterator {
            move_list,
            indices,
            current: 0,
        }
    }

    /// Creates an identity permutation array for move indices.
    ///
    /// This initializes the indices array with [0, 1, 2, ..., MAX_MOVES-1]
    /// which will be rearranged as the iterator finds the best moves.
    const fn create_indices() -> [usize; MAX_MOVES] {
        let mut indices = [0; MAX_MOVES];
        let mut i = 0;
        while i < MAX_MOVES {
            indices[i] = i;
            i += 1;
        }
        indices
    }
}

impl<'a> Iterator for BestFirstMoveIterator<'a> {
    type Item = &'a Move;

    /// Returns the next best move from the remaining unexamined moves.
    ///
    /// This implements selection sort behavior: find the maximum value among
    /// remaining moves, swap it to the current position, and return it.
    /// This gives O(n²) total complexity but is efficient when only the
    /// first few moves are needed.
    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've exhausted all moves
        if self.current == self.move_list.count {
            return None;
        }

        // Find the move with the highest value among remaining moves
        let mut max_idx = self.current;
        for i in (self.current + 1)..self.move_list.count {
            if self.move_list.move_buffer[self.indices[i]].value
                > self.move_list.move_buffer[self.indices[max_idx]].value
            {
                max_idx = i;
            }
        }

        // Move the best move to the current position
        self.indices.swap(self.current, max_idx);
        let result = Some(&self.move_list.move_buffer[self.indices[self.current]]);
        self.current += 1;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::Piece;
    use std::collections::HashSet;

    /// Tests move generation for the starting position.
    #[test]
    fn test_move_list_new() {
        let board = Board::new();
        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count, 4);

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
            Piece::Black,
        );

        let move_list = MoveList::new(&board);
        assert!(move_list.count > 0);

        // Verify all moves have valid flipped pieces
        for mv in move_list.iter() {
            assert!(mv.flipped != 0);
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
        assert_eq!(move_list.count, 0);
        assert!(move_list.first().is_none());
    }

    /// Tests Move struct creation and properties.
    #[test]
    fn test_move_new() {
        let sq = Square::E4;
        let flipped = 0x0000001000000000u64;
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
        move_list.move_buffer[0].value = 10;
        move_list.move_buffer[1].value = 30;
        move_list.move_buffer[2].value = 20;
        move_list.move_buffer[3].value = 40;

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
        move_list.move_buffer[0].value = 10;
        move_list.move_buffer[1].value = 5;
        move_list.move_buffer[2].value = 15;
        move_list.move_buffer[3].value = 2;
        move_list.count = 4;

        let mut iter = move_list.best_first_iter();
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
        for i in 0..move_list.count {
            move_list.move_buffer[i].value = 100;
        }

        let iter = move_list.best_first_iter();
        let mut count = 0;
        for mv in iter {
            assert_eq!(mv.value, 100);
            count += 1;
        }
        assert_eq!(count, move_list.count);
    }

    /// Tests best-first iterator behavior with no legal moves.
    #[test]
    fn test_best_first_iter_empty_list() {
        let board = Board::from_bitboards(u64::MAX, 0);
        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count, 0);

        let mut iter = move_list.best_first_iter();
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
            Piece::Black,
        );

        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count, 1);

        let mut iter = move_list.best_first_iter();
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }

    /// Tests best-first iterator preserves all moves.
    #[test]
    fn test_best_first_iter_completeness() {
        let board = Board::new();
        let mut move_list = MoveList::new(&board);

        // Set unique values
        for i in 0..move_list.count {
            move_list.move_buffer[i].value = (i * 10) as i32;
        }

        let iter = move_list.best_first_iter();
        let mut seen_values = HashSet::new();

        for mv in iter {
            assert!(seen_values.insert(mv.value));
        }

        assert_eq!(seen_values.len(), move_list.count);
    }

    /// Tests concurrent move iterator.
    #[test]
    fn test_concurrent_move_iterator() {
        let board = Board::new();
        let move_list = MoveList::new(&board);
        let concurrent_iter = ConcurrentMoveIterator::new(move_list);

        assert_eq!(concurrent_iter.count(), 4);

        // Get all moves
        let mut moves = Vec::new();
        while let Some((mv, idx)) = concurrent_iter.next() {
            moves.push((mv.sq, idx));
        }

        assert_eq!(moves.len(), 4);
        // Verify indices are 1-based and sequential
        for (i, (_, idx)) in moves.iter().enumerate() {
            assert_eq!(*idx, i + 1);
        }

        // Verify no more moves
        assert!(concurrent_iter.next().is_none());
    }

    /// Tests evaluate_moves_fast with various scenarios.
    #[test]
    fn test_evaluate_moves_fast() {
        let board = Board::new();
        let mut move_list = MoveList::new(&board);

        // Test with no TT move
        move_list.evaluate_moves_fast(&board, Square::None);

        // All moves should have heuristic values
        for mv in move_list.iter() {
            assert_ne!(mv.value, i32::MIN);
            assert_ne!(mv.value, WIPEOUT_VALUE); // No wipeout in starting position
            assert_ne!(mv.value, TT_MOVE_VALUE); // No TT move set
        }

        // Test with TT move
        let tt_move = move_list.first().unwrap().sq;
        move_list.evaluate_moves_fast(&board, tt_move);

        let tt_move_found = move_list.iter().any(|m| m.value == TT_MOVE_VALUE);
        assert!(tt_move_found);
    }

    /// Tests wipeout move detection.
    #[test]
    fn test_wipeout_detection() {
        // Create position where next move captures all opponent pieces
        let board = Board::from_string(
            "........\
             ........\
             ........\
             ...XX...\
             ...XO...\
             ........\
             ........\
             ........",
            Piece::Black,
        );

        let mut move_list = MoveList::new(&board);

        // Find the wipeout move manually
        let mut wipeout_move = None;
        for mv in move_list.iter() {
            if mv.flipped == board.opponent {
                wipeout_move = Some(mv.sq);
                break;
            }
        }

        if let Some(wipeout_sq) = wipeout_move {
            move_list.evaluate_moves_fast(&board, Square::None);

            let wipeout_mv = move_list.iter().find(|m| m.sq == wipeout_sq).unwrap();
            assert_eq!(wipeout_mv.value, WIPEOUT_VALUE);

            let best = move_list.best_first_iter().next().unwrap();
            assert_eq!(best.sq, wipeout_sq);
        }
    }
}
