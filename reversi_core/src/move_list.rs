use std::sync::atomic;

use crate::bitboard;
use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::constants::SCORE_INF;
use crate::flip;
use crate::search::midgame;
use crate::search::search_context::SearchContext;
use crate::square::Square;
use crate::types::Depth;

const MAX_MOVES: usize = 34;
const WIPEOUT_VALUE: i32 = 1 << 30;
const TT_MOVE_VALUE: i32 = 1 << 20;
const MOBILITY_WEIGHT: i32 = 1 << 14;
const CORNER_STABILITY_WEIGHT: i32 = 1 << 11;

/// Represents a single move in the game.
#[derive(Clone, Copy)]
pub struct Move {
    /// The square where the move is placed.
    pub sq: Square,
    /// The bitboard representing the pieces flipped by this move.
    pub flipped: u64,
    /// The value associated with this move.
    pub value: i32,

    pub reduction_depth: Depth,
}

impl Move {
    /// Creates a new `Move` with the specified square and flipped bitboard.
    ///
    /// # Arguments
    ///
    /// * `sq` - The square where the move is placed.
    /// * `flipped` - The bitboard representing the pieces flipped by this move.
    ///
    /// # Returns
    ///
    /// A new instance of `Move`.
    pub fn new(sq: Square, flipped: u64) -> Move {
        Move {
            sq,
            flipped,
            value: 0,
            reduction_depth: 0,
        }
    }
}

/// A list of possible moves.
pub struct MoveList {
    /// Buffer storing the moves.
    moves: [Move; MAX_MOVES],
    /// The count of moves in the list.
    pub count: usize,
}

impl MoveList {
    /// Creates a new `MoveList` based on the current `Board` state.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the current game board.
    ///
    /// # Returns
    ///
    /// A new instance of `MoveList` containing all valid moves.
    pub fn new(board: &Board) -> MoveList {
        let mut move_buffer = [Move {
            sq: Square::None,
            flipped: 0,
            value: 0,
            reduction_depth: 0,
        }; MAX_MOVES];
        let mut count = 0;
        for sq in BitboardIterator::new(board.get_moves()) {
            move_buffer[count].sq = sq;
            move_buffer[count].flipped = flip::flip(sq, board.player, board.opponent);
            move_buffer[count].value = i32::MIN;
            count += 1;
        }

        MoveList {
            moves: move_buffer,
            count,
        }
    }

    /// Retrieves the first move in the list, if any.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the first `Move` if the list is not empty, otherwise `None`.
    #[inline]
    pub fn first(&self) -> Option<&Move> {
        if self.count == 0 {
            None
        } else {
            Some(&self.moves[0])
        }
    }

    /// Returns an iterator over the moves in the list.
    ///
    /// # Returns
    ///
    /// An instance of `MoveIterator`.
    #[inline]
    pub fn iter(&self) -> MoveIterator {
        MoveIterator {
            move_list: self,
            current: 0,
        }
    }

    /// Returns an iterator over the moves in the list sorted by their values.
    ///
    /// # Returns
    /// An instance of `SortedMoveIterator`.
    #[inline]
    pub fn best_first_iter(&self) -> BestFirstMoveIterator {
        BestFirstMoveIterator::new(self)
    }

    pub fn evaluate_moves(
        &mut self,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        tt_move: Square,
    ) {
        #[rustfmt::skip]
        const MIN_DEPTH: [u32; 64] = [
            19, 18, 18, 18, 17, 17, 17, 16,
            16, 16, 15, 15, 15, 14, 14, 14,
            13, 13, 13, 12, 12, 12, 11, 11,
            11, 10, 10, 10,  9,  9,  9,  9,
            9,  9,  9,  9,  9,  9,  9,  9,
            9,  9,  9,  9,  9,  9,  9,  9,
            9,  9,  9,  9,  9,  9,  9,  9,
            9,  9,  9,  9,  9,  9,  9,  9
        ];

        if depth < MIN_DEPTH[ctx.empty_list.count as usize] {
            self.evaluate_moves_fast(board, tt_move);
        } else {
            let mut sort_depth = (depth as i32 - 15) / 3;
            sort_depth = sort_depth.clamp(0, 3);

            let mut max_value = -SCORE_INF;
            for mv in self.moves.iter_mut().take(self.count) {
                if mv.flipped == board.opponent {
                    mv.value = WIPEOUT_VALUE;
                } else if mv.sq == tt_move {
                    mv.value = TT_MOVE_VALUE;
                } else {
                    let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                    ctx.update(mv);
                    mv.value = match sort_depth {
                        0 => -midgame::evaluate(ctx, &next),
                        1 => -midgame::evaluate_depth1(ctx, &next, -SCORE_INF, SCORE_INF),
                        2 => -midgame::evaluate_depth2(ctx, &next, -SCORE_INF, SCORE_INF),
                        _ => -midgame::shallow_search::<crate::types::PV>(
                            ctx, &next, sort_depth as Depth, -SCORE_INF, SCORE_INF,
                        ),
                    };
                    ctx.undo(mv);
                    max_value = max_value.max(mv.value);
                };
            }
        }
    }

    /// Evaluates all moves in the list quickly by assigning values based on specific criteria.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the current game board.
    /// * `tt_move` - The move stored in the transposition table.
    pub fn evaluate_moves_fast(&mut self, board: &Board, tt_move: Square) {
        for mv in self.moves.iter_mut().take(self.count) {
            mv.value = if mv.flipped == board.opponent {
                WIPEOUT_VALUE
            } else if mv.sq == tt_move {
                TT_MOVE_VALUE
            } else {
                let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                let mut value =
                    bitboard::get_corner_stability(next.opponent) as i32 * CORNER_STABILITY_WEIGHT;
                value += (36 - (bitboard::corner_weighted_mobility(next.get_moves()) as i32))
                    * MOBILITY_WEIGHT;
                value
            }
        }
    }

    /// Sorts the move list based on their evaluated values.
    #[inline]
    pub fn sort(&mut self) {
        self.moves[..self.count].sort_unstable_by(|a, b| b.value.cmp(&a.value));
    }
}

/// An iterator over the `MoveList`.
pub struct MoveIterator<'a> {
    /// Reference to the `MoveList` being iterated.
    move_list: &'a MoveList,
    /// The current index in the iteration.
    current: usize,
}

impl<'a> Iterator for MoveIterator<'a> {
    type Item = &'a Move;

    /// Returns the next move in the iteration.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the next `Move` if available, otherwise `None`.
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.move_list.count {
            let result = Some(&self.move_list.moves[self.current]);
            self.current += 1;
            result
        } else {
            None
        }
    }
}

/// A thread-safe concurrent move iterator that can be safely shared among multiple threads.
pub struct ConcurrentMoveIterator {
    move_list: MoveList,
    current: atomic::AtomicUsize,
}

impl ConcurrentMoveIterator {
    pub fn new(move_list: MoveList) -> ConcurrentMoveIterator {
        ConcurrentMoveIterator {
            move_list,
            current: atomic::AtomicUsize::new(0),
        }
    }

    /// Returns the next move in the iteration, along with its original index in the `MoveList`.
    ///
    /// # Returns
    ///
    /// An `Option` containing a tuple with a reference to the next `Move` and its original index (1-based)
    /// if available, otherwise `None`. The index is 1-based.
    pub fn next(&self) -> Option<(&Move, usize)> {
        let current = self.current.fetch_add(1, atomic::Ordering::SeqCst);
        if current < self.move_list.count {
            Some((&self.move_list.moves[current], current + 1))
        } else {
            None
        }
    }

    /// Returns the total number of moves in the list.
    ///
    /// # Returns
    ///
    /// The total number of moves in the list.
    #[inline]
    pub fn count(&self) -> usize {
        self.move_list.count
    }
}

/// An iterator that returns moves sorted by their values in descending order.
pub struct BestFirstMoveIterator<'a> {
    /// Reference to the `MoveList` being iterated.
    move_list: &'a MoveList,
    /// Array of indices representing the order of moves.
    indices: [usize; MAX_MOVES],
    /// The current position in the sorted iteration.
    current: usize,
}

impl BestFirstMoveIterator<'_> {
    /// Creates a new `SortedMoveIterator` with initial indices.
    ///
    /// # Arguments
    ///
    /// * `move_list` - A reference to the `MoveList` to iterate over.
    ///
    /// # Returns
    ///
    /// A new instance of `SortedMoveIterator`.
    pub fn new(move_list: &MoveList) -> BestFirstMoveIterator {
        #[rustfmt::skip]
        let indices = BestFirstMoveIterator::create_indices();

        BestFirstMoveIterator {
            move_list,
            indices,
            current: 0,
        }
    }

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

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == self.move_list.count {
            return None;
        }

        let mut max_idx = self.current;
        for i in (self.current + 1)..self.move_list.count {
            if self.move_list.moves[self.indices[i]].value
                > self.move_list.moves[self.indices[max_idx]].value
            {
                max_idx = i;
            }
        }

        self.indices.swap(self.current, max_idx);
        let result = Some(&self.move_list.moves[self.indices[self.current]]);
        self.current += 1;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_list_new() {
        let board = Board::new();
        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count, 4);
    }

    #[test]
    fn test_best_first_iter() {
        let board = Board::new();
        let mut move_list = MoveList::new(&board);
        move_list.moves[0].value = 10;
        move_list.moves[1].value = 5;
        move_list.moves[2].value = 15;
        move_list.moves[3].value = 2;
        move_list.count = 4;

        let mut iter = move_list.best_first_iter();
        assert_eq!(iter.next().unwrap().value, 15);
        assert_eq!(iter.next().unwrap().value, 10);
        assert_eq!(iter.next().unwrap().value, 5);
        assert_eq!(iter.next().unwrap().value, 2);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_best_first_iter_empty_list() {
        let board = Board::from_bitboards(u64::MAX, 0);
        let move_list = MoveList::new(&board);
        assert_eq!(move_list.count, 0);

        let mut iter = move_list.best_first_iter();
        assert!(iter.next().is_none());
    }
}
