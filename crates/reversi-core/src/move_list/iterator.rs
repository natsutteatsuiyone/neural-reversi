//! Iterators over generated move lists.

use std::sync::atomic;

use super::{Move, MoveList};

impl MoveList {
    /// Returns an iterator that yields moves in order of decreasing value.
    #[inline]
    pub fn best_first_iter(&mut self) -> BestFirstMoveIterator<'_> {
        BestFirstMoveIterator::new(&mut self.moves)
    }
}

/// Thread-safe iterator for distributing moves across multiple search threads.
pub struct ConcurrentMoveIterator {
    move_list: MoveList,
    current: atomic::AtomicUsize,
}

impl ConcurrentMoveIterator {
    /// Creates a new concurrent iterator from a move list.
    pub fn new(move_list: MoveList) -> ConcurrentMoveIterator {
        Self::from_offset(move_list, 0)
    }

    /// Creates a concurrent iterator starting from the given offset.
    ///
    /// Moves before `start` are skipped. The `next()` method returns 1-based
    /// indices relative to the original list, so helpers at a split point
    /// correctly see `move_count > 1`.
    pub fn from_offset(move_list: MoveList, start: usize) -> ConcurrentMoveIterator {
        ConcurrentMoveIterator {
            move_list,
            current: atomic::AtomicUsize::new(start),
        }
    }

    /// Returns the next move and its 1-based index atomically.
    #[inline]
    pub fn next(&self) -> Option<(&Move, usize)> {
        let current = self.current.fetch_add(1, atomic::Ordering::Relaxed);
        if current < self.move_list.moves.len() {
            // SAFETY: `current < len` checked above.
            Some((
                unsafe { self.move_list.moves.get_unchecked_ref(current) },
                current + 1,
            ))
        } else {
            None
        }
    }

    /// Returns the total number of moves in this iterator.
    #[inline]
    pub fn count(&self) -> usize {
        self.move_list.count()
    }

    /// Returns the number of moves not yet consumed.
    #[inline]
    pub fn remaining(&self) -> usize {
        let current = self.current.load(atomic::Ordering::Relaxed);
        // Uses `saturating_sub` because concurrent `fetch_add` calls can cause
        // `current` to exceed `count` when multiple threads race past the last move.
        self.move_list.count().saturating_sub(current)
    }
}

/// Lazy-sorting iterator that yields moves in order of decreasing evaluation value.
///
/// This iterator implements a selection-sort approach, finding the best remaining
/// move on each call to next(). This is more efficient than full sorting when
/// only the first few moves are needed, which is common in alpha-beta search
/// due to early cutoffs.
pub struct BestFirstMoveIterator<'a> {
    moves: &'a mut [Move],
    current: usize,
}

impl<'a> BestFirstMoveIterator<'a> {
    #[inline]
    fn new(moves: &'a mut [Move]) -> Self {
        BestFirstMoveIterator { moves, current: 0 }
    }

    /// Returns the number of remaining moves.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.moves.len() - self.current
    }
}

impl Iterator for BestFirstMoveIterator<'_> {
    type Item = Move;

    /// Returns the next best move, consuming it.
    ///
    /// This performs a partial selection step: finds the maximum among the
    /// remaining elements and returns it.
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let len = self.moves.len();
        let current = self.current;
        if current >= len {
            return None;
        }

        // SAFETY: `current < len`; all pointers stay within `0..len`.
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

            let result = *max_ptr;
            if max_ptr != current_ptr {
                *max_ptr = *current_ptr;
            }

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

impl ExactSizeIterator for BestFirstMoveIterator<'_> {}
