//! # Empty Square List Management
//!
//! This module provides an efficient data structure for managing empty squares
//! during Reversi/Othello game search. The `EmptyList` maintains a doubly-linked
//! list of empty squares in strategic order, enabling fast move generation and
//! endgame parity calculations.

use crate::board::Board;
use crate::square::Square;

/// Quadrant bit-mask for each square position
///
/// The board is divided into 4 quadrants for parity calculation:
/// - Top-left (A1-D4): bit mask 1
/// - Top-right (E1-H4): bit mask 2
/// - Bottom-left (A5-D8): bit mask 4
/// - Bottom-right (E5-H8): bit mask 8
///
/// Used for endgame parity optimization where different quadrant
/// combinations affect move ordering efficiency.
#[rustfmt::skip]
pub const QUADRANT_ID: [u8; 64] = [
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
];

/// Strategic ordering of squares for optimal move generation
///
/// Squares are ordered by their strategic importance in Reversi:
/// 1. Corners (most valuable)
/// 2. Stable edge squares
/// 3. Other edge squares
/// 4. Inner squares (decreasing stability)
///
/// This ordering improves alpha-beta pruning efficiency by
/// examining better moves first during search.
#[rustfmt::skip]
const PRESORTED: [Square; 64] = [
    // Corners (highest strategic value)
    Square::A1, Square::H1, Square::A8, Square::H8,

    // Edge squares near corners
    Square::C1, Square::F1, Square::A3, Square::H3,
    Square::A6, Square::H6, Square::C8, Square::F8,

    // Remaining edge squares
    Square::C3, Square::F3, Square::C6, Square::F6,
    Square::D1, Square::E1, Square::A4, Square::H4,

    // Inner ring squares
    Square::A5, Square::H5, Square::D8, Square::E8,
    Square::D3, Square::E3, Square::C4, Square::F4,
    Square::C5, Square::F5, Square::D6, Square::E6,
    Square::D2, Square::E2,

    // Central squares
    Square::B4, Square::G4, Square::B5, Square::G5,
    Square::D7, Square::E7, Square::C2, Square::F2,
    Square::B3, Square::G3, Square::B6, Square::G6,
    Square::C7, Square::F7,

    // Remaining squares
    Square::B1, Square::G1, Square::A2, Square::H2,
    Square::A7, Square::H7, Square::B8, Square::G8,
    Square::B2, Square::G2, Square::B7, Square::G7,
    Square::D4, Square::E4, Square::D5, Square::E5,
];

/// Node in the doubly-linked list of empty squares
///
/// Each square maintains links to its neighbors in the list
/// and its quadrant ID for parity calculations.
#[derive(Clone, Copy, Default)]
struct EmptyNode {
    /// Next square in the linked list (Square::None if last)
    next: Square,
    /// Previous square in the linked list (Square::None if first)
    prev: Square,
    /// Quadrant bit mask (1, 2, 4, or 8) for parity tracking
    quad_id: u8,
}

/// Efficient linked list of empty squares for move generation
///
/// The `EmptyList` maintains a doubly-linked list of empty squares
/// using an array-based implementation for cache efficiency.
#[derive(Clone)]
pub struct EmptyList {
    /// Array of linked list nodes indexed by square position
    nodes: [EmptyNode; 65],
    /// Number of empty squares currently in the list
    pub count: u32,
    /// XOR of all quadrant IDs (used for endgame parity optimization)
    pub parity: u8,
}
impl EmptyList {
    /// Creates a new `EmptyList` by scanning the board for empty squares.
    ///
    /// # Arguments
    ///
    /// * `board` - A reference to the current `Board` state.
    ///
    /// # Returns
    ///
    /// An instance of `EmptyList` containing all empty squares in the presorted order.
    pub fn new(board: &Board) -> Self {
        let mut count = 0;
        let mut parity: u8 = 0;
        let mut nodes: [EmptyNode; 65] = [EmptyNode::default(); 65];

        let mut prev_sq = Square::None;
        let empty_board = board.get_empty();
        for &sq in PRESORTED.iter() {
            if sq.bitboard() & empty_board != 0 {
                let sq_idx = sq.index();
                nodes[prev_sq.index()].next = sq;
                nodes[sq_idx].prev = prev_sq;
                nodes[sq_idx].quad_id = QUADRANT_ID[sq_idx];
                parity ^= nodes[sq_idx].quad_id;
                prev_sq = sq;
                count += 1;
            }
        }

        Self {
            nodes,
            count,
            parity,
        }
    }

    /// Retrieves the first empty square in the list.
    ///
    /// # Returns
    ///
    /// The `Square` representing the first empty square, or `Square::None` if empty.
    #[inline(always)]
    pub fn first(&self) -> Square {
        unsafe { self.nodes.get_unchecked(Square::None.index()).next }
    }

    /// Retrieves the first empty square and its quadrant ID.
    ///
    /// # Returns
    ///
    /// A tuple containing the first `Square` and its `quad_id`,
    /// or (`Square::None`, 0) if the list is empty.
    #[inline(always)]
    pub fn first_with_quad_id(&self) -> (Square, u8) {
        let first_sq = unsafe { self.nodes.get_unchecked(Square::None.index()).next };
        let quad_id = unsafe { self.nodes.get_unchecked(first_sq.index()).quad_id };
        (first_sq, quad_id)
    }

    /// Retrieves the next empty square following the given square.
    ///
    /// # Arguments
    ///
    /// * `sq` - The current `Square`. Must be a valid square in the list.
    ///
    /// # Returns
    ///
    /// The `Square` representing the next empty square, or `Square::None` if `sq` is the last square.
    #[inline(always)]
    pub fn next(&self, sq: Square) -> Square {
        unsafe { self.nodes.get_unchecked(sq.index()).next }
    }

    /// Retrieves the next empty square and its quadrant ID following the given square.
    ///
    /// # Arguments
    ///
    /// * `sq` - The current `Square`. Must be a square currently in the empty list.
    ///
    /// # Returns
    ///
    /// A tuple containing the next `Square` and its `quad_id`.
    /// Returns (`Square::None`, 0) if `sq` is the last square in the list.
    #[inline(always)]
    pub fn next_with_quad_id(&self, sq: Square) -> (Square, u8) {
        let next_sq = unsafe { self.nodes.get_unchecked(sq.index()).next };
        let quad_id = unsafe { self.nodes.get_unchecked(next_sq.index()).quad_id };
        (next_sq, quad_id)
    }

    /// Removes a square from the empty list.
    ///
    /// # Arguments
    ///
    /// * `sq` - The `Square` to remove. Must currently be in the list.
    #[inline(always)]
    pub fn remove(&mut self, sq: Square) {
        let node = self.nodes[sq.index()];
        let prev = node.prev;
        let next = node.next;
        unsafe { self.nodes.get_unchecked_mut(prev.index()).next = next };
        unsafe { self.nodes.get_unchecked_mut(next.index()).prev = prev };
        self.parity ^= node.quad_id;
        self.count -= 1;
    }

    /// Restores a previously removed square to the empty list.
    ///
    /// # Arguments
    ///
    /// * `sq` - The `Square` to restore. Must have been previously removed.
    #[inline(always)]
    pub fn restore(&mut self, sq: Square) {
        let node = self.nodes[sq.index()];
        unsafe { self.nodes.get_unchecked_mut(node.prev.index()).next = sq };
        unsafe { self.nodes.get_unchecked_mut(node.next.index()).prev = sq };
        self.parity ^= node.quad_id;
        self.count += 1;
    }

    /// Returns the current ply (number of moves played).
    ///
    /// In Reversi, the game starts with 4 pieces placed and 60 empty squares.
    /// Each move fills one empty square, so ply = 60 - empty_count.
    #[inline(always)]
    pub fn ply(&self) -> usize {
        (60 - self.count) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_list_new() {
        let board = Board::new();
        let empty_list = EmptyList::new(&board);

        assert_eq!(empty_list.count, 60);
        assert_eq!(empty_list.first(), Square::A1);
        assert_eq!(empty_list.next(Square::A1), Square::H1);
        assert_eq!(empty_list.next(Square::G7), Square::None);
    }

    #[test]
    fn test_empty_list_remove_restore() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        empty_list.remove(Square::A1);
        assert_eq!(empty_list.count, 59);
        assert_eq!(empty_list.first(), Square::H1);
        assert_eq!(empty_list.next(Square::H1), Square::A8);

        empty_list.restore(Square::A1);
        assert_eq!(empty_list.count, 60);
        assert_eq!(empty_list.first(), Square::A1);
        assert_eq!(empty_list.next(Square::A1), Square::H1);
    }

    #[test]
    fn test_presorted_order() {
        let board = Board::new();
        let empty_list = EmptyList::new(&board);

        let mut current = empty_list.first();
        let mut count = 0;
        while current != Square::None {
            assert_eq!(current, PRESORTED[count]);
            current = empty_list.next(current);
            count += 1;
        }
        assert_eq!(count, 60);
    }

    #[test]
    fn test_empty_list_parity() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        assert_eq!(empty_list.parity, 15);

        empty_list.remove(Square::A1);
        assert_eq!(empty_list.parity, 14);

        empty_list.restore(Square::A1);
        assert_eq!(empty_list.parity, 15);

        empty_list.remove(Square::H8);
        assert_eq!(empty_list.parity, 7);

        empty_list.remove(Square::A8);
        assert_eq!(empty_list.parity, 3);

        empty_list.remove(Square::H1);
        assert_eq!(empty_list.parity, 1);

        empty_list.remove(Square::A1);
        assert_eq!(empty_list.parity, 0);
    }

    #[test]
    fn test_quadrant_id_constants() {
        // Test that QUADRANT_ID correctly maps squares to quadrants
        assert_eq!(QUADRANT_ID[Square::A1.index()], 1); // Top-left
        assert_eq!(QUADRANT_ID[Square::D4.index()], 1); // Top-left
        assert_eq!(QUADRANT_ID[Square::E1.index()], 2); // Top-right
        assert_eq!(QUADRANT_ID[Square::H4.index()], 2); // Top-right
        assert_eq!(QUADRANT_ID[Square::A5.index()], 4); // Bottom-left
        assert_eq!(QUADRANT_ID[Square::D8.index()], 4); // Bottom-left
        assert_eq!(QUADRANT_ID[Square::E5.index()], 8); // Bottom-right
        assert_eq!(QUADRANT_ID[Square::H8.index()], 8); // Bottom-right
    }

    #[test]
    fn test_first_with_quad_id() {
        let board = Board::new();
        let empty_list = EmptyList::new(&board);

        let (first_sq, quad_id) = empty_list.first_with_quad_id();
        assert_eq!(first_sq, Square::A1);
        assert_eq!(quad_id, 1); // A1 is in quadrant 1
    }

    #[test]
    fn test_next_with_quad_id() {
        let board = Board::new();
        let empty_list = EmptyList::new(&board);

        let (next_sq, quad_id) = empty_list.next_with_quad_id(Square::A1);
        assert_eq!(next_sq, Square::H1);
        assert_eq!(quad_id, 2); // H1 is in quadrant 2

        let (next_sq, quad_id) = empty_list.next_with_quad_id(Square::H1);
        assert_eq!(next_sq, Square::A8);
        assert_eq!(quad_id, 4); // A8 is in quadrant 4
    }

    #[test]
    fn test_ply_calculation() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        // Initial ply should be 0 (60 empty squares)
        assert_eq!(empty_list.ply(), 0);

        // Remove one square (simulate one move)
        empty_list.remove(Square::A1);
        assert_eq!(empty_list.ply(), 1);

        // Remove more squares
        empty_list.remove(Square::H1);
        empty_list.remove(Square::A8);
        assert_eq!(empty_list.ply(), 3);

        // Restore one square
        empty_list.restore(Square::A1);
        assert_eq!(empty_list.ply(), 2);
    }

    #[test]
    fn test_multiple_remove_restore_cycles() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);
        let initial_parity = empty_list.parity;
        let initial_count = empty_list.count;
        let initial_first = empty_list.first();

        // Test removing and restoring multiple squares
        let squares_to_test = [Square::A1, Square::H8, Square::D4, Square::E5];

        for &sq in &squares_to_test {
            empty_list.remove(sq);
        }

        // Verify count decreased
        assert_eq!(empty_list.count, initial_count - squares_to_test.len() as u32);

        // Restore all squares in reverse order
        for &sq in squares_to_test.iter().rev() {
            empty_list.restore(sq);
        }

        // Verify we're back to original state
        assert_eq!(empty_list.count, initial_count);
        assert_eq!(empty_list.parity, initial_parity);
        assert_eq!(empty_list.first(), initial_first);
    }

    #[test]
    fn test_remove_restore_order_independence() {
        let board = Board::new();
        let mut empty_list1 = EmptyList::new(&board);
        let mut empty_list2 = EmptyList::new(&board);

        // Remove squares in different orders
        empty_list1.remove(Square::A1);
        empty_list1.remove(Square::H8);
        empty_list1.remove(Square::D4);

        empty_list2.remove(Square::H8);
        empty_list2.remove(Square::D4);
        empty_list2.remove(Square::A1);

        // Both should have the same count and parity
        assert_eq!(empty_list1.count, empty_list2.count);
        assert_eq!(empty_list1.parity, empty_list2.parity);
    }

    #[test]
    fn test_edge_cases_empty_list() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        // Remove all corner squares
        let corners = [Square::A1, Square::H1, Square::A8, Square::H8];
        for &corner in &corners {
            empty_list.remove(corner);
        }

        // Verify first square is now the next in presorted order
        assert_eq!(empty_list.first(), Square::C1);
        assert_eq!(empty_list.count, 56);

        // Restore corners and verify order is maintained
        for &corner in &corners {
            empty_list.restore(corner);
        }
        assert_eq!(empty_list.first(), Square::A1);
        assert_eq!(empty_list.count, 60);
    }

    #[test]
    fn test_parity_specific_quadrants() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);
        let initial_parity = empty_list.parity;

        // Test removing a few squares from quadrant 1 (top-left)
        empty_list.remove(Square::A1); // quadrant 1
        empty_list.remove(Square::B1); // quadrant 1

        // Removing 2 squares from quadrant 1: parity changes by 1^1 = 0
        assert_eq!(empty_list.parity, initial_parity);

        // Remove one more from quadrant 1
        empty_list.remove(Square::C1); // quadrant 1

        // Now we've removed 3 squares: 1^1^1 = 1, so parity should change by 1
        assert_eq!(empty_list.parity, initial_parity ^ 1);

        // Test removing from different quadrants
        empty_list.remove(Square::E1); // quadrant 2

        // Parity change: (1^1^1) ^ 2 = 1^2 = 3
        assert_eq!(empty_list.parity, initial_parity ^ 3);
    }

    #[test]
    fn test_strategic_ordering_corners_first() {
        let board = Board::new();
        let empty_list = EmptyList::new(&board);

        // First four squares should be corners
        let corners = [Square::A1, Square::H1, Square::A8, Square::H8];
        let mut current = empty_list.first();

        for &expected_corner in &corners {
            assert_eq!(current, expected_corner);
            current = empty_list.next(current);
        }
    }

    #[test]
    fn test_list_traversal_completeness() {
        let board = Board::new();
        let empty_list = EmptyList::new(&board);

        let mut visited_squares = [false; 64];
        let mut current = empty_list.first();
        let mut count = 0;

        while current != Square::None {
            let sq_idx = current.index();
            assert!(!visited_squares[sq_idx], "Square visited twice: {current:?}");
            visited_squares[sq_idx] = true;
            current = empty_list.next(current);
            count += 1;
        }

        assert_eq!(count, 60);

        // Count visited squares
        let visited_count = visited_squares.iter().filter(|&&v| v).count();
        assert_eq!(visited_count, 60);
    }

    #[test]
    fn test_remove_middle_squares() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        // Remove some middle squares from the presorted order
        empty_list.remove(Square::C3); // 13th in presorted order
        empty_list.remove(Square::F3); // 14th in presorted order

        // Verify list integrity by traversing
        let mut current = empty_list.first();
        let mut count = 0;
        let mut found_c3 = false;
        let mut found_f3 = false;

        while current != Square::None {
            if current == Square::C3 { found_c3 = true; }
            if current == Square::F3 { found_f3 = true; }
            current = empty_list.next(current);
            count += 1;
        }

        assert_eq!(count, 58); // 60 - 2 removed squares
        assert!(!found_c3, "C3 should not be in list after removal");
        assert!(!found_f3, "F3 should not be in list after removal");
    }

    #[test]
    fn test_clone_functionality() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        // Modify the original list
        empty_list.remove(Square::A1);
        empty_list.remove(Square::H8);

        // Clone the list
        let cloned_list = empty_list.clone();

        // Verify clone has same properties
        assert_eq!(cloned_list.count, empty_list.count);
        assert_eq!(cloned_list.parity, empty_list.parity);
        assert_eq!(cloned_list.first(), empty_list.first());

        // Verify independence - modify original
        empty_list.remove(Square::C1);

        // Clone should be unchanged
        assert_ne!(cloned_list.count, empty_list.count);
    }

    #[test]
    fn test_presorted_array_completeness() {
        let mut squares_seen = [false; 64];

        // Mark all squares in PRESORTED
        for &sq in PRESORTED.iter() {
            let sq_idx = sq.index();
            assert!(!squares_seen[sq_idx], "Duplicate square in PRESORTED: {sq:?}");
            squares_seen[sq_idx] = true;
        }

        // Should have exactly 64 squares (A1 to H8, excluding None)
        let count = squares_seen.iter().filter(|&&seen| seen).count();
        assert_eq!(count, 64);

        // Every square from 0-63 should be present
        for (idx, &seen) in squares_seen.iter().enumerate() {
            assert!(seen, "Square index {idx} missing from PRESORTED");
        }
    }
}
