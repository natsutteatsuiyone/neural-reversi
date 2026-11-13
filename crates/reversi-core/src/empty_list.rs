//! Empty square list management.

use crate::board::Board;
use crate::constants::INITIAL_EMPTY_COUNT;
use crate::square::Square;

/// Returns the quadrant ID for a given square.
///
/// The board is divided into 4 quadrants for parity calculation:
/// - Top-left (A1-D4): bit mask 1
/// - Top-right (E1-H4): bit mask 2
/// - Bottom-left (A5-D8): bit mask 4
/// - Bottom-right (E5-H8): bit mask 8
///
/// Used for endgame parity optimization where different quadrant
/// combinations affect move ordering efficiency.
#[inline(always)]
fn get_quadrant_id(sq: Square) -> u8 {
    #[rustfmt::skip]
    const QUADRANT_ID: [u8; 64] = [
        1, 1, 1, 1, 2, 2, 2, 2,
        1, 1, 1, 1, 2, 2, 2, 2,
        1, 1, 1, 1, 2, 2, 2, 2,
        1, 1, 1, 1, 2, 2, 2, 2,
        4, 4, 4, 4, 8, 8, 8, 8,
        4, 4, 4, 4, 8, 8, 8, 8,
        4, 4, 4, 4, 8, 8, 8, 8,
        4, 4, 4, 4, 8, 8, 8, 8,
    ];
    // SAFETY: sq.index() returns 0..64 and QUADRANT_ID has 64 elements.
    unsafe { *QUADRANT_ID.get_unchecked(sq.index()) }
}

/// Strategic ordering of squares for optimal move generation.
#[rustfmt::skip]
const PRESORTED: [Square; 64] = [
    Square::A1, Square::H1, Square::A8, Square::H8,

    Square::C1, Square::F1, Square::A3, Square::H3,
    Square::A6, Square::H6, Square::C8, Square::F8,

    Square::C3, Square::F3, Square::C6, Square::F6,
    Square::D1, Square::E1, Square::A4, Square::H4,

    Square::A5, Square::H5, Square::D8, Square::E8,
    Square::D3, Square::E3, Square::C4, Square::F4,
    Square::C5, Square::F5, Square::D6, Square::E6,
    Square::D2, Square::E2,

    Square::B4, Square::G4, Square::B5, Square::G5,
    Square::D7, Square::E7, Square::C2, Square::F2,
    Square::B3, Square::G3, Square::B6, Square::G6,
    Square::C7, Square::F7,

    Square::B1, Square::G1, Square::A2, Square::H2,
    Square::A7, Square::H7, Square::B8, Square::G8,
    Square::B2, Square::G2, Square::B7, Square::G7,
    Square::D4, Square::E4, Square::D5, Square::E5,
];

/// Index of the head node, which follows the 64 square nodes.
const HEAD: u8 = 64;

/// Node in the circular doubly-linked list of empty squares.
///
/// Links are node indices: a square index, or [`HEAD`]. Nodes of squares
/// that were never empty link to [`HEAD`], so `contains` rejects them.
#[derive(Clone, Copy)]
struct EmptyNode {
    next: u8,
    prev: u8,
}

/// Doubly-linked list of empty squares for move generation.
#[derive(Clone)]
pub struct EmptyList {
    /// Square nodes indexed by square position, then the [`HEAD`] node.
    nodes: [EmptyNode; 65],
    /// Number of empty squares currently in the list.
    count: u32,
    /// XOR of all quadrant IDs (used for endgame parity optimization)
    parity: u8,
}

impl EmptyList {
    /// Creates a new [`EmptyList`] by scanning the board for empty squares in presorted order.
    pub fn new(board: &Board) -> Self {
        let mut count = 0;
        let mut parity: u8 = 0;
        let mut nodes = [EmptyNode {
            next: HEAD,
            prev: HEAD,
        }; 65];

        let mut prev = HEAD;
        let empty_board = board.empty_squares();
        for &sq in PRESORTED.iter() {
            if empty_board.contains(sq) {
                nodes[prev as usize].next = sq as u8;
                nodes[sq.index()].prev = prev;
                parity ^= get_quadrant_id(sq);
                prev = sq as u8;
                count += 1;
            }
        }
        nodes[HEAD as usize].prev = prev;

        Self {
            nodes,
            count,
            parity,
        }
    }

    /// Returns the number of empty squares currently in the list.
    #[inline(always)]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the parity value (XOR of all quadrant IDs).
    ///
    /// Each bit (1, 2, 4, 8) represents a quadrant with an odd number of empties.
    /// When parity is 0, all quadrants have even empty counts. A non-zero bit
    /// indicates the corresponding quadrant has an odd number of empties, which
    /// affects move ordering heuristics in endgame search.
    #[inline(always)]
    pub fn parity(&self) -> u8 {
        self.parity
    }

    /// Returns the first empty square.
    ///
    /// The list must not be empty.
    #[inline(always)]
    pub fn first(&self) -> Square {
        self.square_after(HEAD)
    }

    /// Returns the first empty square and its quadrant ID.
    ///
    /// The list must not be empty.
    #[inline(always)]
    pub fn first_and_quad_id(&self) -> (Square, u8) {
        let sq = self.first();
        (sq, get_quadrant_id(sq))
    }

    /// Returns the next empty square after `sq`.
    ///
    /// `sq` must be in the list and must not be its last square.
    #[inline(always)]
    pub fn next(&self, sq: Square) -> Square {
        debug_assert!(self.contains(sq), "Square {sq:?} is not in the empty list");
        self.square_after(sq as u8)
    }

    #[inline(always)]
    fn square_after(&self, node: u8) -> Square {
        // SAFETY: node is a square index or HEAD, both < 65 = self.nodes.len().
        let next = unsafe { self.nodes.get_unchecked(node as usize).next };
        debug_assert!(next != HEAD, "no empty square follows node {node}");
        // SAFETY: next != HEAD, so it is a square index.
        unsafe { Square::from_u8_unchecked(next) }
    }

    fn contains(&self, sq: Square) -> bool {
        self.nodes[self.nodes[sq.index()].prev as usize].next == sq as u8
    }

    /// Returns the next empty square after `sq` and its quadrant ID.
    ///
    /// `sq` must be in the list and must not be its last square.
    #[inline(always)]
    pub fn next_and_quad_id(&self, sq: Square) -> (Square, u8) {
        let next_sq = self.next(sq);
        let quad_id = get_quadrant_id(next_sq);
        (next_sq, quad_id)
    }

    /// Removes `sq` from the empty list.
    #[inline(always)]
    pub fn remove(&mut self, sq: Square) {
        debug_assert!(self.count > 0, "Cannot remove from an empty list");
        debug_assert!(self.contains(sq), "Square {sq:?} is not in the empty list");

        // SAFETY: square indices and links are all < 65 = self.nodes.len().
        // The node is copied out before any mutable access, so no aliasing.
        unsafe {
            let node = *self.nodes.get_unchecked(sq.index());
            self.nodes.get_unchecked_mut(node.prev as usize).next = node.next;
            self.nodes.get_unchecked_mut(node.next as usize).prev = node.prev;
        }
        self.parity ^= get_quadrant_id(sq);
        self.count -= 1;
    }

    /// Restores a previously removed square to the empty list.
    #[inline(always)]
    pub fn restore(&mut self, sq: Square) {
        debug_assert!(
            self.count < INITIAL_EMPTY_COUNT as u32,
            "Cannot restore: list is already full"
        );
        debug_assert!(
            !self.contains(sq),
            "Square {sq:?} is already in the empty list"
        );

        // SAFETY: square indices and links are all < 65 = self.nodes.len().
        // The node is copied out before any mutable access, so no aliasing.
        unsafe {
            let node = *self.nodes.get_unchecked(sq.index());
            self.nodes.get_unchecked_mut(node.prev as usize).next = sq as u8;
            self.nodes.get_unchecked_mut(node.next as usize).prev = sq as u8;
        }
        self.parity ^= get_quadrant_id(sq);
        self.count += 1;
    }

    /// Returns the current ply (number of disc placements).
    ///
    /// In Reversi, the game starts with 4 discs placed and 60 empty squares.
    /// Each non-pass move fills one empty square, so ply = [`INITIAL_EMPTY_COUNT`] - empty_count.
    #[inline(always)]
    pub fn ply(&self) -> usize {
        (INITIAL_EMPTY_COUNT as u32 - self.count) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_list_remove_restore() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        empty_list.remove(Square::A1);
        assert_eq!(empty_list.count(), 59);
        assert_eq!(empty_list.first(), Square::H1);
        assert_eq!(empty_list.next(Square::H1), Square::A8);

        empty_list.restore(Square::A1);
        assert_eq!(empty_list.count(), 60);
        assert_eq!(empty_list.first(), Square::A1);
        assert_eq!(empty_list.next(Square::A1), Square::H1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "is not in the empty list")]
    fn next_rejects_square_filled_at_construction() {
        let empty_list = EmptyList::new(&Board::from_bitboards(Square::A1.bitboard(), 0u64));
        let _ = empty_list.next(Square::A1);
    }

    #[test]
    fn test_presorted_order() {
        let board = Board::new();
        let empty_list = EmptyList::new(&board);

        let mut current = empty_list.first();
        for (count, &sq) in PRESORTED.iter().take(60).enumerate() {
            if count > 0 {
                current = empty_list.next(current);
            }
            assert_eq!(current, sq);
        }
        assert_eq!(empty_list.count(), 60);
    }

    #[test]
    fn test_empty_list_parity() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        assert_eq!(empty_list.parity(), 15);

        empty_list.remove(Square::A1);
        assert_eq!(empty_list.parity(), 14);

        empty_list.restore(Square::A1);
        assert_eq!(empty_list.parity(), 15);

        empty_list.remove(Square::H8);
        assert_eq!(empty_list.parity(), 7);

        empty_list.remove(Square::A8);
        assert_eq!(empty_list.parity(), 3);

        empty_list.remove(Square::H1);
        assert_eq!(empty_list.parity(), 1);

        empty_list.remove(Square::A1);
        assert_eq!(empty_list.parity(), 0);
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
        assert_eq!(empty_list.count(), 56);

        // Restore corners in reverse order (LIFO) and verify order is maintained
        for &corner in corners.iter().rev() {
            empty_list.restore(corner);
        }
        assert_eq!(empty_list.first(), Square::A1);
        assert_eq!(empty_list.count(), 60);
    }

    #[test]
    fn test_remove_middle_squares() {
        let board = Board::new();
        let mut empty_list = EmptyList::new(&board);

        // Remove some middle squares from the presorted order
        empty_list.remove(Square::C3); // 13th in presorted order
        empty_list.remove(Square::F3); // 14th in presorted order

        let mut current = empty_list.first();
        let mut visited = vec![current];
        for _ in 1..empty_list.count() {
            current = empty_list.next(current);
            visited.push(current);
        }

        assert_eq!(visited.len(), 58);
        assert!(!visited.contains(&Square::C3));
        assert!(!visited.contains(&Square::F3));
    }

    #[test]
    fn test_presorted_array_completeness() {
        let mut squares_seen = [false; 64];

        // Mark all squares in PRESORTED
        for &sq in PRESORTED.iter() {
            let sq_idx = sq.index();
            assert!(
                !squares_seen[sq_idx],
                "Duplicate square in PRESORTED: {sq:?}"
            );
            squares_seen[sq_idx] = true;
        }

        // Should have exactly 64 squares (A1 to H8)
        let count = squares_seen.iter().filter(|&&seen| seen).count();
        assert_eq!(count, 64);

        // Every square from 0-63 should be present
        for (idx, &seen) in squares_seen.iter().enumerate() {
            assert!(seen, "Square index {idx} missing from PRESORTED");
        }
    }
}
