use crate::board::Board;
use crate::square::Square;

#[rustfmt::skip]
pub const QUADRANT_ID: [usize; 64] = [
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 2, 2, 2, 2,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
    4, 4, 4, 4, 8, 8, 8, 8,
];

/// Represents a single empty square in the list.
#[derive(Clone, Copy)]
struct EmptySquare {
    next: Square,
    prev: Square,
    quad_id: usize,
}

/// Manages a linked list of empty squares on the board.
///
/// The `EmptyList` maintains an array of `EmptySquare` structs to efficiently
/// track and manipulate empty squares in a predefined order.
#[derive(Clone)]
pub struct EmptyList {
    squares: [EmptySquare; 65],
    pub count: u32,
    pub parity: usize,
}

/// A presorted list of squares based on strategic importance.
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
        let mut parity: usize = 0;
        let mut squares = [EmptySquare {
            next: Square::None,
            prev: Square::None,
            quad_id: 0,
        }; 65];

        let mut prev_sq = Square::None;
        for &sq in PRESORTED.iter() {
            if board.is_square_empty(sq) {
                let sq_idx = sq as usize;
                squares[prev_sq as usize].next = sq;
                squares[sq_idx].prev = prev_sq;
                squares[sq_idx].quad_id = QUADRANT_ID[sq_idx];
                parity ^= squares[sq_idx].quad_id;
                prev_sq = sq;
                count += 1;
            }
        }

        Self {
            squares,
            count,
            parity,
        }
    }

    /// Retrieves the first empty square in the list.
    ///
    /// # Returns
    ///
    /// The `Square` representing the first empty square.
    #[inline]
    pub fn first(&self) -> Square {
        self.squares[Square::None as usize].next
    }

    pub fn first_with_quad_id(&self) -> (Square, usize) {
        (
            self.squares[Square::None as usize].next,
            self.squares[Square::None as usize].quad_id,
        )
    }

    /// Retrieves the next empty square following the given square.
    ///
    /// # Arguments
    ///
    /// * `sq` - The current `Square`.
    ///
    /// # Returns
    ///
    /// The `Square` representing the next empty square, or `Square::NONE` if `sq` is the last square.
    #[inline]
    pub fn next(&self, sq: Square) -> Square {
        self.squares[sq as usize].next
    }

    #[inline]
    pub fn next_with_quad_id(&self, sq: Square) -> (Square, usize) {
        (
            self.squares[sq as usize].next,
            self.squares[sq as usize].quad_id,
        )
    }

    /// Removes a square from the empty list.
    ///
    /// # Arguments
    ///
    /// * `sq` - The `Square` to remove.
    ///
    /// # Behavior
    ///
    /// Updates the `next` and `prev` references of adjacent squares to exclude `sq`,
    /// toggles the `parity`, and decrements the `count` of empty squares.
    #[inline]
    pub fn remove(&mut self, sq: Square) {
        let square = self.squares[sq as usize];
        let prev = square.prev;
        let next = square.next;
        self.squares[prev as usize].next = next;
        self.squares[next as usize].prev = prev;
        self.parity ^= square.quad_id;
        self.count -= 1;
    }

    /// Restores a previously removed square to the empty list.
    ///
    /// # Arguments
    ///
    /// * `sq` - The `Square` to restore.
    ///
    /// # Behavior
    ///
    /// Updates the `next` and `prev` references of adjacent squares to include `sq`,
    /// toggles the `parity`, and increments the `count` of empty squares.
    #[inline]
    pub fn restore(&mut self, sq: Square) {
        let square = self.squares[sq as usize];
        let prev = square.prev;
        let next = square.next;
        self.squares[prev as usize].next = sq;
        self.squares[next as usize].prev = sq;
        self.parity ^= square.quad_id;
        self.count += 1;
    }

    #[inline]
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
}
