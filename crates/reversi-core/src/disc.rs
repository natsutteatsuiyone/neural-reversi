//! Disc representation for the Reversi board.

/// Represents a disc color on the board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Disc {
    /// An empty spot on the board.
    Empty,
    /// A black disc.
    Black,
    /// A white disc.
    White,
}

impl Disc {
    /// Returns the character representation (`'-'`, `'X'`, or `'O'`).
    pub fn to_char(self) -> char {
        match self {
            Disc::Empty => '-',
            Disc::Black => 'X',
            Disc::White => 'O',
        }
    }

    /// Returns the opposite disc (`Empty` maps to itself).
    pub fn opposite(self) -> Disc {
        match self {
            Disc::Black => Disc::White,
            Disc::White => Disc::Black,
            Disc::Empty => Disc::Empty,
        }
    }
}
