//! Disc representation for the Reversi board.

/// Represents a disc in the game.
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
    /// Converts the disc to its corresponding character representation.
    ///
    /// # Returns
    ///
    /// * `'-'` for `Disc::Empty`
    /// * `'X'` for `Disc::Black`
    /// * `'O'` for `Disc::White`
    pub fn to_char(self) -> char {
        match self {
            Disc::Empty => '-',
            Disc::Black => 'X',
            Disc::White => 'O',
        }
    }

    /// Returns the opposite disc.
    ///
    /// # Returns
    ///
    /// * `Disc::White` for `Disc::Black`
    /// * `Disc::Black` for `Disc::White`
    /// * `Disc::Empty` for `Disc::Empty`
    pub fn opposite(self) -> Disc {
        match self {
            Disc::Black => Disc::White,
            Disc::White => Disc::Black,
            Disc::Empty => Disc::Empty,
        }
    }
}
