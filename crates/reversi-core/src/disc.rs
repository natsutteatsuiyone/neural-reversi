/// Represents a disc in the game.
///
/// The `Disc` enum has three variants:
///
/// * `Empty` - Represents an empty spot on the board.
/// * `Black` - Represents a black disc.
/// * `White` - Represents a white disc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Disc {
    Empty,
    Black,
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
    pub fn opposite(&self) -> Disc {
        match self {
            Disc::Black => Disc::White,
            Disc::White => Disc::Black,
            Disc::Empty => Disc::Empty,
        }
    }
}
