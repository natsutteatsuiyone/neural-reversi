/// Represents a piece in the game.
///
/// The `Piece` enum has three variants:
///
/// * `Empty` - Represents an empty spot on the board.
/// * `Black` - Represents a black piece.
/// * `White` - Represents a white piece.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Piece {
    Empty,
    Black,
    White,
}

impl Piece {
    /// Converts the piece to its corresponding character representation.
    ///
    /// # Returns
    ///
    /// * `'-'` for `Piece::Empty`
    /// * `'X'` for `Piece::Black`
    /// * `'O'` for `Piece::White`
    pub fn to_char(self) -> char {
        match self {
            Piece::Empty => '-',
            Piece::Black => 'X',
            Piece::White => 'O',
        }
    }

    /// Returns the opposite piece.
    ///
    /// # Returns
    ///
    /// * `Piece::White` for `Piece::Black`
    /// * `Piece::Black` for `Piece::White`
    /// * `Piece::Empty` for `Piece::Empty`
    pub fn opposite(&self) -> Piece {
        match self {
            Piece::Black => Piece::White,
            Piece::White => Piece::Black,
            Piece::Empty => Piece::Empty,
        }
    }
}
