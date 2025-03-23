use std::str::FromStr;
use std::fmt;

/// Represents a square on a reversi, ranging from A1 to H8.
///
/// Each variant corresponds to a specific square on the board.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(usize)]
#[rustfmt::skip]
pub enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    None,
}

impl Square {
    /// Converts the `Square` into a 64-bit unsigned integer representation.
    ///
    /// # Returns
    ///
    /// A `u64` value representing the bit position of the `Square`.
    #[inline]
    pub fn bitboard(self) -> u64 {
        debug_assert!((self as usize) < 64, "Index out of bounds for Square enum. self: {:?}", self);
        1 << self as usize
    }

    /// Converts a `usize` value into a `Square` enum.
    ///
    /// # Arguments
    ///
    /// * `index` - The `usize` value to convert.
    ///
    /// # Returns
    ///
    /// The corresponding `Square` variant.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `index` is 64 or higher.
    #[inline]
    pub fn from_usize_unchecked(index: usize) -> Square {
        debug_assert!(index < 65, "Index out of bounds for Square enum. index: {:?}", index);
        unsafe { std::mem::transmute(index) }
    }

    /// Returns an iterator over all the squares on the board.
    ///
    /// # Returns
    /// An iterator over all the squares on the board.
    #[inline]
    pub fn iter() -> impl Iterator<Item = Square>  {
        (0..64).map(Square::from_usize_unchecked)
    }

    /// Validates a string to see if it is a valid square representation.
    ///
    /// # Arguments
    ///
    /// * `sq_str` - The string to validate.
    ///
    /// # Returns
    ///
    /// `true` if the string is a valid square representation, `false` otherwise.
    fn validate_str(sq_str: &str) -> bool {
        if sq_str.len() != 2 {
            return false;
        }
        let file = sq_str.chars().nth(0).unwrap();
        let rank = sq_str.chars().nth(1).unwrap();
        if !('a'..='h').contains(&file) {
            return false;
        }
        if !('1'..='8').contains(&rank) {
            return false;
        }
        true
    }

}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseSquareError;

impl FromStr for Square {
    type Err = ParseSquareError;

    /// Parses a string into a `Square` enum.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse.
    ///
    /// # Returns
    ///
    /// A `Square` enum if the string is a valid square representation.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_lowercase();
        if !Self::validate_str(&s) {
            return Err(ParseSquareError);
        }

        let file = s.chars().nth(0).unwrap();
        let rank = s.chars().nth(1).unwrap();
        let file = file.to_ascii_lowercase() as u8 - b'a';
        let rank = rank as u8 - b'1';
        Ok(Square::from_usize_unchecked(rank as usize * 8 + file as usize))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Square::None {
            return write!(f, "None");
        }

        let file = ((*self as usize) % 8) as u8 + b'a';
        let rank = ((*self as usize) / 8) as u8 + b'1';

        write!(f, "{}{}", file as char, rank as char)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_bitboard() {
        assert_eq!(Square::A1.bitboard(), 1);
        assert_eq!(Square::H8.bitboard(), 0x8000000000000000);
    }

    #[test]
    fn test_from_usize() {
        assert_eq!(Square::from_usize_unchecked(0), Square::A1);
        assert_eq!(Square::from_usize_unchecked(1), Square::B1);
        assert_eq!(Square::from_usize_unchecked(2), Square::C1);
        assert_eq!(Square::from_usize_unchecked(3), Square::D1);
        assert_eq!(Square::from_usize_unchecked(4), Square::E1);
        assert_eq!(Square::from_usize_unchecked(5), Square::F1);
        assert_eq!(Square::from_usize_unchecked(6), Square::G1);
        assert_eq!(Square::from_usize_unchecked(7), Square::H1);
        assert_eq!(Square::from_usize_unchecked(8), Square::A2);
        assert_eq!(Square::from_usize_unchecked(63), Square::H8);
    }

    #[test]
    fn test_iter() {
        let squares: Vec<Square> = Square::iter().collect();
        assert_eq!(squares.len(), 64);
        assert_eq!(squares[0], Square::from_usize_unchecked(0));
        assert_eq!(squares[63], Square::from_usize_unchecked(63));
    }

    #[test]
    fn test_square_from_str() {
        assert_eq!(Square::from_str("a1").unwrap(), Square::A1);
        assert_eq!(Square::from_str("h8").unwrap(), Square::H8);
        assert_eq!(Square::from_str("A1").unwrap(), Square::A1);
        assert_eq!(Square::from_str("H8").unwrap(), Square::H8);
        assert!(Square::from_str("i1").is_err());
        assert!(Square::from_str("a9").is_err());
        assert!(Square::from_str("").is_err());
        assert!(Square::from_str("a").is_err());
        assert!(Square::from_str("abc").is_err());
    }
}
