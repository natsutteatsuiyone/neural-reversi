use std::fmt;
use std::str::FromStr;

/// Represents a square on a reversi board, ranging from A1 to H8.
///
/// The reversi board uses algebraic notation where files (columns) are labeled A-H
/// and ranks (rows) are labeled 1-8. The board is indexed as follows:
///
/// ```text
///   A B C D E F G H
/// 1 00 01 02 03 04 05 06 07
/// 2 08 09 10 11 12 13 14 15
/// 3 16 17 18 19 20 21 22 23
/// 4 24 25 26 27 28 29 30 31
/// 5 32 33 34 35 36 37 38 39
/// 6 40 41 42 43 44 45 46 47
/// 7 48 49 50 51 52 53 54 55
/// 8 56 57 58 59 60 61 62 63
/// ```
///
/// Each variant corresponds to a specific square on the board, with an additional
/// `None` variant representing an invalid or unspecified square.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
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

/// Constants for board dimensions
pub const BOARD_SIZE: usize = 8;
pub const TOTAL_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;

impl Square {
    /// Converts the `Square` into a u64 representation.
    ///
    /// # Returns
    ///
    /// A `u64` value with a single bit set at the position corresponding to this square.
    /// For example, A1 returns 0x1, B1 returns 0x2, H8 returns 0x8000000000000000.
    #[inline]
    pub fn bitboard(self) -> u64 {
        debug_assert!(
            (self as usize) < 64,
            "Index out of bounds for Square enum. self: {self:?}"
        );
        1 << self as u8
    }

    /// Converts the `Square` into a `usize` index.
    ///
    /// # Returns
    ///
    /// A `usize` value representing the index of the `Square` (0-63 for valid squares,
    /// 64 for `Square::None`).
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Converts a `u8` value into a `Square` enum without bounds checking.
    ///
    /// # Arguments
    /// * `index` - The `u8` value to convert (0-63 for valid squares, 64 for `None`).
    ///
    /// # Returns
    /// The corresponding `Square` variant.
    #[inline]
    pub fn from_u8_unchecked(index: u8) -> Square {
        debug_assert!(
            index <= 64,
            "Index out of bounds for Square enum. index: {index:?}"
        );
        unsafe { std::mem::transmute(index) }
    }

    /// Safely converts a `u8` value into a `Square` enum.
    ///
    /// # Arguments
    /// * `index` - The `u8` value to convert.
    ///
    /// # Returns
    /// `Some(Square)` if the index is valid (0-64), `None` otherwise.
    #[inline]
    pub fn from_u8(index: u8) -> Option<Square> {
        if index <= 64 {
            Some(Square::from_u8_unchecked(index))
        } else {
            None
        }
    }

    /// Converts a `u32` value into a `Square` enum without bounds checking.
    ///
    /// # Arguments
    /// * `index` - The `u32` value to convert.
    ///
    /// # Returns
    /// The corresponding `Square` variant.
    #[inline]
    pub fn from_u32_unchecked(index: u32) -> Square {
        debug_assert!(
            index <= 64,
            "Index out of bounds for Square enum. index: {index:?}"
        );
        unsafe { std::mem::transmute(index as u8) }
    }

    /// Safely converts a `u32` value into a `Square` enum.
    ///
    /// # Arguments
    /// * `index` - The `u32` value to convert.
    ///
    /// # Returns
    /// `Some(Square)` if the index is valid (0-64), `None` otherwise.
    #[inline]
    pub fn from_u32(index: u32) -> Option<Square> {
        if index <= 64 {
            Some(Square::from_u32_unchecked(index))
        } else {
            None
        }
    }

    /// Converts a `usize` value into a `Square` enum without bounds checking.
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
    /// Panics in debug mode if `index` > 64.
    #[inline]
    pub fn from_usize_unchecked(index: usize) -> Square {
        debug_assert!(
            index <= 64,
            "Index out of bounds for Square enum. index: {index:?}"
        );
        unsafe { std::mem::transmute(index as u8) }
    }

    /// Safely converts a `usize` value into a `Square` enum.
    ///
    /// # Arguments
    /// * `index` - The `usize` value to convert.
    ///
    /// # Returns
    /// `Some(Square)` if the index is valid (0-64), `None` otherwise.
    #[inline]
    pub fn from_usize(index: usize) -> Option<Square> {
        if index <= 64 {
            Some(Square::from_usize_unchecked(index))
        } else {
            None
        }
    }

    /// Returns the file (column) of this square.
    ///
    /// # Returns
    ///
    /// The file index (0-7) where 0 represents file A and 7 represents file H.
    ///
    /// # Panics
    ///
    /// Panics if called on `Square::None`.
    #[inline]
    pub fn file(self) -> usize {
        assert!(self != Square::None, "Square::file called on Square::None");
        self.index() % BOARD_SIZE
    }

    /// Returns the rank (row) of this square.
    ///
    /// # Returns
    ///
    /// The rank index (0-7) where 0 represents rank 1 and 7 represents rank 8.
    ///
    /// # Panics
    ///
    /// Panics if called on `Square::None`.
    #[inline]
    pub fn rank(self) -> usize {
        assert!(self != Square::None, "Square::rank called on Square::None");
        self.index() / BOARD_SIZE
    }

    /// Creates a `Square` from file and rank coordinates.
    ///
    /// # Arguments
    ///
    /// * `file` - The file index (0-7) where 0 is file A and 7 is file H.
    /// * `rank` - The rank index (0-7) where 0 is rank 1 and 7 is rank 8.
    ///
    /// # Returns
    ///
    /// The corresponding `Square` variant.
    ///
    /// # Panics
    ///
    /// Panics if either `file` or `rank` is >= 8.
    pub fn from_file_rank(file: u8, rank: u8) -> Square {
        assert!(file < BOARD_SIZE as u8, "Invalid file: {file}");
        assert!(rank < BOARD_SIZE as u8, "Invalid rank: {rank}");
        Self::from_usize_unchecked(rank as usize * BOARD_SIZE + file as usize)
    }

    /// Returns an iterator over all 64 valid squares on the board.
    ///
    /// The iterator yields squares in order from A1 to H8, following the
    /// index order (0-63). This does not include `Square::None`.
    ///
    /// # Returns
    /// An iterator that yields all 64 board squares.
    #[inline]
    pub fn iter() -> impl Iterator<Item = Square> {
        (0..TOTAL_SQUARES as u8).map(Square::from_u8_unchecked)
    }
}

// We want Square::None as the default value, not the first variant (A1)
// which would be chosen by #[derive(Default)]
#[allow(clippy::derivable_impls)]
impl Default for Square {
    fn default() -> Self {
        Square::None
    }
}

/// Error type for square-related operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SquareError {
    /// Invalid square string format (must be 2 characters)
    InvalidFormat,
    /// Invalid file character (must be a-h or A-H)
    InvalidFile(char),
    /// Invalid rank character (must be 1-8)
    InvalidRank(char),
}

impl fmt::Display for SquareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SquareError::InvalidFormat => write!(
                f,
                "Invalid square format: must be 2 characters (e.g., 'a1')"
            ),
            SquareError::InvalidFile(c) => write!(f, "Invalid file '{c}': must be a-h or A-H"),
            SquareError::InvalidRank(c) => write!(f, "Invalid rank '{c}': must be 1-8"),
        }
    }
}

impl std::error::Error for SquareError {}

/// Alias for the old error type name for backward compatibility
pub type ParseSquareError = SquareError;

impl FromStr for Square {
    type Err = SquareError;

    /// Parses a string into a `Square` enum.
    ///
    /// The string must be in algebraic notation (e.g., "a1", "h8").
    /// Both uppercase and lowercase letters are accepted.
    ///
    /// # Arguments
    ///
    /// * `s` - The string to parse.
    ///
    /// # Returns
    ///
    /// * `Ok(Square)` - The parsed square if the string is valid.
    /// * `Err(ParseSquareError)` - If the string is not a valid square representation.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.len() != 2 {
            return Err(SquareError::InvalidFormat);
        }

        let chars: Vec<char> = s.chars().collect();
        let file_char = chars[0].to_ascii_lowercase();
        let rank_char = chars[1];

        if !('a'..='h').contains(&file_char) {
            return Err(SquareError::InvalidFile(chars[0]));
        }

        if !('1'..='8').contains(&rank_char) {
            return Err(SquareError::InvalidRank(rank_char));
        }

        let file = file_char as u8 - b'a';
        let rank = rank_char as u8 - b'1';
        Ok(Square::from_usize_unchecked(
            rank as usize * BOARD_SIZE + file as usize,
        ))
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

        // Test specific error types
        match Square::from_str("").unwrap_err() {
            SquareError::InvalidFormat => (),
            _ => panic!("Expected InvalidFormat error"),
        }
        match Square::from_str("z1").unwrap_err() {
            SquareError::InvalidFile('z') => (),
            _ => panic!("Expected InvalidFile error"),
        }
        match Square::from_str("a0").unwrap_err() {
            SquareError::InvalidRank('0') => (),
            _ => panic!("Expected InvalidRank error"),
        }
    }

    #[test]
    fn test_safe_conversions() {
        assert_eq!(Square::from_u8(0), Some(Square::A1));
        assert_eq!(Square::from_u8(63), Some(Square::H8));
        assert_eq!(Square::from_u8(64), Some(Square::None));
        assert_eq!(Square::from_u8(65), None);

        assert_eq!(Square::from_u32(0), Some(Square::A1));
        assert_eq!(Square::from_u32(64), Some(Square::None));
        assert_eq!(Square::from_u32(65), None);

        assert_eq!(Square::from_usize(63), Some(Square::H8));
        assert_eq!(Square::from_usize(64), Some(Square::None));
        assert_eq!(Square::from_usize(65), None);
    }

    #[test]
    fn test_index() {
        assert_eq!(Square::A1.index(), 0);
        assert_eq!(Square::B1.index(), 1);
        assert_eq!(Square::H1.index(), 7);
        assert_eq!(Square::A2.index(), 8);
        assert_eq!(Square::D4.index(), 27); // 3 * 8 + 3
        assert_eq!(Square::E5.index(), 36); // 4 * 8 + 4
        assert_eq!(Square::H8.index(), 63);
        assert_eq!(Square::None.index(), 64);
    }

    #[test]
    fn test_file() {
        assert_eq!(Square::A1.file(), 0);
        assert_eq!(Square::B1.file(), 1);
        assert_eq!(Square::C1.file(), 2);
        assert_eq!(Square::D1.file(), 3);
        assert_eq!(Square::E1.file(), 4);
        assert_eq!(Square::F1.file(), 5);
        assert_eq!(Square::G1.file(), 6);
        assert_eq!(Square::H1.file(), 7);

        // Test various ranks
        assert_eq!(Square::A8.file(), 0);
        assert_eq!(Square::H8.file(), 7);
        assert_eq!(Square::D4.file(), 3);
        assert_eq!(Square::E5.file(), 4);
    }

    #[test]
    #[should_panic(expected = "Square::file called on Square::None")]
    fn test_file_panics_on_none() {
        let _ = Square::None.file();
    }

    #[test]
    fn test_rank() {
        assert_eq!(Square::A1.rank(), 0);
        assert_eq!(Square::B1.rank(), 0);
        assert_eq!(Square::H1.rank(), 0);

        assert_eq!(Square::A2.rank(), 1);
        assert_eq!(Square::A3.rank(), 2);
        assert_eq!(Square::A4.rank(), 3);
        assert_eq!(Square::A5.rank(), 4);
        assert_eq!(Square::A6.rank(), 5);
        assert_eq!(Square::A7.rank(), 6);
        assert_eq!(Square::A8.rank(), 7);

        // Test various files
        assert_eq!(Square::H8.rank(), 7);
        assert_eq!(Square::D4.rank(), 3);
        assert_eq!(Square::E5.rank(), 4);
    }

    #[test]
    #[should_panic(expected = "Square::rank called on Square::None")]
    fn test_rank_panics_on_none() {
        let _ = Square::None.rank();
    }

    #[test]
    fn test_from_file_rank() {
        assert_eq!(Square::from_file_rank(0, 0), Square::A1);
        assert_eq!(Square::from_file_rank(1, 0), Square::B1);
        assert_eq!(Square::from_file_rank(7, 0), Square::H1);
        assert_eq!(Square::from_file_rank(0, 1), Square::A2);
        assert_eq!(Square::from_file_rank(0, 7), Square::A8);
        assert_eq!(Square::from_file_rank(7, 7), Square::H8);
        assert_eq!(Square::from_file_rank(3, 3), Square::D4);
        assert_eq!(Square::from_file_rank(4, 4), Square::E5);

        // Test all squares
        for square in Square::iter() {
            let file = square.file();
            let rank = square.rank();
            assert_eq!(Square::from_file_rank(file as u8, rank as u8), square);
        }
    }

    #[test]
    #[should_panic(expected = "Invalid file: 8")]
    fn test_from_file_rank_invalid_file() {
        let _ = Square::from_file_rank(8, 0);
    }

    #[test]
    #[should_panic(expected = "Invalid rank: 8")]
    fn test_from_file_rank_invalid_rank() {
        let _ = Square::from_file_rank(0, 8);
    }

    #[test]
    fn test_default() {
        assert_eq!(Square::default(), Square::None);
        let square: Square = Default::default();
        assert_eq!(square, Square::None);
    }

    #[test]
    fn test_display() {
        assert_eq!(Square::A1.to_string(), "a1");
        assert_eq!(Square::B2.to_string(), "b2");
        assert_eq!(Square::C3.to_string(), "c3");
        assert_eq!(Square::D4.to_string(), "d4");
        assert_eq!(Square::E5.to_string(), "e5");
        assert_eq!(Square::F6.to_string(), "f6");
        assert_eq!(Square::G7.to_string(), "g7");
        assert_eq!(Square::H8.to_string(), "h8");
        assert_eq!(Square::None.to_string(), "None");

        // Test format! macro
        assert_eq!(format!("{}", Square::A1), "a1");
        assert_eq!(format!("{}", Square::H8), "h8");
        assert_eq!(format!("{}", Square::None), "None");
    }

    #[test]
    fn test_square_error_display() {
        assert_eq!(
            SquareError::InvalidFormat.to_string(),
            "Invalid square format: must be 2 characters (e.g., 'a1')"
        );
        assert_eq!(
            SquareError::InvalidFile('z').to_string(),
            "Invalid file 'z': must be a-h or A-H"
        );
        assert_eq!(
            SquareError::InvalidRank('9').to_string(),
            "Invalid rank '9': must be 1-8"
        );
    }

    #[test]
    fn test_from_str_edge_cases() {
        // Test with whitespace
        assert_eq!(Square::from_str(" a1 ").unwrap(), Square::A1);
        assert_eq!(Square::from_str("\ta1\n").unwrap(), Square::A1);

        // Test all valid squares
        for square in Square::iter() {
            let s = square.to_string();
            assert_eq!(Square::from_str(&s).unwrap(), square);
            // Test uppercase
            assert_eq!(Square::from_str(&s.to_uppercase()).unwrap(), square);
        }
    }

    #[test]
    fn test_unchecked_conversions() {
        // Test u8 unchecked
        assert_eq!(Square::from_u8_unchecked(0), Square::A1);
        assert_eq!(Square::from_u8_unchecked(63), Square::H8);
        assert_eq!(Square::from_u8_unchecked(64), Square::None);

        // Test u32 unchecked
        assert_eq!(Square::from_u32_unchecked(0), Square::A1);
        assert_eq!(Square::from_u32_unchecked(63), Square::H8);
        assert_eq!(Square::from_u32_unchecked(64), Square::None);
    }

    #[test]
    fn test_roundtrip_conversions() {
        // Test that index() and from_*_unchecked are inverses
        for i in 0..=64 {
            let square = Square::from_usize_unchecked(i);
            assert_eq!(square.index(), i);

            let square_u8 = Square::from_u8_unchecked(i as u8);
            assert_eq!(square_u8.index(), i);

            let square_u32 = Square::from_u32_unchecked(i as u32);
            assert_eq!(square_u32.index(), i);
        }
    }
}
