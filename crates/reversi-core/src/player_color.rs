//! Player colors for the side to move.

use std::fmt;
use std::str::FromStr;

use crate::disc::Disc;

/// The color of a player, such as the side to move.
///
/// Unlike [`Disc`], it has no empty state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PlayerColor {
    /// The black player; numbered like [`Disc::Black`].
    Black = 1,
    /// The white player; numbered like [`Disc::White`].
    White = 2,
}

impl PlayerColor {
    /// Returns the other player's color.
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }

    /// Returns the character representation (`'X'` or `'O'`).
    #[inline]
    pub fn to_char(self) -> char {
        Disc::from(self).to_char()
    }

    /// Returns the lowercase color name (`"black"` or `"white"`).
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Black => "black",
            Self::White => "white",
        }
    }

    /// Returns `0` for [`Self::Black`] and `1` for [`Self::White`].
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize - 1
    }
}

impl fmt::Display for PlayerColor {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error returned when parsing a string into a [`PlayerColor`] fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerColorError;

impl fmt::Display for PlayerColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid player color (expected 'b', 'w', 'black', or 'white')")
    }
}

impl std::error::Error for PlayerColorError {}

impl FromStr for PlayerColor {
    type Err = PlayerColorError;

    /// Parses `"b"`, `"black"`, `"w"`, or `"white"` (ASCII case-insensitive,
    /// surrounding whitespace trimmed) into a [`PlayerColor`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("b") || s.eq_ignore_ascii_case("black") {
            Ok(Self::Black)
        } else if s.eq_ignore_ascii_case("w") || s.eq_ignore_ascii_case("white") {
            Ok(Self::White)
        } else {
            Err(PlayerColorError)
        }
    }
}

impl From<PlayerColor> for Disc {
    #[inline]
    fn from(color: PlayerColor) -> Self {
        match color {
            PlayerColor::Black => Disc::Black,
            PlayerColor::White => Disc::White,
        }
    }
}

/// Error returned when converting [`Disc::Empty`] to a [`PlayerColor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyDiscError;

impl fmt::Display for EmptyDiscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "an empty square has no player color")
    }
}

impl std::error::Error for EmptyDiscError {}

impl TryFrom<Disc> for PlayerColor {
    type Error = EmptyDiscError;

    #[inline]
    fn try_from(disc: Disc) -> Result<Self, Self::Error> {
        match disc {
            Disc::Black => Ok(Self::Black),
            Disc::White => Ok(Self::White),
            Disc::Empty => Err(EmptyDiscError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_accepts_short_and_long_names_case_insensitively_with_whitespace() {
        for token in ["b", "B", "black", "BLACK", "Black", " b ", "\tBLACK\n"] {
            assert_eq!(
                token.parse::<PlayerColor>(),
                Ok(PlayerColor::Black),
                "{token:?}"
            );
        }
        for token in ["w", "W", "white", "WHITE", "White", " w ", "\tWhite\r\n"] {
            assert_eq!(
                token.parse::<PlayerColor>(),
                Ok(PlayerColor::White),
                "{token:?}"
            );
        }
    }

    #[test]
    fn from_str_rejects_board_disc_symbols_and_unknown_tokens() {
        for token in ["", "   ", "-", "x", "X", "o", "O", "bl", "wh", "b w"] {
            assert_eq!(
                token.parse::<PlayerColor>(),
                Err(PlayerColorError),
                "{token:?}"
            );
        }
    }

    #[test]
    fn display_as_str_and_index_round_trip() {
        assert_eq!(PlayerColor::Black.as_str(), "black");
        assert_eq!(PlayerColor::White.as_str(), "white");
        assert_eq!(PlayerColor::Black.to_string(), "black");
        assert_eq!(PlayerColor::White.to_string(), "white");
        assert_eq!(PlayerColor::Black.index(), 0);
        assert_eq!(PlayerColor::White.index(), 1);
        for color in [PlayerColor::Black, PlayerColor::White] {
            assert_eq!(color.to_string().parse::<PlayerColor>(), Ok(color));
        }
    }

    #[test]
    fn error_types_format_non_empty_messages() {
        assert_eq!(
            PlayerColorError.to_string(),
            "invalid player color (expected 'b', 'w', 'black', or 'white')"
        );
        assert_eq!(
            EmptyDiscError.to_string(),
            "an empty square has no player color"
        );
    }
}
