//! Player colors for the side to move.

use std::fmt;

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
