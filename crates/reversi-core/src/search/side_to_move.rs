//! Side to move representation for search.

/// Which side is to move in the current search position.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SideToMove {
    /// The side to move at the search root.
    Player = 0,
    /// The side not to move at the search root.
    Opponent = 1,
}

impl SideToMove {
    /// Returns the opposite side to move.
    #[inline(always)]
    pub fn switch(self) -> Self {
        match self {
            SideToMove::Player => SideToMove::Opponent,
            SideToMove::Opponent => SideToMove::Player,
        }
    }
}
