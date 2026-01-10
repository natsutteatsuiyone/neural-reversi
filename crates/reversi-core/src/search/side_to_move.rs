/// Represents which side is to move in the current search position.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SideToMove {
    /// The player being searched for (typically the engine)
    Player = 0,
    /// The opponent (typically the human or other engine)
    Opponent = 1,
}

impl SideToMove {
    /// Switches the side to move to the opposite player.
    ///
    /// # Returns
    /// The opposite side to move
    pub fn switch(self) -> Self {
        match self {
            SideToMove::Player => SideToMove::Opponent,
            SideToMove::Opponent => SideToMove::Player,
        }
    }
}
