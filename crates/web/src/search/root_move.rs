use reversi_core::{score::Score, square::Square};

/// A root move with its search results.
#[derive(Clone)]
pub struct RootMove {
    /// The move square.
    pub sq: Square,
    /// Current best score for this move in the current iteration.
    pub score: Score,
    /// Score from the previous iteration, used for aspiration windows.
    pub previous_score: Score,
    /// Running average score across iterations, used to center the aspiration
    /// window.
    pub average_score: Score,
}

impl RootMove {
    /// Creates a new root move for the given square.
    pub fn new(sq: Square) -> Self {
        Self {
            sq,
            score: -Score::INF,
            previous_score: -Score::INF,
            average_score: -Score::INF,
        }
    }
}
