use crate::constants::SCORE_INF;
use crate::square::Square;
use crate::types::Score;

/// Represents a root move with its search results and statistics.
#[derive(Clone, Debug)]
pub struct RootMove {
    /// The move square
    pub sq: Square,
    /// Current best score for this move in the current iteration
    pub score: Score,
    /// Score from the previous iteration, used for aspiration windows
    pub previous_score: Score,
    /// Running average score across iterations (for stability analysis)
    pub average_score: Score,
    /// Principal variation line starting from this move
    pub pv: Vec<Square>,
}

impl RootMove {
    /// Creates a new RootMove for the given square.
    pub fn new(sq: Square) -> Self {
        Self {
            sq,
            score: -SCORE_INF,
            previous_score: -SCORE_INF,
            average_score: -SCORE_INF,
            pv: Vec::new(),
        }
    }
}
