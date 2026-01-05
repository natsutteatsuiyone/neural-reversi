use crate::square::Square;
use crate::types::ScaledScore;

/// Represents a root move with its search results and statistics.
#[derive(Clone, Debug)]
pub struct RootMove {
    /// The move square
    pub sq: Square,
    /// Current best score for this move in the current iteration
    pub score: ScaledScore,
    /// Score from the previous iteration, used for aspiration windows
    pub previous_score: ScaledScore,
    /// Running average score across iterations (for stability analysis)
    pub average_score: ScaledScore,
    /// Principal variation line starting from this move
    pub pv: Vec<Square>,
}

impl RootMove {
    /// Creates a new RootMove for the given square.
    pub fn new(sq: Square) -> Self {
        Self {
            sq,
            score: -ScaledScore::INF,
            previous_score: -ScaledScore::INF,
            average_score: -ScaledScore::INF,
            pv: Vec::new(),
        }
    }
}
