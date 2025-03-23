use crate::square::Square;
use crate::types::Score;

#[derive(Clone, Debug)]
pub struct RootMove {
    pub sq: Square,
    pub score: Score,
    pub average_score: Score,
    pub pv: Vec<Square>,
}
