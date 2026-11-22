use std::num::NonZeroUsize;

use crate::constants::BOARD_SQUARES;
use crate::score::{DiscScore, Score};
use crate::square::Square;
use crate::types::Depth;

const SKIP_SIZE: [Depth; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
const SKIP_PHASE: [Depth; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];

const VOTE_MARGIN: Score = Score::from_disc_diff(DiscScore::new(1));

pub(super) fn skips_depth(helper: NonZeroUsize, depth_ply: Depth) -> bool {
    let i = (helper.get() - 1) % SKIP_SIZE.len();
    (depth_ply + SKIP_PHASE[i]) / SKIP_SIZE[i] % 2 == 1
}

#[derive(Clone, Copy)]
pub(super) struct Ballot {
    pub(super) sq: Square,
    pub(super) score: Score,
    pub(super) depth: Depth,
}

pub(super) fn winner<T>(candidates: &[T], ballot: impl Fn(&T) -> Ballot) -> Option<usize> {
    let min_score = candidates.iter().map(|c| ballot(c).score).min()?;
    let mut votes = [0i64; BOARD_SQUARES];
    for candidate in candidates {
        let Ballot { sq, score, depth } = ballot(candidate);
        votes[sq.index()] +=
            i64::from((score - min_score + VOTE_MARGIN).value()) * i64::from(depth);
    }

    let mut best = 0;
    let first = ballot(&candidates[best]);
    let mut best_key = (votes[first.sq.index()], first.depth);
    for (i, candidate) in candidates.iter().enumerate().skip(1) {
        let Ballot { sq, depth, .. } = ballot(candidate);
        let key = (votes[sq.index()], depth);
        if key > best_key {
            best = i;
            best_key = key;
        }
    }
    Some(best)
}
