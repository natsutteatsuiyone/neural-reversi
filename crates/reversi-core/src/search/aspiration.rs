use crate::score::{DiscScore, Score};
use crate::types::Depth;

pub(super) const ASPIRATION_MIN_DEPTH: Depth = 5;
const MIDGAME_DELTA: Score = Score::from_disc_diff(DiscScore::new(2));
const MSS_DIVISOR: i64 = 10193;
const HELPER_WIDTH_STEP: i32 = Score::SCALE / 4;
const HELPER_WIDTH_CYCLE: usize = 8;

pub(super) fn midgame_window(
    depth: Depth,
    history: Option<(Score, i64)>,
    helper_index: usize,
) -> (Score, Score, Score) {
    if let Some((average, mean_squared)) = history.filter(|_| depth >= ASPIRATION_MIN_DEPTH) {
        let steps = (helper_index % HELPER_WIDTH_CYCLE) as i32;
        let delta =
            MIDGAME_DELTA + HELPER_WIDTH_STEP * steps + (mean_squared.abs() / MSS_DIVISOR) as i32;
        (
            (average - delta).max(-Score::INF),
            (average + delta).min(Score::INF),
            delta,
        )
    } else {
        (-Score::INF, Score::INF, MIDGAME_DELTA)
    }
}

pub(super) fn retry_midgame(
    alpha: &mut Score,
    beta: &mut Score,
    mut delta: Score,
    mut search: impl FnMut(Score, Score) -> (Score, bool),
) -> (Score, bool) {
    let mut window_clean = true;
    loop {
        let (score, aborted) = search(*alpha, *beta);
        if aborted || !widen_aspiration_window(score, alpha, beta, delta) {
            return (score, window_clean);
        }
        window_clean = false;
        delta += delta / 2;
    }
}

pub(crate) fn widen_aspiration_window(
    score: Score,
    alpha: &mut Score,
    beta: &mut Score,
    delta: Score,
) -> bool {
    if score <= *alpha {
        *beta = *alpha;
        *alpha = (score - delta).max(-Score::INF);
    } else if score >= *beta {
        *alpha = (*beta - delta).max(*alpha);
        *beta = (score + delta).min(Score::INF);
    } else {
        return false;
    }
    true
}
