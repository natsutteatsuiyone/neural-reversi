//! Endgame ProbCut: a fitted score model and the cut window built on it.
//!
//! A cut takes two steps at a node:
//!
//! 1. Screen: the verification is skipped when the static evaluation is
//!    more than [`SCREEN_MARGINS`] below the verification threshold.
//! 2. Verification: a depth-[`VERIFY_DEPTH`] search. The node is cut when
//!    its score reaches `beta + w · sigma(x) − mean(x)`.
//!
//! [`VerifyModel`] predicts the exact score as the verification score plus
//! `mean(x) + sigma(x) · Z`. The features `x` are known before the
//! verification runs, so the threshold has a closed form. The tail
//! multiplier `w` comes from [`TAIL_SCHEDULE`].
//!
//! `datagen fit-endgame-probcut` fits the model and the screen margins on
//! positions traced from the search tree. Game records are not used: in the
//! tree, the exact score runs well above the shallow score, unlike in game
//! records.

use crate::level::Level;
use crate::probcut::Selectivity;
use crate::types::{Depth, ScaledScore};

/// Depth of the verification search.
pub const VERIFY_DEPTH: Depth = 2;

/// Empty counts the model was fitted on; other counts use the nearest end.
const MODEL_EMPTIES: std::ops::RangeInclusive<u32> = 13..=30;

/// Pass probability below which the verification is not searched, indexed
/// by `cut_node`: 10% at expected all nodes, 0.1% at expected cut nodes.
pub const SCREEN_RATES: [f64; 2] = [0.1, 0.001];

/// The selectivity levels endgame ProbCut runs at: [`Level::ENDGAME_SELECTIVITY`]
/// without its final exact step.
pub const ENDGAME_LEVELS: [Selectivity; 4] = {
    let [levels @ .., Selectivity::None] = Level::ENDGAME_SELECTIVITY else {
        panic!("the endgame selectivity ladder must end at Selectivity::None");
    };
    levels
};

/// Tail multiplier per level: `w = a + b·E + c·(E mod 2)`.
/// Rows are `[a, b, c]` for Level1 to Level4.
///
/// Nodes with more empties get a larger multiplier, because a wrong cut
/// there reaches the root more often.
///
/// [`Selectivity::probability`] is the chance that the root score is exact.
/// The intercepts are calibrated to hit it: on exactly labeled game positions
/// with 20 to 28 empties (`datagen endgame-holdout`), the root score is exact
/// 72.7%, 85.9%, 94.8%, and 99.1% of the time. The slopes make the model cut
/// as often as its predecessor did at each empty count.
const TAIL_SCHEDULE: [[f64; 3]; 4] = [
    [0.3786, 0.1317, 0.3825],
    [-1.3215, 0.2985, 0.6360],
    [0.0170, 0.3131, 0.7175],
    [-0.4242, 0.4874, 0.9121],
];

/// Tail multiplier `w` of a level at `empties`: `a + b·E + c·(E mod 2)`.
#[inline(always)]
pub fn endgame_tail_multiplier(selectivity: Selectivity, empties: u32) -> f64 {
    let [a, b, c] = TAIL_SCHEDULE[selectivity as usize - 1];
    a + b * empties as f64 + c * (empties & 1) as f64
}

/// Number of [`VerifyModel`] regression features.
pub const NUM_VERIFY_FEATURES: usize = 8;

/// Largest `score − beta` the verification model is fitted on.
pub const VERIFY_FIT_T_MAX: f64 = 32.0;

/// Location-scale model of the exact score given the verification score.
///
/// `exact = score + mean(x) + sigma(x) · Z`, with `mean` and `ln sigma`
/// linear in the features `x` ([`VerifyModel::features`]). It is fitted only
/// where cuts are decided: `0 ≤ score − beta ≤` [`VERIFY_FIT_T_MAX`]. Scores
/// are in discs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifyModel {
    /// Coefficients of the mean of `exact − score`.
    pub mean: [f64; NUM_VERIFY_FEATURES],
    /// Coefficients of the log scale of `exact − score`.
    pub ln_sigma: [f64; NUM_VERIFY_FEATURES],
}

impl VerifyModel {
    /// Regression features of a node:
    /// `[1, E, E mod 2, cut_node, d, |d + 3/8|, u, u²]`.
    ///
    /// - `E`: empty count, clamped to the fitted range.
    /// - `d = (eval − beta) / 8`, clamped to ±2. `|d + 3/8|` has its minimum
    ///   where `eval` is 3 discs below `beta`; there the exact score runs
    ///   lowest and spreads least.
    /// - `u = beta / 32`.
    #[inline(always)]
    pub fn features(
        empties: u32,
        cut_node: bool,
        beta: f64,
        eval: f64,
    ) -> [f64; NUM_VERIFY_FEATURES] {
        let e = empties.clamp(*MODEL_EMPTIES.start(), *MODEL_EMPTIES.end());
        let d = ((eval - beta) / 8.0).clamp(-2.0, 2.0);
        let u = beta / 32.0;
        [
            1.0,
            e as f64,
            (e & 1) as f64,
            f64::from(u8::from(cut_node)),
            d,
            (d + 0.375).abs(),
            u,
            u * u,
        ]
    }

    /// Mean of `exact − score` at features `x`.
    #[inline(always)]
    pub fn mean(&self, x: &[f64; NUM_VERIFY_FEATURES]) -> f64 {
        dot(&self.mean, x)
    }

    /// Scale of `exact − score` at features `x`.
    #[inline(always)]
    pub fn sigma(&self, x: &[f64; NUM_VERIFY_FEATURES]) -> f64 {
        dot(&self.ln_sigma, x).exp()
    }

    /// Margin `m ≥ 0` such that the node is cut when the verification score
    /// is at least `beta + m`, at tail multiplier `w`.
    #[inline(always)]
    pub fn margin(&self, empties: u32, cut_node: bool, beta: f64, eval: f64, w: f64) -> f64 {
        let x = Self::features(empties, cut_node, beta, eval);
        (w * self.sigma(&x) - self.mean(&x)).max(0.0)
    }
}

#[inline(always)]
fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Raw score bounds of one endgame ProbCut attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EndgameWindow {
    /// The verification passes when its score is at least this.
    pub pc_beta: ScaledScore,
    /// The verification runs only when the static evaluation is at least this.
    pub screen_beta: ScaledScore,
}

/// Returns the ProbCut window of a node.
///
/// `beta` is the null-window beta and `eval` the main-network static
/// evaluation. `pc_beta` is clamped to [`ScaledScore::MAX`], which no
/// verification reaches, so a clamped window never cuts.
///
/// `selectivity` must be one of [`ENDGAME_LEVELS`].
#[inline]
pub fn endgame_probcut_window(
    empties: Depth,
    selectivity: Selectivity,
    cut_node: bool,
    beta: ScaledScore,
    eval: ScaledScore,
) -> EndgameWindow {
    let scale = ScaledScore::SCALE as f64;
    let max = ScaledScore::MAX.value();
    let margin = ENDGAME_VERIFY_MODEL.margin(
        empties,
        cut_node,
        beta.value() as f64 / scale,
        eval.value() as f64 / scale,
        endgame_tail_multiplier(selectivity, empties),
    );
    let pc_beta = (beta.value() + (margin * scale).ceil() as i32).min(max);
    let screen = margin - SCREEN_MARGINS[usize::from(cut_node)];
    let screen_beta = (beta.value() + (screen * scale).floor() as i32).clamp(-max, max);
    EndgameWindow {
        pc_beta: ScaledScore::from_raw(pc_beta),
        screen_beta: ScaledScore::from_raw(screen_beta),
    }
}

/// Samples of endgame ProbCut attempts, used to fit and check the models on
/// the positions the search actually visits.
#[cfg(feature = "probcut-trace")]
pub mod trace {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    use crate::board::Board;
    use crate::probcut::Selectivity;
    use crate::types::ScaledScore;

    /// One endgame ProbCut attempt.
    #[derive(Clone, Copy, Debug)]
    pub struct TraceRecord {
        /// Position at the node, side to move first.
        pub board: Board,
        /// Null-window beta at the node.
        pub beta: ScaledScore,
        /// Selectivity the attempt ran at.
        pub selectivity: Selectivity,
        /// Whether the node was searched as an expected cut node.
        pub cut_node: bool,
        /// The score ProbCut returned, if it cut.
        pub result: Option<ScaledScore>,
    }

    /// Number of empty counts a position can have.
    const NUM_EMPTIES: usize = 61;

    static ONE_IN: AtomicU64 = AtomicU64::new(0);
    static MIN_EMPTIES: AtomicU32 = AtomicU32::new(0);
    static MAX_EMPTIES: AtomicU32 = AtomicU32::new(u32::MAX);

    /// Most records kept per empty count, so a trace's memory stays bounded.
    const MAX_RECORDS_PER_EMPTIES: usize = 200_000;

    /// Records not yet returned by [`take`], and the number kept so far per
    /// empty count (not reset by [`take`]).
    static TRACE: Mutex<(Vec<TraceRecord>, [usize; NUM_EMPTIES])> =
        Mutex::new((Vec::new(), [0; NUM_EMPTIES]));

    /// Records attempts whose empty count is in `empties` and whose position
    /// hash is divisible by `one_in`. `one_in = 0` disables recording.
    pub fn set_sampling(one_in: u64, empties: std::ops::RangeInclusive<u32>) {
        MIN_EMPTIES.store(*empties.start(), Ordering::Relaxed);
        MAX_EMPTIES.store(*empties.end(), Ordering::Relaxed);
        TRACE.lock().unwrap().1 = [0; NUM_EMPTIES];
        ONE_IN.store(one_in, Ordering::Relaxed);
    }

    /// Returns and clears the records collected so far.
    pub fn take() -> Vec<TraceRecord> {
        std::mem::take(&mut TRACE.lock().unwrap().0)
    }

    pub(crate) fn record(
        board: &Board,
        beta: ScaledScore,
        selectivity: Selectivity,
        cut_node: bool,
        result: Option<ScaledScore>,
    ) {
        let one_in = ONE_IN.load(Ordering::Relaxed);
        let empties = board.get_empty_count();
        if one_in != 0
            && (MIN_EMPTIES.load(Ordering::Relaxed)..=MAX_EMPTIES.load(Ordering::Relaxed))
                .contains(&empties)
            && board.hash().is_multiple_of(one_in)
        {
            let mut trace = TRACE.lock().unwrap();
            let (records, counts) = &mut *trace;
            let count = &mut counts[empties as usize];
            if *count >= MAX_RECORDS_PER_EMPTIES {
                return;
            }
            *count += 1;
            records.push(TraceRecord {
                board: *board,
                beta,
                selectivity,
                cut_node,
                result,
            });
        }
    }
}

// Generated by `datagen fit-endgame-probcut`.
/// Main-network weights the endgame model was fitted against.
pub const ENDGAME_PROBCUT_WEIGHTS: &str = "eval-e6bbc4f6.zst";

/// Exact score given the depth-2 verification score.
#[rustfmt::skip]
const ENDGAME_VERIFY_MODEL: VerifyModel = VerifyModel {
    mean: [3.2996552433, -0.0583166555, -0.2489634608, -0.3459216837, 0.2330285133, 0.3175907433, 0.5569823145, -0.6444854153],
    ln_sigma: [0.9479918545, -0.0175477071, -0.0846528866, -0.0265623046, -0.3431613432, 0.3472355880, -0.0296237147, -0.1353929169],
};

/// Upper quantiles of `s2 − eval` at [`SCREEN_RATES`], by `cut_node`.
const SCREEN_MARGINS: [f64; 2] = [4.9062500000, 17.7304690000];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_was_fitted_for_the_engine_main_network() {
        assert_eq!(ENDGAME_PROBCUT_WEIGHTS, crate::eval::EVAL_FILE_NAME);
    }

    #[test]
    fn windows_tighten_with_each_level() {
        for empties in 13..=36 {
            for cut_node in [false, true] {
                for (beta, eval) in [(0, 0), (-10, -4), (20, 26), (8, 2), (-40, -20)] {
                    let windows = ENDGAME_LEVELS.map(|level| {
                        endgame_probcut_window(
                            empties,
                            level,
                            cut_node,
                            ScaledScore::from_disc_diff(beta),
                            ScaledScore::from_disc_diff(eval),
                        )
                    });
                    for pair in windows.windows(2) {
                        assert!(pair[0].pc_beta < pair[1].pc_beta);
                    }
                    for window in windows {
                        assert!(window.pc_beta > ScaledScore::from_disc_diff(beta));
                        assert!(window.screen_beta < window.pc_beta);
                    }
                }
            }
        }
    }
}
