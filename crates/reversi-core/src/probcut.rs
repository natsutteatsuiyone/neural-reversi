//! ProbCut forward pruning implementation.

use crate::score::Score;
use crate::types::Depth;

mod endgame;
#[cfg(test)]
mod midgame_params;

#[cfg(feature = "probcut-trace")]
pub use endgame::trace;
pub use endgame::{
    ENDGAME_LEVELS, ENDGAME_PROBCUT_WEIGHTS, EndgameWindow, NUM_VERIFY_FEATURES, SCREEN_RATES,
    VERIFY_DEPTH, VERIFY_FIT_T_MAX, VerifyModel, endgame_probcut_window, endgame_tail_multiplier,
};

/// Number of steps in [`Selectivity::ENDGAME_LADDER`].
pub const ENDGAME_STEPS: usize = 5;

/// Selectivity level for search pruning (ProbCut confidence levels).
///
/// `Mid` is used for time-controlled midgame search; on the endgame ladder
/// (`Level1`..`Level4`), lower levels are more aggressive (prune more) and
/// higher levels are more conservative. `None` disables ProbCut entirely.
///
/// Midgame ProbCut scales its margin by [`Selectivity::t_value`]. Endgame
/// ProbCut uses [`endgame_tail_multiplier`] instead, calibrated so the root
/// score is exact with [`Selectivity::probability`].
///
/// The order is the declaration order, not the confidence order: `Mid`
/// compares below `Level1` although its confidence is higher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Selectivity {
    /// Midgame-only level: 86% confidence (t=1.5). Never part of the
    /// endgame selectivity ladder.
    Mid = 0,
    /// Most aggressive endgame-ladder level: 73% confidence (t=1.1)
    #[default]
    Level1 = 1,
    /// 86% confidence
    Level2 = 2,
    /// 95% confidence
    Level3 = 3,
    /// Most conservative: 99% confidence
    Level4 = 4,
    /// ProbCut disabled.
    None = 5,
}

impl Selectivity {
    /// Selectivity steps of an endgame solve in the order they run, from the
    /// most aggressive pruning to the exact solve.
    pub const ENDGAME_LADDER: [Selectivity; ENDGAME_STEPS] = [
        Selectivity::Level1,
        Selectivity::Level2,
        Selectivity::Level3,
        Selectivity::Level4,
        Selectivity::None,
    ];

    /// Returns the statistical confidence multiplier (t-value).
    #[inline]
    pub fn t_value(self) -> f64 {
        match self {
            Selectivity::Mid => 1.5,
            Selectivity::Level1 => 1.1,
            Selectivity::Level2 => 1.5,
            Selectivity::Level3 => 2.0,
            Selectivity::Level4 => 3.3,
            Selectivity::None => 999.0,
        }
    }

    /// Returns the probability percentage for this level.
    #[inline]
    pub fn probability(self) -> i32 {
        match self {
            Selectivity::Mid => 86,
            Selectivity::Level1 => 73,
            Selectivity::Level2 => 86,
            Selectivity::Level3 => 95,
            Selectivity::Level4 => 99,
            Selectivity::None => 100,
        }
    }

    /// Converts to [`u8`].
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates a [`Selectivity`] from a [`u8`] value, clamping to valid range.
    ///
    /// Values > 5 are clamped to [`Selectivity::None`].
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => Selectivity::Mid,
            1 => Selectivity::Level1,
            2 => Selectivity::Level2,
            3 => Selectivity::Level3,
            4 => Selectivity::Level4,
            _ => Selectivity::None,
        }
    }

    /// Returns `true` if ProbCut is enabled at this selectivity level.
    #[inline]
    pub fn is_enabled(self) -> bool {
        self != Selectivity::None
    }
}

/// Pre-scaled model coefficients for one deep search depth.
///
/// The shallow depth is a pure function of the deep depth ([`probcut_depth`]),
/// and the pre-screen always compares against depth 0, so these four values
/// cover every midgame lookup.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProbcutCoefficients {
    /// Predicted mean difference between the shallow and deep scores.
    pub mean: f64,
    /// Predicted standard deviation of that difference.
    pub sigma: f64,
    /// Mean at shallow depth 0, for the static-evaluation pre-screen.
    pub mean0: f64,
    /// Sigma at shallow depth 0, for the static-evaluation pre-screen.
    pub sigma0: f64,
}

/// Returns the shallow search depth midgame ProbCut verifies with.
#[inline(always)]
pub const fn probcut_depth(deep: Depth) -> Depth {
    2 * (deep / 6)
}

include!(concat!(env!("OUT_DIR"), "/midgame_probcut_table.rs"));

/// Returns the midgame coefficients for the given ply and deep search depth.
///
/// # Panics
/// Panics if either coordinate is 60 or greater.
#[inline(always)]
pub fn midgame_coefficients(ply: usize, deep: Depth) -> &'static ProbcutCoefficients {
    &MIDGAME_TABLE[ply][deep as usize]
}

fn score_threshold(rounded: f64) -> Score {
    let bound = Score::INF.value();
    Score::from_raw((rounded as i32).clamp(-bound, bound))
}

/// Computes the ProbCut beta threshold for verification search.
#[inline]
pub fn compute_probcut_beta(beta: Score, t: f64, pc: &ProbcutCoefficients) -> Score {
    score_threshold((beta.value() as f64 + t * pc.sigma - pc.mean).ceil())
}

/// Computes the evaluation threshold for ProbCut pre-screening.
#[inline(always)]
pub fn compute_eval_beta(beta: Score, t: f64, pc: &ProbcutCoefficients, cut_node: bool) -> Score {
    let eval_mean = 0.5 * pc.mean0 + pc.mean;
    let eval_sigma = t * 0.5 * pc.sigma0 + pc.sigma;
    let all_node_margin = if cut_node { 0.0 } else { pc.sigma0 * 1.5 };
    score_threshold((beta.value() as f64 - eval_sigma - eval_mean + all_node_margin).floor())
}

#[cfg(test)]
mod tests {
    use super::midgame_params::{NUM_DEPTH, PROBCUT_PARAMS, ProbcutMidgameParams, SCORE_SCALE};
    use super::*;
    use crate::score::DiscScore;

    #[test]
    fn midgame_model_credits_half_the_deep_parity_and_uses_sqrt_for_sigma() {
        let params = ProbcutMidgameParams {
            mean_intercept: 0.25,
            mean_coef_shallow: -0.5,
            mean_coef_deep: 0.125,
            mean_coef_parity: 1.25,
            std_intercept: 0.75,
            std_coef_shallow: -0.25,
            std_coef_deep: 0.5,
        };

        // Even deep: no parity contribution.
        assert_eq!(params.mean(2.0, 8.0), 0.25);
        // Odd deep: the fitted parity offset enters at PARITY_CREDIT (0.5),
        // so 0.25 - 1.0 + 1.125 + 0.5 * 1.25 = 1.0.
        assert_eq!(params.mean(2.0, 9.0), 1.0);
        assert_eq!(params.sigma(2.0, 9.0), (0.75_f64 - 0.5 + 0.5 * 3.0).exp());
    }

    #[test]
    fn from_u8_maps_to_the_six_supported_selectivities() {
        assert_eq!(Selectivity::from_u8(0), Selectivity::Mid);
        assert_eq!(Selectivity::from_u8(1), Selectivity::Level1);
        assert_eq!(Selectivity::from_u8(2), Selectivity::Level2);
        assert_eq!(Selectivity::from_u8(3), Selectivity::Level3);
        assert_eq!(Selectivity::from_u8(4), Selectivity::Level4);
        assert_eq!(Selectivity::from_u8(5), Selectivity::None);
        assert_eq!(Selectivity::from_u8(6), Selectivity::None);
    }

    #[test]
    fn compute_probcut_beta_applies_t_sigma_minus_mean_and_rounds_up() {
        let pc = |mean, sigma| ProbcutCoefficients {
            mean,
            sigma,
            ..Default::default()
        };
        // raw value 2560; 2560 + 2.0*1.0 - 0.5 = 2561.5 -> ceil -> 2562
        assert_eq!(
            compute_probcut_beta(
                Score::from_disc_diff(DiscScore::new(10)),
                2.0,
                &pc(0.5, 1.0)
            ),
            Score::from_raw(2562)
        );
        // 0 + 1.0*0.4 - 0.0 = 0.4 -> ceil -> 1
        assert_eq!(
            compute_probcut_beta(Score::ZERO, 1.0, &pc(0.0, 0.4)),
            Score::from_raw(1)
        );
    }

    #[test]
    fn compute_eval_beta_blends_models_and_gates_the_all_node_margin() {
        let beta = Score::from_disc_diff(DiscScore::new(20)); // raw value 5120
        let t = 2.0;
        let pc = ProbcutCoefficients {
            mean: 1.0,
            sigma: 2.0,
            mean0: 4.0,
            sigma0: 3.0,
        };
        // eval_mean = 0.5*mean0 + mean = 3; eval_sigma = t*0.5*sigma0 + sigma = 5

        // cut node: margin 0 -> floor(5120 - 5 - 3 + 0) = 5112
        assert_eq!(compute_eval_beta(beta, t, &pc, true), Score::from_raw(5112));
        // all node: margin sigma0*1.5 = 4.5 -> floor(5120 - 5 - 3 + 4.5) = 5116
        assert_eq!(
            compute_eval_beta(beta, t, &pc, false),
            Score::from_raw(5116)
        );
    }

    #[test]
    fn fitted_verification_threshold_at_the_upper_sentinel_skips_the_cut() {
        let pc = midgame_coefficients(1, 4);
        assert_eq!(
            compute_probcut_beta(Score::INF, Selectivity::Mid.t_value(), pc),
            Score::INF
        );
    }

    #[test]
    fn fitted_screen_at_the_lower_sentinel_still_passes_legal_evaluations() {
        let pc = midgame_coefficients(1, 4);
        assert_eq!(
            compute_eval_beta(-Score::INF + 1, Selectivity::Mid.t_value(), pc, true),
            -Score::INF,
        );
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn invalid_depth_does_not_alias_the_next_ply() {
        let _ = midgame_coefficients(0, NUM_DEPTH as Depth);
    }

    #[test]
    fn generated_table_matches_the_parameter_formulas() {
        let approx = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(1.0);

        for (ply, params) in PROBCUT_PARAMS.iter().enumerate() {
            for deep in 0..NUM_DEPTH as Depth {
                let shallow = probcut_depth(deep) as f64;
                let deep_f = deep as f64;
                let pc = midgame_coefficients(ply, deep);
                for (got, want) in [
                    (pc.mean, params.mean(shallow, deep_f)),
                    (pc.sigma, params.sigma(shallow, deep_f)),
                    (pc.mean0, params.mean(0.0, deep_f)),
                    (pc.sigma0, params.sigma(0.0, deep_f)),
                ] {
                    assert!(
                        approx(got, want * f64::from(SCORE_SCALE)),
                        "ply {ply}, deep {deep}"
                    );
                }
            }
        }
    }
}
