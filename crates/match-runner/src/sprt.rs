//! Sequential Probability Ratio Test (SPRT) for early match termination.
//!
//! Implements a constrained multinomial GSPRT over the pentanomial model of
//! paired game results, following the approach used by Fishtest.
//! After every completed opening pair the accumulated log-likelihood ratio
//! (LLR) of H1 (`elo1`) versus H0 (`elo0`) is compared against decision
//! bounds derived from the configured error rates, tightened by Fishtest's
//! dynamic overshoot estimate (Siegmund, Sequential Analysis, Corollary
//! 8.33); crossing either bound ends the match early.

use crate::statistics::PentanomialFrequencies;

/// SPRT hypothesis test parameters.
#[derive(Debug, Clone, Copy)]
pub struct SprtConfig {
    /// Elo difference (engine 1 minus engine 2) under the null hypothesis H0.
    pub elo0: f64,
    /// Elo difference (engine 1 minus engine 2) under the alternative hypothesis H1.
    pub elo1: f64,
    /// Type I error rate (probability of accepting H1 when H0 is true).
    pub alpha: f64,
    /// Type II error rate (probability of accepting H0 when H1 is true).
    pub beta: f64,
}

/// Outcome of an SPRT evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtStatus {
    /// Neither bound crossed; keep playing.
    Continue,
    /// Upper bound crossed: accept the configured numerical H1 hypothesis.
    AcceptH1,
    /// Lower bound crossed: accept the configured numerical H0 hypothesis.
    AcceptH0,
}

/// LLR value, decision bounds, and resulting status of one SPRT evaluation.
#[derive(Debug, Clone, Copy)]
pub struct SprtResult {
    /// Accumulated log-likelihood ratio of H1 versus H0.
    pub llr: f64,
    /// Lower decision bound `ln(beta / (1 - alpha))`.
    pub lower: f64,
    /// Upper decision bound `ln((1 - beta) / alpha)`.
    pub upper: f64,
    /// Decision after comparing the LLR against the bounds, each tightened
    /// by the estimated mean overshoot; a decision may therefore trigger
    /// with an LLR still strictly inside `(lower, upper)`.
    pub status: SprtStatus,
}

impl SprtConfig {
    /// Compute the SPRT decision bounds `(lower, upper)` from the error rates.
    pub fn bounds(&self) -> (f64, f64) {
        let lower = (self.beta / (1.0 - self.alpha)).ln();
        let upper = ((1.0 - self.beta) / self.alpha).ln();
        (lower, upper)
    }
}

/// Running SPRT state: the hypothesis parameters plus the dynamic overshoot
/// estimate of the LLR jump sizes at both decision bounds.
#[derive(Debug, Clone)]
pub struct Sprt {
    pub config: SprtConfig,
    /// Number of completed pairs seen by the last overshoot update.
    last_update: u32,
    ref0: f64,
    m0: f64,
    sq0: f64,
    ref1: f64,
    m1: f64,
    sq1: f64,
}

impl Sprt {
    pub fn new(config: SprtConfig) -> Self {
        Self {
            config,
            last_update: 0,
            ref0: 0.0,
            m0: 0.0,
            sq0: 0.0,
            ref1: 0.0,
            m1: 0.0,
            sq1: 0.0,
        }
    }

    /// Recompute the LLR from the observed pair frequencies and evaluate the
    /// stop decision.
    ///
    /// Safe to call repeatedly with unchanged frequencies: the overshoot
    /// estimate only accumulates when exactly one new pair has completed
    /// since the previous call, and resets its excursion references when
    /// updates were skipped.
    pub fn update(&mut self, freq: &PentanomialFrequencies) -> SprtResult {
        let (lower, upper) = self.config.bounds();
        let llr = llr_pentanomial(freq, self.config.elo0, self.config.elo1);

        let pairs = freq.total_pairs();
        if pairs == self.last_update + 1 {
            if llr < self.ref0 {
                let delta = llr - self.ref0;
                self.m0 += delta;
                self.sq0 += delta * delta;
                self.ref0 = llr;
            }
            if llr > self.ref1 {
                let delta = llr - self.ref1;
                self.m1 += delta;
                self.sq1 += delta * delta;
                self.ref1 = llr;
            }
        } else if pairs != self.last_update {
            self.ref0 = llr;
            self.ref1 = llr;
        }
        self.last_update = pairs;

        let o0 = if self.m0 != 0.0 {
            -self.sq0 / self.m0 / 2.0
        } else {
            0.0
        };
        let o1 = if self.m1 != 0.0 {
            self.sq1 / self.m1 / 2.0
        } else {
            0.0
        };

        let status = if llr < lower + o0 {
            SprtStatus::AcceptH0
        } else if llr > upper - o1 {
            SprtStatus::AcceptH1
        } else {
            SprtStatus::Continue
        };

        SprtResult {
            llr,
            lower,
            upper,
            status,
        }
    }
}

/// Expected score in [0, 1] for a given logistic Elo difference.
fn logistic(elo: f64) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf(-elo / 400.0))
}

/// Compute the generalized log-likelihood ratio of H1 (`elo1`) versus H0
/// (`elo0`) for the observed pentanomial pair distribution.
fn llr_pentanomial(freq: &PentanomialFrequencies, elo0: f64, elo1: f64) -> f64 {
    // Pair scores normalized to [0, 1]; DD and WL both score 0.5 and are merged.
    const SCORES: [f64; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
    // Match Fishtest's regularization so every empirical category has support
    // in both constrained likelihood maximizations.
    const EPSILON: f64 = 1e-3;

    let raw = [freq.ll, freq.ld, freq.dd + freq.wl, freq.wd, freq.ww];
    if raw.iter().all(|&count| count == 0) {
        return 0.0;
    }
    let counts = raw.map(|count| {
        if count == 0 {
            EPSILON
        } else {
            f64::from(count)
        }
    });

    // Logistic scores beyond this interval are indistinguishable from its
    // endpoints in f64 and would put the secular root at infinity.
    let s0 = logistic(elo0).clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let s1 = logistic(elo1).clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let root0 = constrained_mle_root(&counts, &SCORES, s0);
    let root1 = constrained_mle_root(&counts, &SCORES, s1);

    counts
        .iter()
        .zip(SCORES)
        .map(|(count, score)| {
            let denominator0 = 1.0 + root0 * (score - s0);
            let denominator1 = 1.0 + root1 * (score - s1);
            count * (denominator0 / denominator1).ln()
        })
        .sum()
}

/// Find the Lagrange multiplier for the multinomial MLE constrained to
/// `expected_score`.
fn constrained_mle_root(counts: &[f64; 5], scores: &[f64; 5], expected_score: f64) -> f64 {
    // The secular function is strictly decreasing between these poles. All
    // bisection midpoints remain in the open interval, so every denominator
    // is positive without an arbitrary root tolerance near either pole.
    let mut lower = -1.0 / (1.0 - expected_score);
    let mut upper = 1.0 / expected_score;

    for _ in 0..100 {
        let root = lower * 0.5 + upper * 0.5;
        let value = counts
            .iter()
            .zip(scores)
            .map(|(count, score)| {
                let centered_score = score - expected_score;
                count * centered_score / (1.0 + root * centered_score)
            })
            .sum::<f64>();

        if value > 0.0 {
            lower = root;
        } else {
            upper = root;
        }
    }

    lower * 0.5 + upper * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYMMETRIC: SprtConfig = SprtConfig {
        elo0: -10.0,
        elo1: 10.0,
        alpha: 0.05,
        beta: 0.05,
    };

    /// One-shot evaluation on fresh state; the overshoot estimate stays
    /// zeroed, so the decision uses the raw bounds.
    fn one_shot(config: SprtConfig, freq: &PentanomialFrequencies) -> SprtResult {
        Sprt::new(config).update(freq)
    }

    #[test]
    fn bounds_match_error_rates() {
        let (lower, upper) = SYMMETRIC.bounds();

        assert!((lower - (-2.9444389791664403)).abs() < 1e-12);
        assert!((upper - 2.9444389791664403).abs() < 1e-12);
    }

    #[test]
    fn no_pairs_yield_zero_llr_and_continue() {
        let result = one_shot(SYMMETRIC, &PentanomialFrequencies::default());

        assert_eq!(result.llr, 0.0);
        assert_eq!(result.status, SprtStatus::Continue);
    }

    #[test]
    fn balanced_results_continue_under_symmetric_bounds() {
        let freq = PentanomialFrequencies {
            ll: 20,
            ld: 10,
            wl: 40,
            wd: 10,
            ww: 20,
            ..PentanomialFrequencies::default()
        };
        let result = one_shot(SYMMETRIC, &freq);

        assert!(result.llr.abs() < 1e-9);
        assert_eq!(result.status, SprtStatus::Continue);
    }

    #[test]
    fn engine1_dominance_accepts_h1() {
        let freq = PentanomialFrequencies {
            dd: 20,
            wd: 30,
            ww: 30,
            ..PentanomialFrequencies::default()
        };
        let result = one_shot(SYMMETRIC, &freq);

        assert!(result.llr > result.upper);
        assert_eq!(result.status, SprtStatus::AcceptH1);
    }

    #[test]
    fn engine2_dominance_accepts_h0() {
        let freq = PentanomialFrequencies {
            ll: 30,
            ld: 30,
            dd: 20,
            ..PentanomialFrequencies::default()
        };
        let result = one_shot(SYMMETRIC, &freq);

        assert!(result.llr < result.lower);
        assert_eq!(result.status, SprtStatus::AcceptH0);
    }

    #[test]
    fn mirrored_results_negate_llr_under_symmetric_bounds() {
        let freq = PentanomialFrequencies {
            ll: 5,
            ld: 10,
            dd: 15,
            wl: 20,
            wd: 25,
            ww: 30,
        };
        let mirrored = PentanomialFrequencies {
            ll: 30,
            ld: 25,
            dd: 15,
            wl: 20,
            wd: 10,
            ww: 5,
        };

        let llr = one_shot(SYMMETRIC, &freq).llr;
        let mirrored_llr = one_shot(SYMMETRIC, &mirrored).llr;

        assert!((llr + mirrored_llr).abs() < 1e-9);
    }

    #[test]
    fn fishtest_style_bounds_reject_regression() {
        // elo0=0, elo1=10: an even match should eventually accept H0.
        let config = SprtConfig {
            elo0: 0.0,
            elo1: 10.0,
            alpha: 0.05,
            beta: 0.05,
        };
        let freq = PentanomialFrequencies {
            ll: 800,
            ld: 400,
            dd: 800,
            wl: 800,
            wd: 400,
            ww: 800,
        };
        let result = one_shot(config, &freq);

        assert_eq!(result.status, SprtStatus::AcceptH0);
    }

    #[test]
    fn overshoot_correction_stops_inside_raw_bounds() {
        let mut sprt = Sprt::new(SYMMETRIC);
        let mut freq = PentanomialFrequencies::default();

        for _ in 0..100 {
            freq.ww += 1;
            let result = sprt.update(&freq);
            if result.status == SprtStatus::AcceptH1 {
                // The mean upward jump tightens the upper bound, so a pure
                // win stream stops one pair before the raw crossing.
                assert!(result.llr < result.upper);
                assert!(result.llr > result.upper - 0.06);
                return;
            }
        }
        panic!("pure win stream must accept H1");
    }

    #[test]
    fn repeated_same_data_updates_do_not_change_the_decision() {
        let run = |updates_per_pair: usize| {
            let mut sprt = Sprt::new(SYMMETRIC);
            let mut freq = PentanomialFrequencies::default();
            for pair in 1..=200u32 {
                if pair % 3 == 0 {
                    freq.wl += 1;
                } else {
                    freq.ww += 1;
                }
                let mut last = sprt.update(&freq);
                for _ in 1..updates_per_pair {
                    last = sprt.update(&freq);
                }
                if last.status != SprtStatus::Continue {
                    return (pair, last.llr);
                }
            }
            panic!("dominant stream must reach a decision");
        };

        assert_eq!(run(1), run(3));
    }
}
