//! Fits the endgame ProbCut model from [`TraceRow`] samples.
//!
//! - Verification ([`VerifyModel`]): the exact score given the
//!   depth-[`VERIFY_DEPTH`] score. Fitted on attempts with
//!   `0 ≤ score − beta ≤` [`VERIFY_FIT_T_MAX`].
//!   - Mean of `exact − score`: least squares on [`VerifyModel::features`].
//!   - Log scale: least squares on the log of the negative residuals only
//!     (the side fail-high errors come from). The tail multiplier of each
//!     level absorbs the constant offset.
//! - Screen margins: upper quantiles of `score − eval` at [`SCREEN_RATES`],
//!   one per node type.
//!
//! Games in [`HOLDOUT_FOLD`] never enter the fit. The engine's decisions are
//! replayed on them and, with `--validate`, on a separate trace.

use std::path::Path;

use reversi_core::{
    eval::EVAL_FILE_NAME,
    probcut::{
        ENDGAME_LEVELS, NUM_VERIFY_FEATURES, SCREEN_RATES, VERIFY_DEPTH, VERIFY_FIT_T_MAX,
        VerifyModel, endgame_tail_multiplier,
    },
};

use crate::endgame_probcut::{self, TraceRow};
use crate::least_squares::weighted_least_squares;
use crate::probcut::{HOLDOUT_FOLD, game_fold};

/// Smallest negative residual, in discs, whose log enters the scale fit.
const MIN_RESIDUAL: f64 = 1.0 / 64.0;

fn row_features(row: &TraceRow) -> [f64; NUM_VERIFY_FEATURES] {
    VerifyModel::features(row.empties, row.cut_node, row.beta, row.eval)
}

/// Fits the model from `input` and prints the Rust block; with `validate`,
/// also replays the engine's decisions on an `endgame-trace` file.
pub(crate) fn execute(input: &Path, validate: Option<&Path>) -> anyhow::Result<()> {
    let rows = endgame_probcut::read_trace_rows(input)?;
    let (model, screen, holdout_rows) = fit_from_rows(&rows)?;

    if !holdout_rows.is_empty() {
        eprintln!("\n=== decisions replayed on the held-out games ===");
        replay(&model, &screen, &holdout_rows);
    }
    if let Some(validate) = validate {
        eprintln!("\n=== decisions replayed on the validation trace ===");
        let trace_rows = endgame_probcut::read_trace_rows(validate)?;
        replay(&model, &screen, &trace_rows.iter().collect::<Vec<_>>());
    }
    emit(&model, &screen);
    Ok(())
}

fn fit_from_rows(rows: &[TraceRow]) -> anyhow::Result<(VerifyModel, [f64; 2], Vec<&TraceRow>)> {
    let (fit_rows, holdout_rows): (Vec<&TraceRow>, Vec<&TraceRow>) =
        rows.iter().partition(|r| game_fold(r.game) != HOLDOUT_FOLD);
    eprintln!(
        "{} rows: {} for fitting, {} held out",
        rows.len(),
        fit_rows.len(),
        holdout_rows.len()
    );

    anyhow::ensure!(!fit_rows.is_empty(), "no samples in the fitting folds");

    let mut screen = [0.0; 2];
    for (cut_node, margin) in screen.iter_mut().enumerate() {
        let mut gaps: Vec<f64> = fit_rows
            .iter()
            .filter(|r| usize::from(r.cut_node) == cut_node)
            .map(|r| r.score - r.eval)
            .collect();
        anyhow::ensure!(
            !gaps.is_empty(),
            "no samples with cut_node={cut_node} in the fitting folds"
        );
        gaps.sort_by(f64::total_cmp);
        *margin = gaps[((1.0 - SCREEN_RATES[cut_node]) * gaps.len() as f64) as usize];
    }
    let model = fit_verify(&fit_rows)?;
    Ok((model, screen, holdout_rows))
}

fn fit_verify(rows: &[&TraceRow]) -> anyhow::Result<VerifyModel> {
    let rows: Vec<&TraceRow> = rows
        .iter()
        .copied()
        .filter(|r| (0.0..=VERIFY_FIT_T_MAX).contains(&(r.score - r.beta)))
        .collect();
    anyhow::ensure!(
        !rows.is_empty(),
        "no fitting samples with 0 <= score - beta <= {VERIFY_FIT_T_MAX}"
    );
    let mean = weighted_least_squares(
        rows.iter()
            .map(|r| (row_features(r), r.exact - r.score, 1.0)),
    )
    .ok_or_else(|| {
        anyhow::anyhow!("singular mean fit: insufficient independent fitting samples")
    })?;
    let ln_sigma = weighted_least_squares(rows.iter().filter_map(|r| {
        let x = row_features(r);
        let residual = r.exact - r.score - dot(&mean, &x);
        (residual < 0.0).then(|| (x, (-residual).max(MIN_RESIDUAL).ln(), 1.0))
    }))
    .ok_or_else(|| {
        anyhow::anyhow!("singular scale fit: insufficient independent negative-residual samples")
    })?;
    Ok(VerifyModel { mean, ln_sigma })
}

fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Replays the cut decisions (screen and verification) of every level on
/// `rows`, and prints attempts, cuts, and wrong cuts per level.
fn replay(model: &VerifyModel, screen: &[f64; 2], rows: &[&TraceRow]) {
    eprintln!("  level  attempts     cuts  wrong");
    for level in ENDGAME_LEVELS {
        let (mut cuts, mut wrong) = (0, 0);
        for r in rows {
            let w = endgame_tail_multiplier(level, r.empties);
            let margin = model.margin(r.empties, r.cut_node, r.beta, r.eval, w);
            let screened = r.eval >= r.beta + margin - screen[usize::from(r.cut_node)];
            if screened && r.score >= r.beta + margin {
                cuts += 1;
                wrong += usize::from(r.exact < r.beta);
            }
        }
        eprintln!(
            "  {:>5} {:>9} {cuts:>8} {wrong:>6}",
            level.as_u8(),
            rows.len()
        );
    }
}

fn emit(model: &VerifyModel, screen: &[f64; 2]) {
    println!("// Generated by `datagen fit-endgame-probcut`.");
    println!("/// Main-network weights the endgame model was fitted against.");
    println!("pub const ENDGAME_PROBCUT_WEIGHTS: &str = \"{EVAL_FILE_NAME}\";");
    println!();
    println!("/// Exact score given the depth-{VERIFY_DEPTH} verification score.");
    println!("#[rustfmt::skip]");
    println!("const ENDGAME_VERIFY_MODEL: VerifyModel = VerifyModel {{");
    println!("    mean: {},", coefficients(&model.mean));
    println!("    ln_sigma: {},", coefficients(&model.ln_sigma));
    println!("}};");
    println!();
    println!("/// Upper quantiles of `s{VERIFY_DEPTH} − eval` at [`SCREEN_RATES`], by `cut_node`.");
    println!("const SCREEN_MARGINS: [f64; 2] = {};", coefficients(screen));
}

fn coefficients(values: &[f64]) -> String {
    let values: Vec<_> = values.iter().map(|v| format!("{v:.10}")).collect();
    format!("[{}]", values.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_from_rows_rejects_missing_samples_in_fitting_folds() {
        let fit_game = (0..).find(|&game| game_fold(game) != HOLDOUT_FOLD).unwrap();
        let held_game = (0..).find(|&game| game_fold(game) == HOLDOUT_FOLD).unwrap();
        let row = |game, cut_node| TraceRow {
            game,
            empties: 20,
            eval: 0.0,
            score: 0.0,
            exact: -1.0,
            beta: 0.0,
            cut_node,
        };
        for (rows, message) in [
            (vec![], "no samples in the fitting folds"),
            (
                vec![row(held_game, false)],
                "no samples in the fitting folds",
            ),
            (vec![row(fit_game, false)], "no samples with cut_node=1"),
            (vec![row(fit_game, true)], "no samples with cut_node=0"),
            (
                vec![row(fit_game, false), row(held_game, true)],
                "no samples with cut_node=1",
            ),
        ] {
            let error = fit_from_rows(&rows).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
    }

    fn fitting_rows() -> Vec<TraceRow> {
        let mut rows = Vec::new();
        for empties in [14, 15, 18, 19] {
            for beta in [-16.0, 0.0, 16.0] {
                for gap in [-8.0, 0.0, 8.0] {
                    for cut_node in [false, true] {
                        for residual in [-1.0, 1.0] {
                            rows.push(TraceRow {
                                game: 0,
                                empties,
                                eval: beta + gap,
                                score: beta,
                                exact: beta + residual,
                                beta,
                                cut_node,
                            });
                        }
                    }
                }
            }
        }
        rows
    }

    #[test]
    fn fit_verify_recovers_balanced_unit_residuals() {
        let rows = fitting_rows();
        let model = fit_verify(&rows.iter().collect::<Vec<_>>()).unwrap();
        for coefficient in model.mean.iter().chain(&model.ln_sigma) {
            assert!(coefficient.abs() < 1e-10, "{coefficient}");
        }
    }

    #[test]
    fn fit_verify_rejects_singular_and_filtered_samples() {
        let mut rows = fitting_rows();
        let error = fit_verify(&rows[..2].iter().collect::<Vec<_>>())
            .err()
            .unwrap();
        assert!(error.to_string().contains("singular mean fit"));

        for row in &mut rows {
            row.exact = row.score;
        }
        let error = fit_verify(&rows.iter().collect::<Vec<_>>()).err().unwrap();
        assert!(error.to_string().contains("singular scale fit"));

        for row in &mut rows {
            row.score = row.beta - 1.0;
        }
        let error = fit_verify(&rows.iter().collect::<Vec<_>>()).err().unwrap();
        assert!(error.to_string().contains("no fitting samples with"));
    }
}
