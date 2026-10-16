//! Fits the endgame ProbCut model from `endgame-trace` samples.
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

use std::{
    fs::File,
    io::{self, BufRead, BufReader},
};

use reversi_core::{
    eval::EVAL_FILE_NAME,
    probcut::{
        ENDGAME_LEVELS, NUM_VERIFY_FEATURES, SCREEN_RATES, VERIFY_DEPTH, VERIFY_FIT_T_MAX,
        VerifyModel, endgame_tail_multiplier,
    },
};

use crate::endgame_probcut::{HOLDOUT_FOLD, trace_header};
use crate::fit_probcut::{game_fold, weighted_least_squares};

/// Smallest negative residual, in discs, whose log enters the scale fit.
const MIN_RESIDUAL: f64 = 1.0 / 64.0;

/// One traced ProbCut attempt. Scores are in discs.
struct Row {
    game: u64,
    empties: u32,
    /// Static evaluation.
    eval: f64,
    /// Verification score.
    score: f64,
    exact: f64,
    beta: f64,
    cut_node: bool,
}

impl Row {
    fn features(&self) -> [f64; NUM_VERIFY_FEATURES] {
        VerifyModel::features(self.empties, self.cut_node, self.beta, self.eval)
    }
}

/// Fits the model from `input` and prints the Rust block; with `validate`,
/// also replays the engine's decisions on an `endgame-trace` file.
pub fn execute(input: &str, validate: Option<&str>) -> io::Result<()> {
    let rows = read_rows(input)?;
    let (fit_rows, holdout_rows): (Vec<&Row>, Vec<&Row>) =
        rows.iter().partition(|r| game_fold(r.game) != HOLDOUT_FOLD);
    eprintln!(
        "{} rows: {} for fitting, {} held out",
        rows.len(),
        fit_rows.len(),
        holdout_rows.len()
    );

    let model = fit_verify(&fit_rows);
    let screen: [f64; 2] = std::array::from_fn(|cut_node| {
        let mut gaps: Vec<f64> = fit_rows
            .iter()
            .filter(|r| usize::from(r.cut_node) == cut_node)
            .map(|r| r.score - r.eval)
            .collect();
        gaps.sort_by(f64::total_cmp);
        gaps[((1.0 - SCREEN_RATES[cut_node]) * gaps.len() as f64) as usize]
    });

    if !holdout_rows.is_empty() {
        eprintln!("\n=== decisions replayed on the held-out games ===");
        replay(&model, &screen, &holdout_rows);
    }
    if let Some(validate) = validate {
        eprintln!("\n=== decisions replayed on the validation trace ===");
        let trace_rows = read_rows(validate)?;
        replay(&model, &screen, &trace_rows.iter().collect::<Vec<_>>());
    }
    emit(&model, &screen);
    Ok(())
}

fn read_rows(input: &str) -> io::Result<Vec<Row>> {
    let reader = BufReader::new(File::open(input)?);
    let invalid = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
    let mut lines = reader.lines();
    let weights_line = lines
        .next()
        .ok_or_else(|| invalid("empty input".into()))??;
    let weights = weights_line
        .strip_prefix("# main_weights=")
        .ok_or_else(|| invalid(format!("missing weights header: '{weights_line}'")))?;
    if weights != EVAL_FILE_NAME {
        return Err(invalid(format!(
            "samples were generated with {weights}, but the engine uses {EVAL_FILE_NAME}"
        )));
    }
    let header = lines
        .next()
        .ok_or_else(|| invalid("missing CSV header".into()))??;
    if header != trace_header() {
        return Err(invalid(format!(
            "'{header}' is not the endgame-trace header '{}'",
            trace_header()
        )));
    }

    let mut rows = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let line = line?;
        let fields: Vec<&str> = line.split(',').collect();
        let field = |i: usize| -> io::Result<f64> {
            fields
                .get(i)
                .and_then(|f| f.parse().ok())
                .ok_or_else(|| invalid(format!("line {}: bad field {i}", line_no + 3)))
        };
        rows.push(Row {
            game: field(0)? as u64,
            empties: field(1)? as u32,
            eval: field(2)?,
            score: field(3)?,
            exact: field(4)?,
            beta: field(5)?,
            cut_node: field(7)? != 0.0,
        });
    }
    Ok(rows)
}

fn fit_verify(rows: &[&Row]) -> VerifyModel {
    let rows: Vec<&Row> = rows
        .iter()
        .copied()
        .filter(|r| (0.0..=VERIFY_FIT_T_MAX).contains(&(r.score - r.beta)))
        .collect();
    let mean = weighted_least_squares(rows.iter().map(|r| (r.features(), r.exact - r.score, 1.0)))
        .expect("singular mean fit");
    let ln_sigma = weighted_least_squares(rows.iter().filter_map(|r| {
        let x = r.features();
        let residual = r.exact - r.score - dot(&mean, &x);
        (residual < 0.0).then(|| (x, (-residual).max(MIN_RESIDUAL).ln(), 1.0))
    }))
    .expect("singular scale fit");
    VerifyModel { mean, ln_sigma }
}

fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Replays the cut decisions (screen and verification) of every level on
/// `rows`, and prints attempts, cuts, and wrong cuts per level.
fn replay(model: &VerifyModel, screen: &[f64; 2], rows: &[&Row]) {
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
