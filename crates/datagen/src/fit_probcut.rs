//! Fits the midgame ProbCut model from generated CSV samples.
//!
//! Game ids are assigned to deterministic folds before any cell aggregation.
//! Cross-validation therefore keeps every position from a game on the same
//! side of a train/holdout split. The final emitted parameters are refitted on
//! all games after the fold diagnostics have been reported.

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader},
};

use reversi_core::constants::INITIAL_EMPTY_COUNT;

const NUM_PLY: usize = INITIAL_EMPTY_COUNT;
const NUM_FOLDS: usize = 5;
const MIN_POSITIONS_WITHOUT_SMOOTHING: usize = 200;
const MIN_MEAN_CELLS: usize = 4;
const MIN_SIGMA_CELLS: usize = 3;
const MIN_CELL_SAMPLES: f64 = 4.0;
const MIN_SIGMA: f64 = 1e-8;
// = MIN_SIGMA.ln(); sentinel rows predict the sigma floor.
const DEFAULT_STD_INTERCEPT: f64 = -18.420680743952367;

type DepthPair = (u32, u32);

#[derive(Clone, Copy, Debug, Default)]
struct CellStats {
    n: f64,
    sum: f64,
    sum_sq: f64,
}

impl CellStats {
    fn add_sample(&mut self, value: f64) {
        self.n += 1.0;
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn add_scaled(&mut self, other: Self, weight: f64) {
        self.n += weight * other.n;
        self.sum += weight * other.sum;
        self.sum_sq += weight * other.sum_sq;
    }

    fn mean(self) -> f64 {
        self.sum / self.n
    }

    fn sample_std(self) -> Option<f64> {
        if self.n < MIN_CELL_SAMPLES {
            return None;
        }
        let centered_sum_sq = (self.sum_sq - self.sum * self.sum / self.n).max(0.0);
        let std = (centered_sum_sq / (self.n - 1.0)).sqrt();
        (std >= MIN_SIGMA).then_some(std)
    }
}

#[derive(Debug)]
struct Sample {
    game: u64,
    ply: usize,
    shallow: u32,
    deep: u32,
    diff: f64,
}

struct FoldData {
    cells: HashMap<(usize, u32, u32), CellStats>,
    positions: [HashSet<u64>; NUM_PLY],
}

impl Default for FoldData {
    fn default() -> Self {
        Self {
            cells: HashMap::new(),
            positions: std::array::from_fn(|_| HashSet::new()),
        }
    }
}

#[derive(Default)]
struct Dataset {
    folds: [FoldData; NUM_FOLDS],
    samples: u64,
    games: HashSet<u64>,
}

impl Dataset {
    fn add(&mut self, sample: Sample) {
        let fold = game_fold(sample.game);
        let data = &mut self.folds[fold];
        data.positions[sample.ply].insert(sample.game);
        data.cells
            .entry((sample.ply, sample.shallow, sample.deep))
            .or_default()
            .add_sample(sample.diff);
        self.games.insert(sample.game);
        self.samples += 1;
    }
}

struct AggregatedData {
    cells: [HashMap<DepthPair, CellStats>; NUM_PLY],
    positions: [usize; NUM_PLY],
}

impl Default for AggregatedData {
    fn default() -> Self {
        Self {
            cells: std::array::from_fn(|_| HashMap::new()),
            positions: [0; NUM_PLY],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FittedParams {
    mean: [f64; 4],
    sigma: [f64; 3],
}

const DEFAULT_PARAMS: FittedParams = FittedParams {
    mean: [0.0; 4],
    sigma: [DEFAULT_STD_INTERCEPT, 0.0, 0.0],
};

#[derive(Clone, Copy, Debug)]
struct PlyFit {
    params: FittedParams,
    positions: usize,
    cells: usize,
    smoothed: bool,
    fitted: bool,
    carried_from: Option<usize>,
    legacy_residual_sigma: f64,
    redesigned_residual_sigma: f64,
}

impl Default for PlyFit {
    fn default() -> Self {
        Self {
            params: DEFAULT_PARAMS,
            positions: 0,
            cells: 0,
            smoothed: false,
            fitted: false,
            carried_from: None,
            legacy_residual_sigma: f64::NAN,
            redesigned_residual_sigma: f64::NAN,
        }
    }
}

/// Fits midgame ProbCut parameters from the CSV produced by `probcut`.
///
/// Fold and in-sample diagnostics are printed to stderr so stdout can be
/// redirected directly into a source snippet.
pub fn execute(input: &str) -> io::Result<()> {
    let dataset = read_dataset(input)?;
    if dataset.samples == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no samples found in '{input}'"),
        ));
    }

    eprintln!(
        "loaded {} samples from {} games",
        dataset.samples,
        dataset.games.len()
    );

    let all = aggregate_folds(&dataset, |_| true);
    let fits = fit_all_plies(&all);
    if !fits.iter().any(|fit| fit.fitted) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no ply could be fitted from '{input}'; collect more samples"),
        ));
    }

    report_cross_validation(&dataset);
    report_in_sample(&fits);
    emit_params(&fits);
    Ok(())
}

fn read_dataset(input: &str) -> io::Result<Dataset> {
    let file = File::open(input).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("failed to open input file '{input}': {e}"),
        )
    })?;
    let reader = BufReader::new(file);

    let mut dataset = Dataset::default();

    for (line_no, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("failed to read line {}: {e}", line_no + 1),
            )
        })?;
        let line = line.trim();
        if line.is_empty() || line.starts_with("game,") {
            continue;
        }
        if line.starts_with("ply,") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "CSV has no game column; regenerate it with the current `probcut` command",
            ));
        }

        let sample = parse_row(line).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid CSV row at line {}: '{line}'", line_no + 1),
            )
        })?;
        dataset.add(sample);
    }

    Ok(dataset)
}

fn parse_row(line: &str) -> Option<Sample> {
    let mut fields = line.split(',');
    let game = fields.next()?.trim().parse().ok()?;
    let ply = fields.next()?.trim().parse().ok()?;
    let shallow = fields.next()?.trim().parse().ok()?;
    let shallow_score: f64 = fields.next()?.trim().parse().ok()?;
    let deep = fields.next()?.trim().parse().ok()?;
    let deep_score: f64 = fields.next()?.trim().parse().ok()?;
    // The diff column is recomputed from the scores; only its presence is checked.
    fields.next()?;
    // The ply bound keeps `Dataset::add` array indexing in range.
    if fields.next().is_some() || ply >= NUM_PLY {
        return None;
    }

    Some(Sample {
        game,
        ply,
        shallow,
        deep,
        diff: deep_score - shallow_score,
    })
}

fn game_fold(game: u64) -> usize {
    // SplitMix64 gives stable, well-mixed folds even when game ids are
    // consecutive input line numbers.
    let mut value = game.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) % NUM_FOLDS as u64) as usize
}

fn aggregate_folds(dataset: &Dataset, include: impl Fn(usize) -> bool) -> AggregatedData {
    let mut aggregated = AggregatedData::default();
    for (fold_index, fold) in dataset.folds.iter().enumerate() {
        if !include(fold_index) {
            continue;
        }
        for (&(ply, shallow, deep), &stats) in &fold.cells {
            aggregated.cells[ply]
                .entry((shallow, deep))
                .or_default()
                .add_scaled(stats, 1.0);
        }
        for (ply, positions) in fold.positions.iter().enumerate() {
            aggregated.positions[ply] += positions.len();
        }
    }
    aggregated
}

fn cells_for_ply(data: &AggregatedData, ply: usize) -> (Vec<(u32, u32, CellStats)>, bool) {
    let smoothed = data.positions[ply] < MIN_POSITIONS_WITHOUT_SMOOTHING;
    let mut cells: Vec<_> = if smoothed {
        let mut merged: HashMap<DepthPair, CellStats> = HashMap::new();
        let first = ply.saturating_sub(1);
        let last = (ply + 1).min(NUM_PLY - 1);
        for source_ply in first..=last {
            let weight = if source_ply == ply { 2.0 } else { 1.0 };
            for (&pair, &stats) in &data.cells[source_ply] {
                merged.entry(pair).or_default().add_scaled(stats, weight);
            }
        }
        merged
            .into_iter()
            .map(|((shallow, deep), stats)| (shallow, deep, stats))
            .collect()
    } else {
        data.cells[ply]
            .iter()
            .map(|(&(shallow, deep), &stats)| (shallow, deep, stats))
            .collect()
    };
    cells.sort_unstable_by_key(|&(shallow, deep, _)| (shallow, deep));
    (cells, smoothed)
}

fn fit_all_plies(data: &AggregatedData) -> [PlyFit; NUM_PLY] {
    let mut fits: [PlyFit; NUM_PLY] = std::array::from_fn(|ply| fit_ply(data, ply));
    carry_nearest_fitted_params(&mut fits);
    fits
}

/// Copies the nearest fitted ply's parameters into interior plies whose own
/// fit failed, so the search never sees a near-zero sentinel sigma at a
/// reachable ply. Plies 0 and 59 keep their sentinel rows; equidistant ties
/// resolve to the lower ply.
fn carry_nearest_fitted_params(fits: &mut [PlyFit; NUM_PLY]) {
    let fitted: Vec<usize> = (0..NUM_PLY).filter(|&ply| fits[ply].fitted).collect();
    for ply in 1..NUM_PLY - 1 {
        if fits[ply].fitted {
            continue;
        }
        let Some(&source) = fitted
            .iter()
            .min_by_key(|&&source| (source.abs_diff(ply), source))
        else {
            continue;
        };
        fits[ply].params = fits[source].params;
        fits[ply].carried_from = Some(source);
    }
}

fn fit_ply(data: &AggregatedData, ply: usize) -> PlyFit {
    if ply == 0 || ply == NUM_PLY - 1 {
        return PlyFit {
            positions: data.positions[ply],
            ..Default::default()
        };
    }

    let (cells, smoothed) = cells_for_ply(data, ply);
    let Some(params) = fit_group(&cells) else {
        return PlyFit {
            positions: data.positions[ply],
            cells: cells.len(),
            smoothed,
            ..Default::default()
        };
    };

    let legacy_mean = weighted_least_squares(cells.iter().map(|&(shallow, deep, stats)| {
        ([1.0, shallow as f64, deep as f64], stats.mean(), stats.n)
    }));
    let redesigned_residual_sigma =
        residual_sigma(&cells, |shallow, deep| predict_mean(params, shallow, deep));
    let legacy_residual_sigma = legacy_mean
        .map(|coefs| {
            residual_sigma(&cells, |shallow, deep| {
                coefs[0] + coefs[1] * shallow + coefs[2] * deep
            })
        })
        .unwrap_or(f64::NAN);

    PlyFit {
        params,
        positions: data.positions[ply],
        cells: cells.len(),
        smoothed,
        fitted: true,
        carried_from: None,
        legacy_residual_sigma,
        redesigned_residual_sigma,
    }
}

fn fit_group(cells: &[(u32, u32, CellStats)]) -> Option<FittedParams> {
    let mean_cells: Vec<_> = cells
        .iter()
        .filter(|cell| mean_model_shallow(cell.0))
        .copied()
        .collect();
    if mean_cells.len() < MIN_MEAN_CELLS {
        return None;
    }
    let mean = weighted_least_squares(mean_cells.iter().map(|&(shallow, deep, stats)| {
        (
            mean_features(shallow as f64, deep as f64),
            stats.mean(),
            stats.n,
        )
    }))?;

    let sigma_rows: Vec<_> = cells
        .iter()
        .filter_map(|&(shallow, deep, stats)| {
            stats.sample_std().map(|std| {
                (
                    sigma_features(shallow as f64, deep as f64),
                    std.ln(),
                    stats.n,
                )
            })
        })
        .collect();
    if sigma_rows.len() < MIN_SIGMA_CELLS {
        return None;
    }
    let sigma = weighted_least_squares(sigma_rows.into_iter())?;

    Some(FittedParams { mean, sigma })
}

/// Only even shallow depths inform the mean model; odd shallows carry a tempo
/// offset and contribute to sigma only.
fn mean_model_shallow(shallow: u32) -> bool {
    shallow & 1 == 0
}

fn mean_features(shallow: f64, deep: f64) -> [f64; 4] {
    [1.0, shallow, deep, ((deep as u32) & 1) as f64]
}

fn sigma_features(shallow: f64, deep: f64) -> [f64; 3] {
    [1.0, shallow, deep.sqrt()]
}

fn dot<const N: usize>(coefs: [f64; N], features: [f64; N]) -> f64 {
    coefs
        .iter()
        .zip(features)
        .map(|(&coef, feature)| coef * feature)
        .sum()
}

fn predict_mean(params: FittedParams, shallow: f64, deep: f64) -> f64 {
    dot(params.mean, mean_features(shallow, deep))
}

fn predict_ln_sigma(params: FittedParams, shallow: f64, deep: f64) -> f64 {
    dot(params.sigma, sigma_features(shallow, deep))
}

fn residual_sigma(cells: &[(u32, u32, CellStats)], predict: impl Fn(f64, f64) -> f64) -> f64 {
    let mut n = 0.0;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &(shallow, deep, stats) in cells {
        if !mean_model_shallow(shallow) {
            continue;
        }
        let predicted = predict(shallow as f64, deep as f64);
        n += stats.n;
        sum += stats.sum - stats.n * predicted;
        sum_sq += stats.sum_sq - 2.0 * predicted * stats.sum + stats.n * predicted * predicted;
    }
    if n == 0.0 {
        return f64::NAN;
    }
    (sum_sq / n - (sum / n).powi(2)).max(0.0).sqrt()
}

fn weighted_least_squares<const N: usize>(
    rows: impl Iterator<Item = ([f64; N], f64, f64)>,
) -> Option<[f64; N]> {
    let mut matrix = [[0.0f64; N]; N];
    let mut vector = [0.0f64; N];
    for (features, response, weight) in rows {
        for row in 0..N {
            for column in 0..N {
                matrix[row][column] += weight * features[row] * features[column];
            }
            vector[row] += weight * features[row] * response;
        }
    }
    solve(matrix, vector)
}

fn solve<const N: usize>(mut matrix: [[f64; N]; N], mut vector: [f64; N]) -> Option<[f64; N]> {
    let norm = matrix
        .iter()
        .flatten()
        .fold(0.0f64, |acc, &value| acc.max(value.abs()));
    let threshold = norm * 1e-12;

    for column in 0..N {
        let pivot = (column..N).max_by(|&left, &right| {
            matrix[left][column]
                .abs()
                .total_cmp(&matrix[right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= threshold {
            return None;
        }
        matrix.swap(column, pivot);
        vector.swap(column, pivot);

        let (pivot_rows, remaining) = matrix.split_at_mut(column + 1);
        let pivot_row = &pivot_rows[column];
        for (offset, row) in remaining.iter_mut().enumerate() {
            let factor = row[column] / pivot_row[column];
            for (destination, source) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                *destination -= factor * source;
            }
            vector[column + 1 + offset] -= factor * vector[column];
        }
    }

    let mut solution = [0.0f64; N];
    for row in (0..N).rev() {
        let mut value = vector[row];
        for (coefficient, solved) in matrix[row][row + 1..].iter().zip(&solution[row + 1..]) {
            value -= coefficient * solved;
        }
        solution[row] = value / matrix[row][row];
    }
    Some(solution)
}

fn cross_validation_rmse(fits: &[PlyFit; NUM_PLY], holdout: &AggregatedData) -> (f64, f64) {
    let mut mean_weight = 0.0;
    let mut mean_squared_error = 0.0;
    let mut sigma_weight = 0.0;
    let mut sigma_squared_error = 0.0;
    for (ply, fit) in fits.iter().enumerate().take(NUM_PLY - 1).skip(1) {
        if !fit.fitted && fit.carried_from.is_none() {
            continue;
        }
        for (&(shallow, deep), &stats) in &holdout.cells[ply] {
            if mean_model_shallow(shallow) {
                let error = stats.mean() - predict_mean(fit.params, shallow as f64, deep as f64);
                mean_weight += stats.n;
                mean_squared_error += stats.n * error * error;
            }
            if let Some(std) = stats.sample_std() {
                let error = std.ln() - predict_ln_sigma(fit.params, shallow as f64, deep as f64);
                sigma_weight += stats.n;
                sigma_squared_error += stats.n * error * error;
            }
        }
    }

    (
        (mean_squared_error / mean_weight).sqrt(),
        (sigma_squared_error / sigma_weight).sqrt(),
    )
}

fn report_cross_validation(dataset: &Dataset) {
    for holdout_fold in 0..NUM_FOLDS {
        let train = aggregate_folds(dataset, |fold| fold != holdout_fold);
        let holdout = aggregate_folds(dataset, |fold| fold == holdout_fold);
        let fits = fit_all_plies(&train);

        let (mean_rmse, sigma_rmse) = cross_validation_rmse(&fits, &holdout);
        eprintln!(
            "holdout fold {holdout_fold}: mean RMSE {mean_rmse:.4} discs, \
             ln-sigma RMSE {sigma_rmse:.4}"
        );
    }
}

fn report_in_sample(fits: &[PlyFit; NUM_PLY]) {
    for (ply, fit) in fits.iter().enumerate() {
        if ply == 0 || ply == NUM_PLY - 1 {
            eprintln!("ply {ply:2}: sentinel");
        } else if let Some(source) = fit.carried_from {
            eprintln!(
                "ply {ply:2}: {} positions, {} cells -> carried from ply {source}",
                fit.positions, fit.cells
            );
        } else if !fit.fitted {
            eprintln!(
                "ply {ply:2}: {} positions, {} cells -> sentinel (insufficient data)",
                fit.positions, fit.cells
            );
        } else {
            let shrink = 100.0 * (fit.legacy_residual_sigma - fit.redesigned_residual_sigma)
                / fit.legacy_residual_sigma;
            eprintln!(
                "ply {ply:2}: {} positions, {} cells{}; residual sigma {:.4} -> {:.4} ({shrink:+.2}%)",
                fit.positions,
                fit.cells,
                if fit.smoothed { " (±1 smoothed)" } else { "" },
                fit.legacy_residual_sigma,
                fit.redesigned_residual_sigma,
            );
        }
    }
}

fn emit_params(fits: &[PlyFit; NUM_PLY]) {
    println!("/// Statistical parameters for midgame ProbCut indexed by ply.");
    println!("#[rustfmt::skip]");
    println!("const PROBCUT_PARAMS: [ProbcutMidgameParams; {NUM_PLY}] = [");
    for fit in fits {
        let params = fit.params;
        println!("    ProbcutMidgameParams {{");
        println!("        mean_intercept: {:.10},", params.mean[0]);
        println!("        mean_coef_shallow: {:.10},", params.mean[1]);
        println!("        mean_coef_deep: {:.10},", params.mean[2]);
        println!("        mean_coef_parity: {:.10},", params.mean[3]);
        println!("        std_intercept: {:.10},", params.sigma[0]);
        println!("        std_coef_shallow: {:.10},", params.sigma[1]);
        println!("        std_coef_deep: {:.10},", params.sigma[2]);
        println!("    }},");
    }
    println!("];");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_cells(mean: [f64; 4], sigma: [f64; 3]) -> Vec<(u32, u32, CellStats)> {
        let mut cells = Vec::new();
        for shallow in 0..=7u32 {
            for deep in (shallow + 3)..=14 {
                let mut cell_mean = mean[0]
                    + mean[1] * shallow as f64
                    + mean[2] * deep as f64
                    + mean[3] * (deep & 1) as f64;
                if shallow & 1 != 0 {
                    // Odd shallows carry a deliberately different tempo term.
                    // They must inform sigma but not the runtime mean model.
                    cell_mean += 20.0 - 0.1 * shallow as f64;
                }
                let cell_std =
                    (sigma[0] + sigma[1] * shallow as f64 + sigma[2] * (deep as f64).sqrt()).exp();
                cells.push((
                    shallow,
                    deep,
                    CellStats {
                        n: 4.0,
                        sum: 4.0 * cell_mean,
                        // Four symmetric observations with sample std `cell_std`.
                        sum_sq: 4.0 * cell_mean * cell_mean + 3.0 * cell_std * cell_std,
                    },
                ));
            }
        }
        cells
    }

    #[test]
    fn parses_explicit_game_ids_and_recomputes_diff() {
        let sample = parse_row("17,12,2,1.25,7,-0.5,999").unwrap();

        assert_eq!(sample.game, 17);
        assert_eq!(sample.ply, 12);
        assert_eq!(sample.shallow, 2);
        assert_eq!(sample.deep, 7);
        assert_eq!(sample.diff, -1.75);
    }

    #[test]
    fn fit_uses_even_shallows_for_mean_and_all_shallows_for_sigma() {
        let expected_mean = [0.35, -0.04, 0.02, 1.2];
        let expected_sigma = [0.8, -0.06, 0.3];
        let fit = fit_group(&synthetic_cells(expected_mean, expected_sigma)).unwrap();

        for (actual, expected) in fit.mean.into_iter().zip(expected_mean) {
            assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
        }
        for (actual, expected) in fit.sigma.into_iter().zip(expected_sigma) {
            assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
        }
    }

    #[test]
    fn sparse_ply_uses_triangular_neighbor_weights() {
        let mut data = AggregatedData::default();
        data.positions[10] = MIN_POSITIONS_WITHOUT_SMOOTHING - 1;
        data.cells[9].insert(
            (2, 7),
            CellStats {
                n: 3.0,
                sum: 3.0,
                sum_sq: 3.0,
            },
        );
        data.cells[10].insert(
            (2, 7),
            CellStats {
                n: 5.0,
                sum: 10.0,
                sum_sq: 20.0,
            },
        );
        data.cells[11].insert(
            (2, 7),
            CellStats {
                n: 7.0,
                sum: 21.0,
                sum_sq: 63.0,
            },
        );

        let (cells, smoothed) = cells_for_ply(&data, 10);

        assert!(smoothed);
        assert_eq!(cells.len(), 1);
        let stats = cells[0].2;
        assert_eq!(stats.n, 3.0 + 2.0 * 5.0 + 7.0);
        assert_eq!(stats.sum, 3.0 + 2.0 * 10.0 + 21.0);
        assert_eq!(stats.sum_sq, 3.0 + 2.0 * 20.0 + 63.0);
    }

    #[test]
    fn game_folds_are_stable_and_non_degenerate() {
        // Pinned values guard the cross-run stability that game-blocked splits
        // rely on.
        assert_eq!(
            (1..=8).map(game_fold).collect::<Vec<_>>(),
            [0, 0, 3, 3, 3, 2, 2, 2]
        );
        let assignments: Vec<_> = (1..=100).map(game_fold).collect();
        for fold in 0..NUM_FOLDS {
            assert!(assignments.contains(&fold));
        }
    }

    #[test]
    fn game_split_precedes_cell_aggregation() {
        let first_game = 1;
        let first_fold = game_fold(first_game);
        let second_game = (2..).find(|&game| game_fold(game) != first_fold).unwrap();
        let mut dataset = Dataset::default();
        dataset.add(Sample {
            game: first_game,
            ply: 12,
            shallow: 2,
            deep: 7,
            diff: 1.0,
        });
        dataset.add(Sample {
            game: second_game,
            ply: 12,
            shallow: 2,
            deep: 7,
            diff: 9.0,
        });

        let train = aggregate_folds(&dataset, |fold| fold != first_fold);
        let cell = train.cells[12][&(2, 7)];

        assert_eq!(cell.n, 1.0);
        assert_eq!(cell.mean(), 9.0);
        assert_eq!(train.positions[12], 1);
    }

    #[test]
    fn endpoint_plies_remain_sentinels() {
        let data = AggregatedData::default();

        assert!(!fit_ply(&data, 0).fitted);
        assert_eq!(fit_ply(&data, 0).params.mean, [0.0; 4]);
        assert!(!fit_ply(&data, NUM_PLY - 1).fitted);
        assert_eq!(
            fit_ply(&data, NUM_PLY - 1).params.sigma,
            [DEFAULT_STD_INTERCEPT, 0.0, 0.0]
        );
    }

    #[test]
    fn unfitted_interior_plies_carry_the_nearest_fitted_params() {
        let mut fits: [PlyFit; NUM_PLY] = std::array::from_fn(|_| PlyFit::default());
        let lower = FittedParams {
            mean: [1.0, 0.0, 0.0, 0.0],
            sigma: [0.1, 0.0, 0.0],
        };
        let upper = FittedParams {
            mean: [2.0, 0.0, 0.0, 0.0],
            sigma: [0.2, 0.0, 0.0],
        };
        fits[10] = PlyFit {
            params: lower,
            fitted: true,
            ..Default::default()
        };
        fits[20] = PlyFit {
            params: upper,
            fitted: true,
            ..Default::default()
        };

        carry_nearest_fitted_params(&mut fits);

        assert_eq!(fits[12].carried_from, Some(10));
        assert_eq!(fits[12].params.sigma, lower.sigma);
        assert_eq!(fits[17].carried_from, Some(20));
        assert_eq!(fits[17].params.mean, upper.mean);
        // Equidistant ties resolve to the lower ply.
        assert_eq!(fits[15].carried_from, Some(10));
        assert_eq!(fits[0].carried_from, None);
        assert_eq!(fits[0].params.sigma, [DEFAULT_STD_INTERCEPT, 0.0, 0.0]);
        assert_eq!(fits[59].carried_from, None);
        assert_eq!(fits[59].params.sigma, [DEFAULT_STD_INTERCEPT, 0.0, 0.0]);
    }

    #[test]
    fn execute_rejects_a_nonempty_dataset_when_no_ply_can_be_fitted() {
        let path = std::env::temp_dir().join(format!(
            "neural-reversi-fit-probcut-no-fit-{}.csv",
            std::process::id()
        ));
        std::fs::write(
            &path,
            "game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n\
             1,10,0,0,3,1,1\n",
        )
        .unwrap();

        let result = execute(path.to_str().unwrap());
        std::fs::remove_file(path).unwrap();
        let error = result.expect_err("an all-sentinel parameter table must not be emitted");

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("no ply could be fitted"));
    }

    #[test]
    fn cross_validation_evaluates_carried_params() {
        let mut fits: [PlyFit; NUM_PLY] = std::array::from_fn(|_| PlyFit::default());
        fits[10] = PlyFit {
            params: FittedParams {
                mean: [0.0; 4],
                sigma: [0.0; 3],
            },
            fitted: true,
            ..Default::default()
        };
        fits[11] = PlyFit {
            params: fits[10].params,
            carried_from: Some(10),
            ..Default::default()
        };
        let mut holdout = AggregatedData::default();
        holdout.cells[11].insert(
            (0, 3),
            CellStats {
                n: 4.0,
                sum: 0.0,
                sum_sq: 3.0,
            },
        );

        let (mean_rmse, sigma_rmse) = cross_validation_rmse(&fits, &holdout);

        assert_eq!(mean_rmse, 0.0);
        assert_eq!(sigma_rmse, 0.0);
    }
}
