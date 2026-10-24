//! Evaluation Test Suite Runner for Reversi AI Engines
//!
//! This program runs evaluation test suites loaded from OBF (Othello Board Format) files,
//! containing challenging positions used to evaluate the solving capabilities
//! of Reversi/Othello engines.
//!
//! The test runner measures:
//! - Solving accuracy (correct score and best move)
//! - Search performance (time and nodes)
//! - Move selection quality (best move percentage)

mod test_case;

use clap::Parser;
use colored::Colorize;
use num_format::{Locale, ToFormattedString};
use reversi_core::{
    self,
    board::Board,
    disc::Disc,
    level::{ENDGAME_STEPS, Level},
    probcut::Selectivity,
    search::{
        self, SearchProgress, SearchRunOptions,
        counters::SearchCounters,
        options::SearchOptions,
        result::{PvMove, SearchResult},
    },
    square::Square,
    types::{Depth, Scoref},
};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use test_case::{TestCase, find_problem_dir, load_all_problems, load_problems};

/// Score tolerance levels for evaluation
const SCORE_TOLERANCE_PERFECT: Scoref = 3.0;
const SCORE_TOLERANCE_GOOD: Scoref = 6.0;
const SCORE_TOLERANCE_ACCEPTABLE: Scoref = 9.0;

/// Performance thresholds for color coding
const PERFORMANCE_EXCELLENT: f64 = 80.0;
const PERFORMANCE_GOOD: f64 = 60.0;

/// Collected iteration progress data for verbose output.
#[derive(Debug)]
struct IterationData {
    depth: Depth,
    score: Scoref,
    best_move: Square,
    nodes: u64,
    probability: i32,
    counters: SearchCounters,
    tt_fill: f64,
    is_endgame: bool,
    elapsed: std::time::Duration,
}

/// Accuracy and cost of one endgame selectivity level, accumulated over test cases.
///
/// Nodes and time are the level's own share: the delta from the previous
/// iteration, so earlier levels and midgame iterations are not counted again.
#[derive(Default, Clone, Copy)]
struct LevelStats {
    cases: usize,
    exact: usize,
    wdl_flips: usize,
    abs_error_sum: f64,
    nodes: u64,
    time: std::time::Duration,
    moves_judged: usize,
    optimal_moves: usize,
    move_loss_sum: f64,
}

impl LevelStats {
    fn merge(&mut self, other: &LevelStats) {
        self.moves_judged += other.moves_judged;
        self.optimal_moves += other.optimal_moves;
        self.move_loss_sum += other.move_loss_sum;
        self.cases += other.cases;
        self.exact += other.exact;
        self.wdl_flips += other.wdl_flips;
        self.abs_error_sum += other.abs_error_sum;
        self.nodes += other.nodes;
        self.time += other.time;
    }
}

/// Result of a single test case execution
#[derive(Debug)]
struct TestResult {
    elapsed: std::time::Duration,
    nodes: u64,
    score: Scoref,
    depth: Depth,
    selectivity: Selectivity,
    pv_line: Vec<Square>,
    score_difference: Scoref,
    move_accuracy: MoveAccuracy,
    counters: SearchCounters,
    pv_moves: Vec<PvMove>,
}

/// Move accuracy classification
#[derive(Debug, Clone, Copy)]
enum MoveAccuracy {
    Best,
    SecondBest,
    ThirdBest,
    Other,
}

/// Statistics collector for aggregating search results across all test cases
#[derive(Default)]
struct SearchStats {
    total_time: std::time::Duration,
    total_nodes: u64,
    total_count: usize,
    score_differences: Vec<Scoref>,
    best_move_count: usize,
    top2_move_count: usize,
    top3_move_count: usize,
    perfect_score_count: usize,
    good_score_count: usize,
    acceptable_score_count: usize,
    total_counters: SearchCounters,
    mpv_score_differences: Vec<Scoref>,
    mpv_perfect_count: usize,
    mpv_good_count: usize,
    mpv_acceptable_count: usize,
    /// Endgame levels keyed by their probability percentage (100 = exact).
    levels: std::collections::BTreeMap<i32, LevelStats>,
}

/// Mean absolute error and root mean squared error of a slice of absolute
/// error samples. Returns `(0.0, 0.0)` for an empty slice.
fn mae_rmse(samples: &[Scoref]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let n = samples.len() as f64;
    let mae = samples.iter().map(|&d| d as f64).sum::<f64>() / n;
    let mse = samples
        .iter()
        .map(|&d| (d as f64) * (d as f64))
        .sum::<f64>()
        / n;
    (mae, mse.sqrt())
}

/// Round duration to 0.01ms precision for consistent display
fn round_duration(elapsed: std::time::Duration) -> std::time::Duration {
    let rounded_micros = (elapsed.as_secs_f64() * 10000.0).round() as u64 * 100;
    std::time::Duration::from_micros(rounded_micros)
}

/// Format a ratio as "XX.X% (n/total)", or "N/A" when total is zero
fn format_ratio(count: usize, total: usize) -> String {
    if total > 0 {
        format!(
            "{:.1}% ({}/{})",
            (count as f64 / total as f64) * 100.0,
            count,
            total
        )
    } else {
        "N/A".to_string()
    }
}

impl SearchStats {
    /// Update statistics with results from a single test case
    fn update(&mut self, result: &TestResult) {
        self.total_time += round_duration(result.elapsed);
        self.total_nodes += result.nodes;
        self.total_count += 1;

        self.score_differences.push(result.score_difference);

        if result.score_difference <= SCORE_TOLERANCE_PERFECT {
            self.perfect_score_count += 1;
        }
        if result.score_difference <= SCORE_TOLERANCE_GOOD {
            self.good_score_count += 1;
        }
        if result.score_difference <= SCORE_TOLERANCE_ACCEPTABLE {
            self.acceptable_score_count += 1;
        }

        match result.move_accuracy {
            MoveAccuracy::Best => {
                self.best_move_count += 1;
                self.top2_move_count += 1;
                self.top3_move_count += 1;
            }
            MoveAccuracy::SecondBest => {
                self.top2_move_count += 1;
                self.top3_move_count += 1;
            }
            MoveAccuracy::ThirdBest => {
                self.top3_move_count += 1;
            }
            MoveAccuracy::Other => {}
        }

        self.total_counters.merge(&result.counters);
    }

    /// Accumulate per-move error statistics against OBF-listed expected scores.
    fn update_multipv(&mut self, result: &TestResult, test_case: &TestCase) {
        if test_case.is_pass() {
            return;
        }
        for pv in &result.pv_moves {
            let Some(expected) = test_case.expected_score_for_move(pv.sq) else {
                continue;
            };
            let rounded = (pv.score * 10.0).round() / 10.0;
            let diff = (rounded - expected as Scoref).abs();
            self.mpv_score_differences.push(diff);
            if diff <= SCORE_TOLERANCE_PERFECT {
                self.mpv_perfect_count += 1;
            }
            if diff <= SCORE_TOLERANCE_GOOD {
                self.mpv_good_count += 1;
            }
            if diff <= SCORE_TOLERANCE_ACCEPTABLE {
                self.mpv_acceptable_count += 1;
            }
        }
    }

    /// Classify the move accuracy based on the search result
    fn classify_move(result: &SearchResult, test_case: &TestCase) -> MoveAccuracy {
        let Some(&first_pv) = result.pv_line().first() else {
            return MoveAccuracy::Other;
        };
        match test_case.rank_of(first_pv) {
            Some(0) => MoveAccuracy::Best,
            Some(1) => MoveAccuracy::SecondBest,
            Some(2) => MoveAccuracy::ThirdBest,
            _ => MoveAccuracy::Other,
        }
    }

    /// Calculate MAE and RMSE of score differences
    fn calculate_statistics(&self) -> (f64, f64) {
        mae_rmse(&self.score_differences)
    }

    /// Merge another SearchStats into this one
    fn merge(&mut self, other: &SearchStats) {
        self.total_time += other.total_time;
        self.total_nodes += other.total_nodes;
        self.total_count += other.total_count;
        self.score_differences.extend(&other.score_differences);
        self.best_move_count += other.best_move_count;
        self.top2_move_count += other.top2_move_count;
        self.top3_move_count += other.top3_move_count;
        self.perfect_score_count += other.perfect_score_count;
        self.good_score_count += other.good_score_count;
        self.acceptable_score_count += other.acceptable_score_count;
        self.total_counters.merge(&other.total_counters);
        self.mpv_score_differences
            .extend(&other.mpv_score_differences);
        self.mpv_perfect_count += other.mpv_perfect_count;
        self.mpv_good_count += other.mpv_good_count;
        self.mpv_acceptable_count += other.mpv_acceptable_count;
        for (&probability, level) in &other.levels {
            self.levels.entry(probability).or_default().merge(level);
        }
    }

    /// Accumulate per-level accuracy and cost. `move_values` holds the exact
    /// value of each chosen move, used for the move loss.
    ///
    /// Levels are judged against the OBF score, or, if given, `reference`
    /// (the run's own exact solve) for positions without exact scores.
    fn update_levels(
        &mut self,
        iterations: &[IterationData],
        test_case: &TestCase,
        move_values: &std::collections::HashMap<Square, Scoref>,
        reference: Option<&IterationData>,
    ) {
        let n_empties = test_case.board().empty_count();
        let orient = |score: Scoref| if test_case.is_pass() { -score } else { score };
        let expected = reference.map_or(test_case.expected_score() as Scoref, |exact| {
            orient(exact.score)
        });
        let mut prev_nodes = 0;
        let mut prev_elapsed = std::time::Duration::ZERO;
        for iter in dedup_iterations(iterations) {
            let (delta_nodes, delta_time) = (iter.nodes - prev_nodes, iter.elapsed - prev_elapsed);
            prev_nodes = iter.nodes;
            prev_elapsed = iter.elapsed;
            if !iter.is_endgame || iter.depth != n_empties {
                continue;
            }
            let score = orient(iter.score);
            let error = ((score * 10.0).round() / 10.0 - expected).abs();
            let level = self.levels.entry(iter.probability).or_default();
            level.cases += 1;
            level.exact += usize::from(error < 0.05);
            level.wdl_flips += usize::from(score.partial_cmp(&0.0) != expected.partial_cmp(&0.0));
            level.abs_error_sum += error as f64;
            level.nodes += delta_nodes;
            level.time += delta_time;
            if let Some(&value) = move_values.get(&iter.best_move) {
                let loss = (expected - value).max(0.0);
                level.moves_judged += 1;
                level.optimal_moves += usize::from(loss < 0.05);
                level.move_loss_sum += loss as f64;
            }
        }
    }

    fn print_levels(&self) {
        if self.levels.is_empty() {
            return;
        }
        println!("\n### Endgame Levels:");
        println!(
            "| level | cases | exact | MAE | W/D/L flips | best move | move loss | nodes | time |"
        );
        println!(
            "| ----- | ----: | ----: | --: | ----------: | --------: | --------: | ----: | ---: |"
        );
        for (probability, level) in &self.levels {
            let (best_move, move_loss) = if level.moves_judged > 0 {
                (
                    format_ratio(level.optimal_moves, level.moves_judged),
                    format!("{:.3}", level.move_loss_sum / level.moves_judged as f64),
                )
            } else {
                ("-".to_string(), "-".to_string())
            };
            println!(
                "| {probability}% | {} | {} | {:.3} | {} | {best_move} | {move_loss} | {} | {:.4}s |",
                level.cases,
                format_ratio(level.exact, level.cases),
                level.abs_error_sum / level.cases as f64,
                level.wdl_flips,
                level.nodes.to_formatted_string(&Locale::en),
                level.time.as_secs_f64(),
            );
        }
    }

    /// Print formatted statistics summary with color coding
    fn print(&self, verbose: bool, multipv: bool) {
        if self.total_count == 0 {
            println!("\n### No test cases were run");
            return;
        }

        let (mae, rmse) = self.calculate_statistics();
        let nps = if self.total_time.as_secs_f64() > 0.0 {
            (self.total_nodes as f64 / self.total_time.as_secs_f64()).round() as u64
        } else {
            0
        };

        let stats = [
            (
                "Total time",
                format!("{:.4}s", self.total_time.as_secs_f64()),
            ),
            (
                "Total nodes",
                self.total_nodes.to_formatted_string(&Locale::en),
            ),
            ("NPS", nps.to_formatted_string(&Locale::en)),
            (
                "Best move",
                format_ratio(self.best_move_count, self.total_count),
            ),
            (
                "Top 2 move",
                format_ratio(self.top2_move_count, self.total_count),
            ),
            (
                "Top 3 move",
                format_ratio(self.top3_move_count, self.total_count),
            ),
            (
                "Score ±3",
                format_ratio(self.perfect_score_count, self.total_count),
            ),
            (
                "Score ±6",
                format_ratio(self.good_score_count, self.total_count),
            ),
            (
                "Score ±9",
                format_ratio(self.acceptable_score_count, self.total_count),
            ),
            ("MAE", format!("{mae:.2}")),
            ("RMSE", format!("{rmse:.2}")),
        ];

        let max_label_len = stats
            .iter()
            .map(|(label, _)| label.len())
            .max()
            .unwrap_or(0);

        println!("\n### Statistics:");
        for (label, value) in stats {
            let formatted_value = match label {
                "Best move" | "Top 2 move" | "Top 3 move" | "Score ±3" | "Score ±6"
                | "Score ±9" => Self::colorize_percentage(&value),
                _ => value.normal(),
            };
            println!("- {label:<max_label_len$}: {formatted_value}");
        }

        self.print_levels();

        if verbose {
            let c = &self.total_counters;
            let counter_stats = [
                (
                    "TT hit rate",
                    format_rate_with_counts(c.tt_hits, c.tt_probes),
                ),
                (
                    "ProbCut rate",
                    format_rate_with_counts(c.probcut_cuts, c.probcut_attempts),
                ),
                (
                    "ETC rate",
                    format_rate_with_counts(c.etc_cuts, c.etc_attempts),
                ),
                (
                    "Stability cuts",
                    c.stability_cuts.to_formatted_string(&Locale::en),
                ),
            ];
            println!("\n### Search Counters:");
            for (label, value) in counter_stats {
                println!("- {label:<14}: {value}");
            }
        }

        let mpv_count = self.mpv_score_differences.len();
        if multipv && mpv_count > 0 {
            let (mpv_mae, _) = mae_rmse(&self.mpv_score_differences);
            let mpv_stats = [
                ("Moves compared", mpv_count.to_formatted_string(&Locale::en)),
                ("Move MAE", format!("{mpv_mae:.2}")),
                ("Move ±3", format_ratio(self.mpv_perfect_count, mpv_count)),
                ("Move ±6", format_ratio(self.mpv_good_count, mpv_count)),
                (
                    "Move ±9",
                    format_ratio(self.mpv_acceptable_count, mpv_count),
                ),
            ];
            let max_label_len = mpv_stats
                .iter()
                .map(|(label, _)| label.len())
                .max()
                .unwrap_or(0);
            println!("\n### Multi-PV Statistics:");
            for (label, value) in mpv_stats {
                let formatted_value = match label {
                    "Move ±3" | "Move ±6" | "Move ±9" => Self::colorize_percentage(&value),
                    _ => value.normal(),
                };
                println!("- {label:<max_label_len$}: {formatted_value}");
            }
        }
    }

    /// Apply color coding to percentage values
    fn colorize_percentage(value: &str) -> colored::ColoredString {
        if let Some(percentage_str) = value.split('%').next()
            && let Ok(percentage) = percentage_str.parse::<f64>()
        {
            if percentage >= PERFORMANCE_EXCELLENT {
                return value.bright_green();
            } else if percentage >= PERFORMANCE_GOOD {
                return value.bright_yellow();
            } else {
                return value.bright_red();
            }
        }
        value.normal()
    }
}

/// Print the table header for test results
fn print_header(num_width: usize) {
    println!(
        "| {:^num_width$} | {:^6} | {:^9} | {:^15} | {:^13} | {:^8} | {:^6} | {:<32} |",
        "#", "Depth", "Time(s)", "Nodes", "NPS", "Line", "Score", "Expected"
    );
    let dashes = "-".repeat(num_width);
    println!(
        "|{dashes}-:|-------:|----------:|----------------:|--------------:|:---------|-------:|:---------------------------------|"
    );
}

/// Format depth information including probability if not 100%
fn format_depth(depth: Depth, selectivity: Selectivity) -> String {
    let probability = selectivity.probability();
    if probability == 100 {
        format!("{depth}")
    } else {
        format!("{depth}@{probability}%")
    }
}

/// Format principal variation line (top 3 moves)
fn format_pv_line(pv_line: &[Square], max_moves: usize) -> String {
    pv_line
        .iter()
        .take(max_moves)
        .map(|sq| format!("{sq:?}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Apply color coding to score based on accuracy
fn colorize_score(score: Scoref, difference: Scoref) -> colored::ColoredString {
    let score_str = format!("{score:.1}");
    if difference <= SCORE_TOLERANCE_PERFECT {
        score_str.bright_green()
    } else if difference <= SCORE_TOLERANCE_GOOD {
        score_str.bright_yellow()
    } else if difference <= SCORE_TOLERANCE_ACCEPTABLE {
        score_str.bright_cyan()
    } else {
        score_str.bright_red()
    }
}

/// Apply color coding to move based on accuracy
fn colorize_move(move_str: String, accuracy: MoveAccuracy) -> colored::ColoredString {
    match accuracy {
        MoveAccuracy::Best => move_str.bright_green(),
        MoveAccuracy::SecondBest => move_str.bright_yellow(),
        MoveAccuracy::ThirdBest => move_str.bright_cyan(),
        MoveAccuracy::Other => move_str.bright_red(),
    }
}

/// Execute a single test case and return the result
fn execute_test_case(
    test_case: &TestCase,
    search: &mut search::Search,
    level: Level,
    verbose: bool,
    multipv: bool,
) -> (TestResult, Vec<IterationData>) {
    let board = test_case.board();
    // For pass positions, switch sides and search from the opponent's perspective
    let search_board = if test_case.is_pass() {
        board.switch_players()
    } else {
        board
    };
    search.init();

    let iterations: Arc<Mutex<Vec<IterationData>>> = Arc::new(Mutex::new(Vec::new()));
    let iter_clone = iterations.clone();
    let tt = search.tt().clone();
    let start = Instant::now();
    let options = SearchRunOptions::with_level(level)
        .multi_pv(multipv)
        .callback(move |progress: SearchProgress| {
            let elapsed = start.elapsed();
            // Usage scans the table, so only verbose output pays for it.
            let tt_fill = if verbose { tt.usage_rate() } else { 0.0 };
            iter_clone.lock().unwrap().push(IterationData {
                depth: progress.depth,
                score: progress.score,
                best_move: progress.best_move,
                nodes: progress.nodes,
                probability: progress.probability,
                counters: progress.counters,
                tt_fill,
                is_endgame: progress.is_endgame,
                elapsed,
            });
        });

    let result = search.run(&search_board, &options);
    let elapsed = start.elapsed();

    // For pass positions, negate the score back to the original player's perspective.
    let raw_score = result.score().expect("search returned no legal move");
    let score = if test_case.is_pass() {
        -raw_score
    } else {
        raw_score
    };
    let score = (score * 10.0).round() / 10.0;
    let score_difference = (score - test_case.expected_score() as Scoref).abs();
    let move_accuracy = if test_case.is_pass() {
        MoveAccuracy::Best // Pass positions don't have move choices
    } else {
        SearchStats::classify_move(&result, test_case)
    };

    let test_result = TestResult {
        elapsed,
        nodes: result.n_nodes(),
        score,
        depth: result.depth(),
        selectivity: result.selectivity(),
        pv_line: result.pv_line().to_vec(),
        score_difference,
        move_accuracy,
        counters: result.counters(),
        pv_moves: result.pv_moves().to_vec(),
    };

    let iterations = std::mem::take(&mut *iterations.lock().unwrap());
    (test_result, iterations)
}

/// Compute common display metrics from a test result.
fn compute_result_metrics(
    test_case: &TestCase,
    result: &TestResult,
) -> (f64, u64, colored::ColoredString, colored::ColoredString) {
    let rounded_secs = round_duration(result.elapsed).as_secs_f64();
    let nps = if rounded_secs > 0.0 {
        (result.nodes as f64 / rounded_secs).round() as u64
    } else {
        0
    };
    let pv_line_str = if test_case.is_pass() {
        "PS".to_string()
    } else {
        format_pv_line(&result.pv_line, 3)
    };
    let pv_colored = colorize_move(pv_line_str, result.move_accuracy);
    let score_colored = colorize_score(result.score, result.score_difference);
    (rounded_secs, nps, pv_colored, score_colored)
}

/// Print a single test result row
fn print_test_result(test_case: &TestCase, result: &TestResult, num_width: usize) {
    let (rounded_secs, nps, pv_line_colored, score_colored) =
        compute_result_metrics(test_case, result);
    let nodes_formatted = result.nodes.to_formatted_string(&Locale::en);
    let nps_formatted = nps.to_formatted_string(&Locale::en);
    let depth_str = format_depth(result.depth, result.selectivity);
    let expected = format!(
        "{:>3} : {}",
        test_case.expected_score(),
        test_case.expected_moves_str()
    );

    println!(
        "| {:>num_width$} | {:^6} | {:>9.4} | {:>15} | {:>13} | {:<8} | {:>6} | {:<32} |",
        test_case.line_number,
        depth_str,
        rounded_secs,
        nodes_formatted,
        nps_formatted,
        pv_line_colored,
        score_colored,
        expected
    );
}

/// Deduplicate consecutive iterations with the same depth, selectivity, and
/// node count.
///
/// `Search::run()` emits a final progress callback that repeats the last
/// iteration; this drops it. A level that adds no nodes still differs in
/// selectivity, so it is kept.
fn dedup_iterations(iterations: &[IterationData]) -> Vec<&IterationData> {
    let mut result = Vec::with_capacity(iterations.len());
    for iter in iterations {
        if result.last().is_none_or(|prev: &&IterationData| {
            (prev.nodes, prev.depth, prev.probability, prev.is_endgame)
                != (iter.nodes, iter.depth, iter.probability, iter.is_endgame)
        }) {
            result.push(iter);
        }
    }
    result
}

/// Format a rate as "X.Y%" or "-" if the denominator is zero.
fn format_rate(numerator: u64, denominator: u64) -> String {
    if denominator > 0 {
        format!("{:.1}%", numerator as f64 / denominator as f64 * 100.0)
    } else {
        "-".to_string()
    }
}

/// Format a rate as "X.Y% (num/denom)" with thousand-separated counts, or "-" if zero.
fn format_rate_with_counts(numerator: u64, denominator: u64) -> String {
    if denominator > 0 {
        format!(
            "{} ({}/{})",
            format_rate(numerator, denominator),
            numerator.to_formatted_string(&Locale::en),
            denominator.to_formatted_string(&Locale::en),
        )
    } else {
        "-".to_string()
    }
}

/// A pre-formatted row for the verbose markdown iteration table.
struct IterationRow {
    depth: String,
    score: String,
    best: String,
    nodes: String,
    time: String,
    tt_rate: String,
    pc_rate: String,
    etc_rate: String,
    stab: String,
}

impl IterationRow {
    /// Format this row as a markdown table row with the given column widths.
    fn format(&self, w: &ColWidths) -> String {
        format!(
            "| {:<dw$} | {:>scw$} | {:>bw$} | {:>nw$} | {:>tiw$} | {:>tw$} | {:>pw$} | {:>ew$} | {:>sw$} |",
            self.depth,
            self.score,
            self.best,
            self.nodes,
            self.time,
            self.tt_rate,
            self.pc_rate,
            self.etc_rate,
            self.stab,
            dw = w.depth,
            scw = w.score,
            bw = w.best,
            nw = w.nodes,
            tiw = w.time,
            tw = w.tt_rate,
            pw = w.pc_rate,
            ew = w.etc_rate,
            sw = w.stab,
        )
    }
}

/// Column widths for the markdown iteration table.
struct ColWidths {
    depth: usize,
    score: usize,
    best: usize,
    nodes: usize,
    time: usize,
    tt_rate: usize,
    pc_rate: usize,
    etc_rate: usize,
    stab: usize,
}

impl ColWidths {
    fn compute(header: &IterationRow, rows: &[IterationRow], total: &IterationRow) -> Self {
        let max_of = |f: fn(&IterationRow) -> usize| -> usize {
            rows.iter()
                .map(f)
                .max()
                .unwrap_or(0)
                .max(f(total))
                .max(f(header))
        };
        ColWidths {
            depth: max_of(|r| r.depth.len()),
            score: max_of(|r| r.score.len()),
            best: max_of(|r| r.best.len()),
            nodes: max_of(|r| r.nodes.len()),
            time: max_of(|r| r.time.len()),
            tt_rate: max_of(|r| r.tt_rate.len()),
            pc_rate: max_of(|r| r.pc_rate.len()),
            etc_rate: max_of(|r| r.etc_rate.len()),
            stab: max_of(|r| r.stab.len()),
        }
    }

    /// Format the markdown separator row.
    fn separator(&self) -> String {
        let sep = |w: usize| format!("{:-<w$}:", "", w = w - 1);
        format!(
            "| {:-<dw$} | {} | {} | {} | {} | {} | {} | {} | {} |",
            "",
            sep(self.score),
            sep(self.best),
            sep(self.nodes),
            sep(self.time),
            sep(self.tt_rate),
            sep(self.pc_rate),
            sep(self.etc_rate),
            sep(self.stab),
            dw = self.depth,
        )
    }
}

/// Print verbose iteration progress for a test case as a markdown table.
///
/// Each row shows per-iteration deltas (not cumulative totals).
/// A final "total" row shows cumulative values for the entire search.
fn print_verbose_iterations(iterations: &[IterationData]) {
    let iterations = dedup_iterations(iterations);
    if iterations.is_empty() {
        return;
    }

    // Pre-format all rows using per-iteration deltas.
    let mut prev_nodes: u64 = 0;
    let mut prev_elapsed = std::time::Duration::ZERO;
    let mut prev_counters = SearchCounters::default();
    let rows: Vec<IterationRow> = iterations
        .iter()
        .map(|iter| {
            let delta_nodes = iter.nodes - prev_nodes;
            let delta_time = iter.elapsed - prev_elapsed;
            prev_elapsed = iter.elapsed;
            let delta_tt_hits = iter.counters.tt_hits - prev_counters.tt_hits;
            let delta_tt_probes = iter.counters.tt_probes - prev_counters.tt_probes;
            let delta_pc_cuts = iter.counters.probcut_cuts - prev_counters.probcut_cuts;
            let delta_pc_attempts = iter.counters.probcut_attempts - prev_counters.probcut_attempts;
            let delta_etc_cuts = iter.counters.etc_cuts - prev_counters.etc_cuts;
            let delta_etc_attempts = iter.counters.etc_attempts - prev_counters.etc_attempts;
            let delta_stab = iter.counters.stability_cuts - prev_counters.stability_cuts;

            prev_nodes = iter.nodes;
            prev_counters = iter.counters.clone();

            let depth = if iter.probability < 100 {
                format!("{}@{}%", iter.depth, iter.probability)
            } else {
                iter.depth.to_string()
            };
            IterationRow {
                depth,
                score: format!("{:>+.1}", iter.score),
                best: format!("{:?}", iter.best_move),
                nodes: delta_nodes.to_formatted_string(&Locale::en),
                time: format!("{:.4}s", delta_time.as_secs_f64()),
                tt_rate: format_rate(delta_tt_hits, delta_tt_probes),
                pc_rate: format_rate(delta_pc_cuts, delta_pc_attempts),
                etc_rate: format_rate(delta_etc_cuts, delta_etc_attempts),
                stab: delta_stab.to_formatted_string(&Locale::en),
            }
        })
        .collect();

    // Total row from last iteration's cumulative values.
    let last = iterations.last().unwrap();
    let total_row = IterationRow {
        depth: "total".to_string(),
        score: String::new(),
        best: String::new(),
        nodes: last.nodes.to_formatted_string(&Locale::en),
        time: format!("{:.4}s", last.elapsed.as_secs_f64()),
        tt_rate: format_rate(last.counters.tt_hits, last.counters.tt_probes),
        pc_rate: format_rate(last.counters.probcut_cuts, last.counters.probcut_attempts),
        etc_rate: format_rate(last.counters.etc_cuts, last.counters.etc_attempts),
        stab: last
            .counters
            .stability_cuts
            .to_formatted_string(&Locale::en),
    };

    let header = IterationRow {
        depth: "depth".into(),
        score: "score".into(),
        best: "best".into(),
        nodes: "nodes".into(),
        time: "time".into(),
        tt_rate: "TT%".into(),
        pc_rate: "ProbCut%".into(),
        etc_rate: "ETC%".into(),
        stab: "Stab".into(),
    };
    let w = ColWidths::compute(&header, &rows, &total_row);
    println!("{}", header.format(&w));
    println!("{}", w.separator());

    for row in &rows {
        println!("{}", row.format(&w));
    }

    println!("{}", total_row.format(&w));
}

/// Format a board with coordinates for verbose display.
fn format_board_with_coords(board: &Board, current_player: Disc) -> String {
    let board_str = board.to_string_as_board(current_player);
    let mut out = String::new();
    out.push_str("    A B C D E F G H\n");
    for (i, line) in board_str.lines().enumerate() {
        out.push_str(&format!("  {} ", i + 1));
        for (j, ch) in line.chars().enumerate() {
            if j > 0 {
                out.push(' ');
            }
            out.push(ch);
        }
        out.push('\n');
    }
    out
}

/// Print a verbose test case: header, board, iterations, result, expected.
fn print_verbose_test_case(
    test_case: &TestCase,
    result: &TestResult,
    iterations: &[IterationData],
) {
    println!("--- #{} ---", test_case.line_number);

    let board = test_case.board();
    print!(
        "{}",
        format_board_with_coords(&board, test_case.side_to_move())
    );

    print_verbose_iterations(iterations);

    let (rounded_secs, nps, pv_colored, score_colored) = compute_result_metrics(test_case, result);

    let tt_fill = iterations
        .last()
        .map(|i| format!("{:.1}%", i.tt_fill * 100.0))
        .unwrap_or_else(|| "-".to_string());

    let expect_moves = test_case.expected_moves_str();

    println!(
        "- score: {} (expected: {:>+.1})",
        score_colored,
        test_case.expected_score() as Scoref
    );
    println!("- move: {} (expected: {})", pv_colored, expect_moves);
    println!(
        "- nodes: {}  time: {:.4}s  NPS: {}",
        result.nodes.to_formatted_string(&Locale::en),
        rounded_secs,
        nps.to_formatted_string(&Locale::en),
    );
    println!("- TT fill: {}", tt_fill);
    println!();
}

/// Print a per-case Multi-PV comparison table joining the engine's per-move
/// scores against the OBF-listed expected scores.
///
/// Moves that the OBF entry did not enumerate render their Expected and Δ
/// cells as `-` and are excluded from the aggregate statistics.
fn print_multipv_test_result(test_case: &TestCase, result: &TestResult) {
    if test_case.is_pass() {
        println!(
            "### #{} (PS, expected: {:+})",
            test_case.line_number,
            test_case.expected_score()
        );
        println!();
        return;
    }

    let best_diff = result.score_difference;
    let best_score_colored = colorize_score(result.score, best_diff);
    println!(
        "### #{} (searched: {}, expected: {:+}, Δ: {:.1})",
        test_case.line_number,
        best_score_colored,
        test_case.expected_score(),
        best_diff,
    );

    struct Row {
        move_str: String,
        searched_plain: String,
        searched_colored: colored::ColoredString,
        expected: String,
        delta: String,
    }

    let rows: Vec<Row> = result
        .pv_moves
        .iter()
        .map(|pv| {
            let move_str = format!("{:?}", pv.sq);
            let rounded = (pv.score * 10.0).round() / 10.0;
            // Match `colorize_score`'s `{:.1}` format so plain length matches
            // the colored display width (ANSI escapes aside) used for padding.
            let searched_plain = format!("{rounded:.1}");
            let expected_score = test_case.expected_score_for_move(pv.sq);
            let (expected, delta, searched_colored) = match expected_score {
                Some(exp) => {
                    let diff = (rounded - exp as Scoref).abs();
                    (
                        format!("{exp:+}"),
                        format!("{diff:.1}"),
                        colorize_score(rounded, diff),
                    )
                }
                None => ("-".to_string(), "-".to_string(), searched_plain.normal()),
            };
            Row {
                move_str,
                searched_plain,
                searched_colored,
                expected,
                delta,
            }
        })
        .collect();

    let move_w = rows
        .iter()
        .map(|r| r.move_str.len())
        .max()
        .unwrap_or(0)
        .max("Move".len());
    let searched_w = rows
        .iter()
        .map(|r| r.searched_plain.len())
        .max()
        .unwrap_or(0)
        .max("Searched".len());
    let expected_w = rows
        .iter()
        .map(|r| r.expected.len())
        .max()
        .unwrap_or(0)
        .max("Expected".len());
    let delta_w = rows
        .iter()
        .map(|r| r.delta.len())
        .max()
        .unwrap_or(0)
        .max("Δ".chars().count());

    println!(
        "| {:<mw$} | {:>sw$} | {:>ew$} | {:>dw$} |",
        "Move",
        "Searched",
        "Expected",
        "Δ",
        mw = move_w,
        sw = searched_w,
        ew = expected_w,
        dw = delta_w,
    );
    println!(
        "|{:-<mw$}|{:->sw$}:|{:->ew$}:|{:->dw$}:|",
        "",
        "",
        "",
        "",
        mw = move_w + 2,
        sw = searched_w + 1,
        ew = expected_w + 1,
        dw = delta_w + 1,
    );
    // The colored cell embeds ANSI escapes so its char width differs from the
    // plain form. Pad using the plain width first, then swap in the colored
    // version for display.
    for row in &rows {
        let searched_pad = " ".repeat(searched_w.saturating_sub(row.searched_plain.len()));
        println!(
            "| {:<mw$} | {pad}{searched} | {:>ew$} | {:>dw$} |",
            row.move_str,
            row.expected,
            row.delta,
            mw = move_w,
            ew = expected_w,
            dw = delta_w,
            pad = searched_pad,
            searched = row.searched_colored,
        );
    }
    println!();
}

/// Exact value, for the side to move, of each move an endgame level chose.
///
/// The best move's value is already known (the OBF score, or `reference`
/// if given). Other moves are solved exactly after the timed search.
fn exact_move_values(
    search: &mut search::Search,
    test_case: &TestCase,
    iterations: &[IterationData],
    reference: Option<&IterationData>,
) -> std::collections::HashMap<Square, Scoref> {
    let board = test_case.board();
    let n_empties = board.empty_count();
    let full_depth = |it: &&IterationData| it.is_endgame && it.depth == n_empties;
    // A move whose value needs no solve: the reference's best move.
    let known = |sq: Square| match reference {
        Some(exact) => (sq == exact.best_move).then_some(exact.score),
        None => (test_case.rank_of(sq) == Some(0)).then(|| test_case.expected_score() as Scoref),
    };
    let mut values = std::collections::HashMap::new();
    for iter in iterations.iter().filter(full_depth) {
        if values.contains_key(&iter.best_move) {
            continue;
        }
        let value = if let Some(value) = known(iter.best_move) {
            value
        } else {
            let child = board.make_move(iter.best_move);
            let opponent = if child.has_legal_moves() {
                Some(child)
            } else {
                Some(child.switch_players()).filter(Board::has_legal_moves)
            };
            match opponent {
                Some(position) => {
                    search.init();
                    let score = search
                        .run(&position, &SearchRunOptions::with_level(Level::perfect()))
                        .score()
                        .expect("position has a legal move");
                    if position == child { -score } else { score }
                }
                None => -(child.solve(child.empty_count()) as Scoref),
            }
        };
        values.insert(iter.best_move, value);
    }
    values
}

/// Execute a section of test cases and return aggregated statistics
fn execute_section(
    section_name: &str,
    test_cases: &[TestCase],
    search: &mut search::Search,
    level: Level,
    args: &Args,
) -> SearchStats {
    let Args {
        verbose,
        multipv,
        move_loss,
        solve_reference,
        ..
    } = *args;
    println!("\n## {section_name} ({} cases)\n", test_cases.len());

    let mut stats = SearchStats::default();

    let num_width = if verbose || multipv {
        0
    } else {
        let max_num = test_cases
            .iter()
            .map(|tc| tc.line_number)
            .max()
            .unwrap_or(0);
        let nw = max_num.to_string().len().max(3);
        print_header(nw);
        nw
    };

    for test_case in test_cases {
        let (result, iterations) = execute_test_case(test_case, search, level, verbose, multipv);
        stats.update(&result);
        if multipv {
            stats.update_multipv(&result, test_case);
            print_multipv_test_result(test_case, &result);
            continue;
        }
        // Per-level stats skip multi-PV: it reports each PV line as its own
        // iteration, which would mix lines in one level.
        let n_empties = test_case.board().empty_count();
        let reference = if solve_reference {
            iterations
                .iter()
                .rfind(|it| it.is_endgame && it.depth == n_empties && it.probability == 100)
        } else {
            None
        };
        // Without the run's own exact solve there is nothing to judge against.
        if !solve_reference || reference.is_some() {
            let move_values = if move_loss && !test_case.is_pass() {
                exact_move_values(search, test_case, &iterations, reference)
            } else {
                std::collections::HashMap::new()
            };
            stats.update_levels(&iterations, test_case, &move_values, reference);
        }
        if verbose {
            print_verbose_test_case(test_case, &result, &iterations);
        } else {
            print_test_result(test_case, &result, num_width);
        }
    }

    stats.print(verbose, multipv);
    stats
}

/// Command line arguments for the evaluation test runner
#[derive(Parser)]
#[command(author, version, about = "Evaluation Test Suite Runner for Reversi AI")]
struct Args {
    /// Maximum search depth in plies
    #[arg(short, long, default_value = "60")]
    depth: Depth,

    /// Transposition table size in MB
    #[arg(long, default_value = "1024", value_parser = clap::value_parser!(u16).range(1..))]
    hash_size: u16,

    /// Number of parallel search threads
    #[arg(long)]
    threads: Option<usize>,

    /// Problem set to run (preset: fforum, hard-20, hard-25, hard-30; or file path)
    /// Can be specified multiple times. Default: all .obf files in problem directory.
    #[arg(long)]
    problem: Vec<String>,

    /// Path to the problem directory containing .obf files
    #[arg(long)]
    problem_dir: Option<String>,

    /// Enable verbose output with iterative deepening progress and search statistics
    #[arg(short, long)]
    verbose: bool,

    /// Multi-PV: score every legal move and compare against the OBF per-move
    /// expected scores. Prints a per-case comparison table and an aggregate
    /// Multi-PV Statistics block. Overrides verbose per-case formatting when
    /// both are set.
    #[arg(long)]
    multipv: bool,

    /// Solve each move an endgame level chooses, and report per level how
    /// often it is optimal and how many discs it loses on average. Each
    /// non-optimal move costs an exact solve, outside the timed search.
    #[arg(long)]
    move_loss: bool,

    /// Judge each endgame level against the run's own exact solve instead of
    /// the OBF score, for positions without exact labels.
    #[arg(long)]
    solve_reference: bool,

    /// Skip the exact solve at the end of the endgame ladder; judge the levels
    /// against the OBF score.
    #[arg(long, conflicts_with = "solve_reference")]
    selective_only: bool,
}

fn main() {
    let args = Args::parse();

    let problem_dir = if let Some(ref dir) = args.problem_dir {
        let path = std::path::PathBuf::from(dir);
        if !path.is_dir() {
            eprintln!("Error: Problem directory not found: {dir}");
            std::process::exit(1);
        }
        path
    } else {
        find_problem_dir().unwrap_or_else(|| {
            eprintln!("Error: Cannot find problem directory. Use --problem-dir to specify.");
            std::process::exit(1);
        })
    };

    let problem_sets = if args.problem.is_empty() {
        load_all_problems(&problem_dir)
    } else {
        load_problems(&args.problem, &problem_dir)
    };

    if problem_sets.is_empty() || problem_sets.iter().all(|ps| ps.cases.is_empty()) {
        eprintln!("Error: No test cases found");
        std::process::exit(1);
    }

    let search_options = SearchOptions::new(args.hash_size as usize).with_threads(args.threads);
    let mut search = search::Search::new(&search_options);
    let level = if args.selective_only {
        // An end depth of 0 for the last step stops the ladder before it.
        let mut end_depth = [args.depth; ENDGAME_STEPS];
        end_depth[ENDGAME_STEPS - 1] = 0;
        Level::with_depths(args.depth, end_depth)
    } else {
        Level::uniform(args.depth, args.depth)
    };

    let mut overall_stats = SearchStats::default();

    for problem_set in &problem_sets {
        let stats = execute_section(
            &problem_set.name,
            &problem_set.cases,
            &mut search,
            level,
            &args,
        );
        overall_stats.merge(&stats);
    }

    if problem_sets.len() > 1 {
        println!("\n## Overall ({} cases)", overall_stats.total_count);
        overall_stats.print(args.verbose, args.multipv);
    }
}
