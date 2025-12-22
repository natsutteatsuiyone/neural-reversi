//! FFO Test Suite Runner for Reversi AI Engines
//!
//! This program runs the FFO (French Federation of Othello) test suite,
//! a collection of 79 challenging endgame positions used to evaluate
//! the solving capabilities of Reversi/Othello engines.
//!
//! The test runner measures:
//! - Solving accuracy (correct score and best move)
//! - Search performance (time and nodes)
//! - Move selection quality (best move percentage)

mod test_case;

use clap::Parser;
use colored::*;
use num_format::{Locale, ToFormattedString};
use reversi_core::{
    self,
    level::Level,
    search::{self, SearchOptions, search_result::SearchResult},
    square::Square,
    types::{Depth, Scoref, Selectivity},
};
use std::time::Instant;
use test_case::TestCase;

/// Score tolerance levels for evaluation
const SCORE_TOLERANCE_PERFECT: Scoref = 3.0;
const SCORE_TOLERANCE_GOOD: Scoref = 6.0;
const SCORE_TOLERANCE_ACCEPTABLE: Scoref = 9.0;

/// Performance thresholds for color coding
const PERFORMANCE_EXCELLENT: f64 = 80.0;
const PERFORMANCE_GOOD: f64 = 60.0;

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
}

/// Round duration to 0.01ms precision for consistent display
fn round_duration(elapsed: std::time::Duration) -> std::time::Duration {
    let rounded_micros = (elapsed.as_secs_f64() * 10000.0).round() as u64 * 100;
    std::time::Duration::from_micros(rounded_micros)
}

impl SearchStats {
    /// Update statistics with results from a single test case
    fn update(
        &mut self,
        elapsed: std::time::Duration,
        result: &SearchResult,
        test_case: &TestCase,
    ) {
        self.total_time += round_duration(elapsed);
        self.total_nodes += result.n_nodes;
        self.total_count += 1;

        let score_difference = (result.score - test_case.expected_score as Scoref).abs();
        self.score_differences.push(score_difference);

        if score_difference <= SCORE_TOLERANCE_PERFECT {
            self.perfect_score_count += 1;
        }
        if score_difference <= SCORE_TOLERANCE_GOOD {
            self.good_score_count += 1;
        }
        if score_difference <= SCORE_TOLERANCE_ACCEPTABLE {
            self.acceptable_score_count += 1;
        }

        match Self::classify_move(result, test_case) {
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
    }

    /// Classify the move accuracy based on the search result
    fn classify_move(result: &SearchResult, test_case: &TestCase) -> MoveAccuracy {
        if let Some(first_pv) = result.pv_line.first() {
            let move_str = format!("{first_pv:?}");
            if test_case.is_best_move(&move_str) {
                MoveAccuracy::Best
            } else if test_case.is_second_best_move(&move_str) {
                MoveAccuracy::SecondBest
            } else if test_case.is_third_best_move(&move_str) {
                MoveAccuracy::ThirdBest
            } else {
                MoveAccuracy::Other
            }
        } else {
            MoveAccuracy::Other
        }
    }

    /// Calculate mean and standard deviation of score differences
    fn calculate_statistics(&self) -> (f64, f64) {
        if self.total_count == 0 {
            return (0.0, 0.0);
        }

        let mean = self
            .score_differences
            .iter()
            .map(|&d| d as f64)
            .sum::<f64>()
            / self.total_count as f64;

        let variance = self
            .score_differences
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.total_count as f64;

        (mean, variance.sqrt())
    }

    /// Print formatted statistics summary with color coding
    fn print(&self) {
        if self.total_count == 0 {
            println!("\n### No test cases were run");
            return;
        }

        let (mean_diff, std_dev) = self.calculate_statistics();
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
                format!(
                    "{:.1}% ({}/{})",
                    (self.best_move_count as f64 / self.total_count as f64) * 100.0,
                    self.best_move_count,
                    self.total_count
                ),
            ),
            (
                "Top 2 move",
                format!(
                    "{:.1}% ({}/{})",
                    (self.top2_move_count as f64 / self.total_count as f64) * 100.0,
                    self.top2_move_count,
                    self.total_count
                ),
            ),
            (
                "Top 3 move",
                format!(
                    "{:.1}% ({}/{})",
                    (self.top3_move_count as f64 / self.total_count as f64) * 100.0,
                    self.top3_move_count,
                    self.total_count
                ),
            ),
            (
                "Score ±3",
                format!(
                    "{:.1}% ({}/{})",
                    (self.perfect_score_count as f64 / self.total_count as f64) * 100.0,
                    self.perfect_score_count,
                    self.total_count
                ),
            ),
            (
                "Score ±6",
                format!(
                    "{:.1}% ({}/{})",
                    (self.good_score_count as f64 / self.total_count as f64) * 100.0,
                    self.good_score_count,
                    self.total_count
                ),
            ),
            (
                "Score ±9",
                format!(
                    "{:.1}% ({}/{})",
                    (self.acceptable_score_count as f64 / self.total_count as f64) * 100.0,
                    self.acceptable_score_count,
                    self.total_count
                ),
            ),
            ("MAE", format!("{mean_diff:.2}")),
            ("Std Dev.", format!("{std_dev:.2}")),
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
fn print_header() {
    println!(
        "| {:^3} | {:^6} | {:^9} | {:^14} | {:^13} | {:^8} | {:^6} | {:<32} |",
        "#", "Depth", "Time(s)", "Nodes", "NPS", "Line", "Score", "Expected"
    );
    println!(
        "|----:|-------:|----------:|---------------:|--------------:|:---------|-------:|:---------------------------------|"
    );
}

/// Format depth information including probability if not 100%
fn format_depth(result: &SearchResult) -> String {
    let probability = result.get_probability();
    if probability == 100 {
        format!("{}", result.depth)
    } else {
        format!("{}@{}%", result.depth, probability)
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
    selectivity: Selectivity,
) -> TestResult {
    let board = test_case.get_board();
    search.init();

    let start = Instant::now();
    let result = search.test(&board, level, selectivity);
    let elapsed = start.elapsed();

    let score_difference = (result.score - test_case.expected_score as Scoref).abs();
    let move_accuracy = SearchStats::classify_move(&result, test_case);

    TestResult {
        elapsed,
        nodes: result.n_nodes,
        score: result.score,
        depth: result.depth,
        selectivity: result.selectivity,
        pv_line: result.pv_line,
        score_difference,
        move_accuracy,
    }
}

/// Print a single test result row
fn print_test_result(test_case: &TestCase, result: &TestResult) {
    let nodes_formatted = result.nodes.to_formatted_string(&Locale::en);
    let rounded_time = round_duration(result.elapsed);
    let rounded_secs = rounded_time.as_secs_f64();
    let nps = if rounded_secs > 0.0 {
        (result.nodes as f64 / rounded_secs).round() as u64
    } else {
        0
    };
    let nps_formatted = nps.to_formatted_string(&Locale::en);

    let pv_line_str = format_pv_line(&result.pv_line, 3);
    let score_colored = colorize_score(result.score, result.score_difference);
    let pv_line_colored = colorize_move(pv_line_str, result.move_accuracy);
    // Create a temporary SearchResult just for formatting depth
    let temp_result = SearchResult {
        score: result.score,
        best_move: result.pv_line.first().copied(),
        n_nodes: result.nodes,
        pv_line: result.pv_line.clone(),
        depth: result.depth,
        selectivity: result.selectivity,
        game_phase: reversi_core::search::search_context::GamePhase::EndGame,
    };
    let depth_str = format_depth(&temp_result);

    println!(
        "| {:>3} | {:^6} | {:>9.4} | {:>14} | {:>13} | {:<8} | {:>6} | {:<32} |",
        test_case.no,
        depth_str,
        rounded_secs,
        nodes_formatted,
        nps_formatted,
        pv_line_colored,
        score_colored,
        format!(
            "{:>3} : {}",
            test_case.expected_score,
            test_case.get_best_moves_str()
        )
    );
}

/// Execute the FFO test suite with given parameters
///
/// # Arguments
/// * `test_cases` - Vector of test positions to solve
/// * `search_options` - Search configuration (hash size, threads)
/// * `depth` - Maximum search depth
/// * `selectivity` - Search selectivity level (1-6)
fn execute(
    test_cases: &[&TestCase],
    search_options: &SearchOptions,
    depth: Depth,
    selectivity: Selectivity,
) {
    print_header();

    let mut search = search::Search::new(search_options);
    let mut stats = SearchStats::default();
    let level = Level {
        mid_depth: depth,
        end_depth: [depth; 7],
    };

    for test_case in test_cases {
        let result = execute_test_case(test_case, &mut search, level, selectivity);
        // Convert TestResult back to SearchResult for stats update
        let search_result = SearchResult {
            score: result.score,
            best_move: result.pv_line.first().copied(),
            n_nodes: result.nodes,
            pv_line: result.pv_line.clone(),
            depth: result.depth,
            selectivity: result.selectivity,
            game_phase: reversi_core::search::search_context::GamePhase::EndGame,
        };
        stats.update(result.elapsed, &search_result, test_case);
        print_test_result(test_case, &result);
    }

    stats.print();
}

/// Command line arguments for the FFO test runner
#[derive(Parser)]
#[command(author, version, about = "FFO Test Suite Runner for Reversi AI")]
struct Args {
    /// Maximum search depth in plies
    #[arg(short, long, default_value = "60")]
    depth: Depth,

    /// Search selectivity (1: 73%, 2: 87%, 3: 95%, 4: 98%, 5: 99%, 6: 100%)
    #[arg(long, default_value = "1", value_parser = clap::value_parser!(u8).range(0..=6))]
    selectivity: u8,

    /// Transposition table size in MB
    #[arg(long, default_value = "512", value_parser = clap::value_parser!(u16).range(1..))]
    hash_size: u16,

    /// Number of parallel search threads
    #[arg(long)]
    threads: Option<usize>,

    /// Run only a specific test case number (1-79)
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=79))]
    case: Option<u8>,

    /// Start from this test case number
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=79))]
    from: Option<u8>,

    /// Run up to this test case number
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=79))]
    to: Option<u8>,
}

fn main() {
    let args = Args::parse();

    // Validate argument combinations
    if args.case.is_some() && (args.from.is_some() || args.to.is_some()) {
        eprintln!("Error: --case cannot be used with --from or --to");
        std::process::exit(1);
    }

    let test_cases = test_case::get_test_cases();
    let filtered: Vec<&TestCase> = test_cases
        .iter()
        .filter(|test| {
            let case_matches = args.case.is_none_or(|case_no| test.no == case_no as usize);
            let from_matches = args.from.is_none_or(|from| test.no >= from as usize);
            let to_matches = args.to.is_none_or(|to| test.no <= to as usize);
            case_matches && from_matches && to_matches
        })
        .collect();

    if filtered.is_empty() {
        eprintln!("Error: No test cases match the specified criteria");
        std::process::exit(1);
    }

    let search_options =
        search::SearchOptions::new(args.hash_size as usize).with_threads(args.threads);

    execute(
        &filtered,
        &search_options,
        args.depth,
        Selectivity::from_u8(args.selectivity),
    );
}
