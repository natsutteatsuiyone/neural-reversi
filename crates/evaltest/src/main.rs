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

mod obf;
mod test_case;

use clap::Parser;
use colored::*;
use num_format::{Locale, ToFormattedString};
use reversi_core::{
    self,
    level::Level,
    probcut::Selectivity,
    search::{self, SearchRunOptions, options::SearchOptions, search_result::SearchResult},
    square::Square,
    types::{Depth, Scoref},
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
    selectivity: Selectivity,
) -> TestResult {
    let board = test_case.get_board();
    // For pass positions, switch sides and search from the opponent's perspective
    let search_board = if test_case.is_pass() {
        board.switch_players()
    } else {
        board
    };
    search.init();

    let start = Instant::now();
    let options = SearchRunOptions::with_level(level, selectivity);
    let result = search.run(&search_board, &options);
    let elapsed = start.elapsed();

    // For pass positions, negate the score back to the original player's perspective
    let score = if test_case.is_pass() {
        -result.score
    } else {
        result.score
    };
    let score_difference = (score - test_case.expected_score as Scoref).abs();
    let move_accuracy = if test_case.is_pass() {
        MoveAccuracy::Best // Pass positions don't have move choices
    } else {
        SearchStats::classify_move(&result, test_case)
    };

    TestResult {
        elapsed,
        nodes: result.n_nodes,
        score,
        depth: result.depth,
        selectivity: result.selectivity,
        pv_line: result.pv_line,
        score_difference,
        move_accuracy,
    }
}

/// Print a single test result row
fn print_test_result(test_case: &TestCase, result: &TestResult, num_width: usize) {
    let nodes_formatted = result.nodes.to_formatted_string(&Locale::en);
    let rounded_time = round_duration(result.elapsed);
    let rounded_secs = rounded_time.as_secs_f64();
    let nps = if rounded_secs > 0.0 {
        (result.nodes as f64 / rounded_secs).round() as u64
    } else {
        0
    };
    let nps_formatted = nps.to_formatted_string(&Locale::en);

    let pv_line_str = if test_case.is_pass() {
        "PS".to_string()
    } else {
        format_pv_line(&result.pv_line, 3)
    };
    let score_colored = colorize_score(result.score, result.score_difference);
    let pv_line_colored = colorize_move(pv_line_str, result.move_accuracy);
    let depth_str = format_depth(result.depth, result.selectivity);

    println!(
        "| {:>num_width$} | {:^6} | {:>9.4} | {:>15} | {:>13} | {:<8} | {:>6} | {:<32} |",
        test_case.line_number,
        depth_str,
        rounded_secs,
        nodes_formatted,
        nps_formatted,
        pv_line_colored,
        score_colored,
        if test_case.is_pass() {
            format!("{:>3} : PS", test_case.expected_score)
        } else {
            format!(
                "{:>3} : {}",
                test_case.expected_score,
                test_case.get_best_moves_str()
            )
        }
    );
}

/// Execute a section of test cases and return aggregated statistics
fn execute_section(
    section_name: &str,
    test_cases: &[TestCase],
    search: &mut search::Search,
    level: Level,
    selectivity: Selectivity,
) -> SearchStats {
    println!("\n## {section_name} ({} cases)\n", test_cases.len());

    let max_num = test_cases
        .iter()
        .map(|tc| tc.line_number)
        .max()
        .unwrap_or(0);
    let num_width = max_num.to_string().len().max(3);

    print_header(num_width);

    let mut stats = SearchStats::default();

    for test_case in test_cases {
        let result = execute_test_case(test_case, search, level, selectivity);
        stats.update(&result);
        print_test_result(test_case, &result, num_width);
    }

    stats.print();
    stats
}

/// Command line arguments for the evaluation test runner
#[derive(Parser)]
#[command(author, version, about = "Evaluation Test Suite Runner for Reversi AI")]
struct Args {
    /// Maximum search depth in plies
    #[arg(short, long, default_value = "60")]
    depth: Depth,

    /// Search selectivity (0: 73%, 1: 87%, 2: 95%, 3: 98%, 4: 99%, 5: 100%)
    #[arg(long, default_value = "0", value_parser = clap::value_parser!(u8).range(0..=5))]
    selectivity: u8,

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
        obf::find_problem_dir().unwrap_or_else(|| {
            eprintln!("Error: Cannot find problem directory. Use --problem-dir to specify.");
            std::process::exit(1);
        })
    };

    let problem_sets = if args.problem.is_empty() {
        obf::load_all_problems(&problem_dir)
    } else {
        obf::load_problems(&args.problem, &problem_dir)
    };

    if problem_sets.is_empty() || problem_sets.iter().all(|ps| ps.cases.is_empty()) {
        eprintln!("Error: No test cases found");
        std::process::exit(1);
    }

    let search_options = SearchOptions::new(args.hash_size as usize).with_threads(args.threads);
    let mut search = search::Search::new(&search_options);
    let level = Level {
        mid_depth: args.depth,
        end_depth: [args.depth; 4],
    };
    let selectivity = Selectivity::from_u8(args.selectivity);

    let mut overall_stats = SearchStats::default();

    for problem_set in &problem_sets {
        let stats = execute_section(
            &problem_set.name,
            &problem_set.cases,
            &mut search,
            level,
            selectivity,
        );
        overall_stats.merge(&stats);
    }

    if problem_sets.len() > 1 {
        println!("\n## Overall ({} cases)", overall_stats.total_count);
        overall_stats.print();
    }
}
