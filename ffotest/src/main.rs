mod test_case;

use clap::Parser;
use colored::*;
use num_format::{Locale, ToFormattedString};
use reversi_core::{
    self,
    level::Level,
    search::{self, search_result::SearchResult, SearchOptions},
    types::{Depth, Scoref, Selectivity},
};
use std::time::Instant;
use test_case::TestCase;

#[derive(Default)]
struct SearchStats {
    total_time: std::time::Duration,
    total_nodes: u64,
    total_count: usize,
    score_differences: Vec<Scoref>,
    best_move_count: usize,
    top2_move_count: usize,
    top3_move_count: usize,
    score_count2: usize,
    score_count4: usize,
    score_count8: usize,
}

impl SearchStats {
    fn update(
        &mut self,
        elapsed: std::time::Duration,
        result: &SearchResult,
        test_case: &TestCase,
    ) {
        self.total_time += elapsed;
        self.total_nodes += result.n_nodes;
        self.total_count += 1;

        let score_difference = (result.score - test_case.expected_score as Scoref).abs();
        self.score_differences.push(score_difference);

        if score_difference <= 3.0 {
            self.score_count2 += 1;
        }
        if score_difference <= 6.0 {
            self.score_count4 += 1;
        }
        if score_difference <= 9.0 {
            self.score_count8 += 1;
        }

        if let Some(first_pv) = result.pv_line.first() {
            let move_str = format!("{:?}", first_pv);
            if test_case.is_best_move(&move_str) {
                self.best_move_count += 1;
                self.top2_move_count += 1;
                self.top3_move_count += 1;
            } else if test_case.is_second_best_move(&move_str) {
                self.top2_move_count += 1;
                self.top3_move_count += 1;
            } else if test_case.is_third_best_move(&move_str) {
                self.top3_move_count += 1;
            }
        }
    }

    fn print(&self) {
        let mean_diff = self
            .score_differences
            .iter()
            .map(|&d| d as f64)
            .sum::<f64>()
            / self.total_count as f64;
        let variance = self
            .score_differences
            .iter()
            .map(|&d| {
                let diff = d as f64 - mean_diff;
                diff * diff
            })
            .sum::<f64>()
            / self.total_count as f64;
        let std_dev = variance.sqrt();

        let stats = [
            (
                "Total time",
                format!("{:.3}s", self.total_time.as_secs_f64()),
            ),
            (
                "Total nodes",
                self.total_nodes.to_formatted_string(&Locale::en),
            ),
            (
                "NPS",
                ((self.total_nodes as f64 / self.total_time.as_secs_f64()) as u64)
                    .to_formatted_string(&Locale::en),
            ),
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
                    (self.score_count2 as f64 / self.total_count as f64) * 100.0,
                    self.score_count2,
                    self.total_count
                ),
            ),
            (
                "Score ±6",
                format!(
                    "{:.1}% ({}/{})",
                    (self.score_count4 as f64 / self.total_count as f64) * 100.0,
                    self.score_count4,
                    self.total_count
                ),
            ),
            (
                "Score ±9",
                format!(
                    "{:.1}% ({}/{})",
                    (self.score_count8 as f64 / self.total_count as f64) * 100.0,
                    self.score_count8,
                    self.total_count
                ),
            ),
            ("MAE", format!("{:.2}", mean_diff)),
            ("Std Dev.", format!("{:.2}", std_dev)),
        ];

        let max_label_len = stats
            .iter()
            .map(|(label, _)| label.len())
            .max()
            .unwrap_or(0);

        println!("\n### Statistics:");
        for (label, value) in stats {
            match label {
                "Best move" | "Top 2 move" | "Top 3 move" | "Score ±3" | "Score ±6"
                | "Score ±9" => {
                    let percentage = value.split('%').next().unwrap().parse::<f64>().unwrap();
                    let colored_value = if percentage >= 80.0 {
                        value.bright_green()
                    } else if percentage >= 60.0 {
                        value.bright_yellow()
                    } else {
                        value.bright_red()
                    };
                    println!(
                        "- {:<width$}: {}",
                        label,
                        colored_value,
                        width = max_label_len
                    );
                }
                "NPS" => {
                    println!("- {:<width$}: {}", label, value, width = max_label_len);
                }
                _ => {
                    println!("- {:<width$}: {}", label, value, width = max_label_len);
                }
            }
        }
    }
}

fn print_header() {
    println!(
        "| {:^3} | {:^6} | {:^8} | {:^14} | {:^12} | {:^8} | {:^6} | {:<32} |",
        "#", "Depth", "Time(s)", "Nodes", "NPS", "Line", "Score", "Expected"
    );
    println!(
        "|----:|-------:|---------:|---------------:|-------------:|:---------|-------:|:---------------------------------|"
    );
}

fn execute(
    test_cases: &Vec<&TestCase>,
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
        let board = test_case.get_board();
        let start = Instant::now();
        let result = search.test(&board, level, selectivity);
        let elapsed = start.elapsed();

        stats.update(elapsed, &result, test_case);

        let nodes_formatted = result.n_nodes.to_formatted_string(&Locale::en);
        let nps_formatted = ((result.n_nodes as f64 / elapsed.as_secs_f64()) as u64)
            .to_formatted_string(&Locale::en);

        let pv_line = result
            .pv_line
            .iter()
            .take(3)
            .map(|&sq| format!("{:?}", &sq))
            .collect::<Vec<String>>()
            .join(" ");

        // score
        let score_diff = (result.score - test_case.expected_score as Scoref).abs();
        let score_str = if score_diff <= 3.0 {
            format!("{:.1}", result.score).bright_green()
        } else if score_diff <= 6.0 {
            format!("{:.1}", result.score).to_string().bright_yellow()
        } else if score_diff <= 9.0 {
            format!("{:.1}", result.score).to_string().bright_cyan()
        } else {
            format!("{:.1}", result.score).to_string().bright_red()
        };

        // pv line
        let pv_line_color = if let Some(first_pv) = result.pv_line.first() {
            let move_str = format!("{:?}", first_pv);
            if test_case.is_best_move(&move_str) {
                pv_line.bright_green()
            } else if test_case.is_second_best_move(&move_str) {
                pv_line.bright_yellow()
            } else if test_case.is_third_best_move(&move_str) {
                pv_line.bright_cyan()
            } else {
                pv_line.bright_red()
            }
        } else {
            pv_line.bright_white()
        };

        let depth = if result.get_probability() == 100 {
            format!("{:?}", result.depth)
        } else {
            format!("{:?}@{:?}%", result.depth, result.get_probability())
        };

        println!(
            "| {:>3} | {:^6} | {:>8.3} | {:>14} | {:>12} | {:<8} | {:>6} | {:<32} |",
            test_case.no.to_string(),
            depth,
            elapsed.as_secs_f64(),
            nodes_formatted,
            nps_formatted,
            pv_line_color,
            score_str,
            format!(
                "{:>3} : {}",
                test_case.expected_score.to_string(),
                test_case.get_best_moves_str()
            )
        );
    }

    stats.print();
}

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "60")]
    depth: Depth,

    #[arg(long, default_value = "1")]
    selectivity: Selectivity,

    #[arg(long, default_value = "256")]
    hash_size: i32,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long)]
    case: Option<usize>,

    #[arg(long)]
    from: Option<usize>,

    #[arg(long)]
    to: Option<usize>,
}

fn main() {
    reversi_core::init();
    let args = Args::parse();
    let test_cases = test_case::get_test_cases();
    let filtered: Vec<&TestCase> = test_cases
        .iter()
        .filter(|test| {
            let case_matches = args.case.is_none_or(|case_no| test.no == case_no);
            let from_matches = args.from.is_none_or(|from| test.no >= from);
            let to_matches = args.to.is_none_or(|to| test.no <= to);
            case_matches && from_matches && to_matches
        })
        .collect();

    let mut search_options = search::SearchOptions {
        tt_mb_size: args.hash_size,
        ..Default::default()
    };
    if let Some(threads) = args.threads {
        search_options.n_threads = threads;
    }

    execute(&filtered, &search_options, args.depth, args.selectivity);
}
