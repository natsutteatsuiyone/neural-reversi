//! Full round-robin weight tournament.

use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::Args;
use crate::game::{MatchResult, MatchWinner, play_match};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Weight {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Standing {
    games: usize,
    wins: usize,
    losses: usize,
    draws: usize,
    game_points_twice: usize,
    disc_score: i64,
    opponents: Vec<usize>,
}

pub(crate) fn run(args: &Args) -> Result<()> {
    let weights = read_weight_files(&args.weights_dir)?;
    if weights.len() < 2 {
        bail!("weights-dir must contain at least two .zst files");
    }

    let openings = match &args.opening_file {
        Some(path) => read_opening_file(path)?,
        None => vec![String::new()],
    };
    if openings.is_empty() {
        bail!("no playable openings found");
    }

    let jobs = args.jobs.get().min((weights.len() / 2).max(1));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .context("failed to create comparison thread pool")?;
    let pairings = build_round_robin_pairings(weights.len());
    let comparison_count = pairings.len();
    let games_per_comparison = openings.len() * 2;

    println!("Weights: {}", weights.len());
    println!("Openings: {}", openings.len());
    println!("Jobs: {jobs}");
    println!("Comparisons: {comparison_count}");
    println!("Games: {}\n", comparison_count * games_per_comparison);

    let progress = create_progress_bar(comparison_count * games_per_comparison)?;
    let match_results: Result<Vec<_>> = pool.install(|| {
        pairings
            .par_iter()
            .map(|&(engine1, engine2)| {
                play_match(&weights[engine1], &weights[engine2], &openings, || {
                    progress.inc(1);
                })
            })
            .collect()
    });
    progress.finish_and_clear();
    let match_results = match_results?;
    let mut standings = vec![Standing::default(); weights.len()];

    for (comparison, ((engine1, engine2), result)) in
        pairings.into_iter().zip(match_results).enumerate()
    {
        add_match_to_standings(&mut standings, engine1, engine2, result);

        let winner = match result.winner() {
            MatchWinner::Engine1 => &weights[engine1].name,
            MatchWinner::Engine2 => &weights[engine2].name,
            MatchWinner::Draw => "Draw",
        };
        println!(
            "[{}/{comparison_count}] {} vs {}: {}-{}-{}, score {:+}; winner {winner}",
            comparison + 1,
            weights[engine1].name,
            weights[engine2].name,
            result.engine1_wins,
            result.engine2_wins,
            result.draws,
            result.engine1_score,
        );
    }

    let ranking = ranked_weights(&weights, &standings);
    print_standings(&weights, &standings, &ranking);
    println!("\n## Result\n");
    println!("Strongest: {}", weights[ranking[0]].name);
    Ok(())
}

fn create_progress_bar(total_games: usize) -> Result<ProgressBar> {
    let progress = ProgressBar::new(total_games as u64);
    let style = ProgressStyle::with_template(
        "{spinner:.green} Games [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) elapsed {elapsed_precise} ETA {eta_precise}",
    )
    .context("failed to configure progress bar")?
    .progress_chars("=>-");
    progress.set_style(style);
    progress.enable_steady_tick(Duration::from_millis(100));
    Ok(progress)
}

fn read_weight_files(dir: &Path) -> Result<Vec<Weight>> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("failed to read weights directory {}", dir.display()))?;
    let mut weights = Vec::new();

    for entry in entries {
        let entry =
            entry.with_context(|| format!("failed to read an entry in {}", dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("failed to inspect {}", entry.path().display()))?;
        if !file_type.is_file() {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".zst") {
            weights.push(Weight {
                name,
                path: entry.path(),
            });
        }
    }

    weights.sort_by(|a, b| natural_cmp(&a.name, &b.name));
    Ok(weights)
}

fn read_opening_file(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open opening file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut openings = Vec::new();

    for line in reader.lines() {
        let line =
            line.with_context(|| format!("failed to read opening file {}", path.display()))?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            openings.push(line.to_owned());
        }
    }

    Ok(openings)
}

fn build_round_robin_pairings(weight_count: usize) -> Vec<(usize, usize)> {
    let mut pairings = Vec::with_capacity(weight_count * weight_count.saturating_sub(1) / 2);
    for engine1 in 0..weight_count {
        for engine2 in engine1 + 1..weight_count {
            pairings.push((engine1, engine2));
        }
    }
    pairings
}

fn add_match_to_standings(
    standings: &mut [Standing],
    engine1: usize,
    engine2: usize,
    result: MatchResult,
) {
    let engine1_points_twice = result.engine1_wins * 2 + result.draws;
    let engine2_points_twice = result.engine2_wins * 2 + result.draws;

    standings[engine1].games += result.games;
    standings[engine1].wins += result.engine1_wins;
    standings[engine1].losses += result.engine2_wins;
    standings[engine1].draws += result.draws;
    standings[engine1].game_points_twice += engine1_points_twice;
    standings[engine1].disc_score += result.engine1_score;
    standings[engine1].opponents.push(engine2);

    standings[engine2].games += result.games;
    standings[engine2].wins += result.engine2_wins;
    standings[engine2].losses += result.engine1_wins;
    standings[engine2].draws += result.draws;
    standings[engine2].game_points_twice += engine2_points_twice;
    standings[engine2].disc_score -= result.engine1_score;
    standings[engine2].opponents.push(engine1);
}

fn score_rate(standing: &Standing) -> f64 {
    if standing.games == 0 {
        0.5
    } else {
        standing.game_points_twice as f64 / (standing.games * 2) as f64
    }
}

fn average_disc_score(standing: &Standing) -> f64 {
    if standing.games == 0 {
        0.0
    } else {
        standing.disc_score as f64 / standing.games as f64
    }
}

fn opponent_score_rate(standing: &Standing, standings: &[Standing]) -> f64 {
    if standing.opponents.is_empty() {
        0.5
    } else {
        standing
            .opponents
            .iter()
            .map(|&opponent| score_rate(&standings[opponent]))
            .sum::<f64>()
            / standing.opponents.len() as f64
    }
}

fn ranked_weights(weights: &[Weight], standings: &[Standing]) -> Vec<usize> {
    let mut order = (0..weights.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| compare_standings(a, b, weights, standings));
    order
}

fn compare_standings(a: usize, b: usize, weights: &[Weight], standings: &[Standing]) -> Ordering {
    score_rate(&standings[b])
        .total_cmp(&score_rate(&standings[a]))
        .then_with(|| {
            opponent_score_rate(&standings[b], standings)
                .total_cmp(&opponent_score_rate(&standings[a], standings))
        })
        .then_with(|| {
            average_disc_score(&standings[b]).total_cmp(&average_disc_score(&standings[a]))
        })
        .then_with(|| natural_cmp(&weights[a].name, &weights[b].name))
}

fn print_standings(weights: &[Weight], standings: &[Standing], ranking: &[usize]) {
    println!("\n## Standings\n");
    println!("| # | Weight | Score | Games | W-L-D | Disc/game | Opp score |");
    println!("|--:|--------|------:|------:|------:|----------:|----------:|");

    for (rank, &weight) in ranking.iter().enumerate() {
        let standing = &standings[weight];
        println!(
            "| {} | {} | {:.1}% | {} | {}-{}-{} | {:+.2} | {:.1}% |",
            rank + 1,
            weights[weight].name,
            score_rate(standing) * 100.0,
            standing.games,
            standing.wins,
            standing.losses,
            standing.draws,
            average_disc_score(standing),
            opponent_score_rate(standing, standings) * 100.0,
        );
    }
}

fn natural_cmp(a: &str, b: &str) -> Ordering {
    let mut a = a.as_bytes();
    let mut b = b.as_bytes();

    while !a.is_empty() && !b.is_empty() {
        if a[0].is_ascii_digit() && b[0].is_ascii_digit() {
            let a_digits = a
                .iter()
                .position(|byte| !byte.is_ascii_digit())
                .unwrap_or(a.len());
            let b_digits = b
                .iter()
                .position(|byte| !byte.is_ascii_digit())
                .unwrap_or(b.len());
            let a_trimmed = trim_leading_zeroes(&a[..a_digits]);
            let b_trimmed = trim_leading_zeroes(&b[..b_digits]);
            let ordering = a_trimmed
                .len()
                .cmp(&b_trimmed.len())
                .then_with(|| a_trimmed.cmp(b_trimmed))
                .then_with(|| a_digits.cmp(&b_digits));
            if ordering != Ordering::Equal {
                return ordering;
            }
            a = &a[a_digits..];
            b = &b[b_digits..];
        } else {
            let ordering = a[0].cmp(&b[0]);
            if ordering != Ordering::Equal {
                return ordering;
            }
            a = &a[1..];
            b = &b[1..];
        }
    }

    a.len().cmp(&b.len())
}

fn trim_leading_zeroes(digits: &[u8]) -> &[u8] {
    let first_nonzero = digits
        .iter()
        .position(|&digit| digit != b'0')
        .unwrap_or(digits.len().saturating_sub(1));
    &digits[first_nonzero..]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn round_robin_pairings_cover_each_pair_once() {
        let pairings = build_round_robin_pairings(5);
        let unique = pairings.iter().copied().collect::<HashSet<_>>();

        assert_eq!(pairings.len(), 10);
        assert_eq!(unique.len(), 10);
        assert!(pairings.iter().all(|&(engine1, engine2)| engine1 < engine2));
    }

    #[test]
    fn natural_order_compares_numeric_filename_parts() {
        assert_eq!(natural_cmp("weight-2.zst", "weight-10.zst"), Ordering::Less);
    }
}
