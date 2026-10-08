use crate::colors::ThemeColor;
use crate::sprt::{Sprt, SprtConfig, SprtResult, SprtStatus};
use colored::*;
use reversi_core::disc::Disc;
use std::io::{self, IsTerminal};

const ELO_K: f64 = 400.0;

#[derive(Debug, Clone, Default)]
pub struct MatchStatistics {
    pub engine1_wins: u32,
    pub engine2_wins: u32,
    pub draws: u32,
    pub total_score: i32,
    pub recent_results: Vec<GameHistory>,
    pub(crate) random_seed: Option<u64>,
    /// GGS synchro mode: W/L/D, Elo, and SPRT count matches (game pairs)
    /// instead of games.
    pub(crate) synchro: bool,
    completed_games: u32,
    rating_sum: f64,
    paired_frequencies: PentanomialFrequencies,
}

#[derive(Debug, Clone)]
pub struct GameHistory {
    pub winner: MatchWinner,
    pub score: i32,
    pub opening: String,
    pub engine1_color: Option<Disc>,
}

impl MatchStatistics {
    /// Records one finished game. Disc margins always count per game; in
    /// synchro mode the outcome waits for [`Self::add_pair`].
    pub fn add_game(
        &mut self,
        winner: MatchWinner,
        score: i32,
        opening: String,
        engine1_is_black: bool,
    ) {
        self.completed_games += 1;
        self.total_score += score;
        if self.synchro {
            return;
        }
        let color = if engine1_is_black {
            Disc::Black
        } else {
            Disc::White
        };
        self.record_outcome(winner, score, opening, Some(color));
    }

    /// Records both games of one opening, each as `(winner, engine1 margin)`,
    /// after both were passed to [`Self::add_game`].
    pub fn add_pair(
        &mut self,
        first: (MatchWinner, i32),
        second: (MatchWinner, i32),
        opening: String,
    ) {
        if !self.synchro {
            self.paired_frequencies.add_pair(first.0, second.0);
            return;
        }
        let sum = first.1 + second.1;
        let winner = match sum.cmp(&0) {
            std::cmp::Ordering::Greater => MatchWinner::Engine1,
            std::cmp::Ordering::Less => MatchWinner::Engine2,
            std::cmp::Ordering::Equal => MatchWinner::Draw,
        };
        let margin = f64::from(sum) / 2.0;
        self.rating_sum += 0.5 + 0.375 * margin / (1.0 + 0.75 * margin.abs());
        self.record_outcome(winner, sum, opening, None);
    }

    fn record_outcome(
        &mut self,
        winner: MatchWinner,
        score: i32,
        opening: String,
        engine1_color: Option<Disc>,
    ) {
        match winner {
            MatchWinner::Engine1 => self.engine1_wins += 1,
            MatchWinner::Engine2 => self.engine2_wins += 1,
            MatchWinner::Draw => self.draws += 1,
        }
        self.recent_results.push(GameHistory {
            winner,
            score,
            opening,
            engine1_color,
        });
        if self.recent_results.len() > 5 {
            self.recent_results.remove(0);
        }
    }

    /// Finished games, including the first game of an unfinished pair.
    pub fn total_games(&self) -> u32 {
        self.completed_games
    }

    /// Scored W/L/D outcomes: games, or synchro matches in synchro mode.
    pub(crate) fn total_results(&self) -> u32 {
        self.engine1_wins + self.engine2_wins + self.draws
    }

    pub(crate) fn update_sprt(&self, sprt: &mut Sprt) -> SprtResult {
        if self.synchro {
            sprt.update_trinomial(self.engine1_wins, self.engine2_wins, self.draws)
        } else {
            sprt.update(&self.paired_frequencies)
        }
    }

    pub fn print_final_results(
        &self,
        engine1_name: &str,
        engine2_name: &str,
        sprt: Option<(SprtConfig, SprtResult)>,
    ) -> io::Result<()> {
        let total_games = self.total_games();

        if let Some(seed) = self.random_seed {
            println!("GGS random openings seed {seed}");
        }

        if total_games == 0 {
            println!("{}", "No games played yet.".info());
            return Ok(());
        }

        if io::stdout().is_terminal() {
            println!("\r\x1B[2K");
        }
        println!("{}", "═".repeat(80).info().bold());
        let header = format!("{} vs {}", engine1_name, engine2_name);
        println!("{:^80}", header.primary().bold());
        println!("{}", "═".repeat(80).info().bold());
        println!();

        self.print_summary();

        if let Some((config, result)) = sprt {
            println!();
            self.print_sprt_summary(engine1_name, engine2_name, &config, &result);
        }
        println!();

        println!("{}", "═".repeat(80).info().bold());

        Ok(())
    }

    fn print_sprt_summary(
        &self,
        engine1_name: &str,
        engine2_name: &str,
        config: &SprtConfig,
        result: &SprtResult,
    ) {
        let llr_str = match result.status {
            SprtStatus::AcceptH1 => format!("{:.2}", result.llr).success().bold(),
            SprtStatus::AcceptH0 => format!("{:.2}", result.llr).failure().bold(),
            SprtStatus::Continue => format!("{:.2}", result.llr).info(),
        };

        println!(
            "{} LLR {} ({:.2}, {:.2}) {} {}",
            "SPRT:".text().bold(),
            llr_str,
            result.lower,
            result.upper,
            format!("[{:.2}, {:.2}]", config.elo0, config.elo1).subtext(),
            format!("α={} β={}", config.alpha, config.beta).subtext()
        );

        let verdict = sprt_verdict(engine1_name, engine2_name, config, result.status);
        let verdict = match result.status {
            SprtStatus::AcceptH1 => verdict.success().bold(),
            SprtStatus::AcceptH0 => verdict.failure().bold(),
            SprtStatus::Continue => verdict.warning(),
        };
        println!("{} {}", "Verdict:".text().bold(), verdict);
    }

    fn print_summary(&self) {
        let is_ggs = self.synchro;
        // Calculate pentanomial frequencies
        let freq = self.paired_frequencies;

        // Combine DD and WL for pentanomial representation (0-2 format)
        let ptnml = [
            freq.ll,           // 0: Both losses
            freq.ld,           // 1: Loss-Draw
            freq.dd + freq.wl, // 2: Draw-Draw or Win-Loss
            freq.wd,           // 3: Win-Draw
            freq.ww,           // 4: Both wins
        ];

        if is_ggs {
            println!("Completed games: {}", self.total_games());
        }
        let label = if is_ggs {
            "GGS synchro matches:"
        } else {
            "Games:"
        };
        println!(
            "{} {} {} {} {} {} {} {}",
            label.text().bold(),
            self.total_results().to_string().warning(),
            "W:".text().bold(),
            self.engine1_wins.to_string().success(),
            "L:".text().bold(),
            self.engine2_wins.to_string().failure(),
            "D:".text().bold(),
            self.draws.to_string().info()
        );

        if !is_ggs {
            println!(
                "{} {}, {}, {}, {}, {}",
                "Ptnml(0-2):".text().bold(),
                ptnml[0].to_string().subtext(),
                ptnml[1].to_string().subtext(),
                ptnml[2].to_string().subtext(),
                ptnml[3].to_string().subtext(),
                ptnml[4].to_string().subtext()
            );
        }

        // Calculate and display Elo and LOS
        let stats = if is_ggs {
            (self.total_results() > 0).then(|| {
                calculate_trinomial_stats(self.engine1_wins, self.engine2_wins, self.draws)
            })
        } else {
            (freq.total_pairs() > 0).then(|| calculate_pentanomial_stats(&freq))
        };
        if let Some(stats) = stats {
            let los = stats.calculate_los();

            // Format Elo
            let elo_str = if stats.elo_diff.is_infinite() {
                if stats.elo_diff > 0.0 {
                    format!("{} ± {}", "∞".success().bold(), "∞".subtext())
                } else {
                    format!("{} ± {}", "-∞".failure().bold(), "∞".subtext())
                }
            } else {
                let elo_colored = if stats.elo_diff >= 10.0 {
                    format!("{:+.1}", stats.elo_diff).success().bold()
                } else if stats.elo_diff > 0.0 {
                    format!("{:+.1}", stats.elo_diff).success()
                } else if stats.elo_diff == 0.0 {
                    format!("{:+.1}", stats.elo_diff).subtext()
                } else if stats.elo_diff >= -10.0 {
                    format!("{:+.1}", stats.elo_diff).failure()
                } else {
                    format!("{:+.1}", stats.elo_diff).failure().bold()
                };
                format!(
                    "{} ± {}",
                    elo_colored,
                    format!("{:.1}", stats.confidence_interval).subtext()
                )
            };

            let unit = if is_ggs { " (per match)" } else { "" };
            println!("{} {}", format!("Elo{unit}:").text().bold(), elo_str);

            // Calculate and display expected win rate from Elo
            if !stats.elo_diff.is_infinite() {
                let win_rate = 1.0 / (1.0 + 10.0_f64.powf(-stats.elo_diff / 400.0));
                let win_rate_pct = win_rate * 100.0;
                let win_rate_str = if win_rate_pct >= 55.0 {
                    format!("{:.1}%", win_rate_pct).success().bold()
                } else if win_rate_pct > 50.0 {
                    format!("{:.1}%", win_rate_pct).success()
                } else if win_rate_pct == 50.0 {
                    format!("{:.1}%", win_rate_pct).subtext()
                } else if win_rate_pct >= 45.0 {
                    format!("{:.1}%", win_rate_pct).failure()
                } else {
                    format!("{:.1}%", win_rate_pct).failure().bold()
                };
                println!(
                    "{} {}",
                    format!("Win rate{unit}:").text().bold(),
                    win_rate_str
                );
            }

            // Format LOS
            let los_pct = los * 100.0;
            let los_str = if los_pct >= 99.0 {
                format!("{:.2}%", los_pct).success().bold()
            } else if los_pct >= 95.0 {
                format!("{:.1}%", los_pct).success()
            } else if los_pct >= 80.0 {
                format!("{:.1}%", los_pct).info()
            } else if los_pct >= 60.0 {
                format!("{:.1}%", los_pct).warning()
            } else if los_pct >= 40.0 {
                format!("{:.1}%", los_pct).danger()
            } else {
                format!("{:.1}%", los_pct).failure().bold()
            };

            println!("{} {}", "LOS:".text().bold(), los_str);
        }

        if is_ggs && self.total_results() > 0 {
            println!(
                "{} {:.6}",
                "GGS average rating score:".text().bold(),
                self.rating_sum / f64::from(self.total_results())
            );
        }

        // Display Disc diff
        let avg_score = if self.total_games() > 0 {
            self.total_score as f64 / self.total_games() as f64
        } else {
            0.0
        };

        let disc_diff_str = if self.total_score >= 10 {
            format!(
                "{} ({})",
                format!("+{}", self.total_score).success().bold(),
                format!("{:+.2}/game", avg_score).subtext()
            )
        } else if self.total_score > 0 {
            format!(
                "{} ({})",
                format!("+{}", self.total_score).success(),
                format!("{:+.2}/game", avg_score).subtext()
            )
        } else if self.total_score == 0 {
            format!(
                "{} ({})",
                self.total_score.to_string().subtext(),
                format!("{:+.2}/game", avg_score).subtext()
            )
        } else if self.total_score >= -10 {
            format!(
                "{} ({})",
                self.total_score.to_string().failure(),
                format!("{:+.2}/game", avg_score).subtext()
            )
        } else {
            format!(
                "{} ({})",
                self.total_score.to_string().failure().bold(),
                format!("{:+.2}/game", avg_score).subtext()
            )
        };

        println!("{} {}", "Disc diff:".text().bold(), disc_diff_str);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchWinner {
    Engine1,
    Engine2,
    Draw,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PentanomialFrequencies {
    pub ll: u32, // 0-2
    pub ld: u32, // 0.5-1.5
    pub dd: u32, // 1-1 (draw-draw)
    pub wl: u32, // 1-1 (win-loss)
    pub wd: u32, // 1.5-0.5
    pub ww: u32, // 2-0
}

impl PentanomialFrequencies {
    pub(crate) fn add_pair(&mut self, game1: MatchWinner, game2: MatchWinner) {
        match (game1, game2) {
            (MatchWinner::Engine2, MatchWinner::Engine2) => self.ll += 1,
            (MatchWinner::Engine2, MatchWinner::Draw)
            | (MatchWinner::Draw, MatchWinner::Engine2) => {
                self.ld += 1;
            }
            (MatchWinner::Draw, MatchWinner::Draw) => self.dd += 1,
            (MatchWinner::Engine1, MatchWinner::Engine2)
            | (MatchWinner::Engine2, MatchWinner::Engine1) => {
                self.wl += 1;
            }
            (MatchWinner::Engine1, MatchWinner::Draw)
            | (MatchWinner::Draw, MatchWinner::Engine1) => {
                self.wd += 1;
            }
            (MatchWinner::Engine1, MatchWinner::Engine1) => self.ww += 1,
        }
    }

    /// Total number of completed opening pairs.
    pub fn total_pairs(&self) -> u32 {
        self.ll + self.ld + self.dd + self.wl + self.wd + self.ww
    }
}

fn sprt_verdict(
    engine1_name: &str,
    engine2_name: &str,
    config: &SprtConfig,
    status: SprtStatus,
) -> String {
    match status {
        SprtStatus::AcceptH1 => format!(
            "H1 accepted — hypothesis: {engine1_name} - {engine2_name} Elo = {:+.2}",
            config.elo1
        ),
        SprtStatus::AcceptH0 => format!(
            "H0 accepted — hypothesis: {engine1_name} - {engine2_name} Elo = {:+.2}",
            config.elo0
        ),
        SprtStatus::Continue => "Inconclusive — LLR did not reach either bound".to_string(),
    }
}

pub struct RatingStats {
    pub elo_diff: f64,
    pub confidence_interval: f64,
}

impl RatingStats {
    pub fn calculate_los(&self) -> f64 {
        if self.elo_diff.is_infinite() {
            return if self.elo_diff > 0.0 { 1.0 } else { 0.0 };
        }

        if self.confidence_interval == 0.0 {
            // Zero variance: all pairs identical. LOS is determined by sign of elo_diff.
            return if self.elo_diff > 0.0 {
                1.0
            } else if self.elo_diff < 0.0 {
                0.0
            } else {
                0.5
            };
        }

        // Using normal approximation
        let z = self.elo_diff / (self.confidence_interval / 1.96);
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    }
}

fn calculate_pentanomial_stats(freq: &PentanomialFrequencies) -> RatingStats {
    let n_pairs = (freq.ll + freq.ld + freq.dd + freq.wl + freq.wd + freq.ww) as f64;

    if n_pairs == 0.0 {
        return RatingStats {
            elo_diff: 0.0,
            confidence_interval: 0.0,
        };
    }

    // Calculate probabilities for true 5-nomial model
    let p0 = freq.ll as f64 / n_pairs; // 0 points per pair
    let p_half = freq.ld as f64 / n_pairs; // 0.5 points per pair
    let p1_dd = freq.dd as f64 / n_pairs; // 1 point per pair (draw-draw)
    let p1_wl = freq.wl as f64 / n_pairs; // 1 point per pair (win-loss)
    let p_three_half = freq.wd as f64 / n_pairs; // 1.5 points per pair
    let p2 = freq.ww as f64 / n_pairs; // 2 points per pair

    // Calculate score (expected points per pair)
    let score = 0.0 * p0 + 0.5 * p_half + 1.0 * p1_dd + 1.0 * p1_wl + 1.5 * p_three_half + 2.0 * p2;
    let mu = score / 2.0; // Per game

    // Calculate pentanomial variance per pair using true 5-nomial model
    // (DD and WL both score 1 point per pair and contribute identically here)
    let e_t_squared =
        0.0 * p0 + 0.25 * p_half + 1.0 * p1_dd + 1.0 * p1_wl + 2.25 * p_three_half + 4.0 * p2;
    let var_pair = e_t_squared - score * score;

    calculate_rating_stats(n_pairs, mu, var_pair / 4.0)
}

fn calculate_trinomial_stats(wins: u32, losses: u32, draws: u32) -> RatingStats {
    let n = f64::from(wins + losses + draws);
    let mean = (f64::from(wins) + 0.5 * f64::from(draws)) / n;
    let variance = (f64::from(wins) + 0.25 * f64::from(draws)) / n - mean * mean;
    calculate_rating_stats(n, mean, variance)
}

fn calculate_rating_stats(samples: f64, mu: f64, variance: f64) -> RatingStats {
    // Calculate ELO difference
    let elo_diff = if mu == 0.0 || mu == 1.0 {
        if mu > 0.5 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        -ELO_K * (-(mu / (1.0 - mu)).ln()) / std::f64::consts::LN_10
    };

    // Calculate confidence interval
    let se_elo = if mu == 0.0 || mu == 1.0 {
        f64::INFINITY
    } else {
        let se_mu = (variance / samples).sqrt();
        (ELO_K / std::f64::consts::LN_10) * se_mu / (mu * (1.0 - mu))
    };

    let z_score = 1.96;
    let confidence_interval = z_score * se_elo;

    RatingStats {
        elo_diff,
        confidence_interval,
    }
}

// Error function approximation for LOS calculation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_report_retains_random_seed() {
        for (case, has_games, seed) in [
            ("completed", true, Some(20261004_u64)),
            ("partial", true, Some(20261004)),
            ("partial_after_pair", true, Some(20261004)),
            ("empty", false, Some(20261004)),
            ("ordinary", true, None),
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "statistics::tests::final_report_fixture",
                    "--nocapture",
                ])
                .env("MATCH_RUNNER_REPORT_CASE", case)
                .output()
                .unwrap();
            assert!(output.status.success());
            let report = String::from_utf8(output.stdout).unwrap();
            if let Some(seed) = seed {
                assert!(
                    report.contains(&format!("GGS random openings seed {seed}")),
                    "{report}"
                );
            } else {
                assert!(!report.contains("GGS random openings seed"), "{report}");
            }
            assert_eq!(
                report.contains("No games played yet."),
                !has_games,
                "{report}"
            );
            assert_eq!(report.contains("Games:"), case == "ordinary", "{report}");
            assert_eq!(
                report.contains("Ptnml(0-2):"),
                case == "ordinary",
                "{report}"
            );
            if case == "completed" || case == "partial_after_pair" {
                assert!(
                    report.contains("GGS synchro matches: 1 W: 1 L: 0 D: 0"),
                    "{report}"
                );
                assert!(report.contains("Elo (per match):"), "{report}");
                assert!(
                    report.contains("GGS average rating score: 0.935484"),
                    "{report}"
                );
            }
            match case {
                "completed" => {
                    assert!(report.contains("Disc diff: +18 (+9.00/game)"), "{report}");
                }
                "partial_after_pair" => {
                    assert!(report.contains("Disc diff: -46 (-15.33/game)"), "{report}");
                }
                "partial" => {
                    assert!(
                        report.contains("GGS synchro matches: 0 W: 0 L: 0 D: 0"),
                        "{report}"
                    );
                    assert!(!report.contains("Elo"), "{report}");
                    assert!(report.contains("Disc diff: +20 (+20.00/game)"), "{report}");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn final_report_fixture() {
        let Ok(case) = std::env::var("MATCH_RUNNER_REPORT_CASE") else {
            return;
        };
        let mut statistics = MatchStatistics {
            random_seed: (case != "ordinary").then_some(20261004),
            synchro: case != "ordinary",
            ..MatchStatistics::default()
        };
        match case.as_str() {
            "empty" => {}
            "partial" => statistics.add_game(MatchWinner::Engine1, 20, String::new(), true),
            _ => play_pair(
                &mut statistics,
                (MatchWinner::Engine1, 20),
                (MatchWinner::Engine2, -2),
            ),
        }
        if case == "partial_after_pair" {
            statistics.add_game(MatchWinner::Engine2, -64, String::new(), true);
        }
        statistics
            .print_final_results("engine1", "engine2", None)
            .unwrap();
    }

    fn play_pair(
        stats: &mut MatchStatistics,
        first: (MatchWinner, i32),
        second: (MatchWinner, i32),
    ) {
        stats.add_game(first.0, first.1, String::new(), true);
        stats.add_game(second.0, second.1, String::new(), false);
        stats.add_pair(first, second, String::new());
    }

    #[test]
    fn synchro_counts_one_outcome_from_the_combined_disc_margin() {
        use MatchWinner::{Draw, Engine1, Engine2};

        let mut stats = MatchStatistics {
            synchro: true,
            ..MatchStatistics::default()
        };
        for (first_winner, first, second_winner, second, expected) in [
            (Engine1, 20, Engine2, -2, Engine1),
            (Engine1, 2, Engine2, -20, Engine2),
            (Engine1, 20, Engine2, -20, Draw),
            (Engine1, 20, Engine1, 2, Engine1),
            (Engine2, -20, Engine2, -2, Engine2),
            (Draw, 0, Draw, 0, Draw),
        ] {
            let results = stats.total_results();
            play_pair(&mut stats, (first_winner, first), (second_winner, second));
            assert_eq!(stats.total_results(), results + 1);
            let history = stats.recent_results.last().unwrap();
            assert_eq!(history.winner, expected);
            assert_eq!(history.score, first + second);
            assert_eq!(history.engine1_color, None);
        }
        assert_eq!(stats.total_games(), 12);
        assert_eq!(
            (stats.engine1_wins, stats.engine2_wins, stats.draws),
            (2, 2, 2)
        );
        assert_eq!(stats.paired_frequencies.total_pairs(), 0);
    }

    #[test]
    fn ordinary_mode_keeps_game_outcomes_and_pentanomial_pairs() {
        let mut stats = MatchStatistics::default();
        play_pair(
            &mut stats,
            (MatchWinner::Engine1, 20),
            (MatchWinner::Engine2, -2),
        );
        assert_eq!(
            (stats.engine1_wins, stats.engine2_wins, stats.draws),
            (1, 1, 0)
        );
        assert_eq!(stats.total_games(), 2);
        assert_eq!(stats.total_score, 18);
        assert_eq!(stats.paired_frequencies.wl, 1);
        assert_eq!(stats.recent_results.len(), 2);
        assert_eq!(stats.recent_results[0].engine1_color, Some(Disc::Black));
        assert_eq!(stats.recent_results[1].engine1_color, Some(Disc::White));
    }

    #[test]
    fn synchro_sprt_uses_match_outcomes_and_ignores_unfinished_pairs() {
        let config = SprtConfig {
            elo0: -10.0,
            elo1: 10.0,
            alpha: 0.05,
            beta: 0.05,
        };
        let mut stats = MatchStatistics {
            synchro: true,
            ..MatchStatistics::default()
        };
        let mut sprt = Sprt::new(config);
        let mut reference = Sprt::new(config);
        let mut previous = stats.update_sprt(&mut sprt);
        for pair in 1..=100 {
            stats.add_game(MatchWinner::Engine1, 20, String::new(), true);
            let partial = stats.update_sprt(&mut sprt);
            assert_eq!(partial.llr, previous.llr);
            assert_eq!(partial.status, previous.status);
            stats.add_game(MatchWinner::Engine2, -2, String::new(), false);
            stats.add_pair(
                (MatchWinner::Engine1, 20),
                (MatchWinner::Engine2, -2),
                String::new(),
            );
            let actual = stats.update_sprt(&mut sprt);
            let expected = reference.update_trinomial(pair, 0, 0);
            assert_eq!(actual.llr, expected.llr);
            assert_eq!(actual.status, expected.status);
            if actual.status == SprtStatus::AcceptH1 {
                return;
            }
            previous = actual;
        }
        panic!("positive synchro margins must eventually accept H1");
    }

    #[test]
    fn synchro_elo_and_uncertainty_count_each_match_once() {
        let stats = calculate_trinomial_stats(40, 20, 40);
        let mean: f64 = 0.6;
        let variance: f64 = 0.14;
        let expected_ci = 1.96 * 400.0 / std::f64::consts::LN_10 * (variance / 100.0).sqrt()
            / (mean * (1.0 - mean));
        assert!((stats.elo_diff - 70.4365).abs() < 0.01);
        assert!((stats.confidence_interval - expected_ci).abs() < 1e-10);
    }

    #[test]
    fn add_pair_updates_pentanomial_frequencies() {
        let mut frequencies = PentanomialFrequencies::default();
        frequencies.add_pair(MatchWinner::Engine2, MatchWinner::Engine2);
        frequencies.add_pair(MatchWinner::Engine2, MatchWinner::Draw);
        frequencies.add_pair(MatchWinner::Draw, MatchWinner::Draw);
        frequencies.add_pair(MatchWinner::Engine1, MatchWinner::Engine2);
        frequencies.add_pair(MatchWinner::Engine1, MatchWinner::Draw);
        frequencies.add_pair(MatchWinner::Engine1, MatchWinner::Engine1);

        assert_eq!(
            frequencies,
            PentanomialFrequencies {
                ll: 1,
                ld: 1,
                dd: 1,
                wl: 1,
                wd: 1,
                ww: 1,
            }
        );
    }

    #[test]
    fn sprt_verdict_names_the_configured_hypothesis() {
        let positive = SprtConfig {
            elo0: 10.0,
            elo1: 20.0,
            alpha: 0.05,
            beta: 0.05,
        };
        let negative = SprtConfig {
            elo0: -20.0,
            elo1: -10.0,
            alpha: 0.05,
            beta: 0.05,
        };
        let one_sided = SprtConfig {
            elo0: 0.0,
            elo1: 10.0,
            alpha: 0.05,
            beta: 0.05,
        };

        assert_eq!(
            sprt_verdict("Engine A", "Engine B", &positive, SprtStatus::AcceptH0),
            "H0 accepted — hypothesis: Engine A - Engine B Elo = +10.00"
        );
        assert_eq!(
            sprt_verdict("Engine A", "Engine B", &negative, SprtStatus::AcceptH1),
            "H1 accepted — hypothesis: Engine A - Engine B Elo = -10.00"
        );
        assert_eq!(
            sprt_verdict("Engine A", "Engine B", &one_sided, SprtStatus::AcceptH0),
            "H0 accepted — hypothesis: Engine A - Engine B Elo = +0.00"
        );
    }

    #[test]
    fn zero_pairs_are_even_with_no_uncertainty() {
        let stats = calculate_pentanomial_stats(&PentanomialFrequencies::default());

        assert_eq!(stats.elo_diff, 0.0);
        assert_eq!(stats.confidence_interval, 0.0);
        assert_eq!(stats.calculate_los(), 0.5);
    }

    #[test]
    fn all_draw_draw_pairs_are_even_with_no_uncertainty() {
        let frequencies = PentanomialFrequencies {
            dd: 50,
            ..PentanomialFrequencies::default()
        };
        let stats = calculate_pentanomial_stats(&frequencies);

        assert_eq!(stats.elo_diff, 0.0);
        assert_eq!(stats.confidence_interval, 0.0);
        assert_eq!(stats.calculate_los(), 0.5);
    }

    #[test]
    fn decisive_pairs_have_infinite_elo_and_certain_los() {
        let all_wins = calculate_pentanomial_stats(&PentanomialFrequencies {
            ww: 50,
            ..PentanomialFrequencies::default()
        });
        let all_losses = calculate_pentanomial_stats(&PentanomialFrequencies {
            ll: 50,
            ..PentanomialFrequencies::default()
        });

        assert!(all_wins.elo_diff.is_infinite() && all_wins.elo_diff > 0.0);
        assert_eq!(all_wins.calculate_los(), 1.0);
        assert!(all_losses.elo_diff.is_infinite() && all_losses.elo_diff < 0.0);
        assert_eq!(all_losses.calculate_los(), 0.0);
    }

    #[test]
    fn symmetric_decisive_and_draw_pairs_are_even() {
        let frequencies = PentanomialFrequencies {
            ww: 30,
            dd: 40,
            ll: 30,
            ..PentanomialFrequencies::default()
        };
        let stats = calculate_pentanomial_stats(&frequencies);

        assert_eq!(stats.elo_diff, 0.0);
        assert!((stats.calculate_los() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn asymmetric_pairs_match_hand_computed_elo() {
        let frequencies = PentanomialFrequencies {
            ww: 40,
            dd: 40,
            ll: 20,
            ..PentanomialFrequencies::default()
        };
        let stats = calculate_pentanomial_stats(&frequencies);

        assert!((stats.elo_diff - 70.4365).abs() < 0.01);
        assert!(stats.calculate_los() > 0.99);
    }

    #[test]
    fn erf_matches_known_values() {
        assert!(erf(0.0).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427).abs() < 1e-3);
    }
}
