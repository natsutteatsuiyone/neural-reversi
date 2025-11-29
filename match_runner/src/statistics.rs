use crate::colors::ThemeColor;
use colored::*;
use reversi_core::piece::Piece;
use std::io;

const ELO_K: f64 = 400.0;

#[derive(Debug, Clone)]
pub struct MatchStatistics {
    pub engine1_wins: u32,
    pub engine2_wins: u32,
    pub draws: u32,
    pub total_score: i32,
    pub games_played: u32,
    pub recent_results: Vec<GameHistory>,
    pub paired_results: Vec<PairedResult>,
}

#[derive(Debug, Clone)]
pub struct GameHistory {
    pub winner: MatchWinner,
    pub score: i32,
    pub opening: String,
    pub engine1_color: Piece,
}

#[derive(Debug, Clone)]
pub struct PairedResult {
    pub game1: (MatchWinner, i32),
    pub game2: (MatchWinner, i32),
}

impl Default for MatchStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchStatistics {
    pub fn new() -> Self {
        Self {
            engine1_wins: 0,
            engine2_wins: 0,
            draws: 0,
            total_score: 0,
            games_played: 0,
            recent_results: Vec::new(),
            paired_results: Vec::new(),
        }
    }

    pub fn add_result(
        &mut self,
        winner: MatchWinner,
        score: i32,
        opening: String,
        engine1_is_black: bool,
    ) {
        match winner {
            MatchWinner::Engine1 => self.engine1_wins += 1,
            MatchWinner::Engine2 => self.engine2_wins += 1,
            MatchWinner::Draw => self.draws += 1,
        }
        self.total_score += score;
        self.games_played += 1;

        let engine1_color = if engine1_is_black {
            Piece::Black
        } else {
            Piece::White
        };

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

    pub fn total_games(&self) -> u32 {
        self.engine1_wins + self.engine2_wins + self.draws
    }

    pub fn engine1_win_rate(&self) -> f64 {
        if self.total_games() == 0 {
            0.0
        } else {
            (self.engine1_wins as f64 / self.total_games() as f64) * 100.0
        }
    }

    pub fn engine2_win_rate(&self) -> f64 {
        if self.total_games() == 0 {
            0.0
        } else {
            (self.engine2_wins as f64 / self.total_games() as f64) * 100.0
        }
    }

    pub fn print_final_results(&self, engine1_name: &str, engine2_name: &str) -> io::Result<()> {
        let total_games = self.total_games();

        if total_games == 0 {
            println!("{}", "No games played yet.".info());
            return Ok(());
        }

        // Clear line and print header
        println!("\r\x1B[2K");
        println!("{}", "═".repeat(80).info().bold());
        let header = format!("{} vs {}", engine1_name, engine2_name);
        println!("{:^80}", header.primary().bold());
        println!("{}", "═".repeat(80).info().bold());
        println!();

        self.print_summary();
        println!();

        println!("{}", "═".repeat(80).info().bold());

        Ok(())
    }

    fn print_summary(&self) {
        // Calculate pentanomial frequencies
        let freq = self.calculate_pentanomial_frequencies();

        // Combine DD and WL for pentanomial representation (0-2 format)
        let ptnml = [
            freq.ll,           // 0: Both losses
            freq.ld,           // 1: Loss-Draw
            freq.dd + freq.wl, // 2: Draw-Draw or Win-Loss
            freq.wd,           // 3: Win-Draw
            freq.ww,           // 4: Both wins
        ];

        println!(
            "{} {} {} {} {} {} {} {}",
            "Games:".text().bold(),
            self.total_games().to_string().warning(),
            "W:".text().bold(),
            self.engine1_wins.to_string().success(),
            "L:".text().bold(),
            self.engine2_wins.to_string().failure(),
            "D:".text().bold(),
            self.draws.to_string().info()
        );

        println!(
            "{} {}, {}, {}, {}, {}",
            "Ptnml(0-2):".text().bold(),
            ptnml[0].to_string().subtext(),
            ptnml[1].to_string().subtext(),
            ptnml[2].to_string().subtext(),
            ptnml[3].to_string().subtext(),
            ptnml[4].to_string().subtext()
        );

        // Calculate and display Elo and LOS
        if !self.paired_results.is_empty() {
            let stats = PentanomialCalculator::calculate(&freq);
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

            println!("{} {}", "Elo:".text().bold(), elo_str);

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
                println!("{} {}", "Win rate:".text().bold(), win_rate_str);
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

        // Display Disk diff
        let avg_score = if self.total_games() > 0 {
            self.total_score as f64 / self.total_games() as f64
        } else {
            0.0
        };

        let disk_diff_str = if self.total_score >= 10 {
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

        println!("{} {}", "Disk diff:".text().bold(), disk_diff_str);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MatchWinner {
    Engine1,
    Engine2,
    Draw,
}

impl MatchStatistics {
    pub fn add_paired_result(&mut self, game1: (MatchWinner, i32), game2: (MatchWinner, i32)) {
        self.paired_results.push(PairedResult { game1, game2 });
    }

    pub fn calculate_pentanomial_frequencies(&self) -> PentanomialFrequencies {
        let mut freq = PentanomialFrequencies::default();

        for paired in &self.paired_results {
            match (paired.game1.0, paired.game2.0) {
                (MatchWinner::Engine2, MatchWinner::Engine2) => freq.ll += 1,
                (MatchWinner::Engine2, MatchWinner::Draw)
                | (MatchWinner::Draw, MatchWinner::Engine2) => freq.ld += 1,
                (MatchWinner::Draw, MatchWinner::Draw) => freq.dd += 1,
                (MatchWinner::Engine1, MatchWinner::Engine2)
                | (MatchWinner::Engine2, MatchWinner::Engine1) => freq.wl += 1,
                (MatchWinner::Engine1, MatchWinner::Draw)
                | (MatchWinner::Draw, MatchWinner::Engine1) => freq.wd += 1,
                (MatchWinner::Engine1, MatchWinner::Engine1) => freq.ww += 1,
            }
        }

        freq
    }
}

#[derive(Debug, Default)]
pub struct PentanomialFrequencies {
    pub ll: u32, // 0-2
    pub ld: u32, // 0.5-1.5
    pub dd: u32, // 1-1 (draw-draw)
    pub wl: u32, // 1-1 (win-loss)
    pub wd: u32, // 1.5-0.5
    pub ww: u32, // 2-0
}

pub struct PentanomialStats {
    pub elo_diff: f64,
    pub confidence_interval: f64,
}

impl PentanomialStats {
    pub fn calculate_los(&self) -> f64 {
        if self.elo_diff.is_infinite() {
            return if self.elo_diff > 0.0 { 1.0 } else { 0.0 };
        }

        // Using normal approximation
        let z = self.elo_diff / (self.confidence_interval / 1.96);
        0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
    }
}

pub struct PentanomialCalculator;

impl PentanomialCalculator {
    pub fn calculate(freq: &PentanomialFrequencies) -> PentanomialStats {
        let n_pairs = (freq.ll + freq.ld + freq.dd + freq.wl + freq.wd + freq.ww) as f64;

        if n_pairs == 0.0 {
            return PentanomialStats {
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
        let score =
            0.0 * p0 + 0.5 * p_half + 1.0 * p1_dd + 1.0 * p1_wl + 1.5 * p_three_half + 2.0 * p2;
        let mu = score / 2.0; // Per game

        // Calculate pentanomial variance per pair using true 5-nomial model
        // Note: DD and WL both give 1 point but have different variance contributions
        let e_t_squared =
            0.0 * p0 + 0.25 * p_half + 1.0 * p1_dd + 1.0 * p1_wl + 2.25 * p_three_half + 4.0 * p2;
        let var_pair = e_t_squared - score * score;

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
            let se_mu = (var_pair / 4.0 / n_pairs).sqrt();
            (ELO_K / std::f64::consts::LN_10) * se_mu / (mu * (1.0 - mu))
        };

        let z_score = 1.96;
        let confidence_interval = z_score * se_elo;

        PentanomialStats {
            elo_diff,
            confidence_interval,
        }
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
