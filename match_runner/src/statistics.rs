use std::io;
use colored::*;
use reversi_core::piece::Piece;

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

    pub fn add_result(&mut self, winner: MatchWinner, score: i32, opening: String, engine1_is_black: bool) {
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

        self.recent_results.push(GameHistory { winner, score, opening, engine1_color });
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

    pub fn engine1_score_rate(&self) -> f64 {
        if self.total_games() == 0 {
            0.0
        } else {
            ((self.engine1_wins as f64 + self.draws as f64 * 0.5) / self.total_games() as f64) * 100.0
        }
    }

    pub fn engine2_score_rate(&self) -> f64 {
        if self.total_games() == 0 {
            0.0
        } else {
            ((self.engine2_wins as f64 + self.draws as f64 * 0.5) / self.total_games() as f64) * 100.0
        }
    }


    pub fn print_final_results(&self, engine1_name: &str, engine2_name: &str) -> io::Result<()> {
        let total_games = self.total_games();

        if total_games == 0 {
            println!("No games were played.");
            return Ok(());
        }

        // Clear line and print header
        println!("\r\x1B[2K");
        println!("{}", "═".repeat(80).bright_cyan());
        println!("{:^80}", "MATCH RESULTS".bright_white().bold());
        println!("{}", "═".repeat(80).bright_cyan());
        println!();

        // Display match summary
        self.print_match_summary(total_games);
        println!();

        // Display detailed statistics
        self.print_detailed_stats(engine1_name, engine2_name)?;
        println!();

        // Display ELO rating
        self.print_elo_rating(engine1_name, engine2_name, total_games)?;
        println!();


        println!("{}", "═".repeat(80).bright_cyan());

        Ok(())
    }

    fn print_match_summary(&self, total_games: u32) {
        println!("{} {}", "Total Games:".bright_white(), total_games.to_string().bright_yellow().bold());
        println!("{} {} / {} / {}",
            "Results:".bright_white(),
            format!("{} wins", self.engine1_wins).bright_green(),
            format!("{} draws", self.draws).bright_blue(),
            format!("{} losses", self.engine2_wins).bright_red()
        );
    }

    fn print_detailed_stats(&self, engine1_name: &str, engine2_name: &str) -> io::Result<()> {
        let name_width = std::cmp::max(engine1_name.len(), engine2_name.len());

        // Header
        println!("{}", format!(
            "┌─{:─<width$}─┬─{:─^7}─┬─{:─^7}─┬─{:─^7}─┬─{:─^10}─┬─{:─^22}─┐",
            "", "", "", "", "", "",
            width = name_width
        ).bright_black());

        println!(
            "│ {:^width$} │ {:^7} │ {:^7} │ {:^7} │ {:^10} │ {:^22} │",
            "Engine".bright_white(),
            "Wins".bright_white(),
            "Losses".bright_white(),
            "Draws".bright_white(),
            "Win Rate".bright_white(),
            "Score (discs)".bright_white(),
            width = name_width
        );

        println!("{}", format!(
            "├─{:─<width$}─┼─{:─^7}─┼─{:─^7}─┼─{:─^7}─┼─{:─^10}─┼─{:─^22}─┤",
            "", "", "", "", "", "",
            width = name_width
        ).bright_black());

        // Engine 1 row
        let avg_score1 = if self.total_games() > 0 {
            self.total_score as f64 / self.total_games() as f64
        } else {
            0.0
        };
        let engine1_score_str = format!("{:+} ({:+.2}/game)", self.total_score, avg_score1);
        let engine1_score = match self.total_score.cmp(&0) {
            std::cmp::Ordering::Greater => engine1_score_str.bright_green(),
            std::cmp::Ordering::Less => engine1_score_str.bright_red(),
            std::cmp::Ordering::Equal => engine1_score_str.bright_yellow(),
        };

        println!(
            "│ {:width$} │ {:^7} │ {:^7} │ {:^7} │ {:^10} │ {:>22} │",
            engine1_name.bright_cyan().bold(),
            self.engine1_wins.to_string().bright_green(),
            self.engine2_wins.to_string().bright_red(),
            self.draws.to_string().bright_blue(),
            format!("{:.1}%", self.engine1_score_rate()).bright_yellow(),
            engine1_score,
            width = name_width
        );

        // Engine 2 row
        let avg_score2 = if self.total_games() > 0 {
            -self.total_score as f64 / self.total_games() as f64
        } else {
            0.0
        };
        let engine2_score_str = format!("{:+} ({:+.2}/game)", -self.total_score, avg_score2);
        let engine2_score = match (-self.total_score).cmp(&0) {
            std::cmp::Ordering::Greater => engine2_score_str.bright_green(),
            std::cmp::Ordering::Less => engine2_score_str.bright_red(),
            std::cmp::Ordering::Equal => engine2_score_str.bright_yellow(),
        };

        println!(
            "│ {:width$} │ {:^7} │ {:^7} │ {:^7} │ {:^10} │ {:>22} │",
            engine2_name.bright_cyan().bold(),
            self.engine2_wins.to_string().bright_green(),
            self.engine1_wins.to_string().bright_red(),
            self.draws.to_string().bright_blue(),
            format!("{:.1}%", self.engine2_score_rate()).bright_yellow(),
            engine2_score,
            width = name_width
        );

        println!("{}", format!(
            "└─{:─<width$}─┴─{:─^7}─┴─{:─^7}─┴─{:─^7}─┴─{:─^10}─┴─{:─^22}─┘",
            "", "", "", "", "", "",
            width = name_width
        ).bright_black());

        Ok(())
    }

    fn print_elo_rating(&self, engine1_name: &str, engine2_name: &str, total_games: u32) -> io::Result<()> {
        // If we have paired results, use pentanomial model for better accuracy
        if !self.paired_results.is_empty() {
            return self.print_combined_elo_stats(engine1_name, engine2_name);
        }

        // Otherwise, use simple trinomial model
        let elo_stats = EloCalculator::calculate_stats(
            self.engine1_wins,
            self.engine2_wins,
            self.draws,
            total_games,
        );

        println!("{}", "Statistical Analysis".bright_white().underline());
        println!();

        self.print_elo_display(&elo_stats, engine1_name, engine2_name);

        Ok(())
    }

    fn print_combined_elo_stats(&self, engine1_name: &str, engine2_name: &str) -> io::Result<()> {
        let freq = self.calculate_pentanomial_frequencies();
        let stats = PentanomialCalculator::calculate(&freq);
        let total_pairs = self.paired_results.len();

        println!("{}", "Statistical Analysis".bright_white().underline());
        println!();

        // Display paired game outcomes in a table format
        println!("{}", "Paired Game Outcomes:".bright_white());

        // Header
        println!("  {:>14} {:>6} {:>8}", "Result", "Count", "Percent");
        println!("  {:>14} {:>6} {:>8}", "-------------", "-----", "-------");

        // Data rows
        let outcomes = [
            ("0-2 (LL)", freq.ll),
            ("½-1½ (LD)", freq.ld),
            ("1-1 (DD)", freq.dd),
            ("1-1 (WL)", freq.wl),
            ("1½-½ (WD)", freq.wd),
            ("2-0 (WW)", freq.ww),
        ];

        for (result, count) in outcomes {
            let percentage = count as f64 / total_pairs as f64 * 100.0;
            let result_colored = if result.starts_with("0-") || result.starts_with("½-") {
                result.bright_red()
            } else if result.starts_with("1-") {
                result.bright_yellow()
            } else {
                result.bright_green()
            };

            println!("  {:>14} {:>6} {:>7.1}%",
                result_colored,
                count,
                percentage
            );
        }

        println!();

        // Display ELO with enhanced confidence interval
        let elo_display = self.format_elo_display(stats.elo_diff, stats.confidence_interval);
        println!("{:>20}: {}", "ELO Difference".bright_white(), elo_display);

        // LOS (Likelihood of Superiority)
        let los = stats.calculate_los();
        println!("{:>20}: {}", "LOS".bright_white(),
            self.format_los(los * 100.0)
        );

        // Performance assessment
        self.print_performance_assessment(stats.elo_diff, stats.confidence_interval, engine1_name, engine2_name);

        Ok(())
    }

    fn print_elo_display(&self, stats: &EloStats, engine1_name: &str, engine2_name: &str) {
        let elo_display = self.format_elo_display(stats.elo_diff, stats.confidence_interval);
        println!("{:>20}: {}", "ELO Difference".bright_white(), elo_display);

        if !stats.elo_diff.is_infinite() {
            let win_rate = 1.0 / (1.0 + 10.0_f64.powf(-stats.elo_diff / 400.0));
            println!("{:>20}: {}", "Expected Win Rate".bright_white(),
                self.format_percentage(win_rate * 100.0, stats.elo_diff)
            );
        }

        self.print_performance_assessment(stats.elo_diff, stats.confidence_interval, engine1_name, engine2_name);
    }

    fn format_elo_display(&self, elo_diff: f64, confidence_interval: f64) -> String {
        if elo_diff.is_infinite() {
            if elo_diff > 0.0 {
                format!("{} {}", "∞".bright_green().bold(), "(Dominant)".bright_black())
            } else {
                format!("{} {}", "-∞".bright_red().bold(), "(Dominant)".bright_black())
            }
        } else {
            let elo_str = format!("{:+.1}", elo_diff);
            let confidence_str = format!("± {:.1}", confidence_interval);
            let colored_elo = if elo_diff > 0.0 {
                elo_str.bright_green().bold()
            } else if elo_diff < 0.0 {
                elo_str.bright_red().bold()
            } else {
                elo_str.bright_yellow().bold()
            };
            format!("{} {}", colored_elo, confidence_str.bright_black())
        }
    }

    fn format_percentage(&self, percentage: f64, elo_diff: f64) -> ColoredString {
        let pct_str = format!("{:.1}%", percentage);
        if elo_diff > 0.0 {
            pct_str.bright_green()
        } else if elo_diff < 0.0 {
            pct_str.bright_red()
        } else {
            pct_str.bright_yellow()
        }
    }

    fn format_los(&self, los_percentage: f64) -> ColoredString {
        let los_str = format!("{:.1}%", los_percentage);
        if los_percentage > 95.0 {
            los_str.bright_green().bold()
        } else if los_percentage > 75.0 {
            los_str.bright_green()
        } else if los_percentage > 50.0 {
            los_str.bright_yellow()
        } else {
            los_str.bright_red()
        }
    }

    fn print_performance_assessment(&self, elo_diff: f64, confidence_interval: f64, engine1_name: &str, engine2_name: &str) {
        let ci_crosses_zero = (elo_diff - confidence_interval) <= 0.0 &&
                             (elo_diff + confidence_interval) >= 0.0;

        let assessment = self.format_performance_assessment(
            elo_diff,
            ci_crosses_zero,
            engine1_name,
            engine2_name
        );

        println!("{:>20}: {}", "Performance".bright_white(), assessment);
    }

    fn format_performance_assessment(&self, elo_diff: f64, ci_crosses_zero: bool, engine1_name: &str, engine2_name: &str) -> ColoredString {
        // Define performance levels with their thresholds
        const SIGNIFICANT_THRESHOLD: f64 = 100.0;
        const CLEAR_THRESHOLD: f64 = 50.0;
        const SLIGHT_THRESHOLD: f64 = 20.0;

        // Early returns for special cases
        if ci_crosses_zero {
            return "Statistically equivalent".bright_blue();
        }

        if elo_diff.is_infinite() {
            return if elo_diff > 0.0 {
                format!("{} is overwhelmingly superior", engine1_name).bright_green().bold()
            } else {
                format!("{} is overwhelmingly superior", engine2_name).bright_red().bold()
            };
        }

        // Determine which engine is stronger
        let (stronger, is_positive) = if elo_diff > 0.0 {
            (engine1_name, true)
        } else {
            (engine2_name, false)
        };

        // Select assessment level and color
        let abs_diff = elo_diff.abs();
        let (assessment_text, color_fn): (&str, fn(String) -> ColoredString) = if abs_diff > SIGNIFICANT_THRESHOLD {
            ("{} is decisively stronger (+{:.0} ELO)", if is_positive {
                |s| s.bright_green()
            } else {
                |s| s.bright_red()
            })
        } else if abs_diff > CLEAR_THRESHOLD {
            ("{} is clearly stronger (+{:.0} ELO)", if is_positive {
                |s| s.bright_green()
            } else {
                |s| s.bright_red()
            })
        } else if abs_diff > SLIGHT_THRESHOLD {
            ("{} is slightly stronger (+{:.0} ELO)", |s| s.bright_yellow())
        } else {
            ("{} is marginally stronger (+{:.0} ELO)", |s| s.bright_yellow())
        };

        color_fn(assessment_text.replace("{}", stronger).replace("{:.0}", &format!("{:.0}", abs_diff)))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MatchWinner {
    Engine1,
    Engine2,
    Draw,
}

pub struct EloStats {
    pub elo_diff: f64,
    pub confidence_interval: f64,
}

pub struct EloCalculator;

impl EloCalculator {
    pub fn calculate_stats(wins: u32, losses: u32, draws: u32, total_games: u32) -> EloStats {
        if total_games == 0 {
            return EloStats {
                elo_diff: 0.0,
                confidence_interval: 0.0,
            };
        }

        let n = total_games as f64;
        let w = wins as f64;
        let d = draws as f64;
        let l = losses as f64;

        let p_hat = (w + 0.5 * d) / n;

        let elo_diff = if p_hat == 0.0 || p_hat == 1.0 {
            if p_hat > 0.5 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            -ELO_K * (-(p_hat / (1.0 - p_hat)).ln()) / std::f64::consts::LN_10
        };

        let wld_var = w * (1.0 - p_hat).powi(2) + l * p_hat.powi(2) + d * (0.5 - p_hat).powi(2);
        let se_elo = if p_hat == 0.0 || p_hat == 1.0 {
            f64::INFINITY
        } else {
            (ELO_K / (std::f64::consts::LN_10 * n)) * wld_var.sqrt() / (p_hat * (1.0 - p_hat))
        };

        let z_score = 1.96;
        let confidence_interval = z_score * se_elo;

        EloStats {
            elo_diff,
            confidence_interval,
        }
    }
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
                (MatchWinner::Engine2, MatchWinner::Draw) | (MatchWinner::Draw, MatchWinner::Engine2) => freq.ld += 1,
                (MatchWinner::Draw, MatchWinner::Draw) => freq.dd += 1,
                (MatchWinner::Engine1, MatchWinner::Engine2) | (MatchWinner::Engine2, MatchWinner::Engine1) => freq.wl += 1,
                (MatchWinner::Engine1, MatchWinner::Draw) | (MatchWinner::Draw, MatchWinner::Engine1) => freq.wd += 1,
                (MatchWinner::Engine1, MatchWinner::Engine1) => freq.ww += 1,
            }
        }

        freq
    }

}

#[derive(Debug, Default)]
pub struct PentanomialFrequencies {
    pub ll: u32,  // 0-2
    pub ld: u32,  // 0.5-1.5
    pub dd: u32,  // 1-1 (draw-draw)
    pub wl: u32,  // 1-1 (win-loss)
    pub wd: u32,  // 1.5-0.5
    pub ww: u32,  // 2-0
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
        let p0 = freq.ll as f64 / n_pairs;        // 0 points per pair
        let p_half = freq.ld as f64 / n_pairs;    // 0.5 points per pair
        let p1_dd = freq.dd as f64 / n_pairs;     // 1 point per pair (draw-draw)
        let p1_wl = freq.wl as f64 / n_pairs;     // 1 point per pair (win-loss)
        let p_three_half = freq.wd as f64 / n_pairs; // 1.5 points per pair
        let p2 = freq.ww as f64 / n_pairs;        // 2 points per pair

        // Calculate score (expected points per pair)
        let score = 0.0 * p0 + 0.5 * p_half + 1.0 * p1_dd + 1.0 * p1_wl + 1.5 * p_three_half + 2.0 * p2;
        let mu = score / 2.0; // Per game

        // Calculate pentanomial variance per pair using true 5-nomial model
        // Note: DD and WL both give 1 point but have different variance contributions
        let e_t_squared = 0.0 * p0 + 0.25 * p_half + 1.0 * p1_dd + 1.0 * p1_wl + 2.25 * p_three_half + 4.0 * p2;
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
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
