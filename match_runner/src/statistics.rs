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
}

#[derive(Debug, Clone)]
pub struct GameHistory {
    pub winner: MatchWinner,
    pub score: i32,
    pub opening: String,
    pub engine1_color: Piece,
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

        // Display results with visual bars
        self.print_visual_results(engine1_name, engine2_name)?;
        println!();

        // Display detailed statistics
        self.print_detailed_stats(engine1_name, engine2_name)?;
        println!();

        // Display ELO rating
        self.print_elo_rating(engine1_name, engine2_name, total_games)?;

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

    fn print_visual_results(&self, engine1_name: &str, engine2_name: &str) -> io::Result<()> {
        let total = self.total_games() as f64;
        let bar_width = 60;  // Match the width used in live display

        // Calculate the maximum width needed for proper alignment
        let max_name_len = std::cmp::max(
            std::cmp::max(engine1_name.len(), engine2_name.len()),
            "Draws".len()
        );
        let name_width = std::cmp::max(max_name_len, 12); // Minimum width of 12

        println!("{}", "Win Rate".bright_white().underline());
        println!();

        // Engine 1 bar
        let engine1_bar_len = ((self.engine1_wins as f64 / total) * bar_width as f64) as usize;
        let engine1_bar = "█".repeat(engine1_bar_len).bright_green();
        let engine1_empty = "░".repeat(bar_width - engine1_bar_len).bright_black();

        // Engine 2 bar
        let engine2_bar_len = ((self.engine2_wins as f64 / total) * bar_width as f64) as usize;
        let engine2_bar = "█".repeat(engine2_bar_len).bright_red();
        let engine2_empty = "░".repeat(bar_width - engine2_bar_len).bright_black();

        // Draw rate visualization
        let draw_percentage = (self.draws as f64 / total) * 100.0;

        // Engine 1
        println!("{:>width$} {} {} {:.1}%",
            engine1_name.bright_cyan().bold(),
            engine1_bar,
            engine1_empty,
            self.engine1_win_rate(),
            width = name_width
        );

        // Draws
        if self.draws > 0 {
            let draw_bar_len = ((self.draws as f64 / total) * bar_width as f64) as usize;
            let draw_bar = "█".repeat(draw_bar_len).bright_blue();
            let draw_empty = "░".repeat(bar_width - draw_bar_len).bright_black();
            println!("{:>width$} {} {} {:.1}%",
                "Draws".bright_white(),
                draw_bar,
                draw_empty,
                draw_percentage,
                width = name_width
            );
        }

        // Engine 2
        println!("{:>width$} {} {} {:.1}%",
            engine2_name.bright_cyan().bold(),
            engine2_bar,
            engine2_empty,
            self.engine2_win_rate(),
            width = name_width
        );

        Ok(())
    }

    fn print_detailed_stats(&self, engine1_name: &str, engine2_name: &str) -> io::Result<()> {
        println!("{}", "Detailed Statistics".bright_white().underline());
        println!();

        let name_width = std::cmp::max(engine1_name.len(), engine2_name.len());

        // Header
        println!("{}", format!(
            "┌─{:─<width$}─┬─{:─^7}─┬─{:─^7}─┬─{:─^7}─┬─{:─^10}─┬─{:─^7}─┐",
            "", "", "", "", "", "",
            width = name_width
        ).bright_black());

        println!(
            "│ {:^width$} │ {:^7} │ {:^7} │ {:^7} │ {:^10} │ {:^7} │",
            "Engine".bright_white(),
            "Wins".bright_white(),
            "Losses".bright_white(),
            "Draws".bright_white(),
            "Win Rate".bright_white(),
            "Score".bright_white(),
            width = name_width
        );

        println!("{}", format!(
            "├─{:─<width$}─┼─{:─^7}─┼─{:─^7}─┼─{:─^7}─┼─{:─^10}─┼─{:─^7}─┤",
            "", "", "", "", "", "",
            width = name_width
        ).bright_black());

        // Engine 1 row
        let engine1_score = match self.total_score.cmp(&0) {
            std::cmp::Ordering::Greater => format!("{:+}", self.total_score).bright_green(),
            std::cmp::Ordering::Less => format!("{:+}", self.total_score).bright_red(),
            std::cmp::Ordering::Equal => format!("{:+}", self.total_score).bright_yellow(),
        };

        println!(
            "│ {:width$} │ {:^7} │ {:^7} │ {:^7} │ {:^10} │ {:>7} │",
            engine1_name.bright_cyan().bold(),
            self.engine1_wins.to_string().bright_green(),
            self.engine2_wins.to_string().bright_red(),
            self.draws.to_string().bright_blue(),
            format!("{:.1}%", self.engine1_win_rate()).bright_yellow(),
            engine1_score,
            width = name_width
        );

        // Engine 2 row
        let engine2_score = match (-self.total_score).cmp(&0) {
            std::cmp::Ordering::Greater => format!("{:+}", -self.total_score).bright_green(),
            std::cmp::Ordering::Less => format!("{:+}", -self.total_score).bright_red(),
            std::cmp::Ordering::Equal => format!("{:+}", -self.total_score).bright_yellow(),
        };

        println!(
            "│ {:width$} │ {:^7} │ {:^7} │ {:^7} │ {:^10} │ {:>7} │",
            engine2_name.bright_cyan().bold(),
            self.engine2_wins.to_string().bright_green(),
            self.engine1_wins.to_string().bright_red(),
            self.draws.to_string().bright_blue(),
            format!("{:.1}%", self.engine2_win_rate()).bright_yellow(),
            engine2_score,
            width = name_width
        );

        println!("{}", format!(
            "└─{:─<width$}─┴─{:─^7}─┴─{:─^7}─┴─{:─^7}─┴─{:─^10}─┴─{:─^7}─┘",
            "", "", "", "", "", "",
            width = name_width
        ).bright_black());

        Ok(())
    }

    fn print_elo_rating(&self, engine1_name: &str, engine2_name: &str, total_games: u32) -> io::Result<()> {
        let elo_stats = EloCalculator::calculate_stats(
            self.engine1_wins,
            self.engine2_wins,
            self.draws,
            total_games,
        );

        println!("{}", "ELO Rating".bright_white().underline());
        println!();


        // ELO difference with visual indicator
        let elo_display = if elo_stats.elo_diff.is_infinite() {
            if elo_stats.elo_diff > 0.0 {
                format!("{} {}", "∞".bright_green().bold(), "(Dominant performance)".bright_black())
            } else {
                format!("{} {}", "-∞".bright_red().bold(), "(Dominant performance)".bright_black())
            }
        } else {
            let elo_str = format!("{:+.2}", elo_stats.elo_diff);
            let confidence_str = format!("± {:.2} (95%)", elo_stats.confidence_interval);
            if elo_stats.elo_diff > 0.0 {
                format!("{} {}", elo_str.bright_green().bold(), confidence_str.bright_black())
            } else if elo_stats.elo_diff < 0.0 {
                format!("{} {}", elo_str.bright_red().bold(), confidence_str.bright_black())
            } else {
                format!("{} {}", elo_str.bright_yellow().bold(), confidence_str.bright_black())
            }
        };

        println!("{:>20}: {}", "ELO Difference".bright_white(), elo_display);
        
        // Win rate from ELO difference
        if !elo_stats.elo_diff.is_infinite() {
            let win_rate = 1.0 / (1.0 + 10.0_f64.powf(-elo_stats.elo_diff / 400.0));
            let win_rate_display = format!("{:.1}%", win_rate * 100.0);
            println!("{:>20}: {}", "Expected Win Rate".bright_white(), 
                if elo_stats.elo_diff > 0.0 {
                    win_rate_display.bright_green()
                } else if elo_stats.elo_diff < 0.0 {
                    win_rate_display.bright_red()
                } else {
                    win_rate_display.bright_yellow()
                }
            );
        }

        // Performance indicator
        if elo_stats.elo_diff.abs() > 100.0 && !elo_stats.elo_diff.is_infinite() {
            println!("{:>20}: {}", "Performance".bright_white(),
                if elo_stats.elo_diff > 0.0 {
                    format!("{engine1_name} is significantly stronger").bright_green()
                } else {
                    format!("{engine2_name} is significantly stronger").bright_red()
                }
            );
        } else if elo_stats.elo_diff.abs() > 50.0 {
            println!("{:>20}: {}", "Performance".bright_white(),
                if elo_stats.elo_diff > 0.0 {
                    format!("{engine1_name} has a clear advantage").bright_green()
                } else {
                    format!("{engine2_name} has a clear advantage").bright_red()
                }
            );
        } else if elo_stats.elo_diff.abs() > 20.0 {
            println!("{:>20}: {}", "Performance".bright_white(),
                if elo_stats.elo_diff > 0.0 {
                    format!("{engine1_name} has a slight edge").bright_yellow()
                } else {
                    format!("{engine2_name} has a slight edge").bright_yellow()
                }
            );
        } else {
            println!("{:>20}: {}", "Performance".bright_white(),
                "Engines are evenly matched".bright_blue()
            );
        }

        Ok(())
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
