//! Display management for automatch UI.
//!
//! This module handles all terminal-based user interface elements including
//! progress visualization, real-time match statistics, and formatted output.

use std::io::{self, Write};
use colored::*;
use crate::statistics::MatchStatistics;

/// Color variants for progress bars.
#[derive(Clone, Copy)]
enum BarColor {
    Green,
    Red,
    #[allow(dead_code)]
    Blue,
}

/// Data structure for rendering engine performance bars.
struct EngineBarData<'a> {
    name: &'a str,
    wins: f64,
    total: f64,
    bar_width: usize,
    name_width: usize,
    win_rate: f64,
    win_count: u32,
    color: BarColor,
}

/// Manages terminal display and user interface for match visualization.
///
/// The DisplayManager handles all aspects of the terminal-based user interface,
/// including real-time progress bars, match statistics visualization, and
/// formatted output display.
pub struct DisplayManager;

impl Default for DisplayManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DisplayManager {
    /// Create a new DisplayManager instance.
    ///
    /// # Returns
    ///
    /// A new DisplayManager ready for use.
    pub fn new() -> Self {
        Self
    }

    /// Clear the terminal screen and move cursor to top-left.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an I/O error if the terminal operation fails.
    pub fn clear_screen(&self) -> io::Result<()> {
        print!("\x1B[2J\x1B[H");
        io::stdout().flush()
    }

    /// Display the match header and reserve space for live visualization.
    ///
    /// Clears the screen and shows a formatted header indicating that a match
    /// is in progress. Also reserves vertical space for the live statistics
    /// visualization.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an I/O error if the terminal operation fails.
    pub fn show_match_header(&self) -> io::Result<()> {
        self.clear_screen()?;
        
        // Reserve space for visualization (12 lines to avoid overlap with progress bar)
        for _ in 0..12 {
            println!();
        }

        Ok(())
    }

    /// Update the live match visualization without scrolling.
    ///
    /// Updates the on-screen statistics display in-place, showing current
    /// win rates, scores, and progress bars for both engines. Uses terminal
    /// cursor positioning to avoid scrolling.
    ///
    /// # Arguments
    ///
    /// * `statistics` - Current match statistics
    /// * `engine1_name` - Name of the first engine
    /// * `engine2_name` - Name of the second engine
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an I/O error if the terminal operation fails.
    pub fn update_live_visualization(
        &self,
        statistics: &MatchStatistics,
        engine1_name: &str,
        engine2_name: &str,
    ) -> io::Result<()> {
        // Save cursor position
        print!("\x1B[s");

        // Move to visualization area (line 3 for statistics display)
        print!("\x1B[3;1H");

        let total = statistics.total_games() as f64;
        if total == 0.0 {
            print!("\x1B[u");
            return Ok(());
        }

        let bar_width = 60;  // Wider bars for better visual
        let name_width = self.calculate_name_width(engine1_name, engine2_name);

        // Display section header
        println!("\x1B[2K  {} vs {}",
            engine1_name.bright_cyan().bold(),
            engine2_name.bright_cyan().bold()
        );
        println!("\x1B[2K{}", "  ".to_string() + &"─".repeat(bar_width + name_width + 15).bright_black());

        let engine1_bar_data = EngineBarData {
            name: engine1_name,
            wins: statistics.engine1_wins as f64,
            total,
            bar_width,
            name_width,
            win_rate: statistics.engine1_win_rate(),
            win_count: statistics.engine1_wins,
            color: BarColor::Green,
        };
        self.display_engine_bar(&engine1_bar_data);

        if statistics.draws > 0 {
            self.display_draw_bar(statistics, total, bar_width, name_width);
        }

        let engine2_bar_data = EngineBarData {
            name: engine2_name,
            wins: statistics.engine2_wins as f64,
            total,
            bar_width,
            name_width,
            win_rate: statistics.engine2_win_rate(),
            win_count: statistics.engine2_wins,
            color: BarColor::Red,
        };
        self.display_engine_bar(&engine2_bar_data);

        println!("\x1B[2K{}", "  ".to_string() + &"─".repeat(bar_width + name_width + 15).bright_black());
        self.display_score_summary(statistics, name_width, bar_width);

        // Restore cursor position
        print!("\x1B[u");
        io::stdout().flush()?;

        Ok(())
    }

    fn calculate_name_width(&self, engine1_name: &str, engine2_name: &str) -> usize {
        let max_name_len = std::cmp::max(
            std::cmp::max(engine1_name.len(), engine2_name.len()),
            "Draws".len()
        );
        std::cmp::max(max_name_len, 12)
    }

    fn display_engine_bar(&self, bar_data: &EngineBarData) {
        let bar_len = ((bar_data.wins / bar_data.total) * bar_data.bar_width as f64) as usize;
        let bar = "█".repeat(bar_len);
        let empty = "░".repeat(bar_data.bar_width - bar_len).bright_black();

        let colored_bar = match bar_data.color {
            BarColor::Green => bar.bright_green(),
            BarColor::Red => bar.bright_red(),
            BarColor::Blue => bar.bright_blue(),
        };

        let padding = "  ";
        println!("\x1B[2K{}{:>width$} {} {} {:.1}% ({})",
            padding,
            bar_data.name.bright_cyan().bold(),
            colored_bar,
            empty,
            bar_data.win_rate,
            bar_data.win_count.to_string().bright_white(),
            width = bar_data.name_width
        );
    }

    fn display_draw_bar(
        &self,
        statistics: &MatchStatistics,
        total: f64,
        bar_width: usize,
        name_width: usize,
    ) {
        let draws = statistics.draws as f64;
        let draw_bar_len = ((draws / total) * bar_width as f64) as usize;
        let draw_bar = "█".repeat(draw_bar_len).bright_blue();
        let draw_empty = "░".repeat(bar_width - draw_bar_len).bright_black();
        let draw_percentage = (draws / total) * 100.0;

        let padding = "  ";
        println!("\x1B[2K{}{:>width$} {} {} {:.1}% ({})",
            padding,
            "Draws".bright_white(),
            draw_bar,
            draw_empty,
            draw_percentage,
            statistics.draws.to_string().bright_white(),
            width = name_width
        );
    }

    fn display_score_summary(&self, statistics: &MatchStatistics, name_width: usize, _bar_width: usize) {
        let score_color = match statistics.total_score.cmp(&0) {
            std::cmp::Ordering::Greater => format!("{:+}", statistics.total_score).bright_green(),
            std::cmp::Ordering::Less => format!("{:+}", statistics.total_score).bright_red(),
            std::cmp::Ordering::Equal => format!("{:+}", statistics.total_score).bright_yellow(),
        };

        let padding = "  ";
        println!("\x1B[2K{}{:>width$}: {}",
            padding,
            "Total Score".bright_white(),
            score_color,
            width = name_width
        );
    }
}
