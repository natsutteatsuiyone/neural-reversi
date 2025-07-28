//! Display management for match runner UI.
//!
//! This module handles all terminal-based user interface elements including
//! progress visualization, real-time match statistics, and formatted output.

use std::io::{self, Write};
use colored::*;
use reversi_core::piece::Piece;
use crate::statistics::{MatchStatistics, MatchWinner};

/// Display constants
const DEFAULT_BAR_WIDTH: usize = 60;
const HEADER_RESERVED_LINES: usize = 16;
const MIN_NAME_WIDTH: usize = 7;
const MAX_OPENING_DISPLAY_LEN: usize = 16;
const VISUALIZATION_START_LINE: &str = "\x1B[3;1H";
const CLEAR_LINE: &str = "\x1B[2K";
const SAVE_CURSOR: &str = "\x1B[s";
const RESTORE_CURSOR: &str = "\x1B[u";
const CLEAR_SCREEN: &str = "\x1B[2J\x1B[H";

/// Color variants for progress bars.
#[derive(Clone, Copy, Debug)]
enum BarColor {
    Green,
    Red,
    Blue,
}

/// Unified data structure for rendering bars.
#[derive(Debug)]
struct BarData {
    label: String,
    count: u32,
    percentage: f64,
    color: BarColor,
}

/// Layout configuration for display elements.
#[derive(Debug)]
struct DisplayLayout {
    bar_width: usize,
    name_width: usize,
    padding: String,
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
    pub fn new() -> Self {
        Self
    }

    /// Clear the terminal screen and move cursor to top-left.
    pub fn clear_screen(&self) -> io::Result<()> {
        print!("{CLEAR_SCREEN}");
        io::stdout().flush()
    }

    /// Display the match header and reserve space for live visualization.
    pub fn show_match_header(&self) -> io::Result<()> {
        self.clear_screen()?;
        
        // Reserve space for visualization to avoid overlap
        for _ in 0..HEADER_RESERVED_LINES {
            println!();
        }
        
        Ok(())
    }

    /// Update the live match visualization without scrolling.
    pub fn update_live_visualization(
        &self,
        statistics: &MatchStatistics,
        engine1_name: &str,
        engine2_name: &str,
    ) -> io::Result<()> {
        self.save_cursor_and_move_to_display()?;
        
        let layout = self.create_display_layout();
        
        self.display_header(engine1_name, engine2_name, &layout)?;
        self.display_statistics_bars(statistics, &layout)?;
        self.display_footer(statistics, &layout)?;
        
        self.restore_cursor_and_flush()?;
        
        Ok(())
    }

    // Helper methods for terminal operations
    fn save_cursor_and_move_to_display(&self) -> io::Result<()> {
        print!("{SAVE_CURSOR}{VISUALIZATION_START_LINE}");
        Ok(())
    }
    
    fn restore_cursor_and_flush(&self) -> io::Result<()> {
        print!("{RESTORE_CURSOR}");
        io::stdout().flush()
    }
    
    // Layout and structure methods
    fn create_display_layout(&self) -> DisplayLayout {
        DisplayLayout {
            bar_width: DEFAULT_BAR_WIDTH,
            name_width: self.calculate_name_width(),
            padding: "  ".to_string(),
        }
    }
    
    fn calculate_name_width(&self) -> usize {
        let required_width = ["Engine1", "Engine2", "Draws", "Disc Diff"]
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        required_width.max(MIN_NAME_WIDTH)
    }
    
    // Display section methods
    fn display_header(&self, engine1_name: &str, engine2_name: &str, layout: &DisplayLayout) -> io::Result<()> {
        println!("{}{} {} vs {}",
            CLEAR_LINE,
            layout.padding,
            engine1_name.bright_cyan().bold(),
            engine2_name.bright_cyan().bold()
        );
        
        let separator = "─".repeat(layout.bar_width + layout.name_width + 15);
        println!("{}{}{}", 
            CLEAR_LINE,
            layout.padding,
            separator.bright_black()
        );
        
        Ok(())
    }
    
    fn display_statistics_bars(&self, statistics: &MatchStatistics, layout: &DisplayLayout) -> io::Result<()> {
        let bars = self.create_bar_data(statistics);
        
        for bar in &bars {
            self.display_bar(bar, layout);
        }
        
        Ok(())
    }
    
    fn create_bar_data(&self, statistics: &MatchStatistics) -> Vec<BarData> {
        let total = statistics.total_games();
        
        vec![
            BarData {
                label: "Engine1".to_string(),
                count: statistics.engine1_wins,
                percentage: statistics.engine1_win_rate(),
                color: BarColor::Green,
            },
            BarData {
                label: "Draws".to_string(),
                count: statistics.draws,
                percentage: if total == 0 { 0.0 } else { (statistics.draws as f64 / total as f64) * 100.0 },
                color: BarColor::Blue,
            },
            BarData {
                label: "Engine2".to_string(),
                count: statistics.engine2_wins,
                percentage: statistics.engine2_win_rate(),
                color: BarColor::Red,
            },
        ]
    }
    
    fn display_bar(&self, bar_data: &BarData, layout: &DisplayLayout) {
        let bar_len = ((bar_data.percentage / 100.0) * layout.bar_width as f64) as usize;
        let filled_bar = "█".repeat(bar_len);
        let empty_bar = "░".repeat(layout.bar_width - bar_len).bright_black();
        
        let colored_bar = match bar_data.color {
            BarColor::Green => filled_bar.bright_green(),
            BarColor::Red => filled_bar.bright_red(),
            BarColor::Blue => filled_bar.bright_blue(),
        };
        
        println!("{}{}{:>width$} {}{} {:.1}% ({})",
            CLEAR_LINE,
            layout.padding,
            bar_data.label.bright_white(),
            colored_bar,
            empty_bar,
            bar_data.percentage,
            bar_data.count.to_string().bright_white(),
            width = layout.name_width
        );
    }
    
    fn display_footer(&self, statistics: &MatchStatistics, layout: &DisplayLayout) -> io::Result<()> {
        let separator = "─".repeat(layout.bar_width + layout.name_width + 15);
        println!("{}{}{}", 
            CLEAR_LINE,
            layout.padding,
            separator.bright_black()
        );
        
        self.display_score_summary(statistics, layout);
        self.display_recent_games(statistics, layout);
        
        Ok(())
    }
    
    fn display_score_summary(&self, statistics: &MatchStatistics, layout: &DisplayLayout) {
        println!("{}{}{:>width$}: {}",
            CLEAR_LINE,
            layout.padding,
            "Disc Diff".bright_white(),
            format!("{:+}", statistics.total_score).bright_cyan(),
            width = layout.name_width
        );
    }
    
    fn display_recent_games(&self, statistics: &MatchStatistics, layout: &DisplayLayout) {
        if statistics.recent_results.is_empty() {
            return;
        }
        
        println!("{}{}", CLEAR_LINE, layout.padding);
        
        let start_game_num = statistics.games_played
            .saturating_sub(statistics.recent_results.len() as u32 - 1);
        
        for (idx, game) in statistics.recent_results.iter().enumerate() {
            let game_number = start_game_num + idx as u32;
            
            let result_symbol = self.format_result_symbol(game.winner);
            let score_colored = self.format_score(game.score);
            let opening_display = self.format_opening(&game.opening);
            let vs_display = self.format_vs_display(game.engine1_color);
            
            println!("{}{}  {:>5}: {} {} {} {}",
                CLEAR_LINE,
                layout.padding,
                game_number.to_string().bright_white(),
                result_symbol,
                score_colored,
                opening_display.bright_black(),
                vs_display.bright_black()
            );
        }
    }
    
    // Formatting helper methods
    fn format_result_symbol(&self, winner: MatchWinner) -> colored::ColoredString {
        match winner {
            MatchWinner::Engine1 => "W".bright_green().bold(),
            MatchWinner::Engine2 => "L".bright_red().bold(),
            MatchWinner::Draw => "D".bright_blue().bold(),
        }
    }
    
    fn format_score(&self, score: i32) -> colored::ColoredString {
        let score_str = format!("{score:+3}");
        match score.cmp(&0) {
            std::cmp::Ordering::Greater => score_str.bright_green(),
            std::cmp::Ordering::Less => score_str.bright_red(),
            std::cmp::Ordering::Equal => score_str.bright_blue(),
        }
    }
    
    fn format_opening(&self, opening: &str) -> String {
        if opening.len() > MAX_OPENING_DISPLAY_LEN {
            format!("{}...", &opening[..MAX_OPENING_DISPLAY_LEN])
        } else {
            opening.to_string()
        }
    }
    
    fn format_vs_display(&self, engine1_color: Piece) -> String {
        let (engine1_symbol, engine2_symbol) = match engine1_color {
            Piece::Black => ("●", "○"),
            Piece::White => ("○", "●"),
            Piece::Empty => ("?", "?"), // This shouldn't happen in normal game flow
        };
        
        format!("{engine1_symbol} Engine1 vs Engine2 {engine2_symbol}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_name_width() {
        let display = DisplayManager::new();
        let width = display.calculate_name_width();
        assert!(width >= MIN_NAME_WIDTH);
    }
    
    #[test]
    fn test_format_opening() {
        let display = DisplayManager::new();
        
        let short_opening = "e4";
        assert_eq!(display.format_opening(short_opening), "e4");
        
        let long_opening = "this_is_a_very_long_opening_name_that_exceeds_limit";
        let formatted = display.format_opening(long_opening);
        assert!(formatted.len() <= MAX_OPENING_DISPLAY_LEN + 3); // +3 for "..."
        assert!(formatted.ends_with("..."));
    }
    
    #[test]
    fn test_format_score() {
        let display = DisplayManager::new();
        
        // Positive score should be green
        let positive = display.format_score(5);
        assert!(positive.to_string().contains("+5"));
        
        // Negative score should be red  
        let negative = display.format_score(-3);
        assert!(negative.to_string().contains("-3"));
        
        // Zero score should be blue
        let zero = display.format_score(0);
        assert!(zero.to_string().contains("+0"));
    }
}