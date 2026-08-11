//! Terminal display for live match statistics.

use std::io::{self, Write};

use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use reversi_core::disc::Disc;

use crate::colors::ThemeColor;
use crate::sprt::SprtResult;
use crate::statistics::{MatchStatistics, MatchWinner};

const BAR_WIDTH: usize = 60;
const NAME_WIDTH: usize = 9;
const HEADER_RESERVED_LINES: usize = 19;
const MAX_OPENING_DISPLAY_LEN: usize = 16;
const PADDING: &str = "  ";
const VISUALIZATION_START_LINE: &str = "\x1B[3;1H";
const CLEAR_LINE: &str = "\x1B[2K";
const SAVE_CURSOR: &str = "\x1B[s";
const RESTORE_CURSOR: &str = "\x1B[u";
const CLEAR_SCREEN: &str = "\x1B[2J\x1B[H";

#[derive(Clone, Copy)]
enum BarColor {
    Green,
    Red,
    Blue,
}

pub(crate) fn clear_screen() -> io::Result<()> {
    print!("{CLEAR_SCREEN}");
    io::stdout().flush()
}

pub(crate) fn show_match_header() -> io::Result<()> {
    clear_screen()?;
    for _ in 0..HEADER_RESERVED_LINES {
        println!();
    }
    Ok(())
}

pub(crate) fn create_progress_bar(total_games: u64) -> ProgressBar {
    let progress_bar = ProgressBar::new(total_games);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.cyan} [{bar:40.cyan}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
    );
    progress_bar
}

pub(crate) fn update_live_visualization(
    statistics: &MatchStatistics,
    engine1_name: &str,
    engine2_name: &str,
    sprt: Option<&SprtResult>,
) -> io::Result<()> {
    print!("{SAVE_CURSOR}{VISUALIZATION_START_LINE}");

    let separator = "─".repeat(BAR_WIDTH + NAME_WIDTH + 15);
    display_header(engine1_name, engine2_name, &separator);
    display_statistics_bars(statistics);
    display_footer(statistics, sprt, &separator);

    print!("{RESTORE_CURSOR}");
    io::stdout().flush()
}

fn display_header(engine1_name: &str, engine2_name: &str, separator: &str) {
    println!(
        "{}{} {} vs {}",
        CLEAR_LINE,
        PADDING,
        engine1_name.primary().bold(),
        engine2_name.primary().bold()
    );
    println!("{CLEAR_LINE}{PADDING}{}", separator.subtext());
}

fn display_statistics_bars(statistics: &MatchStatistics) {
    let total = statistics.total_games();
    display_bar(
        "Engine1",
        statistics.engine1_wins,
        percentage(statistics.engine1_wins, total),
        BarColor::Green,
    );
    display_bar(
        "Draws",
        statistics.draws,
        percentage(statistics.draws, total),
        BarColor::Blue,
    );
    display_bar(
        "Engine2",
        statistics.engine2_wins,
        percentage(statistics.engine2_wins, total),
        BarColor::Red,
    );
}

fn percentage(count: u32, total: u32) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64 * 100.0
    }
}

fn display_bar(label: &str, count: u32, percentage: f64, color: BarColor) {
    let bar_len = ((percentage / 100.0) * BAR_WIDTH as f64) as usize;
    let filled_bar = "█".repeat(bar_len);
    let empty_bar = " ".repeat(BAR_WIDTH - bar_len).bg_dark();
    let colored_bar = match color {
        BarColor::Green => filled_bar.success(),
        BarColor::Red => filled_bar.failure(),
        BarColor::Blue => filled_bar.info(),
    };

    let count = format!("{count:>4}").text();
    let percentage = format!("{percentage:>6.1}%").text();
    println!(
        "{}{}{:>NAME_WIDTH$} {}{} {} ({})",
        CLEAR_LINE,
        PADDING,
        label.text(),
        colored_bar,
        empty_bar,
        percentage,
        count,
    );
}

fn display_footer(statistics: &MatchStatistics, sprt: Option<&SprtResult>, separator: &str) {
    println!("{CLEAR_LINE}{PADDING}{}", separator.subtext());
    println!(
        "{}{}{:>NAME_WIDTH$}: {}",
        CLEAR_LINE,
        PADDING,
        "Disc Diff".text(),
        format!("{:+}", statistics.total_score).primary(),
    );

    if let Some(result) = sprt {
        println!(
            "{}{}{:>NAME_WIDTH$}: {} {}",
            CLEAR_LINE,
            PADDING,
            "SPRT LLR".text(),
            format!("{:+.2}", result.llr).primary(),
            format!("({:.2}, {:.2})", result.lower, result.upper).subtext(),
        );
    }

    println!("{CLEAR_LINE}{PADDING}");
    println!("{CLEAR_LINE}{PADDING}{}", "Recent Games:".subtext());
    println!("{CLEAR_LINE}{PADDING}{}", separator.subtext());
    display_recent_games(statistics);
}

fn display_recent_games(statistics: &MatchStatistics) {
    if statistics.recent_results.is_empty() {
        return;
    }

    println!("{CLEAR_LINE}{PADDING}");
    let start_game_num = statistics
        .total_games()
        .saturating_sub(statistics.recent_results.len() as u32 - 1);

    for (idx, game) in statistics.recent_results.iter().enumerate() {
        let game_number = start_game_num + idx as u32;
        println!(
            "{}{}  {:>5}: {} {} {} {}",
            CLEAR_LINE,
            PADDING,
            game_number.to_string().subtext(),
            format_result_symbol(game.winner),
            format_score(game.score, game.winner),
            format_opening(&game.opening).subtext(),
            format_vs_display(game.engine1_color).subtext(),
        );
    }
}

fn format_result_symbol(winner: MatchWinner) -> colored::ColoredString {
    match winner {
        MatchWinner::Engine1 => "W".success().bold(),
        MatchWinner::Engine2 => "L".failure().bold(),
        MatchWinner::Draw => "D".info().bold(),
    }
}

fn format_score(score: i32, winner: MatchWinner) -> colored::ColoredString {
    let score = format!("{score:+3}");
    match winner {
        MatchWinner::Engine1 => score.success(),
        MatchWinner::Engine2 => score.failure(),
        MatchWinner::Draw => score.info(),
    }
}

fn format_opening(opening: &str) -> String {
    if opening.len() > MAX_OPENING_DISPLAY_LEN {
        format!("{}...", &opening[..MAX_OPENING_DISPLAY_LEN])
    } else {
        opening.to_string()
    }
}

fn format_vs_display(engine1_color: Disc) -> String {
    let (engine1_symbol, engine2_symbol) = match engine1_color {
        Disc::Black => ("●", "○"),
        Disc::White => ("○", "●"),
        Disc::Empty => ("?", "?"),
    };
    format!("{engine1_symbol} Engine1 vs Engine2 {engine2_symbol}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_display_is_truncated() {
        assert_eq!(format_opening("e4"), "e4");
        assert_eq!(format_opening("abcdefghijklmnopq"), "abcdefghijklmnop...");
    }
}
