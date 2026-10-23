//! What a finished search reports: the summary shown with its game and in
//! the transcript, and the text told to the spectator channel.

use std::fmt;
use std::time::Duration;

use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::search::result::SearchResult;
use reversi_core::square::Square;
use reversi_core::types::{Depth, Scoref};

/// Statistics of a finished search that chose a move.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchReport {
    pub sq: Square,
    /// Disc difference from our side.
    pub score: Scoref,
    pub depth: Depth,
    /// Selectivity probability in percent.
    pub probability: i32,
    pub nodes: u64,
    /// Search time, without the wait for the shared thread pool.
    pub elapsed: Duration,
    /// Time spent waiting for the shared thread pool.
    pub waited: Duration,
}

impl SearchReport {
    /// The move and its rounded score, e.g. `f5 +3`.
    pub fn best(&self) -> String {
        format!("{} {:+}", self.sq, self.score.round() as i32)
    }

    /// Depth, node count, time, and speed, e.g. `depth 22@73%`,
    /// `12.3M nodes`, `1.84s (wait 0.20s)`, and `6.7 Mnps`.
    pub fn stats(&self) -> [String; 4] {
        let secs = self.elapsed.as_secs_f64();
        let nodes = self.nodes as f64;
        let wait = if self.waited >= Duration::from_millis(50) {
            format!(" (wait {:.2}s)", self.waited.as_secs_f64())
        } else {
            String::new()
        };
        [
            format!("depth {}@{}%", self.depth, self.probability),
            format!("{} nodes", si_count(nodes)),
            format!("{secs:.2}s{wait}"),
            format!("{:.1} Mnps", nodes / secs.max(1e-9) / 1e6),
        ]
    }
}

/// One line, e.g. `f5 +3 | depth 22@73% | 12.3M nodes | 1.84s (wait 0.20s) | 6.7 Mnps`.
impl fmt::Display for SearchReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} | {}", self.best(), self.stats().join(" | "))
    }
}

/// Formats a count with a K/M/G suffix, e.g. `12.3M`.
fn si_count(n: f64) -> String {
    match n {
        n if n >= 1e9 => format!("{:.1}G", n / 1e9),
        n if n >= 1e6 => format!("{:.1}M", n / 1e6),
        n if n >= 1e3 => format!("{:.1}K", n / 1e3),
        n => format!("{n}"),
    }
}

/// Search summary told to the spectator channel.
/// GGS shows each `\` in a tell as a line break, so the channel prints e.g.
///
/// ```text
/// .nra nra: -----------------------------------------
/// |nra plays F2 in game .1.1 using 8 threads ;
/// |score == -2 at 38@87% ; PV: F2 e7 B6 c7 B3 c2 F8 a2  ;
/// |nodes:  2.149 GN ; time: search = 10.4s, move = 10.5s; speed:  205.7 MN/s.
/// |search out of time
/// ```
///
/// Black's moves are upper case and `pa` is a pass. `move` adds the wait for
/// the shared thread pool to `search`.
#[allow(clippy::too_many_arguments)]
pub(super) fn channel_report(
    login: &str,
    game_id: &str,
    board: &Board,
    color: Disc,
    result: &SearchResult,
    elapsed: Duration,
    waited: Duration,
    n_threads: usize,
    timed_out: bool,
) -> String {
    let mv = report_square(result.best_move().unwrap_or(Square::None), color);
    if result.depth() == 0 {
        // The first move of a game is picked at random, without a search.
        return format!(
            "-----------------------------------------\\\
             {login} plays {mv} in game {game_id}\\\
             random opening move"
        );
    }
    let pv = report_pv(board, color, result.pv_line());
    let score = result.score().unwrap_or_default().round() as i32;
    let depth = result.depth();
    let probability = result.probability();
    let nodes = result.n_nodes() as f64;
    let search = elapsed.as_secs_f64();
    let status = if timed_out {
        "search out of time"
    } else {
        "search completed"
    };
    format!(
        "-----------------------------------------\\\
         {login} plays {mv} in game {game_id} using {n_threads} threads ;\\\
         score == {score:+} at {depth}@{probability}% ; PV: {pv} ;\\\
         nodes: {}N ; time: search = {search:.1}s, move = {:.1}s; speed: {}N/s.\\\
         {status}",
        report_count(nodes),
        search + waited.as_secs_f64(),
        report_count(nodes / search.max(1e-9)),
    )
}

/// A report move: upper case for Black, `pa` for a pass.
fn report_square(sq: Square, color: Disc) -> String {
    let s = if sq == Square::None {
        "pa".to_string()
    } else {
        sq.to_string()
    };
    if color == Disc::Black {
        s.to_uppercase()
    } else {
        s
    }
}

/// The first 8 moves of `pv`, each followed by a space. PV lines omit forced
/// passes, so the line is replayed from `board` to put them back.
fn report_pv(board: &Board, color: Disc, pv: &[Square]) -> String {
    let mut line = String::new();
    let mut mover = color;
    let mut current = *board;
    let mut tokens = 0;
    for &sq in pv {
        if tokens >= 8 {
            break;
        }
        if !current.is_legal_move(sq) {
            line += &report_square(Square::None, mover);
            line.push(' ');
            current = current.switch_players();
            mover = mover.opposite();
            tokens += 1;
        }
        if tokens >= 8 || !current.is_legal_move(sq) {
            break;
        }
        line += &report_square(sq, mover);
        line.push(' ');
        current = current.make_move(sq);
        mover = mover.opposite();
        tokens += 1;
    }
    line
}

/// Formats a count with four significant digits and an SI prefix, e.g. ` 2.149 G` or ` 98.52 M` (right-aligned to six characters).
fn report_count(n: f64) -> String {
    let (value, prefix) = match n {
        n if n >= 1e9 => (n / 1e9, "G"),
        n if n >= 1e6 => (n / 1e6, "M"),
        n if n >= 1e3 => (n / 1e3, "K"),
        n => return format!("{n:6.0} "),
    };
    let decimals = match value {
        v if v < 10.0 => 3,
        v if v < 100.0 => 2,
        _ => 1,
    };
    format!("{value:6.decimals$} {prefix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_pv_restores_forced_passes() {
        // After Black's c1, White has no move and Black plays again.
        let board = Board::from_string(&format!("XO------XO------{}", "-".repeat(48)), Disc::Black)
            .unwrap();
        assert_eq!(
            report_pv(&board, Disc::Black, &[Square::C1, Square::C3]),
            "C1 pa C3 "
        );
    }

    #[test]
    fn report_count_keeps_four_significant_digits() {
        assert_eq!(report_count(2.149e9), " 2.149 G");
        assert_eq!(report_count(98.52e6), " 98.52 M");
        assert_eq!(report_count(753.8e6), " 753.8 M");
        assert_eq!(report_count(25.0), "    25 ");
    }
}
