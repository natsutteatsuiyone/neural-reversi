use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use num_format::{Locale, ToFormattedString};
use reversi_core::{
    board::Board,
    engine::{Engine, GameId},
    level::Level,
    obf::ObfPosition,
    player_color::PlayerColor,
    score::Score,
    search::SearchRunOptions,
    square::Square,
};

use crate::config::EngineConfig;
use crate::text::visible;

const NUM_WIDTH: usize = 5;
const DEPTH_WIDTH: usize = 6;
const SCORE_WIDTH: usize = 5;
const TIME_WIDTH: usize = 11;
const NODES_WIDTH: usize = 19;
const NPS_WIDTH: usize = 13;
const PV_WIDTH: usize = 23;

pub fn solve(
    file_path: &Path,
    config: &EngineConfig,
    exact: bool,
    all_moves: bool,
) -> Result<(), String> {
    let file =
        File::open(file_path).map_err(|e| format!("cannot open {}: {e}", file_path.display()))?;
    let reader = BufReader::new(file);

    let engine = config.engine()?;

    print_header(file_path, config.hash_size, engine.n_threads());

    let game = engine.open();
    let level_config = if exact {
        Level::perfect()
    } else {
        Level::preset(config.level)
    };

    if !all_moves {
        print_table_header();
    }

    let mut total_time = Duration::ZERO;
    let mut total_nodes: u64 = 0;
    let mut n_positions = 0usize;
    let mut n_invalid_lines = 0usize;

    for (line_num, line) in reader.lines().enumerate() {
        let raw = line.map_err(|e| format!("cannot read line {}: {e}", line_num + 1))?;
        let pos = match ObfPosition::parse(&raw) {
            Ok(Some(pos)) => pos,
            Ok(None) => continue,
            Err(e) => {
                eprintln!(
                    "Error parsing line {}: {}",
                    line_num + 1,
                    visible(&e.to_string())
                );
                n_invalid_lines += 1;
                continue;
            }
        };
        // The search assumes at most 60 empties.
        if pos.board.player_count() + pos.board.opponent_count() < 4 {
            eprintln!(
                "Error parsing line {}: At least 4 discs required",
                line_num + 1
            );
            n_invalid_lines += 1;
            continue;
        }
        n_positions += 1;
        let (elapsed, nodes) = solve_position(
            &engine,
            game,
            pos.board,
            pos.side_to_move,
            level_config,
            line_num + 1,
            all_moves,
        );
        total_time += elapsed;
        total_nodes += nodes;
        if all_moves {
            print_position_stats(elapsed, nodes);
            println!();
        }
    }

    let total_secs = total_time.as_secs_f64();
    let total_nps = if total_secs > 0.0 {
        total_nodes as f64 / total_secs
    } else {
        0.0
    };
    if all_moves {
        println!(
            "Total: time {}, nodes {}, n/s {}",
            format_time(total_time),
            total_nodes.to_formatted_string(&Locale::en),
            (total_nps.round() as u64).to_formatted_string(&Locale::en)
        );
    } else {
        print_row(
            "Total",
            "",
            "",
            format_time(total_time),
            total_nodes.to_formatted_string(&Locale::en),
            (total_nps.round() as u64).to_formatted_string(&Locale::en),
            "",
        );
    }
    println!();

    if n_invalid_lines > 0 {
        return Err(format!("skipped {n_invalid_lines} invalid line(s)"));
    }
    if n_positions == 0 {
        return Err(format!("no positions in {}", file_path.display()));
    }
    Ok(())
}

fn print_table_header() {
    println!(
        "| {:^NUM_WIDTH$} | {:^DEPTH_WIDTH$} | {:^SCORE_WIDTH$} | {:^TIME_WIDTH$} | {:^NODES_WIDTH$} | {:^NPS_WIDTH$} | {:^PV_WIDTH$} |",
        "#", "Depth", "Score", "Time", "Nodes", "N/s", "Principal Variation"
    );
    println!(
        "| {} | {} | {} | {} | {} | {} | {} |",
        markdown_align_right(NUM_WIDTH),
        markdown_align_center(DEPTH_WIDTH),
        markdown_align_center(SCORE_WIDTH),
        markdown_align_right(TIME_WIDTH),
        markdown_align_right(NODES_WIDTH),
        markdown_align_right(NPS_WIDTH),
        markdown_align_left(PV_WIDTH),
    );
}

fn print_all_moves_table_header() {
    println!(
        "| {:^SCORE_WIDTH$} | {:^PV_WIDTH$} |",
        "Score", "Principal Variation"
    );
    println!(
        "| {} | {} |",
        markdown_align_center(SCORE_WIDTH),
        markdown_align_left(PV_WIDTH)
    );
}

fn print_position_stats(elapsed: Duration, nodes: u64) {
    let nps = if elapsed.as_secs_f64() > 0.0 {
        nodes as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    println!(
        "Time: {}  Nodes: {}  N/s: {}",
        format_time(elapsed),
        nodes.to_formatted_string(&Locale::en),
        (nps.round() as u64).to_formatted_string(&Locale::en)
    );
}

fn print_header(file_path: &Path, tt_mb_size: usize, n_threads: usize) {
    let file_name = file_path
        .file_name()
        .unwrap_or(file_path.as_os_str())
        .to_string_lossy();
    println!(
        "Neural Reversi v{} ({})",
        env!("CARGO_PKG_VERSION"),
        env!("TARGET")
    );
    println!();
    println!("- File:      {file_name}");
    println!("- Hash size: {tt_mb_size} MB");
    println!("- Threads:   {n_threads}");
    println!();
}

fn solve_position(
    engine: &Engine,
    game: GameId,
    board: Board,
    side_to_move: PlayerColor,
    level: Level,
    position_num: usize,
    all_moves: bool,
) -> (Duration, u64) {
    let is_pass = !board.has_legal_moves();

    if is_pass && !board.switch_players().has_legal_moves() {
        let score = board.solve_disc_diff(board.empty_count());
        if all_moves {
            println!("Position #{}  Depth: END", position_num);
            print_all_moves_table_header();
            print_all_moves_row(format!("{:+03}", score), "--");
            return (Duration::ZERO, 0);
        }
        print_row(
            position_num,
            "END",
            format!("{:+03}", score),
            format_time(Duration::ZERO),
            "0",
            "0",
            "--",
        );
        return (Duration::ZERO, 0);
    }
    let search_board = if is_pass {
        board.switch_players()
    } else {
        board
    };

    engine.reset(game);
    let start_time = Instant::now();
    let options = SearchRunOptions::with_level(level).multi_pv(all_moves);
    let result = engine.search(game, &search_board, &options);
    let elapsed = start_time.elapsed();

    let depth = if result.probability() == 100 {
        format!("{}", result.depth())
    } else {
        format!("{}@{}", result.depth(), result.selectivity())
    };

    let nodes_per_sec = if elapsed.as_secs_f64() > 0.0 {
        result.n_nodes() as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let move_side = if is_pass {
        side_to_move.opposite()
    } else {
        side_to_move
    };

    if all_moves && !result.pv_moves().is_empty() {
        println!("Position #{}  Depth: {}", position_num, depth);
        print_all_moves_table_header();
        for pv_move in result.pv_moves() {
            let score = disc_score(pv_move.score, is_pass);
            let pv_string = if pv_move.pv_line.is_empty() {
                format_root_move(pv_move.sq, move_side, is_pass, side_to_move)
            } else {
                format_pv_with_passes(&board, side_to_move, &pv_move.pv_line, 8)
            };

            print_all_moves_row(format!("{:+03}", score), pv_string);
        }

        return (elapsed, result.n_nodes());
    }
    let result_score = result.score().expect("search returned no legal move");
    let score = disc_score(result_score, is_pass);
    let pv_string = if result.pv_line().is_empty() {
        result
            .best_move()
            .map_or("--".to_string(), |m| format_square(m, move_side))
    } else {
        format_pv_with_passes(&board, side_to_move, result.pv_line(), 8)
    };

    print_row(
        position_num,
        depth,
        format!("{:+03}", score),
        format_time(elapsed),
        result.n_nodes().to_formatted_string(&Locale::en),
        (nodes_per_sec.round() as u64).to_formatted_string(&Locale::en),
        pv_string,
    );

    (elapsed, result.n_nodes())
}

/// Rounds `score` to whole discs for the side to move in the position file.
///
/// `searched_after_pass` means that side had to pass, so `score` is from the
/// opponent's point of view.
fn disc_score(score: Score, searched_after_pass: bool) -> i32 {
    let discs = score.to_disc_diff_f32().round() as i32;
    if searched_after_pass { -discs } else { discs }
}

fn print_all_moves_row(score: impl Display, pv: impl Display) {
    println!("| {score:^SCORE_WIDTH$} | {pv:<PV_WIDTH$} |");
}

fn print_row(
    num: impl Display,
    depth: impl Display,
    score: impl Display,
    time: impl Display,
    nodes: impl Display,
    nps: impl Display,
    pv: impl Display,
) {
    println!(
        "| {num:>NUM_WIDTH$} | {depth:^DEPTH_WIDTH$} | {score:^SCORE_WIDTH$} | {time:>TIME_WIDTH$} | {nodes:>NODES_WIDTH$} | {nps:>NPS_WIDTH$} | {pv:<PV_WIDTH$} |"
    );
}

fn markdown_align_right(width: usize) -> String {
    format!("{:->width$}:", "", width = width - 1)
}

fn markdown_align_center(width: usize) -> String {
    format!(":{:-<width$}:", "", width = width - 2)
}

fn markdown_align_left(width: usize) -> String {
    format!(":{:-<width$}", "", width = width - 1)
}

fn format_pv_with_passes(
    board: &Board,
    side_to_move: PlayerColor,
    pv_line: &[Square],
    max_tokens: usize,
) -> String {
    let push_token = |s: &mut String, t: &str| {
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(t);
    };

    let mut result = String::new();
    let mut current = *board;
    let mut side = side_to_move;
    let mut count = 0;

    for &sq in pv_line {
        if count >= max_tokens {
            break;
        }
        if !current.has_legal_moves() {
            push_token(&mut result, format_pass(side));
            current = current.switch_players();
            side = side.opposite();
            count += 1;
            if count >= max_tokens {
                break;
            }
        }
        if !current.is_legal_move(sq) {
            break;
        }
        push_token(&mut result, &format_square(sq, side));
        current = current.make_move(sq);
        side = side.opposite();
        count += 1;
    }

    result
}

fn format_root_move(
    sq: Square,
    move_side: PlayerColor,
    is_pass: bool,
    side_to_move: PlayerColor,
) -> String {
    if !is_pass {
        return format_square(sq, move_side);
    }

    let mut result = String::from(format_pass(side_to_move));
    result.push(' ');
    result.push_str(&format_square(sq, move_side));
    result
}

fn format_time(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    // Whole milliseconds, so rounding cannot print 60.000 seconds.
    let frac_secs = (duration.as_millis() % 60_000) as f64 / 1000.0;
    if total_secs >= 3600 {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        format!("{hours}:{mins:02}:{frac_secs:06.3}")
    } else {
        let mins = total_secs / 60;
        format!("{mins}:{frac_secs:06.3}")
    }
}

fn format_square(sq: Square, side: PlayerColor) -> String {
    let s = sq.to_string();
    if side == PlayerColor::White {
        s.to_uppercase()
    } else {
        s
    }
}

fn format_pass(side: PlayerColor) -> &'static str {
    if side == PlayerColor::White {
        "PS"
    } else {
        "ps"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Black owns C1-H1 and C8-H8 and White owns B1 and B8, so only Black can
    /// move (A1 or A8), before and after either move.
    fn only_black_can_move(side_to_move: PlayerColor) -> Board {
        let black = 0xFC00_0000_0000_00FCu64;
        let white = Square::B1.bitboard() | Square::B8.bitboard();
        match side_to_move {
            PlayerColor::Black => Board::from_bitboards(black, white),
            PlayerColor::White => Board::from_bitboards(white, black),
        }
    }

    #[test]
    fn format_time_keeps_seconds_below_60() {
        let cases = [
            (Duration::ZERO, "0:00.000"),
            (Duration::from_micros(59_999_500), "0:59.999"),
            (Duration::from_secs(60), "1:00.000"),
            (Duration::from_micros(3_599_999_500), "59:59.999"),
            (Duration::from_secs(3600), "1:00:00.000"),
            (Duration::from_millis(3_661_001), "1:01:01.001"),
        ];
        for (duration, expected) in cases {
            assert_eq!(format_time(duration), expected, "{duration:?}");
        }
    }

    #[test]
    fn disc_score_rounds_to_nearest_and_negates_after_a_pass() {
        let cases = [
            (383, 1, -1),
            (384, 2, -2),
            (-383, -1, 1),
            (-384, -2, 2),
            (Score::MAX.value(), 64, -64),
            (Score::MIN.value(), -64, 64),
        ];
        for (raw, expected, expected_after_pass) in cases {
            let score = Score::from_raw(raw);
            assert_eq!(disc_score(score, false), expected, "raw {raw}");
            assert_eq!(disc_score(score, true), expected_after_pass, "raw {raw}");
        }
    }

    #[test]
    fn format_pv_inserts_a_pass_in_the_middle_of_the_line() {
        let board = only_black_can_move(PlayerColor::Black);
        let pv = [Square::A1, Square::A8];
        assert_eq!(
            format_pv_with_passes(&board, PlayerColor::Black, &pv, 8),
            "a1 PS a8"
        );
    }

    #[test]
    fn format_pv_starts_with_a_pass_when_the_side_to_move_has_no_move() {
        let board = only_black_can_move(PlayerColor::White);
        let pv = [Square::A1, Square::A8];
        assert_eq!(
            format_pv_with_passes(&board, PlayerColor::White, &pv, 8),
            "PS a1 PS a8"
        );
    }

    #[test]
    fn format_pv_counts_passes_toward_max_tokens() {
        let board = only_black_can_move(PlayerColor::White);
        let pv = [Square::A1, Square::A8];
        assert_eq!(
            format_pv_with_passes(&board, PlayerColor::White, &pv, 2),
            "PS a1"
        );
    }

    #[test]
    fn format_pv_stops_at_an_illegal_move() {
        let board = only_black_can_move(PlayerColor::Black);
        let pv = [Square::D4, Square::A1];
        assert_eq!(
            format_pv_with_passes(&board, PlayerColor::Black, &pv, 8),
            ""
        );
    }
}
