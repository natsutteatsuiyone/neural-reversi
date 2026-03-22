use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Duration;

use reversi_core::search::options::SearchOptions;
use reversi_core::{
    board::Board,
    disc::Disc,
    level::{Level, get_level},
    probcut::Selectivity,
    search::{Search, SearchRunOptions},
    square::Square,
};

#[allow(clippy::too_many_arguments)]
pub fn solve(
    file_path: &Path,
    hash_size: usize,
    level: usize,
    selectivity: Selectivity,
    threads: Option<usize>,
    eval_path: Option<&Path>,
    eval_sm_path: Option<&Path>,
    exact: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let search_options = SearchOptions::new(hash_size)
        .with_threads(threads)
        .with_eval_paths(eval_path, eval_sm_path);

    let mut search = Search::new(&search_options);
    let level_config = if exact {
        Level::perfect()
    } else {
        get_level(level)
    };

    println!(
        "| {:^3} | {:^6} | {:^5} | {:^9} | {:^12} | {:^10} | {:^23} |",
        "#", "Depth", "Score", "Time", "Nodes", "N/s", "Principal Variation"
    );
    println!(
        "|-----|--------|-------|-----------|--------------|------------|-------------------------|"
    );

    let mut total_time = Duration::ZERO;
    let mut total_nodes: u64 = 0;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;

        let line = if let Some(comment_pos) = line.find('%') {
            &line[..comment_pos]
        } else {
            &line
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        match parse_position_line(line) {
            Ok((board, side_to_move)) => {
                let (elapsed, nodes) = solve_position(
                    &mut search,
                    board,
                    side_to_move,
                    level_config,
                    selectivity,
                    line_num + 1,
                );
                total_time += elapsed;
                total_nodes += nodes;
            }
            Err(e) => {
                eprintln!("Error parsing line {}: {}", line_num + 1, e);
                continue;
            }
        }
    }

    // Print summary
    let total_secs = total_time.as_secs_f64();
    let total_nps = if total_secs > 0.0 {
        total_nodes as f64 / total_secs
    } else {
        0.0
    };
    println!(
        "Total: {:.3}s, {} nodes, {:.0} N/s",
        total_secs, total_nodes, total_nps
    );

    Ok(())
}

fn parse_position_line(line: &str) -> Result<(Board, Disc), String> {
    let fields: Vec<&str> = line.split(';').collect();

    if fields.is_empty() {
        return Err("Empty line".to_string());
    }

    let board_field = fields[0].trim();
    if board_field.len() < 65 {
        return Err("Invalid board format".to_string());
    }

    let board_str = &board_field[..64];
    let side_char = board_field.chars().nth(65).unwrap_or('X');

    let side_to_move = match side_char {
        'X' => Disc::Black,
        'O' => Disc::White,
        _ => return Err(format!("Invalid side to move: {side_char}")),
    };

    let board = Board::from_string(board_str, side_to_move)
        .map_err(|e| format!("Invalid board string: {e}"))?;

    Ok((board, side_to_move))
}

fn solve_position(
    search: &mut Search,
    board: Board,
    side_to_move: Disc,
    level: reversi_core::level::Level,
    selectivity: Selectivity,
    position_num: usize,
) -> (Duration, u64) {
    use std::time::Instant;

    let is_pass = !board.has_legal_moves();

    if is_pass && !board.switch_players().has_legal_moves() {
        let score = board.solve(board.get_empty_count());
        println!(
            "| {:^3} | {:^6} | {:^+5} | {:>2}:{:06.3} | {:>12} | {:>10.0} | {:23} |",
            position_num, "END", score, 0, 0.0, 0, 0.0, "--"
        );
        return (Duration::ZERO, 0);
    }
    let search_board = if is_pass {
        board.switch_players()
    } else {
        board
    };

    // search
    search.init();
    let start_time = Instant::now();
    let options = SearchRunOptions::with_level(level, selectivity);
    let result = search.run(&search_board, &options);
    let elapsed = start_time.elapsed();

    let score = if is_pass {
        -(result.score as i32)
    } else {
        result.score as i32
    };

    let depth = if result.get_probability() == 100 {
        format!("{}", result.depth)
    } else {
        format!("{}@{}%", result.depth, result.get_probability())
    };

    // N/s
    let nodes_per_sec = if elapsed.as_secs_f64() > 0.0 {
        result.n_nodes as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    // pv
    let move_side = if is_pass {
        side_to_move.opposite()
    } else {
        side_to_move
    };
    let pv_string = if result.pv_line.is_empty() {
        result
            .best_move
            .map_or("--".to_string(), |m| format_square(m, move_side))
    } else {
        format_pv_with_passes(&board, side_to_move, &result.pv_line, 8)
    };

    println!(
        "| {:^3} | {:^6} | {:^+5} | {:>2}:{:06.3} | {:>12} | {:>10.0} | {:23} |",
        position_num,
        depth,
        score,
        elapsed.as_secs() / 60,
        elapsed.as_secs_f64() % 60.0,
        result.n_nodes,
        nodes_per_sec,
        pv_string
    );

    (elapsed, result.n_nodes)
}

fn format_pv_with_passes(
    board: &Board,
    side_to_move: Disc,
    pv_line: &[Square],
    max_tokens: usize,
) -> String {
    let mut result = String::new();
    let mut current = *board;
    let mut side = side_to_move;
    let mut count = 0;

    for &sq in pv_line {
        if count >= max_tokens {
            break;
        }
        if !current.has_legal_moves() {
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(format_pass(side));
            current = current.switch_players();
            side = side.opposite();
            count += 1;
            if count >= max_tokens {
                break;
            }
        }
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(&format_square(sq, side));
        current = current.make_move(sq);
        side = side.opposite();
        count += 1;
    }

    result
}

fn format_square(sq: Square, side: Disc) -> String {
    let s = sq.to_string();
    if side == Disc::White {
        s.to_uppercase()
    } else {
        s
    }
}

fn format_pass(side: Disc) -> &'static str {
    if side == Disc::White { "PS" } else { "ps" }
}
