use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Duration;

use reversi_core::{
    board::Board,
    level::{Level, get_level},
    piece::Piece,
    search::{Search, SearchConstraint, SearchOptions},
    types::Selectivity,
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

fn parse_position_line(line: &str) -> Result<(Board, Piece), String> {
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
        'X' => Piece::Black,
        'O' => Piece::White,
        _ => return Err(format!("Invalid side to move: {side_char}")),
    };

    let board = Board::from_string(board_str, side_to_move);

    Ok((board, side_to_move))
}

fn solve_position(
    search: &mut Search,
    board: Board,
    _side_to_move: Piece,
    level: reversi_core::level::Level,
    selectivity: Selectivity,
    position_num: usize,
) -> (Duration, u64) {
    use std::time::Instant;

    // search
    search.init();
    let start_time = Instant::now();
    let constraint = SearchConstraint::Level(level);
    let result = search.run(
        &board,
        constraint,
        selectivity,
        false,
        None::<fn(reversi_core::search::SearchProgress)>,
    );
    let elapsed = start_time.elapsed();

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
    let pv_string = if result.pv_line.is_empty() {
        result.best_move.map_or("--".to_string(), |m| m.to_string())
    } else {
        result
            .pv_line
            .iter()
            .take(8)
            .map(|sq| sq.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };

    println!(
        "| {:^3} | {:^6} | {:^+5} | {:>2}:{:06.3} | {:>12} | {:>10.0} | {:23} |",
        position_num,
        depth,
        result.score as i32,
        elapsed.as_secs() / 60,
        elapsed.as_secs_f64() % 60.0,
        result.n_nodes,
        nodes_per_sec,
        pv_string
    );

    (elapsed, result.n_nodes)
}
