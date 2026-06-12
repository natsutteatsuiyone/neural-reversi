use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use num_format::{Locale, ToFormattedString};
use reversi_core::{
    board::Board,
    disc::Disc,
    level::{Level, get_level},
    obf::ObfPosition,
    probcut::Selectivity,
    search::{Search, SearchRunOptions, options::SearchOptions},
    square::Square,
};

use crate::config::EngineConfig;

const NUM_WIDTH: usize = 5;

pub fn solve(
    file_path: &Path,
    config: &EngineConfig,
    exact: bool,
    all_moves: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let search_options = SearchOptions::new(config.hash_size)
        .with_threads(config.threads)
        .with_eval_paths(config.eval_file.as_deref(), config.eval_sm_file.as_deref());

    print_header(file_path, &search_options);

    let mut search = Search::new(&search_options);
    let level_config = if exact {
        Level::perfect()
    } else {
        get_level(config.level)
    };

    if !all_moves {
        print_table_header();
    }

    let mut total_time = Duration::ZERO;
    let mut total_nodes: u64 = 0;

    for (line_num, line) in reader.lines().enumerate() {
        let raw = line?;
        let pos = match ObfPosition::parse(&raw) {
            Ok(Some(pos)) => pos,
            Ok(None) => continue,
            Err(e) => {
                eprintln!("Error parsing line {}: {}", line_num + 1, e);
                continue;
            }
        };
        let (elapsed, nodes) = solve_position(
            &mut search,
            pos.board,
            pos.side_to_move,
            level_config,
            config.selectivity,
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

    Ok(())
}

fn print_table_header() {
    println!(
        "| {:^NUM_WIDTH$} | {:^6} | {:^5} | {:^11} | {:^19} | {:^13} | {:^23} |",
        "#", "Depth", "Score", "Time", "Nodes", "N/s", "Principal Variation"
    );
    println!(
        "|{:-<width$}|--------|-------|-------------|---------------------|---------------|-------------------------|",
        "",
        width = NUM_WIDTH + 2
    );
}

fn print_all_moves_table_header() {
    println!("| {:^5} | {:^23} |", "Score", "Principal Variation");
    println!("|-------|-------------------------|");
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

fn print_header(file_path: &Path, options: &SearchOptions) {
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
    println!("- Hash size: {} MB", options.tt_mb_size);
    println!("- Threads:   {}", options.n_threads);
    println!();
}

fn solve_position(
    search: &mut Search,
    board: Board,
    side_to_move: Disc,
    level: Level,
    selectivity: Selectivity,
    position_num: usize,
    all_moves: bool,
) -> (Duration, u64) {
    let is_pass = !board.has_legal_moves();

    if is_pass && !board.switch_players().has_legal_moves() {
        let score = board.solve(board.get_empty_count());
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

    search.init();
    let start_time = Instant::now();
    let options = SearchRunOptions::with_level(level, selectivity).multi_pv(all_moves);
    let result = search.run(&search_board, &options);
    let elapsed = start_time.elapsed();

    let depth = if result.get_probability() == 100 {
        format!("{}", result.depth())
    } else {
        format!("{}@{}%", result.depth(), result.get_probability())
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
            let score = if is_pass {
                -(pv_move.score as i32)
            } else {
                pv_move.score as i32
            };
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
    let score = if is_pass {
        -(result_score as i32)
    } else {
        result_score as i32
    };
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

fn print_all_moves_row(score: impl Display, pv: impl Display) {
    println!("| {score:^5} | {pv:<23} |");
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
        "| {num:>NUM_WIDTH$} | {depth:^6} | {score:^5} | {time:>11} | {nodes:>19} | {nps:>13} | {pv:<23} |"
    );
}

fn format_pv_with_passes(
    board: &Board,
    side_to_move: Disc,
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
        push_token(&mut result, &format_square(sq, side));
        current = current.make_move(sq);
        side = side.opposite();
        count += 1;
    }

    result
}

fn format_root_move(sq: Square, move_side: Disc, is_pass: bool, side_to_move: Disc) -> String {
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
    let frac_secs = duration.as_secs_f64() % 60.0;
    if total_secs >= 3600 {
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        format!("{hours}:{mins:02}:{frac_secs:06.3}")
    } else {
        let mins = total_secs / 60;
        format!("{mins}:{frac_secs:06.3}")
    }
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
