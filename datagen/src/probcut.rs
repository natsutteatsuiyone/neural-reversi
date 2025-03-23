use clap::Parser;
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};

use reversi_core::{
    board::Board,
    level::get_level,
    piece::Piece,
    search::{Search, SearchOptions},
    square::Square,
    types::{Depth, Scoref},
};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Debug)]
struct ProbCutSample {
    ply: u32,
    shallow_depth: Depth,
    shallow_score: Scoref,
    deep_depth: Depth,
    deep_score: Scoref,
}

pub fn execute(input: &str, output: &str) {
    let options = SearchOptions {
        tt_mb_size: 256,
        ..Default::default()
    };
    let mut search = Search::new(&options);

    let file = File::open(input).unwrap();
    let reader = BufReader::new(file);

    let file = File::create(output).unwrap();
    let mut file = std::io::BufWriter::new(file);
    file.write_all(b"ply,shallow_depth,deep_depth,diff\n")
        .unwrap();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut samples = Vec::new();
        let mut board = Board::new();
        let mut side_to_move = Piece::Black;

        for token in line.as_bytes().chunks_exact(2) {
            let m = std::str::from_utf8(token).unwrap();
            let sq = m.parse::<Square>().unwrap();

            if !board.has_legal_moves() {
                board = board.switch_players();
                side_to_move = side_to_move.opposite();
                if !board.has_legal_moves() {
                    break;
                }
            }

            let num_depth = 14;
            let max_shallow_depth = 8;
            let ply = 60 - board.get_empty_count();

            search.init();
            let depth_scores: Vec<(Depth, Scoref)> = (0..num_depth)
                .map(|depth| {
                    let mut lv = get_level(depth);
                    lv.end_depth = [depth as Depth; 7];
                    let result = search.test(&board, lv, 6);
                    (depth as Depth, result.score)
                })
                .collect();

            for (shallow_depth, shallow_score) in depth_scores.iter().take(max_shallow_depth + 1) {
                samples.extend(
                    depth_scores.iter()
                        .filter(|(deep_depth, _)| *deep_depth > *shallow_depth + 2)
                        .map(|(deep_depth, deep_score)| ProbCutSample {
                            ply,
                            shallow_depth: *shallow_depth,
                            shallow_score: *shallow_score,
                            deep_depth: *deep_depth,
                            deep_score: *deep_score,
                        })
                );
            }

            board = board.make_move(sq);
        }

        for sample in samples.iter() {
            let line = format!(
                "{},{},{},{}\n",
                sample.ply,
                sample.shallow_depth,
                sample.deep_depth,
                sample.deep_score - sample.shallow_score
            );
            file.write_all(line.as_bytes()).unwrap();
        }
        file.flush().unwrap();

        println!("{}", line_no);
    }
}
