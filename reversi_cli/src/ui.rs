use reversi_core::{
    self, level,
    piece::Piece,
    search::{self, SearchOptions},
    square::Square, types::Selectivity,
};
use rustyline::{error::ReadlineError, DefaultEditor};

use crate::game::GameState;

pub fn ui_loop(selectvity: Selectivity) {
    let mut rl = DefaultEditor::new().unwrap();
    let mut game = GameState::new();
    let mut search = search::Search::new(&SearchOptions::default());
    let mut level: usize = 10;
    let mut game_mode: usize = 3;

    loop {
        game.print();
        println!();

        let current_side = game.get_side_to_move();
        let should_ai_play = matches!((game_mode, current_side), (0, Piece::White) | (1, Piece::Black) | (2, _));

        if should_ai_play && !game.board.is_game_over() {
            let result = search.run(
                &game.board,
                level::get_level(level),
                selectvity,
                None::<fn(reversi_core::search::SearchProgress)>,
            );
            if let Some(computer_move) = result.pv_line.first() {
                game.make_move(*computer_move);
                println!("Computer plays {:?}\n", computer_move);
                continue;
            }
        }

        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(&line);
                let mut parts = line.split_whitespace();
                if let Some(cmd) = parts.next() {
                    println!();

                    match cmd {
                        "init" | "i" => {
                            game = GameState::new();
                            search.init();
                        }
                        "new" | "n" => {
                            game = GameState::new();
                            search.init();
                        }
                        "undo" | "u" => {
                            if game.undo() {
                            } else {
                                println!("Cannot undo.");
                            }
                        }
                        "level" | "l" => {
                            if let Some(lvl_str) = parts.next() {
                                if let Ok(lvl) = lvl_str.parse::<usize>() {
                                    level = lvl;
                                }
                            }
                        }
                        "mode" | "m" => {
                            if let Some(mode_str) = parts.next() {
                                if let Ok(mode) = mode_str.parse::<usize>() {
                                    if mode <= 3 {
                                        game_mode = mode;
                                        println!("Mode changed to: {}", mode);
                                    } else {
                                        println!("Invalid mode number. Please specify a value between 0-3.");
                                    }
                                } else {
                                    println!(
                                        "Invalid mode number. Please specify a value between 0-3."
                                    );
                                }
                            } else {
                                println!("Current mode: {}", game_mode);
                                println!("0: Black-Human, White-AI");
                                println!("1: Black-AI, White-Human");
                                println!("2: Black-AI, White-AI");
                                println!("3: Black-Human, White-Human");
                            }
                        }
                        "go" => {
                            let result = search.run(
                                &game.board,
                                level::get_level(level),
                                selectvity,
                                None::<fn(reversi_core::search::SearchProgress)>,
                            );
                            if let Some(computer_move) = result.pv_line.first() {
                                game.make_move(*computer_move);

                                println!("depth | score | nodes ");
                                println!("----------------------");
                                println!(
                                    "{}@{} | {} | {}\n",
                                    result.depth,
                                    result.get_probability(),
                                    result.score,
                                    result.n_nodes
                                );
                                println!("Computer plays {:?}\n", computer_move);
                            }
                        }
                        "play" => {
                            if let Some(moves_str) = parts.next() {
                                for sq_str in moves_str.chars().collect::<Vec<_>>().chunks(2) {
                                    let sq_str: String = sq_str.iter().collect();
                                    let sq = sq_str.parse::<Square>();
                                    if let Ok(sq) = sq {
                                        if game.board.is_legal_move(sq) {
                                            game.make_move(sq);
                                        } else {
                                            break;
                                        }
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }
                        "quit" | "q" => break,
                        _ => {
                            let sq = cmd.parse::<Square>();
                            if let Ok(sq) = sq {
                                if game.board.is_legal_move(sq) {
                                    game.make_move(sq);
                                } else {
                                    println!("Illegal move: {}\n", cmd);
                                }
                            } else {
                                println!("Unknown command: {}\n", cmd);
                            }
                        }
                    }
                } else {
                    continue;
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
}
