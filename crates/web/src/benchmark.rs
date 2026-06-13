use reversi_core::board::Board;
use reversi_core::constants::INITIAL_EMPTY_COUNT;
use reversi_core::disc::Disc;
use reversi_core::eval::pattern_feature::{PatternFeature, PatternFeatures};
use reversi_core::move_list::MoveList;
use reversi_core::search::side_to_move::SideToMove;
use reversi_core::types::Depth;
use std::hint::black_box;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::config::{DEFAULT_TT_MB, MIDGAME_SELECTIVITY};
use crate::eval::Eval;
use crate::level::Level;
use crate::search::{Search, search_context::SearchContext, search_result::SearchResult};
use crate::transposition_table::TranspositionTable;

const BENCH_TEST_POSITIONS: usize = 11;
const BENCH_MOVES_PER_POSITION_BASE: usize = 10;
const BENCH_MOVES_STEP: usize = 2;

const FFO_40_BOARD_STR: &str = "O--OOOOX-OOOOOOXOOXXOOOXOOXOOOXXOOOOOOXX---OOOOX----O--X--------";
const FFO_40_EXPECTED_SCORE: i32 = 38;

const FFO_41_BOARD_STR: &str = "-OOOOO----OOOOX--OOOOOO-XXXXXOO--XXOOX--OOXOXX----OXXO---OOO--O-";
const FFO_41_EXPECTED_SCORE: i32 = 0;

struct NetworkBenchInput {
    pattern_feature: PatternFeature,
    ply: usize,
}

impl NetworkBenchInput {
    fn from_board(board: &Board) -> Self {
        let ply = INITIAL_EMPTY_COUNT - board.get_empty_count() as usize;
        let pattern_features = PatternFeatures::new(board, ply);
        Self {
            pattern_feature: *pattern_features.p_feature(ply),
            ply,
        }
    }
}

/// Aggregated timing statistics returned by a benchmark run.
#[wasm_bindgen]
pub struct BenchmarkResult {
    name: String,
    iterations: u32,
    total_time_ms: f64,
    avg_time_us: f64,
    ops_per_sec: f64,
}

#[wasm_bindgen]
impl BenchmarkResult {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    #[wasm_bindgen(getter)]
    pub fn total_time_ms(&self) -> f64 {
        self.total_time_ms
    }

    #[wasm_bindgen(getter)]
    pub fn avg_time_us(&self) -> f64 {
        self.avg_time_us
    }

    #[wasm_bindgen(getter)]
    pub fn ops_per_sec(&self) -> f64 {
        self.ops_per_sec
    }
}

/// Driver that runs the WebAssembly micro-benchmarks against a fixed corpus.
#[wasm_bindgen]
pub struct BenchmarkRunner {
    eval: Rc<Eval>,
    test_boards: Vec<Board>,
    network_inputs: Vec<NetworkBenchInput>,
}

#[wasm_bindgen]
impl BenchmarkRunner {
    /// Creates a benchmark runner and pre-generates the test board corpus.
    ///
    /// # Errors
    ///
    /// Returns an error if the evaluation network fails to load.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BenchmarkRunner, JsValue> {
        console_error_panic_hook::set_once();

        let eval = Rc::new(Eval::new().map_err(|e| {
            JsValue::from_str(&format!("Failed to load evaluation network: {}", e))
        })?);

        let test_boards = Self::generate_test_boards();
        let network_inputs = test_boards
            .iter()
            .map(NetworkBenchInput::from_board)
            .collect();

        Ok(BenchmarkRunner {
            eval,
            test_boards,
            network_inputs,
        })
    }

    /// Generates a variety of test positions for benchmarking.
    fn generate_test_boards() -> Vec<Board> {
        let mut boards = Vec::with_capacity(BENCH_TEST_POSITIONS);
        boards.push(Board::new());

        for seed in 0..10 {
            let mut board = Board::new();
            let moves_to_make = BENCH_MOVES_PER_POSITION_BASE + (seed * BENCH_MOVES_STEP);

            for step in 0..moves_to_make {
                let moves = MoveList::new(&board);
                if moves.count() == 0 {
                    break;
                }
                let move_idx = (seed + step) % moves.count();
                if let Some(mv) = moves.iter().nth(move_idx) {
                    board = board.make_move(mv.sq);
                } else {
                    break;
                }
            }
            boards.push(board);
        }

        boards
    }

    /// Measures execution time and calculates benchmark statistics.
    fn measure_benchmark<F>(
        name: &str,
        iterations: u32,
        ops_per_iteration: u32,
        mut benchmark_fn: F,
    ) -> BenchmarkResult
    where
        F: FnMut(),
    {
        let start = web_sys::window().unwrap().performance().unwrap().now();

        for _ in 0..iterations {
            benchmark_fn();
        }

        let end = web_sys::window().unwrap().performance().unwrap().now();

        let total_time_ms = end - start;
        let total_ops = iterations * ops_per_iteration;
        let avg_time_us = (total_time_ms * 1000.0) / total_ops as f64;
        let ops_per_sec = total_ops as f64 / (total_time_ms / 1000.0);

        BenchmarkResult {
            name: name.to_string(),
            iterations: total_ops,
            total_time_ms,
            avg_time_us,
            ops_per_sec,
        }
    }

    /// Benchmarks move generation performance.
    pub fn bench_move_generation(&self, iterations: u32) -> BenchmarkResult {
        let boards = &self.test_boards;

        Self::measure_benchmark("Move Generation", iterations, boards.len() as u32, || {
            for board in boards {
                let moves = MoveList::new(board);
                let _ = moves.count();
            }
        })
    }

    /// Benchmarks neural network evaluation performance.
    pub fn bench_evaluation(&self, iterations: u32) -> BenchmarkResult {
        let boards = &self.test_boards;
        let tt = Rc::new(TranspositionTable::new(DEFAULT_TT_MB));
        let eval = Rc::clone(&self.eval);

        Self::measure_benchmark(
            "Neural Network Evaluation",
            iterations,
            boards.len() as u32,
            || {
                for board in boards {
                    let ctx = SearchContext::new(
                        board,
                        MIDGAME_SELECTIVITY,
                        Rc::clone(&tt),
                        Rc::clone(&eval),
                        None,
                    );
                    let _ = eval.evaluate(&ctx, board);
                }
            },
        )
    }

    /// Returns how many positions are evaluated by one network-forward iteration.
    pub fn network_forward_positions(&self) -> u32 {
        self.network_inputs.len() as u32
    }

    /// Runs raw neural network forward passes without cache or context setup.
    pub fn run_network_forward(&self, iterations: u32) -> i32 {
        let eval = &self.eval;
        let inputs = &self.network_inputs;
        let mut checksum = 0i32;

        for _ in 0..iterations {
            for input in inputs {
                let score =
                    eval.evaluate_network(black_box(&input.pattern_feature), black_box(input.ply));
                let score_value = black_box(score.value());
                checksum = checksum
                    .wrapping_add(score_value.wrapping_mul(31))
                    .wrapping_add(input.ply as i32);
            }
        }

        black_box(checksum)
    }

    /// Benchmarks search performance at the given fixed depth.
    pub fn bench_search(&self, depth: u8, iterations: u32) -> BenchmarkResult {
        let tt = Rc::new(TranspositionTable::new(DEFAULT_TT_MB));
        let mut search = Search::new(Rc::clone(&tt), Rc::clone(&self.eval));
        let board = Board::new();
        let level = Level {
            mid_depth: depth as Depth,
            end_depth: depth as Depth,
            perfect_depth: depth as Depth,
        };

        Self::measure_benchmark(&format!("Search (depth {})", depth), iterations, 1, || {
            let _ = search.run(&board, level, MIDGAME_SELECTIVITY, None);
            tt.clear();
        })
    }

    /// Benchmarks endgame search performance using FFO #40 and #41.
    pub fn bench_endgame(&self, iterations: u32) -> BenchmarkResult {
        let tt = Rc::new(TranspositionTable::new(DEFAULT_TT_MB));
        let mut search = Search::new(Rc::clone(&tt), Rc::clone(&self.eval));

        let board_40 = Board::from_string(FFO_40_BOARD_STR, Disc::Black).unwrap();
        let empty_count_40 = board_40.get_empty_count();
        let level_40 = Level {
            mid_depth: empty_count_40 as Depth,
            end_depth: empty_count_40 as Depth,
            perfect_depth: empty_count_40 as Depth,
        };

        let board_41 = Board::from_string(FFO_41_BOARD_STR, Disc::Black).unwrap();
        let empty_count_41 = board_41.get_empty_count();
        let level_41 = Level {
            mid_depth: empty_count_41 as Depth,
            end_depth: empty_count_41 as Depth,
            perfect_depth: empty_count_41 as Depth,
        };

        let mut result_40_opt: Option<(SearchResult, f64)> = None;
        let mut result_41_opt: Option<(SearchResult, f64)> = None;

        let benchmark_result =
            Self::measure_benchmark("Endgame Search (FFO #40-41)", iterations, 2, || {
                let perf = web_sys::window().unwrap().performance().unwrap();

                let start_40 = perf.now();
                let result_40 = search.run(&board_40, level_40, MIDGAME_SELECTIVITY, None);
                let elapsed_40 = perf.now() - start_40;
                result_40_opt = Some((result_40, elapsed_40));

                let start_41 = perf.now();
                let result_41 = search.run(&board_41, level_41, MIDGAME_SELECTIVITY, None);
                let elapsed_41 = perf.now() - start_41;
                result_41_opt = Some((result_41, elapsed_41));
            });

        if let Some((result, elapsed)) = result_40_opt {
            let nps = (result.n_nodes as f64) / (elapsed / 1000.0);
            let best_move = match result.best_move {
                Some(sq) => sq.to_string(),
                None => "None".to_string(),
            };
            web_sys::console::log_1(&format!(
                "FFO #40 ({} empties) - Score: {} (expected: {}), Best Move: {}, Nodes: {}, NPS: {:.0}",
                empty_count_40, result.score, FFO_40_EXPECTED_SCORE, best_move, result.n_nodes, nps
            ).into());
        }

        if let Some((result, elapsed)) = result_41_opt {
            let nps = (result.n_nodes as f64) / (elapsed / 1000.0);
            let best_move = match result.best_move {
                Some(sq) => sq.to_string(),
                None => "None".to_string(),
            };
            web_sys::console::log_1(&format!(
                "FFO #41 ({} empties) - Score: {} (expected: {}), Best Move: {}, Nodes: {}, NPS: {:.0}",
                empty_count_41, result.score, FFO_41_EXPECTED_SCORE, best_move, result.n_nodes, nps
            ).into());
        }

        benchmark_result
    }

    /// Benchmarks perft (performance test) for move generation.
    pub fn bench_perft(&self, depth: u32, iterations: u32) -> BenchmarkResult {
        let board = Board::new();

        let result =
            Self::measure_benchmark(&format!("Perft (depth {})", depth), iterations, 1, || {
                let mut pattern_features = PatternFeatures::new(&board, 0);
                let side_to_move = SideToMove::Player;
                let _ = Self::perft(&board, &mut pattern_features, 0, side_to_move, depth);
            });

        let mut pattern_features = PatternFeatures::new(&board, 0);
        let nodes = Self::perft(&board, &mut pattern_features, 0, SideToMove::Player, depth);
        web_sys::console::log_1(&format!("Perft depth {} - Nodes: {}", depth, nodes).into());

        result
    }

    /// Recursively counts leaf nodes for perft.
    fn perft(
        board: &Board,
        pattern_feature: &mut PatternFeatures,
        ply: usize,
        side_to_move: SideToMove,
        depth: u32,
    ) -> u64 {
        let mut nodes = 0;
        let move_list = MoveList::new(board);

        if move_list.count() > 0 {
            for m in move_list.iter() {
                let next = board.make_move_with_flipped(m.flipped, m.sq);
                pattern_feature.update(m.sq, m.flipped, ply, side_to_move);

                if depth <= 1 {
                    nodes += 1;
                } else {
                    nodes += Self::perft(
                        &next,
                        pattern_feature,
                        ply + 1,
                        side_to_move.switch(),
                        depth - 1,
                    );
                }
            }
        } else {
            let next = board.switch_players();
            if next.has_legal_moves() {
                nodes += Self::perft(&next, pattern_feature, ply, side_to_move.switch(), depth);
            } else {
                nodes += 1;
            }
        }
        nodes
    }
}
