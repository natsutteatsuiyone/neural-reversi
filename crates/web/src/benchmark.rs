use reversi_core::board::Board;
use reversi_core::constants::INITIAL_EMPTY_COUNT;
use reversi_core::eval::pattern_feature::{PatternFeature, PatternFeatures};
use reversi_core::move_list::MoveList;
use std::hint::black_box;
use wasm_bindgen::prelude::*;

use crate::eval::Eval;

const BENCH_TEST_POSITIONS: usize = 11;
const BENCH_MOVES_PER_POSITION_BASE: usize = 10;
const BENCH_MOVES_STEP: usize = 2;

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

/// Driver that runs the WebAssembly micro-benchmarks against a fixed corpus.
#[wasm_bindgen]
pub struct BenchmarkRunner {
    eval: Eval,
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

        let eval = Eval::new()
            .map_err(|e| JsValue::from_str(&format!("Failed to load evaluation network: {}", e)))?;

        let network_inputs = Self::generate_test_boards()
            .iter()
            .map(NetworkBenchInput::from_board)
            .collect();

        Ok(BenchmarkRunner {
            eval,
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
}
