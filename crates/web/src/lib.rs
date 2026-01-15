mod eval;
mod level;
mod move_list;
mod probcut;
mod search;

use crate::{
    eval::Eval,
    level::Level,
    search::{Search, search_context::SearchContext},
};
use js_sys::Function;
use rand::Rng;
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::eval::pattern_feature::PatternFeatures;
use reversi_core::move_list::MoveList;
use reversi_core::search::side_to_move::SideToMove;
use reversi_core::square::{Square, TOTAL_SQUARES};
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::probcut::Selectivity;
use reversi_core::types::Depth;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

const DEFAULT_TT_MB: usize = 32;
const DEFAULT_MID_DEPTH: Depth = 7;
const MIDGAME_SELECTIVITY: Selectivity = Selectivity::Level2;
const MIN_MID_DEPTH: u8 = 1;
const MAX_MID_DEPTH: u8 = 15;

struct EngineState {
    search: Search,
    tt: Rc<TranspositionTable>,
}

impl EngineState {
    fn new() -> Self {
        let tt = Rc::new(TranspositionTable::new(DEFAULT_TT_MB));
        let eval = Rc::new(Eval::new().expect("Failed to load evaluation network"));
        let search = Search::new(Rc::clone(&tt), eval);
        EngineState { search, tt }
    }

    fn reset(&self) {
        self.tt.clear();
    }

    fn search(
        &mut self,
        board: &Board,
        level: Level,
        progress_callback: Option<Function>,
    ) -> Option<Square> {
        let result = self
            .search
            .run(board, level, MIDGAME_SELECTIVITY, progress_callback);
        if let Some(best_move) = result.best_move {
            return Some(best_move);
        }

        let moves = MoveList::new(board);
        moves.first().map(|mv| mv.sq)
    }
}

#[wasm_bindgen]
pub struct Game {
    board: Board,
    current_player: Disc,
    human_player: Disc,
    ai_player: Disc,
    engine: EngineState,
    mid_depth: Depth,
    progress_callback: Option<Function>,
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new(human_is_black: bool) -> Game {
        console_error_panic_hook::set_once();

        let mut game = Game {
            board: Board::new(),
            current_player: Disc::Black,
            human_player: Disc::Black,
            ai_player: Disc::White,
            engine: EngineState::new(),
            mid_depth: DEFAULT_MID_DEPTH,
            progress_callback: None,
        };
        game.set_players(human_is_black);
        game
    }

    pub fn set_progress_callback(&mut self, callback: Option<Function>) {
        self.progress_callback = callback;
    }

    pub fn reset(&mut self, human_is_black: bool) {
        self.set_players(human_is_black);
    }

    pub fn board(&self) -> Vec<u8> {
        let (black_bits, white_bits) = self.color_bitboards();
        let mut cells = vec![0u8; TOTAL_SQUARES];
        for (idx, cell) in cells.iter_mut().enumerate().take(TOTAL_SQUARES) {
            let mask = 1u64 << idx;
            if (black_bits & mask) != 0 {
                *cell = 1;
            } else if (white_bits & mask) != 0 {
                *cell = 2;
            }
        }
        cells
    }

    pub fn legal_moves(&self) -> Vec<u8> {
        let moves = MoveList::new(&self.board);
        let mut result = Vec::with_capacity(moves.count());
        for mv in moves.iter() {
            result.push(mv.sq.index() as u8);
        }
        result
    }

    pub fn human_move(&mut self, index: u8) -> bool {
        if self.board.is_game_over() || self.current_player != self.human_player {
            return false;
        }

        let square = match Square::from_u8(index) {
            Some(Square::None) | None => return false,
            Some(sq) => sq,
        };

        if let Some(next_board) = self.board.try_make_move(square) {
            self.board = next_board;
            self.current_player = self.current_player.opposite();
            self.handle_forced_passes();
            true
        } else {
            false
        }
    }

    pub fn pass(&mut self) -> bool {
        if self.board.is_game_over()
            || self.current_player != self.human_player
            || self.board.has_legal_moves()
        {
            return false;
        }

        self.board = self.board.switch_players();
        self.current_player = self.current_player.opposite();
        self.handle_forced_passes();
        true
    }

    pub fn ai_move(&mut self) -> Option<u8> {
        if self.board.is_game_over() || self.current_player != self.ai_player {
            return None;
        }

        if !self.board.has_legal_moves() {
            self.board = self.board.switch_players();
            self.current_player = self.current_player.opposite();
            self.handle_forced_passes();
            return None;
        }

        let best_square = self.select_ai_move()?;
        self.board = self.board.make_move(best_square);
        self.current_player = self.current_player.opposite();
        self.handle_forced_passes();
        Some(best_square.index() as u8)
    }

    pub fn current_player(&self) -> u8 {
        piece_to_u8(self.current_player)
    }

    pub fn human_color(&self) -> u8 {
        piece_to_u8(self.human_player)
    }

    pub fn ai_color(&self) -> u8 {
        piece_to_u8(self.ai_player)
    }

    pub fn is_game_over(&self) -> bool {
        self.board.is_game_over()
    }

    pub fn score(&self) -> Vec<u8> {
        let (black_bits, white_bits) = self.color_bitboards();
        vec![black_bits.count_ones() as u8, white_bits.count_ones() as u8]
    }

    pub fn empty_count(&self) -> u8 {
        self.board.get_empty_count() as u8
    }

    pub fn set_level(&mut self, level: u8) {
        let clamped = level.clamp(MIN_MID_DEPTH, MAX_MID_DEPTH);
        self.mid_depth = clamped as Depth;
    }

    /// Make a move without checking whose turn it is (for replay purposes)
    pub fn make_move_unchecked(&mut self, index: u8) -> bool {
        if self.board.is_game_over() {
            return false;
        }

        let square = match Square::from_u8(index) {
            Some(Square::None) | None => return false,
            Some(sq) => sq,
        };

        if let Some(next_board) = self.board.try_make_move(square) {
            self.board = next_board;
            self.current_player = self.current_player.opposite();
            self.handle_forced_passes();
            true
        } else {
            false
        }
    }
}

impl Game {
    fn set_players(&mut self, human_is_black: bool) {
        self.engine.reset();
        self.human_player = if human_is_black {
            Disc::Black
        } else {
            Disc::White
        };
        self.ai_player = self.human_player.opposite();
        self.board = Board::new();
        self.current_player = Disc::Black;
        self.handle_forced_passes();
    }

    fn color_bitboards(&self) -> (u64, u64) {
        match self.current_player {
            Disc::Black => (self.board.player.0, self.board.opponent.0),
            Disc::White => (self.board.opponent.0, self.board.player.0),
            Disc::Empty => (0, 0),
        }
    }

    fn handle_forced_passes(&mut self) {
        while !self.board.is_game_over() && !self.board.has_legal_moves() {
            self.board = self.board.switch_players();
            self.current_player = self.current_player.opposite();
        }
    }

    fn select_ai_move(&mut self) -> Option<Square> {
        let level = level_for_position(self.mid_depth);
        self.engine
            .search(&self.board, level, self.progress_callback.clone())
    }
}

fn level_for_position(mid_depth: Depth) -> Level {
    let end_depth = (mid_depth as f64 * 1.6).round() as Depth;
    Level {
        mid_depth,
        end_depth,
        perfect_depth: if end_depth > 10 {
            (end_depth - 2).max(10)
        } else {
            end_depth
        },
    }
}

fn piece_to_u8(piece: Disc) -> u8 {
    match piece {
        Disc::Empty => 0,
        Disc::Black => 1,
        Disc::White => 2,
    }
}

// Benchmark module
const BENCH_TEST_POSITIONS: usize = 11; // 1 opening + 10 midgame
const BENCH_MOVES_PER_POSITION_BASE: usize = 10;
const BENCH_MOVES_STEP: usize = 2;

// FFO test cases for endgame search benchmark
const FFO_40_BOARD_STR: &str = "O--OOOOX-OOOOOOXOOXXOOOXOOXOOOXXOOOOOOXX---OOOOX----O--X--------";
const FFO_40_EXPECTED_SCORE: i32 = 38;

const FFO_41_BOARD_STR: &str = "-OOOOO----OOOOX--OOOOOO-XXXXXOO--XXOOX--OOXOXX----OXXO---OOO--O-";
const FFO_41_EXPECTED_SCORE: i32 = 0;

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

#[wasm_bindgen]
pub struct BenchmarkRunner {
    eval: Rc<Eval>,
    test_boards: Vec<Board>,
}

#[wasm_bindgen]
impl BenchmarkRunner {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<BenchmarkRunner, JsValue> {
        console_error_panic_hook::set_once();

        let eval = Rc::new(Eval::new().map_err(|e| {
            JsValue::from_str(&format!("Failed to load evaluation network: {}", e))
        })?);

        let test_boards = Self::generate_test_boards();

        Ok(BenchmarkRunner { eval, test_boards })
    }

    /// Generate a variety of test positions for benchmarking
    fn generate_test_boards() -> Vec<Board> {
        let mut boards = Vec::with_capacity(BENCH_TEST_POSITIONS);
        boards.push(Board::new()); // Opening position

        // Create midgame positions by making random moves
        let mut rng = rand::rng();
        for seed in 0..10 {
            let mut board = Board::new();
            let moves_to_make = BENCH_MOVES_PER_POSITION_BASE + (seed * BENCH_MOVES_STEP);

            for _ in 0..moves_to_make {
                let moves = MoveList::new(&board);
                if moves.count() == 0 {
                    break;
                }
                let move_idx = rng.random_range(0..moves.count());
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

    /// Measure execution time and calculate benchmark statistics
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

    /// Benchmark move generation performance
    pub fn bench_move_generation(&self, iterations: u32) -> BenchmarkResult {
        let boards = &self.test_boards;

        Self::measure_benchmark("Move Generation", iterations, boards.len() as u32, || {
            for board in boards {
                let moves = MoveList::new(board);
                // Force evaluation to prevent optimization
                let _ = moves.count();
            }
        })
    }

    /// Benchmark neural network evaluation performance
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
                    // Create a temporary SearchContext for evaluation
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

    /// Benchmark search performance (fixed depth)
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
            let _ = search.run(&board, level.clone(), MIDGAME_SELECTIVITY, None);
            tt.clear();
        })
    }

    /// Benchmark endgame search performance using FFO #40 and #41
    pub fn bench_endgame(&self, iterations: u32) -> BenchmarkResult {
        use crate::search::search_result::SearchResult;

        let tt = Rc::new(TranspositionTable::new(DEFAULT_TT_MB));
        let mut search = Search::new(Rc::clone(&tt), Rc::clone(&self.eval));

        // FFO #40
        let board_40 = Board::from_string(FFO_40_BOARD_STR, Disc::Black);
        let empty_count_40 = board_40.get_empty_count();
        let level_40 = Level {
            mid_depth: empty_count_40 as Depth,
            end_depth: empty_count_40 as Depth,
            perfect_depth: empty_count_40 as Depth,
        };

        // FFO #41
        let board_41 = Board::from_string(FFO_41_BOARD_STR, Disc::Black);
        let empty_count_41 = board_41.get_empty_count();
        let level_41 = Level {
            mid_depth: empty_count_41 as Depth,
            end_depth: empty_count_41 as Depth,
            perfect_depth: empty_count_41 as Depth,
        };

        // Store results for logging outside the benchmark
        let mut result_40_opt: Option<(SearchResult, f64)> = None;
        let mut result_41_opt: Option<(SearchResult, f64)> = None;

        let benchmark_result = Self::measure_benchmark(
            "Endgame Search (FFO #40-41)",
            iterations,
            2, // 2 positions per iteration
            || {
                let perf = web_sys::window().unwrap().performance().unwrap();

                // FFO #40
                let start_40 = perf.now();
                let result_40 = search.run(&board_40, level_40.clone(), MIDGAME_SELECTIVITY, None);
                let elapsed_40 = perf.now() - start_40;
                result_40_opt = Some((result_40, elapsed_40));

                // FFO #41
                let start_41 = perf.now();
                let result_41 = search.run(&board_41, level_41.clone(), MIDGAME_SELECTIVITY, None);
                let elapsed_41 = perf.now() - start_41;
                result_41_opt = Some((result_41, elapsed_41));
            },
        );

        // Log results after benchmark measurement
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

    /// Benchmark perft (performance test) for move generation
    pub fn bench_perft(&self, depth: u32, iterations: u32) -> BenchmarkResult {
        let board = Board::new();

        let result =
            Self::measure_benchmark(&format!("Perft (depth {})", depth), iterations, 1, || {
                let mut pattern_features = PatternFeatures::new(&board, 0);
                let side_to_move = SideToMove::Player;
                let _ = Self::perft(&board, &mut pattern_features, 0, side_to_move, depth);
            });

        // Log node count after benchmark measurement for verification
        let mut pattern_features = PatternFeatures::new(&board, 0);
        let nodes = Self::perft(&board, &mut pattern_features, 0, SideToMove::Player, depth);
        web_sys::console::log_1(&format!("Perft depth {} - Nodes: {}", depth, nodes).into());

        result
    }

    /// Helper function for perft recursion
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
