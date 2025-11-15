mod eval;
mod level;
mod move_list;
mod probcut;
mod search;

use crate::{eval::Eval, level::Level, search::Search};
use js_sys::Function;
use reversi_core::board::Board;
use reversi_core::move_list::MoveList;
use reversi_core::piece::Piece;
use reversi_core::square::{Square, TOTAL_SQUARES};
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::types::Depth;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

const DEFAULT_TT_MB: usize = 32;
const DEFAULT_MID_DEPTH: Depth = 7;
const MIDGAME_SELECTIVITY: u8 = 2;
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
    current_player: Piece,
    human_player: Piece,
    ai_player: Piece,
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
            current_player: Piece::Black,
            human_player: Piece::Black,
            ai_player: Piece::White,
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
        for idx in 0..TOTAL_SQUARES {
            let mask = 1u64 << idx;
            if (black_bits & mask) != 0 {
                cells[idx] = 1;
            } else if (white_bits & mask) != 0 {
                cells[idx] = 2;
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
            Piece::Black
        } else {
            Piece::White
        };
        self.ai_player = self.human_player.opposite();
        self.board = Board::new();
        self.current_player = Piece::Black;
        self.handle_forced_passes();
    }

    fn color_bitboards(&self) -> (u64, u64) {
        match self.current_player {
            Piece::Black => (self.board.player, self.board.opponent),
            Piece::White => (self.board.opponent, self.board.player),
            Piece::Empty => (0, 0),
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
    Level {
        mid_depth,
        end_depth: (mid_depth as f64 * 1.45).round() as Depth,
    }
}

fn piece_to_u8(piece: Piece) -> u8 {
    match piece {
        Piece::Empty => 0,
        Piece::Black => 1,
        Piece::White => 2,
    }
}
