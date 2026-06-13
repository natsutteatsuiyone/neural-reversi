use js_sys::{Function, Object, Reflect};
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::move_list::MoveList;
use reversi_core::square::{Square, TOTAL_SQUARES};
use reversi_core::types::Depth;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::config::{
    DEFAULT_MID_DEPTH, DEFAULT_TT_MB, MAX_MID_DEPTH, MIDGAME_SELECTIVITY, MIN_MID_DEPTH,
};
use crate::eval::Eval;
use crate::level::Level;
use crate::search::Search;
use crate::transposition_table::TranspositionTable;

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

/// Game session exposed to JavaScript that pairs a board with the AI engine.
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
    /// Creates a new game with the human playing the given color.
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

    /// Makes a move without checking whose turn it is (for replay purposes).
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

    /// Evaluates all legal moves for the human player and returns
    /// `[{ move: u8, score: f32 }]`. Streams per-move progress via `callback`.
    pub fn hint(&mut self, callback: Option<Function>) -> JsValue {
        let scores = self.hint_scores(callback);
        let arr = js_sys::Array::new();
        for (sq, score) in scores {
            let obj = Object::new();
            let _ = Reflect::set(
                &obj,
                &JsValue::from_str("move"),
                &JsValue::from_f64(sq.index() as f64),
            );
            let _ = Reflect::set(
                &obj,
                &JsValue::from_str("score"),
                &JsValue::from_f64(score as f64),
            );
            arr.push(&obj);
        }
        arr.into()
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
            Disc::Black => (self.board.player().bits(), self.board.opponent().bits()),
            Disc::White => (self.board.opponent().bits(), self.board.player().bits()),
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

    /// Evaluates every legal move for the human player. Does not mutate the board.
    fn hint_scores(&mut self, callback: Option<Function>) -> Vec<(Square, f32)> {
        if self.board.is_game_over()
            || self.current_player != self.human_player
            || !self.board.has_legal_moves()
        {
            return Vec::new();
        }

        let level = level_for_position(self.mid_depth);
        self.engine
            .search
            .run_multi_pv(&self.board, level, MIDGAME_SELECTIVITY, callback)
            .multi_pv_scores
    }
}

fn level_for_position(mid_depth: Depth) -> Level {
    let end_depth = ((mid_depth as f64 * 1.2).round() as Depth).min(26);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_initial_state(game: &Game) {
        assert_eq!(game.score(), vec![2, 2]);
        assert_eq!(game.empty_count(), 60);
        assert_eq!(game.current_player(), 1);
        assert!(!game.is_game_over());

        let mut moves = game.legal_moves();
        moves.sort_unstable();
        assert_eq!(moves, vec![19, 26, 37, 44]);
    }

    fn assert_conservation(game: &Game) {
        let score = game.score();
        let occupied = u16::from(score[0]) + u16::from(score[1]);
        assert_eq!(
            occupied + u16::from(game.empty_count()),
            TOTAL_SQUARES as u16
        );
    }

    #[test]
    fn new_game_has_standard_opening_state() {
        let game = Game::new(true);

        assert_initial_state(&game);
    }

    #[test]
    fn human_move_rejected_when_not_humans_turn() {
        let mut game = Game::new(false);

        assert!(!game.human_move(19));
        assert_eq!(game.score(), vec![2, 2]);
        assert_eq!(game.current_player(), 1);
    }

    #[test]
    fn human_move_rejects_illegal_squares() {
        let mut game = Game::new(true);

        assert!(!game.human_move(0));
        assert!(!game.human_move(64));
        assert!(!game.human_move(255));
        assert_initial_state(&game);
    }

    #[test]
    fn pass_rejected_while_legal_moves_exist() {
        let mut game = Game::new(true);

        assert!(!game.pass());
        assert_eq!(game.current_player(), 1);
    }

    #[test]
    fn human_move_flips_discs_and_switches_turn() {
        let mut game = Game::new(true);

        assert!(game.human_move(19));
        assert_eq!(game.score(), vec![4, 1]);
        assert_eq!(game.current_player(), 2);
        assert_eq!(game.empty_count(), 59);
        assert_eq!(game.board()[19], 1);
    }

    #[test]
    fn replaying_same_moves_reproduces_identical_state() {
        let mut game_a = Game::new(true);
        let mut replayed_moves = Vec::new();

        for _ in 0..12 {
            if game_a.is_game_over() {
                break;
            }
            let moves = game_a.legal_moves();
            if moves.is_empty() {
                break;
            }
            let next_move = moves[0];
            assert!(game_a.make_move_unchecked(next_move));
            replayed_moves.push(next_move);
            assert_conservation(&game_a);
        }

        let mut game_b = Game::new(true);
        for mv in replayed_moves {
            assert!(game_b.make_move_unchecked(mv));
        }

        assert_eq!(game_a.board(), game_b.board());
        assert_eq!(game_a.score(), game_b.score());
        assert_eq!(game_a.current_player(), game_b.current_player());
    }

    #[test]
    fn greedy_full_game_terminates_and_is_consistent() {
        let mut game = Game::new(true);

        for _ in 0..70 {
            if game.is_game_over() {
                break;
            }
            let moves = game.legal_moves();
            assert!(!moves.is_empty());
            assert!(game.make_move_unchecked(moves[0]));
        }

        assert!(game.is_game_over());
        assert_conservation(&game);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut game = Game::new(true);

        assert!(game.make_move_unchecked(19));
        assert!(game.make_move_unchecked(game.legal_moves()[0]));
        game.reset(true);

        assert_initial_state(&game);
    }

    #[test]
    fn ai_move_plays_a_legal_move() {
        let mut game = Game::new(false);
        game.set_level(1);
        let legal = game.legal_moves();

        let mv = game.ai_move();

        let mv = mv.expect("AI should choose a legal opening move");
        assert!(legal.contains(&mv));
        assert_eq!(game.current_player(), 2);
        assert_eq!(game.empty_count(), 59);
    }

    #[test]
    fn hint_scores_returns_an_entry_per_legal_move_and_keeps_board_intact() {
        let mut game = Game::new(true);
        game.set_level(1);
        let board_before = game.board();
        let legal: Vec<u8> = game.legal_moves();

        let hints = game.hint_scores(None);

        assert_eq!(hints.len(), legal.len());
        for (sq, score) in &hints {
            assert!(legal.contains(&(sq.index() as u8)));
            assert!(score.is_finite());
        }
        assert_eq!(game.board(), board_before);
        assert_eq!(game.legal_moves(), legal);
    }

    #[test]
    fn hint_scores_is_empty_when_not_humans_turn() {
        let mut game = Game::new(false);
        game.set_level(1);

        assert!(game.hint_scores(None).is_empty());
    }

    #[test]
    fn hint_results_are_sorted_best_first() {
        let mut game = Game::new(true);
        game.set_level(2);

        let hints = game.hint_scores(None);

        let best = hints
            .iter()
            .cloned()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(hints[0].1, best.1);
    }

    #[test]
    fn hint_scores_covers_endgame_path() {
        let mut game = Game::new(true);
        game.set_level(10);

        // Greedily play moves until the human is to move with few empties,
        // so the position routes through the endgame Multi-PV driver.
        loop {
            if game.is_game_over() {
                panic!("greedy line no longer reaches an endgame hint position");
            }
            if game.empty_count() <= 12 && game.current_player() == game.human_color() {
                break;
            }
            let moves = game.legal_moves();
            assert!(!moves.is_empty());
            assert!(game.make_move_unchecked(moves[0]));
        }

        let legal = game.legal_moves();
        let hints = game.hint_scores(None);

        assert_eq!(hints.len(), legal.len());
        for (sq, score) in &hints {
            assert!(legal.contains(&(sq.index() as u8)));
            assert!(score.is_finite());
        }
    }
}
