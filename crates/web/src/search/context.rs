use js_sys::{Function, Object, Reflect};
use std::cmp::Reverse;
use std::rc::Rc;

use reversi_core::{
    bitboard::Bitboard,
    board::Board,
    empty_list::EmptyList,
    eval::pattern_feature::{PatternFeature, PatternFeatures},
    probcut::Selectivity,
    search::{root_move::RootMove, side_to_move::SideToMove},
    square::Square,
    types::{Depth, ScaledScore, Scoref},
};

use crate::transposition_table::TranspositionTable;

use crate::{eval::Eval, move_list::MoveList};
use wasm_bindgen::JsValue;

/// Maintains all mutable state during a single search operation.
pub struct SearchContext {
    /// Number of nodes searched so far.
    pub n_nodes: u64,
    /// Current side to move.
    pub side_to_move: SideToMove,
    /// Selectivity level.
    pub selectivity: Selectivity,
    /// List of empty squares, optimized for quick access.
    pub empty_list: EmptyList,
    /// Transposition table for storing search results.
    pub tt: Rc<TranspositionTable>,
    /// Neural network evaluator.
    pub eval: Rc<Eval>,
    /// Root moves being searched.
    pub root_moves: Vec<RootMove>,
    /// Pattern features for neural network input.
    pub pattern_features: PatternFeatures,
    /// Current PV index for Multi-PV search. Moves at indices < pv_idx are
    /// already finalized as earlier PV lines.
    pv_idx: usize,
    /// Optional callback for reporting progress to the JavaScript UI.
    progress_callback: Option<Function>,
}

impl SearchContext {
    /// Creates a new search context for the given board position.
    pub fn new(
        board: &Board,
        selectivity: Selectivity,
        tt: Rc<TranspositionTable>,
        eval: Rc<Eval>,
        progress_callback: Option<Function>,
    ) -> Self {
        let empty_list = EmptyList::new(board);
        let ply = empty_list.ply();

        SearchContext {
            n_nodes: 0,
            side_to_move: SideToMove::Player,
            selectivity,
            empty_list,
            tt,
            eval,
            root_moves: Self::create_root_moves(board),
            pattern_features: PatternFeatures::new(board, ply),
            pv_idx: 0,
            progress_callback,
        }
    }

    /// Switches the side to move.
    #[inline]
    fn switch_players(&mut self) {
        self.side_to_move = self.side_to_move.switch();
    }

    /// Updates state after making a midgame move.
    #[inline]
    pub fn update(&mut self, sq: Square, flipped: Bitboard) {
        self.increment_nodes();
        self.pattern_features
            .update(sq, flipped, self.ply(), self.side_to_move);
        self.switch_players();
        self.empty_list.remove(sq);
    }

    /// Undoes a midgame move.
    #[inline]
    pub fn undo(&mut self, sq: Square) {
        self.empty_list.restore(sq);
        self.switch_players();
    }

    /// Updates state after making an endgame move (skips pattern feature update).
    #[inline]
    pub fn update_endgame(&mut self, sq: Square) {
        self.increment_nodes();
        self.empty_list.remove(sq);
    }

    /// Undoes an endgame move.
    #[inline]
    pub fn undo_endgame(&mut self, sq: Square) {
        self.empty_list.restore(sq);
    }

    /// Updates state for a pass move.
    #[inline]
    pub fn update_pass(&mut self) {
        self.increment_nodes();
        self.switch_players();
    }

    /// Undoes a pass move.
    #[inline]
    pub fn undo_pass(&mut self) {
        self.switch_players();
    }

    /// Returns the current ply (number of moves played from the opening).
    #[inline]
    pub fn ply(&self) -> usize {
        self.empty_list.ply()
    }

    /// Increments the node counter.
    #[inline]
    pub fn increment_nodes(&mut self) {
        self.n_nodes += 1;
    }

    /// Returns the current pattern feature for neural network evaluation.
    #[inline]
    pub fn get_pattern_feature(&self) -> &PatternFeature {
        let ply = self.ply();
        if self.side_to_move == SideToMove::Player {
            self.pattern_features.p_feature(ply)
        } else {
            self.pattern_features.o_feature(ply)
        }
    }

    /// Updates a root move with its search score.
    pub fn update_root_move(
        &mut self,
        sq: Square,
        score: ScaledScore,
        move_count: usize,
        alpha: ScaledScore,
    ) {
        let is_pv = move_count == 1 || score > alpha;

        let rm = self.root_moves.iter_mut().find(|rm| rm.sq == sq).unwrap();
        rm.average_score = if rm.average_score == -ScaledScore::INF {
            score
        } else {
            (rm.average_score + score) / 2
        };

        if is_pv {
            rm.score = score;
        } else {
            rm.score = -ScaledScore::INF;
        }
    }

    /// Returns the root move with the highest score.
    pub fn get_best_root_move(&self) -> Option<RootMove> {
        self.root_moves.iter().max_by_key(|rm| rm.score).cloned()
    }

    /// Sets the current PV index for Multi-PV search.
    #[inline]
    pub fn set_pv_idx(&mut self, idx: usize) {
        self.pv_idx = idx;
    }

    /// Returns the current PV index.
    #[inline]
    pub fn pv_idx(&self) -> usize {
        self.pv_idx
    }

    /// Returns the root move at the current PV index, or [`None`] if out of bounds.
    ///
    /// The caller must sort the list beforehand for this to return the best
    /// move of the current PV line.
    pub fn current_pv_root_move(&self) -> Option<&RootMove> {
        self.root_moves.get(self.pv_idx)
    }

    /// Saves current scores as previous scores before starting a new iteration.
    pub fn save_previous_scores(&mut self) {
        for rm in self.root_moves.iter_mut() {
            rm.previous_score = rm.score;
        }
    }

    /// Sorts root moves from pv_idx to end by descending score (stable sort).
    pub fn sort_root_moves_from_pv_idx(&mut self) {
        if self.pv_idx < self.root_moves.len() {
            self.root_moves[self.pv_idx..].sort_by_key(|rm| Reverse(rm.score));
        }
    }

    /// Checks whether a move square exists in the remaining moves (from pv_idx onwards).
    pub fn root_move_in_pv_window(&self, sq: Square) -> bool {
        self.root_moves[self.pv_idx..].iter().any(|rm| rm.sq == sq)
    }

    /// Creates the initial list of root moves from the board's legal moves.
    fn create_root_moves(board: &Board) -> Vec<RootMove> {
        let move_list = MoveList::new(board);
        let mut root_moves = Vec::<RootMove>::with_capacity(move_list.count());
        for m in move_list.iter() {
            root_moves.push(RootMove::new(m.sq));
        }
        root_moves
    }

    /// Sends search progress to the JavaScript UI via the registered callback.
    pub fn notify_progress(
        &self,
        depth: Depth,
        score: Scoref,
        best_move: Square,
        selectivity: Selectivity,
    ) {
        let Some(callback) = &self.progress_callback else {
            return;
        };

        let payload = Object::new();
        let _ = Reflect::set(&payload, &JsValue::from_str("depth"), &JsValue::from(depth));
        let _ = Reflect::set(
            &payload,
            &JsValue::from_str("score"),
            &JsValue::from_f64(score as f64),
        );
        let _ = Reflect::set(
            &payload,
            &JsValue::from_str("probcut"),
            &JsValue::from_f64(selectivity.probability() as f64),
        );
        let _ = Reflect::set(
            &payload,
            &JsValue::from_str("nodes"),
            &JsValue::from_f64(self.n_nodes as f64),
        );
        let (best_move_value, best_move_index) = if best_move == Square::None {
            (JsValue::NULL, JsValue::NULL)
        } else {
            (
                JsValue::from(best_move.to_string()),
                JsValue::from_f64(best_move.index() as f64),
            )
        };
        let _ = Reflect::set(&payload, &JsValue::from_str("bestMove"), &best_move_value);
        let _ = Reflect::set(
            &payload,
            &JsValue::from_str("bestMoveIndex"),
            &best_move_index,
        );

        let _ = callback.call1(&JsValue::NULL, &payload);
    }
}
