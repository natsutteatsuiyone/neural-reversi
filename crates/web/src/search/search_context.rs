use js_sys::{Function, Object, Reflect};
use std::rc::Rc;

use reversi_core::{
    bitboard::Bitboard,
    board::Board,
    constants::MAX_PLY,
    empty_list::EmptyList,
    eval::pattern_feature::{PatternFeature, PatternFeatures},
    probcut::Selectivity,
    search::{root_move::RootMove, search_context::StackRecord, side_to_move::SideToMove},
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
    /// Optional callback for reporting progress to the JavaScript UI.
    progress_callback: Option<Function>,
    /// Search stack for PV and state at each ply.
    stack: [StackRecord; MAX_PLY],
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
            progress_callback,
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
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

    /// Updates a root move with its search score and PV.
    pub fn update_root_move(
        &mut self,
        sq: Square,
        score: ScaledScore,
        move_count: usize,
        alpha: ScaledScore,
    ) {
        let is_pv = move_count == 1 || score > alpha;
        if is_pv {
            self.update_pv(sq);
        }

        let ply = self.ply();
        let rm = self.root_moves.iter_mut().find(|rm| rm.sq == sq).unwrap();
        rm.average_score = if rm.average_score == -ScaledScore::INF {
            score
        } else {
            (rm.average_score + score) / 2
        };

        if is_pv {
            rm.score = score;
            rm.pv.clear();
            for sq in self.stack[ply].pv.iter() {
                if *sq == Square::None {
                    break;
                }
                rm.pv.push(*sq);
            }
        } else {
            rm.score = -ScaledScore::INF;
        }
    }

    /// Returns the root move with the highest score.
    pub fn get_best_root_move(&self) -> Option<RootMove> {
        self.root_moves.iter().max_by_key(|rm| rm.score).cloned()
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

    /// Updates the principal variation at the current ply.
    pub fn update_pv(&mut self, sq: Square) {
        let ply = self.ply();
        self.stack[ply].pv[0] = sq;
        if ply == 0 {
            return;
        }
        let mut idx = 0;
        while idx < self.stack[ply + 1].pv.len() && self.stack[ply + 1].pv[idx] != Square::None {
            self.stack[ply].pv[idx + 1] = self.stack[ply + 1].pv[idx];
            idx += 1;
        }
        self.stack[ply].pv[idx + 1] = Square::None;
    }

    /// Clears the principal variation at the current ply.
    pub fn clear_pv(&mut self) {
        self.stack[self.ply()].pv.fill(Square::None);
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
        let best_move_value = if best_move == Square::None {
            JsValue::NULL
        } else {
            JsValue::from(best_move.to_string())
        };
        let _ = Reflect::set(&payload, &JsValue::from_str("bestMove"), &best_move_value);

        let _ = callback.call1(&JsValue::NULL, &payload);
    }
}
