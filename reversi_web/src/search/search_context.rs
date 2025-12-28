use js_sys::{Function, Object, Reflect};
use std::rc::Rc;

use reversi_core::{
    board::Board,
    constants::{MAX_PLY, SCORE_INF},
    empty_list::EmptyList,
    eval::pattern_feature::{PatternFeature, PatternFeatures},
    search::{root_move::RootMove, search_context::StackRecord, side_to_move::SideToMove},
    square::Square,
    transposition_table::TranspositionTable,
    types::{Depth, Score, Scoref, Selectivity},
};

use crate::{eval::Eval, move_list::MoveList};
use wasm_bindgen::JsValue;

/// The search context that maintains all state during search operations.
pub struct SearchContext {
    /// Number of nodes searched in this context
    pub n_nodes: u64,
    /// Current side to move
    pub side_to_move: SideToMove,
    /// Transposition table generation counter for aging entries
    pub generation: u8,
    /// Selectivity level
    pub selectivity: Selectivity,
    /// List of empty squares on the board, optimized for quick access
    pub empty_list: EmptyList,
    /// Transposition table for storing search results
    pub tt: Rc<TranspositionTable>,
    /// Neural network evaluator for position assessment
    pub eval: Rc<Eval>,
    /// List of root moves being searched
    pub root_moves: Vec<RootMove>,
    /// Pattern features for efficient neural network input
    pub pattern_features: PatternFeatures,
    /// Optional callback for reporting progress back to JavaScript UI
    progress_callback: Option<Function>,
    /// Search stack for maintaining PV and search state at each ply
    stack: [StackRecord; MAX_PLY],
}

impl SearchContext {
    /// Creates a new search context for the given board position.
    ///
    /// # Arguments
    /// * `board` - The current board position
    /// * `generation` - Transposition table generation for aging
    /// * `selectivity` - Selectivity level
    /// * `tt` - Transposition table
    /// * `eval` - Neural network evaluator
    /// * `progress_callback` - Optional JavaScript callback for reporting progress
    ///
    /// # Returns
    /// A new SearchContext ready for search operations
    pub fn new(
        board: &Board,
        generation: u8,
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
            generation,
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

    /// Switches the side to move in this context.
    #[inline]
    fn switch_players(&mut self) {
        self.side_to_move = self.side_to_move.switch();
    }

    /// Updates the search context after making a move in midgame search.
    ///
    /// # Arguments
    /// * `sq` - The square where the move is played
    /// * `flipped` - The number of pieces flipped by the move
    #[inline]
    pub fn update(&mut self, sq: Square, flipped: u64) {
        self.increment_nodes();
        self.pattern_features
            .update(sq, flipped, self.ply(), self.side_to_move);
        self.switch_players();
        self.empty_list.remove(sq);
    }

    /// Undoes a move in the search context.
    ///
    /// # Arguments
    /// * `sq` - The square where the move is played
    #[inline]
    pub fn undo(&mut self, sq: Square) {
        self.empty_list.restore(sq);
        self.switch_players();
    }

    /// Updates the context for an endgame move where pattern features aren't used.
    ///
    /// # Arguments
    /// * `sq` - The square where the move is played
    #[inline]
    pub fn update_endgame(&mut self, sq: Square) {
        self.increment_nodes();
        self.empty_list.remove(sq);
    }

    /// Undoes an endgame move by restoring the played square.
    ///
    /// # Arguments
    /// * `sq` - The square to restore to the empty list
    #[inline]
    pub fn undo_endgame(&mut self, sq: Square) {
        self.empty_list.restore(sq);
    }

    /// Updates the context when a pass move is made.
    #[inline]
    pub fn update_pass(&mut self) {
        self.increment_nodes();
        self.switch_players();
    }

    /// Undoes a pass move by switching back the side to move.
    #[inline]
    pub fn undo_pass(&mut self) {
        self.switch_players();
    }

    /// Returns the current ply in the search tree.
    ///
    /// The ply is calculated from the number of empty squares remaining,
    /// representing how far we are from the start of the game.
    ///
    /// # Returns
    /// Current search depth/ply
    #[inline]
    pub fn ply(&self) -> usize {
        self.empty_list.ply()
    }

    /// Increments the node counter for search statistics.
    #[inline]
    pub fn increment_nodes(&mut self) {
        self.n_nodes += 1;
    }

    /// Gets the current pattern feature for neural network evaluation.
    ///
    /// # Returns
    /// Reference to the current pattern feature for the position
    #[inline]
    pub fn get_pattern_feature(&self) -> &PatternFeature {
        let ply = self.ply();
        if self.side_to_move == SideToMove::Player {
            &self.pattern_features.p_features[ply]
        } else {
            &self.pattern_features.o_features[ply]
        }
    }

    /// Updates a root move with its search results.
    ///
    /// # Arguments
    /// * `sq` - The square of the root move being updated
    /// * `score` - The score returned from searching this move
    /// * `move_count` - Which move this is in the search order (1-based)
    /// * `alpha` - The current alpha bound
    pub fn update_root_move(&mut self, sq: Square, score: Score, move_count: usize, alpha: Score) {
        let is_pv = move_count == 1 || score > alpha;
        if is_pv {
            self.update_pv(sq);
        }

        let ply = self.ply();
        let rm = self.root_moves.iter_mut().find(|rm| rm.sq == sq).unwrap();
        rm.average_score = if rm.average_score == -SCORE_INF {
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
            rm.score = -SCORE_INF;
        }
    }

    /// Gets the best root move (the one with highest score).
    ///
    /// # Returns
    /// The best root move, or None if no moves exist
    pub fn get_best_root_move(&self) -> Option<RootMove> {
        self.root_moves.iter().max_by_key(|rm| rm.score).cloned()
    }

    /// Creates the initial list of root moves from the current board position.
    ///
    /// # Arguments
    /// * `board` - The current board position
    ///
    /// # Returns
    /// Vector of initialized root moves
    fn create_root_moves(board: &Board) -> Vec<RootMove> {
        let move_list = MoveList::new(board);
        let mut root_moves = Vec::<RootMove>::with_capacity(move_list.count());
        for m in move_list.iter() {
            root_moves.push(RootMove::new(m.sq));
        }
        root_moves
    }

    /// Updates the principal variation at the current ply.
    ///
    /// # Arguments
    /// * `sq` - The best move at the current ply
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

    /// Notifies the UI of search progress through the registered callback.
    ///
    /// # Arguments
    /// * `depth` - Current search depth
    /// * `score` - Current best score (from engine's perspective)
    /// * `best_move` - Current best move
    /// * `selectivity` - Current selectivity level
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
