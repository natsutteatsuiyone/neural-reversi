//! Search context for maintaining state during game tree search.

use std::sync::Arc;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::empty_list::EmptyList;
use crate::eval::Eval;
use crate::eval::EvalMode;
use crate::eval::pattern_feature::{PatternFeature, PatternFeatures};
use crate::probcut::Selectivity;
use crate::search::root_move::{RootMove, RootMoves};
use crate::search::side_to_move::SideToMove;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::ScaledScore;

/// A record stored for each ply in the search stack.
#[derive(Clone, Copy)]
pub struct StackRecord {
    /// Principal variation line from this ply to the end of search.
    pub pv: [Square; MAX_PLY],
}

/// The search context that maintains all state during search operations.
pub struct SearchContext {
    /// Number of nodes searched in this context
    pub n_nodes: u64,
    /// Current side to move
    pub side_to_move: SideToMove,
    /// Selectivity level
    pub selectivity: Selectivity,
    /// List of empty squares on the board, optimized for quick access
    pub empty_list: EmptyList,
    /// Shared transposition table for storing search results
    pub tt: Arc<TranspositionTable>,
    /// Container for root moves and Multi-PV state
    pub root_moves: RootMoves,
    /// Neural network evaluator for position assessment
    pub eval: Arc<Eval>,
    /// Pattern features for efficient neural network input
    pub pattern_features: PatternFeatures,
    /// Search stack for maintaining PV and search state at each ply
    stack: [StackRecord; MAX_PLY],
    /// Current evaluation mode (midgame vs endgame).
    pub eval_mode: EvalMode,
}

impl SearchContext {
    /// Creates a new search context for the given board position.
    ///
    /// # Arguments
    /// * `board` - Current board position.
    /// * `selectivity` - Selectivity level.
    /// * `tt` - Transposition table.
    /// * `eval` - Neural network evaluator.
    pub fn new(
        board: &Board,
        selectivity: Selectivity,
        tt: Arc<TranspositionTable>,
        eval: Arc<Eval>,
    ) -> SearchContext {
        let empty_list = EmptyList::new(board);
        let ply = empty_list.ply();
        SearchContext {
            n_nodes: 0,
            side_to_move: SideToMove::Player,
            selectivity,
            empty_list,
            tt,
            root_moves: RootMoves::new(board),
            eval,
            pattern_features: PatternFeatures::new(board, ply),
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
            eval_mode: EvalMode::Main,
        }
    }

    /// Creates a new search context from a parallel search split point.
    ///
    /// # Arguments
    /// * `sp` - Split point reference.
    #[inline]
    pub fn from_split_point(sp: &Arc<SplitPoint>) -> SearchContext {
        let state = sp.state();
        let task = state.task.as_ref().unwrap();
        let empty_list = task.empty_list.clone();
        let ply = empty_list.ply();
        let pattern_features = if task.side_to_move == SideToMove::Player {
            PatternFeatures::new(&task.board, ply)
        } else {
            PatternFeatures::new(&task.board.switch_players(), ply)
        };
        SearchContext {
            n_nodes: 0,
            side_to_move: task.side_to_move,
            empty_list,
            selectivity: task.selectivity,
            tt: task.tt.clone(),
            root_moves: task.root_moves.clone(),
            eval: task.eval.clone(),
            pattern_features,
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
            eval_mode: task.eval_mode,
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
    /// * `sq` - Move square.
    /// * `flipped` - Flipped discs bitboard.
    #[inline]
    pub fn update(&mut self, sq: Square, flipped: Bitboard) {
        self.increment_nodes();
        self.pattern_features
            .update(sq, flipped, self.ply(), self.side_to_move);
        self.switch_players();
        self.empty_list.remove(sq);
    }

    /// Undoes a move in the search context.
    ///
    /// # Arguments
    /// * `sq` - Move square to restore.
    #[inline]
    pub fn undo(&mut self, sq: Square) {
        self.empty_list.restore(sq);
        self.switch_players();
    }

    /// Updates the context for an endgame move (no pattern feature update).
    ///
    /// # Arguments
    /// * `sq` - Move square.
    #[inline]
    pub fn update_endgame(&mut self, sq: Square) {
        self.increment_nodes();
        self.empty_list.remove(sq);
    }

    /// Undoes an endgame move by restoring the played square.
    ///
    /// # Arguments
    /// * `sq` - Move square to restore.
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
    /// Current search depth/ply.
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
    /// Reference to the current pattern feature.
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
    /// * `sq` - Root move square.
    /// * `score` - Search score.
    /// * `move_count` - Move index in search order (1-based).
    /// * `alpha` - Alpha bound.
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

        let pv = self.stack[self.ply()].pv;
        self.root_moves.update(sq, score, move_count, alpha, &pv);
    }

    /// Gets the root move at the current PV index.
    ///
    /// In Multi-PV mode, this returns the move at the current PV position
    /// which should be searched next.
    ///
    /// # Returns
    /// Root move at current PV index, or None if out of bounds.
    pub fn get_current_pv_root_move(&self) -> Option<RootMove> {
        self.root_moves.get_current_pv()
    }

    /// Gets the best root move (the one at index 0 after sorting).
    ///
    /// # Returns
    /// Best root move, or None if no moves exist.
    pub fn get_best_root_move(&self) -> Option<RootMove> {
        self.root_moves.get_best()
    }

    /// Sets the current PV index for Multi-PV search.
    ///
    /// # Arguments
    /// * `idx` - PV index.
    pub fn set_pv_idx(&self, idx: usize) {
        self.root_moves.set_pv_idx(idx);
    }

    /// Returns the current PV index.
    #[inline]
    pub fn pv_idx(&self) -> usize {
        self.root_moves.pv_idx()
    }

    /// Saves current scores as previous scores before starting a new iteration.
    ///
    /// This is called at the beginning of each iterative deepening iteration
    /// to preserve scores for aspiration window calculation.
    pub fn save_previous_scores(&self) {
        self.root_moves.save_previous_scores();
    }

    /// Sorts root moves from pv_idx to end by score (stable sort).
    pub fn sort_root_moves_from_pv_idx(&self) {
        self.root_moves.sort_from_pv_idx();
    }

    /// Sorts all root moves by score for final result ordering.
    pub fn sort_all_root_moves(&self) {
        self.root_moves.sort_all();
    }

    /// Updates the principal variation at the current ply.
    ///
    /// # Arguments
    /// * `sq` - Best move at current ply.
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

    /// Gets the principal variation at the current ply.
    pub fn get_pv(&self) -> &[Square; MAX_PLY] {
        &self.stack[self.ply()].pv
    }

    /// Sets the principal variation at the current ply.
    pub fn set_pv(&mut self, pv: &[Square; MAX_PLY]) {
        self.stack[self.ply()].pv.copy_from_slice(pv);
    }

    /// Returns the number of root moves available from the current position.
    ///
    /// # Returns
    /// Count of legal moves from root position.
    pub fn root_moves_count(&self) -> usize {
        self.root_moves.count()
    }
}
