use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::empty_list::EmptyList;
use crate::eval::Eval;
use crate::eval::pattern_feature::{PatternFeature, PatternFeatures};
use crate::move_list::MoveList;
use crate::search::SearchProgress;
use crate::search::SearchProgressCallback;
use crate::search::root_move::RootMove;
use crate::search::side_to_move::SideToMove;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, ScaledScore, Scoref, Selectivity};

/// Represents the current phase of the game for search strategy selection.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    /// Middle game phase using the regular neural network model
    MidGame,
    /// End game phase using the smaller neural network model and perfect search
    EndGame,
}

/// A record stored for each ply in the search stack.
#[derive(Clone, Copy)]
pub struct StackRecord {
    /// Principal variation line from this ply to the end of search
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
    /// Shared list of root moves being searched (sorted by score after each PV line)
    pub root_moves: Arc<std::sync::Mutex<Vec<RootMove>>>,
    /// Current PV index for Multi-PV search
    /// Moves at indices < pv_idx are already part of earlier PV lines
    pub pv_idx: Arc<AtomicUsize>,
    /// Neural network evaluator for position assessment
    pub eval: Arc<Eval>,
    /// Pattern features for efficient neural network input
    pub pattern_features: PatternFeatures,
    /// Optional callback for reporting search progress to UI
    pub callback: Option<Arc<SearchProgressCallback>>,
    /// Search stack for maintaining PV and search state at each ply
    stack: [StackRecord; MAX_PLY],
    /// Current phase of the game (midgame vs endgame)
    pub game_phase: GamePhase,
}

impl SearchContext {
    /// Creates a new search context for the given board position.
    ///
    /// # Arguments
    /// * `board` - The current board position
    /// * `selectivity` - Selectivity level
    /// * `tt` - Shared transposition table
    /// * `eval` - Neural network evaluator
    ///
    /// # Returns
    /// A new SearchContext ready for search operations
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
            root_moves: Arc::new(std::sync::Mutex::new(Self::create_root_moves(board))),
            pv_idx: Arc::new(AtomicUsize::new(0)),
            eval,
            pattern_features: PatternFeatures::new(board, ply),
            callback: None,
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
            game_phase: GamePhase::MidGame,
        }
    }

    /// Creates a new search context from a parallel search split point.
    ///
    /// # Arguments
    /// * `sp` - The split point containing the search task to inherit from
    ///
    /// # Returns
    /// A new SearchContext configured for parallel search from the split point
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
            pv_idx: task.pv_idx.clone(),
            eval: task.eval.clone(),
            pattern_features,
            callback: None,
            stack: [StackRecord {
                pv: [Square::None; MAX_PLY],
            }; MAX_PLY],
            game_phase: task.game_phase,
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
    /// * `flipped` - The flipped pieces as a bitboard
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
    /// * `sq` - The square to undo
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

        let mut root_moves = self.root_moves.lock().unwrap();
        let rm = root_moves.iter_mut().find(|rm| rm.sq == sq).unwrap();
        rm.average_score = if rm.average_score == -ScaledScore::INF {
            score
        } else {
            (rm.average_score + score) / 2
        };

        if is_pv {
            let ply = self.ply();
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

    /// Gets the root move at the current PV index.
    ///
    /// In Multi-PV mode, this returns the move at the current PV position
    /// which should be searched next.
    ///
    /// # Returns
    /// The root move at the current PV index, or None if index is out of bounds
    pub fn get_current_pv_root_move(&self) -> Option<RootMove> {
        let root_moves = self.root_moves.lock().unwrap();
        root_moves.get(self.pv_idx()).cloned()
    }

    /// Gets the best root move (the one at index 0 after sorting).
    ///
    /// # Returns
    /// The best root move, or None if no moves exist
    pub fn get_best_root_move(&self) -> Option<RootMove> {
        let root_moves = self.root_moves.lock().unwrap();
        root_moves.first().cloned()
    }

    /// Sets the current PV index for Multi-PV search.
    ///
    /// # Arguments
    /// * `idx` - The new PV index
    pub fn set_pv_idx(&self, idx: usize) {
        self.pv_idx.store(idx, Ordering::Relaxed);
    }

    /// Returns the current PV index.
    #[inline]
    pub fn pv_idx(&self) -> usize {
        self.pv_idx.load(Ordering::Relaxed)
    }

    /// Saves current scores as previous scores before starting a new iteration.
    ///
    /// This is called at the beginning of each iterative deepening iteration
    /// to preserve scores for aspiration window calculation.
    pub fn save_previous_scores(&mut self) {
        let mut root_moves = self.root_moves.lock().unwrap();
        for rm in root_moves.iter_mut() {
            rm.previous_score = rm.score;
        }
    }

    /// Sorts root moves from pv_idx to end by score (stable sort).
    pub fn sort_root_moves_from_pv_idx(&self) {
        let pv_idx = self.pv_idx();
        let mut root_moves = self.root_moves.lock().unwrap();
        if pv_idx < root_moves.len() {
            root_moves[pv_idx..].sort_by(|a, b| b.score.cmp(&a.score));
        }
    }

    /// Sorts all root moves by score for final result ordering.
    pub fn sort_all_root_moves(&self) {
        let mut root_moves = self.root_moves.lock().unwrap();
        root_moves.sort_by(|a, b| b.score.cmp(&a.score));
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

    /// Gets the principal variation at the current ply.
    pub fn get_pv(&self) -> &[Square; MAX_PLY] {
        &self.stack[self.ply()].pv
    }

    /// Sets the principal variation at the current ply.
    pub fn set_pv(&mut self, pv: &[Square; MAX_PLY]) {
        self.stack[self.ply()].pv.copy_from_slice(pv);
    }

    /// The callback is invoked periodically during search to report progress
    /// to the UI, including current depth, best move, and evaluation.
    ///
    /// # Arguments
    /// * `callback` - The progress callback function
    pub fn set_callback(&mut self, callback: Arc<SearchProgressCallback>) {
        self.callback = Some(callback);
    }

    /// Notifies the UI of search progress through the registered callback.
    ///
    /// # Arguments
    /// * `depth` - Current search depth
    /// * `target_depth` - Target search depth (max_depth for midgame, n_empties for endgame)
    /// * `score` - Current best score (from engine's perspective)
    /// * `best_move` - Current best move
    /// * `selectivity` - Current selectivity level
    /// * `nodes` - Number of nodes searched
    /// * `pv_line` - Principal variation line
    #[allow(clippy::too_many_arguments)]
    pub fn notify_progress(
        &self,
        depth: Depth,
        target_depth: Depth,
        score: Scoref,
        best_move: Square,
        selectivity: Selectivity,
        nodes: u64,
        pv_line: Vec<Square>,
    ) {
        if let Some(ref callback) = self.callback {
            callback(SearchProgress {
                depth,
                target_depth,
                score,
                best_move,
                probability: selectivity.probability(),
                nodes,
                pv_line,
                game_phase: self.game_phase,
            });
        }
    }

    /// Returns the number of root moves available from the current position.
    ///
    /// # Returns
    /// The count of legal moves from the root position
    pub fn root_moves_count(&self) -> usize {
        self.root_moves.lock().unwrap().len()
    }
}
