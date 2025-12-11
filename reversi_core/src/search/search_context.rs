use std::sync::Arc;

use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::constants::SCORE_INF;
use crate::empty_list::EmptyList;
use crate::eval::Eval;
use crate::eval::pattern_feature::{PatternFeature, PatternFeatures};
use crate::move_list::{Move, MoveList};
use crate::search::SearchProgress;
use crate::search::SearchProgressCallback;
use crate::search::root_move::RootMove;
use crate::search::side_to_move::SideToMove;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, Score, Scoref, Selectivity};

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
    /// Transposition table generation counter for aging entries
    pub generation: u8,
    /// Selectivity level
    pub selectivity: Selectivity,
    /// List of empty squares on the board, optimized for quick access
    pub empty_list: EmptyList,
    /// Shared transposition table for storing search results
    pub tt: Arc<TranspositionTable>,
    /// Shared list of root moves being searched
    pub root_moves: Arc<std::sync::Mutex<Vec<RootMove>>>,
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
    /// * `generation` - Transposition table generation for aging
    /// * `selectivity` - Selectivity level
    /// * `tt` - Shared transposition table
    /// * `eval` - Neural network evaluator
    ///
    /// # Returns
    /// A new SearchContext ready for search operations
    pub fn new(
        board: &Board,
        generation: u8,
        selectivity: Selectivity,
        tt: Arc<TranspositionTable>,
        eval: Arc<Eval>,
    ) -> SearchContext {
        let empty_list = EmptyList::new(board);
        let ply = empty_list.ply();
        SearchContext {
            n_nodes: 0,
            side_to_move: SideToMove::Player,
            generation,
            selectivity,
            empty_list,
            tt,
            root_moves: Arc::new(std::sync::Mutex::new(Self::create_root_moves(board))),
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
            generation: task.generation,
            selectivity: task.selectivity,
            tt: task.tt.clone(),
            root_moves: task.root_moves.clone(),
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
    /// * `mv` - The move being made, including square and flipped pieces
    #[inline]
    pub fn update(&mut self, mv: &Move) {
        self.increment_nodes();
        self.pattern_features
            .update(mv.sq, mv.flipped, self.ply(), self.side_to_move);
        self.switch_players();
        self.empty_list.remove(mv.sq);
    }

    /// Undoes a move in the search context.
    ///
    /// # Arguments
    /// * `mv` - The move to undo
    #[inline]
    pub fn undo(&mut self, mv: &Move) {
        self.empty_list.restore(mv.sq);
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

        let mut root_moves = self.root_moves.lock().unwrap();
        let rm = root_moves.iter_mut().find(|rm| rm.sq == sq).unwrap();
        rm.average_score = if rm.average_score == -SCORE_INF {
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
            rm.score = -SCORE_INF;
        }
    }

    /// Gets the best root move based on current search results.
    ///
    /// # Arguments
    /// * `skip_seached_move` - If true, only consider unsearched moves
    ///
    /// # Returns
    /// The best root move, or None if no moves match the criteria
    pub fn get_best_root_move(&self, skip_seached_move: bool) -> Option<RootMove> {
        let root_moves = self
            .root_moves
            .lock()
            .expect("Failed to acquire lock on root_moves");
        if skip_seached_move {
            root_moves
                .iter()
                .filter(|rm| !rm.searched)
                .max_by_key(|rm| rm.score)
                .cloned()
        } else {
            root_moves.iter().max_by_key(|rm| rm.score).cloned()
        }
    }

    /// Marks a root move as having been searched.
    ///
    /// # Arguments
    /// * `sq` - The square of the root move to mark as searched
    pub fn mark_root_move_searched(&mut self, sq: Square) {
        let mut root_moves = self.root_moves.lock().unwrap();
        if let Some(rm) = root_moves.iter_mut().find(|rm| rm.sq == sq) {
            rm.searched = true;
        }
    }

    /// Resets the searched flag for all root moves.
    pub fn reset_root_move_searched(&mut self) {
        let mut root_moves = self.root_moves.lock().unwrap();
        for rm in root_moves.iter_mut() {
            rm.searched = false;
        }
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
            root_moves.push(RootMove {
                sq: m.sq,
                score: -SCORE_INF,
                average_score: -SCORE_INF,
                pv: Vec::new(),
                searched: false,
            });
        }
        root_moves
    }

    /// Checks if a specific root move has been searched.
    ///
    /// # Arguments
    /// * `sq` - The square to check
    ///
    /// # Returns
    /// True if the move has been searched, false otherwise
    pub fn is_move_searched(&self, sq: Square) -> bool {
        let root_moves = self.root_moves.lock().unwrap();
        root_moves.iter().any(|rm| rm.sq == sq && rm.searched)
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
        if let Some(ref callback) = self.callback {
            callback(SearchProgress {
                depth,
                score,
                best_move,
                probability: selectivity.probability(),
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
