use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::move_list::MoveList;
use crate::square::Square;
use crate::types::ScaledScore;

/// Represents a root move with its search results and statistics.
#[derive(Clone, Debug)]
pub struct RootMove {
    /// The move square
    pub sq: Square,
    /// Current best score for this move in the current iteration
    pub score: ScaledScore,
    /// Score from the previous iteration, used for aspiration windows
    pub previous_score: ScaledScore,
    /// Running average score across iterations (for stability analysis)
    pub average_score: ScaledScore,
    /// Principal variation line starting from this move
    pub pv: Vec<Square>,
}

impl RootMove {
    /// Creates a new RootMove for the given square.
    pub fn new(sq: Square) -> Self {
        Self {
            sq,
            score: -ScaledScore::INF,
            previous_score: -ScaledScore::INF,
            average_score: -ScaledScore::INF,
            pv: Vec::new(),
        }
    }
}

/// Thread-safe container for root moves during search.
///
/// This struct manages the list of moves being searched at the root position,
/// along with Multi-PV state. It can be cloned to share across threads via Arc.
#[derive(Clone)]
pub struct RootMoves {
    /// Shared list of root moves being searched (sorted by score after each PV line)
    moves: Arc<Mutex<Vec<RootMove>>>,
    /// Current PV index for Multi-PV search.
    /// Moves at indices < pv_idx are already part of earlier PV lines.
    pv_idx: Arc<AtomicUsize>,
}

impl RootMoves {
    /// Creates a new RootMoves container from the current board position.
    ///
    /// # Arguments
    /// * `board` - The current board position
    ///
    /// # Returns
    /// A new RootMoves with all legal moves initialized
    pub fn new(board: &Board) -> Self {
        let move_list = MoveList::new(board);
        let mut moves = Vec::<RootMove>::with_capacity(move_list.count());
        for m in move_list.iter() {
            moves.push(RootMove::new(m.sq));
        }
        Self {
            moves: Arc::new(Mutex::new(moves)),
            pv_idx: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Updates a root move with its search results.
    ///
    /// # Arguments
    /// * `sq` - The square of the root move being updated
    /// * `score` - The score returned from searching this move
    /// * `move_count` - Which move this is in the search order (1-based)
    /// * `alpha` - The current alpha bound
    /// * `pv` - The principal variation from the search stack
    pub fn update(
        &self,
        sq: Square,
        score: ScaledScore,
        move_count: usize,
        alpha: ScaledScore,
        pv: &[Square; MAX_PLY],
    ) {
        let is_pv = move_count == 1 || score > alpha;

        let mut moves = self.moves.lock().unwrap();
        let rm = moves.iter_mut().find(|rm| rm.sq == sq).unwrap();
        rm.average_score = if rm.average_score == -ScaledScore::INF {
            score
        } else {
            (rm.average_score + score) / 2
        };

        if is_pv {
            rm.score = score;
            rm.pv.clear();
            for sq in pv.iter() {
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
    pub fn get_current_pv(&self) -> Option<RootMove> {
        let moves = self.moves.lock().unwrap();
        moves.get(self.pv_idx()).cloned()
    }

    /// Gets the best root move (the one at index 0 after sorting).
    ///
    /// # Returns
    /// The best root move, or None if no moves exist
    pub fn get_best(&self) -> Option<RootMove> {
        let moves = self.moves.lock().unwrap();
        moves.first().cloned()
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
    pub fn save_previous_scores(&self) {
        let mut moves = self.moves.lock().unwrap();
        for rm in moves.iter_mut() {
            rm.previous_score = rm.score;
        }
    }

    /// Sorts root moves from pv_idx to end by score (stable sort).
    pub fn sort_from_pv_idx(&self) {
        let pv_idx = self.pv_idx();
        let mut moves = self.moves.lock().unwrap();
        if pv_idx < moves.len() {
            moves[pv_idx..].sort_by(|a, b| b.score.cmp(&a.score));
        }
    }

    /// Sorts all root moves by score for final result ordering.
    pub fn sort_all(&self) {
        let mut moves = self.moves.lock().unwrap();
        moves.sort_by(|a, b| b.score.cmp(&a.score));
    }

    /// Returns the number of root moves available.
    ///
    /// # Returns
    /// The count of legal moves from the root position
    pub fn count(&self) -> usize {
        self.moves.lock().unwrap().len()
    }

    /// Applies a function to all root moves and collects the results.
    ///
    /// # Arguments
    /// * `f` - A function to apply to each RootMove
    ///
    /// # Returns
    /// A vector containing the results of applying f to each move
    pub fn map<T, F>(&self, f: F) -> Vec<T>
    where
        F: FnMut(&RootMove) -> T,
    {
        let moves = self.moves.lock().unwrap();
        moves.iter().map(f).collect()
    }

    /// Checks if a move square exists in the remaining moves (from pv_idx onwards).
    ///
    /// # Arguments
    /// * `sq` - The square to check
    ///
    /// # Returns
    /// true if the move exists in moves[pv_idx..], false otherwise
    pub fn contains_from_pv_idx(&self, sq: Square) -> bool {
        let pv_idx = self.pv_idx();
        let moves = self.moves.lock().unwrap();
        moves.iter().skip(pv_idx).any(|rm| rm.sq == sq)
    }
}
