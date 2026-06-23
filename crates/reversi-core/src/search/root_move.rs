//! Root move management.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::move_list::MoveList;
use crate::square::Square;
use crate::types::ScaledScore;

/// A root move with its search results and statistics.
#[derive(Clone, Debug)]
pub struct RootMove {
    /// The move square.
    pub sq: Square,
    /// Current best score for this move in the current iteration.
    pub score: ScaledScore,
    /// Score from the previous iteration, used for aspiration windows.
    pub previous_score: ScaledScore,
    /// Running average score across iterations (for stability analysis).
    pub average_score: ScaledScore,
    /// Principal variation line starting from this move.
    pub pv: Vec<Square>,
}

impl RootMove {
    /// Creates a new root move for the given square.
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
    /// Shared list of root moves being searched.
    moves: Arc<Mutex<Vec<RootMove>>>,
    /// Current PV index for Multi-PV search.
    /// Moves at indices < pv_idx are already part of earlier PV lines.
    pv_idx: Arc<AtomicUsize>,
}

impl RootMoves {
    /// Creates a new root moves container from the current board position.
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
    /// # Panics
    ///
    /// Panics if `sq` is not found in the root move list.
    pub fn update(&self, sq: Square, score: ScaledScore, is_pv: bool, pv: &[Square; MAX_PLY]) {
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

    /// Returns the root move at the current PV index, or [`None`] if out of bounds.
    pub fn get_current_pv(&self) -> Option<RootMove> {
        let moves = self.moves.lock().unwrap();
        moves.get(self.pv_idx()).cloned()
    }

    /// Returns the root move at `idx`, or [`None`] if out of bounds.
    pub fn get(&self, idx: usize) -> Option<RootMove> {
        let moves = self.moves.lock().unwrap();
        moves.get(idx).cloned()
    }

    /// Returns the first root move, or [`None`] if no moves exist.
    ///
    /// The caller must sort the list beforehand (via [`sort_from_pv_idx`](Self::sort_from_pv_idx)
    /// or [`sort_all`](Self::sort_all)) for this to return the highest-scoring move.
    pub fn get_best(&self) -> Option<RootMove> {
        let moves = self.moves.lock().unwrap();
        moves.first().cloned()
    }

    /// Sets the current PV index for Multi-PV search.
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

    /// Returns a detached snapshot of the current root move order and scores.
    pub fn snapshot(&self) -> Vec<RootMove> {
        self.moves.lock().unwrap().clone()
    }

    /// Sorts root moves from pv_idx to end by score (stable sort).
    pub fn sort_from_pv_idx(&self) {
        let pv_idx = self.pv_idx();
        let mut moves = self.moves.lock().unwrap();
        if pv_idx < moves.len() {
            moves[pv_idx..].sort_by_key(|m| std::cmp::Reverse(m.score));
        }
    }

    /// Sorts all root moves by score for final result ordering.
    pub fn sort_all(&self) {
        let mut moves = self.moves.lock().unwrap();
        moves.sort_by_key(|m| std::cmp::Reverse(m.score));
    }

    /// Returns the number of root moves available.
    pub fn count(&self) -> usize {
        self.moves.lock().unwrap().len()
    }

    /// Applies a function to all root moves and collects the results.
    pub fn map<T, F>(&self, f: F) -> Vec<T>
    where
        F: FnMut(&RootMove) -> T,
    {
        let moves = self.moves.lock().unwrap();
        moves.iter().map(f).collect()
    }

    /// Checks whether a move square exists in the remaining moves (from pv_idx onwards).
    pub fn contains_from_pv_idx(&self, sq: Square) -> bool {
        let pv_idx = self.pv_idx();
        let moves = self.moves.lock().unwrap();
        moves.iter().skip(pv_idx).any(|rm| rm.sq == sq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pv_array(line: &[Square]) -> [Square; MAX_PLY] {
        let mut pv = [Square::None; MAX_PLY];
        pv[..line.len()].copy_from_slice(line);
        pv
    }

    fn find<T>(rms: &RootMoves, sq: Square, f: impl Fn(&RootMove) -> T) -> T {
        rms.map(|rm| (rm.sq, f(rm)))
            .into_iter()
            .find(|(s, _)| *s == sq)
            .map(|(_, v)| v)
            .expect("square should be present in the root moves")
    }

    #[test]
    fn new_builds_one_root_move_per_legal_move() {
        let rms = RootMoves::new(&Board::new());
        assert_eq!(rms.count(), 4); // the standard opening has four legal moves
        assert_eq!(rms.pv_idx(), 0);
    }

    #[test]
    fn update_seeds_then_halves_the_running_average() {
        let rms = RootMoves::new(&Board::new());
        let sq = rms.map(|rm| rm.sq)[0];
        let pv = pv_array(&[sq]);

        rms.update(sq, ScaledScore::from_disc_diff(4), true, &pv);
        assert_eq!(
            find(&rms, sq, |rm| rm.score),
            ScaledScore::from_disc_diff(4)
        );
        // Seeded from the -INF sentinel, the average takes the first score verbatim.
        assert_eq!(
            find(&rms, sq, |rm| rm.average_score),
            ScaledScore::from_disc_diff(4)
        );

        rms.update(sq, ScaledScore::from_disc_diff(8), true, &pv);
        assert_eq!(
            find(&rms, sq, |rm| rm.score),
            ScaledScore::from_disc_diff(8)
        );
        // (4 + 8) / 2 == 6
        assert_eq!(
            find(&rms, sq, |rm| rm.average_score),
            ScaledScore::from_disc_diff(6)
        );
    }

    #[test]
    fn update_non_pv_resets_score_but_still_folds_the_average() {
        let rms = RootMoves::new(&Board::new());
        let sq = rms.map(|rm| rm.sq)[0];
        let pv = pv_array(&[sq]);

        rms.update(sq, ScaledScore::from_disc_diff(4), true, &pv);
        rms.update(sq, ScaledScore::from_disc_diff(8), false, &pv);

        assert_eq!(find(&rms, sq, |rm| rm.score), -ScaledScore::INF);
        assert_eq!(
            find(&rms, sq, |rm| rm.average_score),
            ScaledScore::from_disc_diff(6)
        );
    }

    #[test]
    fn update_copies_the_pv_until_the_none_sentinel() {
        let rms = RootMoves::new(&Board::new());
        let sq = rms.map(|rm| rm.sq)[0];
        let pv = pv_array(&[sq, Square::C3, Square::H8]); // trailing entries stay None

        rms.update(sq, ScaledScore::from_disc_diff(1), true, &pv);

        assert_eq!(
            find(&rms, sq, |rm| rm.pv.clone()),
            vec![sq, Square::C3, Square::H8]
        );
    }

    #[test]
    #[should_panic(expected = "on a `None` value")]
    fn update_panics_for_a_square_not_in_the_root_list() {
        // The expected message pins the missing-square `Option::unwrap()` panic,
        // distinguishing it from the lock's `Result::unwrap()` (an `Err` value).
        let rms = RootMoves::new(&Board::new());
        // A1 is never a legal opening move, so it is absent from the root list.
        rms.update(
            Square::A1,
            ScaledScore::ZERO,
            true,
            &pv_array(&[Square::A1]),
        );
    }

    #[test]
    fn sort_from_pv_idx_orders_the_tail_by_descending_score() {
        let rms = RootMoves::new(&Board::new());
        let sqs = rms.map(|rm| rm.sq);
        for (i, &sq) in sqs.iter().enumerate() {
            rms.update(
                sq,
                ScaledScore::from_disc_diff(i as i32),
                true,
                &pv_array(&[sq]),
            );
        }

        rms.sort_from_pv_idx();

        let best = rms.get_best().unwrap();
        assert_eq!(best.sq, sqs[sqs.len() - 1]);
        assert_eq!(
            best.score,
            ScaledScore::from_disc_diff((sqs.len() - 1) as i32)
        );
    }

    #[test]
    fn sort_from_pv_idx_is_a_noop_when_pv_idx_is_out_of_range() {
        let rms = RootMoves::new(&Board::new());
        let before = rms.map(|rm| rm.sq);

        rms.set_pv_idx(rms.count() + 5);
        rms.sort_from_pv_idx(); // must neither panic nor reorder

        assert_eq!(rms.map(|rm| rm.sq), before);
    }

    #[test]
    fn current_pv_tracks_the_pv_index_and_its_bounds() {
        let rms = RootMoves::new(&Board::new());

        rms.set_pv_idx(1);
        assert_eq!(rms.pv_idx(), 1);
        assert!(rms.get_current_pv().is_some());

        rms.set_pv_idx(rms.count());
        assert!(rms.get_current_pv().is_none());
    }

    #[test]
    fn get_returns_the_indexed_root_move() {
        let rms = RootMoves::new(&Board::new());
        let sqs = rms.map(|rm| rm.sq);

        assert_eq!(rms.get(1).map(|rm| rm.sq), Some(sqs[1]));
    }

    #[test]
    fn get_returns_none_for_out_of_bounds_index() {
        let rms = RootMoves::new(&Board::new());

        assert!(rms.get(rms.count()).is_none());
    }

    #[test]
    fn contains_from_pv_idx_respects_membership_and_the_pv_window() {
        let rms = RootMoves::new(&Board::new());
        let sq = rms.map(|rm| rm.sq)[0];

        assert!(rms.contains_from_pv_idx(sq));
        assert!(!rms.contains_from_pv_idx(Square::A1));

        rms.set_pv_idx(rms.count()); // skip every move
        assert!(!rms.contains_from_pv_idx(sq));
    }

    #[test]
    fn save_previous_scores_snapshots_the_current_scores() {
        let rms = RootMoves::new(&Board::new());
        let sq = rms.map(|rm| rm.sq)[0];
        rms.update(sq, ScaledScore::from_disc_diff(5), true, &pv_array(&[sq]));

        rms.save_previous_scores();

        assert_eq!(
            find(&rms, sq, |rm| rm.previous_score),
            ScaledScore::from_disc_diff(5)
        );
    }

    #[test]
    fn snapshot_is_detached_from_later_updates() {
        let rms = RootMoves::new(&Board::new());
        let sq = rms.map(|rm| rm.sq)[0];
        rms.update(sq, ScaledScore::from_disc_diff(5), true, &pv_array(&[sq]));

        let snapshot = rms.snapshot();

        rms.update(
            sq,
            ScaledScore::from_disc_diff(9),
            true,
            &pv_array(&[sq, Square::C3]),
        );

        let snapshotted = snapshot
            .iter()
            .find(|rm| rm.sq == sq)
            .expect("square should be present in the snapshot");
        assert_eq!(snapshotted.score, ScaledScore::from_disc_diff(5));
        assert_eq!(snapshotted.pv, vec![sq]);
    }

    #[test]
    fn empty_root_moves_have_no_best_or_current_pv() {
        // A full board has no empty squares and thus no legal moves.
        let board = Board::from_bitboards(u64::MAX, 0);
        let rms = RootMoves::new(&board);

        assert_eq!(rms.count(), 0);
        assert!(rms.get_best().is_none());
        assert!(rms.get_current_pv().is_none());
    }
}
