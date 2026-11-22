//! Lazy SMP for the midgame root search.
//!
//! The main thread runs the iterative deepening of [`search_root`]; every
//! other thread of the pool runs a helper search on the same root. The
//! threads share results only through the transposition table and the
//! evaluation cache, and each helper walks the tree its own way: it skips
//! some depths and widens its aspiration windows by an amount taken from its
//! index, so the threads spread over different parts of the tree. A timed
//! search reports the move that the threads' last completed iterations vote
//! for.
//!
//! [`search_root`]: super::search_root

mod policy;

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::board::Board;
use crate::eval::{Eval, EvalMode};
use crate::probcut::Selectivity;
use crate::search::SearchTask;
use crate::search::context::SearchContext;
use crate::search::correction_history::CorrectionHistories;
use crate::search::counters::SearchCounters;
use crate::search::result::Iteration;
use crate::search::root_move::RootMoves;
use crate::search::threading::{HelperJob, Thread, ThreadPool};
use crate::transposition_table::TranspositionTable;
use crate::types::Depth;

use super::{aspiration_search, aspiration_window};
use policy::{Ballot, skips_depth, winner};

/// A Lazy SMP helper thread, identified by its index in the pool.
#[derive(Clone, Copy)]
pub(in crate::search) struct Helper(NonZeroUsize);

impl Helper {
    /// Returns whether this helper skips the iteration whose depth plus root
    /// ply is `depth_ply`.
    ///
    /// Each helper searches alternating runs of `SKIP_SIZE` depths, shifted
    /// by its phase, so at any moment the helpers spread across neighboring
    /// depths instead of repeating the main thread's iteration.
    fn skips_depth(self, depth_ply: Depth) -> bool {
        skips_depth(self.0, depth_ply)
    }

    pub(super) fn index(self) -> usize {
        self.0.get()
    }
}

/// The helper threads of one midgame search, aborted when dropped.
pub(super) struct Helpers<'a> {
    /// Pool running the helpers; `None` once stopped or without other threads.
    pool: Option<&'a ThreadPool>,
    report: Arc<HelperReport>,
}

/// What the helper threads of one search hand back to the main thread.
#[derive(Default)]
struct HelperReport {
    /// Nodes of the helpers' finished iterations, for progress reports.
    nodes: AtomicU64,
    /// Work of the helpers that have ended.
    results: Mutex<HelperResults>,
}

/// Work of the helper threads of one search.
#[derive(Default)]
pub(super) struct HelperResults {
    /// Counters summed over the helpers.
    pub(super) counters: SearchCounters,
    /// Last completed iteration of every helper that completed one.
    pub(super) completed: Vec<Iteration>,
}

impl<'a> Helpers<'a> {
    /// Starts a helper search of `task`'s root, up to `max_depth`, on every
    /// other thread of the pool.
    pub(super) fn start(task: &'a SearchTask, max_depth: Depth) -> Self {
        let report = Arc::new(HelperReport::default());
        let pool = (task.pool.size > 1).then(|| {
            let search = HelperSearch {
                board: task.board,
                selectivity: task.mid_selectivity,
                tt: task.tt.clone(),
                eval: task.eval.clone(),
                eval_mode: task.eval_mode,
                corrections: task.corrections.clone(),
                max_depth,
                report: report.clone(),
            };
            let job: HelperJob = Arc::new(move |thread: &Arc<Thread>| search.run(thread));
            task.pool.start_helpers(&job);
            &*task.pool
        });
        Self { pool, report }
    }

    /// Returns the nodes of the helpers' finished iterations so far.
    pub(super) fn nodes(&self) -> u64 {
        self.report.nodes.load(Ordering::Relaxed)
    }

    /// Aborts the helpers and returns their work.
    pub(super) fn stop(mut self) -> HelperResults {
        if let Some(pool) = self.pool.take() {
            pool.stop_helpers();
        }
        std::mem::take(&mut *self.report.results.lock().unwrap())
    }
}

impl Drop for Helpers<'_> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            pool.stop_helpers();
        }
    }
}

/// Root search a helper thread runs alongside the main thread.
struct HelperSearch {
    board: Board,
    selectivity: Selectivity,
    tt: Arc<TranspositionTable>,
    eval: Arc<Eval>,
    eval_mode: Option<EvalMode>,
    corrections: Arc<CorrectionHistories>,
    max_depth: Depth,
    report: Arc<HelperReport>,
}

impl HelperSearch {
    /// Iterates on the root on `thread` until the maximum depth is completed
    /// or the main thread aborts the helpers, then reports to the main thread.
    fn run(&self, thread: &Arc<Thread>) {
        let helper =
            Helper(NonZeroUsize::new(thread.idx()).expect("helpers run on worker threads"));
        let root_moves = RootMoves::new(&self.board);
        let mut ctx = SearchContext::new(
            &self.board,
            self.selectivity,
            &root_moves,
            &self.tt,
            &self.eval,
        );
        ctx.helper = Some(helper);
        ctx.correction_history = Some(self.corrections.take(thread.idx()));
        if let Some(mode) = self.eval_mode {
            ctx.eval_mode = mode;
        }

        let root_ply = ctx.ply() as Depth;
        let mut reported_nodes = 0;
        let mut last = None;
        for depth in 1..=self.max_depth {
            if helper.skips_depth(depth + root_ply) {
                continue;
            }
            let (mut alpha, mut beta, delta) = aspiration_window(&ctx, depth);
            aspiration_search(
                &mut ctx,
                &self.board,
                depth,
                &mut alpha,
                &mut beta,
                delta,
                thread,
            );
            self.report
                .nodes
                .fetch_add(ctx.n_nodes - reported_nodes, Ordering::Relaxed);
            reported_nodes = ctx.n_nodes;
            if thread.is_search_aborted() {
                break;
            }
            ctx.root_moves.sort_from_pv_idx();
            last = Some(Iteration::capture(&ctx, depth, false));
        }
        if let Some(history) = ctx.correction_history.take() {
            self.corrections.put(thread.idx(), history);
        }

        let mut results = self.report.results.lock().unwrap();
        results.counters.merge(&ctx.counters());
        results.completed.extend(last);
    }
}

/// Picks the reported iteration among the main thread's and the helpers'.
///
/// Each iteration votes for its best move with weight
/// `(score - lowest score + VOTE_MARGIN) * depth`; the iteration whose move
/// collects the most votes wins, the deeper one on a tie, and the main
/// thread on a full tie.
pub(super) fn vote_best(main: Option<Iteration>, helpers: Vec<Iteration>) -> Option<Iteration> {
    let mut candidates: Vec<Iteration> = main.into_iter().chain(helpers).collect();
    let best = winner(&candidates, |it| {
        let root = it.best_move();
        Ballot {
            sq: root.sq,
            score: root.score,
            depth: it.depth,
        }
    })?;
    Some(candidates.swap_remove(best))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{DiscScore, Score};
    use crate::search::root_move::RootMove;
    use crate::square::Square;

    #[test]
    fn equal_votes_and_depth_keep_the_first_complete_snapshot() {
        let snapshot = |sq, next, selectivity| {
            let mut root = RootMove::new(sq);
            root.score = Score::from_raw(-137);
            root.pv = vec![sq, next];
            Iteration {
                root_moves: vec![root],
                depth: 10,
                selectivity,
                is_endgame: false,
            }
        };
        let tied_helper = || snapshot(Square::F5, Square::F6, Selectivity::None);
        let first = || snapshot(Square::D3, Square::C3, Selectivity::Mid);
        let main_wins = vote_best(Some(first()), vec![tied_helper()]).unwrap();
        let helper_wins = vote_best(None, vec![first(), tied_helper()]).unwrap();
        for result in [main_wins, helper_wins] {
            assert_eq!(result.best_move().sq, Square::D3);
            assert_eq!(result.best_move().pv, vec![Square::D3, Square::C3]);
            assert_eq!(result.selectivity, Selectivity::Mid);
        }
        assert!(vote_best(None, vec![]).is_none());
    }

    #[test]
    fn vote_prefers_the_move_backed_by_more_and_deeper_iterations() {
        let iteration = |sq, discs, depth| {
            let mut rm = RootMove::new(sq);
            rm.score = Score::from_disc_diff(DiscScore::new(discs));
            Iteration {
                root_moves: vec![rm],
                depth,
                selectivity: Selectivity::None,
                is_endgame: false,
            }
        };
        let lead = |c: Option<Iteration>| c.unwrap().root_moves[0].sq;

        // Two helpers agreeing on D3 outvote a deeper lone F5.
        let best = vote_best(
            Some(iteration(Square::C4, 0, 12)),
            vec![
                iteration(Square::D3, 0, 11),
                iteration(Square::D3, 0, 11),
                iteration(Square::F5, 0, 13),
            ],
        );
        assert_eq!(lead(best), Square::D3);

        // On equal votes the deeper iteration of the same move is reported.
        let best = vote_best(
            Some(iteration(Square::C4, 2, 10)),
            vec![iteration(Square::C4, 2, 12)],
        );
        assert_eq!(best.unwrap().depth, 12);

        // A main thread without a completed iteration never wins.
        let best = vote_best(None, vec![iteration(Square::D3, -5, 4)]);
        assert_eq!(lead(best), Square::D3);
    }
}
