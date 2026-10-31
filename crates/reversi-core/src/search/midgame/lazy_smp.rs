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
use crate::search::threading::{HelperJob, Thread, ThreadPool};
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, ScaledScore};

use super::{aspiration_search, aspiration_window};

/// Depth-skip run lengths per helper; see [`Helper::skips_depth`].
const SKIP_SIZE: [Depth; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];

/// Depth-skip phases per helper; see [`Helper::skips_depth`].
const SKIP_PHASE: [Depth; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];

/// Extra aspiration window width per step of a helper's index.
const ASPIRATION_STEP: ScaledScore = ScaledScore::from_raw(ScaledScore::SCALE / 4);

/// Number of distinct aspiration window widths; see
/// [`Helper::aspiration_widening`].
const ASPIRATION_CYCLE: usize = 8;

/// Vote margin added to every score so the worst candidate still counts.
const VOTE_MARGIN: ScaledScore = ScaledScore::from_disc_diff(1);

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
        let i = (self.0.get() - 1) % SKIP_SIZE.len();
        (depth_ply + SKIP_PHASE[i]) / SKIP_SIZE[i] % 2 == 1
    }

    /// Returns how much wider than the main thread's this helper opens its
    /// aspiration windows, so the threads fail high and low at different
    /// scores.
    pub(super) fn aspiration_widening(self) -> ScaledScore {
        let steps = (self.0.get() % ASPIRATION_CYCLE) as i32;
        ScaledScore::from_raw(ASPIRATION_STEP.value() * steps)
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
        let mut ctx = SearchContext::new(
            &self.board,
            self.selectivity,
            self.tt.clone(),
            self.eval.clone(),
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
            ctx.root_moves.save_previous_scores();
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
                .fetch_add(ctx.counters.n_nodes - reported_nodes, Ordering::Relaxed);
            reported_nodes = ctx.counters.n_nodes;
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
        results.counters.merge(&ctx.counters);
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
    let min_score = candidates.iter().map(|c| c.best_move().score).min()?;

    let mut votes = [0i64; 64];
    for c in &candidates {
        let rm = c.best_move();
        votes[rm.sq as usize] +=
            i64::from((rm.score - min_score + VOTE_MARGIN).value()) * i64::from(c.depth);
    }
    let votes_of = |c: &Iteration| votes[c.best_move().sq as usize];

    let mut best = 0;
    for (i, c) in candidates.iter().enumerate().skip(1) {
        let (c_votes, best_votes) = (votes_of(c), votes_of(&candidates[best]));
        if c_votes > best_votes || (c_votes == best_votes && c.depth > candidates[best].depth) {
            best = i;
        }
    }
    Some(candidates.swap_remove(best))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::root_move::RootMove;
    use crate::square::Square;

    #[test]
    fn vote_prefers_the_move_backed_by_more_and_deeper_iterations() {
        let iteration = |sq, discs, depth| {
            let mut rm = RootMove::new(sq);
            rm.score = ScaledScore::from_disc_diff(discs);
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
