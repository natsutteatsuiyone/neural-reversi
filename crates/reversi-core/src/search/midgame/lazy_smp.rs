//! Lazy SMP for the midgame root search.
//!
//! The main thread runs the iterative deepening of [`search_root`]; every
//! other thread of the pool runs a helper search on the same root. The
//! threads share results only through the transposition table, and each
//! helper walks the tree its own way: it skips some depths, opens wider
//! aspiration windows, and orders moves differently, so the threads spread
//! over different parts of the tree. A timed search reports the move that
//! the threads' last completed iterations vote for.
//!
//! [`search_root`]: super::search_root

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::board::Board;
use crate::eval::{Eval, EvalMode};
use crate::move_list::MoveList;
use crate::probcut::Selectivity;
use crate::search::SearchTask;
use crate::search::context::SearchContext;
use crate::search::counters::SearchCounters;
use crate::search::node_type::NodeType;
use crate::search::result::CompletedState;
use crate::search::threading::{HelperJob, Thread, ThreadPool};
use crate::square::Square;
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

/// Right shift that turns a 64-bit hash into a move-ordering bias of 0 to
/// 2 discs (in 1/256 disc).
const ORDERING_BIAS_SHIFT: u32 = 55;

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

    /// Sorts the moves of the node with hash `key` into this helper's order.
    ///
    /// Inside the tree, moves are sorted by their values plus a bias of up to
    /// two discs hashed from the node, the move, and the helper; the values
    /// themselves stay unbiased because LMR reads them. At the root, the moves
    /// after the first are rotated by an offset taken from the helper's index.
    /// The TT move stays first either way.
    #[inline(always)]
    pub(in crate::search) fn sort_moves<NT: NodeType>(self, move_list: &mut MoveList, key: u64) {
        let idx = self.0.get();
        if NT::ROOT_NODE {
            move_list.sort();
            let n_moves = move_list.count();
            if n_moves > 2 {
                move_list.rotate_left_from(1, (idx - 1) % (n_moves - 1));
            }
        } else {
            move_list.sort_with_bias(|mv| ordering_bias(key, mv.sq, idx));
        }
    }
}

/// Returns the move-ordering bias helper `idx` gives `sq` at the node with
/// hash `key`.
#[inline(always)]
fn ordering_bias(key: u64, sq: Square, idx: usize) -> i32 {
    let h = (key
        ^ (sq.index() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (idx as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
    .wrapping_mul(0xD6E8_FEB8_6659_FD93);
    (h >> ORDERING_BIAS_SHIFT) as i32
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
    pub(super) completed: Vec<CompletedState>,
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
                max_depth,
                reduce_fail_high_depth: task.time_manager.is_some(),
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
    max_depth: Depth,
    reduce_fail_high_depth: bool,
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
                self.reduce_fail_high_depth,
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
            last = Some(CompletedState {
                root_moves: ctx.root_moves.snapshot(),
                depth,
                selectivity: ctx.selectivity,
                is_endgame: false,
            });
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
pub(super) fn vote_best(main: CompletedState, helpers: Vec<CompletedState>) -> CompletedState {
    // An aborted search can leave the main thread without a completed iteration.
    let lead = |c: &CompletedState| {
        c.root_moves
            .first()
            .filter(|_| c.depth > 0)
            .map(|rm| (rm.sq, rm.score))
    };
    let Some(min_score) = std::iter::once(&main)
        .chain(&helpers)
        .filter_map(lead)
        .map(|(_, score)| score)
        .min()
    else {
        return main;
    };

    let mut votes = [0i64; 64];
    for c in std::iter::once(&main).chain(&helpers) {
        if let Some((sq, score)) = lead(c) {
            votes[sq as usize] +=
                i64::from((score - min_score + VOTE_MARGIN).value()) * i64::from(c.depth);
        }
    }
    let votes_of = |c: &CompletedState| lead(c).map_or(-1, |(sq, _)| votes[sq as usize]);

    let mut best = main;
    for c in helpers {
        let (c_votes, best_votes) = (votes_of(&c), votes_of(&best));
        if c_votes > best_votes || (c_votes == best_votes && c.depth > best.depth) {
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::root_move::RootMove;

    #[test]
    fn vote_prefers_the_move_backed_by_more_and_deeper_iterations() {
        let iteration = |sq, discs, depth| {
            let mut rm = RootMove::new(sq);
            rm.score = ScaledScore::from_disc_diff(discs);
            CompletedState {
                root_moves: vec![rm],
                depth,
                selectivity: Selectivity::None,
                is_endgame: false,
            }
        };
        let lead = |c: &CompletedState| c.root_moves[0].sq;

        // Two helpers agreeing on D3 outvote a deeper lone F5.
        let best = vote_best(
            iteration(Square::C4, 0, 12),
            vec![
                iteration(Square::D3, 0, 11),
                iteration(Square::D3, 0, 11),
                iteration(Square::F5, 0, 13),
            ],
        );
        assert_eq!(lead(&best), Square::D3);

        // On equal votes the deeper iteration of the same move is reported.
        let best = vote_best(
            iteration(Square::C4, 2, 10),
            vec![iteration(Square::C4, 2, 12)],
        );
        assert_eq!(best.depth, 12);

        // A main thread without a completed iteration never wins.
        let best = vote_best(
            iteration(Square::C4, 0, 0),
            vec![iteration(Square::D3, -5, 4)],
        );
        assert_eq!(lead(&best), Square::D3);
    }
}
