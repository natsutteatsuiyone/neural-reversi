//! Game tree search engine.
//!
//! Provides the main [`Search`] engine and dispatches root searches to the
//! midgame and endgame drivers; the shared alpha-beta core lives in the
//! `pvs` module.

pub mod context;
pub mod counters;
mod endgame;
pub(crate) mod midgame;
pub mod node_type;
pub mod options;
pub mod progress;
mod pvs;
pub mod result;
pub mod root_move;
pub mod side_to_move;
pub mod stack;
mod stop;
pub mod strategy;
pub mod threading;
pub mod time_control;

#[doc(hidden)]
pub use endgame::{EndGameCaches, null_window_search, solve_last1};
pub use options::{SearchConstraint, SearchRunOptions};
pub use progress::{SearchProgress, SearchProgressCallback};
pub(crate) use pvs::{LMR_MIN_DEPTH, search};
pub use stop::StopToken;

use std::sync::Arc;

use crate::board::Board;
use crate::constants::{INITIAL_EMPTY_COUNT, MAX_THREADS};
use crate::eval::{Eval, EvalMode};
use crate::level::Level;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::search::options::{SearchOptions, available_cpus};
use crate::search::result::SearchResult;
use crate::search::threading::{Thread, ThreadPool};
use crate::search::time_control::TimeManager;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, ScaledScore};

/// Lowest endgame selectivity whose completion lets later timed searches
/// skip the midgame iterations.
///
/// A Level1 solve can finish on one move and still abort on the next, which
/// then has no midgame result to fall back on.
const ENDGAME_TRACKING_MIN_SELECTIVITY: Selectivity = Selectivity::Level2;

/// Main search engine that coordinates game tree exploration.
///
/// Manages the transposition table, thread pool, and evaluation function
/// used during search. Create one instance per game session and call
/// [`Search::init`] between games to reset state.
pub struct Search {
    tt: Arc<TranspositionTable>,
    threads: Arc<ThreadPool>,
    eval: Arc<Eval>,
    endgame_start_n_empties: Option<Depth>,
    timed_out: bool,
}

/// Shared heavyweight search resources that can back multiple [`Search`]
/// engines concurrently.
///
/// Engines created from the same resource bundle share the evaluation network;
/// the transposition table is per-engine, and so is the thread pool unless
/// [`SearchSharedResources::share_thread_pool`] was called. Useful for callers
/// that need more than one independent search in flight, such as GGS synchro
/// child games, without reloading neural-network weights for every worker.
pub struct SearchSharedResources {
    eval: Arc<Eval>,
    tt_mb_size: usize,
    n_threads: usize,
    thread_pool: Option<Arc<ThreadPool>>,
}

/// Task descriptor passed to search threads.
///
/// Contains all shared state needed for a search thread to independently
/// execute a search on the given board position.
#[derive(Clone)]
pub struct SearchTask {
    /// Board position to search.
    pub board: Board,
    /// Midgame selectivity: [`Selectivity::Level1`] for fixed-level searches,
    /// [`Selectivity::Mid`] for time-controlled searches (both 73%), and
    /// [`Selectivity::None`] when ProbCut is disabled. The endgame driver uses
    /// its own selectivity ladder and ignores this value.
    pub mid_selectivity: Selectivity,
    /// Shared transposition table.
    pub tt: Arc<TranspositionTable>,
    /// Shared thread pool for parallel search.
    pub pool: Arc<ThreadPool>,
    /// Shared evaluation function.
    pub eval: Arc<Eval>,
    /// Search depth and endgame configuration.
    pub level: Level,
    /// Whether to report multiple principal variations.
    pub multi_pv: bool,
    /// Optional callback invoked to report search progress.
    pub callback: Option<Arc<SearchProgressCallback>>,
    /// Optional time manager for time-controlled searches.
    pub time_manager: Option<Arc<TimeManager>>,
    /// Optional override for evaluation mode.
    pub eval_mode: Option<EvalMode>,
}

impl SearchTask {
    /// Returns how many PV lines to search: every root move in Multi-PV mode,
    /// otherwise one.
    pub(crate) fn pv_count(&self, root_moves_count: usize) -> usize {
        if self.multi_pv { root_moves_count } else { 1 }
    }
}

fn midgame_selectivity(options: &SearchRunOptions) -> Selectivity {
    if options.probcut_disabled {
        return Selectivity::None;
    }

    match &options.constraint {
        SearchConstraint::Level(_) => Selectivity::Level1,
        SearchConstraint::Time(_) => Selectivity::Mid,
    }
}

impl SearchSharedResources {
    /// Creates a reusable search-resource bundle from search options.
    ///
    /// # Panics
    ///
    /// Panics if the evaluation weight files cannot be loaded.
    pub fn new(options: &SearchOptions) -> Self {
        let n_threads = options
            .n_threads
            .min(available_cpus())
            .clamp(1, MAX_THREADS);
        let eval = Eval::with_weight_files(
            options.eval_path.as_deref(),
            options.eval_sm_path.as_deref(),
        )
        .unwrap_or_else(|err| panic!("failed to load evaluation weights: {err}"));

        // Ensure ProbCut tables are initialized before any engine is spawned.
        probcut::init();

        Self {
            eval: Arc::new(eval),
            tt_mb_size: options.tt_mb_size,
            n_threads,
            thread_pool: None,
        }
    }

    /// Makes all engines created from this bundle after this call share one
    /// thread pool of `n_threads` threads.
    ///
    /// A pool runs one search at a time: engines sharing it must never search
    /// concurrently, and the caller is responsible for serializing them.
    pub fn share_thread_pool(&mut self) {
        self.thread_pool = Some(ThreadPool::new(self.n_threads));
    }

    /// Returns the search thread count after clamping to the CPU count and
    /// the engine limit.
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }
}

impl Search {
    /// Creates a new search engine with the given options.
    ///
    /// Initializes the evaluation function, transposition table, and thread pool.
    /// The number of threads is clamped to the available CPU count and [`MAX_THREADS`].
    ///
    /// # Panics
    ///
    /// Panics if the evaluation weight files cannot be loaded.
    pub fn new(options: &SearchOptions) -> Self {
        let shared = SearchSharedResources::new(options);
        Self::from_shared_resources(&shared)
    }

    /// Creates a new search engine from a shared-resource bundle.
    pub fn from_shared_resources(shared: &SearchSharedResources) -> Self {
        Self::from_shared_resources_with_tt(
            shared,
            Arc::new(TranspositionTable::new(shared.tt_mb_size)),
        )
    }

    /// Creates a new search engine from a shared-resource bundle that uses
    /// `tt` instead of allocating its own table.
    ///
    /// Engines sharing a table must never search concurrently.
    pub fn from_shared_resources_with_tt(
        shared: &SearchSharedResources,
        tt: Arc<TranspositionTable>,
    ) -> Self {
        Self {
            tt,
            threads: shared
                .thread_pool
                .clone()
                .unwrap_or_else(|| ThreadPool::new(shared.n_threads)),
            eval: shared.eval.clone(),
            endgame_start_n_empties: None,
            timed_out: false,
        }
    }

    /// Returns whether the last [`Search::run`] used up its time and aborted
    /// the iteration in progress, rather than stopping between iterations.
    pub fn timed_out(&self) -> bool {
        self.timed_out
    }

    /// Returns a reference to the transposition table.
    pub fn tt(&self) -> &Arc<TranspositionTable> {
        &self.tt
    }

    /// Returns a reference to the shared evaluation function.
    pub fn eval(&self) -> &Arc<Eval> {
        &self.eval
    }

    /// Resets the search state for a new game.
    ///
    /// Clears the transposition table, resets the TT generation counter,
    /// flushes the evaluation cache, and resets endgame tracking.
    pub fn init(&mut self) {
        self.tt.clear();
        self.tt.reset_generation();
        self.eval.clear_cache();
        self.endgame_start_n_empties = None;
    }

    /// Resizes the transposition table to `mb_size` MiB.
    ///
    /// Replaces the table only when the requested size differs from the
    /// current one, avoiding unnecessary reallocation.
    pub fn resize_tt(&mut self, mb_size: usize) {
        if self.tt.mb_size() != mb_size {
            self.tt = Arc::new(TranspositionTable::new(mb_size));
        }
    }

    /// Replaces the transposition table, e.g. with one shared by another engine.
    ///
    /// Engines sharing a table must never search concurrently.
    pub fn set_tt(&mut self, tt: Arc<TranspositionTable>) {
        self.tt = tt;
    }

    /// Runs a search on the given board position.
    ///
    /// Selects the appropriate search strategy based on the constraint (fixed level
    /// or time-controlled) and executes the search.
    ///
    /// # Panics
    ///
    /// Panics if `board` has fewer than 4 discs.
    pub fn run(&mut self, board: &Board, options: &SearchRunOptions) -> SearchResult {
        assert!(
            board.empty_count() as usize <= INITIAL_EMPTY_COUNT,
            "a search needs at most {INITIAL_EMPTY_COUNT} empty squares, got {}",
            board.empty_count()
        );
        let callback = options.callback.clone();
        let n_empties = board.empty_count();

        let (time_manager, mut effective_level) =
            self.build_time_controls(n_empties, &options.constraint);
        let is_time_mode = time_manager.is_some();
        let deadline = time_manager.clone();

        if is_time_mode {
            self.maybe_extend_endgame_depth(n_empties, &mut effective_level);
        }

        let task = SearchTask {
            board: *board,
            mid_selectivity: midgame_selectivity(options),
            tt: self.tt.clone(),
            pool: self.threads.clone(),
            eval: self.eval.clone(),
            level: effective_level,
            multi_pv: options.multi_pv,
            callback: callback.clone(),
            time_manager,
            eval_mode: options.eval_mode,
        };

        let result = self.execute_search(task, options.stop_token.as_ref());
        self.timed_out = deadline.is_some_and(|tm| tm.timed_out());

        if let Some(callback) = callback {
            callback(SearchProgress::from_result(&result));
        }

        if is_time_mode {
            self.update_endgame_tracking(n_empties, &result);
        }

        result
    }

    fn build_time_controls(
        &self,
        n_empties: Depth,
        constraint: &SearchConstraint,
    ) -> (Option<Arc<TimeManager>>, Level) {
        match constraint {
            SearchConstraint::Level(level) => (None, *level),
            SearchConstraint::Time(mode) => {
                let tm = Arc::new(TimeManager::new(
                    *mode,
                    self.threads.abort_state(),
                    n_empties,
                ));
                (Some(tm), Level::unlimited())
            }
        }
    }

    /// Lifts the endgame depth cap to [`Level::perfect`] once a previous
    /// time-controlled search has reached the endgame phase.
    ///
    /// Time-controlled searches default to [`Level::unlimited`], which caps the
    /// endgame at 14 ply. Once the endgame has been entered, subsequent searches
    /// should instead solve all the way to the end.
    fn maybe_extend_endgame_depth(&mut self, n_empties: Depth, level: &mut Level) {
        let Some(start) = self.endgame_start_n_empties else {
            return;
        };
        if n_empties > start {
            self.endgame_start_n_empties = None;
        } else {
            *level = Level::with_depths(level.mid_depth, Level::perfect().end_depth);
        }
    }

    /// Records the empty count at which an endgame solve first finished
    /// [`ENDGAME_TRACKING_MIN_SELECTIVITY`]. Later time-controlled searches
    /// then skip the midgame iterations and extend their end depth.
    ///
    /// Until then, each search starts with midgame iterations, so a result
    /// exists even if the endgame solve aborts before finishing a level.
    fn update_endgame_tracking(&mut self, n_empties: Depth, result: &SearchResult) {
        if n_empties > 0
            && self.endgame_start_n_empties.is_none()
            && result.is_endgame()
            && result.selectivity() >= ENDGAME_TRACKING_MIN_SELECTIVITY
        {
            self.endgame_start_n_empties = Some(n_empties - 1);
        }
    }

    fn execute_search(&mut self, task: SearchTask, stop: Option<&StopToken>) -> SearchResult {
        self.tt.increment_generation();

        let board = task.board;
        let result_receiver = self.threads.start_thinking(task, stop);

        let result = result_receiver.recv().unwrap_or_else(|_| {
            // Channel closed - search thread may have panicked. Return fallback.
            result::quick_move(&board, &self.eval)
        });

        self.threads.stop_timer();

        result
    }

    /// Returns the [`ThreadPool`] used by this search engine.
    ///
    /// [`ThreadPool`]: threading::ThreadPool
    #[cfg(test)]
    pub(crate) fn thread_pool(&self) -> Arc<threading::ThreadPool> {
        self.threads.clone()
    }
}

/// Widens the aspiration window around `score` after a fail-low or fail-high.
///
/// Returns `true` if the window was widened and the search must be repeated,
/// or `false` when `score` lies inside the window and the result stands.
pub(crate) fn widen_aspiration_window(
    score: ScaledScore,
    alpha: &mut ScaledScore,
    beta: &mut ScaledScore,
    delta: ScaledScore,
) -> bool {
    if score <= *alpha {
        *beta = *alpha;
        *alpha = (score - delta).max(-ScaledScore::INF);
    } else if score >= *beta {
        *alpha = (*beta - delta).max(*alpha);
        *beta = (score + delta).min(ScaledScore::INF);
    } else {
        return false;
    }
    true
}

/// Dispatches to midgame or endgame search based on remaining empties.
///
/// Compares the minimum endgame depth from the level configuration against the
/// number of empty squares. If the endgame depth covers all empties, delegates
/// to the endgame solver; otherwise delegates to the midgame search.
fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let min_end_depth = task.level.min_end_depth();
    let n_empties = task.board.empty_count();

    if min_end_depth >= n_empties {
        return endgame::search_root(task, thread);
    }

    midgame::search_root(task, thread)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::counters::SearchCounters;
    use crate::search::time_control::TimeControlMode;
    use crate::square::Square;

    fn one_thread_options() -> SearchOptions {
        SearchOptions::new(0).with_threads(Some(1))
    }

    #[test]
    fn fixed_level_uses_level1_midgame_selectivity() {
        let options = SearchRunOptions::with_level(Level::uniform(8, 8));

        assert_eq!(midgame_selectivity(&options), Selectivity::Level1);
    }

    #[test]
    fn time_constraint_uses_mid_midgame_selectivity() {
        let options = SearchRunOptions::with_time(TimeControlMode::Infinite);

        assert_eq!(midgame_selectivity(&options), Selectivity::Mid);
    }

    #[test]
    fn disabling_probcut_overrides_constraint_midgame_selectivity() {
        let fixed = SearchRunOptions::with_level(Level::uniform(8, 8)).disable_probcut();
        let timed = SearchRunOptions::with_time(TimeControlMode::Infinite).disable_probcut();

        assert_eq!(midgame_selectivity(&fixed), Selectivity::None);
        assert_eq!(midgame_selectivity(&timed), Selectivity::None);
    }

    #[test]
    fn shared_resources_reuse_eval_and_share_the_thread_pool_only_on_request() {
        let mut shared = SearchSharedResources::new(&SearchOptions::new(1).with_threads(Some(1)));

        let first = Search::from_shared_resources(&shared);
        let second = Search::from_shared_resources(&shared);

        assert!(Arc::ptr_eq(&first.eval, &second.eval));
        assert!(!Arc::ptr_eq(first.tt(), second.tt()));
        assert_eq!(first.tt().mb_size(), 1);
        assert!(!Arc::ptr_eq(&first.thread_pool(), &second.thread_pool()));

        shared.share_thread_pool();
        let first = Search::from_shared_resources(&shared);
        let second = Search::from_shared_resources(&shared);

        assert!(Arc::ptr_eq(&first.thread_pool(), &second.thread_pool()));
        assert!(!Arc::ptr_eq(first.tt(), second.tt()));
    }

    #[test]
    fn resize_tt_reuses_same_size_replaces_changed_size_and_init_resets_generation() {
        let mut search = Search::new(&one_thread_options());
        let original_tt = search.tt().clone();

        assert_eq!(search.tt().mb_size(), 0);
        assert_eq!(search.tt().increment_generation(), 1);

        search.resize_tt(0);
        assert!(Arc::ptr_eq(search.tt(), &original_tt));

        search.init();
        assert_eq!(search.tt().generation(), 0);
        assert_eq!(search.tt().usage_rate(), 0.0);

        search.resize_tt(1);
        assert!(!Arc::ptr_eq(search.tt(), &original_tt));
        assert_eq!(search.tt().mb_size(), 1);
        assert_eq!(search.tt().generation(), 0);
    }

    #[test]
    fn midgame_iterations_are_skipped_only_after_a_level2_endgame_solve() {
        let mut search = Search::new(&one_thread_options());
        let endgame = |selectivity| SearchResult::BestMove {
            sq: Square::A1,
            score: 0.0,
            n_nodes: 0,
            pv_line: vec![],
            depth: 30,
            selectivity,
            is_endgame: true,
            pv_moves: vec![],
            counters: SearchCounters::default(),
        };

        search.update_endgame_tracking(30, &endgame(Selectivity::Level1));
        assert_eq!(search.endgame_start_n_empties, None);

        search.update_endgame_tracking(28, &endgame(Selectivity::Level2));
        assert_eq!(search.endgame_start_n_empties, Some(27));
    }
}
