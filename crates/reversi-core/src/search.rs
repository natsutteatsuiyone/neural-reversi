//! Game tree search engine.
//!
//! Provides the main [`Search`] engine and dispatches root searches to the
//! midgame and endgame drivers; the shared alpha-beta core lives in the
//! `pvs` module.

mod aspiration;
pub mod context;
pub(crate) mod correction_history;
pub mod counters;
mod endgame;
pub(crate) mod midgame;
pub(crate) mod node_type;
pub mod options;
pub mod progress;
mod pvs;
pub mod result;
pub mod root_move;
pub mod side_to_move;
pub mod stack;
mod stop;
pub mod strategy;
pub(crate) mod threading;
pub mod time_control;

pub(crate) use aspiration::widen_aspiration_window;
#[doc(hidden)]
pub use endgame::solve_last1;
pub use options::{SearchConstraint, SearchRunOptions};
pub use progress::{SearchProgress, SearchProgressCallback};
pub(crate) use pvs::{LMR_MIN_DEPTH, search};
pub use stop::StopToken;

use std::sync::Arc;

use crate::board::Board;
use crate::constants::{INITIAL_EMPTY_COUNT, MAX_THREADS};
use crate::eval::{Eval, EvalMode};
use crate::probcut::Selectivity;
use crate::search::correction_history::CorrectionHistories;
use crate::search::options::{SearchOptions, available_cpus};
use crate::search::result::SearchResult;
use crate::search::threading::{Thread, ThreadPool};
use crate::search::time_control::TimeManager;
use crate::transposition_table::TranspositionTable;
use crate::types::Depth;

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
    corrections: Arc<CorrectionHistories>,
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

const TIMED_ENDGAME_ROOT_MAX_EMPTIES: Depth = 14;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RootPhase {
    Midgame { max_depth: Depth },
    Endgame { last: Selectivity },
}

/// Position, resolved search settings, and shared resources for one root search.
pub(crate) struct SearchTask {
    /// Board position to search.
    pub board: Board,
    /// Midgame selectivity: [`Selectivity::Level1`] (73%) for fixed-level searches,
    /// [`Selectivity::Mid`] (86%) for time-controlled searches, and
    /// [`Selectivity::None`] when ProbCut is disabled. The endgame driver uses
    /// its own selectivity ladder and ignores this value.
    pub mid_selectivity: Selectivity,
    /// Shared transposition table.
    pub tt: Arc<TranspositionTable>,
    /// Shared thread pool for parallel search.
    pub pool: Arc<ThreadPool>,
    /// Shared evaluation function.
    pub eval: Arc<Eval>,
    pub phase: RootPhase,
    /// Whether to report multiple principal variations.
    pub multi_pv: bool,
    /// Optional callback invoked to report search progress.
    pub callback: Option<Arc<SearchProgressCallback>>,
    /// Optional time manager for time-controlled searches.
    pub time_manager: Option<Arc<TimeManager>>,
    /// Optional override for evaluation mode.
    pub eval_mode: Option<EvalMode>,
    /// Evaluation corrections of this game's search threads.
    pub corrections: Arc<CorrectionHistories>,
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
        let eval = Eval::with_weight_files(
            options.eval_path.as_deref(),
            options.eval_sm_path.as_deref(),
        )
        .unwrap_or_else(|err| panic!("failed to load evaluation weights: {err}"));

        Self::from_eval(Arc::new(eval), options.tt_mb_size, Some(options.n_threads))
    }

    pub fn from_eval(eval: Arc<Eval>, tt_mb_size: usize, n_threads: Option<usize>) -> Self {
        let n_threads = n_threads
            .unwrap_or_else(available_cpus)
            .min(available_cpus())
            .clamp(1, MAX_THREADS);

        Self {
            eval,
            tt_mb_size,
            n_threads,
            thread_pool: None,
        }
    }

    pub fn eval(&self) -> &Arc<Eval> {
        &self.eval
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
            corrections: Arc::default(),
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
    /// flushes the evaluation cache and corrections, and resets endgame tracking.
    pub fn init(&mut self) {
        match Arc::get_mut(&mut self.tt) {
            Some(tt) => tt.clear(),
            None => self.tt.clear_shared(),
        }
        self.tt.reset_generation();
        self.eval.clear_cache();
        self.corrections.clear();
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

        let (time_manager, phase) = match &options.constraint {
            SearchConstraint::Level(level) => {
                let phase = match level.end_selectivity(n_empties) {
                    Some(last) => RootPhase::Endgame { last },
                    None => RootPhase::Midgame {
                        max_depth: level.mid_depth,
                    },
                };
                (None, phase)
            }
            SearchConstraint::Time(mode) => {
                let tm = Arc::new(TimeManager::new(
                    *mode,
                    self.threads.abort_state(),
                    n_empties,
                ));
                (Some(tm), self.timed_root_phase(n_empties))
            }
        };
        let is_time_mode = time_manager.is_some();
        let deadline = time_manager.clone();

        // The timer starts with the search, so an endgame root must not run
        // under the smaller midgame deadline even briefly.
        if let Some(tm) = &time_manager
            && matches!(phase, RootPhase::Endgame { .. })
        {
            tm.set_endgame_mode(true);
        }

        let task = SearchTask {
            board: *board,
            mid_selectivity: midgame_selectivity(options),
            tt: self.tt.clone(),
            pool: self.threads.clone(),
            eval: self.eval.clone(),
            phase,
            multi_pv: options.multi_pv,
            callback: callback.clone(),
            time_manager,
            eval_mode: options.eval_mode,
            corrections: self.corrections.clone(),
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

    fn timed_root_phase(&self, n_empties: Depth) -> RootPhase {
        let solved_before = self
            .endgame_start_n_empties
            .is_some_and(|start| n_empties <= start);

        if n_empties <= TIMED_ENDGAME_ROOT_MAX_EMPTIES || solved_before {
            RootPhase::Endgame {
                last: Selectivity::None,
            }
        } else {
            RootPhase::Midgame {
                max_depth: n_empties,
            }
        }
    }

    /// Records the empty count at which an endgame solve first finished
    /// [`ENDGAME_TRACKING_MIN_SELECTIVITY`].
    ///
    /// Until then, each search starts with midgame iterations, so a result
    /// exists even if the endgame solve aborts before finishing a level.
    fn update_endgame_tracking(&mut self, n_empties: Depth, result: &SearchResult) {
        if self
            .endgame_start_n_empties
            .is_some_and(|start| n_empties > start)
        {
            self.endgame_start_n_empties = None;
        }

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

fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    match task.phase {
        RootPhase::Midgame { max_depth } => midgame::search_root(task, max_depth, thread),
        RootPhase::Endgame { last } => endgame::search_root(task, last, thread),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::level::Level;
    use crate::player_color::PlayerColor;
    use crate::score::Score;
    use crate::search::counters::SearchCounters;
    use crate::search::time_control::TimeControlMode;
    use crate::square::Square;

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
        let mut shared = SearchSharedResources::from_eval(Eval::shared_for_tests(), 1, Some(1));

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

    fn search_for_tests() -> Search {
        let shared = SearchSharedResources::from_eval(Eval::shared_for_tests(), 0, Some(1));
        Search::from_shared_resources(&shared)
    }

    #[test]
    fn timed_midgame_handoff_reuses_an_interrupted_search() {
        // FFO #9 has 15 empties. G7 and A4 both score -8.
        let board = Board::from_string(
            "--XOXX--O-OOXXXX-OOOXXXX-XOXXXOXXXOXOOOXOXXOXOXX--OXOO----OOOO--",
            PlayerColor::White,
        )
        .unwrap();
        let mut search = search_for_tests();
        let pool = search.thread_pool();
        let timed = || SearchRunOptions::with_time(TimeControlMode::Infinite);
        let earlier = search.run(
            &board,
            &timed().callback(move |progress| {
                if progress.depth == 7 && !progress.is_endgame {
                    pool.abort_search();
                }
            }),
        );
        assert_eq!(earlier.depth(), 7);
        assert!(!earlier.is_endgame());

        // Level1 has written the endgame tables, but the saved midgame line
        // remains the public result until Level2 completes.
        let pool = search.thread_pool();
        let saw_level1 = Arc::new(AtomicBool::new(false));
        let seen = saw_level1.clone();
        let after_handoff = search.run(
            &board,
            &timed().callback(move |p| {
                if p.is_endgame && p.probability == Selectivity::Level1.probability() {
                    seen.store(true, Ordering::Relaxed);
                    pool.abort_search();
                }
            }),
        );
        assert!(saw_level1.load(Ordering::Relaxed));
        assert_eq!(after_handoff.depth(), 15);
        assert!(!after_handoff.is_endgame());

        let progress = Arc::new(Mutex::new(Vec::new()));
        let events = progress.clone();
        let result = search.run(
            &board,
            &timed().callback(move |p| events.lock().unwrap().push((p.depth, p.is_endgame))),
        );
        let events = progress.lock().unwrap();
        assert!(
            events
                .windows(2)
                .any(|pair| pair == [(15, false), (15, true)])
        );
        assert!(result.is_endgame());
        assert_eq!(result.selectivity(), Selectivity::None);
        assert!(matches!(result.best_move(), Some(Square::G7 | Square::A4)));
        assert_eq!(result.score().unwrap().value(), -8 * 256);
    }

    #[test]
    fn resize_tt_reuses_same_size_replaces_changed_size_and_init_resets_generation() {
        let mut search = search_for_tests();
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

    fn endgame_result(selectivity: Selectivity) -> SearchResult {
        SearchResult::BestMove {
            sq: Square::A1,
            score: Score::ZERO,
            n_nodes: 0,
            pv_line: vec![],
            depth: 30,
            selectivity,
            is_endgame: true,
            pv_moves: vec![],
            counters: SearchCounters::default(),
        }
    }

    #[test]
    fn midgame_iterations_are_skipped_only_after_a_level2_endgame_solve() {
        let mut search = search_for_tests();

        search.update_endgame_tracking(30, &endgame_result(Selectivity::Level1));
        assert_eq!(search.endgame_start_n_empties, None);

        search.update_endgame_tracking(28, &endgame_result(Selectivity::Level2));
        assert_eq!(search.endgame_start_n_empties, Some(27));
    }

    #[test]
    fn endgame_tracking_is_dropped_by_a_search_with_more_empties() {
        let mut search = search_for_tests();
        search.endgame_start_n_empties = Some(27);

        search.update_endgame_tracking(27, &endgame_result(Selectivity::Level1));
        assert_eq!(search.endgame_start_n_empties, Some(27));

        search.update_endgame_tracking(28, &endgame_result(Selectivity::Level1));
        assert_eq!(search.endgame_start_n_empties, None);
    }

    #[test]
    fn timed_root_starts_in_the_endgame_near_the_end_or_after_a_tracked_solve() {
        let mut search = search_for_tests();
        let endgame = RootPhase::Endgame {
            last: Selectivity::None,
        };
        let midgame = |max_depth| RootPhase::Midgame { max_depth };

        assert_eq!(
            search.timed_root_phase(TIMED_ENDGAME_ROOT_MAX_EMPTIES),
            endgame
        );
        assert_eq!(
            search.timed_root_phase(TIMED_ENDGAME_ROOT_MAX_EMPTIES + 1),
            midgame(TIMED_ENDGAME_ROOT_MAX_EMPTIES + 1)
        );

        search.endgame_start_n_empties = Some(27);
        assert_eq!(search.timed_root_phase(27), endgame);
        assert_eq!(search.timed_root_phase(28), midgame(28));
    }
}
