//! Game tree search engine.
//!
//! Provides the main [`Search`] engine and dispatches root searches to the
//! midgame and endgame drivers; the shared alpha-beta core lives in the
//! `pvs` module.

pub mod context;
pub mod counters;
mod endgame;
pub mod midgame;
pub mod node_type;
pub mod options;
mod pvs;
pub mod result;
pub mod root_move;
pub mod side_to_move;
pub mod stack;
pub mod strategy;
pub mod threading;
pub mod time_control;

#[doc(hidden)]
pub use endgame::{EndGameCaches, null_window_search, solve_last1};
pub use options::{SearchConstraint, SearchRunOptions};
pub use pvs::search;

use std::sync::Arc;

use crate::board::Board;
use crate::constants::MAX_THREADS;
use crate::eval::{Eval, EvalMode};
use crate::flip;
use crate::level::Level;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::search::counters::SearchCounters;
use crate::search::options::{SearchOptions, available_cpus};
use crate::search::result::SearchResult;
use crate::search::threading::{Thread, ThreadPool};
use crate::search::time_control::TimeManager;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, ScaledScore, Scoref};

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
}

/// Shared heavyweight search resources that can back multiple [`Search`]
/// engines concurrently.
///
/// Engines created from the same resource bundle share the transposition table
/// and evaluation network, but each engine gets its own thread pool and
/// endgame-tracking state. This is useful for callers that sometimes need more
/// than one independent search in flight at once, such as GGS synchro child
/// games, without reloading neural-network weights for every worker.
pub struct SearchSharedResources {
    tt: Arc<TranspositionTable>,
    eval: Arc<Eval>,
    n_threads: usize,
}

/// Task descriptor passed to search threads.
///
/// Contains all shared state needed for a search thread to independently
/// execute a search on the given board position.
#[derive(Clone)]
pub struct SearchTask {
    /// Board position to search.
    pub board: Board,
    /// Selectivity level controlling ProbCut pruning aggressiveness.
    pub selectivity: Selectivity,
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

/// Progress information reported during an ongoing search.
pub struct SearchProgress {
    /// Current search depth completed.
    pub depth: Depth,
    /// Target search depth for this iteration.
    pub target_depth: Depth,
    /// Best score found so far (in disc difference).
    pub score: Scoref,
    /// Best move found so far.
    pub best_move: Square,
    /// Probability percentage from the current [`Selectivity`] level.
    pub probability: i32,
    /// Total nodes searched.
    pub nodes: u64,
    /// Principal variation (sequence of best moves).
    pub pv_line: Vec<Square>,
    /// Whether the search is in endgame phase.
    pub is_endgame: bool,
    /// Snapshot of search counters at this point.
    pub counters: SearchCounters,
}

/// Callback invoked to report [`SearchProgress`] during a search.
pub type SearchProgressCallback = dyn Fn(SearchProgress) + Send + Sync + 'static;

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
            tt: Arc::new(TranspositionTable::new(options.tt_mb_size)),
            eval: Arc::new(eval),
            n_threads,
        }
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
        Self {
            tt: shared.tt.clone(),
            threads: ThreadPool::new(shared.n_threads),
            eval: shared.eval.clone(),
            endgame_start_n_empties: None,
        }
    }

    /// Returns a reference to the transposition table.
    pub fn tt(&self) -> &Arc<TranspositionTable> {
        &self.tt
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

    /// Runs a search on the given board position.
    ///
    /// Selects the appropriate search strategy based on the constraint (fixed level
    /// or time-controlled), executes the search, and falls back to [`Search::quick_move`]
    /// if the search is aborted before completing any iteration.
    pub fn run(&mut self, board: &Board, options: &SearchRunOptions) -> SearchResult {
        let callback = options.callback.clone();
        let n_empties = board.get_empty_count();

        let (time_manager, mut effective_level) =
            self.build_time_controls(n_empties, &options.constraint);
        let is_time_mode = time_manager.is_some();

        if is_time_mode {
            self.maybe_extend_endgame_depth(n_empties, &mut effective_level);
        }

        let task = SearchTask {
            board: *board,
            selectivity: options.selectivity,
            tt: self.tt.clone(),
            pool: self.threads.clone(),
            eval: self.eval.clone(),
            level: effective_level,
            multi_pv: options.multi_pv,
            callback: callback.clone(),
            time_manager,
            eval_mode: options.eval_mode,
        };

        let mut result = self.execute_search(task);
        self.apply_fallback_if_invalid(board, &mut result);

        if let Some(callback) = callback {
            callback(progress_from_result(&result));
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
            level.end_depth = Level::perfect().end_depth;
        }
    }

    /// Replaces an aborted-search sentinel result with a shallow
    /// [`Self::quick_move`] fallback.
    ///
    /// When the search is cancelled before finishing a single iteration the
    /// result score is still the initial sentinel; in that case a minimal
    /// best move must still be provided to the caller.
    fn apply_fallback_if_invalid(&self, board: &Board, result: &mut SearchResult) {
        if !result.is_invalid_sentinel() {
            return;
        }
        *result = self.quick_move(board);
    }

    /// Records the empty-square count at which the endgame phase first became
    /// reachable, so future time-controlled searches know to extend their end depth.
    fn update_endgame_tracking(&mut self, n_empties: Depth, result: &SearchResult) {
        if n_empties > 0
            && self.endgame_start_n_empties.is_none()
            && result.depth() + 1 >= n_empties
        {
            self.endgame_start_n_empties = Some(n_empties - 1);
        }
    }

    fn execute_search(&mut self, task: SearchTask) -> SearchResult {
        self.tt.increment_generation();

        let board = task.board;
        let time_manager = task.time_manager.clone();

        let result_receiver = self.threads.start_thinking(task);

        if let Some(tm) = time_manager.as_ref()
            && tm.deadline().is_some()
        {
            self.threads.start_timer(tm.clone());
        }

        let result = result_receiver.recv().unwrap_or_else(|_| {
            // Channel closed - search thread may have panicked. Return fallback.
            self.quick_move(&board)
        });

        self.threads.stop_timer();

        result
    }

    /// Aborts the current search.
    pub fn abort(&self) {
        self.threads.stop_timer();
        self.threads.abort_search();
    }

    /// Returns whether the search has been aborted.
    pub fn is_aborted(&self) -> bool {
        self.threads.is_aborted()
    }

    /// Returns the [`ThreadPool`] used by this search engine.
    ///
    /// [`ThreadPool`]: threading::ThreadPool
    pub fn thread_pool(&self) -> Arc<threading::ThreadPool> {
        self.threads.clone()
    }

    /// Selects a move quickly for time-critical situations.
    ///
    /// Performs a shallow 1-ply search to find the best move when there is
    /// not enough time for a full search. This is a fallback for situations
    /// where the main search would return invalid results.
    pub fn quick_move(&self, board: &Board) -> SearchResult {
        let moves = board.get_moves();
        if moves.is_empty() {
            return SearchResult::NoLegalMove;
        }

        let mut best_move = Square::None;
        let mut best_score = -ScaledScore::INF;

        for sq in moves.iter() {
            let flipped = flip::flip(sq, board.player(), board.opponent());
            let next = board.make_move_with_flipped(flipped, sq);
            let score = -self.eval.evaluate_simple(&next);

            if score > best_score {
                best_score = score;
                best_move = sq;
            }
        }

        SearchResult::BestMove {
            sq: best_move,
            score: best_score.to_disc_diff_f32(),
            n_nodes: moves.count() as u64,
            pv_line: vec![best_move],
            depth: 1,
            selectivity: Selectivity::None,
            is_endgame: false,
            pv_moves: vec![],
            counters: SearchCounters::default(),
        }
    }
}

fn progress_from_result(result: &SearchResult) -> SearchProgress {
    SearchProgress {
        depth: result.depth(),
        target_depth: result.depth(),
        score: result.score().unwrap_or(0.0),
        probability: result.get_probability(),
        best_move: result.best_move().unwrap_or(Square::None),
        nodes: result.n_nodes(),
        pv_line: result.pv_line().to_vec(),
        is_endgame: result.is_endgame(),
        counters: result.counters(),
    }
}

/// Dispatches to midgame or endgame search based on remaining empties.
///
/// Compares the minimum endgame depth from the level configuration against the
/// number of empty squares. If the endgame depth covers all empties, delegates
/// to the endgame solver; otherwise delegates to the midgame search.
fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let min_end_depth = task.level.min_end_depth();
    let n_empties = task.board.get_empty_count();

    if min_end_depth >= n_empties {
        return endgame::search_root(task, thread);
    }

    midgame::search_root(task, thread)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_thread_options() -> SearchOptions {
        SearchOptions::new(0).with_threads(Some(1))
    }

    #[test]
    fn shared_resources_reuse_tt_and_eval_but_create_independent_thread_pools() {
        let shared = SearchSharedResources::new(&one_thread_options());

        let first = Search::from_shared_resources(&shared);
        let second = Search::from_shared_resources(&shared);

        assert!(Arc::ptr_eq(first.tt(), second.tt()));
        assert!(Arc::ptr_eq(&first.eval, &second.eval));

        let first_pool = first.thread_pool();
        let second_pool = second.thread_pool();
        assert!(!Arc::ptr_eq(&first_pool, &second_pool));
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
    fn quick_move_returns_legal_one_ply_result_or_no_legal_move() {
        let search = Search::new(&one_thread_options());
        let board = Board::new();

        let result = search.quick_move(&board);
        let best_move = result.best_move().expect("initial board has legal moves");

        assert!(board.is_legal_move(best_move));
        assert_eq!(result.depth(), 1);
        assert_eq!(result.n_nodes(), board.get_moves().count() as u64);
        assert_eq!(result.pv_line(), &[best_move]);
        assert_eq!(result.selectivity(), Selectivity::None);
        assert!(!result.is_endgame());

        let no_move_board = Board::from_bitboards(Square::A1.bitboard(), 0);
        assert!(matches!(
            search.quick_move(&no_move_board),
            SearchResult::NoLegalMove
        ));
    }
}
