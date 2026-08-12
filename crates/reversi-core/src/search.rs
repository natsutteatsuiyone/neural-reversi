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
pub mod strategy;
pub mod threading;
pub mod time_control;

#[doc(hidden)]
pub use endgame::{EndGameCaches, null_window_search, solve_last1};
pub use options::{SearchConstraint, SearchRunOptions};
pub use progress::{SearchProgress, SearchProgressCallback};
pub(crate) use pvs::{LMR_MIN_DEPTH, search};

use std::sync::Arc;

use crate::board::Board;
use crate::constants::{MAX_PLY, MAX_THREADS};
use crate::eval::{Eval, EvalMode};
use crate::level::Level;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::search::context::SearchContext;
use crate::search::counters::SearchCounters;
use crate::search::options::{SearchOptions, available_cpus};
use crate::search::result::SearchResult;
use crate::search::threading::{Thread, ThreadPool};
use crate::search::time_control::TimeManager;
use crate::square::Square;
use crate::transposition_table::{Bound, TranspositionTable};
use crate::types::{Depth, ScaledScore};

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
/// Engines created from the same resource bundle share the evaluation network;
/// the transposition table and thread pool are per-engine. Useful for callers
/// that need more than one independent search in flight, such as GGS synchro
/// child games, without reloading neural-network weights for every worker.
pub struct SearchSharedResources {
    eval: Arc<Eval>,
    tt_mb_size: usize,
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
    /// Midgame selectivity: [`Selectivity::Level1`] (73%) for fixed-level searches,
    /// [`Selectivity::Mid`] (63%) for time-controlled searches, and
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
            tt: Arc::new(TranspositionTable::new(shared.tt_mb_size)),
            threads: ThreadPool::new(shared.n_threads),
            eval: shared.eval.clone(),
            endgame_start_n_empties: None,
        }
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

        let mut result = self.execute_search(task);
        self.apply_fallback_if_invalid(board, &mut result);

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
            let next = board.make_move(sq);
            let score = -self.eval.evaluate_simple(&next, EvalMode::Main);

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

/// Re-stores a completed principal variation into the transposition table.
///
/// Walks `pv` from `board`, re-inserting the forced passes that PV lines omit,
/// and stores an exact entry wherever the table no longer holds the PV move.
/// `score` and `depth` describe the head position and are negamax-mirrored and
/// decremented along the line.
pub(crate) fn store_pv_in_tt(
    ctx: &SearchContext,
    board: &Board,
    pv: &[Square],
    mut score: ScaledScore,
    mut depth: Depth,
    is_endgame: bool,
) {
    let mut board = *board;
    let mut protected_indices = [0; MAX_PLY];
    let mut protected_len = 0;

    for &sq in pv {
        if !board.is_legal_move(sq) {
            board = board.switch_players();
            score = -score;
        }

        if let Some(probe) =
            ctx.tt
                .probe_for_pv(&board, board.hash(), &protected_indices[..protected_len])
        {
            let index = probe.index();
            if probe.best_move() != sq {
                ctx.tt.store(
                    index,
                    &board,
                    score,
                    Bound::Exact,
                    depth,
                    sq,
                    ctx.selectivity,
                    is_endgame,
                );
            }
            protected_indices[protected_len] = index;
            protected_len += 1;
        }

        board = board.make_move(sq);
        score = -score;
        depth -= 1;
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
    use crate::search::time_control::TimeControlMode;
    use std::sync::OnceLock;

    fn one_thread_options() -> SearchOptions {
        SearchOptions::new(0).with_threads(Some(1))
    }

    #[test]
    fn fixed_level_uses_73_percent_midgame_selectivity() {
        let options = SearchRunOptions::with_level(Level::uniform(8, 8));
        let selectivity = midgame_selectivity(&options);

        assert_eq!(selectivity, Selectivity::Level1);
        assert_eq!(selectivity.probability(), 73);
    }

    #[test]
    fn time_constraint_keeps_63_percent_midgame_selectivity() {
        let options = SearchRunOptions::with_time(TimeControlMode::Infinite);
        let selectivity = midgame_selectivity(&options);

        assert_eq!(selectivity, Selectivity::Mid);
        assert_eq!(selectivity.probability(), 63);
    }

    #[test]
    fn disabling_probcut_overrides_constraint_midgame_selectivity() {
        let fixed = SearchRunOptions::with_level(Level::uniform(8, 8)).disable_probcut();
        let timed = SearchRunOptions::with_time(TimeControlMode::Infinite).disable_probcut();

        assert_eq!(midgame_selectivity(&fixed), Selectivity::None);
        assert_eq!(midgame_selectivity(&timed), Selectivity::None);
    }

    fn pv_store_ctx(board: &Board, mb_size: usize) -> SearchContext {
        static EVAL: OnceLock<Arc<Eval>> = OnceLock::new();
        let eval = EVAL
            .get_or_init(|| {
                Arc::new(
                    Eval::with_weight_files(None, None)
                        .expect("embedded evaluation weights must load"),
                )
            })
            .clone();
        SearchContext::new(
            board,
            Selectivity::None,
            Arc::new(TranspositionTable::new(mb_size)),
            eval,
        )
    }

    /// Stores an exact entry for `board` into the slot its probe selects.
    fn seed_tt(
        tt: &TranspositionTable,
        board: &Board,
        score: ScaledScore,
        depth: Depth,
        best_move: Square,
    ) {
        let probe = tt.probe(board, board.hash());
        tt.store(
            probe.index(),
            board,
            score,
            Bound::Exact,
            depth,
            best_move,
            Selectivity::None,
            false,
        );
    }

    #[test]
    fn store_pv_in_tt_stores_exact_entries_with_negamax_mirroring() {
        let board = Board::new();
        let mut line = Vec::new();
        let mut b = board;
        for _ in 0..3 {
            let sq = b.get_moves().iter().next().unwrap();
            line.push(sq);
            b = b.make_move(sq);
        }
        let ctx = pv_store_ctx(&board, 1);
        let score = ScaledScore::from_disc_diff(4);

        store_pv_in_tt(&ctx, &board, &line, score, 8, false);

        let mut walk = board;
        let mut expected_score = score;
        let mut expected_depth = 8;
        for &sq in &line {
            let data = ctx
                .tt
                .lookup(&walk, walk.hash())
                .expect("every PV position must be stored");
            assert_eq!(data.best_move(), sq);
            assert_eq!(data.score(), expected_score);
            assert_eq!(data.depth(), expected_depth);
            assert_eq!(data.bound(), Bound::Exact);
            walk = walk.make_move(sq);
            expected_score = -expected_score;
            expected_depth -= 1;
        }
    }

    #[test]
    fn store_pv_in_tt_preserves_prefixes_when_a_full_cluster_collides() {
        let board = Board::new();
        let ctx = pv_store_ctx(&board, 0);
        let cluster = |b: &Board| ctx.tt.get_cluster_idx(b.hash());

        // Walk the greedy PV until a position shares a cluster with an earlier
        // one; a 16-cluster table makes that happen within a few plies.
        let mut line = Vec::new();
        let mut positions: Vec<Board> = Vec::new();
        let mut walk = board;
        let (earlier_idx, later_idx) = loop {
            if !walk.has_legal_moves() {
                walk = walk.switch_players();
            }
            let collision = positions.iter().position(|p| cluster(p) == cluster(&walk));
            let sq = walk.get_moves().iter().next().unwrap();
            positions.push(walk);
            line.push(sq);

            if let Some(earlier_idx) = collision {
                break (earlier_idx, positions.len() - 1);
            }

            walk = walk.make_move(sq);
        };

        // A deeper unrelated entry in the same cluster makes the earlier PV slot
        // the preferred victim, so the prefix survives only if it is protected.
        let blocker = (1..=10_000u64)
            .map(|player| Board::from_bitboards(player, 0))
            .find(|b| cluster(b) == cluster(&positions[later_idx]) && !positions.contains(b))
            .expect("a colliding unrelated board must exist");
        seed_tt(&ctx.tt, &blocker, ScaledScore::ZERO, 60, Square::A1);

        store_pv_in_tt(
            &ctx,
            &board,
            &line,
            ScaledScore::ZERO,
            line.len() as Depth,
            false,
        );

        for idx in [earlier_idx, later_idx] {
            let position = positions[idx];
            let data = ctx
                .tt
                .lookup(&position, position.hash())
                .unwrap_or_else(|| panic!("PV position {idx} must remain probeable"));
            assert_eq!(data.best_move(), line[idx]);
        }
    }

    #[test]
    fn store_pv_in_tt_reinserts_omitted_passes() {
        let board = Board::new().make_move(Square::D3);
        let after_pass = board.switch_players();
        let sq = after_pass
            .get_moves()
            .iter()
            .find(|&sq| !board.is_legal_move(sq))
            .expect("a move legal only after a pass must exist");
        let ctx = pv_store_ctx(&board, 1);
        let score = ScaledScore::from_disc_diff(6);

        store_pv_in_tt(&ctx, &board, &[sq], score, 5, false);

        let data = ctx
            .tt
            .lookup(&after_pass, after_pass.hash())
            .expect("the pass-adjusted position must be stored");
        assert_eq!(data.best_move(), sq);
        assert_eq!(data.score(), -score);
        assert_eq!(data.depth(), 5);
    }

    #[test]
    fn store_pv_in_tt_keeps_entries_that_already_hold_the_pv_move() {
        let board = Board::new();
        let sq = board.get_moves().iter().next().unwrap();
        let ctx = pv_store_ctx(&board, 1);
        seed_tt(&ctx.tt, &board, ScaledScore::from_disc_diff(2), 20, sq);

        store_pv_in_tt(
            &ctx,
            &board,
            &[sq],
            ScaledScore::from_disc_diff(4),
            6,
            false,
        );

        let data = ctx.tt.lookup(&board, board.hash()).unwrap();
        assert_eq!(data.depth(), 20);
        assert_eq!(data.score(), ScaledScore::from_disc_diff(2));
    }

    #[test]
    fn shared_resources_reuse_eval_but_keep_tt_and_thread_pool_per_engine() {
        let shared = SearchSharedResources::new(&SearchOptions::new(1).with_threads(Some(1)));

        let first = Search::from_shared_resources(&shared);
        let second = Search::from_shared_resources(&shared);

        assert!(Arc::ptr_eq(&first.eval, &second.eval));
        assert!(!Arc::ptr_eq(first.tt(), second.tt()));
        assert_eq!(first.tt().mb_size(), 1);
        assert!(!Arc::ptr_eq(&first.thread_pool(), &second.thread_pool()));
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
