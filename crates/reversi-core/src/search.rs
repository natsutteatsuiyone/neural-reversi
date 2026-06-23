//! Game tree search engine.
//!
//! Provides the main [`Search`] engine, alpha-beta search functions shared by
//! midgame and endgame phases, and parallel search support via split points.

mod endgame;
#[path = "search/endgame/cache.rs"]
pub mod endgame_cache;
pub mod midgame;
pub mod node_type;
pub mod options;
pub mod root_move;
pub mod search_context;
pub mod search_counters;
pub mod search_result;
pub mod search_stack;
pub mod search_strategy;
pub mod side_to_move;
pub mod threading;
pub mod time_control;

#[doc(hidden)]
pub use endgame::{EndGameCaches, null_window_search};

use std::sync::Arc;

use crate::board::Board;
use crate::constants::MAX_THREADS;
use crate::eval::{Eval, EvalMode};
use crate::flip;
use crate::level::Level;
use crate::move_list::MoveList;

use crate::probcut;
use crate::probcut::Selectivity;
use crate::search::node_type::{NodeType, NonPV, PV};
use crate::search::options::{SearchOptions, available_cpus};
use crate::search::search_context::SearchContext;
use crate::search::search_counters::SearchCounters;
use crate::search::search_result::SearchResult;
use crate::search::search_strategy::SearchStrategy;
use crate::search::threading::{SplitPoint, Thread, ThreadPool};
use crate::search::time_control::TimeManager;
use crate::square::Square;
use crate::stability::stability_cutoff;
use crate::transposition_table::{Bound, TranspositionTable};
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

// Re-export SearchConstraint and SearchRunOptions for external use
pub use options::{SearchConstraint, SearchRunOptions};

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
                    self.threads.get_abort_flag(),
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
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let min_end_depth = task.level.min_end_depth();
    let n_empties = task.board.get_empty_count();

    if min_end_depth >= n_empties {
        return endgame::search_root(task, thread);
    }

    midgame::search_root(task, thread)
}

/// Performs enhanced transposition cutoff (ETC) by probing child positions.
///
/// For each move in the move list, checks the transposition table for the resulting
/// position. If a child entry produces a score above alpha with sufficient depth
/// and selectivity, stores a lower-bound entry at the parent and returns the cutoff score.
fn enhanced_transposition_cutoff<SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    move_list: &MoveList,
    depth: Depth,
    alpha: ScaledScore,
    tt_entry_index: usize,
) -> Option<ScaledScore> {
    let etc_depth = depth - 1;
    for mv in move_list.iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.increment_nodes();

        let etc_tt_key = next.hash();
        if let Some(etc_tt_data) = ctx.tt.lookup(&next, etc_tt_key)
            && (!SS::IS_ENDGAME || etc_tt_data.is_endgame())
            && etc_tt_data.depth() >= etc_depth
            && etc_tt_data.selectivity() >= ctx.selectivity
        {
            let score = -etc_tt_data.score();
            if (etc_tt_data.bound() == Bound::Exact || etc_tt_data.bound() == Bound::Upper)
                && score > alpha
            {
                ctx.tt.store(
                    tt_entry_index,
                    board,
                    score,
                    Bound::Lower,
                    depth,
                    mv.sq,
                    ctx.selectivity,
                    SS::IS_ENDGAME,
                );
                return Some(score);
            }
        }
    }
    None
}

/// Searches both midgame and endgame positions using Principal Variation Search.
pub fn search<NT: NodeType, SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: ScaledScore,
    beta: ScaledScore,
    thread: &Arc<Thread>,
    cut_node: bool,
) -> ScaledScore {
    let all_node = !NT::PV_NODE && !cut_node;
    let org_alpha = alpha;

    if NT::PV_NODE {
        if depth == 0 {
            ctx.clear_pv();
            return SS::evaluate(ctx, board);
        }
        ctx.prepare_pv();
    } else {
        if depth <= SS::DEPTH_TO_SHALLOW {
            return SS::shallow_search(ctx, board, depth, alpha, beta, thread);
        }

        if let Some(score) = stability_cutoff(board, ctx.empty_list.count(), alpha.to_disc_diff()) {
            ctx.counters.increment_stability_cut();
            return ScaledScore::from_disc_diff(score);
        }
    }

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    // Move generation
    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let child_cut_node = !NT::PV_NODE && !cut_node;
            let score = -search::<NT, SS>(ctx, &next, depth, -beta, -alpha, thread, child_cut_node);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    // Root node: exclude earlier PV moves (before wipeout/TT shortcuts)
    if NT::ROOT_NODE {
        move_list.exclude_earlier_pv_moves(ctx, board);
        if move_list.count() == 0 {
            return -ScaledScore::INF;
        }
    }

    if let Some(sq) = move_list.wipeout_move() {
        if NT::ROOT_NODE {
            ctx.update_root_move(sq, ScaledScore::MAX, true);
        } else if NT::PV_NODE {
            ctx.update_pv(sq);
        }
        return ScaledScore::MAX;
    }

    // Transposition table probe
    let tt_probe_result = ctx.tt.probe(board, tt_key);
    ctx.counters.increment_tt_probe();
    let tt_move = tt_probe_result.best_move();

    // NonPV cutoffs
    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && tt_data.can_cut(beta, depth, ctx.selectivity, SS::IS_ENDGAME)
        {
            ctx.counters.increment_tt_hit();
            return tt_data.score();
        }

        // Enhanced Transposition Cutoff
        if depth >= SS::MIN_ETC_DEPTH {
            ctx.counters.increment_etc_attempt();
            if let Some(score) = enhanced_transposition_cutoff::<SS>(
                ctx,
                board,
                &move_list,
                depth,
                alpha,
                tt_probe_result.index(),
            ) {
                ctx.counters.increment_etc_cut();
                return score;
            }
        }

        // ProbCut
        if depth >= SS::MIN_PROBCUT_DEPTH {
            ctx.counters.increment_probcut_attempt();
            if let Some(score) = SS::try_probcut(ctx, board, depth, beta, cut_node, thread) {
                ctx.counters.increment_probcut_cut();
                return score;
            }
        }
    }

    let n_moves = move_list.count();
    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;
    let mut move_count: usize = 0;

    if !NT::PV_NODE && (n_moves == 1 || tt_move != Square::None) {
        let (sq, flipped) = if tt_move != Square::None {
            (
                tt_move,
                flip::flip(tt_move, board.player(), board.opponent()),
            )
        } else {
            let mv = move_list.get_move(0);
            (mv.sq, mv.flipped)
        };
        move_count = 1;

        let next = board.make_move_with_flipped(flipped, sq);
        ctx.update(sq, flipped);
        let score = -search::<NonPV, SS>(
            ctx,
            &next,
            depth - 1,
            -(alpha + 1),
            -alpha,
            thread,
            !cut_node,
        );
        ctx.undo(sq);

        if thread.should_stop() {
            return ScaledScore::ZERO;
        }

        best_score = score;
        if score > alpha {
            best_move = sq;
            if score >= beta {
                ctx.tt.store(
                    tt_probe_result.index(),
                    board,
                    best_score,
                    Bound::Lower,
                    depth,
                    best_move,
                    ctx.selectivity,
                    SS::IS_ENDGAME,
                );
                return best_score;
            }
            alpha = score;
        }
    }

    // Move ordering
    // Both branches must ensure the TT move ends up at index 0 when present,
    // so the main loop (starting at move_count=1) skips it correctly.
    if n_moves - move_count > 1 {
        move_list.evaluate_moves::<NT, SS>(ctx, board, depth, tt_move, alpha, cut_node);
        move_list.sort();
    } else if n_moves == 2 && move_list.get_move(0).sq != tt_move {
        move_list.swap_moves(0, 1);
    }

    // Main move loop
    let allow_speculative_split = all_node && depth <= SS::SPECULATIVE_SPLIT_MAX_DEPTH;
    while move_count < n_moves {
        // Parallel search split
        if (move_count >= 1 || allow_speculative_split)
            && depth >= SS::MIN_SPLIT_DEPTH
            && (n_moves - move_count) >= 2
            && thread.can_split()
        {
            let (s, m, split_counters) = thread.split(
                ctx,
                board,
                alpha,
                beta,
                best_score,
                best_move,
                depth,
                move_list,
                move_count,
                NT::ID,
                SS::IS_ENDGAME,
                cut_node,
            );
            best_score = s;
            best_move = m;
            ctx.counters.merge(&split_counters);

            if thread.should_stop() {
                return ScaledScore::ZERO;
            }

            break; // Split consumed all remaining moves
        }

        let mv = move_list.get_move(move_count);
        move_count += 1;

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -ScaledScore::INF;

        if !NT::PV_NODE || move_count > 1 {
            let reduction =
                compute_lmr_reduction::<NT, SS>(ctx.selectivity, depth, move_count, n_moves);

            score = -search::<NonPV, SS>(
                ctx,
                &next,
                depth - 1 - reduction,
                -(alpha + 1),
                -alpha,
                thread,
                reduction > 0 || !cut_node,
            );

            if reduction > 0 && score > alpha {
                score = -search::<NonPV, SS>(
                    ctx,
                    &next,
                    depth - 1,
                    -(alpha + 1),
                    -alpha,
                    thread,
                    !cut_node,
                );
            }
        }

        // PV re-search
        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread, false);
        }

        ctx.undo(mv.sq);

        // Abort check
        if thread.should_stop() {
            return ScaledScore::ZERO;
        }

        // Root move update
        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count == 1 || score > alpha);
        }

        // Best score update
        if score > best_score {
            best_score = score;

            if score > alpha {
                best_move = mv.sq;

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                }

                if NT::PV_NODE && score < beta {
                    alpha = score;
                    if alpha >= ScaledScore::MAX {
                        break;
                    }
                } else {
                    break; // Beta cutoff
                }
            }
        }
    }

    // Store in transposition table
    ctx.tt.store(
        tt_probe_result.index(),
        board,
        best_score,
        Bound::classify::<NT>(best_score, org_alpha, beta),
        depth,
        best_move,
        ctx.selectivity,
        SS::IS_ENDGAME,
    );

    best_score
}

/// Searches remaining moves at a split point in parallel search.
///
/// Called by helper threads that join an existing split point. Picks moves from
/// the shared move iterator, searches them, and updates the split
/// point's best score/move under its lock.
pub fn search_split_point<NT: NodeType, SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    thread: &Arc<Thread>,
    split_point: &Arc<SplitPoint>,
) -> ScaledScore {
    let beta = split_point.state().beta;
    let cut_node = split_point.state().cut_node;
    let move_iter = split_point.move_iter();
    let n_moves = move_iter.count();

    while let Some((mv, move_count)) = move_iter.next() {
        split_point.unlock();

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let alpha = split_point.state().alpha();

        debug_assert!(!NT::PV_NODE || move_count > 1);
        let reduction =
            compute_lmr_reduction::<NT, SS>(ctx.selectivity, depth, move_count, n_moves);

        let mut score = -search::<NonPV, SS>(
            ctx,
            &next,
            depth - 1 - reduction,
            -(alpha + 1),
            -alpha,
            thread,
            reduction > 0 || !cut_node,
        );

        if reduction > 0 && score > alpha {
            score = -search::<NonPV, SS>(
                ctx,
                &next,
                depth - 1,
                -(alpha + 1),
                -alpha,
                thread,
                !cut_node,
            );
        }

        // PV re-search
        if NT::PV_NODE && score > alpha {
            let alpha = split_point.state().alpha();
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread, false);
        }

        ctx.undo(mv.sq);

        split_point.lock();

        // Abort check
        if thread.should_stop() {
            return ScaledScore::ZERO;
        }

        let sp = split_point.state();

        // Root move update
        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, score > sp.alpha());
        }

        // Best score update
        if score > sp.best_score() {
            sp.set_best_score(score);

            if score > sp.alpha() {
                sp.set_best_move(mv.sq);

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                    split_point.copy_pv(ctx.get_pv());
                }

                if NT::PV_NODE && score < beta {
                    sp.set_alpha(score);
                    if score >= ScaledScore::MAX {
                        thread.mark_split_point_cutoff(sp);
                        break;
                    }
                } else {
                    thread.mark_split_point_cutoff(sp);
                    break;
                }
            }
        }
    }

    split_point.state().best_score()
}

/// Computes the LMR depth reduction for late moves.
///
/// Disabled for endgame search, PV nodes, and ProbCut verification search
/// (selectivity disabled).
#[inline(always)]
fn compute_lmr_reduction<NT: NodeType, SS: SearchStrategy>(
    selectivity: Selectivity,
    depth: Depth,
    move_count: usize,
    n_moves: usize,
) -> Depth {
    if !SS::IS_ENDGAME
        && !NT::PV_NODE
        && selectivity.is_enabled()
        && depth >= midgame::LMR_MIN_DEPTH
        && move_count > 2
        && n_moves >= 4
    {
        if depth >= midgame::LMR_DEEPER_DEPTH && move_count > 5 {
            2
        } else {
            1
        }
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probcut::Selectivity;
    use crate::search::midgame::{LMR_DEEPER_DEPTH, LMR_MIN_DEPTH};
    use crate::search::node_type::{NonPV, PV};
    use crate::search::search_strategy::{EndGameStrategy, MidGameStrategy};

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

    #[test]
    fn no_reduction_below_the_gating_thresholds() {
        // Depth below LMR_MIN_DEPTH.
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_MIN_DEPTH - 1,
                10,
                10
            ),
            0
        );
        // move_count must exceed 2.
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                2,
                10
            ),
            0
        );
        // n_moves must be at least 4.
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                6,
                3
            ),
            0
        );
        // ProbCut verification search runs with selectivity disabled.
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::None,
                LMR_DEEPER_DEPTH,
                6,
                10
            ),
            0
        );
    }

    #[test]
    fn shallow_late_moves_reduce_by_one() {
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_MIN_DEPTH,
                3,
                4
            ),
            1
        );
        // Deep enough, but not enough late moves for a two-ply reduction.
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                5,
                10
            ),
            1
        );
        // Many late moves, but not deep enough for a two-ply reduction.
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH - 1,
                6,
                10
            ),
            1
        );
    }

    #[test]
    fn deep_and_late_moves_reduce_by_two() {
        assert_eq!(
            compute_lmr_reduction::<NonPV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                6,
                10
            ),
            2
        );
    }

    #[test]
    fn pv_and_endgame_nodes_are_never_reduced() {
        assert_eq!(
            compute_lmr_reduction::<PV, MidGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                6,
                10
            ),
            0
        );
        assert_eq!(
            compute_lmr_reduction::<NonPV, EndGameStrategy>(
                Selectivity::Level1,
                LMR_DEEPER_DEPTH,
                6,
                10
            ),
            0
        );
    }
}
