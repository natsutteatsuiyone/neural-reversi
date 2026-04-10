//! Game tree search engine.
//!
//! Provides the main [`Search`] engine, alpha-beta search functions shared by
//! midgame and endgame phases, and parallel search support via split points.

mod endgame;
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

use std::sync::Arc;

use crate::board::Board;
use crate::constants::MAX_THREADS;
use crate::eval::{Eval, EvalMode};
use crate::flip;
use crate::level::Level;
use crate::move_list::MoveList;

use crate::probcut::Selectivity;
use crate::search::node_type::{NodeType, NonPV, PV};
use crate::search::options::SearchOptions;
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
use crate::{probcut, stability};

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

impl Search {
    /// Creates a new search engine with the given options.
    ///
    /// Initializes the evaluation function, transposition table, and thread pool.
    /// The number of threads is clamped to the available CPU count and [`MAX_THREADS`].
    ///
    /// # Panics
    ///
    /// Panics if the evaluation weight files cannot be loaded.
    pub fn new(options: &SearchOptions) -> Search {
        let n_threads = options.n_threads.min(num_cpus::get()).clamp(1, MAX_THREADS);
        let eval = Eval::with_weight_files(
            options.eval_path.as_deref(),
            options.eval_sm_path.as_deref(),
        )
        .unwrap_or_else(|err| panic!("failed to load evaluation weights: {err}"));

        // Ensure that dependent modules are initialized
        probcut::init();
        stability::init();

        Search {
            tt: Arc::new(TranspositionTable::new(options.tt_mb_size)),
            threads: ThreadPool::new(n_threads),
            eval: Arc::new(eval),
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

        // Configure time manager and level based on constraint
        let (time_manager, mut effective_level) = match &options.constraint {
            SearchConstraint::Level(level) => (None, *level),
            SearchConstraint::Time(mode) => {
                let tm = Arc::new(TimeManager::new(
                    *mode,
                    self.threads.get_abort_flag(),
                    board.get_empty_count(),
                ));
                (Some(tm), Level::unlimited())
            }
        };

        // In Time mode, automatically extend endgame search depth once endgame phase is reached
        let is_time_mode = time_manager.is_some();
        let n_empties = board.get_empty_count();
        if is_time_mode && let Some(endgame_start_n_empties) = self.endgame_start_n_empties {
            if n_empties > endgame_start_n_empties {
                self.endgame_start_n_empties = None;
            } else {
                effective_level.end_depth = [60; 4];
            }
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

        // Fallback to quick_move if score is invalid
        // This can happen when search is aborted before completing any iteration
        if result.score == (-ScaledScore::INF).to_disc_diff_f32() {
            let fallback = self.quick_move(board);
            result.score = fallback.score;
            if result.best_move.is_none() {
                result.best_move = fallback.best_move;
                result.pv_line = fallback.pv_line;
            }
        }

        if let Some(callback) = callback {
            callback(SearchProgress {
                depth: result.depth,
                target_depth: result.depth,
                score: result.score,
                probability: result.get_probability(),
                best_move: result.best_move.unwrap_or(Square::None),
                nodes: result.n_nodes,
                pv_line: result.pv_line.clone(),
                is_endgame: result.is_endgame,
                counters: result.counters.clone(),
            });
        }

        if is_time_mode && self.endgame_start_n_empties.is_none() && result.depth + 1 >= n_empties {
            self.endgame_start_n_empties = Some(n_empties - 1);
        }

        result
    }

    fn execute_search(&mut self, task: SearchTask) -> SearchResult {
        self.tt.increment_generation();

        let board = task.board;

        // Start timer thread if we have a deadline
        if let Some(tm) = task.time_manager.as_ref()
            && tm.deadline().is_some()
        {
            self.threads.start_timer(tm.clone());
        }

        let result_receiver = self.threads.start_thinking(task);
        let result = result_receiver.recv().unwrap_or_else(|_| {
            // Channel closed - search thread may have panicked. Return fallback.
            self.quick_move(&board)
        });

        // Stop timer thread
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
    pub fn get_thread_pool(&self) -> Arc<threading::ThreadPool> {
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
            // No legal moves - return pass
            return SearchResult {
                score: 0.0,
                best_move: None,
                n_nodes: 0,
                pv_line: vec![],
                depth: 0,
                selectivity: Selectivity::None,
                is_endgame: false,
                pv_moves: vec![],
                counters: SearchCounters::default(),
            };
        }

        let mut best_move = Square::None;
        let mut best_score = -ScaledScore::INF;

        // Evaluate each move with depth-1 search
        for sq in moves.iter() {
            let flipped = flip::flip(sq, board.player, board.opponent);
            let next = board.make_move_with_flipped(flipped, sq);

            // Evaluate the resulting position (negamax)
            let score = -self.eval.evaluate_simple(&next);

            if score > best_score {
                best_score = score;
                best_move = sq;
            }
        }

        SearchResult {
            score: best_score.to_disc_diff_f32(),
            best_move: Some(best_move),
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
) -> ScaledScore {
    let org_alpha = alpha;

    if NT::PV_NODE {
        ctx.prepare_pv();
        if depth == 0 {
            return SS::evaluate(ctx, board);
        }
    } else {
        if depth <= SS::DEPTH_TO_SHALLOW {
            return SS::shallow_search(ctx, board, depth, alpha, beta);
        }

        if let Some(score) = stability_cutoff(board, ctx.empty_list.count(), alpha.to_disc_diff()) {
            ctx.counters.stability_cuts += 1;
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
            let score = -search::<NT, SS>(ctx, &next, depth, -beta, -alpha, thread);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    // Root node: exclude earlier PV moves (before wipeout/TT shortcuts)
    if NT::ROOT_NODE {
        move_list.exclude_earlier_pv_moves(ctx);
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
    ctx.counters.tt_probes += 1;
    let tt_move = tt_probe_result.best_move();

    // NonPV cutoffs
    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && (!SS::IS_ENDGAME || tt_data.is_endgame())
            && tt_data.depth() >= depth
            && tt_data.selectivity() >= ctx.selectivity
            && tt_data.can_cut(beta)
        {
            ctx.counters.tt_hits += 1;
            return tt_data.score();
        }

        // Enhanced Transposition Cutoff
        if depth >= SS::MIN_ETC_DEPTH {
            ctx.counters.etc_attempts += 1;
            if let Some(score) = enhanced_transposition_cutoff::<SS>(
                ctx,
                board,
                &move_list,
                depth,
                alpha,
                tt_probe_result.index(),
            ) {
                ctx.counters.etc_cuts += 1;
                return score;
            }
        }

        // ProbCut
        if depth >= SS::MIN_PROBCUT_DEPTH {
            ctx.counters.probcut_attempts += 1;
            if let Some(score) = SS::probcut(ctx, board, depth, beta, thread) {
                ctx.counters.probcut_cuts += 1;
                return score;
            }
        }
    }

    let n_moves = move_list.count();
    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;
    let mut move_count: usize = 0;

    if !NT::PV_NODE && n_moves > 1 && tt_move != Square::None {
        // TT move first: search the TT move before expensive move ordering (NonPV only).
        let flipped = flip::flip(tt_move, board.player, board.opponent);
        debug_assert!(!flipped.is_empty());
        move_count = 1;

        let next = board.make_move_with_flipped(flipped, tt_move);
        ctx.update(tt_move, flipped);
        let score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
        ctx.undo(tt_move);

        if thread.cutoff_occurred() || thread.is_search_aborted() {
            return ScaledScore::ZERO;
        }

        best_score = score;
        if score > alpha {
            best_move = tt_move;
            if score >= beta {
                // Beta cutoff — skip move ordering entirely
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
        move_list.evaluate_moves::<SS>(ctx, board, depth, tt_move);
        move_list.sort();
    } else if n_moves == 2 && move_list.get_move(0).sq != tt_move {
        move_list.swap_moves(0, 1);
    }

    // Main move loop
    while move_count < n_moves {
        // Parallel search split
        if move_count >= 1
            && depth >= SS::MIN_SPLIT_DEPTH
            && (n_moves - move_count) >= 2
            && thread.can_split()
        {
            let (s, m, n, split_counters) = thread.split(
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
            );
            best_score = s;
            best_move = m;
            ctx.n_nodes += n;
            ctx.counters.merge(&split_counters);

            if thread.cutoff_occurred() || thread.is_search_aborted() {
                return ScaledScore::ZERO;
            }

            break; // Split consumed all remaining moves
        }

        let mv = move_list.get_move(move_count);
        move_count += 1;

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -ScaledScore::INF;

        let reduction =
            compute_lmr_reduction::<NT, SS>(ctx.selectivity, depth, move_count, n_moves);

        if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV, SS>(
                ctx,
                &next,
                depth - 1 - reduction,
                -(alpha + 1),
                -alpha,
                thread,
            );

            if reduction > 0 && score > alpha {
                score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
            }
        }

        // PV re-search
        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        // Abort check
        if thread.cutoff_occurred() || thread.is_search_aborted() {
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
        Bound::classify_scaled::<NT>(best_score, org_alpha, beta),
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
    let move_iter = split_point.move_iter();
    let n_moves = move_iter.count();

    while let Some((mv, move_count)) = move_iter.next() {
        split_point.unlock();

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let alpha = split_point.state().alpha();
        let mut score = -ScaledScore::INF;

        let reduction =
            compute_lmr_reduction::<NT, SS>(ctx.selectivity, depth, move_count, n_moves);

        if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV, SS>(
                ctx,
                &next,
                depth - 1 - reduction,
                -(alpha + 1),
                -alpha,
                thread,
            );

            if reduction > 0 && score > alpha {
                score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
            }
        }

        // PV re-search
        if NT::PV_NODE && score > alpha {
            let alpha = split_point.state().alpha();
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        split_point.lock();

        // Abort check
        if thread.cutoff_occurred() || thread.is_search_aborted() {
            return ScaledScore::ZERO;
        }

        let sp = split_point.state();

        // Root move update
        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count == 1 || score > sp.alpha());
        }

        // Best score update
        if score > sp.best_score() {
            sp.set_best_score(score);

            if score > sp.alpha() {
                sp.set_best_move(mv.sq);

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                    split_point.state_mut().copy_pv(ctx.get_pv());
                }

                if NT::PV_NODE && score < beta {
                    sp.set_alpha(score);
                    if score >= ScaledScore::MAX {
                        sp.set_cutoff(true);
                        break;
                    }
                } else {
                    sp.set_cutoff(true);
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
