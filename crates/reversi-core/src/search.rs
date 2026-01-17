//! Game tree search engine.

mod endgame;
pub mod endgame_cache;
pub mod midgame;
pub mod node_type;
pub mod options;
pub mod root_move;
pub mod search_context;
pub mod search_result;
pub mod search_strategy;
pub mod side_to_move;
pub mod threading;
pub mod time_control;

use std::sync::Arc;

use crate::board::Board;
use crate::constants::MAX_THREADS;
use crate::eval::Eval;
use crate::level::Level;
use crate::move_list::{ConcurrentMoveIterator, MoveList};

use crate::probcut::Selectivity;
use crate::search::node_type::{NodeType, NonPV, PV};
use crate::search::options::SearchOptions;
use crate::search::search_context::SearchContext;
use crate::search::search_result::SearchResult;
use crate::search::search_strategy::SearchStrategy;
use crate::search::threading::{SplitPoint, Thread, ThreadPool};
use crate::search::time_control::TimeManager;
use crate::square::Square;
use crate::stability::stability_cutoff;
use crate::transposition_table::{Bound, TranspositionTable};
use crate::types::{Depth, ScaledScore, Scoref};
use crate::{probcut, stability};

/// Main search engine structure.
pub struct Search {
    tt: Arc<TranspositionTable>,
    threads: Arc<ThreadPool>,
    eval: Arc<Eval>,
    endgame_start_n_empties: Option<Depth>,
}

/// Task structure passed to search threads.
#[derive(Clone)]
pub struct SearchTask {
    pub board: Board,
    pub selectivity: Selectivity,
    pub tt: Arc<TranspositionTable>,
    pub pool: Arc<ThreadPool>,
    pub eval: Arc<Eval>,
    pub level: Level,
    pub multi_pv: bool,
    pub callback: Option<Arc<SearchProgressCallback>>,
    pub time_manager: Option<Arc<TimeManager>>,
}

/// Progress information during search.
pub struct SearchProgress {
    pub depth: Depth,
    pub target_depth: Depth,
    pub score: Scoref,
    pub best_move: Square,
    pub probability: i32,
    pub nodes: u64,
    pub pv_line: Vec<Square>,
    pub is_endgame: bool,
}

/// Type alias for search progress callback.
pub type SearchProgressCallback = dyn Fn(SearchProgress) + Send + Sync + 'static;

// Re-export SearchConstraint and SearchRunOptions for external use
pub use options::{SearchConstraint, SearchRunOptions};

impl Search {
    /// Creates a new search engine with the given options.
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

    /// Resets the search state for a new game.
    pub fn init(&mut self) {
        self.tt.clear();
        self.tt.reset_generation();
        self.eval.clear_cache();
        self.endgame_start_n_empties = None;
    }

    /// Runs a search on the given board position.
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
                effective_level.end_depth = [60; 6];
            }
        }

        let mut result = self.execute_search(
            board,
            effective_level,
            options.selectivity,
            options.multi_pv,
            callback.clone(),
            time_manager,
        );

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
            });
        }

        if is_time_mode
            && self.endgame_start_n_empties.is_none()
            && result.depth + 1 >= n_empties
            && result.selectivity >= Selectivity::Level3
        {
            self.endgame_start_n_empties = Some(n_empties - 1);
        }

        result
    }

    fn execute_search(
        &mut self,
        board: &Board,
        level: Level,
        selectivity: Selectivity,
        multi_pv: bool,
        callback: Option<Arc<SearchProgressCallback>>,
        time_manager: Option<Arc<TimeManager>>,
    ) -> SearchResult {
        self.tt.increment_generation();

        // Get deadline for timer thread
        let timer_time_manager = time_manager.clone();

        let task = SearchTask {
            board: *board,
            selectivity,
            tt: self.tt.clone(),
            pool: self.threads.clone(),
            eval: self.eval.clone(),
            level,
            multi_pv,
            callback,
            time_manager,
        };

        // Start timer thread if we have a deadline
        if let Some(tm) = timer_time_manager
            && tm.deadline().is_some()
        {
            self.threads.start_timer(tm);
        }

        let result_receiver = self.threads.start_thinking(task);
        let result = result_receiver.recv().unwrap();

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

    /// Returns the thread pool used by this search engine.
    pub fn get_thread_pool(&self) -> Arc<threading::ThreadPool> {
        self.threads.clone()
    }

    /// Quick move selection for time-critical situations.
    ///
    /// Performs a shallow 1-ply search to find the best move when there's
    /// not enough time for a full search. This is a fallback for situations
    /// where the main search would return invalid results.
    ///
    /// # Arguments
    ///
    /// * `board` - The board position to search
    ///
    /// # Returns
    ///
    /// SearchResult with the best move found by shallow evaluation.
    pub fn quick_move(&self, board: &Board) -> SearchResult {
        use crate::flip;

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
        }
    }
}

/// Dispatches to midgame or endgame search based on remaining empties.
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let min_end_depth = task.level.get_end_depth(Selectivity::Level1);
    let n_empties = task.board.get_empty_count();

    if min_end_depth >= n_empties {
        return endgame::search_root(task, thread);
    }

    midgame::search_root(task, thread)
}

/// Checks child positions in TT for potential cutoffs.
fn enhanced_transposition_cutoff<SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    move_list: &MoveList,
    depth: Depth,
    alpha: ScaledScore,
    tt_key: u64,
    tt_entry_index: usize,
) -> Option<ScaledScore> {
    let etc_depth = depth - 1;
    for mv in move_list.iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.increment_nodes();

        let etc_tt_key = next.hash();
        let etc_tt_probe_result = ctx.tt.probe(etc_tt_key);
        if let Some(etc_tt_data) = etc_tt_probe_result.data()
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
                    tt_key,
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

/// Search function for both midgame and endgame positions.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
/// * `SS` - Search strategy (MidGamePhase or EndGamePhase) determining phase-specific logic.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `depth` - Remaining search depth (for endgame, equals n_empties).
/// * `alpha` - Alpha bound.
/// * `beta` - Beta bound.
/// * `thread` - Thread handle for parallel search.
///
/// # Returns
///
/// Best score found.
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
        if depth == 0 {
            return SS::evaluate(ctx, board);
        }
    } else {
        if depth <= SS::DEPTH_TO_SHALLOW {
            return SS::shallow_search(ctx, board, depth, alpha, beta);
        }

        if let Some(score) = stability_cutoff(board, ctx.empty_list.count, alpha.to_disc_diff()) {
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
            return board.solve_scaled(ctx.empty_list.count);
        }
    } else if let Some(sq) = move_list.wipeout_move() {
        if NT::ROOT_NODE {
            ctx.update_root_move(sq, ScaledScore::MAX, 1, alpha);
        } else if NT::PV_NODE {
            ctx.update_pv(sq);
        }
        return ScaledScore::MAX;
    }

    // Transposition table probe
    let tt_probe_result = ctx.tt.probe(tt_key);
    let tt_move = tt_probe_result.best_move();

    // NonPV cutoffs
    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && (!SS::IS_ENDGAME || tt_data.is_endgame())
            && tt_data.depth() >= depth
            && tt_data.selectivity() >= ctx.selectivity
            && tt_data.can_cut(beta)
        {
            return tt_data.score();
        }

        // Enhanced Transposition Cutoff
        if depth >= SS::MIN_ETC_DEPTH
            && let Some(score) = enhanced_transposition_cutoff::<SS>(
                ctx,
                board,
                &move_list,
                depth,
                alpha,
                tt_key,
                tt_probe_result.index(),
            )
        {
            return score;
        }

        // ProbCut
        if depth >= SS::MIN_PROBCUT_DEPTH
            && let Some(score) = SS::probcut(ctx, board, depth, beta, thread)
        {
            return score;
        }
    }

    // Root node: exclude earlier PV moves
    if NT::ROOT_NODE {
        move_list.exclude_earlier_pv_moves(ctx);
    }

    // Move ordering
    if move_list.count() > 1 {
        move_list.evaluate_moves::<SS>(ctx, board, depth, tt_move);
        move_list.sort();
    }

    let move_iter = Arc::new(ConcurrentMoveIterator::new(move_list));
    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;

    // Main move loop
    while let Some((mv, move_count)) = move_iter.next() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -ScaledScore::INF;

        // Score-based Reduction (midgame only)
        if SS::USE_SBR && depth >= 2 && mv.reduction_depth > 0 {
            let d = (depth - 1).saturating_sub(mv.reduction_depth);
            score = -search::<NonPV, SS>(ctx, &next, d, -(alpha + 1), -alpha, thread);
            if score > alpha {
                score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
            }
        } else if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
        }

        // PV re-search
        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            ctx.clear_pv();
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        // Abort check
        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return ScaledScore::ZERO;
        }

        // Root move update
        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, alpha);
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
                } else {
                    break; // Beta cutoff
                }
            }
        }

        // Parallel search split
        if depth >= SS::MIN_SPLIT_DEPTH && move_iter.remaining() >= 2 && thread.can_split() {
            let (s, m, n) = thread.split(
                ctx,
                board,
                alpha,
                beta,
                best_score,
                best_move,
                depth,
                &move_iter,
                NT::TYPE_ID,
                SS::IS_ENDGAME,
            );
            best_score = s;
            best_move = m;
            ctx.n_nodes += n;

            if thread.is_search_aborted() || thread.cutoff_occurred() {
                return ScaledScore::ZERO;
            }

            if best_score >= beta {
                break;
            }
        }
    }

    // Store in transposition table
    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        best_score,
        Bound::classify::<NT>(best_score.value(), org_alpha.value(), beta.value()),
        depth,
        best_move,
        ctx.selectivity,
        SS::IS_ENDGAME,
    );

    best_score
}

/// Search function for split-point nodes in parallel search.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
/// * `SS` - Search strategy (MidGamePhase or EndGamePhase) determining phase-specific logic.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `depth` - Remaining search depth.
/// * `thread` - Thread handle for parallel search.
/// * `split_point` - Split point for work distribution.
///
/// # Returns
///
/// Best score found.
pub fn search_split_point<NT: NodeType, SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    thread: &Arc<Thread>,
    split_point: &Arc<SplitPoint>,
) -> ScaledScore {
    let beta = split_point.state().beta;
    let move_iter = split_point.state().move_iter.clone().unwrap();

    while let Some((mv, move_count)) = move_iter.next() {
        split_point.unlock();

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let alpha = split_point.state().alpha();
        let mut score = -ScaledScore::INF;

        // Score-based Reduction (midgame only)
        if SS::USE_SBR && depth >= 2 && mv.reduction_depth > 0 {
            let d = (depth - 1).saturating_sub(mv.reduction_depth);
            score = -search::<NonPV, SS>(ctx, &next, d, -(alpha + 1), -alpha, thread);
            if score > alpha {
                let alpha = split_point.state().alpha();
                score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
            }
        } else if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread);
        }

        // PV re-search
        if NT::PV_NODE && score > alpha {
            ctx.clear_pv();
            let alpha = split_point.state().alpha();
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        split_point.lock();

        // Abort check
        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return ScaledScore::ZERO;
        }

        let sp = split_point.state();

        // Root move update
        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, sp.alpha());
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
                } else {
                    sp.set_cutoff(true);
                    break;
                }
            }
        }
    }

    split_point.state().best_score()
}
