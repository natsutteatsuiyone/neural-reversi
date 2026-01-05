mod endgame;
pub mod endgame_cache;
pub mod midgame;
pub mod node_type;
pub mod options;
pub mod root_move;
pub mod search_context;
pub mod search_result;
pub mod side_to_move;
pub mod threading;
pub mod time_control;

use std::sync::Arc;

use crate::board::Board;
use crate::constants::MAX_THREADS;
use crate::eval::Eval;
use crate::level::Level;

use crate::search::search_result::SearchResult;
use crate::search::threading::{Thread, ThreadPool};
use crate::search::time_control::{TimeControlMode, TimeManager};
use crate::square::Square;
use crate::transposition_table::{Bound, TranspositionTable};
use crate::types::{Depth, ScaledScore, Scoref, Selectivity};
use crate::{move_list, probcut, stability};

/// Main search engine structure
pub struct Search {
    tt: Arc<TranspositionTable>,
    threads: Arc<ThreadPool>,
    eval: Arc<Eval>,
    endgame_start_n_empties: Option<Depth>,
}

/// Task structure passed to search threads
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

/// Progress information during search
pub struct SearchProgress {
    pub depth: Depth,
    pub target_depth: Depth,
    pub score: Scoref,
    pub best_move: Square,
    pub probability: i32,
    pub nodes: u64,
    pub pv_line: Vec<Square>,
    pub game_phase: GamePhase,
}

/// Type alias for search progress callback
pub type SearchProgressCallback = dyn Fn(SearchProgress) + Send + Sync + 'static;

pub use options::SearchOptions;
pub use search_context::GamePhase;

/// Search constraint definition
pub enum SearchConstraint {
    Level(Level),
    Time(TimeControlMode),
}

impl Search {
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

    pub fn init(&mut self) {
        self.tt.clear();
        self.tt.reset_generation();
        self.eval.cache.clear();
        self.endgame_start_n_empties = None;
    }

    pub fn run<F>(
        &mut self,
        board: &Board,
        constraint: SearchConstraint,
        selectivity: Selectivity,
        multi_pv: bool,
        callback: Option<F>,
    ) -> SearchResult
    where
        F: Fn(SearchProgress) + Send + Sync + 'static,
    {
        let callback = callback.map(|f| Arc::new(f) as Arc<SearchProgressCallback>);

        // Configure time manager and level based on constraint
        let (time_manager, mut effective_level) = match constraint {
            SearchConstraint::Level(level) => (None, level),
            SearchConstraint::Time(mode) => {
                let tm = Arc::new(TimeManager::new(
                    mode,
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
                effective_level.end_depth = [60; 7];
            }
        }

        let mut result = self.execute_search(
            board,
            effective_level,
            selectivity,
            multi_pv,
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
                game_phase: result.game_phase,
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

    pub fn abort(&self) {
        self.threads.stop_timer();
        self.threads.abort_search();
    }

    pub fn is_aborted(&self) -> bool {
        self.threads.is_aborted()
    }

    pub fn get_thread_pool(&self) -> Arc<threading::ThreadPool> {
        self.threads.clone()
    }

    pub fn test(&mut self, board: &Board, level: Level, selectivity: Selectivity) -> SearchResult {
        self.execute_search(board, level, selectivity, false, None, None)
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
    /// SearchResult with the best move found by shallow evaluation
    pub fn quick_move(&self, board: &Board) -> SearchResult {
        use crate::bitboard::BitboardIterator;
        use crate::flip;

        let moves = board.get_moves();
        if moves == 0 {
            // No legal moves - return pass
            return SearchResult {
                score: 0.0,
                best_move: None,
                n_nodes: 0,
                pv_line: vec![],
                depth: 0,
                selectivity: Selectivity::None,
                game_phase: GamePhase::MidGame,
                pv_moves: vec![],
            };
        }

        let mut best_move = Square::None;
        let mut best_score = -ScaledScore::INF;

        // Evaluate each move with depth-1 search
        for sq in BitboardIterator::new(moves) {
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
            n_nodes: BitboardIterator::new(moves).count() as u64,
            pv_line: vec![best_move],
            depth: 1,
            selectivity: Selectivity::None,
            game_phase: GamePhase::MidGame,
            pv_moves: vec![],
        }
    }
}

impl Drop for Search {
    fn drop(&mut self) {
        assert!(Arc::strong_count(&self.threads) == 1);
    }
}

/// Main search entry point that delegates to midgame or endgame search
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let min_end_depth = task.level.get_end_depth(Selectivity::Level0);
    let n_empties = task.board.get_empty_count();

    if min_end_depth >= n_empties {
        return endgame::search_root(task, thread);
    }

    midgame::search_root(task, thread)
}

/// Enhanced Transposition Cutoff
fn enhanced_transposition_cutoff(
    ctx: &mut search_context::SearchContext,
    board: &Board,
    move_list: &move_list::MoveList,
    depth: u32,
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
        let is_endgame = ctx.game_phase == GamePhase::EndGame;
        if let Some(etc_tt_data) = etc_tt_probe_result.data()
            && etc_tt_data.is_endgame() == is_endgame
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
                    ctx.game_phase == GamePhase::EndGame,
                );
                return Some(score);
            }
        }
    }
    None
}
