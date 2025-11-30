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
use crate::search::search_context::GamePhase;
use crate::search::search_result::SearchResult;
use crate::search::threading::{Thread, ThreadPool};
use crate::search::time_control::{TimeControlMode, TimeManager};
use crate::square::Square;
use crate::transposition_table::{Bound, TranspositionTable};
use crate::types::{Depth, Score, Scoref, Selectivity};
use crate::{move_list, probcut, stability};

/// Main search engine structure
pub struct Search {
    tt: Arc<TranspositionTable>,
    generation: u8,
    threads: Arc<ThreadPool>,
    eval: Arc<Eval>,
    endgame_start_n_empties: Option<Depth>,
}

/// Task structure passed to search threads
#[derive(Clone)]
pub struct SearchTask {
    pub board: Board,
    pub generation: u8,
    pub selectivity: u8,
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
    pub score: Scoref,
    pub best_move: Square,
    pub probability: i32,
}

/// Type alias for search progress callback
pub type SearchProgressCallback = dyn Fn(SearchProgress) + Send + Sync + 'static;

pub use options::SearchOptions;

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
            generation: 0,
            threads: ThreadPool::new(n_threads),
            eval: Arc::new(eval),
            endgame_start_n_empties: None,
        }
    }

    pub fn init(&mut self) {
        self.tt.clear();
        self.eval.cache.clear();
        self.generation = 0;
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
        use crate::constants::SCORE_INF;
        use crate::types::Scoref;

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

        let n_empties = board.get_empty_count();
        if let Some(endgame_start_n_empties) = self.endgame_start_n_empties {
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
        if result.score == (-SCORE_INF) as Scoref {
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
                score: result.score,
                probability: result.get_probability(),
                best_move: result.best_move.unwrap_or(Square::None),
            });
        }

        if self.endgame_start_n_empties.is_none()
            && result.depth + 1 >= n_empties
            && result.selectivity >= 3
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
        self.generation += 1;

        // Get deadline for timer thread
        let timer_time_manager = time_manager.clone();

        let task = SearchTask {
            board: *board,
            generation: self.generation,
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
        use crate::constants::{SCORE_INF, unscale_score_f32};
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
                selectivity: 0,
                game_phase: GamePhase::MidGame,
            };
        }

        let mut best_move = Square::None;
        let mut best_score = -SCORE_INF;

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
            score: unscale_score_f32(best_score),
            best_move: Some(best_move),
            n_nodes: BitboardIterator::new(moves).count() as u64,
            pv_line: vec![best_move],
            depth: 1,
            selectivity: 0,
            game_phase: GamePhase::MidGame,
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
    let min_end_depth = task.level.get_end_depth(0);
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
    alpha: Score,
    tt_key: u64,
    tt_entry_index: usize,
) -> Option<Score> {
    let etc_depth = depth - 1;
    for mv in move_list.iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.increment_nodes();

        let etc_tt_key = next.hash();
        let (etc_tt_hit, etc_tt_data, _tt_entry_index) = ctx.tt.probe(etc_tt_key, ctx.generation);
        let is_endgame = ctx.game_phase == GamePhase::EndGame;
        if etc_tt_hit
            && etc_tt_data.is_endgame == is_endgame
            && etc_tt_data.depth >= etc_depth
            && etc_tt_data.selectivity >= ctx.selectivity
        {
            let score = -etc_tt_data.score;
            if (etc_tt_data.bound == Bound::Exact || etc_tt_data.bound == Bound::Upper)
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
                    ctx.generation,
                    ctx.game_phase == GamePhase::EndGame,
                );
                return Some(score);
            }
        }
    }
    None
}
