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

use std::sync::Arc;

use search_result::SearchResult;
use threading::{Thread, ThreadPool};

use crate::board::Board;
use crate::eval::Eval;
use crate::level::Level;
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
}

/// Task structure passed to search threads
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

impl Search {
    pub fn new(options: &SearchOptions) -> Search {
        let n_threads = options.n_threads.min(num_cpus::get()).max(1);
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
        }
    }

    pub fn init(&mut self) {
        self.tt.clear();
        self.eval.cache.clear();
        self.generation = 0;
    }

    pub fn run(
        &mut self,
        board: &Board,
        level: Level,
        selectivity: Selectivity,
        multi_pv: bool,
    ) -> SearchResult {
        self.run_with_callback::<fn(SearchProgress)>(board, level, selectivity, multi_pv, None)
    }

    pub fn run_with_callback<F>(
        &mut self,
        board: &Board,
        level: Level,
        selectivity: Selectivity,
        multi_pv: bool,
        callback: Option<F>,
    ) -> SearchResult
    where
        F: Fn(SearchProgress) + Send + Sync + 'static,
    {
        let callback = callback.map(|f| Arc::new(f) as Arc<SearchProgressCallback>);
        self.execute_search(board, level, selectivity, multi_pv, callback)
    }

    fn execute_search(
        &mut self,
        board: &Board,
        level: Level,
        selectivity: Selectivity,
        multi_pv: bool,
        callback: Option<Arc<SearchProgressCallback>>,
    ) -> SearchResult {
        self.generation += 1;

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
        };

        let result_receiver = self.threads.start_thinking(task);
        result_receiver.recv().unwrap()
    }

    pub fn abort(&self) {
        self.threads.abort_search();
    }

    pub fn is_aborted(&self) -> bool {
        self.threads.is_aborted()
    }

    pub fn get_thread_pool(&self) -> Arc<threading::ThreadPool> {
        self.threads.clone()
    }

    pub fn test(&mut self, board: &Board, level: Level, selectivity: Selectivity) -> SearchResult {
        self.execute_search(board, level, selectivity, false, None)
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

    if min_end_depth < n_empties {
        midgame::search_root(task, thread)
    } else {
        endgame::search_root(task, thread)
    }
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
        if etc_tt_hit
            && etc_tt_data.depth >= etc_depth
            && etc_tt_data.selectivity >= ctx.selectivity
        {
            let score = -etc_tt_data.score;
            if (etc_tt_data.bound == Bound::Exact as u8 || etc_tt_data.bound == Bound::Upper as u8)
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
                );
                return Some(score);
            }
        }
    }
    None
}
