mod endgame;
pub mod midgame;
pub mod node_type;
mod root_move;
pub mod search_context;
pub mod search_result;
pub mod threading;

use std::sync::Arc;

use search_result::SearchResult;
use threading::{Thread, ThreadPool};

use crate::board::Board;
use crate::eval::Eval;
use crate::level::Level;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, Scoref, Selectivity};

/// Main search engine structure
pub struct Search {
    tt: Arc<TranspositionTable>,
    generation: u8,
    threads: Arc<ThreadPool>,
    eval: Arc<Eval>,
}

/// Configuration options for the search engine
pub struct SearchOptions {
    pub tt_mb_size: usize,
    pub n_threads: usize,
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

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            tt_mb_size: 64,
            n_threads: num_cpus::get(),
        }
    }
}


impl Search {
    pub fn new(options: &SearchOptions) -> Search {
        let n_threads = options.n_threads.min(num_cpus::get()).max(1);
        Search {
            tt: Arc::new(TranspositionTable::new(options.tt_mb_size)),
            generation: 0,
            threads: ThreadPool::new(n_threads),
            eval: Arc::new(Eval::new().unwrap()),
        }
    }

    pub fn init(&mut self) {
        self.tt.clear();
        self.eval.cache.clear();
        self.generation = 0;
    }

    pub fn run(&mut self, board: &Board, level: Level, selectivity: Selectivity, multi_pv: bool) -> SearchResult {
        self.run_with_callback::<fn(SearchProgress)>(board, level, selectivity, multi_pv, None)
    }

    pub fn run_with_callback<F>(&mut self, board: &Board, level: Level, selectivity: Selectivity, multi_pv: bool, callback: Option<F>) -> SearchResult
    where
        F: Fn(SearchProgress) + Send + Sync + 'static,
    {
        let callback = callback.map(|f| Arc::new(f) as Arc<SearchProgressCallback>);
        self.execute_search(board, level, selectivity, multi_pv, callback)
    }

    fn execute_search(&mut self, board: &Board, level: Level, selectivity: Selectivity, multi_pv: bool, callback: Option<Arc<SearchProgressCallback>>) -> SearchResult {
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
