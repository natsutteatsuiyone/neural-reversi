mod endgame;
pub mod midgame;
mod root_move;
pub mod search_context;
pub mod search_result;
mod spinlock;
pub mod threading;

use rand;
use rand::seq::IteratorRandom;
use search_context::SearchContext;
use search_result::SearchResult;
use threading::ThreadPool;
use std::sync::Arc;

use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::constants::SCORE_INF;
use crate::eval::Eval;
use crate::level::Level;
use crate::probcut::NO_SELECTIVITY;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, Score, Scoref, Selectivity};

pub type SearchProgressCallback = dyn Fn(SearchProgress) + Send + Sync + 'static;

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

pub struct SearchProgress {
    pub depth: Depth,
    pub score: Scoref,
    pub best_move: Square,
    pub probability: i32,
}

pub struct Search {
    tt: Arc<TranspositionTable>,
    generation: u8,
    threads: Arc<ThreadPool>,
    eval: Arc<Eval>,
}

pub struct SearchOptions {
    pub tt_mb_size: i32,
    pub n_threads: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            tt_mb_size: 64,
            n_threads: num_cpus::get(),
        }
    }
}

pub trait SearchCallback: Send {
    fn on_depth_completed(&self, depth: Depth, score: Score, best_move: Option<Square>);
}

impl Search {
    pub fn new(options: &SearchOptions) -> Search {
        let mut pool = ThreadPool::new(options.n_threads);
        pool.init();
        Search {
            tt: Arc::new(TranspositionTable::new(options.tt_mb_size)),
            generation: 0,
            threads: Arc::new(pool),
            eval: Arc::new(Eval::new("eval.zst").unwrap()),
        }
    }

    pub fn init(&mut self) {
        self.tt.clear();
        self.eval.cache.clear();
        self.generation = 0;
    }

    pub fn run<F>(&mut self, board: &Board, level: Level, selectivity: Selectivity, multi_pv: bool, callback: Option<F>) -> SearchResult
    where
        F: Fn(SearchProgress) + Send + Sync + 'static,
    {
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
            callback: callback.map(|f| Arc::new(f) as Arc<SearchProgressCallback>),
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
        self.generation += 1;

        let task = SearchTask {
            board: *board,
            generation: self.generation,
            selectivity,
            tt: self.tt.clone(),
            pool: self.threads.clone(),
            eval: self.eval.clone(),
            level,
            multi_pv: false,
            callback: None,
        };

        let result_receiver = self.threads.start_thinking(task);
        result_receiver.recv().unwrap()
    }
}

pub fn search_root(ctx: &mut SearchContext, board: &Board, level: Level, multi_pv: bool) -> (Scoref, Depth, u8) {
    let min_end_depth = level.get_end_depth(0);
    let n_empties = ctx.empty_list.count;

    if n_empties == 60 && !multi_pv {
        ctx.update_root_move(random_move(board), 0, 1, -SCORE_INF);
        (0.0, 0, NO_SELECTIVITY)
    } else if min_end_depth < n_empties {
        midgame::search_root(ctx, board, level, multi_pv)
    } else {
        endgame::search_root(ctx, board, level, multi_pv)
    }
}

fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    BitboardIterator::new(board.get_moves())
        .choose(&mut rng)
        .unwrap()
}

impl Drop for Search {
    fn drop(&mut self) {
        assert!(Arc::strong_count(&self.threads) == 1);
    }
}
