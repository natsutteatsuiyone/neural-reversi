//! Process-wide search resources and the games that search on them.

mod games;
mod shared_table;
mod turn;

pub use games::{Finished, GameId, SearchKind, SearchStart};

use std::sync::{Arc, Mutex};

use crate::constants::MAX_THREADS;
use crate::eval::Eval;
use crate::search::Search;
use crate::search::options::{SearchOptions, available_cpus};
use crate::search::threading::ThreadPool;
use crate::transposition_table::TranspositionTable;
pub(crate) use shared_table::SharedTable;
pub(crate) use turn::{SearchTurn, TurnState};

/// Search resources of one process and the search state of every game opened
/// on them: the evaluation networks, one thread pool, and for each game a
/// transposition table, correction histories and endgame tracking.
///
/// A game is opened with [`Engine::open`] and named by its [`GameId`].
/// The pool runs one search at a time, so the games of an engine take turns:
/// a search waits until the engine's other searches are done. Searches that
/// run side by side each need their own engine; [`Engine::from_eval`] builds
/// several over one copy of the networks.
pub struct Engine {
    eval: Arc<Eval>,
    pool: Arc<ThreadPool>,
    tt_mb_size: usize,
    turn: Arc<TurnState>,
    games: Arc<Mutex<games::Games>>,
}

impl Engine {
    /// Creates an engine from search options, loading the evaluation networks.
    ///
    /// # Panics
    ///
    /// Panics if the evaluation weight files cannot be loaded.
    pub fn new(options: &SearchOptions) -> Self {
        let eval = Eval::with_weight_files(
            options.eval_path.as_deref(),
            options.eval_sm_path.as_deref(),
        )
        .unwrap_or_else(|err| panic!("failed to load evaluation weights: {err}"));

        Self::from_eval(Arc::new(eval), options.tt_mb_size, Some(options.n_threads))
    }

    /// Creates an engine over evaluation networks that are already loaded.
    ///
    /// The thread count defaults to the CPU count and is clamped to it and to
    /// [`MAX_THREADS`].
    pub fn from_eval(eval: Arc<Eval>, tt_mb_size: usize, n_threads: Option<usize>) -> Self {
        let n_threads = n_threads
            .unwrap_or_else(available_cpus)
            .min(available_cpus())
            .clamp(1, MAX_THREADS);

        Self {
            eval,
            pool: ThreadPool::new(n_threads),
            tt_mb_size,
            turn: Arc::default(),
            games: Arc::default(),
        }
    }

    /// Returns the shared evaluation networks.
    pub fn eval(&self) -> &Arc<Eval> {
        &self.eval
    }

    /// Returns the size of the thread pool.
    pub fn n_threads(&self) -> usize {
        self.pool.size
    }

    /// Creates the search state of a new game, with its own transposition table.
    pub(crate) fn new_game(&self) -> Search {
        self.game(Arc::new(self.new_table()), None)
    }

    /// Creates the search state of a new game on `shared`: it searches on the
    /// table of the game that joined before it while that game still holds
    /// it, and on its own table otherwise. Returns [`None`] when two games
    /// joined `shared` already.
    ///
    /// # Panics
    ///
    /// Panics if a game of another engine joined `shared`.
    fn new_game_sharing(&self, shared: &SharedTable) -> Option<Search> {
        let (member, tt) = shared.join(&self.turn, || self.new_table())?;
        Some(self.game(tt, Some((shared.clone(), member))))
    }

    fn new_table(&self) -> TranspositionTable {
        TranspositionTable::new(self.tt_mb_size)
    }

    fn game(&self, tt: Arc<TranspositionTable>, shared: Option<(SharedTable, u32)>) -> Search {
        Search::new(
            tt,
            self.pool.clone(),
            self.eval.clone(),
            self.turn.clone(),
            shared,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::level::Level;
    use crate::search::SearchRunOptions;
    use crate::square::Square;

    #[test]
    fn a_game_that_leaves_the_shared_line_searches_on_a_copy_of_the_table() {
        let engine = Engine::from_eval(Eval::shared_for_tests(), 1, Some(1));
        let shared = SharedTable::default();
        let mut first = engine.new_game_sharing(&shared).unwrap();
        let mut second = engine.new_game_sharing(&shared).unwrap();
        assert!(engine.new_game_sharing(&shared).is_none());
        assert!(Arc::ptr_eq(first.tt(), second.tt()));

        let board = Board::new().make_move(Square::F5);
        let result = first.run(&board, &SearchRunOptions::with_level(Level::uniform(4, 0)));
        first.record_move(board.make_move(result.best_move().unwrap()));
        let stored = first.tt().lookup(&board, board.hash()).unwrap();

        // `second` is to move from another board than the one `first` left.
        let turn = engine.turn.wait();
        assert!(second.split_table_if_diverged(&turn, &board));
        assert!(!Arc::ptr_eq(first.tt(), second.tt()));
        let copied = second.tt().lookup(&board, board.hash()).unwrap();
        assert_eq!(copied.score(), stored.score());
        assert_eq!(copied.best_move(), stored.best_move());

        assert!(!first.split_table_if_diverged(&turn, &board));
        assert!(!second.split_table_if_diverged(&turn, &board));
    }
}
