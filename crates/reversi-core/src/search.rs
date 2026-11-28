//! Game tree search.
//!
//! Dispatches each root search to the midgame or endgame driver; the shared
//! alpha-beta core lives in the `pvs` module. `Search` starts the runs and
//! keeps the state of one game that outlives one.

mod aspiration;
pub mod context;
pub(crate) mod correction_history;
pub mod counters;
mod endgame;
pub(crate) mod midgame;
pub(crate) mod node_type;
pub mod options;
pub mod progress;
mod pvs;
pub mod result;
pub mod root_move;
pub mod side_to_move;
pub mod stack;
mod stop;
pub mod strategy;
pub(crate) mod threading;
pub mod time_control;

pub(crate) use aspiration::widen_aspiration_window;
#[doc(hidden)]
pub use endgame::{Empties, odd_quadrants, quadrant, solve_last1};
pub use options::{SearchConstraint, SearchRunOptions};
pub use progress::{SearchProgress, SearchProgressCallback};
pub(crate) use pvs::{LMR_MIN_DEPTH, search};
pub use stop::StopToken;

use std::sync::Arc;

use crate::board::Board;
use crate::constants::INITIAL_EMPTY_COUNT;
use crate::engine::{SearchTurn, SharedTable, TurnState};
use crate::eval::{Eval, EvalMode};
use crate::probcut::Selectivity;
use crate::search::correction_history::CorrectionHistories;
use crate::search::result::{SearchResult, quick_move};
use crate::search::threading::{Thread, ThreadPool};
use crate::search::time_control::TimeManager;
use crate::transposition_table::TranspositionTable;
use crate::types::Depth;

/// Lowest endgame selectivity whose completion lets later timed searches
/// skip the midgame iterations.
///
/// A Level1 solve can finish on one move and still abort on the next, which
/// then has no midgame result to fall back on.
pub(crate) const ENDGAME_TRACKING_MIN_SELECTIVITY: Selectivity = Selectivity::Level2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RootPhase {
    Midgame { max_depth: Depth },
    Endgame { last: Selectivity },
}

/// Position, resolved search settings, and shared resources for one root search.
pub(crate) struct SearchTask {
    /// Board position to search.
    pub board: Board,
    /// Midgame selectivity: [`Selectivity::Mid`], or [`Selectivity::None`]
    /// when ProbCut is disabled. The endgame driver uses its own selectivity
    /// ladder and ignores this value.
    pub mid_selectivity: Selectivity,
    /// Shared transposition table.
    pub tt: Arc<TranspositionTable>,
    /// Shared thread pool for parallel search.
    pub pool: Arc<ThreadPool>,
    /// Shared evaluation function.
    pub eval: Arc<Eval>,
    pub phase: RootPhase,
    /// Whether to report multiple principal variations.
    pub multi_pv: bool,
    /// Optional callback invoked to report search progress.
    pub callback: Option<Arc<SearchProgressCallback>>,
    /// Optional time manager for time-controlled searches.
    pub time_manager: Option<Arc<TimeManager>>,
    /// Optional override for evaluation mode.
    pub eval_mode: Option<EvalMode>,
    /// Evaluation corrections of this game's search threads.
    pub corrections: Arc<CorrectionHistories>,
}

impl SearchTask {
    /// Returns how many PV lines to search: every root move in Multi-PV mode,
    /// otherwise one.
    pub(crate) fn pv_count(&self, root_moves_count: usize) -> usize {
        if self.multi_pv { root_moves_count } else { 1 }
    }
}

pub(crate) fn midgame_selectivity(options: &SearchRunOptions) -> Selectivity {
    if options.probcut_disabled {
        Selectivity::None
    } else {
        Selectivity::Mid
    }
}

fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    match task.phase {
        RootPhase::Midgame { max_depth } => midgame::search_root(task, max_depth, thread),
        RootPhase::Endgame { last } => endgame::search_root(task, last, thread),
    }
}

/// Search state of one game on an [`Engine`]: its transposition table,
/// correction histories, and endgame tracking.
///
/// The engine keeps one for each open game.
///
/// [`Engine`]: crate::engine::Engine
pub(crate) struct Search {
    tt: Arc<TranspositionTable>,
    threads: Arc<ThreadPool>,
    eval: Arc<Eval>,
    corrections: Arc<CorrectionHistories>,
    endgame_start_n_empties: Option<Depth>,
    timed_out: bool,
    turn: Arc<TurnState>,
    /// The shared table this game joined, and its number there.
    shared: Option<(SharedTable, u32)>,
}

const TIMED_ENDGAME_ROOT_MAX_EMPTIES: Depth = 14;

// The MIR checks under `verification/` name this block `search-{impl#1}`: it
// must stay the second `impl` written in this file.
impl Search {
    /// Creates the search state of a new game on `tt`, for the engine that
    /// owns `turn`.
    pub(crate) fn new(
        tt: Arc<TranspositionTable>,
        threads: Arc<ThreadPool>,
        eval: Arc<Eval>,
        turn: Arc<TurnState>,
        shared: Option<(SharedTable, u32)>,
    ) -> Self {
        Self {
            tt,
            threads,
            eval,
            corrections: Arc::default(),
            endgame_start_n_empties: None,
            timed_out: false,
            turn,
            shared,
        }
    }

    /// Returns whether the last [`Search::run`] used up its time and aborted
    /// the iteration in progress, rather than stopping between iterations.
    pub fn timed_out(&self) -> bool {
        self.timed_out
    }

    /// Returns a reference to the transposition table.
    #[cfg(test)]
    pub(crate) fn tt(&self) -> &Arc<TranspositionTable> {
        &self.tt
    }

    /// Estimates the fraction of occupied transposition table entries.
    pub fn tt_usage_rate(&self) -> f64 {
        self.tt.usage_rate()
    }

    /// Resets the search state for a new game.
    ///
    /// Clears the transposition table, resets the TT generation counter,
    /// flushes the evaluation cache and corrections, and resets endgame tracking.
    ///
    /// Waits for the engine's other searches first, as [`Search::run`]
    /// does.
    pub fn init(&mut self) {
        let _turn = self.turn.wait();
        match Arc::get_mut(&mut self.tt) {
            Some(tt) => tt.clear(),
            None => self.tt.clear_shared(),
        }
        self.tt.reset_generation();
        self.eval.clear_cache();
        self.corrections.clear();
        self.endgame_start_n_empties = None;
    }

    /// Resizes the transposition table to `mb_size` MiB. Returns whether the
    /// table was replaced.
    ///
    /// Replaces the table only when the requested size differs from the
    /// current one, avoiding unnecessary reallocation. A game that shared its
    /// table stops sharing it then.
    pub fn resize_tt(&mut self, mb_size: usize) -> bool {
        let replace = self.tt.mb_size() != mb_size;
        if replace {
            self.tt = Arc::new(TranspositionTable::new(mb_size));
            if let Some((shared, _)) = self.shared.take() {
                shared.dissolve();
            }
        }
        replace
    }

    /// Gives this game a private copy of its shared table once `board`, the
    /// position it is to move from, is not the one another game on the table
    /// left by its latest move. Returns whether the game got the copy, which
    /// takes time in proportion to the table size.
    ///
    /// A game that shares no table keeps its table and returns `false`.
    ///
    /// # Panics
    ///
    /// Panics if `turn` is the turn of another engine.
    pub(crate) fn split_table_if_diverged(&mut self, turn: &SearchTurn, board: &Board) -> bool {
        let Some((shared, member)) = &self.shared else {
            return false;
        };
        let Some(copy) = shared.split_off(turn, *member, board) else {
            return false;
        };
        self.tt = Arc::new(copy);
        true
    }

    /// Records `left`, the position after the move this game just played,
    /// for the other games on its shared table to compare with.
    pub(crate) fn record_move(&self, left: Board) {
        if let Some((shared, member)) = &self.shared {
            shared.record_move(*member, left);
        }
    }

    /// Runs a search on the given board position, once the engine's other
    /// searches are done.
    ///
    /// Selects the appropriate search strategy based on the constraint (fixed level
    /// or time-controlled) and executes the search.
    ///
    /// # Panics
    ///
    /// Panics if `board` has fewer than 4 discs.
    pub fn run(&mut self, board: &Board, options: &SearchRunOptions) -> SearchResult {
        let turn = self.turn.wait();
        self.run_in_turn(&turn, board, options)
    }

    /// Runs a search on the given board position within `turn`.
    ///
    /// A thread that holds the turn calls this instead of [`Search::run`],
    /// which would wait for that turn forever.
    ///
    /// # Panics
    ///
    /// Panics if `board` has fewer than 4 discs, or if `turn` is the turn of
    /// another engine.
    pub(crate) fn run_in_turn(
        &mut self,
        turn: &SearchTurn,
        board: &Board,
        options: &SearchRunOptions,
    ) -> SearchResult {
        assert!(
            turn.belongs_to(&self.turn),
            "the search turn belongs to another engine"
        );
        assert!(
            board.empty_count() as usize <= INITIAL_EMPTY_COUNT,
            "a search needs at most {INITIAL_EMPTY_COUNT} empty squares, got {}",
            board.empty_count()
        );
        let callback = options.callback.clone();
        let n_empties = board.empty_count();

        let (time_manager, phase) = match &options.constraint {
            SearchConstraint::Level(level) => {
                let phase = match level.end_selectivity(n_empties) {
                    Some(last) => RootPhase::Endgame { last },
                    None => RootPhase::Midgame {
                        max_depth: level.mid_depth,
                    },
                };
                (None, phase)
            }
            SearchConstraint::Time(mode) => {
                let tm = Arc::new(TimeManager::new(
                    *mode,
                    self.threads.abort_state(),
                    n_empties,
                ));
                (Some(tm), self.timed_root_phase(n_empties))
            }
        };
        let is_time_mode = time_manager.is_some();
        let deadline = time_manager.clone();

        // The timer starts with the search, so an endgame root must not run
        // under the smaller midgame deadline even briefly.
        if let Some(tm) = &time_manager
            && matches!(phase, RootPhase::Endgame { .. })
        {
            tm.set_endgame_mode(true);
        }

        let task = SearchTask {
            board: *board,
            mid_selectivity: midgame_selectivity(options),
            tt: self.tt.clone(),
            pool: self.threads.clone(),
            eval: self.eval.clone(),
            phase,
            multi_pv: options.multi_pv,
            callback: callback.clone(),
            time_manager,
            eval_mode: options.eval_mode,
            corrections: self.corrections.clone(),
        };

        let result = self.execute_search(task, options.stop_token.as_ref());
        self.timed_out = deadline.is_some_and(|tm| tm.timed_out());

        if let Some(callback) = callback {
            callback(SearchProgress::from_result(&result));
        }

        if is_time_mode {
            self.update_endgame_tracking(n_empties, &result);
        }

        result
    }

    fn timed_root_phase(&self, n_empties: Depth) -> RootPhase {
        let solved_before = self
            .endgame_start_n_empties
            .is_some_and(|start| n_empties <= start);

        if n_empties <= TIMED_ENDGAME_ROOT_MAX_EMPTIES || solved_before {
            RootPhase::Endgame {
                last: Selectivity::None,
            }
        } else {
            RootPhase::Midgame {
                max_depth: n_empties,
            }
        }
    }

    /// Records the empty count at which an endgame solve first finished
    /// [`ENDGAME_TRACKING_MIN_SELECTIVITY`].
    ///
    /// Until then, each search starts with midgame iterations, so a result
    /// exists even if the endgame solve aborts before finishing a level.
    fn update_endgame_tracking(&mut self, n_empties: Depth, result: &SearchResult) {
        if self
            .endgame_start_n_empties
            .is_some_and(|start| n_empties > start)
        {
            self.endgame_start_n_empties = None;
        }

        if n_empties > 0
            && self.endgame_start_n_empties.is_none()
            && result.is_endgame()
            && result.selectivity() >= ENDGAME_TRACKING_MIN_SELECTIVITY
        {
            self.endgame_start_n_empties = Some(n_empties - 1);
        }
    }

    fn execute_search(&mut self, task: SearchTask, stop: Option<&StopToken>) -> SearchResult {
        self.tt.increment_generation();

        let board = task.board;
        let result_receiver = self.threads.start_thinking(task, stop);

        let result = result_receiver.recv().unwrap_or_else(|_| {
            // Channel closed - search thread may have panicked. Return fallback.
            quick_move(&board, &self.eval)
        });

        self.threads.stop_timer();

        result
    }

    /// Returns the [`ThreadPool`] this game searches on.
    #[cfg(test)]
    pub(crate) fn thread_pool(&self) -> Arc<ThreadPool> {
        self.threads.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::engine::Engine;
    use crate::level::Level;
    use crate::player_color::PlayerColor;
    use crate::score::{DiscScore, Score};
    use crate::search::counters::SearchCounters;
    use crate::search::time_control::TimeControlMode;
    use crate::square::Square;
    use crate::transposition_table::Bound;

    #[test]
    fn fixed_level_and_time_constraints_share_the_midgame_selectivity() {
        let fixed = SearchRunOptions::with_level(Level::uniform(8, 8));
        let timed = SearchRunOptions::with_time(TimeControlMode::Infinite);

        assert_eq!(midgame_selectivity(&fixed), Selectivity::Mid);
        assert_eq!(midgame_selectivity(&timed), Selectivity::Mid);
    }

    #[test]
    fn disabling_probcut_overrides_constraint_midgame_selectivity() {
        let fixed = SearchRunOptions::with_level(Level::uniform(8, 8)).disable_probcut();
        let timed = SearchRunOptions::with_time(TimeControlMode::Infinite).disable_probcut();

        assert_eq!(midgame_selectivity(&fixed), Selectivity::None);
        assert_eq!(midgame_selectivity(&timed), Selectivity::None);
    }

    #[test]
    fn games_of_one_engine_share_the_networks_and_the_pool_but_not_the_table() {
        let engine = Engine::from_eval(Eval::shared_for_tests(), 1, Some(1));

        let first = engine.new_game();
        let second = engine.new_game();

        assert!(Arc::ptr_eq(&first.eval, &second.eval));
        assert!(Arc::ptr_eq(&first.thread_pool(), &second.thread_pool()));
        assert!(!Arc::ptr_eq(first.tt(), second.tt()));
        assert_eq!(first.tt().mb_size(), 1);
    }

    fn search_for_tests() -> Search {
        Engine::from_eval(Eval::shared_for_tests(), 0, Some(1)).new_game()
    }

    #[test]
    fn timed_midgame_handoff_reuses_an_interrupted_search() {
        // FFO #9 has 15 empties. G7 and A4 both score -8.
        let board = Board::from_string(
            "--XOXX--O-OOXXXX-OOOXXXX-XOXXXOXXXOXOOOXOXXOXOXX--OXOO----OOOO--",
            PlayerColor::White,
        )
        .unwrap();
        let mut search = search_for_tests();
        let pool = search.thread_pool();
        let timed = || SearchRunOptions::with_time(TimeControlMode::Infinite);
        let earlier = search.run(
            &board,
            &timed().callback(move |progress| {
                if progress.depth == 7 && !progress.is_endgame {
                    pool.abort_search();
                }
            }),
        );
        assert_eq!(earlier.depth(), 7);
        assert!(!earlier.is_endgame());

        // Level1 has written the endgame tables, but the saved midgame line
        // remains the public result until Level2 completes.
        let pool = search.thread_pool();
        let saw_level1 = Arc::new(AtomicBool::new(false));
        let seen = saw_level1.clone();
        let after_handoff = search.run(
            &board,
            &timed().callback(move |p| {
                if p.is_endgame && p.selectivity == Selectivity::Level1 {
                    seen.store(true, Ordering::Relaxed);
                    pool.abort_search();
                }
            }),
        );
        assert!(saw_level1.load(Ordering::Relaxed));
        assert_eq!(after_handoff.depth(), 15);
        assert!(!after_handoff.is_endgame());

        let progress = Arc::new(Mutex::new(Vec::new()));
        let events = progress.clone();
        let result = search.run(
            &board,
            &timed().callback(move |p| events.lock().unwrap().push((p.depth, p.is_endgame))),
        );
        let events = progress.lock().unwrap();
        assert!(
            events
                .windows(2)
                .any(|pair| pair == [(15, false), (15, true)])
        );
        assert!(result.is_endgame());
        assert_eq!(result.selectivity(), Selectivity::None);
        assert!(matches!(result.best_move(), Some(Square::G7 | Square::A4)));
        assert_eq!(result.score().unwrap().value(), -8 * 256);
    }

    #[test]
    fn resize_tt_reuses_same_size_replaces_changed_size_and_init_resets_generation() {
        let mut search = search_for_tests();
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

    fn endgame_result(selectivity: Selectivity) -> SearchResult {
        SearchResult::BestMove {
            sq: Square::A1,
            score: Score::ZERO,
            n_nodes: 0,
            pv_line: vec![],
            depth: 30,
            selectivity,
            is_endgame: true,
            pv_moves: vec![],
            counters: SearchCounters::default(),
        }
    }

    #[test]
    fn midgame_iterations_are_skipped_only_after_a_level2_endgame_solve() {
        let mut search = search_for_tests();

        search.update_endgame_tracking(30, &endgame_result(Selectivity::Level1));
        assert_eq!(search.endgame_start_n_empties, None);

        search.update_endgame_tracking(28, &endgame_result(Selectivity::Level2));
        assert_eq!(search.endgame_start_n_empties, Some(27));
    }

    #[test]
    fn endgame_tracking_is_dropped_by_a_search_with_more_empties() {
        let mut search = search_for_tests();
        search.endgame_start_n_empties = Some(27);

        search.update_endgame_tracking(27, &endgame_result(Selectivity::Level1));
        assert_eq!(search.endgame_start_n_empties, Some(27));

        search.update_endgame_tracking(28, &endgame_result(Selectivity::Level1));
        assert_eq!(search.endgame_start_n_empties, None);
    }

    #[test]
    fn timed_root_starts_in_the_endgame_near_the_end_or_after_a_tracked_solve() {
        let mut search = search_for_tests();
        let endgame = RootPhase::Endgame {
            last: Selectivity::None,
        };
        let midgame = |max_depth| RootPhase::Midgame { max_depth };

        assert_eq!(
            search.timed_root_phase(TIMED_ENDGAME_ROOT_MAX_EMPTIES),
            endgame
        );
        assert_eq!(
            search.timed_root_phase(TIMED_ENDGAME_ROOT_MAX_EMPTIES + 1),
            midgame(TIMED_ENDGAME_ROOT_MAX_EMPTIES + 1)
        );

        search.endgame_start_n_empties = Some(27);
        assert_eq!(search.timed_root_phase(27), endgame);
        assert_eq!(search.timed_root_phase(28), midgame(28));
    }

    fn table_game() -> Search {
        Engine::from_eval(Eval::shared_for_tests(), 8, Some(1)).new_game()
    }

    fn assert_exact_score(result: &SearchResult, discs: i32) {
        let SearchResult::BestMove {
            score,
            is_endgame: true,
            selectivity: Selectivity::None,
            ..
        } = result
        else {
            panic!("expected a completed exact endgame result");
        };
        assert_eq!(score.value(), 256 * discs);
    }

    #[test]
    fn exact_solve_reuses_a_selective_root_entry_across_generation_wrap() {
        let mut engine = table_game();
        let full = SearchRunOptions::with_level(Level::perfect());
        let weak = SearchRunOptions::with_level(Level::perfect().stopping_at(Selectivity::Level1));
        let warmup = Board::from_string(
            "-OXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-",
            PlayerColor::Black,
        )
        .unwrap();
        for _ in 0..126 {
            engine.run(&warmup, &full);
        }

        let board = Board::from_bitboards(0x0012060f3c000080_u64, 0x04e9b0f0c27efc7f_u64);
        let previous = engine.run(&board, &weak);
        assert!(previous.is_endgame());
        assert_eq!(previous.selectivity(), Selectivity::Level1);
        let stored = engine.tt().lookup(&board, board.hash()).unwrap();
        assert_eq!(stored.bound(), Bound::Exact);
        assert_eq!(stored.selectivity(), Selectivity::Level1);
        assert_eq!(stored.score(), previous.score().unwrap());
        assert_eq!(stored.generation(), 127);

        // Ensure the selective root entry carries a score distinct from the exact +20
        // even when Level1 ProbCut happens to match on this position.
        let probe = engine.tt().probe(&board, board.hash());
        let stale_score = Score::from_disc_diff(DiscScore::new(0));
        engine.tt().store(
            probe.slot(),
            &board,
            stale_score,
            Bound::Exact,
            stored.depth(),
            stored.best_move(),
            Selectivity::Level1,
            true,
            true,
        );
        let seeded = engine.tt().lookup(&board, board.hash()).unwrap();
        assert_eq!(seeded.score(), stale_score);
        assert_eq!(seeded.generation(), 127);

        // Independent directional minimax gives +20 on this board.
        let exact = engine.run(&board, &full);
        assert_exact_score(&exact, 20);
        assert_eq!(engine.tt().generation(), 0);
    }

    #[test]
    fn midgame_root_score_can_seed_an_exact_solve() {
        let mut engine = table_game();
        let board = Board::from_bitboards(0x0012060f3c000080_u64, 0x04e9b0f0c27efc7f_u64);
        let shallow = engine.run(&board, &SearchRunOptions::with_level(Level::uniform(1, 0)));
        assert!(!shallow.is_endgame());
        let prior = engine.tt().lookup(&board, board.hash()).unwrap();
        assert!(!prior.is_endgame());
        assert_eq!(prior.bound(), Bound::Exact);
        assert_eq!(prior.score(), shallow.score().unwrap());
        assert_ne!(prior.score().value(), 20 * 256);

        let exact = engine.run(&board, &SearchRunOptions::with_level(Level::perfect()));
        assert_exact_score(&exact, 20);
    }
}
