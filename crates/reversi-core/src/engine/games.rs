//! The games of an engine and the searches started for them.

use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use std::sync::MutexGuard;

use super::{Engine, SearchTurn, SharedTable, TurnState};
use crate::board::Board;
use crate::search::result::SearchResult;
use crate::search::{Search, SearchRunOptions, StopToken};

/// Names an open game of one [`Engine`]. An id is never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GameId(u64);

/// What a search started with [`Engine::start`] is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchKind {
    /// The search for the move the game plays. On a shared table its best
    /// move counts as played: the lines of the games are compared before the
    /// search and its move is recorded after it.
    Move,
    /// A ponder or a hint, which leaves the line of the game alone.
    Analysis,
}

/// What a started search knows when it is about to run.
pub struct SearchStart {
    /// Time from the request to the search turn.
    pub waited: Duration,
    /// Whether the game just left the line of the games it shared a table
    /// with and now searches on a copy of that table.
    pub split_table: bool,
}

/// A started search that ran.
pub struct Finished {
    pub result: SearchResult,
    /// Time from the request to the search turn.
    pub waited: Duration,
    /// Time the search itself took.
    pub elapsed: Duration,
    /// Whether the search used up its time and aborted an iteration.
    pub timed_out: bool,
}

/// The open games of an engine.
#[derive(Default)]
pub(super) struct Games {
    next_id: u64,
    next_ticket: u64,
    open: HashMap<GameId, Slot>,
}

struct Slot {
    search: Arc<Mutex<Search>>,
    /// Passed from one search or reset of the game to the next. Each takes it
    /// before locking `search`, so one that waits behind another can still
    /// be stopped.
    line: Arc<TurnState>,
    /// Set while the game shares its table; another game can join it.
    shared: Option<SharedTable>,
    /// Ticket and stop handle of every started search that has not ended.
    started: Vec<(u64, StopToken)>,
    /// Ticket of the search of each [`SearchKind`] started last, until it is
    /// stopped.
    latest: [Option<u64>; 2],
}

impl Engine {
    /// Opens a game with a transposition table of its own.
    pub fn open(&self) -> GameId {
        self.insert(self.new_game(), None)
    }

    /// Opens a game on a table that two games following the same moves share,
    /// like the games of a synchro match: the table `with` shares, or a fresh
    /// one when `with` is [`None`], is closed, shares none, or already shares
    /// with another game.
    ///
    /// The two share the table until a [`SearchKind::Move`] search finds that
    /// their lines diverged, which gives that game a copy, or until
    /// [`Engine::resize_tt`] replaces the table of one.
    pub fn open_sharing(&self, with: Option<GameId>) -> GameId {
        let offered =
            with.and_then(|with| self.games.lock().unwrap().open.get(&with)?.shared.clone());
        let (search, shared) = offered
            .and_then(|shared| Some((self.new_game_sharing(&shared)?, shared)))
            .unwrap_or_else(|| {
                let shared = SharedTable::default();
                let search = self.new_game_sharing(&shared);
                (search.expect("a new shared table has room"), shared)
            });
        self.insert(search, Some(shared))
    }

    fn insert(&self, search: Search, shared: Option<SharedTable>) -> GameId {
        let mut games = self.games.lock().unwrap();
        let id = GameId(games.next_id);
        games.next_id += 1;
        let slot = Slot {
            search: Arc::new(Mutex::new(search)),
            line: Arc::default(),
            shared,
            started: Vec::new(),
            latest: [None; 2],
        };
        games.open.insert(id, slot);
        id
    }

    /// Closes `game` and stops every search it started, queued or running. A
    /// stopped search still reports to its `done`. Does nothing for a game
    /// that is not open.
    pub fn close(&self, game: GameId) {
        let slot = self.games.lock().unwrap().open.remove(&game);
        for (_, stop) in slot.into_iter().flat_map(|slot| slot.started) {
            stop.stop();
        }
    }

    /// Searches `board` for `game` on the calling thread once the game's and
    /// the engine's other searches are done. Leaves the line of a shared
    /// table alone.
    ///
    /// # Panics
    ///
    /// Panics if `game` is not open or `board` has fewer than 4 discs.
    pub fn search(&self, game: GameId, board: &Board, options: &SearchRunOptions) -> SearchResult {
        let (record, line) = self.record(game);
        let _place = line.wait();
        let mut search = record.lock().unwrap();
        search.run(board, options)
    }

    /// Starts a search of `board` for `game`, requested at `queued_at`, in a
    /// new thread.
    ///
    /// Once the search has the engine's turn, `begin` chooses its options from
    /// the time it waited; the stop token of the options is replaced. `done`
    /// receives the outcome after the turn is given up, or [`None`] when
    /// [`Engine::stop`] or [`Engine::close`] came before the turn. Neither
    /// stops an earlier search of the game: the new one waits for it.
    ///
    /// # Errors
    ///
    /// Returns the error of spawning the thread; `done` is not called then.
    ///
    /// # Panics
    ///
    /// Panics if `game` is not open.
    pub fn start(
        &self,
        game: GameId,
        kind: SearchKind,
        board: Board,
        queued_at: Instant,
        begin: impl FnOnce(&SearchStart) -> SearchRunOptions + Send + 'static,
        done: impl FnOnce(Option<Finished>) + Send + 'static,
    ) -> io::Result<()> {
        let stop = StopToken::new();
        let (record, line, ticket) = {
            let mut games = self.games.lock().unwrap();
            let ticket = games.next_ticket;
            games.next_ticket += 1;
            let slot = games.open.get_mut(&game).expect("game is not open");
            slot.started.push((ticket, stop.clone()));
            slot.latest[kind as usize] = Some(ticket);
            (Arc::clone(&slot.search), Arc::clone(&slot.line), ticket)
        };
        let turn = Arc::clone(&self.turn);
        let games = Arc::clone(&self.games);
        let spawned = thread::Builder::new().spawn(move || {
            let outcome = admit(&line, &record, &turn, &stop).map(|(_place, mut search, turn)| {
                let split_table =
                    kind == SearchKind::Move && search.split_table_if_diverged(&turn, &board);
                // Measured after the copy: the game's clock ran through both.
                let waited = queued_at.elapsed();
                let options = begin(&SearchStart {
                    waited,
                    split_table,
                })
                .stop_token(stop);
                let started = Instant::now();
                let result = search.run_in_turn(&turn, &board, &options);
                let elapsed = started.elapsed();
                if kind == SearchKind::Move
                    && let Some(sq) = result.best_move()
                {
                    search.record_move(board.make_move(sq));
                }
                Finished {
                    result,
                    waited,
                    elapsed,
                    timed_out: search.timed_out(),
                }
            });
            forget(&games, game, ticket);
            drop((record, line));
            done(outcome);
        });
        if spawned.is_err() {
            forget(&self.games, game, ticket);
        }
        spawned.map(drop)
    }

    /// Stops the search of `kind` that `game` started last, queued or
    /// running. Does nothing for a game that is not open.
    pub fn stop(&self, game: GameId, kind: SearchKind) {
        let mut games = self.games.lock().unwrap();
        let Some(slot) = games.open.get_mut(&game) else {
            return;
        };
        let latest = slot.latest[kind as usize].take();
        if let Some((_, stop)) = slot
            .started
            .iter()
            .find(|(ticket, _)| Some(*ticket) == latest)
        {
            stop.stop();
        }
    }

    /// Clears the transposition table, corrections and endgame tracking of
    /// `game` once its and the engine's other searches are done.
    ///
    /// # Panics
    ///
    /// Panics if `game` is not open.
    pub fn reset(&self, game: GameId) {
        let (record, line) = self.record(game);
        let _place = line.wait();
        record.lock().unwrap().init();
    }

    /// Gives `game` a transposition table of `mb_size` MiB unless its table
    /// has that size already. A game that shared its table stops sharing it.
    ///
    /// # Panics
    ///
    /// Panics if `game` is not open.
    pub fn resize_tt(&self, game: GameId, mb_size: usize) {
        let (record, _) = self.record(game);
        let replaced = record.lock().unwrap().resize_tt(mb_size);
        if replaced && let Some(slot) = self.games.lock().unwrap().open.get_mut(&game) {
            slot.shared = None;
        }
    }

    /// Returns whether the last search of `game` used up its time and aborted
    /// an iteration.
    ///
    /// # Panics
    ///
    /// Panics if `game` is not open.
    pub fn timed_out(&self, game: GameId) -> bool {
        self.record(game).0.lock().unwrap().timed_out()
    }

    /// Estimates the fraction of occupied entries in the transposition table
    /// of `game`.
    ///
    /// # Panics
    ///
    /// Panics if `game` is not open.
    pub fn tt_usage_rate(&self, game: GameId) -> f64 {
        self.record(game).0.lock().unwrap().tt_usage_rate()
    }

    /// Returns the search state of `game` and the line its searches wait in.
    fn record(&self, game: GameId) -> (Arc<Mutex<Search>>, Arc<TurnState>) {
        let games = self.games.lock().unwrap();
        let slot = games.open.get(&game).expect("game is not open");
        (Arc::clone(&slot.search), Arc::clone(&slot.line))
    }
}

/// Waits for the place of a started search in the line of its game and then
/// for the engine's turn. Returns [`None`] once `stop` is stopped.
fn admit<'a>(
    line: &Arc<TurnState>,
    record: &'a Mutex<Search>,
    turn: &Arc<TurnState>,
    stop: &StopToken,
) -> Option<(SearchTurn, MutexGuard<'a, Search>, SearchTurn)> {
    let place = line.wait_unless_stopped(stop)?;
    let search = record.lock().unwrap();
    let turn = turn.wait_unless_stopped(stop)?;
    Some((place, search, turn))
}

/// Drops the stop handle of a started search that has ended.
fn forget(games: &Mutex<Games>, game: GameId, ticket: u64) {
    if let Some(slot) = games.lock().unwrap().open.get_mut(&game) {
        slot.started.retain(|(started, ..)| *started != ticket);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{self, Receiver};

    use super::*;
    use crate::eval::Eval;
    use crate::level::Level;
    use crate::search::time_control::TimeControlMode;
    use crate::square::Square;
    use crate::transposition_table::TranspositionTable;

    const WAIT: Duration = Duration::from_secs(60);

    fn engine() -> Engine {
        Engine::from_eval(Eval::shared_for_tests(), 1, Some(1))
    }

    fn quick(_: &SearchStart) -> SearchRunOptions {
        SearchRunOptions::with_level(Level::uniform(2, 0))
    }

    fn endless(_: &SearchStart) -> SearchRunOptions {
        SearchRunOptions::with_time(TimeControlMode::Infinite)
    }

    fn board() -> Board {
        Board::new().make_move(Square::F5)
    }

    /// Starts a search of `board()` and returns the channel its outcome
    /// arrives on.
    fn start(
        engine: &Engine,
        game: GameId,
        kind: SearchKind,
        begin: fn(&SearchStart) -> SearchRunOptions,
    ) -> Receiver<Option<Finished>> {
        let (done_tx, done) = mpsc::channel();
        engine
            .start(game, kind, board(), Instant::now(), begin, move |outcome| {
                done_tx.send(outcome).unwrap()
            })
            .unwrap();
        done
    }

    /// Starts an endless analysis of `game` and waits until it has the turn.
    fn start_analysis(engine: &Engine, game: GameId) -> Receiver<Option<Finished>> {
        let (begun_tx, begun) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        engine
            .start(
                game,
                SearchKind::Analysis,
                board(),
                Instant::now(),
                move |start| {
                    begun_tx.send(()).unwrap();
                    endless(start)
                },
                move |outcome| done_tx.send(outcome).unwrap(),
            )
            .unwrap();
        begun.recv_timeout(WAIT).unwrap();
        done
    }

    /// Searches the move of `game` on `board` and returns whether its table
    /// split and the board its move leaves.
    fn play(engine: &Engine, game: GameId, board: Board) -> (bool, Board) {
        let (split_tx, split) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        engine
            .start(
                game,
                SearchKind::Move,
                board,
                Instant::now(),
                move |start| {
                    split_tx.send(start.split_table).unwrap();
                    quick(start)
                },
                move |outcome| done_tx.send(outcome).unwrap(),
            )
            .unwrap();
        let found = done.recv_timeout(WAIT).unwrap().unwrap();
        let left = board.make_move(found.result.best_move().unwrap());
        (split.recv_timeout(WAIT).unwrap(), left)
    }

    #[test]
    fn a_started_move_reports_its_move_and_its_wait_for_the_turn() {
        let engine = engine();
        let game = engine.open();
        let turn = engine.turn.wait();
        let done = start(&engine, game, SearchKind::Move, quick);
        let held = Duration::from_millis(20);
        thread::sleep(held);
        drop(turn);

        let found = done.recv_timeout(WAIT).unwrap().unwrap();
        assert!(board().is_legal_move(found.result.best_move().unwrap()));
        assert!(found.waited >= held);
        assert!(!found.timed_out);
    }

    #[test]
    fn a_search_stopped_while_queued_is_never_run() {
        let engine = engine();
        let game = engine.open();
        let turn = engine.turn.wait();
        let done = start(&engine, game, SearchKind::Move, |_| {
            unreachable!("a stopped search must not begin")
        });

        engine.stop(game, SearchKind::Move);
        assert!(done.recv_timeout(WAIT).unwrap().is_none());
        drop(turn);
    }

    #[test]
    fn a_search_stopped_behind_another_of_its_game_is_never_run() {
        let engine = engine();
        let game = engine.open();
        let analysis = start_analysis(&engine, game);
        let queued = start(&engine, game, SearchKind::Move, |_| {
            unreachable!("a stopped search must not begin")
        });

        engine.stop(game, SearchKind::Move);
        assert!(queued.recv_timeout(WAIT).unwrap().is_none());
        assert!(analysis.recv_timeout(Duration::from_millis(50)).is_err());
        engine.stop(game, SearchKind::Analysis);
        assert!(analysis.recv_timeout(WAIT).unwrap().is_some());
    }

    #[test]
    fn stopping_reaches_only_the_search_of_its_kind_started_last() {
        let engine = engine();
        let game = engine.open();
        let running = start_analysis(&engine, game);
        let queued = start(&engine, game, SearchKind::Analysis, |_| {
            unreachable!("a stopped search must not begin")
        });

        engine.stop(game, SearchKind::Analysis);
        assert!(queued.recv_timeout(WAIT).unwrap().is_none());
        engine.stop(game, SearchKind::Analysis);
        assert!(running.recv_timeout(Duration::from_millis(50)).is_err());
        engine.close(game);
        assert!(running.recv_timeout(WAIT).unwrap().is_some());
    }

    #[test]
    fn closing_a_game_stops_its_searches_and_leaves_other_games_alone() {
        let engine = engine();
        let [closed, kept] = [engine.open(), engine.open()];
        let running = start_analysis(&engine, closed);
        // Of the same kind as the running one: both must be stopped.
        let queued = start(&engine, closed, SearchKind::Analysis, |_| {
            unreachable!("a stopped search must not begin")
        });
        let other = start(&engine, kept, SearchKind::Move, quick);

        engine.close(closed);
        assert!(running.recv_timeout(WAIT).unwrap().is_some());
        assert!(queued.recv_timeout(WAIT).unwrap().is_none());
        assert!(other.recv_timeout(WAIT).unwrap().is_some());
        // A closed game takes no further requests but may be stopped or closed again.
        engine.stop(closed, SearchKind::Move);
        engine.close(closed);
    }

    #[test]
    fn games_share_a_table_until_one_leaves_the_line_of_the_other() {
        let engine = engine();
        let first = engine.open_sharing(None);
        let second = engine.open_sharing(Some(first));

        let (split, left) = play(&engine, first, board());
        assert!(!split);
        // The opponent repeated our move in the other game.
        let (split, _) = play(&engine, second, left);
        assert!(!split);
        // The first game is back on a board the other did not leave.
        let (split, _) = play(&engine, first, board());
        assert!(split);
        let (split, _) = play(&engine, second, board());
        assert!(!split);
    }

    fn table(engine: &Engine, game: GameId) -> Arc<TranspositionTable> {
        Arc::clone(engine.record(game).0.lock().unwrap().tt())
    }

    #[test]
    fn a_shared_table_takes_two_games_and_loses_one_that_is_resized() {
        let engine = engine();
        let first = engine.open_sharing(None);
        let second = engine.open_sharing(Some(first));
        assert!(Arc::ptr_eq(&table(&engine, first), &table(&engine, second)));
        let third = engine.open_sharing(Some(first));
        assert!(!Arc::ptr_eq(&table(&engine, first), &table(&engine, third)));

        play(&engine, first, board());
        engine.resize_tt(second, 2);
        let resized = table(&engine, second);
        assert_eq!(resized.mb_size(), 2);
        // The second game moves from a board the first did not leave.
        let (split, _) = play(&engine, second, board());
        assert!(!split);
        assert!(Arc::ptr_eq(&table(&engine, second), &resized));
        let fourth = engine.open_sharing(Some(second));
        assert!(!Arc::ptr_eq(&table(&engine, fourth), &resized));
    }

    #[test]
    fn a_search_on_the_calling_thread_follows_the_stopped_analysis_of_its_game() {
        let engine = engine();
        let game = engine.open();
        let options = SearchRunOptions::with_level(Level::uniform(2, 0));
        for _ in 0..20 {
            let analysis = start_analysis(&engine, game);
            engine.stop(game, SearchKind::Analysis);
            let result = engine.search(game, &board(), &options);
            assert!(board().is_legal_move(result.best_move().unwrap()));
            assert!(analysis.recv_timeout(WAIT).unwrap().is_some());
        }
    }
}
