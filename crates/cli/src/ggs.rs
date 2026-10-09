//! GGS (Othello Generic Game Server) play mode.
//!
//! Connects to `localhost:5000` by default (or host/port override), logs in,
//! and runs an interactive terminal frontend with a plain line fallback. The engine accepts supported
//! asks and plays every active `/os` match automatically.

pub mod connection;
mod frontend;
pub mod protocol;
pub mod session;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant};

use reversi_core::board::Board;
use reversi_core::level::{MAX_LEVEL, get_level};
use reversi_core::search::result::SearchResult;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{self, SearchRunOptions, options::SearchOptions};
use reversi_core::square::Square;
use reversi_core::transposition_table::TranspositionTable;
use reversi_core::types::Scoref;

use crate::config::EngineConfig;

const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(50);

struct SearchPool {
    events: Sender<connection::Event>,
    shared: search::SearchSharedResources,
    capacity: usize,
    engines: Mutex<HashMap<String, EngineSlot>>,
    ponder_handles: Mutex<HashMap<String, WorkerHandle>>,
    search_handles: Mutex<HashMap<String, WorkerHandle>>,
    /// Held around every search and ponder: all engines share one thread
    /// pool, which runs one search at a time.
    search_turn: Arc<SearchTurn>,
}

/// The search lock. Unlike a plain mutex, a cancelled waiter leaves the
/// queue at once instead of holding its engine and transposition table until
/// its turn comes.
#[derive(Default)]
struct SearchTurn {
    busy: Mutex<bool>,
    changed: Condvar,
}

impl SearchTurn {
    /// Waits for the turn; `false` when `cancelled` is set first.
    fn acquire(&self, cancelled: &AtomicBool) -> bool {
        let mut busy = self.busy.lock().unwrap();
        while *busy && !cancelled.load(Ordering::Acquire) {
            busy = self.changed.wait(busy).unwrap();
        }
        if *busy {
            return false;
        }
        *busy = true;
        true
    }

    fn release(&self) {
        *self.busy.lock().unwrap() = false;
        self.changed.notify_all();
    }

    /// Wakes waiters to recheck their cancellation flag.
    fn wake(&self) {
        // Taking the lock orders this after any waiter's flag check.
        drop(self.busy.lock().unwrap());
        self.changed.notify_all();
    }
}

/// One game's engine plus the thread pool it searches on, captured so the
/// event loop never has to lock the engine mutex (which a winding-down ponder
/// may hold) just to reach the pool.
#[derive(Clone)]
struct EngineSlot {
    search: Arc<Mutex<search::Search>>,
    pool: Arc<search::threading::ThreadPool>,
    /// Set for synchro child games, shared with the sibling's slot.
    shared_tt: Option<Arc<Mutex<SharedTranspositionTable>>>,
}

/// Transposition table a synchro pair shares while both games follow the
/// same moves. The first game whose board leaves that line searches on a
/// copy from then on.
struct SharedTranspositionTable {
    /// Weak so each table lives exactly as long as the engines using it.
    tt: Weak<TranspositionTable>,
    /// Game id and the board it left after our latest move.
    last_left: Option<(String, Board)>,
    diverged: bool,
}

impl SharedTranspositionTable {
    /// Returns a copy of the table for `game_id` once its board is not the one
    /// the sibling game left after our latest move there. Must be called
    /// while holding the search lock: the copy needs the table free of writers.
    fn split_off(&mut self, game_id: &str, board: &Board) -> Option<TranspositionTable> {
        let diverged = !self.diverged
            && self
                .last_left
                .as_ref()
                .is_some_and(|(id, left)| id != game_id && left != board);
        self.diverged |= diverged;
        diverged
            .then(|| self.tt.upgrade())
            .flatten()
            // SAFETY: the caller holds the search lock, so no search writes the table.
            .map(|tt| unsafe { tt.duplicate() })
    }

    fn record_move(&mut self, game_id: &str, left: Board) {
        if !self.diverged {
            self.last_left = Some((game_id.to_string(), left));
        }
    }
}

/// Lifecycle of one search or ponder worker. `Running` holds exactly while
/// the worker owns the search lock around its search, so an abort issued
/// under the state mutex in `Running` can only ever hit that worker — never
/// another search that won the search lock before it (state still
/// `Starting`) or after it exited (`Finished`).
#[derive(Clone, Copy, PartialEq, Debug)]
enum WorkerState {
    /// Worker spawned but not yet past its cancellation check; the pool may
    /// even be held by another search right now. No abort may be issued.
    Starting,
    /// Worker holds the search lock and is searching (or about to).
    Running,
    /// Worker exited or was never able to start. No abort may be issued.
    Finished,
}

/// Cancellation state for one in-flight search or ponder.
struct WorkerHandle {
    cancelled: Arc<AtomicBool>,
    state: Arc<Mutex<WorkerState>>,
    pool: Arc<search::threading::ThreadPool>,
    turn: Arc<SearchTurn>,
}

/// Cancels a worker: one that has not started never searches, and a running
/// one is aborted until it exits.
fn stop_worker(handle: WorkerHandle, events: &Sender<connection::Event>) {
    handle.cancelled.store(true, Ordering::Release);
    handle.turn.wake();
    if *handle.state.lock().unwrap() != WorkerState::Running {
        return;
    }
    // start_thinking resets the pool's abort flag, so an abort landing in
    // the worker's startup window is silently lost — and a direct endgame
    // search would then hold the search lock for a full solve while
    // another search starves. Re-issue the abort until the worker exits;
    // aborts fire only in `Running`, i.e. only while the worker owns the
    // search lock, so they can never reach another search. The handle was
    // already removed from its map, so if the retry worker cannot be
    // spawned this is the last chance to stop the worker: retry inline.
    if thread::Builder::new()
        .spawn({
            let state = Arc::clone(&handle.state);
            let pool = Arc::clone(&handle.pool);
            let events = events.clone();
            move || {
                if !retry_abort_while_running(&state, &|| pool.abort_search()) {
                    let _ = events.send(connection::Event::Log(
                        "worker failed to stop within 30s".into(),
                    ));
                }
            }
        })
        .is_err()
    {
        let _ = events.send(connection::Event::Log(
            "failed to spawn worker stopper; retrying inline".into(),
        ));
        if !retry_abort_while_running(&handle.state, &|| handle.pool.abort_search()) {
            let _ = events.send(connection::Event::Log(
                "worker failed to stop within 30s".into(),
            ));
        }
    }
}

fn take_game_handles(handles: &mut HashMap<String, WorkerHandle>, id: &str) -> Vec<WorkerHandle> {
    handles
        .extract_if(|game_id, _| session::is_game_of(id, game_id))
        .map(|(_, handle)| handle)
        .collect()
}

/// Transitions a worker out of `Starting`: to `Running` when it may
/// search, to `Finished` when it was cancelled first. Must be called while
/// holding the search lock; the state mutex makes the cancellation check
/// atomic against [`stop_worker`].
fn begin_worker(cancelled: &AtomicBool, state: &Mutex<WorkerState>) -> bool {
    let mut state = state.lock().unwrap();
    if cancelled.load(Ordering::Acquire) {
        *state = WorkerState::Finished;
        false
    } else {
        *state = WorkerState::Running;
        true
    }
}

impl SearchPool {
    fn new(options: &SearchOptions, capacity: usize, events: Sender<connection::Event>) -> Self {
        let mut shared = search::SearchSharedResources::new(options);
        shared.share_thread_pool();
        Self {
            events,
            shared,
            capacity,
            engines: Mutex::new(HashMap::new()),
            ponder_handles: Mutex::new(HashMap::new()),
            search_handles: Mutex::new(HashMap::new()),
            search_turn: Arc::default(),
        }
    }

    fn stop_ponder(&self, game_id: &str) {
        if let Some(handle) = self.ponder_handles.lock().unwrap().remove(game_id) {
            stop_worker(handle, &self.events);
        }
    }

    fn engine_for(&self, game_id: &str) -> Result<EngineSlot, String> {
        let mut engines = self.engines.lock().unwrap();
        if let Some(slot) = engines.get(game_id) {
            return Ok(slot.clone());
        }
        if engines.len() == self.capacity {
            return Err(format!(
                "search engine capacity exhausted while starting {game_id}"
            ));
        }
        let sibling = session::synchro_sibling(game_id).map(|sibling| {
            let shared_tt = engines
                .get(&sibling)
                .and_then(|slot| slot.shared_tt.clone());
            let tt = shared_tt
                .as_ref()
                .and_then(|shared_tt| shared_tt.lock().unwrap().tt.upgrade());
            (shared_tt, tt)
        });
        let search = match sibling.as_ref().and_then(|(_, tt)| tt.clone()) {
            Some(tt) => search::Search::from_shared_resources_with_tt(&self.shared, tt),
            None => search::Search::from_shared_resources(&self.shared),
        };
        let shared_tt = sibling.map(|(shared_tt, _)| {
            shared_tt.unwrap_or_else(|| {
                Arc::new(Mutex::new(SharedTranspositionTable {
                    tt: Arc::downgrade(search.tt()),
                    last_left: None,
                    diverged: false,
                }))
            })
        });
        let slot = EngineSlot {
            pool: search.thread_pool(),
            search: Arc::new(Mutex::new(search)),
            shared_tt,
        };
        engines.insert(game_id.to_string(), slot.clone());
        Ok(slot)
    }

    /// Drops the engines of `id` (a match id, or a synchro game id from
    /// `/os: end`) and cancels their running or queued searches; their
    /// results still arrive and are discarded as stale.
    fn end_match(&self, id: &str) {
        self.engines
            .lock()
            .unwrap()
            .retain(|game_id, _| !session::is_game_of(id, game_id));
        let ended = take_game_handles(&mut self.search_handles.lock().unwrap(), id);
        for handle in ended {
            stop_worker(handle, &self.events);
        }
    }
}

/// Connection, authentication, and presentation settings for GGS mode.
pub struct GgsOptions<'a> {
    pub host: &'a str,
    pub port: u16,
    pub user: &'a str,
    pub password: &'a str,
    pub max_games: usize,
    pub ponder: bool,
    pub plain: bool,
    pub join_tournaments: bool,
    pub startup_commands: &'a [String],
    /// Appends the server transcript here, starting after the password.
    pub log_file: Option<&'a std::path::Path>,
}

/// Runs a GGS session with automatic engine play and a terminal or plain frontend.
pub fn run_ggs(options: GgsOptions<'_>, config: &EngineConfig) -> Result<(), String> {
    let GgsOptions {
        host,
        port,
        user,
        password,
        max_games,
        ponder,
        plain,
        join_tournaments,
        startup_commands,
        log_file,
    } = options;
    let startup_lines = startup_lines(user, password, max_games, startup_commands)?;
    let log_file = log_file
        .map(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("failed to open {}: {e}", path.display()))
        })
        .transpose()?;
    let mandatory_line_count = startup_lines.len() - startup_commands.len();
    let mut search_options = config.search_options();
    let max_game_slots = max_games * 2;
    search_options.tt_mb_size = partition_tt_size(search_options.tt_mb_size, max_games)?;
    let (mut conn, event_tx) = connection::Connection::connect(host, port)
        .map_err(|e| format!("failed to connect to {host}:{port}: {e}"))?;
    let search_pool = Arc::new(SearchPool::new(
        &search_options,
        max_game_slots,
        event_tx.clone(),
    ));
    let mut frontend =
        frontend::Frontend::start(plain, &event_tx, format!("{user}@{host}:{port}"))?;

    // The first two lines are the user name and password, kept out of the log.
    let (login, setup) = startup_lines[..mandatory_line_count].split_at(2);
    for line in login {
        conn.send_line(line)
            .map_err(|e| format!("login send failed: {e}"))?;
    }
    if let Some(file) = log_file {
        conn.log_to(file);
    }
    for line in setup {
        conn.send_line(line)
            .map_err(|e| format!("login send failed: {e}"))?;
    }
    for line in &startup_lines[mandatory_line_count..] {
        submit_command(&mut conn, &mut frontend, line)?;
    }

    let mut session = session::Session::new(user.to_string(), config.level, max_games);
    if ponder {
        session.enable_ponder();
    }
    let mut accumulator = connection::LineAccumulator::new();
    let mut finished_searches = Vec::new();
    frontend.draw()?;

    loop {
        let Some(event) = conn.recv_or_keepalive(KEEPALIVE_INTERVAL)? else {
            continue;
        };
        // Other updates wait for the next 100 ms tick, so a burst of server
        // lines costs one redraw instead of one per line.
        let redraw = matches!(
            event,
            connection::Event::Tick | connection::Event::Terminal(_)
        );
        match event {
            connection::Event::Server(line) => {
                frontend.server_line(&line);
                if is_os_offline(&line) {
                    return Err("/os went offline; reconnect required".to_string());
                }
                if join_tournaments {
                    for join in tournament_join(&line) {
                        submit_command(&mut conn, &mut frontend, &join)?;
                    }
                }
                if let Some(chunk) = accumulator.push(line) {
                    let chunk_refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
                    for (parsed, snapshot) in protocol::parse_server_messages(&chunk_refs, user) {
                        frontend.on_os_event(&parsed);
                        if let Some(snapshot) = snapshot {
                            frontend.snapshot(snapshot);
                        }
                        if let protocol::OsEvent::MatchEnd { id, .. } = &parsed {
                            search_pool.end_match(id);
                        }
                        for action in session.on_event(parsed) {
                            frontend.on_action(&action);
                            apply_action(action, &mut conn, &search_pool, &event_tx)?;
                        }
                    }
                }
            }
            connection::Event::Stdin(line) => {
                submit_command(&mut conn, &mut frontend, &line)?;
            }
            connection::Event::SearchStarted {
                game_id,
                search_id,
                ponder,
            } => {
                frontend.worker_started(&game_id, search_id, ponder);
            }
            connection::Event::SearchReport {
                game_id,
                search_id,
                message,
            } => {
                frontend.search_report(&game_id, search_id, &message);
            }
            connection::Event::PonderFinished { game_id, search_id } => {
                frontend.worker_finished(&game_id, search_id);
            }
            connection::Event::SearchFinished {
                game_id,
                search_id,
                queued_at,
                result,
            } => {
                frontend.worker_finished(&game_id, search_id);
                if let Ok(Some((_, score))) = &result {
                    frontend.eval(&game_id, search_id, *score);
                }
                finished_searches.push((game_id, search_id, queued_at, result));
            }
            connection::Event::Terminal(event) => match frontend.input(event) {
                frontend::Input::None => {}
                frontend::Input::Quit => break,
                frontend::Input::Commands(lines) => {
                    for line in lines {
                        submit_command(&mut conn, &mut frontend, &line)?;
                    }
                }
            },
            connection::Event::Log(message) => frontend.log(&message),
            connection::Event::Tick => {}
            connection::Event::InputFailed(error) => {
                return Err(format!("input read failed: {error}"));
            }
            connection::Event::SocketClosed => break,
            connection::Event::SocketReadFailed(error) => {
                return Err(format!("socket read failed: {error}"));
            }
        }
        if !accumulator.has_pending_lines() {
            for (game_id, search_id, queued_at, result) in finished_searches.drain(..) {
                let actions = match result {
                    Ok(mv) => session.on_move_ready(&game_id, search_id, mv, queued_at.elapsed()),
                    Err(error) => {
                        frontend.log(&format!("search failed for game {game_id}: {error}"));
                        session.on_search_aborted(&game_id, search_id)
                    }
                };
                for action in actions {
                    frontend.on_action(&action);
                    apply_action(action, &mut conn, &search_pool, &event_tx)?;
                }
            }
        }
        if redraw {
            frontend.draw()?;
        }
    }
    Ok(())
}

fn submit_command(
    conn: &mut connection::Connection,
    frontend: &mut frontend::Frontend,
    line: &str,
) -> Result<(), String> {
    frontend.command(line);
    if let Some(clock) = unsupported_ask_clock(line) {
        frontend.log(&format!(
            "not sending ask with unsupported or unknown clock {clock}"
        ));
        Ok(())
    } else {
        conn.send_line(line)
            .map_err(|e| format!("stdin->socket failed: {e}"))
    }
}

/// Builds the exact startup wire order. Authentication must come first; the
/// persistent prompt and parser modes are then normalized, with `open 0`
/// first so no ask is accepted under settings persisted by another client.
///
/// # Errors
///
/// Returns an error if a startup command contains a line break.
fn startup_lines(
    user: &str,
    password: &str,
    max_games: usize,
    startup_commands: &[String],
) -> Result<Vec<String>, String> {
    for (index, command) in startup_commands.iter().enumerate() {
        if command.contains('\r') || command.contains('\n') {
            return Err(format!("--command {} must be a single line", index + 1));
        }
    }

    let mut lines: Vec<String> = [
        user,
        password,
        "tell /os open 0",
        "vt100 -",
        "var _ready",
        "var _alert",
        "notify + /os",
        "tell /os client -",
        "tell /os trust +",
        "tell /os vt100 -",
        "tell /os aform",
        "tell /os dform size != 8 | anti | mt1 < 60 | mm1 != 0 | !ml1 | mm2 != 1 | !ma2",
        "tell /os request +",
        &format!("tell /os open {max_games}"),
    ]
    .map(str::to_string)
    .into();
    lines.extend_from_slice(startup_commands);
    Ok(lines)
}

fn is_os_offline(line: &str) -> bool {
    let mut fields = line.trim().trim_start_matches('\x07').split_whitespace();
    matches!(
        (fields.next(), fields.next(), fields.next()),
        (Some(":"), Some("-"), Some("/os"))
    )
}

/// The tournament joins to send when a user tells us `tell /td join <id>`,
/// e.g. `HCyrano: tell /td join .3`. The tournament announcement asks to relay
/// `tell /os join <id>`, but the GGS source has no `/os join` and handles joins
/// in `/td`, so that form is sent to both. Only joins are relayed, so any
/// sender is accepted.
fn tournament_join(line: &str) -> Vec<String> {
    let mut fields = line.trim_start_matches('\x07').split_whitespace();
    match (
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
    ) {
        (
            Some(from),
            Some("tell" | "t"),
            Some(service @ ("/td" | "/os")),
            Some("join"),
            Some(id),
            None,
        ) if from.len() > 1 && from.ends_with(':') && !from.starts_with('/') => {
            let mut joins = vec![format!("tell /td join {id}")];
            if service == "/os" {
                joins.push(format!("tell /os join {id}"));
            }
            joins
        }
        _ => Vec::new(),
    }
}

fn unsupported_ask_clock(line: &str) -> Option<&str> {
    let mut fields = line.split_whitespace();
    if !matches!(
        (fields.next(), fields.next(), fields.next()),
        (Some("tell"), Some("/os"), Some("ask"))
    ) {
        return None;
    }
    let game_type = fields.next()?;
    if game_type.starts_with('.') {
        return Some(game_type);
    }
    let clock = fields.next()?;
    (!session::is_supported_ask_clock(clock)).then_some(clock)
}

fn partition_tt_size(total_tt_mb: usize, max_matches: usize) -> Result<usize, String> {
    let max_game_slots = max_matches * 2;
    if total_tt_mb < max_game_slots {
        return Err(format!(
            "--max-games {max_matches} requires at least {max_game_slots} MB of hash for synchro boards (got {total_tt_mb} MB)"
        ));
    }
    Ok(total_tt_mb / max_game_slots)
}

fn apply_action(
    action: session::SessionAction,
    conn: &mut connection::Connection,
    search_pool: &Arc<SearchPool>,
    event_tx: &Sender<connection::Event>,
) -> Result<(), String> {
    match action {
        session::SessionAction::Send(line) => {
            conn.send_line(&line)
                .map_err(|e| format!("send failed: {e}"))?;
        }
        session::SessionAction::Log(msg) => {
            let _ = event_tx.send(connection::Event::Log(msg));
        }
        session::SessionAction::StartSearch {
            game_id,
            search_id,
            board,
            search_limit,
        } => {
            let queued_at = Instant::now();
            // Each game id keeps one engine for the full game so its
            // endgame-depth tracking cannot leak to or be lost to another
            // concurrently active game.
            let slot = match search_pool.engine_for(&game_id) {
                Ok(slot) => slot,
                Err(error) => {
                    let _ = event_tx.send(connection::Event::SearchFinished {
                        game_id,
                        search_id,
                        queued_at,
                        result: Err(error),
                    });
                    return Ok(());
                }
            };
            let shared_tt = slot.shared_tt.clone();
            let job_id = game_id.clone();
            let tx = event_tx.clone();
            let done_id = game_id.clone();
            let log_tx = event_tx.clone();
            let spawned = spawn_worker(
                search_pool,
                &search_pool.search_handles,
                &game_id,
                slot,
                move |search| {
                    if let Some(shared_tt) = &shared_tt
                        && let Some(tt) = shared_tt.lock().unwrap().split_off(&job_id, &board)
                    {
                        search.set_tt(Arc::new(tt));
                        let _ = log_tx.send(connection::Event::Log(format!(
                            "{job_id} left the synchro line; searching on a TT copy"
                        )));
                    }
                    // Our clock kept running while another game held the pool.
                    let waited = queued_at.elapsed();
                    let search_limit = deduct_wait(search_limit, waited);
                    let options = build_search_run_options(search_limit);
                    let _ = log_tx.send(connection::Event::SearchStarted {
                        game_id: job_id.clone(),
                        search_id,
                        ponder: false,
                    });
                    let started = Instant::now();
                    let result = search.run(&board, &options);
                    let elapsed = started.elapsed();
                    let mv = result.best_move().zip(result.score());
                    if let (Some(shared_tt), Some((sq, _))) = (&shared_tt, mv) {
                        shared_tt
                            .lock()
                            .unwrap()
                            .record_move(&job_id, board.make_move(sq));
                    }
                    let report = mv.map(|(sq, score)| {
                        search_report(&job_id, sq, score, &result, elapsed, waited)
                    });
                    (mv, report)
                },
                move |result| {
                    let (result, report) = match result {
                        Some((mv, report)) => (Ok(mv), report),
                        None => (
                            Err("search cancelled because its match ended".to_string()),
                            None,
                        ),
                    };
                    let _ = tx.send(connection::Event::SearchFinished {
                        game_id: done_id.clone(),
                        search_id,
                        queued_at,
                        result,
                    });
                    // Printed after the move is handed off, so it costs no clock time.
                    if let Some(report) = report {
                        let _ = tx.send(connection::Event::SearchReport {
                            game_id: done_id,
                            search_id,
                            message: report,
                        });
                    }
                },
            );
            if let Err(error) = spawned {
                let _ = event_tx.send(connection::Event::SearchFinished {
                    game_id,
                    search_id,
                    queued_at,
                    result: Err(format!("failed to spawn search worker: {error}")),
                });
            }
        }
        session::SessionAction::StartPonder {
            game_id,
            search_id,
            board,
        } => {
            let slot = match search_pool.engine_for(&game_id) {
                Ok(slot) => slot,
                Err(error) => {
                    let _ = event_tx.send(connection::Event::Log(format!(
                        "ponder skipped for {game_id}: {error}"
                    )));
                    let _ = event_tx.send(connection::Event::PonderFinished { game_id, search_id });
                    return Ok(());
                }
            };
            let _ = event_tx.send(connection::Event::Log(format!("pondering {game_id}")));
            let tx = event_tx.clone();
            let done_id = game_id.clone();
            let started_tx = event_tx.clone();
            let started_id = game_id.clone();
            let spawned = spawn_worker(
                search_pool,
                &search_pool.ponder_handles,
                &game_id,
                slot,
                move |search| {
                    let _ = started_tx.send(connection::Event::SearchStarted {
                        game_id: started_id,
                        search_id,
                        ponder: true,
                    });
                    search.run(
                        &board,
                        &SearchRunOptions::with_time(TimeControlMode::Infinite),
                    );
                },
                move |_| {
                    let _ = tx.send(connection::Event::PonderFinished {
                        game_id: done_id,
                        search_id,
                    });
                },
            );
            if let Err(error) = spawned {
                let _ = event_tx.send(connection::Event::Log(format!(
                    "failed to spawn ponder worker: {error}"
                )));
                let _ = event_tx.send(connection::Event::PonderFinished { game_id, search_id });
            }
        }
        session::SessionAction::StopPonder { game_id } => {
            search_pool.stop_ponder(&game_id);
        }
    }
    Ok(())
}

/// Runs `job` on the game's engine in a new worker thread once it holds the
/// search lock, then hands its result to `done` after releasing the locks
/// (`None` when the worker was cancelled before it began). The worker is
/// registered in `handles` under `game_id` so it can be stopped.
fn spawn_worker<T: Send + 'static>(
    search_pool: &Arc<SearchPool>,
    handles: &Mutex<HashMap<String, WorkerHandle>>,
    game_id: &str,
    slot: EngineSlot,
    job: impl FnOnce(&mut search::Search) -> T + Send + 'static,
    done: impl FnOnce(Option<T>) + Send + 'static,
) -> std::io::Result<()> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let state = Arc::new(Mutex::new(WorkerState::Starting));
    handles.lock().unwrap().insert(
        game_id.to_string(),
        WorkerHandle {
            cancelled: Arc::clone(&cancelled),
            state: Arc::clone(&state),
            pool: Arc::clone(&slot.pool),
            turn: Arc::clone(&search_pool.search_turn),
        },
    );
    let turn = Arc::clone(&search_pool.search_turn);
    let worker_state = Arc::clone(&state);
    let spawned = thread::Builder::new().spawn(move || {
        if !turn.acquire(&cancelled) {
            *worker_state.lock().unwrap() = WorkerState::Finished;
            drop(slot);
            done(None);
            return;
        }
        let mut search = slot.search.lock().unwrap();
        let result = begin_worker(&cancelled, &worker_state).then(|| job(&mut search));
        // Publish exit while the search lock is still held: a stopper's
        // abort retries stop before another search can acquire the pool.
        *worker_state.lock().unwrap() = WorkerState::Finished;
        drop(search);
        turn.release();
        drop(slot);
        done(result);
    });
    if spawned.is_err() {
        *state.lock().unwrap() = WorkerState::Finished;
        handles.lock().unwrap().remove(game_id);
    }
    spawned.map(drop)
}

/// One-line summary of a finished search, e.g.
/// `.5 f5 +3 | depth 22@73% | 12.3M nodes | 1.84s (wait 0.20s) | 6.7 Mnps`.
fn search_report(
    game_id: &str,
    sq: Square,
    score: Scoref,
    result: &SearchResult,
    elapsed: Duration,
    waited: Duration,
) -> String {
    let secs = elapsed.as_secs_f64();
    let nodes = result.n_nodes() as f64;
    let wait = if waited >= Duration::from_millis(50) {
        format!(" (wait {:.2}s)", waited.as_secs_f64())
    } else {
        String::new()
    };
    format!(
        "{game_id} {sq} {:+} | depth {}@{}% | {} nodes | {secs:.2}s{wait} | {:.1} Mnps",
        score.round() as i32,
        result.depth(),
        result.get_probability(),
        si_count(nodes),
        nodes / secs.max(1e-9) / 1e6,
    )
}

/// Formats a count with a K/M/G suffix, e.g. `12.3M`.
fn si_count(n: f64) -> String {
    match n {
        n if n >= 1e9 => format!("{:.1}G", n / 1e9),
        n if n >= 1e6 => format!("{:.1}M", n / 1e6),
        n if n >= 1e3 => format!("{:.1}K", n / 1e3),
        n => format!("{n}"),
    }
}

/// Takes the time spent waiting for the shared pool off the clock's main time.
fn deduct_wait(search_limit: session::SearchLimit, waited: Duration) -> session::SearchLimit {
    match search_limit {
        session::SearchLimit::Time(TimeControlMode::Fischer {
            main_time_ms,
            increment_ms,
        }) => session::SearchLimit::Time(TimeControlMode::Fischer {
            main_time_ms: main_time_ms.saturating_sub(waited.as_millis() as u64),
            increment_ms,
        }),
        other => other,
    }
}

fn build_search_run_options(search_limit: session::SearchLimit) -> SearchRunOptions {
    match search_limit {
        session::SearchLimit::Time(time_mode) => SearchRunOptions::with_time(time_mode),
        session::SearchLimit::Level { level } => {
            SearchRunOptions::with_level(get_level(level.min(MAX_LEVEL)))
        }
    }
}

/// Re-issues `abort` under the state mutex for as long as the worker is
/// `Running` (bounded at 30s). See [`stop_worker`] for why a single abort is
/// not enough and why aborting outside `Running` is unsafe.
fn retry_abort_while_running(state: &Mutex<WorkerState>, abort: &dyn Fn()) -> bool {
    for _ in 0..30_000 {
        {
            let state = state.lock().unwrap();
            match *state {
                WorkerState::Running => abort(),
                // Finished: worker exited. Starting: the worker will see
                // `cancelled` and never search, while the pool may belong to
                // another search an abort would hit.
                WorkerState::Finished | WorkerState::Starting => return true,
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_budget_reserves_two_boards_per_match() {
        assert_eq!(partition_tt_size(256, 1).unwrap(), 128);
        assert_eq!(partition_tt_size(256, 2).unwrap(), 64);
        assert!(partition_tt_size(1, 1).is_err());
    }

    #[test]
    fn startup_logs_in_then_normalizes_modes() {
        assert_eq!(
            startup_lines("alice", "secret", 2, &[]).unwrap(),
            [
                "alice",
                "secret",
                "tell /os open 0",
                "vt100 -",
                "var _ready",
                "var _alert",
                "notify + /os",
                "tell /os client -",
                "tell /os trust +",
                "tell /os vt100 -",
                "tell /os aform",
                "tell /os dform size != 8 | anti | mt1 < 60 | mm1 != 0 | !ml1 | mm2 != 1 | !ma2",
                "tell /os request +",
                "tell /os open 2",
            ]
            .map(str::to_string)
        );
    }

    #[test]
    fn startup_commands_follow_normalization_in_order() {
        let commands = [
            "tell /os rated +".to_string(),
            "  info spaced  ".to_string(),
            String::new(),
        ];

        let lines = startup_lines("alice", "secret", 2, &commands).unwrap();

        assert_eq!(&lines[lines.len() - commands.len()..], commands);
        assert_eq!(lines[lines.len() - commands.len() - 1], "tell /os open 2");
    }

    #[test]
    fn startup_commands_reject_line_breaks() {
        for command in ["info first\ninfo second", "info first\rinfo second"] {
            let error = startup_lines("alice", "secret", 1, &[command.to_string()]).unwrap_err();
            assert_eq!(error, "--command 1 must be a single line");
        }
    }

    #[test]
    fn only_tournament_join_tells_are_relayed() {
        for (line, id) in [
            ("HCyrano: tell /td join .3", ".3"),
            ("\x07bob: t /td join .12", ".12"),
            ("bob: tell /td join 3", "3"),
        ] {
            assert_eq!(tournament_join(line), [format!("tell /td join {id}")]);
        }
        assert_eq!(
            tournament_join("HCyrano: tell /os join .3"),
            ["tell /td join .3", "tell /os join .3"]
        );
        for line in [
            "HCyrano: tell /td withdraw .3",
            "HCyrano: tell /os ask s8r14 3:00//0:30 bob",
            "HCyrano: tell /td join .3 now",
            "/os: tell /td join .3",
            "tell /td join .3",
        ] {
            assert!(tournament_join(line).is_empty(), "{line}");
        }
    }

    #[test]
    fn only_os_offline_notification_requires_reconnect() {
        assert!(!is_os_offline(": + /os 1"));
        assert!(is_os_offline(": - /os 1"));
        assert!(is_os_offline("\x07: - /os 0"));
        assert!(!is_os_offline("/os: - match .1"));
    }

    #[test]
    fn unsupported_outbound_ask_clock_is_found() {
        assert_eq!(
            unsupported_ask_clock("tell /os ask 8 01:00,N0//02:00 bob"),
            Some("01:00,N0//02:00")
        );
        assert_eq!(
            unsupported_ask_clock("tell /os ask 8 01:00//02:00 bob"),
            None
        );
        assert_eq!(unsupported_ask_clock("tell /os ask .42"), Some(".42"));
        assert_eq!(
            unsupported_ask_clock("tell /os ask .42 ignored"),
            Some(".42")
        );
        assert_eq!(unsupported_ask_clock("tell /os ask 8"), None);
    }

    #[test]
    fn ponder_stop_in_starting_never_aborts_and_worker_self_cancels() {
        use std::sync::atomic::AtomicUsize;

        // Startup race: the real search won the search lock first, so the
        // worker is still `Starting` when the stop arrives. No abort may be
        // issued — it would hit the real search through the shared pool.
        let cancelled = AtomicBool::new(false);
        let state = Mutex::new(WorkerState::Starting);
        cancelled.store(true, Ordering::Release);

        let aborts = AtomicUsize::new(0);
        retry_abort_while_running(&state, &|| {
            aborts.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(aborts.load(Ordering::Relaxed), 0);

        // The worker later observes the cancellation and never searches.
        assert!(!begin_worker(&cancelled, &state));
        assert_eq!(*state.lock().unwrap(), WorkerState::Finished);
    }

    #[test]
    fn synchro_pair_splits_tt_only_when_a_board_leaves_the_sibling_line() {
        let tt = Arc::new(TranspositionTable::new(0));
        let mut shared = SharedTranspositionTable {
            tt: Arc::downgrade(&tt),
            last_left: None,
            diverged: false,
        };
        let start = Board::new();
        let [a, b] = [Square::D3, Square::C4].map(|sq| start.make_move(sq));

        assert!(shared.split_off(".5.0", &start).is_none());
        shared.record_move(".5.0", a);
        // Same game again (the opponent passed there): no comparison yet.
        assert!(shared.split_off(".5.0", &b).is_none());
        // The opponent repeated our move in the sibling game.
        assert!(shared.split_off(".5.1", &a).is_none());
        shared.record_move(".5.1", a.make_move(Square::C3));

        assert!(shared.split_off(".5.0", &a.make_move(Square::E3)).is_some());
        assert!(shared.diverged);
        // Only the engines keep the original table alive.
        drop(tt);
        assert!(shared.tt.upgrade().is_none());
        assert!(shared.split_off(".5.1", &b).is_none());
    }

    #[test]
    fn cancelled_waiter_leaves_a_busy_search_turn() {
        let turn = Arc::new(SearchTurn::default());
        let never = AtomicBool::new(false);
        assert!(turn.acquire(&never));

        let cancelled = Arc::new(AtomicBool::new(false));
        let waiter = thread::spawn({
            let turn = Arc::clone(&turn);
            let cancelled = Arc::clone(&cancelled);
            move || turn.acquire(&cancelled)
        });
        cancelled.store(true, Ordering::Release);
        turn.wake();
        assert!(!waiter.join().unwrap());

        // The turn is still held, so a live waiter gets it only on release.
        assert!(*turn.busy.lock().unwrap());
        turn.release();
        assert!(turn.acquire(&never));
    }

    #[test]
    fn match_end_cancels_only_its_queued_searches() {
        let pool = search::threading::ThreadPool::new(1);
        let mut handles: HashMap<String, WorkerHandle> = [".5", ".5.0", ".5.1", ".50", ".6"]
            .into_iter()
            .map(|id| {
                let handle = WorkerHandle {
                    cancelled: Arc::new(AtomicBool::new(false)),
                    state: Arc::new(Mutex::new(WorkerState::Starting)),
                    pool: Arc::clone(&pool),
                    turn: Arc::default(),
                };
                (id.to_string(), handle)
            })
            .collect();

        let ended = take_game_handles(&mut handles, ".5");
        let workers: Vec<_> = ended
            .iter()
            .map(|handle| (Arc::clone(&handle.cancelled), Arc::clone(&handle.state)))
            .collect();
        for handle in ended {
            stop_worker(handle, &std::sync::mpsc::channel().0);
        }

        assert_eq!(workers.len(), 3);
        for (cancelled, state) in &workers {
            assert!(!begin_worker(cancelled, state));
        }
        let mut remaining: Vec<_> = handles.keys().map(String::as_str).collect();
        remaining.sort_unstable();
        assert_eq!(remaining, [".50", ".6"]);
    }

    #[test]
    fn ponder_stopper_aborts_only_while_running() {
        use std::sync::atomic::AtomicUsize;

        let cancelled = AtomicBool::new(false);
        let state = Arc::new(Mutex::new(WorkerState::Starting));
        assert!(begin_worker(&cancelled, &state));
        assert_eq!(*state.lock().unwrap(), WorkerState::Running);

        let aborts = Arc::new(AtomicUsize::new(0));
        let finisher = {
            let state = Arc::clone(&state);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(10));
                *state.lock().unwrap() = WorkerState::Finished;
            })
        };
        retry_abort_while_running(&state, &|| {
            aborts.fetch_add(1, Ordering::Relaxed);
        });
        finisher.join().unwrap();
        assert!(aborts.load(Ordering::Relaxed) >= 1);

        // Once Finished, retries must abort nothing.
        let late_aborts = AtomicUsize::new(0);
        retry_abort_while_running(&state, &|| {
            late_aborts.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(late_aborts.load(Ordering::Relaxed), 0);
    }
}
