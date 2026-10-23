//! GGS (Othello Generic Game Server) play mode.
//!
//! Connects to `localhost:5000` by default (or host/port override), logs in,
//! and runs an interactive terminal frontend with a plain line fallback. The engine accepts supported
//! asks and plays every active `/os` match automatically.

pub mod connection;
mod frontend;
pub mod protocol;
pub mod report;
pub mod session;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread;
use std::time::{Duration, Instant};

use reversi_core::board::Board;
use reversi_core::level::{MAX_LEVEL, get_level};
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{self, SearchRunOptions, StopToken, options::SearchOptions};
use reversi_core::transposition_table::TranspositionTable;

use crate::config::EngineConfig;

const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(50);

struct SearchPool {
    shared: search::SearchSharedResources,
    capacity: usize,
    login: String,
    n_threads: usize,
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

/// One game's engine.
#[derive(Clone)]
struct EngineSlot {
    search: Arc<Mutex<search::Search>>,
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

/// Cancellation state for one in-flight search or ponder.
struct WorkerHandle {
    /// Keeps a worker still waiting for the search lock from ever searching.
    cancelled: Arc<AtomicBool>,
    /// Stops the worker's search, even one that is just starting. It never
    /// reaches another engine's search on the shared pool.
    stop: StopToken,
    turn: Arc<SearchTurn>,
}

/// Cancels a worker: one that has not started never searches, and a running
/// one stops at once.
fn stop_worker(handle: WorkerHandle) {
    handle.cancelled.store(true, Ordering::Release);
    handle.stop.stop();
    handle.turn.wake();
}

fn take_game_handles(handles: &mut HashMap<String, WorkerHandle>, id: &str) -> Vec<WorkerHandle> {
    handles
        .extract_if(|game_id, _| session::is_game_of(id, game_id))
        .map(|(_, handle)| handle)
        .collect()
}

impl SearchPool {
    fn new(options: &SearchOptions, login: &str, capacity: usize) -> Self {
        let mut shared = search::SearchSharedResources::new(options);
        shared.share_thread_pool();
        let n_threads = shared.n_threads();
        Self {
            shared,
            capacity,
            login: login.to_string(),
            n_threads,
            engines: Mutex::new(HashMap::new()),
            ponder_handles: Mutex::new(HashMap::new()),
            search_handles: Mutex::new(HashMap::new()),
            search_turn: Arc::default(),
        }
    }

    fn stop_ponder(&self, game_id: &str) {
        if let Some(handle) = self.ponder_handles.lock().unwrap().remove(game_id) {
            stop_worker(handle);
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
            stop_worker(handle);
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

/// Why a GGS session ended with an error.
#[derive(Debug)]
pub enum GgsError {
    /// The server refused the login or password; reconnecting cannot help.
    LoginRejected(String),
    Other(String),
}

impl From<String> for GgsError {
    fn from(error: String) -> Self {
        Self::Other(error)
    }
}

impl std::fmt::Display for GgsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoginRejected(line) => write!(f, "login rejected: {line}"),
            Self::Other(error) => f.write_str(error),
        }
    }
}

/// Runs a GGS session with automatic engine play and a terminal or plain frontend.
pub fn run_ggs(options: GgsOptions<'_>, config: &EngineConfig) -> Result<(), GgsError> {
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
    let search_pool = Arc::new(SearchPool::new(&search_options, user, max_game_slots));
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
    // The server sends its first READY (ALERT when low on memory) only after
    // accepting the login; later password errors come from `passw` and must
    // not end the session.
    let mut logged_in = false;
    frontend.draw(session.games())?;

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
                    return Err("/os went offline; reconnect required".to_string().into());
                }
                if !logged_in {
                    if is_login_rejected(&line) {
                        return Err(GgsError::LoginRejected(
                            line.trim().trim_start_matches(": ").to_string(),
                        ));
                    }
                    logged_in = connection::is_message_boundary(&line);
                }
                if join_tournaments {
                    for join in tournament_join(&line) {
                        submit_command(&mut conn, &mut frontend, &join)?;
                    }
                }
                if let Some(chunk) = accumulator.push(line) {
                    let chunk_refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
                    for parsed in protocol::parse_server_messages(&chunk_refs, user) {
                        for action in session.on_event(parsed) {
                            apply_action(action, &mut conn, &search_pool, &event_tx)?;
                        }
                    }
                }
            }
            connection::Event::Stdin(line) => {
                submit_command(&mut conn, &mut frontend, &line)?;
            }
            connection::Event::SearchStarted { game_id, search_id } => {
                session.on_search_started(&game_id, search_id);
            }
            connection::Event::PonderFinished { game_id, search_id } => {
                session.on_ponder_finished(&game_id, search_id);
            }
            connection::Event::SearchFinished {
                game_id,
                search_id,
                queued_at,
                result,
            } => {
                finished_searches.push((game_id, search_id, queued_at, result));
            }
            connection::Event::Terminal(event) => match frontend.input(event, session.games()) {
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
                return Err(format!("input read failed: {error}").into());
            }
            connection::Event::SocketClosed => break,
            connection::Event::SocketReadFailed(error) => {
                return Err(format!("socket read failed: {error}").into());
            }
        }
        if !accumulator.has_pending_lines() {
            for (game_id, search_id, queued_at, result) in finished_searches.drain(..) {
                let elapsed = queued_at.elapsed();
                for action in session.on_search_finished(&game_id, search_id, result, elapsed) {
                    apply_action(action, &mut conn, &search_pool, &event_tx)?;
                }
            }
        }
        if redraw {
            frontend.draw(session.games())?;
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
        &format!("chann + {}", log_channel(user)),
        "tell /os client -",
        "tell /os trust +",
        "tell /os vt100 -",
        "tell /os aform",
        "tell /os dform size != 8 | anti | komi | saved | mt1 < 60 | mm1 != 0 | !ml1 | mm2 != 1 | !ma2",
        "tell /os request +",
        &format!("tell /os open {max_games}"),
    ]
    .map(str::to_string)
    .into();
    lines.extend_from_slice(startup_commands);
    Ok(lines)
}

/// The channel spectators join (`chann + .name`) to follow our search reports:
/// the login after a dot, cut to GGS's 8-character channel name limit.
fn log_channel(user: &str) -> String {
    format!(".{}", user.chars().take(7).collect::<String>())
}

/// The server's answers to a bad password or login name, sent before it asks
/// for the login again.
fn is_login_rejected(line: &str) -> bool {
    let line = line.trim();
    line.starts_with(": ERR Password") || line.starts_with(": ERR Login")
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
        session::SessionAction::Broadcast(report) => {
            // A channel tell overwrites `_send`, the target of an operator's
            // `tell $_send` reply; keep the private one.
            for line in [
                "var _nrsend $_send".to_string(),
                format!("tell {} {report}", log_channel(&search_pool.login)),
                "var _send $_nrsend".to_string(),
            ] {
                conn.send_line(&line)
                    .map_err(|e| format!("send failed: {e}"))?;
            }
        }
        session::SessionAction::EndMatch { id } => search_pool.end_match(&id),
        session::SessionAction::StartSearch {
            game_id,
            search_id,
            board,
            color,
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
            let n_threads = search_pool.n_threads;
            let login = search_pool.login.clone();
            let spawned = spawn_worker(
                search_pool,
                &search_pool.search_handles,
                &game_id,
                slot,
                move |search, stop| {
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
                    let options = build_search_run_options(search_limit).stop_token(stop);
                    let _ = log_tx.send(connection::Event::SearchStarted {
                        game_id: job_id.clone(),
                        search_id,
                    });
                    let started = Instant::now();
                    let result = search.run(&board, &options);
                    let elapsed = started.elapsed();
                    let timed_out = search.timed_out();
                    let mv = result.best_move().zip(result.score());
                    if let (Some(shared_tt), Some((sq, _))) = (&shared_tt, mv) {
                        shared_tt
                            .lock()
                            .unwrap()
                            .record_move(&job_id, board.make_move(sq));
                    }
                    mv.map(|(sq, score)| session::FoundMove {
                        report: report::SearchReport {
                            sq,
                            score,
                            depth: result.depth(),
                            probability: result.probability(),
                            nodes: result.n_nodes(),
                            elapsed,
                            waited,
                        },
                        broadcast: report::channel_report(
                            &login, &job_id, &board, color, &result, elapsed, waited, n_threads,
                            timed_out,
                        ),
                    })
                },
                move |mv| {
                    let _ = tx.send(connection::Event::SearchFinished {
                        game_id: done_id,
                        search_id,
                        queued_at,
                        result: mv
                            .ok_or_else(|| "search cancelled because its match ended".to_string()),
                    });
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
                move |search, stop| {
                    let _ = started_tx.send(connection::Event::SearchStarted {
                        game_id: started_id,
                        search_id,
                    });
                    search.run(
                        &board,
                        &SearchRunOptions::with_time(TimeControlMode::Infinite).stop_token(stop),
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
/// registered in `handles` under `game_id` so it can be stopped; `job` must
/// give its search the stop token it receives.
fn spawn_worker<T: Send + 'static>(
    search_pool: &Arc<SearchPool>,
    handles: &Mutex<HashMap<String, WorkerHandle>>,
    game_id: &str,
    slot: EngineSlot,
    job: impl FnOnce(&mut search::Search, StopToken) -> T + Send + 'static,
    done: impl FnOnce(Option<T>) + Send + 'static,
) -> std::io::Result<()> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let stop = StopToken::new();
    handles.lock().unwrap().insert(
        game_id.to_string(),
        WorkerHandle {
            cancelled: Arc::clone(&cancelled),
            stop: stop.clone(),
            turn: Arc::clone(&search_pool.search_turn),
        },
    );
    let turn = Arc::clone(&search_pool.search_turn);
    let spawned = thread::Builder::new().spawn(move || {
        if !turn.acquire(&cancelled) {
            drop(slot);
            done(None);
            return;
        }
        let mut search = slot.search.lock().unwrap();
        let result = (!cancelled.load(Ordering::Acquire)).then(|| job(&mut search, stop));
        drop(search);
        turn.release();
        drop(slot);
        done(result);
    });
    if spawned.is_err() {
        handles.lock().unwrap().remove(game_id);
    }
    spawned.map(drop)
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

#[cfg(test)]
mod tests {
    use reversi_core::square::Square;

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
                "chann + .alice",
                "tell /os client -",
                "tell /os trust +",
                "tell /os vt100 -",
                "tell /os aform",
                "tell /os dform size != 8 | anti | komi | saved | mt1 < 60 | mm1 != 0 | !ml1 | mm2 != 1 | !ma2",
                "tell /os request +",
                "tell /os open 2",
            ]
            .map(str::to_string)
        );
    }

    #[test]
    fn log_channel_fits_ggs_name_limit() {
        assert_eq!(log_channel("Melody"), ".Melody");
        assert_eq!(log_channel("Rhapsody"), ".Rhapsod");
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
    fn only_login_errors_reject_the_login() {
        assert!(is_login_rejected(": ERR Password is incorrect."));
        assert!(is_login_rejected(
            ": ERR Login (2-8 chars) can not start with '.' or '_' chars"
        ));
        assert!(!is_login_rejected(": ERR 3: unknown command"));
        assert!(!is_login_rejected("bob: ERR Password is incorrect."));
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
        let mut handles: HashMap<String, WorkerHandle> = [".5", ".5.0", ".5.1", ".50", ".6"]
            .into_iter()
            .map(|id| {
                let handle = WorkerHandle {
                    cancelled: Arc::new(AtomicBool::new(false)),
                    stop: StopToken::new(),
                    turn: Arc::default(),
                };
                (id.to_string(), handle)
            })
            .collect();

        let ended = take_game_handles(&mut handles, ".5");
        let cancelled: Vec<_> = ended
            .iter()
            .map(|handle| Arc::clone(&handle.cancelled))
            .collect();
        for handle in ended {
            stop_worker(handle);
        }

        assert_eq!(cancelled.len(), 3);
        assert!(cancelled.iter().all(|flag| flag.load(Ordering::Acquire)));
        let mut remaining: Vec<_> = handles.keys().map(String::as_str).collect();
        remaining.sort_unstable();
        assert_eq!(remaining, [".50", ".6"]);
    }
}
