//! GGS (Othello Generic Game Server) play mode.
//!
//! Connects to `localhost:5000` by default (or host/port override), replays
//! an init script, and runs a transparent REPL. The engine only
//! auto-generates one kind of output: moves played when it is our turn in an
//! /os match.

pub mod connection;
pub mod protocol;
pub mod session;
pub mod time;

use std::path::Path;
use std::sync::mpsc::{RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use reversi_core::level::{MAX_LEVEL, get_level};
use reversi_core::probcut::Selectivity;
use reversi_core::search::{self, SearchRunOptions, options::SearchOptions};

/// Send a `tell /os continue` keepalive whenever we have not sent anything
/// on the socket for this long. Edax uses 60 s (`ggs.c:1093`) measured
/// against the last *send* time, not the last receive — a kibitz-only RX
/// stream during a deep search still leaves us silent from the server's
/// perspective. We undercut 60 s so drift and scheduling jitter do not push
/// us past the server's idle timer.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(50);

struct SearchPool {
    shared: search::SearchSharedResources,
    idle: Mutex<Vec<search::Search>>,
}

struct SearchLease {
    search: Option<search::Search>,
    pool: Arc<SearchPool>,
}

impl SearchPool {
    fn new(options: &SearchOptions) -> Self {
        let shared = search::SearchSharedResources::new(options);
        let initial_search = search::Search::from_shared_resources(&shared);
        Self {
            shared,
            idle: Mutex::new(vec![initial_search]),
        }
    }

    fn acquire(self: &Arc<Self>) -> SearchLease {
        let search = lock_idle(&self.idle)
            .pop()
            .unwrap_or_else(|| search::Search::from_shared_resources(&self.shared));
        SearchLease {
            search: Some(search),
            pool: Arc::clone(self),
        }
    }

    fn release(&self, search: search::Search) {
        lock_idle(&self.idle).push(search);
    }
}

impl SearchLease {
    fn search_mut(&mut self) -> &mut search::Search {
        self.search
            .as_mut()
            .expect("search lease must hold an engine until drop")
    }

    fn mark_broken(&mut self) {
        self.search.take();
    }
}

impl Drop for SearchLease {
    fn drop(&mut self) {
        if let Some(search) = self.search.take() {
            self.pool.release(search);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_ggs(
    script: &Path,
    host: &str,
    port: u16,
    user: &str,
    hash_size: usize,
    level: usize,
    selectivity: Selectivity,
    threads: Option<usize>,
    eval_file: Option<&Path>,
    eval_sm_file: Option<&Path>,
) -> Result<(), String> {
    let search_options = SearchOptions::new(hash_size)
        .with_threads(threads)
        .with_eval_paths(eval_file, eval_sm_file);
    let search_pool = Arc::new(SearchPool::new(&search_options));

    // Blank and `#`-prefixed lines are skipped so the `init.ggs.example`
    // template works as distributed; GGS commands never start with `#`.
    let script_lines: Vec<String> = std::fs::read_to_string(script)
        .map_err(|e| format!("failed to read script {}: {e}", script.display()))?
        .lines()
        .filter_map(|raw| {
            let trimmed = raw.trim_start();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                Some(raw.to_string())
            }
        })
        .collect();

    let (mut conn, event_tx) = connection::Connection::connect(host, port)
        .map_err(|e| format!("failed to connect to {host}:{port}: {e}"))?;

    for line in &script_lines {
        conn.send_line(line)
            .map_err(|e| format!("script send failed: {e}"))?;
    }

    let mut session = session::Session::new(user.to_string(), level, selectivity);
    let mut accumulator = connection::LineAccumulator::new();

    loop {
        let event = match conn.recv_timeout(conn.time_until_keepalive(KEEPALIVE_INTERVAL)) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => {
                if let Err(e) = conn.send_line("tell /os continue") {
                    return Err(format!("keepalive failed: {e}"));
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match event {
            connection::Event::Server(line) => {
                println!("{line}");
                if let Some(chunk) = accumulator.push(line) {
                    let chunk_refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
                    let parsed = protocol::parse_server_chunk(&chunk_refs, session.my_username());
                    for action in session.on_event(parsed) {
                        apply_action(action, &mut conn, &search_pool, &event_tx)?;
                    }
                }
            }
            connection::Event::Stdin(line) => {
                conn.send_line(&line)
                    .map_err(|e| format!("stdin->socket failed: {e}"))?;
            }
            connection::Event::MoveReady { match_id, mv } => {
                for action in session.on_move_ready(&match_id, mv) {
                    apply_action(action, &mut conn, &search_pool, &event_tx)?;
                }
            }
            connection::Event::SearchFailed { match_id, error } => {
                eprintln!("[ggs] search panicked for match {match_id}: {error}");
                for action in session.on_search_aborted(&match_id) {
                    apply_action(action, &mut conn, &search_pool, &event_tx)?;
                }
            }
            connection::Event::SocketClosed => {
                eprintln!("[ggs] server closed the connection");
                break;
            }
            connection::Event::StdinClosed => {
                // Running without stdin is fine (daemon-ish). Keep the loop
                // alive and let it exit via SocketClosed.
            }
        }
    }

    Ok(())
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
            eprintln!("[ggs] {msg}");
        }
        session::SessionAction::StartSearch {
            match_id,
            board,
            search_limit,
            selectivity,
        } => {
            let search_pool = Arc::clone(search_pool);
            let tx = event_tx.clone();
            thread::spawn(move || {
                // Lease an independent engine so synchro children can search
                // concurrently instead of serialising on one `Search`.
                // `catch_unwind` prevents a panic in `Search::run` (malformed
                // weights, internal assert, OOM in eval, ...) from wedging
                // the main loop.
                let mut search = search_pool.acquire();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let options = build_search_run_options(search_limit, selectivity);
                    search.search_mut().run(&board, &options)
                }));
                match result {
                    Ok(r) => {
                        let _ = tx.send(connection::Event::MoveReady {
                            match_id,
                            mv: r.best_move,
                        });
                    }
                    Err(payload) => {
                        search.mark_broken();
                        let _ = tx.send(connection::Event::SearchFailed {
                            match_id,
                            error: panic_message(&payload),
                        });
                    }
                }
            });
        }
    }
    Ok(())
}

fn build_search_run_options(
    search_limit: session::SearchLimit,
    selectivity: Selectivity,
) -> SearchRunOptions {
    match search_limit {
        session::SearchLimit::Time(time_mode) => {
            SearchRunOptions::with_time(time_mode, selectivity)
        }
        session::SearchLimit::Level { level } => {
            SearchRunOptions::with_level(get_level(level.min(MAX_LEVEL)), selectivity)
        }
    }
}

/// Lock the idle-engine pool, recovering from poisoning.
fn lock_idle(idle: &Mutex<Vec<search::Search>>) -> std::sync::MutexGuard<'_, Vec<search::Search>> {
    idle.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Extract a best-effort message from a `catch_unwind` panic payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}
