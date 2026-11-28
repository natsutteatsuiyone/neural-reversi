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
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use reversi_core::engine::{Engine, Finished, GameId, SearchKind};
use reversi_core::level::{Level, MAX_LEVEL};
use reversi_core::search::SearchRunOptions;
use reversi_core::search::time_control::TimeControlMode;

use crate::config::EngineConfig;

const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(50);

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
    let search_options = config.search_options();
    let max_game_slots = max_games * 2;
    let (mut conn, event_tx) = connection::Connection::connect(host, port)
        .map_err(|e| format!("failed to connect to {host}:{port}: {e}"))?;
    let mut games = Games::new(Engine::new(&search_options), max_game_slots);
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
    // Set once we log ourselves out, after which the server closing the socket is
    // the requested logout rather than a lost connection.
    let mut quit_sent = false;
    for line in &startup_lines[mandatory_line_count..] {
        quit_sent |= submit_command(&mut conn, &mut frontend, line)?;
    }

    let mut session = session::Session::new(user.to_string(), config.level, max_games);
    if ponder {
        session.enable_ponder();
    }
    let mut accumulator = connection::LineAccumulator::new();
    // `(sender, tournament id)` of each join request whose `/td finger` reply
    // is still to come.
    let mut pending_joins: Vec<(String, String)> = Vec::new();
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
                if join_tournaments && let Some((from, id)) = tournament_join_request(&line) {
                    pending_joins.push((from.to_string(), id.to_string()));
                    submit_command(&mut conn, &mut frontend, &format!("tell /td finger {id}"))?;
                }
                if let Some(chunk) = accumulator.push(line) {
                    let chunk_refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
                    for verdict in tournament_join_verdicts(&chunk_refs, &mut pending_joins) {
                        match verdict {
                            Ok(join) => {
                                submit_command(&mut conn, &mut frontend, &join)?;
                            }
                            Err(refusal) => frontend.log(&refusal),
                        }
                    }
                    for parsed in protocol::parse_server_messages(&chunk_refs, user) {
                        for action in session.on_event(parsed) {
                            apply_action(action, &mut conn, &mut games, user, &event_tx)?;
                        }
                    }
                }
            }
            connection::Event::Stdin(line) => {
                quit_sent |= submit_command(&mut conn, &mut frontend, &line)?;
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
                        quit_sent |= submit_command(&mut conn, &mut frontend, &line)?;
                    }
                }
            },
            connection::Event::Log(message) => frontend.log(&message),
            connection::Event::Tick => {}
            connection::Event::InputFailed(error) => {
                return Err(format!("input read failed: {error}").into());
            }
            connection::Event::SocketClosed if quit_sent => break,
            connection::Event::SocketClosed => {
                return Err("server closed the connection".to_string().into());
            }
            connection::Event::SocketReadFailed(error) => {
                return Err(format!("socket read failed: {error}").into());
            }
        }
        if !accumulator.has_pending_lines() {
            for (game_id, search_id, queued_at, result) in finished_searches.drain(..) {
                let elapsed = queued_at.elapsed();
                for action in session.on_search_finished(&game_id, search_id, result, elapsed) {
                    apply_action(action, &mut conn, &mut games, user, &event_tx)?;
                }
            }
        }
        if redraw {
            frontend.draw(session.games())?;
        }
    }
    Ok(())
}

/// Sends a user command and returns whether it logs us out: `quit` with no
/// arguments or its standard GGS alias `q` (`quit <user>` disconnects another user).
fn submit_command(
    conn: &mut connection::Connection,
    frontend: &mut frontend::Frontend,
    line: &str,
) -> Result<bool, String> {
    frontend.command(line);
    if let Some(clock) = unsupported_ask_clock(line) {
        frontend.log(&format!(
            "not sending ask with unsupported or unknown clock {clock}"
        ));
        Ok(false)
    } else {
        conn.send_line(line)
            .map_err(|e| format!("stdin->socket failed: {e}"))?;
        Ok(matches!(line.trim(), "quit" | "q"))
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

/// The `(sender, tournament id)` of a user's tell asking us to join a `/td`
/// tournament, e.g. `HCyrano: tell /td join 3`. Only the tournament's director
/// can enter another player this way, since `/td` lets nobody but a superuser
/// join on someone else's behalf. `/os` in place of `/td` (as some tournament
/// announcements write it) and a `.` before the id are accepted; `/td` ids are
/// plain numbers.
fn tournament_join_request(line: &str) -> Option<(&str, &str)> {
    let mut fields = line.trim_start_matches('\x07').split_whitespace();
    let (Some(from), Some("tell" | "t"), Some("/td" | "/os"), Some("join"), Some(id), None) = (
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
        fields.next(),
    ) else {
        return None;
    };
    let from = from.strip_suffix(':')?;
    let id = id.trim_prefix('.');
    (!from.is_empty()
        && !from.starts_with('/')
        && !id.is_empty()
        && id.bytes().all(|b| b.is_ascii_digit()))
    .then_some((from, id))
}

/// Settles the pending join requests that the `/td` replies in `chunk` answer:
/// `Ok` is the join command to send, `Err` the reason for not joining.
///
/// A `finger` reply settles every request for its tournament, and we join only
/// an `/os` tournament directed by one of the requesters whose game type and
/// clock the engine plays: `/td` starts tournament matches itself, past the
/// `dform` that declines unsupported asks. A `finger` error names no
/// tournament, so it settles the oldest request.
fn tournament_join_verdicts(
    chunk: &[&str],
    pending: &mut Vec<(String, String)>,
) -> Vec<Result<String, String>> {
    let mut verdicts = Vec::new();
    // A `|` line continues the message above it, so text a user puts in a
    // tell can never start a message of its own.
    for message in chunk.chunk_by(|_, next| next.starts_with('|')) {
        if pending.is_empty() {
            break;
        }
        let Some(reply) = message[0].trim_start_matches('\x07').strip_prefix("/td: ") else {
            continue;
        };
        if let Some(id) = reply.strip_prefix("finger: ") {
            let id = id.trim();
            let requesters: Vec<String> = pending
                .extract_if(.., |(_, wanted)| wanted == id)
                .map(|(from, _)| from)
                .collect();
            if requesters.is_empty() {
                continue;
            }
            let field = |name: &str| {
                message[1..]
                    .iter()
                    .find_map(|line| {
                        let (key, value) = line.strip_prefix('|')?.split_once(':')?;
                        (key.trim() == name).then_some(value.trim())
                    })
                    .unwrap_or("")
            };
            let (director, service, game_type, clock) = (
                field("director"),
                field("service"),
                field("type"),
                field("clock"),
            );
            let refusal = if !requesters.iter().any(|from| from == director) {
                Some(format!(
                    "asked by {}, but its director is {director}",
                    requesters.join(", ")
                ))
            } else if service != "/os" {
                Some(format!("its service is {service}, not /os"))
            } else if !session::is_supported_ask_game(game_type) {
                Some(format!("unsupported game type {game_type}"))
            } else if !session::is_supported_ask_clock(clock) {
                Some(format!("unsupported clock {clock}"))
            } else {
                None
            };
            verdicts.push(match refusal {
                None => Ok(format!("tell /td join {id}")),
                Some(reason) => Err(format!("not joining tournament {id}: {reason}")),
            });
        } else if reply
            .strip_prefix("ERR ")
            .and_then(|error| error.split_once(": "))
            .is_some_and(|(_, error)| error.starts_with("finger:"))
        {
            let (from, id) = pending.remove(0);
            verdicts.push(Err(format!(
                "not joining tournament {id} for {from}: /td answered {reply}"
            )));
        }
    }
    verdicts
}

fn unsupported_ask_clock(line: &str) -> Option<&str> {
    let mut fields = line.split_whitespace();
    if !matches!(
        (fields.next(), fields.next(), fields.next()),
        (Some("tell" | "t"), Some("/os"), Some("ask"))
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

fn apply_action(
    action: session::SessionAction,
    conn: &mut connection::Connection,
    games: &mut Games,
    login: &str,
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
                format!("tell {} {report}", log_channel(login)),
                "var _send $_nrsend".to_string(),
            ] {
                conn.send_line(&line)
                    .map_err(|e| format!("send failed: {e}"))?;
            }
        }
        // `id` is a match id, or a synchro game id from `/os: end`. Results of
        // the stopped searches still arrive and are discarded as stale.
        session::SessionAction::EndMatch { id } => {
            games.close_match(&id);
        }
        session::SessionAction::StartSearch {
            game_id,
            search_id,
            board,
            color,
            search_limit,
        } => {
            let queued_at = Instant::now();
            // Each game id keeps one search state for the full game so its
            // endgame-depth tracking cannot leak to or be lost to another
            // concurrently active game.
            let Some(game) = games.open(&game_id) else {
                let _ = event_tx.send(connection::Event::SearchFinished {
                    result: Err(capacity_error(&game_id)),
                    game_id,
                    search_id,
                    queued_at,
                });
                return Ok(());
            };
            let job_id = game_id.clone();
            let tx = event_tx.clone();
            let done_id = game_id.clone();
            let log_tx = event_tx.clone();
            let n_threads = games.engine.n_threads();
            let login = login.to_string();
            let spawned = games.engine.start(
                game,
                SearchKind::Move,
                board,
                queued_at,
                move |start| {
                    if start.split_table {
                        let _ = log_tx.send(connection::Event::Log(format!(
                            "{job_id} left the synchro line; searching on a TT copy"
                        )));
                    }
                    // Our clock kept running while another game held the pool.
                    let search_limit = deduct_wait(search_limit, start.waited);
                    let _ = log_tx.send(connection::Event::SearchStarted {
                        game_id: job_id,
                        search_id,
                    });
                    build_search_run_options(search_limit)
                },
                move |found| {
                    let mv = found.map(|found| {
                        let Finished {
                            result,
                            waited,
                            elapsed,
                            timed_out,
                        } = found;
                        let mv = result.best_move().zip(result.score());
                        mv.map(|(sq, score)| session::FoundMove {
                            report: report::SearchReport {
                                sq,
                                score,
                                depth: result.depth(),
                                selectivity: result.selectivity(),
                                nodes: result.n_nodes(),
                                elapsed,
                                waited,
                            },
                            broadcast: report::channel_report(
                                &login, &done_id, &board, color, &result, elapsed, waited,
                                n_threads, timed_out,
                            ),
                        })
                    });
                    let _ = tx.send(connection::Event::SearchFinished {
                        game_id: done_id,
                        search_id,
                        queued_at,
                        result: mv.ok_or_else(|| "search cancelled before it started".to_string()),
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
            let Some(game) = games.open(&game_id) else {
                let _ = event_tx.send(connection::Event::Log(format!(
                    "ponder skipped for {game_id}: {}",
                    capacity_error(&game_id)
                )));
                let _ = event_tx.send(connection::Event::PonderFinished { game_id, search_id });
                return Ok(());
            };
            let _ = event_tx.send(connection::Event::Log(format!("pondering {game_id}")));
            let tx = event_tx.clone();
            let done_id = game_id.clone();
            let started_tx = event_tx.clone();
            let started_id = game_id.clone();
            let spawned = games.engine.start(
                game,
                SearchKind::Analysis,
                board,
                Instant::now(),
                move |_| {
                    let _ = started_tx.send(connection::Event::SearchStarted {
                        game_id: started_id,
                        search_id,
                    });
                    SearchRunOptions::with_time(TimeControlMode::Infinite)
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
            games.stop(&game_id, SearchKind::Analysis);
        }
        session::SessionAction::StopSearch { game_id } => {
            games.stop(&game_id, SearchKind::Move);
        }
    }
    Ok(())
}

/// Our GGS games on the engine, by game id.
struct Games {
    engine: Engine,
    /// Most games open at a time; a synchro match takes two.
    capacity: usize,
    open: HashMap<String, GameId>,
}

impl Games {
    fn new(engine: Engine, capacity: usize) -> Self {
        Self {
            engine,
            capacity,
            open: HashMap::new(),
        }
    }

    /// Returns the engine game of `game_id`, opened on first use, or [`None`]
    /// when every slot is taken. A synchro game shares the transposition
    /// table of its sibling until their lines diverge.
    fn open(&mut self, game_id: &str) -> Option<GameId> {
        if let Some(&game) = self.open.get(game_id) {
            return Some(game);
        }
        if self.open.len() == self.capacity {
            return None;
        }
        let game = match session::synchro_sibling(game_id) {
            Some(sibling) => self.engine.open_sharing(self.open.get(&sibling).copied()),
            None => self.engine.open(),
        };
        self.open.insert(game_id.to_string(), game);
        Some(game)
    }

    /// Stops the search of `kind` that `game_id` started last.
    fn stop(&self, game_id: &str, kind: SearchKind) {
        if let Some(&game) = self.open.get(game_id) {
            self.engine.stop(game, kind);
        }
    }

    /// Closes the games of match `id` (or the synchro game `id`) and stops
    /// their searches.
    fn close_match(&mut self, id: &str) {
        self.open.retain(|game_id, game| {
            let ended = session::is_game_of(id, game_id);
            if ended {
                self.engine.close(*game);
            }
            !ended
        });
    }
}

fn capacity_error(game_id: &str) -> String {
    format!("search engine capacity exhausted while starting {game_id}")
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
            SearchRunOptions::with_level(Level::preset(level.min(MAX_LEVEL)))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn a_full_engine_takes_no_new_game_until_a_match_ends() {
        let mut games = Games::new(crate::config::test_engine(1, 1), 3);
        let [first, _, plain] = [".5.0", ".5.1", ".6"].map(|id| games.open(id).unwrap());
        assert_eq!(games.open(".5.0"), Some(first));
        assert_eq!(games.open(".7"), None);

        games.close_match(".5");
        assert!(games.open(".7").is_some());
        assert_eq!(games.open(".6"), Some(plain));
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
    fn only_tournament_join_tells_are_requests() {
        for (line, request) in [
            ("HCyrano: tell /td join 3", ("HCyrano", "3")),
            ("\x07bob: t /td join .12", ("bob", "12")),
            ("HCyrano: tell /os join .3", ("HCyrano", "3")),
        ] {
            assert_eq!(tournament_join_request(line), Some(request), "{line}");
        }
        for line in [
            "HCyrano: tell /td withdraw 3",
            "HCyrano: tell /os ask s8r14 3:00//0:30 bob",
            "HCyrano: tell /td join 3 now",
            "HCyrano: tell /td join 3x",
            "HCyrano: tell /td join .",
            "/os: tell /td join 3",
            ": tell /td join 3",
            "tell /td join 3",
        ] {
            assert_eq!(tournament_join_request(line), None, "{line}");
        }
    }

    fn finger_reply(id: &str, director: &str, service: &str, game_type: &str) -> Vec<String> {
        [
            format!("/td: finger: {id}"),
            "|t nra finger: ".to_string(),
            format!("|id        : {id}"),
            "|joining   : open".to_string(),
            format!("|director  : {director}"),
            format!("|service   : {service}"),
            "|rounds    : 0/1".to_string(),
            "|style     : rrobin".to_string(),
            "|clock     : 1:00//0:30".to_string(),
            format!("|type      : {game_type}"),
            "|breaks    :  1:00".to_string(),
            "|players   : 1 mallory ".to_string(),
            "|round 1 begins in  4:58".to_string(),
        ]
        .into()
    }

    fn verdicts(
        chunk: &[String],
        pending: &mut Vec<(String, String)>,
    ) -> Vec<Result<String, String>> {
        let chunk: Vec<&str> = chunk.iter().map(String::as_str).collect();
        tournament_join_verdicts(&chunk, pending)
    }

    fn pending(requests: &[(&str, &str)]) -> Vec<(String, String)> {
        requests
            .iter()
            .map(|(from, id)| (from.to_string(), id.to_string()))
            .collect()
    }

    #[test]
    fn a_tournament_is_joined_only_for_its_director() {
        let mut requests = pending(&[("mallory", "3"), ("HCyrano", "4"), ("HCyrano", "3")]);
        assert_eq!(
            verdicts(&finger_reply("3", "HCyrano", "/os", "s8r20"), &mut requests),
            [Ok("tell /td join 3".to_string())]
        );
        assert_eq!(requests, pending(&[("HCyrano", "4")]));

        // The reply to mallory's own finger request finds nothing left to settle.
        assert!(verdicts(&finger_reply("3", "HCyrano", "/os", "s8r20"), &mut requests).is_empty());

        let mut requests = pending(&[("mallory", "3")]);
        assert_eq!(
            verdicts(&finger_reply("3", "HCyrano", "/os", "s8r20"), &mut requests),
            [Err(
                "not joining tournament 3: asked by mallory, but its director is HCyrano"
                    .to_string()
            )]
        );
        assert!(requests.is_empty());
    }

    #[test]
    fn a_tournament_the_engine_cannot_play_is_not_joined() {
        for (service, game_type, reason) in [
            ("/os", "8a", "unsupported game type 8a"),
            ("/os", "s10r20", "unsupported game type s10r20"),
            ("/ams", "s8r20", "its service is /ams, not /os"),
        ] {
            let mut requests = pending(&[("HCyrano", "3")]);
            assert_eq!(
                verdicts(
                    &finger_reply("3", "HCyrano", service, game_type),
                    &mut requests
                ),
                [Err(format!("not joining tournament 3: {reason}"))]
            );
        }
    }

    #[test]
    fn a_finger_reply_inside_a_tell_settles_nothing() {
        let mut requests = pending(&[("mallory", "3")]);
        let mut chunk = vec!["mallory: hello".to_string()];
        chunk.extend(
            finger_reply("3", "mallory", "/os", "s8r20")
                .into_iter()
                .map(|line| format!("|{}", line.trim_start_matches('|'))),
        );
        assert!(verdicts(&chunk, &mut requests).is_empty());
        assert_eq!(requests, pending(&[("mallory", "3")]));
    }

    #[test]
    fn a_finger_error_settles_the_oldest_request() {
        let mut requests = pending(&[("HCyrano", "9"), ("HCyrano", "3")]);
        let chunk = [
            "/os: + match .1 1720 alice 1720 bob 8 R".to_string(),
            "/td: ERR 16: finger: tournament id does not exist".to_string(),
            "/td: ERR 12: join: tournament is closed".to_string(),
        ];
        assert_eq!(
            verdicts(&chunk, &mut requests),
            [Err(
                "not joining tournament 9 for HCyrano: /td answered ERR 16: finger: tournament id does not exist"
                    .to_string()
            )]
        );
        assert_eq!(requests, pending(&[("HCyrano", "3")]));
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
            unsupported_ask_clock("t /os ask 8 01:00,N0//02:00 bob"),
            Some("01:00,N0//02:00")
        );
        assert_eq!(
            unsupported_ask_clock("tell /os ask .42 ignored"),
            Some(".42")
        );
        assert_eq!(unsupported_ask_clock("tell /os ask 8"), None);
    }
}
