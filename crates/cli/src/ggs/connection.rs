//! TCP connection and line-to-chunk accumulator for GGS mode.
//!
//! The [`Connection`] owns the write half of a TCP stream to the GGS server
//! and spawns a socket reader that funnels [`Event`]s into a channel consumed by the main loop in
//! `run_ggs`. The [`LineAccumulator`] buffers incoming server lines into
//! chunks delimited by a `READY` or `ALERT` line, which is the unit that
//! [`crate::ggs::protocol::parse_server_messages`] consumes.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use reversi_core::square::Square;
use reversi_core::types::Scoref;

/// Events streamed to the main event loop in `run_ggs`.
#[derive(Debug)]
pub enum Event {
    /// One line received from the GGS server (newline stripped).
    Server(String),
    /// One line read from stdin (newline stripped).
    Stdin(String),
    /// A terminal input or resize event from the interactive frontend.
    Terminal(crossterm::event::Event),
    /// Periodic redraw request from the interactive frontend, for running clocks.
    Tick,
    /// A diagnostic emitted by a worker.
    Log(String),
    /// A worker acquired the engine and started searching the position.
    SearchStarted {
        game_id: String,
        search_id: u64,
        ponder: bool,
    },
    /// Completed search statistics, retained with the displayed game.
    SearchReport {
        game_id: String,
        search_id: u64,
        message: String,
    },
    /// Reading interactive input failed.
    InputFailed(String),
    /// A search for `game_id` ended: its best move and disc-difference score
    /// (`None` means no legal move — a pass), or why it produced no move.
    SearchFinished {
        game_id: String,
        search_id: u64,
        /// The instant the turn was queued, including time waiting for the engine.
        queued_at: Instant,
        result: Result<Option<(Square, Scoref)>, String>,
    },
    /// A ponder worker exited (aborted, completed, or never started); its
    /// engine and thread pool are idle again.
    PonderFinished { game_id: String, search_id: u64 },
    /// Server closed the TCP connection.
    SocketClosed,
    /// Reading from the server connection failed.
    SocketReadFailed(String),
}

/// Owns the write-side of the TCP connection and the event channel receiver.
pub struct Connection {
    writer: TcpStream,
    events: Receiver<Event>,
    last_send: Instant,
    /// Transcript of every line sent and received after [`Connection::log_to`].
    log: Option<File>,
}

impl Connection {
    /// Connects to `host:port`, spawns a socket reader,
    /// and returns the `Connection` along with a cloned
    /// `Sender<Event>` that downstream code (e.g. the search worker thread)
    /// can use to feed search results back into the loop.
    pub fn connect(host: &str, port: u16) -> std::io::Result<(Self, Sender<Event>)> {
        let writer = TcpStream::connect((host, port))?;
        let reader_stream = writer.try_clone()?;

        let (tx, rx) = mpsc::channel::<Event>();

        // Socket reader thread.
        let socket_tx = tx.clone();
        thread::spawn(move || {
            // GGS relays tells byte-for-byte, so lines are not guaranteed UTF-8.
            let reader = BufReader::new(reader_stream);
            for line in reader.split(b'\n') {
                match line {
                    Ok(bytes) => {
                        let mut line = String::from_utf8_lossy(&bytes).into_owned();
                        if line.ends_with('\r') {
                            line.pop();
                        }
                        if socket_tx.send(Event::Server(line)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = socket_tx.send(Event::SocketReadFailed(error.to_string()));
                        return;
                    }
                }
            }
            let _ = socket_tx.send(Event::SocketClosed);
        });

        Ok((
            Connection {
                writer,
                events: rx,
                last_send: Instant::now(),
                log: None,
            },
            tx,
        ))
    }

    /// Sends one line to the GGS server, appending `\n`. Returns IO errors verbatim.
    ///
    /// An IO error returned here should be treated as terminal: the writer may
    /// be in a half-closed state and subsequent writes will likely fail too.
    pub fn send_line(&mut self, line: &str) -> std::io::Result<()> {
        // The line and its `\n` must go out in a single write_all: split into
        // two writes, the kernel may deliver them in separate TCP segments,
        // and the reference GGS build then rejects the login prompt with
        // "ERR Login (2-8 chars) ..." and leaves every later line unconsumed.
        self.writer.write_all(format!("{line}\n").as_bytes())?;
        self.last_send = Instant::now();
        self.log_line('>', line);
        Ok(())
    }

    /// Appends every later sent (`>`) and received (`<`) line to `file`,
    /// each prefixed with its Unix time in seconds.
    pub fn log_to(&mut self, file: File) {
        self.log = Some(file);
    }

    fn log_line(&mut self, direction: char, line: &str) {
        if let Some(log) = &mut self.log {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default();
            // A failing log must not abort live games.
            let _ = writeln!(
                log,
                "{}.{:03} {direction} {line}",
                now.as_secs(),
                now.subsec_millis()
            );
        }
    }

    /// Waits for the next event until `interval` has passed since the last
    /// send, then sends a keepalive instead and returns `None`.
    pub fn recv_or_keepalive(&mut self, interval: Duration) -> Result<Option<Event>, String> {
        let remaining = interval.saturating_sub(self.last_send.elapsed());
        if !remaining.is_zero() {
            match self.events.recv_timeout(remaining) {
                Ok(event) => {
                    if let Event::Server(line) = &event {
                        self.log_line('<', line);
                    }
                    return Ok(Some(event));
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("GGS event channel disconnected".to_string());
                }
            }
        }
        self.send_line(":")
            .map_err(|e| format!("keepalive failed: {e}"))?;
        Ok(None)
    }
}

/// Whether a server line terminates the current response block.
fn is_message_boundary(line: &str) -> bool {
    matches!(line.trim(), "READY" | "ALERT")
}

/// Accumulates incoming server lines into chunks terminated by `READY` or `ALERT`.
///
/// Lines passed to `push` are expected to have already been stripped of any
/// trailing `\r` (GGS uses `\r\n` terminators); that is the socket reader's
/// job, and the accumulator does not re-trim.
#[derive(Debug, Default)]
pub struct LineAccumulator {
    buffer: Vec<String>,
}

impl LineAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Buffers `line` and returns `None`, or — when `line` is a boundary —
    /// drains the buffer and returns it. Blank lines inside a chunk are
    /// preserved (some GGS events have them), but an all-blank chunk yields
    /// `None` (nothing to parse).
    pub fn push(&mut self, line: String) -> Option<Vec<String>> {
        if is_message_boundary(&line) {
            if self.buffer.iter().all(|l| l.trim().is_empty()) {
                self.buffer.clear();
                return None;
            }
            return Some(std::mem::take(&mut self.buffer));
        }
        self.buffer.push(line);
        None
    }

    /// Whether a server message has begun but its terminating boundary has not
    /// arrived yet.
    pub fn has_pending_lines(&self) -> bool {
        !self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn multiple_chunks_are_independent() {
        let mut acc = LineAccumulator::new();
        assert_eq!(acc.push("first line of chunk one".into()), None);
        let chunk1 = acc.push("READY".into()).unwrap();
        assert_eq!(chunk1, vec!["first line of chunk one"]);

        assert_eq!(acc.push("first line of chunk two".into()), None);
        assert_eq!(acc.push("second line of chunk two".into()), None);
        let chunk2 = acc.push("READY".into()).unwrap();
        assert_eq!(
            chunk2,
            vec!["first line of chunk two", "second line of chunk two"]
        );
    }

    #[test]
    fn pending_lines_clear_only_at_ready_boundary() {
        let mut acc = LineAccumulator::new();
        assert!(!acc.has_pending_lines());
        assert_eq!(acc.push("/os: - match .5 ...".into()), None);
        assert!(acc.has_pending_lines());
        let _ = acc.push("READY".into());
        assert!(!acc.has_pending_lines());
    }

    #[test]
    fn leading_blank_lines_inside_chunk_are_preserved() {
        let mut acc = LineAccumulator::new();
        assert_eq!(acc.push(String::new()), None);
        assert_eq!(acc.push("/os: update .5 8b K?".into()), None);
        let chunk = acc.push("READY".into()).unwrap();
        assert_eq!(chunk.first().map(String::as_str), Some(""));
        assert_eq!(
            chunk.last().map(String::as_str),
            Some("/os: update .5 8b K?")
        );
    }

    #[test]
    fn all_blank_chunk_before_ready_returns_none() {
        let mut acc = LineAccumulator::new();
        assert_eq!(acc.push(String::new()), None);
        assert_eq!(acc.push("  ".into()), None);
        assert_eq!(acc.push("READY".into()), None);
        // buffer should also be empty afterwards
        assert_eq!(acc.push("READY".into()), None);
    }

    #[test]
    fn ready_with_trailing_whitespace_is_still_ready() {
        let mut acc = LineAccumulator::new();
        assert_eq!(acc.push("hello".into()), None);
        let chunk = acc.push("READY   ".into()).unwrap();
        assert_eq!(chunk, vec!["hello"]);
    }

    #[test]
    fn alert_is_also_a_message_boundary() {
        let mut acc = LineAccumulator::new();
        assert_eq!(acc.push("/os: ERR test".into()), None);
        assert_eq!(acc.push("ALERT".into()).unwrap(), vec!["/os: ERR test"]);
        assert!(!acc.has_pending_lines());
    }

    #[test]
    fn invalid_utf8_line_does_not_end_the_reader() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (mut conn, _events) = Connection::connect("127.0.0.1", port).unwrap();
        let (mut stream, _) = listener.accept().unwrap();
        stream.write_all(b"caf\xe9\r\nREADY\r\n").unwrap();

        let mut next = || match conn.recv_or_keepalive(Duration::from_secs(5)).unwrap() {
            Some(Event::Server(line)) => line,
            _ => panic!("expected a server line"),
        };
        assert_eq!(next(), "caf\u{fffd}");
        assert_eq!(next(), "READY");
    }

    #[test]
    fn transcript_starts_at_log_to() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (mut conn, _events) = Connection::connect("127.0.0.1", port).unwrap();
        let (mut stream, _) = listener.accept().unwrap();
        let path = std::env::temp_dir().join(format!("ggs-log-{port}.txt"));

        conn.send_line("secret").unwrap();
        conn.log_to(File::create(&path).unwrap());
        conn.send_line("tell /os open 1").unwrap();
        stream.write_all(b"READY\r\n").unwrap();
        assert!(matches!(
            conn.recv_or_keepalive(Duration::from_secs(5)).unwrap(),
            Some(Event::Server(_))
        ));

        let log = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let lines: Vec<&str> = log
            .lines()
            .map(|line| line.split_once(' ').unwrap().1)
            .collect();
        assert_eq!(lines, ["> tell /os open 1", "< READY"]);
    }

    #[test]
    fn outbound_idle_sends_ggs_refresh() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (mut conn, _events) = Connection::connect("127.0.0.1", port).unwrap();
        let (stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();

        assert!(conn.recv_or_keepalive(Duration::ZERO).unwrap().is_none());
        let mut line = String::new();
        BufReader::new(stream).read_line(&mut line).unwrap();
        assert_eq!(line, ":\n");
    }
}
