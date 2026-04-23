//! TCP connection, stdin reader, and line-to-chunk accumulator for GGS mode.
//!
//! The [`Connection`] owns the write half of a TCP stream to the GGS server
//! and spawns two reader threads — one for the socket, one for stdin — that
//! funnel [`Event`]s into an `mpsc` channel consumed by the main loop in
//! `run_ggs`. The [`LineAccumulator`] buffers incoming server lines into
//! chunks delimited by a literal `READY` line, which is the unit that
//! [`crate::ggs::protocol::parse_server_chunk`] consumes.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use reversi_core::square::Square;

/// Events streamed to the main event loop in `run_ggs`.
#[derive(Debug)]
pub enum Event {
    /// One line received from the GGS server (newline stripped).
    Server(String),
    /// One line read from stdin (newline stripped).
    Stdin(String),
    /// Search completed; deliver best move for the given match.
    /// `mv = None` means no legal move — a pass.
    MoveReady {
        match_id: String,
        mv: Option<Square>,
    },
    /// Search terminated without a usable move (for example due to panic).
    SearchFailed { match_id: String, error: String },
    /// Server closed the TCP connection or a read error occurred.
    SocketClosed,
    /// Stdin hit EOF or a read error occurred. Loop should continue running
    /// (daemon mode); the main loop decides what to do.
    StdinClosed,
}

/// Owns the write-side of the TCP connection and the event channel receiver.
pub struct Connection {
    writer: TcpStream,
    events: Receiver<Event>,
    /// Wall-clock timestamp of the most recent `send_line`. Drives the
    /// TX-based keepalive in `run_ggs` so the server's idle timer cannot
    /// reap us while we are silent but still receiving (e.g. observer
    /// kibitz while we are deep in a search).
    last_send: Instant,
}

impl Connection {
    /// Connects to `host:port`, spawns background threads for socket reads
    /// and stdin reads, and returns the `Connection` along with a cloned
    /// `Sender<Event>` that downstream code (e.g. the search worker thread)
    /// can use to feed search results back into the loop.
    pub fn connect(host: &str, port: u16) -> std::io::Result<(Self, Sender<Event>)> {
        let writer = TcpStream::connect((host, port))?;
        let reader_stream = writer.try_clone()?;

        let (tx, rx) = mpsc::channel::<Event>();

        // Socket reader thread.
        let socket_tx = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(reader_stream);
            for line in reader.lines() {
                match line {
                    Ok(mut line) => {
                        // GGS terminates lines with `\r\n`; `BufRead::lines()` strips
                        // the `\n` but leaves the `\r`. Drop it here so downstream
                        // parsers never see carriage returns.
                        if line.ends_with('\r') {
                            line.pop();
                        }
                        if socket_tx.send(Event::Server(line)).is_err() {
                            return;
                        }
                    }
                    Err(_) => {
                        let _ = socket_tx.send(Event::SocketClosed);
                        return;
                    }
                }
            }
            let _ = socket_tx.send(Event::SocketClosed);
        });

        // Stdin reader thread.
        let stdin_tx = tx.clone();
        thread::spawn(move || {
            let stdin = std::io::stdin();
            let handle = stdin.lock();
            for line in handle.lines() {
                match line {
                    Ok(line) => {
                        if stdin_tx.send(Event::Stdin(line)).is_err() {
                            return;
                        }
                    }
                    Err(_) => {
                        let _ = stdin_tx.send(Event::StdinClosed);
                        return;
                    }
                }
            }
            let _ = stdin_tx.send(Event::StdinClosed);
        });

        Ok((
            Connection {
                writer,
                events: rx,
                last_send: Instant::now(),
            },
            tx,
        ))
    }

    /// Sends one line to the GGS server, appending `\n`. Returns IO errors verbatim.
    ///
    /// The line and its terminating `\n` are written in a single `write_all`
    /// call so a strict line-oriented peer sees the line atomically. Splitting
    /// them into two writes lets the kernel deliver the line and its `\n` in
    /// separate TCP segments; the local reference GGS build at
    /// `localhost:5000` rejects the login prompt in that case with a generic
    /// "ERR Login (2-8 chars) ..." and leaves every later line unconsumed.
    ///
    /// An IO error returned here should be treated as terminal: the writer may
    /// be in a half-closed state and subsequent writes will likely fail too.
    pub fn send_line(&mut self, line: &str) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(line.len() + 1);
        bytes.extend_from_slice(line.as_bytes());
        bytes.push(b'\n');
        self.writer.write_all(&bytes)?;
        self.writer.flush()?;
        self.last_send = Instant::now();
        Ok(())
    }

    /// Receives the next event or returns [`RecvTimeoutError::Timeout`] after
    /// `duration`. The main loop uses this to drive periodic keepalives while
    /// the session is otherwise idle; [`RecvTimeoutError::Disconnected`]
    /// means every sender has dropped and is terminal.
    pub fn recv_timeout(&self, duration: Duration) -> Result<Event, RecvTimeoutError> {
        self.events.recv_timeout(duration)
    }

    /// Duration remaining before `since_last_send` reaches `interval`.
    /// Saturates to zero when we are already past that threshold so the
    /// caller can pass the result straight to `recv_timeout`.
    ///
    /// Used to drive TX-based keepalives (Edax-style, `ggs.c:1092`).
    pub fn time_until_keepalive(&self, interval: Duration) -> Duration {
        interval.saturating_sub(self.last_send.elapsed())
    }
}

/// Accumulates incoming server lines into chunks terminated by a literal `READY` line.
/// `push(line)` returns `Some(chunk)` when `line.trim() == "READY"` and the chunk is
/// non-empty; otherwise it buffers the line and returns `None`. Blank leading lines
/// are tolerated inside the buffer (some GGS events have them), but an all-blank
/// chunk followed by `READY` yields `None` (nothing to parse).
///
/// Lines passed to `push` are expected to have already been stripped of any trailing
/// `\r` (GGS uses `\r\n` terminators). That stripping is the socket reader's job, not
/// the accumulator's — `LineAccumulator` does not re-validate or re-trim carriage
/// returns itself.
#[derive(Debug, Default)]
pub struct LineAccumulator {
    buffer: Vec<String>,
}

impl LineAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `line` to the buffer (unless it is `READY`, in which case the buffer is
    /// drained and returned). Ownership of the line is taken — the caller typically
    /// clones or echoes it before calling `push` since the buffer consumes it.
    pub fn push(&mut self, line: String) -> Option<Vec<String>> {
        if line.trim() == "READY" {
            if self.buffer.iter().all(|l| l.trim().is_empty()) {
                self.buffer.clear();
                return None;
            }
            return Some(std::mem::take(&mut self.buffer));
        }
        self.buffer.push(line);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_ready_returns_buffered_lines() {
        let mut acc = LineAccumulator::new();
        assert_eq!(
            acc.push("/os: + match .5 1720 alice 1720 bob 8b U".into()),
            None
        );
        let chunk = acc.push("READY".into()).unwrap();
        assert_eq!(chunk, vec!["/os: + match .5 1720 alice 1720 bob 8b U"]);
    }

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
    fn ready_with_empty_buffer_returns_none() {
        let mut acc = LineAccumulator::new();
        assert_eq!(acc.push("READY".into()), None);
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
    fn realistic_update_chunk_round_trips_through_accumulator() {
        let lines = vec![
            "/os: update .5 8b K?".to_string(),
            "|  1720 alice vs 1720 bob".to_string(),
            "|  move 5: F5".to_string(),
            "|  time  5:00   5:00".to_string(),
        ];
        let mut acc = LineAccumulator::new();
        for line in &lines {
            assert_eq!(acc.push(line.clone()), None);
        }
        let chunk = acc.push("READY".into()).unwrap();
        assert_eq!(chunk, lines);
    }
}
