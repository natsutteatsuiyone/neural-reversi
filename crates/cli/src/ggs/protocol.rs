//! GGS `/os` server-event parser.
//!
//! Parses one complete chunk (the block of lines delimited by `READY`
//! markers on the receive stream) into a structured [`OsEvent`].

use reversi_core::board::Board;
use reversi_core::disc::Disc;

/// Clock information parsed from our GGS clock line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerClock {
    /// Remaining active-budget time for us in ms.
    pub main_ms: u64,
    /// Our increment in ms; 0 when absent or `0:00`.
    pub increment_ms: u64,
    /// Our extension / byoyomi period in ms; 0 when absent or `0:00`.
    pub byoyomi_ms: u64,
}

/// A structured event extracted from a single GGS receive chunk.
#[derive(Debug, Clone)]
pub enum OsEvent {
    /// `/os: + match <id> <r1> <n1> <r2> <n2> <variant> <type>` chunk.
    ///
    /// No color here: the `+ match` name order is request order, and GGS
    /// Othello picks colors from the variant qualifier (`8w` inverts the
    /// usual first-is-Black heuristic). See `MatchState::my_color` instead.
    MatchCreated {
        id: String,
        /// `None` when we are not a player in this match (spectator / observer).
        opponent: Option<String>,
        /// Raw variant token (`8b`, `s8r20`, ...) — `None` when the header
        /// is truncated. Session logs this for diagnostics and uses the
        /// leading `s` to detect synchro matches, whose two child games are
        /// reported under `.<id>.0` / `.<id>.1`.
        variant: Option<String>,
    },
    /// `/os: join <id> ...` OR `/os: update <id> ...` chunk, both carrying a full board.
    MatchState {
        id: String,
        board: Board,
        side_to_move: Disc,
        /// Actual color marker for `my_username` in this game, parsed from
        /// the clock line. This is authoritative because `+ match` lists
        /// request players, not necessarily black then white.
        my_color: Option<Disc>,
        /// `None` when our clock line is absent or unparseable.
        my_clock: Option<PlayerClock>,
    },
    /// `/os: - match <id> ...` or `/os: end <id> ...` chunk.
    MatchEnd {
        id: String,
        /// Match result text as the server prints it (`-64.00`, `aborted`, `name left`, ...).
        result: String,
        /// Signed score as the server prints it (from first-player perspective), if numeric.
        score: Option<f64>,
    },
    /// `/os: <id> A <rating> <from>: <message>` (match-attached kibitz/tell).
    Kibitz {
        id: String,
        from: String,
        message: String,
    },
    /// `/os: +   .<id> <r0> <n0> <clock> <type> <flag> <r1> <n1> [<my_clock>] [<stored_id>]`
    /// — a match-request ("ask") offer from `from` (the challenger, `req.p1`
    /// on the server) to `to` (the challengee, `req.p2`). See
    /// `EXE_Service::ask` / `EXE_Service::accept` in the server and the
    /// matching branch in `BoardGameServiceClient.java` for format details.
    AskOffer {
        id: String,
        /// Challenger username.
        from: String,
        /// Challengee username. When this equals our login the offer is
        /// addressed to us; otherwise it is a third-party offer (or our own
        /// outgoing request being echoed back).
        to: String,
        /// GGS game type / board variant, for example `8b`.
        game_type: String,
        /// GGS ask mode, for example `U`.
        mode: String,
    },
    /// `/os: -   .<id> ...` — an ask offer has been removed from the server's
    /// queue. Emitted both when the challenger manually withdraws the offer
    /// AND when the server consumes the request during match creation.
    AskWithdrawn { id: String },
    /// `/os: error <id> <message>` — the server rejected our last command for
    /// the given match. In Othello play this most commonly means an illegal
    /// move ("not your turn", "illegal move"), which the engine would
    /// otherwise wait to timeout on. Surfacing the raw message is the only
    /// diagnostic available: see `Game::err_string` in
    /// `Service/GameLib/src/Game.C`.
    ServerError { id: String, message: String },
    /// Everything else: MOTD, `help ...` output, `who` output, `+ack:` echoes, etc.
    Unknown,
}

/// Parses one complete server chunk (the lines between two `READY` markers,
/// with the `READY` itself NOT included). Blank lines inside the chunk are tolerated.
/// `my_username` is used to decide `my_color` / `my_clock` by
/// comparing against the names in `+ match` and the `|<name> (<rating> X)` clock lines.
pub fn parse_server_chunk(lines: &[&str], my_username: &str) -> OsEvent {
    // Find the first non-blank line (the header).
    let header = match lines.iter().find(|l| !l.trim().is_empty()) {
        Some(l) => *l,
        None => return OsEvent::Unknown,
    };

    let rest = match header.strip_prefix("/os: ") {
        Some(r) => r.trim_start(),
        None => return OsEvent::Unknown,
    };

    // Dispatch on the header's leading tag.
    if let Some(body) = rest.strip_prefix("+ match ") {
        return parse_match_created(body, my_username);
    }
    if let Some(body) = rest.strip_prefix("- match ") {
        return parse_match_end(body);
    }
    if let Some(body) = rest.strip_prefix("end ") {
        return parse_child_match_end(body);
    }
    if let Some(body) = rest.strip_prefix("join ") {
        return parse_match_state(body, lines, my_username);
    }
    if let Some(body) = rest.strip_prefix("update ") {
        return parse_match_state(body, lines, my_username);
    }
    if let Some(body) = rest.strip_prefix("error ") {
        return parse_server_error(body);
    }
    // Ask offer: `+   .<id> ...`. Must be checked AFTER `+ match` so we don't
    // swallow match-created chunks. The server uses variable whitespace
    // between `+` and the id (`Form` with padded fields), so trim before the
    // `.<id>` check.
    if let Some(body) = rest.strip_prefix("+ ") {
        let body = body.trim_start();
        if body.starts_with('.') {
            return parse_ask_offer(body).unwrap_or(OsEvent::Unknown);
        }
    }
    // Ask withdrawal: `-   .<id> ...`. Same variable-whitespace caveat as
    // `+`. We only need the id; the rest is ignored.
    if let Some(body) = rest.strip_prefix("- ")
        && let Some(first) = body.split_whitespace().next()
        && first.starts_with('.')
        && first.len() > 1
    {
        return OsEvent::AskWithdrawn {
            id: first.to_string(),
        };
    }

    // Kibitz: `<id> A <rating> <from>: <message>`.
    if let Some(ev) = parse_kibitz(rest) {
        return ev;
    }

    OsEvent::Unknown
}

/// Parse a `+   .<id> <r0> <n0> <clock> <type> <flag> <r1> <n1> [...]` ask offer.
///
/// The trailing optional fields (`<my_clock>`, `<stored_id>`) are ignored.
/// Returns `None` on any missing or shape-mismatched token, which the caller
/// maps to `OsEvent::Unknown`.
fn parse_ask_offer(body: &str) -> Option<OsEvent> {
    let mut it = body.split_whitespace();
    let id = it.next().filter(|s| s.starts_with('.'))?.to_string();
    let _r0 = it.next()?;
    let from = it.next()?.to_string();
    let _clock = it.next()?;
    let game_type = it.next()?.to_string();
    let mode = it.next()?.to_string();
    let _r1 = it.next()?;
    let to = it.next()?.to_string();
    Some(OsEvent::AskOffer {
        id,
        from,
        to,
        game_type,
        mode,
    })
}

fn parse_match_created(body: &str, my_username: &str) -> OsEvent {
    // Expect: <id> <r1> <n1> <r2> <n2> <variant> <type>
    let mut it = body.split_whitespace();
    let id = match it.next() {
        Some(s) => s.to_string(),
        None => return OsEvent::Unknown,
    };
    let _r1 = it.next();
    let n1 = match it.next() {
        Some(s) => s,
        None => return OsEvent::Unknown,
    };
    let _r2 = it.next();
    let n2 = match it.next() {
        Some(s) => s,
        None => return OsEvent::Unknown,
    };
    let variant = it.next().map(str::to_string);

    let opponent = if my_username == n1 {
        Some(n2.to_string())
    } else if my_username == n2 {
        Some(n1.to_string())
    } else {
        None
    };

    OsEvent::MatchCreated {
        id,
        opponent,
        variant,
    }
}

fn parse_match_end(body: &str) -> OsEvent {
    // Expect: <id> <r1> <n1> <r2> <n2> <variant> <type> <result...>
    // The result is numeric for completed games, but can be text such as
    // `aborted` or `<name> left` for interrupted games. The Java client and
    // Edax both treat any `- match` as a match end, regardless of score shape.
    let parts: Vec<&str> = body.split_whitespace().collect();
    if parts.len() < 8 {
        return OsEvent::Unknown;
    }
    let id = parts[0].to_string();
    let result = parts[7..].join(" ");
    let score = parts[7].parse::<f64>().ok();
    OsEvent::MatchEnd { id, result, score }
}

fn parse_child_match_end(body: &str) -> OsEvent {
    // Synchro child games are reported as `/os: end <id> ...`. The trailing
    // payload is not needed for session cleanup, so parse only the child id
    // and preserve any remainder as best-effort result text.
    let mut parts = body.split_whitespace();
    let Some(id) = parts.next() else {
        return OsEvent::Unknown;
    };
    let remainder: Vec<&str> = parts.collect();
    let result = if remainder.is_empty() {
        "end".to_string()
    } else {
        remainder.join(" ")
    };
    let score = remainder.first().and_then(|s| s.parse::<f64>().ok());
    OsEvent::MatchEnd {
        id: id.to_string(),
        result,
        score,
    }
}

/// Parse `error <id> <message>`. The id is typically a match id (`.5`)
/// but the server also uses `error` for out-of-match failures (e.g. before a
/// match exists). We accept any non-empty first token.
fn parse_server_error(body: &str) -> OsEvent {
    let body = body.trim();
    if body.is_empty() {
        return OsEvent::Unknown;
    }
    let (id, message) = match body.split_once(char::is_whitespace) {
        Some((id, msg)) => (id.to_string(), msg.trim().to_string()),
        None => (body.to_string(), String::new()),
    };
    OsEvent::ServerError { id, message }
}

fn parse_kibitz(rest: &str) -> Option<OsEvent> {
    // `<id> <tag> <rating> <from>: <message>` where `<tag>` is:
    //   * `A` — tell all (default)
    //   * `O` — tell observers only
    //   * `P` — tell players only
    // See `EXE_Service::tell` in `Service/GameLib/src/EXE_Service.C` on the
    // GGS server for the emission site.
    //
    let (id, rest) = take_word(rest)?;
    let (tag, rest) = take_word(rest)?;
    if !matches!(tag, "A" | "O" | "P") {
        return None;
    }
    let (_rating, remainder) = take_word(rest)?;
    let remainder = remainder.trim_start();
    // remainder: "<from>: <message>"
    let (from, message) = remainder.split_once(": ")?;
    Some(OsEvent::Kibitz {
        id: id.to_string(),
        from: from.to_string(),
        message: message.to_string(),
    })
}

fn take_word(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    let end = s.find(char::is_whitespace).unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((&s[..end], &s[end..]))
}

fn parse_match_state(header_body: &str, lines: &[&str], my_username: &str) -> OsEvent {
    // Header body: `<id> <variant> <type>`
    let mut it = header_body.split_whitespace();
    let id = match it.next() {
        Some(s) => s.to_string(),
        None => return OsEvent::Unknown,
    };

    let mut my_clock = None;
    let mut my_color = None;

    let mut board_rows: Vec<[u8; 8]> = Vec::with_capacity(16);
    let mut side_to_move: Option<Disc> = None;

    for raw in lines.iter() {
        let line = *raw;

        if let Some(clock) = parse_clock_line(line, my_username) {
            if clock.is_mine {
                my_color = clock.color;
                my_clock = clock.clock;
            }
            continue;
        }

        if let Some(row) = parse_board_row(line) {
            board_rows.push(row);
            continue;
        }

        if let Some(stm) = parse_side_to_move(line) {
            side_to_move = Some(stm);
        }
    }

    if board_rows.len() < 8 {
        return OsEvent::Unknown;
    }
    let board_rows = &board_rows[board_rows.len() - 8..];
    let stm = match side_to_move {
        Some(s) => s,
        None => return OsEvent::Unknown,
    };

    // Assemble the 64 bytes in Square::iter order (A1..H1, A2..H2, ..., A8..H8).
    // `parse_board_row` already normalises cells into Board's alphabet.
    let mut board_bytes = [0u8; 64];
    for (i, row) in board_rows.iter().enumerate() {
        board_bytes[i * 8..i * 8 + 8].copy_from_slice(row);
    }
    let board_string = std::str::from_utf8(&board_bytes).expect("ASCII cells");
    let board = match Board::from_string(board_string, stm) {
        Ok(b) => b,
        Err(_) => return OsEvent::Unknown,
    };

    OsEvent::MatchState {
        id,
        board,
        side_to_move: stm,
        my_color,
        my_clock,
    }
}

/// A parsed `|<name> (<rating> <mark>) <main>/<increment>/<extension>` clock line.
struct ClockLine {
    is_mine: bool,
    color: Option<Disc>,
    clock: Option<PlayerClock>,
}

/// Parse `|<name>{spaces}(<rating> <colormark>) <main>/<increment>/<extension>`.
/// Only populates `color` / `clock` when the line belongs to `my_username` —
/// everyone else's clock is discarded before any per-field parsing or
/// allocation, since GGS chunks list every watcher on match kibitz.
fn parse_clock_line(line: &str, my_username: &str) -> Option<ClockLine> {
    let body = line.strip_prefix('|')?;
    let (name_part, after) = body.split_once('(')?;
    let name = name_part.trim();
    if name.is_empty() {
        return None;
    }
    let is_mine = name == my_username;
    let (rating_block, after_paren) = after.split_once(')')?;
    let (color, clock) = if is_mine {
        // The rating block ends in the live color mark, e.g. `1720.0 *`.
        // Empty increment is valid: the common Othello default prints as
        // `<main>//<extension>`.
        let c = rating_block
            .split_whitespace()
            .last()
            .and_then(parse_color_mark);
        (c, parse_player_clock(after_paren.trim_start()))
    } else {
        (None, None)
    };
    Some(ClockLine {
        is_mine,
        color,
        clock,
    })
}

fn parse_player_clock(fields: &str) -> Option<PlayerClock> {
    let mut fields = fields.split('/');
    let main_ms = parse_clock_time_field(fields.next()?)?;
    let increment_ms = fields.next().and_then(parse_clock_time_field).unwrap_or(0);
    let byoyomi_ms = fields.next().and_then(parse_clock_time_field).unwrap_or(0);
    Some(PlayerClock {
        main_ms,
        increment_ms,
        byoyomi_ms,
    })
}

fn parse_color_mark(mark: &str) -> Option<Disc> {
    match mark {
        "*" | "X" => Some(Disc::Black),
        "O" => Some(Disc::White),
        _ => None,
    }
}

/// Parse the leading time component from a comma-separated field and return it
/// in milliseconds.
///
/// The server's `HHMMSS::print` (`Server/Central/src/HHMMSS.C`) uses `setw(5)`
/// and typically emits:
///   * `SS`         when compact or manually written messages use seconds
///   * `MM:SS`      when time < 1 hour
///   * `HH:MM:SS`   when time ≥ 1 hour
///   * `DD.HH:MM:SS` when time ≥ 24 hours
///
/// We accept all four. Negative fields such as `-00:05` are clamped to 0 ms
/// instead of dropping the whole clock, because GGS emits them once a player
/// has already overstepped the active budget. Anything else still returns
/// `None`, which causes the enclosing clock line to be treated as unparseable.
fn parse_clock_time_field(field: &str) -> Option<u64> {
    let time_str = field.split(',').next()?.trim();
    if time_str.is_empty() {
        return None;
    }

    let (negative, time_str) = match time_str.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, time_str),
    };
    if time_str.is_empty() {
        return None;
    }

    // Optional leading `<days>.` segment.
    let (days, rest) = match time_str.split_once('.') {
        Some((d, r)) => (d.trim().parse::<u64>().ok()?, r),
        None => (0u64, time_str),
    };

    // Remaining segment is `SS`, `MM:SS`, or `HH:MM:SS`.
    let parts: Vec<&str> = rest.split(':').collect();
    let (hh, mm, ss) = match parts.as_slice() {
        [ss] => (0u64, 0u64, ss.trim().parse::<u64>().ok()?),
        [mm, ss] => (
            0u64,
            mm.trim().parse::<u64>().ok()?,
            ss.trim().parse::<u64>().ok()?,
        ),
        [hh, mm, ss] => (
            hh.trim().parse::<u64>().ok()?,
            mm.trim().parse::<u64>().ok()?,
            ss.trim().parse::<u64>().ok()?,
        ),
        _ => return None,
    };

    if negative {
        Some(0)
    } else {
        Some((((days * 24 + hh) * 60 + mm) * 60 + ss) * 1000)
    }
}

/// Parse a board row line: `| <digit> <c> <c> <c> <c> <c> <c> <c> <c> <digit>`.
/// Cells are emitted in `Board::from_string`'s alphabet (`-`, `X`, `O`) so the
/// caller can concatenate rows into a 64-byte board string directly.
fn parse_board_row(line: &str) -> Option<[u8; 8]> {
    let body = line.strip_prefix('|')?;
    let body = body.trim_start();
    let mut it = body.split_whitespace();
    let rank = it.next()?;
    if rank.len() != 1 || !rank.as_bytes()[0].is_ascii_digit() {
        return None;
    }
    let mut cells = [b'-'; 8];
    for cell in cells.iter_mut() {
        let tok = it.next()?;
        if tok.len() != 1 {
            return None;
        }
        *cell = match tok.as_bytes()[0] {
            b'-' => b'-',
            b'*' => b'X',
            b'O' => b'O',
            _ => return None,
        };
    }
    Some(cells)
}

/// Parse `|* to move` / `|O to move`.
fn parse_side_to_move(line: &str) -> Option<Disc> {
    let body = line.strip_prefix('|')?.trim_start();
    let rest = body.strip_suffix("to move")?.trim_end();
    match rest {
        "*" => Some(Disc::Black),
        "O" => Some(Disc::White),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reversi_core::square::Square;

    const SAMPLE: &str = include_str!("../../tests/fixtures/ggs/session-sample.log");

    fn chunks() -> Vec<Vec<&'static str>> {
        let mut out = Vec::new();
        let mut cur: Vec<&str> = Vec::new();
        for line in SAMPLE.lines() {
            if line.trim() == "READY" {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            } else if !line.is_empty() || !cur.is_empty() {
                cur.push(line);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    /// Find the first chunk whose first non-blank line starts with `prefix`.
    fn find_chunk<'a>(all: &'a [Vec<&'static str>], prefix: &str) -> &'a [&'static str] {
        all.iter()
            .find(|c| {
                c.iter()
                    .find(|l| !l.trim().is_empty())
                    .map(|l| l.starts_with(prefix))
                    .unwrap_or(false)
            })
            .expect("chunk not found")
            .as_slice()
    }

    #[test]
    fn match_created_is_parsed() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: + match .5");
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchCreated {
                id,
                opponent,
                variant,
            } => {
                assert_eq!(id, ".5");
                assert_eq!(opponent.as_deref(), Some("bob"));
                assert_eq!(variant.as_deref(), Some("8b"));
            }
            other => panic!("expected MatchCreated, got {other:?}"),
        }
    }

    #[test]
    fn unknown_role_when_username_not_in_match() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: + match .5");
        let ev = parse_server_chunk(chunk, "charlie");
        match ev {
            OsEvent::MatchCreated { id, opponent, .. } => {
                assert_eq!(id, ".5");
                assert!(opponent.is_none());
            }
            other => panic!("expected MatchCreated, got {other:?}"),
        }
    }

    #[test]
    fn synchro_match_created_captures_variant() {
        let ev = parse_server_chunk(&["/os: + match .5 1720 alice 1720 bob s8r20 R"], "alice");
        match ev {
            OsEvent::MatchCreated {
                id,
                opponent,
                variant,
            } => {
                assert_eq!(id, ".5");
                assert_eq!(opponent.as_deref(), Some("bob"));
                assert_eq!(variant.as_deref(), Some("s8r20"));
            }
            other => panic!("expected MatchCreated, got {other:?}"),
        }
    }

    #[test]
    fn match_state_join_is_parsed() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: join .5");
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState {
                id,
                board,
                side_to_move,
                my_color,
                my_clock,
            } => {
                assert_eq!(id, ".5");
                assert_eq!(side_to_move, Disc::Black);
                assert_eq!(my_color, Some(Disc::Black));
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 59_000,
                        increment_ms: 0,
                        byoyomi_ms: 120_000,
                    })
                );
                assert_eq!(board, Board::new());
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn match_state_update_is_parsed() {
        let all = chunks();
        // First update chunk is the `|  1: f5//3.47` one.
        let chunk = all
            .iter()
            .find(|c| {
                c.iter().any(|l| l.starts_with("/os: update .5"))
                    && c.iter().any(|l| l.contains("1: f5"))
            })
            .expect("update chunk not found");
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState {
                id,
                side_to_move,
                my_clock,
                ..
            } => {
                assert_eq!(id, ".5");
                assert_eq!(side_to_move, Disc::White);
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 56_000,
                        increment_ms: 0,
                        byoyomi_ms: 120_000,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn join_with_history_uses_current_board_not_start_board() {
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|2 move(s)",
            "|alice    (1720.0 *) 01:00,0:0//02:00,0:0",
            "|bob      (1720.0 O) 01:00,0:0//02:00,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
            "|  1: f5//3.47",
            "|  2: f6//1.50",
            "|alice    (1720.0 *) 00:56,1:0//02:00,1:0",
            "|bob      (1720.0 O) 00:58,1:0//02:00,1:0",
            "|",
            "|   A B C D E F G H",
            "| 1 * - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O * - - 5 ",
            "| 6 - - - - - O - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { board, .. } => {
                assert_eq!(board.get_disc_at(Square::A1, Disc::Black), Disc::Black);
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn match_end_is_parsed() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: - match .5");
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchEnd { id, result, score } => {
                assert_eq!(id, ".5");
                assert_eq!(result, "-64.00");
                assert!((score.unwrap() - -64.0).abs() < 1e-9);
            }
            other => panic!("expected MatchEnd, got {other:?}"),
        }
    }

    #[test]
    fn interrupted_match_end_is_parsed_without_score() {
        let ev = parse_server_chunk(
            &["/os: - match .5 1720 alice 1720 bob 8b U alice left"],
            "alice",
        );
        match ev {
            OsEvent::MatchEnd { id, result, score } => {
                assert_eq!(id, ".5");
                assert_eq!(result, "alice left");
                assert_eq!(score, None);
            }
            other => panic!("expected MatchEnd, got {other:?}"),
        }
    }

    #[test]
    fn synchro_child_end_is_parsed_as_match_end() {
        let ev = parse_server_chunk(&["/os: end .5.1 0.00"], "alice");
        match ev {
            OsEvent::MatchEnd { id, result, score } => {
                assert_eq!(id, ".5.1");
                assert_eq!(result, "0.00");
                assert_eq!(score, Some(0.0));
            }
            other => panic!("expected MatchEnd, got {other:?}"),
        }
    }

    #[test]
    fn kibitz_is_parsed() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: .5 A");
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::Kibitz { id, from, message } => {
                assert_eq!(id, ".5");
                assert_eq!(from, "bob");
                assert_eq!(message, "good game");
            }
            other => panic!("expected Kibitz, got {other:?}"),
        }
    }

    #[test]
    fn kibitz_with_padded_rating_is_parsed() {
        let ev = parse_server_chunk(&["/os: .5 A    0 bob: hello"], "alice");
        match ev {
            OsEvent::Kibitz { id, from, message } => {
                assert_eq!(id, ".5");
                assert_eq!(from, "bob");
                assert_eq!(message, "hello");
            }
            other => panic!("expected Kibitz, got {other:?}"),
        }
    }

    #[test]
    fn ask_offer_is_parsed_from_sample() {
        // Sample line:
        //   /os: +   .8 1720.0 alice    01:00//02:00       8b U 1720.0 bob
        // alice is the challenger (p1), bob is the challengee (p2).
        let all = chunks();
        let chunk = find_chunk(&all, "/os: +   .");
        let ev = parse_server_chunk(chunk, "bob");
        match ev {
            OsEvent::AskOffer {
                id,
                from,
                to,
                game_type,
                mode,
            } => {
                assert_eq!(id, ".8");
                assert_eq!(from, "alice");
                assert_eq!(to, "bob");
                assert_eq!(game_type, "8b");
                assert_eq!(mode, "U");
            }
            other => panic!("expected AskOffer, got {other:?}"),
        }
    }

    #[test]
    fn ask_withdrawal_is_parsed() {
        // `- <padding> .id ...` is an ask withdrawal (or the server's
        // post-accept cleanup). The session needs to observe it as a distinct
        // event, not classify it as Unknown.
        let ev = parse_server_chunk(
            &["/os: -   .8 1720.0 alice    01:00//02:00       8b U 1720.0 bob"],
            "bob",
        );
        match ev {
            OsEvent::AskWithdrawn { id } => assert_eq!(id, ".8"),
            other => panic!("expected AskWithdrawn, got {other:?}"),
        }
    }

    #[test]
    fn match_end_is_not_classified_as_ask_withdrawal() {
        // Regression: `- match .5 ...` must still route to MatchEnd.
        let ev = parse_server_chunk(
            &["/os: - match .5 1720 alice 1720 bob 8b U -64.00"],
            "alice",
        );
        assert!(matches!(ev, OsEvent::MatchEnd { .. }), "got {ev:?}");
    }

    #[test]
    fn match_created_is_not_classified_as_ask_offer() {
        // Regression: `+ match .5 ...` must still route to MatchCreated even
        // though `+ ` prefix also matches ask-offer dispatch.
        let ev = parse_server_chunk(&["/os: + match .5 1720 alice 1720 bob 8b U"], "alice");
        assert!(matches!(ev, OsEvent::MatchCreated { .. }), "got {ev:?}");
    }

    #[test]
    fn match_state_missing_or_zero_byoyomi_is_zero() {
        // Synthetic chunk: same shape as a join chunk but with `//0:00` on alice's
        // clock and no `//` segment at all on bob's. Both should map to 0 ms.
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 00:59,0:0//0:00,0:0",
            "|bob      (1720.0 O) 01:00,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { my_clock, .. } => {
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 59_000,
                        increment_ms: 0,
                        byoyomi_ms: 0,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn clock_disambiguation_when_my_username_is_second_player() {
        // Same synthetic chunk shape as the byoyomi test, but with alice
        // listed first (different clock) and bob second. With
        // my_username = "bob", the parser must pick bob's clock line, not
        // alice's.
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 00:45,0:0//02:00,0:0",
            "|bob      (1720.0 O) 00:59,0:0//01:30,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "bob");
        match ev {
            OsEvent::MatchState {
                my_color, my_clock, ..
            } => {
                assert_eq!(my_color, Some(Disc::White));
                // bob's clock, NOT alice's 45_000/120_000.
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 59_000,
                        increment_ms: 0,
                        byoyomi_ms: 90_000,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn unparseable_matching_clock_is_reported_missing() {
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) not-a-clock",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState {
                my_color, my_clock, ..
            } => {
                assert_eq!(my_color, Some(Disc::Black));
                assert_eq!(my_clock, None);
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn zero_clock_is_preserved_as_clock_present() {
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 00:00,0:0//0:00,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { my_clock, .. } => {
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 0,
                        increment_ms: 0,
                        byoyomi_ms: 0,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn clock_with_seconds_only_fields_is_parsed() {
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 120,0:0/5,0:1/0,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { my_clock, .. } => {
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 120_000,
                        increment_ms: 5_000,
                        byoyomi_ms: 0,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn increment_clock_field_is_parsed() {
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 05:00,0:0/00:05,0:1/0:00,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { my_clock, .. } => {
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 300_000,
                        increment_ms: 5_000,
                        byoyomi_ms: 0,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn clock_with_hour_field_is_parsed() {
        // A long tournament clock: 1 hour main, 5 minute byoyomi.
        // HHMMSS prints `01:00:00` / `05:00` with setw(5).
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 01:00:00,0:0//05:00,0:0",
            "|bob      (1720.0 O) 01:00:00,0:0//05:00,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { my_clock, .. } => {
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 3_600_000,
                        increment_ms: 0,
                        byoyomi_ms: 300_000,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn clock_with_days_field_is_parsed() {
        // Pathological but real: `HHMMSS::print` emits `<dd>.<HH:MM:SS>` when
        // time ≥ 24 hours. Accept it so the parse doesn't silently fall back
        // to level-based search on very long controls.
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) 1.00:00:00,0:0//05:00,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { my_clock, .. } => {
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 86_400_000,
                        increment_ms: 0,
                        byoyomi_ms: 300_000,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn clock_with_negative_main_field_is_clamped_not_dropped() {
        let chunk: &[&str] = &[
            "/os: join .5 8b K?",
            "|0 move(s)",
            "|  0: PASS",
            "|alice    (1720.0 *) -00:05,0:0//02:00,0:0",
            "|",
            "|   A B C D E F G H",
            "| 1 - - - - - - - - 1 ",
            "| 2 - - - - - - - - 2 ",
            "| 3 - - - - - - - - 3 ",
            "| 4 - - - O * - - - 4 ",
            "| 5 - - - * O - - - 5 ",
            "| 6 - - - - - - - - 6 ",
            "| 7 - - - - - - - - 7 ",
            "| 8 - - - - - - - - 8 ",
            "|   A B C D E F G H",
            "|",
            "|* to move",
        ];
        let ev = parse_server_chunk(chunk, "alice");
        match ev {
            OsEvent::MatchState { my_clock, .. } => {
                assert_eq!(
                    my_clock,
                    Some(PlayerClock {
                        main_ms: 0,
                        increment_ms: 0,
                        byoyomi_ms: 120_000,
                    })
                );
            }
            other => panic!("expected MatchState, got {other:?}"),
        }
    }

    #[test]
    fn kibitz_observer_and_player_tags_are_parsed() {
        // `tell /os o .5 ...` emits `O`; `tell /os p .5 ...` emits `P`.
        // Both should map to Kibitz, not Unknown.
        for tag in ["O", "P"] {
            let line = format!("/os: .5 {tag} 1720 bob: hello");
            let ev = parse_server_chunk(&[line.as_str()], "alice");
            match ev {
                OsEvent::Kibitz { id, from, message } => {
                    assert_eq!(id, ".5");
                    assert_eq!(from, "bob");
                    assert_eq!(message, "hello");
                }
                other => panic!("expected Kibitz for tag {tag}, got {other:?}"),
            }
        }
    }

    #[test]
    fn kibitz_unknown_tag_is_classified_unknown() {
        let ev = parse_server_chunk(&["/os: .5 Z 1720 bob: hello"], "alice");
        assert!(matches!(ev, OsEvent::Unknown), "got {ev:?}");
    }

    #[test]
    fn server_error_chunks_are_parsed() {
        let cases: &[(&str, &str, &str)] = &[
            ("/os: error .5 illegal move", ".5", "illegal move"),
            ("/os: error .5 not your turn", ".5", "not your turn"),
            ("/os: error .5", ".5", ""),
        ];
        for &(chunk, want_id, want_msg) in cases {
            match parse_server_chunk(&[chunk], "alice") {
                OsEvent::ServerError { id, message } => {
                    assert_eq!(id, want_id, "chunk: {chunk}");
                    assert_eq!(message, want_msg, "chunk: {chunk}");
                }
                other => panic!("chunk {chunk}: expected ServerError, got {other:?}"),
            }
        }
    }

    #[test]
    fn malformed_chunk_does_not_panic() {
        assert!(matches!(parse_server_chunk(&[], "alice"), OsEvent::Unknown));
        assert!(matches!(
            parse_server_chunk(&["garbage"], "alice"),
            OsEvent::Unknown
        ));
        assert!(matches!(
            parse_server_chunk(&["/os: join"], "alice"),
            OsEvent::Unknown
        ));
        // Header looks like update but no body at all.
        assert!(matches!(
            parse_server_chunk(&["/os: update .5 8b K?"], "alice"),
            OsEvent::Unknown
        ));
    }
}
