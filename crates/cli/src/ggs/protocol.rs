//! GGS `/os` server-event parser.
//!
//! Parses every service message in a complete chunk (the block of lines
//! delimited by `READY` / `ALERT` markers) into structured [`OsEvent`]s.

use reversi_core::board::Board;
use reversi_core::disc::Disc;

/// Clock information parsed from our GGS clock line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerClock {
    /// Remaining active-budget time for us in ms.
    pub main_ms: u64,
    /// Additive every-move increment in ms; 0 for other increment modes.
    pub increment_ms: u64,
}

/// A structured event extracted from a single GGS receive chunk.
#[derive(Debug, Clone, PartialEq)]
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
    MatchState(MatchState),
    /// A join/update header whose board body was unavailable or unparseable.
    /// This commonly occurs when a reconnect races a persisted `client +`
    /// setting; the session requests one verbose state after forcing
    /// `client -`.
    MatchStateUnavailable {
        id: String,
        variant: Option<String>,
        is_player: bool,
    },
    /// An unparseable join/update whose clock lines all name other players:
    /// we are watching this match, not playing it.
    SpectatedMatchState { id: String, variant: Option<String> },
    /// `/os: - match <id> ...` or `/os: end <id> ...` chunk.
    MatchEnd {
        id: String,
        /// Match result text as the server prints it (`-64.00`, `aborted`, `name left`, ...).
        result: String,
        /// Disc difference from our side, when the result is numeric and we played.
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
    AskOffer(AskOffer),
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
    /// A request/accept command failed before a match was created. GGS emits
    /// these as generic `/os: ERR ...` messages.
    RequestError {
        /// Present for `request '<id>' not found`; the other errors do not
        /// echo the request id and cannot be correlated safely.
        id: Option<String>,
        message: String,
    },
    /// Everything else: MOTD, `help ...` output, `who` output, `+ack:` echoes, etc.
    Unknown,
}

/// Position, side to move, our color and clock, and both players' clocks
/// from one `/os: join` or `/os: update` chunk.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchState {
    pub id: String,
    /// Raw variant token from the join/update header.
    pub variant: Option<String>,
    /// Other player, when it can be recovered from the clock lines.
    pub opponent: Option<String>,
    pub board: Board,
    pub side_to_move: Disc,
    /// Actual color marker for `my_username` in this game, parsed from
    /// the clock line. This is authoritative because `+ match` lists
    /// request players, not necessarily black then white.
    pub my_color: Option<Disc>,
    /// `None` when our clock line is absent or unparseable.
    pub my_clock: Option<PlayerClock>,
    /// Both players' live clock lines, as shown to the user.
    pub clocks: Vec<DisplayClock>,
    /// The last numbered move listed (`f5`, `PASS`); a join lists the whole
    /// game, an update only its last move.
    pub last_move: Option<String>,
}

/// A match-request ("ask") offer; [`OsEvent::AskOffer`] describes the wire format.
#[derive(Debug, Clone, PartialEq)]
pub struct AskOffer {
    pub id: String,
    /// Challenger username.
    pub from: String,
    /// Challengee username. When this equals our login the offer is
    /// addressed to us; otherwise it is a third-party offer (or our own
    /// outgoing request being echoed back).
    pub to: String,
    /// GGS game type / board variant, for example `8b`.
    pub game_type: String,
    /// Raw GGS clock setting, for example `01:00//02:00`.
    pub clock: String,
    /// GGS ask mode, for example `U`.
    pub mode: String,
}

/// Parses every `/os` message in one boundary-delimited receive chunk.
///
/// GGS can emit multiple service messages before the next boundary marker.
/// Each `/os: ` header starts a new message; its following lines belong to it
/// until the next header.
pub(super) fn parse_server_messages(lines: &[&str], my_username: &str) -> Vec<OsEvent> {
    let first_header = lines
        .iter()
        .position(|line| line.starts_with("/os: "))
        .unwrap_or(0);
    lines[first_header..]
        .chunk_by(|_, next| !next.starts_with("/os: "))
        .map(|message| parse_server_chunk(message, my_username))
        .collect()
}

/// The header body of a join/update message.
fn match_state_body<'a>(lines: &[&'a str]) -> Option<&'a str> {
    let header = lines.iter().find(|line| !line.trim().is_empty())?;
    let rest = header
        .strip_prefix("/os: ")?
        .trim_start_matches('\x07')
        .trim_start();
    rest.strip_prefix("join ")
        .or_else(|| rest.strip_prefix("update "))
}

/// Parses one complete server chunk (the lines between boundary markers,
/// with the marker itself NOT included). Blank lines inside the chunk are tolerated.
/// `my_username` is used to decide `my_color` / `my_clock` by
/// comparing against the names in `+ match` and the `|<name> (<rating> X)` clock lines.
pub fn parse_server_chunk(lines: &[&str], my_username: &str) -> OsEvent {
    parse_message(lines, my_username).unwrap_or(OsEvent::Unknown)
}

fn parse_message(lines: &[&str], my_username: &str) -> Option<OsEvent> {
    // Find the first non-blank line (the header).
    let header = lines.iter().find(|l| !l.trim().is_empty())?;
    let rest = header
        .strip_prefix("/os: ")?
        .trim_start_matches('\x07')
        .trim_start();

    // Dispatch on the header's leading tag.
    if let Some(body) = rest.strip_prefix("+ match ") {
        return parse_match_created(body, my_username);
    }
    if let Some(body) = rest.strip_prefix("- match ") {
        return parse_match_end(body, my_username);
    }
    if let Some(body) = rest.strip_prefix("end ") {
        return parse_child_match_end(body, my_username);
    }
    if let Some(body) = match_state_body(lines) {
        return parse_match_state(body, lines, my_username);
    }
    if let Some(body) = rest.strip_prefix("error ") {
        return parse_server_error(body);
    }
    if let Some(body) = rest.strip_prefix("ERR ")
        && let Some(error) = parse_request_error(body)
    {
        return Some(error);
    }
    // Ask offer: `+   .<id> ...`. Must be checked AFTER `+ match` so we don't
    // swallow match-created chunks. The server uses variable whitespace
    // between `+` and the id (`Form` with padded fields), so trim before the
    // `.<id>` check.
    if let Some(body) = rest.strip_prefix("+ ") {
        let body = body.trim_start();
        if body.starts_with('.') {
            return parse_ask_offer(body, my_username);
        }
    }
    // Ask withdrawal: `-   .<id> ...`. Same variable-whitespace caveat as
    // `+`. We only need the id; the rest is ignored.
    if let Some(body) = rest.strip_prefix("- ")
        && let Some(first) = body.split_whitespace().next()
        && first.starts_with('.')
        && first.len() > 1
    {
        return Some(OsEvent::AskWithdrawn {
            id: first.to_string(),
        });
    }

    // Kibitz: `<id> A <rating> <from>: <message>`.
    parse_kibitz(rest)
}

/// Parse a `+   .<id> <r0> <n0> <clock> <type> <flag> <r1> <n1> [...]` ask offer.
///
/// An optional second clock is selected when it belongs to this client;
/// the trailing stored-game id is ignored.
/// Returns `None` on any missing or shape-mismatched token.
fn parse_ask_offer(body: &str, my_username: &str) -> Option<OsEvent> {
    let mut it = body.split_whitespace();
    let id = it.next().filter(|s| s.starts_with('.'))?.to_string();
    let _r0 = it.next()?;
    let from = it.next()?.to_string();
    let challenger_clock = it.next()?;
    let game_type = it.next()?.to_string();
    let mode = it.next()?.to_string();
    let _r1 = it.next()?;
    let to = it.next()?.to_string();
    let second_clock = it.next().filter(|field| field.contains('/'));
    let clock = if from == my_username {
        challenger_clock
    } else {
        second_clock.unwrap_or(challenger_clock)
    }
    .to_string();
    Some(OsEvent::AskOffer(AskOffer {
        id,
        from,
        to,
        game_type,
        clock,
        mode,
    }))
}

fn parse_match_created(body: &str, my_username: &str) -> Option<OsEvent> {
    // Expect: <id> <r1> <n1> <r2> <n2> <variant> <type>
    let mut it = body.split_whitespace();
    let id = it.next()?.to_string();
    let _r1 = it.next()?;
    let n1 = it.next()?;
    let _r2 = it.next()?;
    let n2 = it.next()?;
    let variant = it.next().map(str::to_string);

    let opponent = if my_username == n1 {
        Some(n2.to_string())
    } else if my_username == n2 {
        Some(n1.to_string())
    } else {
        None
    };

    Some(OsEvent::MatchCreated {
        id,
        opponent,
        variant,
    })
}

fn parse_match_end(body: &str, my_username: &str) -> Option<OsEvent> {
    // Expect: <id> <r1> <n1> <r2> <n2> <variant> <type> <result...>
    // The result is numeric for completed games, but can be text such as
    // `aborted` or `<name> left` for interrupted games; any `- match` ends the
    // match, regardless of score shape.
    // Rated matches append a global counter after the result. Logins may be
    // numeric, so `<name> left` must be ruled out before parsing a score.
    let parts: Vec<&str> = body.split_whitespace().collect();
    if parts.len() < 8 {
        return None;
    }
    let id = parts[0].to_string();
    let result = parts[7..].join(" ");
    let score = if parts.get(8) == Some(&"left") {
        None
    } else {
        player_score(parts[7], parts[2], parts[4], my_username)
    };
    Some(OsEvent::MatchEnd { id, result, score })
}

/// Converts a result printed from player 0's side to our side.
fn player_score(result: &str, name0: &str, name1: &str, my_username: &str) -> Option<f64> {
    let score: f64 = result.trim().parse().ok()?;
    if name0 == my_username {
        Some(score)
    } else if name1 == my_username {
        Some(-score)
    } else {
        None
    }
}

fn parse_child_match_end(body: &str, my_username: &str) -> Option<OsEvent> {
    // Synchro child games are reported as `/os: end <id> ( <n0> vs. <n1> ) <r0>`.
    // Session cleanup needs only the child id; the remainder is kept as
    // best-effort result text.
    let mut parts = body.split_whitespace();
    let id = parts.next()?;
    let remainder: Vec<&str> = parts.collect();
    let score = match remainder.as_slice() {
        ["(", name0, "vs.", name1, ")", result] => player_score(result, name0, name1, my_username),
        _ => None,
    };
    let result = if remainder.is_empty() {
        "end".to_string()
    } else {
        remainder.join(" ")
    };
    Some(OsEvent::MatchEnd {
        id: id.to_string(),
        result,
        score,
    })
}

/// Parse `error <id> <message>`. The id is typically a match id (`.5`)
/// but the server also uses `error` for out-of-match failures (e.g. before a
/// match exists). We accept any non-empty first token.
fn parse_server_error(body: &str) -> Option<OsEvent> {
    let (id, message) = take_word(body)?;
    Some(OsEvent::ServerError {
        id: id.to_string(),
        message: message.trim().to_string(),
    })
}

fn parse_request_error(message: &str) -> Option<OsEvent> {
    let id = if message == "your request doesn't fit the opponent's formula."
        || message == "you cannot accept your own match request."
    {
        None
    } else {
        Some(
            message
                .strip_prefix("request '")?
                .strip_suffix("' not found.")?
                .to_string(),
        )
    };
    Some(OsEvent::RequestError {
        id,
        message: message.to_string(),
    })
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

/// One player's clock line as shown to the user.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayClock {
    pub name: String,
    pub color: Disc,
    /// `None` when the clock fields are unparseable.
    pub main_ms: Option<u64>,
}

/// Parses the move of `|  <n>: <move>/<eval>/<time>`.
fn parse_move_line(line: &str) -> Option<String> {
    let (number, rest) = line.strip_prefix('|')?.trim_start().split_once(": ")?;
    number.parse::<usize>().ok()?;
    Some(rest.split('/').next()?.trim().to_string())
}

fn parse_match_state(header_body: &str, lines: &[&str], my_username: &str) -> Option<OsEvent> {
    // Verbose header: `<id> <variant> K...`. Compact join/update payloads do
    // not carry a header variant and are deliberately resynced verbosely.
    let mut it = header_body.split_whitespace();
    let id = it.next()?.to_string();
    let variant = match (it.next(), it.next()) {
        (Some(variant), Some(kind)) if kind.starts_with('K') => Some(variant.to_string()),
        _ => None,
    };

    let mut matching_clocks = Vec::new();
    let mut opponent = None;

    let mut board_rows = Vec::with_capacity(16);
    let mut side_to_move = None;
    let mut clocks = Vec::new();
    let mut last_move = None;

    for &line in lines {
        if let Some(mv) = parse_move_line(line) {
            last_move = Some(mv);
        }
        if let Some(clock) = parse_clock_line(line, my_username) {
            clocks.push(DisplayClock {
                name: clock.name.to_string(),
                color: clock.color,
                main_ms: clock.clock.map(|clock| clock.main_ms),
            });
            if clock.is_mine {
                matching_clocks.push((clock.color, clock.clock));
            } else if opponent.is_none() {
                opponent = Some(clock.name.to_string());
            }
        } else if let Some(row) = parse_board_row(line) {
            board_rows.push(row);
        } else if let Some(stm) = parse_side_to_move(line) {
            side_to_move = Some(stm);
        }
    }

    // History-bearing joins print setup clocks first and live clocks last.
    if clocks.len() > 2 {
        clocks.drain(..clocks.len() - 2);
    }
    let is_player = !matching_clocks.is_empty();
    let is_spectator = !is_player && opponent.is_some();
    if is_spectator {
        return Some(OsEvent::SpectatedMatchState { id, variant });
    }
    let Some((board, stm)) = current_position(&board_rows, side_to_move) else {
        return Some(OsEvent::MatchStateUnavailable {
            id,
            variant,
            is_player,
        });
    };
    // In a supported self-play match both player clock lines have our login.
    // Prefer the last clock for the side to move: history-bearing join chunks
    // print setup clocks first and live clocks last. Ordinary matches fall
    // back to their final matching line regardless of whose turn it is.
    let (my_color, my_clock) = matching_clocks
        .iter()
        .rev()
        .find(|(color, _)| *color == stm)
        .or_else(|| matching_clocks.last())
        .map(|&(color, clock)| (Some(color), clock))
        .unwrap_or((None, None));
    if opponent.is_none() && matching_clocks.len() > 1 {
        opponent = Some(my_username.to_string());
    }

    Some(OsEvent::MatchState(MatchState {
        id,
        variant,
        opponent,
        board,
        side_to_move: stm,
        my_color,
        my_clock,
        clocks,
        last_move,
    }))
}

/// The last printed board with its side to move, or `None` if either is missing or invalid.
fn current_position(rows: &[[u8; 8]], side_to_move: Option<Disc>) -> Option<(Board, Disc)> {
    let side_to_move = side_to_move?;
    // Assemble the 64 bytes in Square::iter order (A1..H1, A2..H2, ..., A8..H8).
    // `parse_board_row` already normalises cells into Board's alphabet.
    let cells: String = rows
        .last_chunk::<8>()?
        .iter()
        .flatten()
        .map(|&cell| char::from(cell))
        .collect();
    let board = Board::from_string(&cells, side_to_move).ok()?;
    Some((board, side_to_move))
}

/// A parsed `|<name> (<rating> <mark>) <main>/<increment>/<extension>` clock line.
struct ClockLine<'a> {
    name: &'a str,
    is_mine: bool,
    color: Disc,
    clock: Option<PlayerClock>,
}

/// Parse `|<name>{spaces}(<rating> <colormark>) <main>/<increment>/<extension>`.
fn parse_clock_line<'a>(line: &'a str, my_username: &str) -> Option<ClockLine<'a>> {
    let body = line.strip_prefix('|')?;
    // Login names may contain `(`. The server prints a space-delimited rating
    // block after its fixed-width name field, so split at that delimiter.
    let (name_part, after) = body.rsplit_once(" (")?;
    let name = name_part.trim();
    if name.is_empty() {
        return None;
    }
    let is_mine = name == my_username;
    let (rating_block, after_paren) = after.split_once(')')?;
    // The rating block ends in the live color mark, e.g. `1720.0 *`. Requiring
    // it keeps parenthesized lines such as `|0 move(s)` out of this parser.
    let color = rating_block
        .split_whitespace()
        .last()
        .and_then(parse_color_mark)?;
    // Empty increment is valid: the common Othello default prints as
    // `<main>//<extension>`.
    let clock = parse_player_clock(after_paren.trim_start());
    Some(ClockLine {
        name,
        is_mine,
        color,
        clock,
    })
}

fn parse_player_clock(fields: &str) -> Option<PlayerClock> {
    let mut fields = fields.split('/');
    let main_ms = parse_clock_time_field(fields.next()?)?;
    let increment_ms = fields
        .next()
        .and_then(parse_fischer_increment_field)
        .unwrap_or(0);
    Some(PlayerClock {
        main_ms,
        increment_ms,
    })
}

/// GGS also supports Bronstein (`N`) and every-N-moves increments. The
/// engine's time manager models neither, so only expose an additive increment
/// whose live metadata says it is awarded every move.
fn parse_fischer_increment_field(field: &str) -> Option<u64> {
    let (_, metadata) = field.split_once(',')?;
    let (counter, every) = metadata.trim().split_once(':')?;
    if counter.starts_with('N') {
        return Some(0);
    }
    counter.parse::<u64>().ok()?;
    if every.parse::<u64>().ok()? != 1 {
        return Some(0);
    }
    parse_clock_time_field(field)
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
    let (negative, time_str) = match time_str.strip_prefix('-') {
        Some(rest) => (true, rest.trim()),
        None => (false, time_str),
    };
    let num = |s: &str| s.trim().parse::<u64>().ok();

    // Optional leading `<days>.` segment.
    let (days, rest) = match time_str.split_once('.') {
        Some((d, r)) => (num(d)?, r),
        None => (0, time_str),
    };

    // Remaining segment is `SS`, `MM:SS`, or `HH:MM:SS`.
    let parts: Vec<&str> = rest.split(':').collect();
    let (hh, mm, ss) = match parts.as_slice() {
        [ss] => (0, 0, num(ss)?),
        [mm, ss] => (0, num(mm)?, num(ss)?),
        [hh, mm, ss] => (num(hh)?, num(mm)?, num(ss)?),
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
    let mut it = line.strip_prefix('|')?.split_whitespace();
    let rank = it.next()?;
    if rank.len() != 1 || !rank.as_bytes()[0].is_ascii_digit() {
        return None;
    }
    let mut cells = [b'-'; 8];
    for cell in &mut cells {
        *cell = match it.next()? {
            "-" => b'-',
            "*" => b'X',
            "O" => b'O',
            _ => return None,
        };
    }
    if it.next()? != rank || it.next().is_some() {
        return None;
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
        assert_eq!(
            parse_server_chunk(chunk, "alice"),
            OsEvent::MatchCreated {
                id: ".5".to_string(),
                opponent: Some("bob".to_string()),
                variant: Some("8b".to_string()),
            }
        );
    }

    #[test]
    fn unknown_role_when_username_not_in_match() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: + match .5");
        let ev = parse_server_chunk(chunk, "charlie");
        let OsEvent::MatchCreated { id, opponent, .. } = ev else {
            panic!("expected MatchCreated, got {ev:?}")
        };
        assert_eq!(id, ".5");
        assert!(opponent.is_none());
    }

    #[test]
    fn match_state_join_is_parsed() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: join .5");
        assert_eq!(
            parse_server_chunk(chunk, "alice"),
            OsEvent::MatchState(MatchState {
                id: ".5".to_string(),
                variant: Some("8b".to_string()),
                opponent: Some("bob".to_string()),
                board: Board::new(),
                side_to_move: Disc::Black,
                my_color: Some(Disc::Black),
                my_clock: Some(PlayerClock {
                    main_ms: 59_000,
                    increment_ms: 0,
                }),
                clocks: vec![
                    DisplayClock {
                        name: "alice".to_string(),
                        color: Disc::Black,
                        main_ms: Some(59_000),
                    },
                    DisplayClock {
                        name: "bob".to_string(),
                        color: Disc::White,
                        main_ms: Some(60_000),
                    },
                ],
                last_move: Some("PASS".to_string()),
            })
        );
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
        let OsEvent::MatchState(MatchState {
            id,
            side_to_move,
            my_clock,
            board,
            ..
        }) = ev
        else {
            panic!("expected MatchState, got {ev:?}")
        };
        assert_eq!(id, ".5");
        assert_eq!(side_to_move, Disc::White);
        assert_eq!(
            board,
            Board::from_bitboards(1 << 27, (1 << 28) | (1 << 35) | (1 << 36) | (1 << 37))
        );
        assert_eq!(
            my_clock,
            Some(PlayerClock {
                main_ms: 56_000,
                increment_ms: 0,
            })
        );
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
        let OsEvent::MatchState(MatchState {
            board,
            my_clock,
            clocks,
            last_move,
            ..
        }) = ev
        else {
            panic!("expected MatchState, got {ev:?}")
        };
        assert_eq!(board.disc_at(Square::A1, Disc::Black), Disc::Black);
        assert_eq!(my_clock.map(|clock| clock.main_ms), Some(56_000));
        assert_eq!(last_move.as_deref(), Some("f6"));
        let clocks: Vec<_> = clocks
            .iter()
            .map(|clock| (clock.name.as_str(), clock.color, clock.main_ms))
            .collect();
        assert_eq!(
            clocks,
            [
                ("alice", Disc::Black, Some(56_000)),
                ("bob", Disc::White, Some(58_000)),
            ]
        );
    }

    #[test]
    fn match_end_is_parsed() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: - match .5");
        assert_eq!(
            parse_server_chunk(chunk, "alice"),
            OsEvent::MatchEnd {
                id: ".5".to_string(),
                result: "-64.00".to_string(),
                score: Some(-64.0),
            }
        );
    }

    #[test]
    fn rated_match_end_score_ignores_counter() {
        assert_eq!(
            parse_server_chunk(
                &["/os: - match .9 2544 lynx 2505 kitty 8 R +30.00  12345"],
                "kitty",
            ),
            OsEvent::MatchEnd {
                id: ".9".to_string(),
                result: "+30.00 12345".to_string(),
                score: Some(-30.0),
            }
        );
    }

    #[test]
    fn interrupted_match_end_is_parsed() {
        assert_eq!(
            parse_server_chunk(
                &["/os: - match .5 1720 alice 1720 bob 8b U alice left"],
                "alice",
            ),
            OsEvent::MatchEnd {
                id: ".5".to_string(),
                result: "alice left".to_string(),
                score: None,
            }
        );
    }

    #[test]
    fn numeric_login_leaving_is_not_a_score() {
        for line in [
            "/os: - match .5 1720 1234 1720 bob 8 R 1234 left 12345",
            "/os: - match .5 1720 1234 1720 bob 8b U 1234 left",
        ] {
            let OsEvent::MatchEnd { score, .. } = parse_server_chunk(&[line], "bob") else {
                panic!("not a match end: {line}");
            };
            assert_eq!(score, None, "{line}");
        }
    }

    #[test]
    fn synchro_child_end_is_parsed_as_match_end() {
        assert_eq!(
            parse_server_chunk(&["/os: end .5.1 0.00"], "alice"),
            OsEvent::MatchEnd {
                id: ".5.1".to_string(),
                result: "0.00".to_string(),
                score: None,
            }
        );
        assert_eq!(
            parse_server_chunk(&["/os: end .5.0 ( bob vs. alice ) +4.00 "], "alice"),
            OsEvent::MatchEnd {
                id: ".5.0".to_string(),
                result: "( bob vs. alice ) +4.00".to_string(),
                score: Some(-4.0),
            }
        );
    }

    #[test]
    fn kibitz_is_parsed() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: .5 A");
        assert_eq!(
            parse_server_chunk(chunk, "alice"),
            OsEvent::Kibitz {
                id: ".5".to_string(),
                from: "bob".to_string(),
                message: "good game".to_string(),
            }
        );
    }

    #[test]
    fn kibitz_with_padded_rating_is_parsed() {
        assert_eq!(
            parse_server_chunk(&["/os: .5 A    0 bob: hello"], "alice"),
            OsEvent::Kibitz {
                id: ".5".to_string(),
                from: "bob".to_string(),
                message: "hello".to_string(),
            }
        );
    }

    #[test]
    fn ask_offer_is_parsed_from_sample() {
        // Sample line:
        //   /os: +   .8 1720.0 alice    01:00//02:00       8b U 1720.0 bob
        // alice is the challenger (p1), bob is the challengee (p2).
        let all = chunks();
        let chunk = find_chunk(&all, "/os: +   .");
        assert_eq!(
            parse_server_chunk(chunk, "bob"),
            OsEvent::AskOffer(AskOffer {
                id: ".8".to_string(),
                from: "alice".to_string(),
                to: "bob".to_string(),
                game_type: "8b".to_string(),
                clock: "01:00//02:00".to_string(),
                mode: "U".to_string(),
            })
        );
    }

    #[test]
    fn ask_offer_uses_our_asymmetric_clock() {
        let ev = parse_server_chunk(
            &["/os: + .8 1720 alice 01:00//02:00 8b U 1720 bob 00:30,N0//01:00"],
            "bob",
        );
        assert!(
            matches!(ev, OsEvent::AskOffer(AskOffer { clock, .. }) if clock == "00:30,N0//01:00")
        );
    }

    #[test]
    fn ask_withdrawal_is_parsed() {
        // `- <padding> .id ...` is an ask withdrawal (or the server's
        // post-accept cleanup). The session needs to observe it as a distinct
        // event, not classify it as Unknown.
        assert_eq!(
            parse_server_chunk(
                &["/os: -   .8 1720.0 alice    01:00//02:00       8b U 1720.0 bob"],
                "bob",
            ),
            OsEvent::AskWithdrawn {
                id: ".8".to_string()
            }
        );
    }

    #[test]
    fn clock_disambiguation_when_my_username_is_second_player() {
        // Alice is listed first with a different clock and bob second. With
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
        let OsEvent::MatchState(MatchState {
            my_color, my_clock, ..
        }) = ev
        else {
            panic!("expected MatchState, got {ev:?}")
        };
        assert_eq!(my_color, Some(Disc::White));
        // bob's clock, NOT alice's 45_000/120_000.
        assert_eq!(
            my_clock,
            Some(PlayerClock {
                main_ms: 59_000,
                increment_ms: 0,
            })
        );
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
        let OsEvent::MatchState(MatchState {
            my_color, my_clock, ..
        }) = ev
        else {
            panic!("expected MatchState, got {ev:?}")
        };
        assert_eq!(my_color, Some(Disc::Black));
        assert_eq!(my_clock, None);
    }

    #[test]
    fn clock_fields_are_parsed() {
        for (fields, main_ms, increment_ms) in [
            ("120,0:0/5,0:1/0,0:0", 120_000, 5_000),
            ("05:00,0:0/00:05,0:1/0:00,0:0", 300_000, 5_000),
            ("01:00:00,0:0//05:00,0:0", 3_600_000, 0),
            ("1.00:00:00,0:0//05:00,0:0", 86_400_000, 0),
            ("-00:05,0:0//02:00,0:0", 0, 0),
        ] {
            assert_eq!(
                parse_player_clock(fields),
                Some(PlayerClock {
                    main_ms,
                    increment_ms,
                })
            );
        }
    }

    #[test]
    fn non_fischer_increments_are_not_exposed_as_per_move_time() {
        assert_eq!(parse_fischer_increment_field("00:05,N0:1"), Some(0));
        assert_eq!(parse_fischer_increment_field("00:05,0:2"), Some(0));
        assert_eq!(parse_fischer_increment_field("00:05,3:1"), Some(5_000));
    }

    #[test]
    fn clock_parser_accepts_parenthesis_in_login() {
        let clock = parse_clock_line("|a(b      (1720.0 *) 00:10,0:0//02:00,0:0", "a(b")
            .expect("clock line");
        assert!(clock.is_mine);
        assert_eq!(clock.color, Disc::Black);
    }

    #[test]
    fn kibitz_observer_and_player_tags_are_parsed() {
        // `tell /os o .5 ...` emits `O`; `tell /os p .5 ...` emits `P`.
        // Both should map to Kibitz, not Unknown.
        for tag in ["O", "P"] {
            let line = format!("/os: .5 {tag} 1720 bob: hello");
            assert_eq!(
                parse_server_chunk(&[line.as_str()], "alice"),
                OsEvent::Kibitz {
                    id: ".5".to_string(),
                    from: "bob".to_string(),
                    message: "hello".to_string(),
                },
                "tag {tag}"
            );
        }
    }

    #[test]
    fn kibitz_unknown_tag_is_classified_unknown() {
        assert_eq!(
            parse_server_chunk(&["/os: .5 Z 1720 bob: hello"], "alice"),
            OsEvent::Unknown
        );
    }

    #[test]
    fn server_error_chunks_are_parsed() {
        let cases: &[(&str, &str, &str)] = &[
            ("/os: error .5 illegal move", ".5", "illegal move"),
            ("/os: error .5 not your turn", ".5", "not your turn"),
            ("/os: error .5", ".5", ""),
        ];
        for &(chunk, want_id, want_msg) in cases {
            assert_eq!(
                parse_server_chunk(&[chunk], "alice"),
                OsEvent::ServerError {
                    id: want_id.to_string(),
                    message: want_msg.to_string(),
                },
                "chunk: {chunk}"
            );
        }
    }

    #[test]
    fn request_error_chunks_are_parsed() {
        let cases = [
            (
                "/os: ERR your request doesn't fit the opponent's formula.",
                None,
                "your request doesn't fit the opponent's formula.",
            ),
            (
                "/os: ERR request '.8' not found.",
                Some(".8"),
                "request '.8' not found.",
            ),
            (
                "/os: ERR you cannot accept your own match request.",
                None,
                "you cannot accept your own match request.",
            ),
        ];
        for (chunk, want_id, want_message) in cases {
            assert_eq!(
                parse_server_chunk(&[chunk], "alice"),
                OsEvent::RequestError {
                    id: want_id.map(str::to_string),
                    message: want_message.to_string(),
                },
                "chunk: {chunk}"
            );
        }
    }

    #[test]
    fn malformed_chunk_does_not_panic() {
        assert_eq!(parse_server_chunk(&[], "alice"), OsEvent::Unknown);
        assert_eq!(parse_server_chunk(&["garbage"], "alice"), OsEvent::Unknown);
        assert_eq!(
            parse_server_chunk(&["/os: join"], "alice"),
            OsEvent::Unknown
        );
        // Header looks like update but no body at all.
        assert!(matches!(
            parse_server_chunk(&["/os: update .5 8b K?"], "alice"),
            OsEvent::MatchStateUnavailable { id, .. } if id == ".5"
        ));
    }

    #[test]
    fn compact_match_state_payload_is_not_a_variant() {
        for header in ["/os: join .5.0 (;GM[Othello]TY[s8])", "/os: update .5.1 f5"] {
            assert!(matches!(
                parse_server_chunk(&[header], "alice"),
                OsEvent::MatchStateUnavailable { id, variant: None, .. }
                    if id == if header.contains(".5.0") { ".5.0" } else { ".5.1" }
            ));
        }
    }

    #[test]
    fn unparseable_state_naming_only_other_players_is_spectated() {
        let chunk = [
            "/os: join .9 10 K?",
            "|bob      (1720.0 *) 01:00//00:30",
            "|carol    (1720.0 O) 01:00//00:30",
            "| 1 - - - - - - - - - - 1",
        ];
        assert_eq!(
            parse_server_chunk(&chunk, "alice"),
            OsEvent::SpectatedMatchState {
                id: ".9".to_string(),
                variant: Some("10".to_string()),
            }
        );
    }

    #[test]
    fn complete_state_naming_only_other_players_is_spectated() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: join .5");
        assert!(matches!(
            parse_server_chunk(chunk, "dave"),
            OsEvent::SpectatedMatchState { id, .. } if id == ".5"
        ));
    }

    #[test]
    fn bell_and_padding_before_join_are_ignored() {
        let all = chunks();
        let chunk = find_chunk(&all, "/os: join .5");
        let mut lines: Vec<String> = chunk.iter().map(|line| (*line).to_string()).collect();
        lines[0] = lines[0].replacen("/os: join ", "/os: \x07   join ", 1);
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        assert_eq!(
            parse_server_chunk(&refs, "alice"),
            parse_server_chunk(chunk, "alice")
        );
    }

    #[test]
    fn oversized_board_row_is_rejected() {
        assert!(
            parse_board_row("| 1 - - - - - - - - - - 1").is_none(),
            "10x10 rows must not be truncated into 8x8 rows"
        );
    }

    #[test]
    fn multiple_messages_before_ready_are_all_parsed() {
        let events = parse_server_messages(
            &[
                "/os: + match .5 1720 alice 1720 bob 8b U",
                "/os: + match .7 1720 alice 1720 carol 8b U",
            ],
            "alice",
        );

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], OsEvent::MatchCreated { id, .. } if id == ".5"));
        assert!(matches!(&events[1], OsEvent::MatchCreated { id, .. } if id == ".7"));
    }

    #[test]
    fn bell_prefixed_event_is_parsed() {
        let event = parse_server_chunk(&["/os: \x07+ match .5 1720 alice 1720 bob 8b U"], "alice");
        assert!(matches!(event, OsEvent::MatchCreated { id, .. } if id == ".5"));
    }

    #[test]
    fn self_play_uses_the_side_to_move_clock() {
        for (side_line, expected_color, expected_ms) in [
            ("|* to move", Disc::Black, 10_000),
            ("|O to move", Disc::White, 20_000),
        ] {
            let event = parse_server_chunk(
                &[
                    "/os: join .5 8b U",
                    "|alice (1720.0 *) 00:10//02:00",
                    "|alice (1720.0 O) 00:20//02:00",
                    "| 1 - - - - - - - - 1",
                    "| 2 - - - - - - - - 2",
                    "| 3 - - - - - - - - 3",
                    "| 4 - - - O * - - - 4",
                    "| 5 - - - * O - - - 5",
                    "| 6 - - - - - - - - 6",
                    "| 7 - - - - - - - - 7",
                    "| 8 - - - - - - - - 8",
                    side_line,
                ],
                "alice",
            );

            let OsEvent::MatchState(MatchState {
                my_color, my_clock, ..
            }) = event
            else {
                panic!("expected self-play state, got {event:?}")
            };
            assert_eq!(my_color, Some(expected_color));
            assert_eq!(my_clock.map(|clock| clock.main_ms), Some(expected_ms));
        }
    }
}
