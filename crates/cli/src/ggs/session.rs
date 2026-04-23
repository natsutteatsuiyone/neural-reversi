//! Pure-logic session state machine for GGS `/os` play mode.
//!
//! Consumes [`OsEvent`]s from the parser plus move-ready notifications from
//! the search thread, and produces [`SessionAction`]s describing what the
//! caller should do (send a line, start a search, log a message). All IO,
//! threading, and TCP plumbing happens at higher layers; this module is
//! fully unit-testable.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::probcut::Selectivity;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::square::Square;

use crate::ggs::protocol::{OsEvent, PlayerClock};
use crate::ggs::time;

/// Search constraint requested by the GGS session layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchLimit {
    Time(TimeControlMode),
    Level { level: usize },
}

/// A concrete action the session asks the caller to perform.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionAction {
    /// A single line to send to the GGS server (no trailing newline).
    Send(String),
    /// Kick off a search on another thread. The caller MUST later call
    /// [`Session::on_move_ready`] with the best move (or `None` for pass).
    StartSearch {
        match_id: String,
        board: Board,
        search_limit: SearchLimit,
        selectivity: Selectivity,
    },
    /// Informational; the caller typically prints this to stderr prefixed with `[ggs]`.
    Log(String),
}

/// Private per-match bookkeeping.
struct ActiveMatch {
    id: String,
    /// Our authoritative color keyed by the game id that carried it. For
    /// regular matches this is a single `<parent_id> -> color` entry; for
    /// synchro matches the two child games (`<id>.0`, `<id>.1`) have
    /// *flipped* colors and each gets its own entry populated from the
    /// corresponding chunk's clock line. Looking up by the event's id (not
    /// the parent id) is what lets a single `Session` play both halves of a
    /// synchro match without confusing which side we are on.
    my_colors: HashMap<String, Disc>,
    opponent: String,
    /// Game ids with a search in flight. Set on
    /// [`SessionAction::StartSearch`] emit; cleared in
    /// [`Session::on_move_ready`] when the corresponding result arrives
    /// (whether the move was sent or dropped as stale).
    searching_games: HashSet<String>,
}

const PENDING_ACCEPT_RECOVERY_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Debug, Clone)]
struct PendingAccept {
    id: String,
    withdrawn_at: Option<Instant>,
}

/// Pure-logic GGS session state machine.
pub struct Session {
    my_username: String,
    level: usize,
    selectivity: Selectivity,
    current_match: Option<ActiveMatch>,
    /// Match id for which we have either sent `tell /os accept` or seen our
    /// own outgoing ask echoed back, but not yet seen the corresponding
    /// `+ match <id>` event. Treated as "busy" by `on_ask_offer` so we don't
    /// accept a second offer while another match request is still in flight,
    /// which would let the server create two matches that the single-match
    /// session model can't both play.
    ///
    /// Cleared when `MatchCreated` arrives with us as a player, or when an ask
    /// withdrawal has not been followed by match creation within
    /// [`PENDING_ACCEPT_RECOVERY_TIMEOUT`].
    pending_accept: Option<PendingAccept>,
}

impl Session {
    /// Creates a new session. Infallible.
    pub fn new(my_username: String, level: usize, selectivity: Selectivity) -> Self {
        Self {
            my_username,
            level,
            selectivity,
            current_match: None,
            pending_accept: None,
        }
    }

    /// Returns the username we identify as on the server.
    pub fn my_username(&self) -> &str {
        &self.my_username
    }

    /// Consume one [`OsEvent`] and return the actions that should be performed.
    pub fn on_event(&mut self, event: OsEvent) -> Vec<SessionAction> {
        match event {
            OsEvent::MatchCreated {
                id,
                opponent,
                variant,
            } => self.on_match_created(id, opponent, variant),
            OsEvent::MatchState {
                id,
                side_to_move,
                board,
                my_color,
                my_clock,
                ..
            } => self.on_match_state(id, side_to_move, my_color, board, my_clock),
            OsEvent::MatchEnd { id, result, score } => self.on_match_end(id, result, score),
            OsEvent::Kibitz { id, from, message } => {
                vec![SessionAction::Log(format!("[{id}] {from}: {message}"))]
            }
            OsEvent::AskOffer {
                id,
                from,
                to,
                game_type,
                mode,
            } => self.on_ask_offer(id, from, to, game_type, mode),
            OsEvent::AskWithdrawn { id } => self.on_ask_withdrawn(id),
            OsEvent::ServerError { id, message } => self.on_server_error(id, message),
            OsEvent::Unknown => Vec::new(),
        }
    }

    /// Handle an incoming match-request ("ask") offer.
    ///
    /// Policy:
    /// * If the offer is our own outgoing ask echoed back, treat it as a
    ///   pending match request but do NOT self-accept (the server rejects
    ///   self-accepts anyway; see `EXE_Service::accept` line 807).
    /// * If the offer isn't addressed to us or the global `*` pool, ignore it
    ///   silently as a third-party offer we can't act on.
    /// * If the offer uses an unsupported game type, decline it. This engine
    ///   only understands 8x8 Othello board states.
    /// * If we're already in a match OR have an accept in flight for a
    ///   previous offer, send `decline` to clear the request. Without the
    ///   pending-accept check, a client running `ts open 3+` with multiple
    ///   queued offers would fire `accept` on every chunk before the first
    ///   `+ match` event landed, leaving the extra matches unplayed because
    ///   `Session` can only track one.
    /// * Otherwise, auto-accept and latch the id in `pending_accept` until
    ///   the match materializes.
    fn on_ask_offer(
        &mut self,
        id: String,
        from: String,
        to: String,
        game_type: String,
        mode: String,
    ) -> Vec<SessionAction> {
        self.recover_expired_pending_accept();
        if from == self.my_username.as_str() {
            if is_supported_ask_game(&game_type)
                && self.current_match.is_none()
                && self.pending_accept.is_none()
            {
                self.pending_accept = Some(PendingAccept::new(id));
            }
            return Vec::new();
        }
        if to != self.my_username && to != "*" {
            return Vec::new();
        }
        if !is_supported_ask_game(&game_type) {
            return decline_offer(&id, &from, format!("unsupported game {game_type} {mode}"));
        }
        if let Some(pending) = &self.pending_accept {
            return decline_offer(
                &id,
                &from,
                format!("match request {} still pending", pending.id),
            );
        }
        if self.current_match.is_some() {
            return decline_offer(&id, &from, "already in a match".to_string());
        }
        self.pending_accept = Some(PendingAccept::new(id.clone()));
        vec![
            SessionAction::Send(format!("tell /os accept {id}")),
            SessionAction::Log(format!(
                "accepting offer {id} from {from} ({game_type} {mode})"
            )),
        ]
    }

    /// Ask withdrawals are also emitted as server post-accept cleanup. Mark a
    /// matching pending accept as withdrawn, but keep the busy latch briefly so
    /// the common cleanup-before-`+ match` ordering cannot accept a second game.
    fn on_ask_withdrawn(&mut self, id: String) -> Vec<SessionAction> {
        if let Some(pending) = &mut self.pending_accept
            && pending.id == id
            && pending.withdrawn_at.is_none()
        {
            pending.withdrawn_at = Some(Instant::now());
        }
        Vec::new()
    }

    /// Feed back a completed search result. `match_id` must identify which
    /// match this move belongs to; if it is not the current match (e.g. the
    /// match ended while we were searching) the result is logged and dropped.
    pub fn on_move_ready(&mut self, match_id: &str, mv: Option<Square>) -> Vec<SessionAction> {
        match &mut self.current_match {
            Some(cur) if cur.contains_game(match_id) => {
                if !cur.searching_games.remove(match_id) {
                    return vec![SessionAction::Log(format!(
                        "dropping stale move for match {match_id}"
                    ))];
                }
                let line = match mv {
                    Some(sq) => format!("tell /os play {match_id} {sq}"),
                    None => format!("tell /os play {match_id} PASS"),
                };
                vec![SessionAction::Send(line)]
            }
            _ => vec![SessionAction::Log(format!(
                "dropping stale move for match {match_id}"
            ))],
        }
    }

    /// Clear an in-flight search that terminated without producing a move.
    ///
    /// This is used when the worker panics or otherwise aborts before it can
    /// return a move. The in-flight marker must still be cleared so the next
    /// state update can retry the search instead of wedging permanently behind
    /// a stale "already in flight" flag.
    pub fn on_search_aborted(&mut self, match_id: &str) -> Vec<SessionAction> {
        match &mut self.current_match {
            Some(cur) if cur.contains_game(match_id) => {
                if cur.searching_games.remove(match_id) {
                    Vec::new()
                } else {
                    vec![SessionAction::Log(format!(
                        "dropping stale search failure for match {match_id}"
                    ))]
                }
            }
            _ => vec![SessionAction::Log(format!(
                "dropping stale search failure for match {match_id}"
            ))],
        }
    }

    fn on_match_created(
        &mut self,
        id: String,
        opponent: Option<String>,
        variant: Option<String>,
    ) -> Vec<SessionAction> {
        // Spectator vs player is resolved by whether our username appears in
        // the `+ match` name list (populated by `parse_match_created`).
        let opponent_name = match opponent {
            Some(n) => n,
            None => {
                return vec![SessionAction::Log(format!("ignoring spectator match {id}"))];
            }
        };

        // The pending accept (if any) resolves as soon as we see a match
        // we're playing in, regardless of whether the ids match — a stale
        // latch from a race must not block later offers.
        self.pending_accept = None;

        if let Some(cur) = &self.current_match {
            if cur.id == id {
                return vec![SessionAction::Log(format!(
                    "duplicate match-created for {id}"
                ))];
            }
            return vec![SessionAction::Log(format!(
                "ignoring second match {id}; already in {}",
                cur.id
            ))];
        }

        let level = self.level;
        let variant_label = variant.as_deref().unwrap_or("?");
        let kind = if variant.as_deref().is_some_and(is_synchro_variant) {
            "synchro "
        } else {
            ""
        };
        let log = SessionAction::Log(format!(
            "{kind}match {id} started vs {opponent_name} ({variant_label}, level {level})"
        ));
        self.current_match = Some(ActiveMatch {
            id,
            my_colors: HashMap::new(),
            opponent: opponent_name,
            searching_games: HashSet::new(),
        });
        vec![log]
    }

    fn on_match_state(
        &mut self,
        id: String,
        side_to_move: Disc,
        my_color: Option<Disc>,
        board: Board,
        my_clock: Option<PlayerClock>,
    ) -> Vec<SessionAction> {
        let cur = match &mut self.current_match {
            Some(c) => c,
            None => {
                return vec![SessionAction::Log(format!(
                    "ignoring state for match {id}; no current match"
                ))];
            }
        };

        if !cur.contains_game(&id) {
            return vec![SessionAction::Log(format!(
                "ignoring state for non-current match {id}"
            ))];
        }

        if let Some(actual_color) = my_color {
            cur.my_colors.insert(id.clone(), actual_color);
        }

        // If we have not yet seen our authoritative color for THIS game id
        // (no parseable clock line on our side in the current chunk), refuse
        // to search. Acting on a guess could send an illegal move for
        // variants like `8w` where the challenger plays White, and for
        // synchro matches where our color is flipped between `.id.0` and
        // `.id.1` — stored-per-match state from the other child game would
        // mislead us here.
        let Some(&my_color) = cur.my_colors.get(&id) else {
            return vec![SessionAction::Log(format!(
                "ignoring state for match {id}; our color not yet known"
            ))];
        };

        if side_to_move != my_color {
            return Vec::new();
        }

        if !cur.searching_games.insert(id.clone()) {
            return vec![SessionAction::Log(format!(
                "search already in flight for match {id}, skipping duplicate state"
            ))];
        }

        let search_limit = match my_clock {
            Some(clock) => SearchLimit::Time(time::derive_mode(
                clock.main_ms,
                clock.increment_ms,
                clock.byoyomi_ms,
            )),
            None => SearchLimit::Level { level: self.level },
        };
        vec![SessionAction::StartSearch {
            match_id: id,
            board,
            search_limit,
            selectivity: self.selectivity,
        }]
    }

    fn on_match_end(
        &mut self,
        id: String,
        result: String,
        score: Option<f64>,
    ) -> Vec<SessionAction> {
        let Some(cur) = self.current_match.as_mut() else {
            return log_non_current_match_end(&id);
        };
        if !cur.contains_game(&id) {
            return log_non_current_match_end(&id);
        }
        let result_text = match score {
            Some(score) => format!("score {score}"),
            None => format!("result {result}"),
        };
        let opponent = cur.opponent.clone();
        if cur.id == id {
            self.current_match = None;
        } else {
            // Synchro child game (`.<parent>.<n>`): the parent match persists.
            cur.searching_games.remove(&id);
        }
        vec![SessionAction::Log(format!(
            "match {id} ended vs {opponent}, {result_text}"
        ))]
    }

    /// Handle a `/os: error <id> <message>` chunk. The server rejects moves
    /// and other commands with this payload. For an auto-play bot the most
    /// common trigger is an illegal move — which, if not surfaced, would
    /// leave the bot silently waiting to time out. Clearing the in-flight
    /// search flag for the offending match lets the next state update
    /// retry rather than permanently wedging behind a stale flag.
    fn on_server_error(&mut self, id: String, message: String) -> Vec<SessionAction> {
        if let Some(cur) = self.current_match.as_mut()
            && cur.contains_game(&id)
        {
            cur.searching_games.remove(&id);
        }
        vec![SessionAction::Log(format!(
            "server error for {id}: {message}"
        ))]
    }

    fn recover_expired_pending_accept(&mut self) {
        if self
            .pending_accept
            .as_ref()
            .is_some_and(PendingAccept::is_expired_withdrawal)
        {
            self.pending_accept = None;
        }
    }
}

impl PendingAccept {
    fn new(id: String) -> Self {
        Self {
            id,
            withdrawn_at: None,
        }
    }

    fn is_expired_withdrawal(&self) -> bool {
        self.withdrawn_at
            .is_some_and(|withdrawn_at| withdrawn_at.elapsed() >= PENDING_ACCEPT_RECOVERY_TIMEOUT)
    }
}

/// Ask-offer variant filter.
///
/// Accepted:
/// * plain 8x8 (`8`, `8b`, `8w`) — the baseline case,
/// * synchro 8x8 (`s8`, `s8b`, `s8w`) — two concurrent child games with
///   flipped colors; the session identifies them via `.id.0` / `.id.1`,
/// * random-start 8x8 (`8r<N>`, `s8r<N>`, and the `b` / `w` color hints
///   thereof) — the engine just searches from whatever initial board the
///   server sends in the `join` chunk.
///
/// Rejected:
/// * komi variants (contain `k`) — komi-move negotiation is a separate
///   opening protocol we don't implement,
/// * anti variants (contain `a`) — our eval network is trained on normal
///   Othello and would play badly.
fn is_supported_ask_game(game_type: &str) -> bool {
    // Strip optional synchro prefix.
    let rest = game_type.strip_prefix('s').unwrap_or(game_type);
    // Board size must be 8.
    let Some(rest) = rest.strip_prefix('8') else {
        return false;
    };
    // Optional initial-color hint `b` / `w`.
    let rest = rest
        .strip_prefix('b')
        .or_else(|| rest.strip_prefix('w'))
        .unwrap_or(rest);
    // Nothing more, or an `r<digits>` random-start suffix.
    match rest.strip_prefix('r') {
        None => rest.is_empty(),
        Some(digits) => !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()),
    }
}

/// Returns `true` when `variant` is a synchro variant (leading `s` qualifier,
/// e.g. `s8b`, `s8r20`). Synchro matches pair two child games with flipped
/// colors; the session tags its match-created log so operators can tell the
/// two modes apart at a glance.
fn is_synchro_variant(variant: &str) -> bool {
    variant.starts_with('s')
}

/// Emit the (`decline` line, log) pair for a refused ask offer. The reason
/// fragment becomes the tail of the log message.
fn decline_offer(id: &str, from: &str, reason: String) -> Vec<SessionAction> {
    vec![
        SessionAction::Send(format!("tell /os decline {id}")),
        SessionAction::Log(format!("declining offer {id} from {from}; {reason}")),
    ]
}

fn log_non_current_match_end(id: &str) -> Vec<SessionAction> {
    vec![SessionAction::Log(format!(
        "match-end for non-current match {id}"
    ))]
}

impl ActiveMatch {
    fn contains_game(&self, id: &str) -> bool {
        id == self.id
            || id
                .strip_prefix(self.id.as_str())
                .is_some_and(|suffix| suffix.starts_with('.'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_session() -> Session {
        Session::new("alice".to_string(), 21, Selectivity::from_u8(0))
    }

    fn match_created(id: &str, opponent: Option<&str>) -> OsEvent {
        match_created_with_variant(id, opponent, Some("8b"))
    }

    fn match_created_with_variant(
        id: &str,
        opponent: Option<&str>,
        variant: Option<&str>,
    ) -> OsEvent {
        OsEvent::MatchCreated {
            id: id.to_string(),
            opponent: opponent.map(|s| s.to_string()),
            variant: variant.map(|s| s.to_string()),
        }
    }

    fn match_state(id: &str, side_to_move: Disc, my_time_ms: u64, my_byoyomi_ms: u64) -> OsEvent {
        match_state_with_clock(
            id,
            side_to_move,
            Some(PlayerClock {
                main_ms: my_time_ms,
                increment_ms: 0,
                byoyomi_ms: my_byoyomi_ms,
            }),
        )
    }

    fn match_state_with_color(
        id: &str,
        side_to_move: Disc,
        my_color: Option<Disc>,
        my_clock: Option<PlayerClock>,
    ) -> OsEvent {
        OsEvent::MatchState {
            id: id.to_string(),
            board: Board::new(),
            side_to_move,
            my_color,
            my_clock,
        }
    }

    fn match_state_with_clock(
        id: &str,
        side_to_move: Disc,
        my_clock: Option<PlayerClock>,
    ) -> OsEvent {
        match_state_with_color(id, side_to_move, None, my_clock)
    }

    /// Drive `MatchCreated` and then — simulating the authoritative clock
    /// line of the first state chunk — resolve our color for the parent
    /// match id. This mirrors the production flow where color is only known
    /// after the first `join` chunk is parsed, but keeps existing non-synchro
    /// tests concise.
    fn enter_match(session: &mut Session, id: &str, color: Disc, opponent: &str) {
        let _ = session.on_event(match_created(id, Some(opponent)));
        if let Some(cur) = session.current_match.as_mut() {
            cur.my_colors.insert(id.to_string(), color);
        }
    }

    #[test]
    fn match_created_transitions_to_in_match() {
        let mut s = new_session();
        let actions = s.on_event(match_created(".5", Some("bob")));
        assert!(s.current_match.is_some());
        assert_eq!(s.current_match.as_ref().unwrap().id, ".5");
        // Color is intentionally unresolved until the first state chunk.
        assert!(s.current_match.as_ref().unwrap().my_colors.is_empty());
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::Log(msg) => {
                assert!(msg.contains(".5"), "log missing id: {msg}");
                assert!(msg.contains("bob"), "log missing opponent: {msg}");
                // Log must not claim a color — the `+ match` header can't
                // tell us (variant-dependent in GGS Othello).
                assert!(
                    !msg.contains("Black") && !msg.contains("White"),
                    "log should not claim a color yet: {msg}"
                );
            }
            other => panic!("expected Log, got {other:?}"),
        }
    }

    #[test]
    fn match_created_while_busy_logs_warning_and_keeps_current() {
        let mut s = new_session();
        let _ = s.on_event(match_created(".5", Some("bob")));
        let actions = s.on_event(match_created(".7", Some("carol")));
        assert_eq!(s.current_match.as_ref().unwrap().id, ".5");
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::Log(msg) => {
                assert!(msg.contains(".7"), "log missing new id: {msg}");
                assert!(msg.contains(".5"), "log missing current id: {msg}");
            }
            other => panic!("expected Log, got {other:?}"),
        }
    }

    #[test]
    fn match_state_on_my_turn_emits_start_search() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::StartSearch {
                match_id,
                search_limit,
                ..
            } => {
                assert_eq!(match_id, ".5");
                assert_eq!(
                    *search_limit,
                    SearchLimit::Time(TimeControlMode::JapaneseByo {
                        main_time_ms: 56_000,
                        time_per_move_ms: 120_000,
                    })
                );
            }
            other => panic!("expected StartSearch, got {other:?}"),
        }
    }

    #[test]
    fn match_state_missing_clock_uses_level_fallback() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_state_with_clock(".5", Disc::Black, None));
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::StartSearch {
                match_id,
                search_limit,
                ..
            } => {
                assert_eq!(match_id, ".5");
                assert_eq!(*search_limit, SearchLimit::Level { level: 21 });
            }
            other => panic!("expected StartSearch, got {other:?}"),
        }
    }

    #[test]
    fn match_state_on_opponent_turn_emits_nothing() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_state(".5", Disc::White, 56_000, 120_000));
        assert!(actions.is_empty(), "expected empty, got {actions:?}");
    }

    #[test]
    fn match_state_color_overrides_match_created_order_hint() {
        let mut s = new_session();
        // `+ match` lists request players, not necessarily black then white.
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_state_with_color(
            ".5",
            Disc::White,
            Some(Disc::White),
            Some(PlayerClock {
                main_ms: 56_000,
                increment_ms: 0,
                byoyomi_ms: 120_000,
            }),
        ));

        assert_eq!(
            s.current_match.as_ref().unwrap().my_colors.get(".5"),
            Some(&Disc::White)
        );
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::StartSearch { .. }));
    }

    #[test]
    fn synchro_child_game_id_is_accepted_and_sent_back() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_state_with_color(
            ".5.1",
            Disc::White,
            Some(Disc::White),
            Some(PlayerClock {
                main_ms: 56_000,
                increment_ms: 0,
                byoyomi_ms: 120_000,
            }),
        ));

        match &actions[0] {
            SessionAction::StartSearch { match_id, .. } => assert_eq!(match_id, ".5.1"),
            other => panic!("expected StartSearch, got {other:?}"),
        }

        let actions = s.on_move_ready(".5.1", Some(Square::F5));
        assert_eq!(
            actions,
            vec![SessionAction::Send("tell /os play .5.1 f5".to_string())]
        );
    }

    #[test]
    fn child_match_end_invalidates_child_search_and_drops_stale_move() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let search = s.on_event(match_state_with_color(
            ".5.1",
            Disc::White,
            Some(Disc::White),
            Some(PlayerClock {
                main_ms: 56_000,
                increment_ms: 0,
                byoyomi_ms: 120_000,
            }),
        ));
        assert!(matches!(&search[0], SessionAction::StartSearch { .. }));

        let ended = s.on_event(OsEvent::MatchEnd {
            id: ".5.1".to_string(),
            result: "0.00".to_string(),
            score: Some(0.0),
        });
        assert!(s.current_match.is_some());
        assert!(
            !s.current_match
                .as_ref()
                .unwrap()
                .searching_games
                .contains(".5.1")
        );
        assert!(matches!(&ended[0], SessionAction::Log(msg) if msg.contains(".5.1")));

        let stale = s.on_move_ready(".5.1", Some(Square::F5));
        assert_eq!(stale.len(), 1);
        assert!(matches!(&stale[0], SessionAction::Log(_)));
        assert!(
            !stale
                .iter()
                .any(|action| matches!(action, SessionAction::Send(_)))
        );
    }

    #[test]
    fn synchro_child_games_track_flipped_colors_independently() {
        // Regression: synchro matches pair `.<id>.0` (we play one color) with
        // `.<id>.1` (we play the opposite color). A single per-match color
        // field would overwrite one child's color with the other's, causing
        // missed turns or wrong-color searches. Per-game color storage lets
        // both halves run correctly in parallel.
        let mut s = new_session();
        let log = s.on_event(match_created_with_variant(".5", Some("bob"), Some("s8r20")));
        match &log[0] {
            SessionAction::Log(msg) => {
                assert!(msg.contains("synchro"), "log should flag synchro: {msg}");
                assert!(msg.contains("s8r20"), "log should include variant: {msg}");
            }
            other => panic!("expected synchro-aware Log, got {other:?}"),
        }

        // `.5.0`: we are Black. Same side to move → search it.
        let first = s.on_event(match_state_with_color(
            ".5.0",
            Disc::Black,
            Some(Disc::Black),
            Some(PlayerClock {
                main_ms: 60_000,
                increment_ms: 0,
                byoyomi_ms: 120_000,
            }),
        ));
        match &first[0] {
            SessionAction::StartSearch { match_id, .. } => assert_eq!(match_id, ".5.0"),
            other => panic!("expected StartSearch for .5.0, got {other:?}"),
        }

        // `.5.1`: we are White. Same side to move → search it without
        // clobbering the `.5.0` color.
        let second = s.on_event(match_state_with_color(
            ".5.1",
            Disc::White,
            Some(Disc::White),
            Some(PlayerClock {
                main_ms: 60_000,
                increment_ms: 0,
                byoyomi_ms: 120_000,
            }),
        ));
        match &second[0] {
            SessionAction::StartSearch { match_id, .. } => assert_eq!(match_id, ".5.1"),
            other => panic!("expected StartSearch for .5.1, got {other:?}"),
        }

        let cur = s.current_match.as_ref().unwrap();
        assert_eq!(cur.my_colors.get(".5.0"), Some(&Disc::Black));
        assert_eq!(cur.my_colors.get(".5.1"), Some(&Disc::White));
        assert!(cur.searching_games.contains(".5.0"));
        assert!(cur.searching_games.contains(".5.1"));

        // Per-game send should use the child id, not the parent.
        let send0 = s.on_move_ready(".5.0", Some(Square::F5));
        assert_eq!(
            send0,
            vec![SessionAction::Send("tell /os play .5.0 f5".to_string())]
        );
        let send1 = s.on_move_ready(".5.1", Some(Square::F5));
        assert_eq!(
            send1,
            vec![SessionAction::Send("tell /os play .5.1 f5".to_string())]
        );
    }

    #[test]
    fn synchro_child_with_unparseable_clock_does_not_inherit_sibling_color() {
        // In a synchro match, if one child's clock line is unparseable, we
        // must NOT fall back to the other child's color — that would flip
        // our view and either miss a turn or fabricate an illegal move.
        let mut s = new_session();
        let _ = s.on_event(match_created_with_variant(".5", Some("bob"), Some("s8")));

        let _ = s.on_event(match_state_with_color(
            ".5.0",
            Disc::Black,
            Some(Disc::Black),
            Some(PlayerClock {
                main_ms: 60_000,
                increment_ms: 0,
                byoyomi_ms: 120_000,
            }),
        ));

        // `.5.1` arrives without a parseable color. We should abstain, not
        // reuse the `.5.0` color.
        let actions = s.on_event(match_state_with_color(
            ".5.1",
            Disc::White,
            /* my_color */ None,
            None,
        ));
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], SessionAction::Log(msg) if msg.contains(".5.1") && msg.contains("color")),
            "expected color-unknown log for .5.1, got {:?}",
            actions[0]
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, SessionAction::StartSearch { .. })),
            "must not start a search on unresolved color, got {actions:?}"
        );
    }

    #[test]
    fn match_state_for_other_match_is_logged_only() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_state(".7", Disc::Black, 56_000, 120_000));
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(msg) if msg.contains(".7")));
    }

    #[test]
    fn match_end_clears_current_match() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(OsEvent::MatchEnd {
            id: ".5".to_string(),
            result: "-64.00".to_string(),
            score: Some(-64.0),
        });
        assert!(s.current_match.is_none());
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::Log(msg) => {
                assert!(msg.contains("-64"), "log missing score: {msg}");
            }
            other => panic!("expected Log, got {other:?}"),
        }
    }

    #[test]
    fn match_end_for_other_match_is_logged_only() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(OsEvent::MatchEnd {
            id: ".7".to_string(),
            result: "12.00".to_string(),
            score: Some(12.0),
        });
        assert_eq!(s.current_match.as_ref().unwrap().id, ".5");
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(_)));
    }

    #[test]
    fn interrupted_match_end_clears_current_match() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(OsEvent::MatchEnd {
            id: ".5".to_string(),
            result: "alice left".to_string(),
            score: None,
        });

        assert!(s.current_match.is_none());
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(msg) if msg.contains("alice left")));
    }

    #[test]
    fn spectator_match_created_is_ignored() {
        let mut s = new_session();
        let actions = s.on_event(match_created(".9", None));
        assert!(s.current_match.is_none());
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(msg) if msg.contains(".9")));
    }

    #[test]
    fn move_ready_emits_tell_os_play() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        let actions = s.on_move_ready(".5", Some(Square::F5));
        assert_eq!(
            actions,
            vec![SessionAction::Send("tell /os play .5 f5".to_string())]
        );
    }

    #[test]
    fn move_ready_pass_emits_tell_os_play_pass() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        let actions = s.on_move_ready(".5", None);
        assert_eq!(
            actions,
            vec![SessionAction::Send("tell /os play .5 PASS".to_string())]
        );
    }

    #[test]
    fn stale_move_ready_is_logged_and_dropped() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(OsEvent::MatchEnd {
            id: ".5".to_string(),
            result: "-64.00".to_string(),
            score: Some(-64.0),
        });
        let actions = s.on_move_ready(".5", Some(Square::F5));
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::Log(msg) => assert!(msg.contains(".5")),
            other => panic!("expected Log, got {other:?}"),
        }
        assert!(!actions.iter().any(|a| matches!(a, SessionAction::Send(_))));
    }

    #[test]
    fn kibitz_is_logged() {
        let mut s = new_session();
        let actions = s.on_event(OsEvent::Kibitz {
            id: ".5".to_string(),
            from: "bob".to_string(),
            message: "gg".to_string(),
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::Log(msg) => {
                assert!(msg.contains(".5"));
                assert!(msg.contains("bob"));
                assert!(msg.contains("gg"));
            }
            other => panic!("expected Log, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_match_state_while_searching_does_not_re_emit_start_search() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");

        // First MatchState on our turn emits StartSearch exactly once.
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::StartSearch { .. }));

        // A second identical MatchState arrives before on_move_ready fires.
        // We must NOT re-emit StartSearch.
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::Log(msg) => {
                assert!(
                    msg.contains("already in flight") && msg.contains(".5"),
                    "expected dedupe log, got {msg}"
                );
            }
            other => panic!("expected Log, got {other:?}"),
        }

        // After on_move_ready, the flag resets and the next MatchState can
        // kick off a fresh search.
        let sent = s.on_move_ready(".5", Some(Square::F5));
        assert!(matches!(&sent[0], SessionAction::Send(_)));
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], SessionAction::StartSearch { .. }),
            "expected fresh StartSearch after flag reset, got {:?}",
            actions[0]
        );
    }

    #[test]
    fn aborted_search_clears_in_flight_flag_and_allows_retry() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");

        let actions = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::StartSearch { .. }));

        let aborted = s.on_search_aborted(".5");
        assert!(
            aborted.is_empty(),
            "expected current search abort to clear silently, got {aborted:?}"
        );

        let actions = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], SessionAction::StartSearch { .. }),
            "expected retry StartSearch after abort, got {:?}",
            actions[0]
        );
    }

    #[test]
    fn unknown_returns_empty_actions() {
        let mut s = new_session();
        let actions = s.on_event(OsEvent::Unknown);
        assert!(actions.is_empty());
    }

    #[test]
    fn server_error_logs_and_clears_in_flight_search() {
        // Regression: `/os: error .5 illegal move` used to map to Unknown,
        // leaving no diagnostic trace and no way to retry — we'd wait to
        // time out. Now it logs the server message and clears the in-flight
        // flag so the next state update can re-search.
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let first = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert!(matches!(&first[0], SessionAction::StartSearch { .. }));

        let err = s.on_event(OsEvent::ServerError {
            id: ".5".to_string(),
            message: "illegal move".to_string(),
        });
        assert_eq!(err.len(), 1);
        match &err[0] {
            SessionAction::Log(msg) => {
                assert!(msg.contains(".5"), "log missing id: {msg}");
                assert!(msg.contains("illegal move"), "log missing message: {msg}");
            }
            other => panic!("expected Log, got {other:?}"),
        }

        // Subsequent state update must be allowed to restart a search.
        let retry = s.on_event(match_state(".5", Disc::Black, 56_000, 120_000));
        assert!(matches!(&retry[0], SessionAction::StartSearch { .. }));
    }

    #[test]
    fn server_error_for_unknown_match_is_still_logged() {
        let mut s = new_session();
        let actions = s.on_event(OsEvent::ServerError {
            id: ".nonexistent".to_string(),
            message: "not found".to_string(),
        });
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            SessionAction::Log(msg) if msg.contains(".nonexistent")
        ));
    }

    #[test]
    fn match_state_without_resolved_color_does_not_search() {
        // Regression: if a state update arrives before we have an
        // authoritative color (clock line missing or unparseable), we must
        // NOT guess from the `+ match` name ordering. GGS variant `8w`
        // flips which listed player plays Black, so guessing can fabricate
        // illegal moves.
        let mut s = new_session();
        let _ = s.on_event(match_created(".5", Some("bob")));
        assert!(s.current_match.as_ref().unwrap().my_colors.is_empty());
        let actions = s.on_event(match_state_with_color(
            ".5",
            Disc::Black,
            /* my_color */ None,
            Some(PlayerClock {
                main_ms: 56_000,
                increment_ms: 0,
                byoyomi_ms: 120_000,
            }),
        ));
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], SessionAction::Log(msg) if msg.contains(".5") && msg.contains("color")),
            "expected color-unknown log, got {:?}",
            actions[0]
        );
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, SessionAction::StartSearch { .. }))
        );
    }

    fn ask_offer(id: &str, from: &str, to: &str) -> OsEvent {
        ask_offer_with_game(id, from, to, "8b", "U")
    }

    fn pending_accept_id(session: &Session) -> Option<&str> {
        session
            .pending_accept
            .as_ref()
            .map(|pending| pending.id.as_str())
    }

    fn ask_offer_with_game(id: &str, from: &str, to: &str, game_type: &str, mode: &str) -> OsEvent {
        OsEvent::AskOffer {
            id: id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            game_type: game_type.to_string(),
            mode: mode.to_string(),
        }
    }

    #[test]
    fn ask_offer_to_me_is_auto_accepted_when_idle() {
        let mut s = new_session();
        let actions = s.on_event(ask_offer(".8", "bob", "alice"));
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            SessionAction::Send("tell /os accept .8".to_string())
        );
        assert!(
            matches!(&actions[1], SessionAction::Log(msg) if msg.contains(".8") && msg.contains("bob")),
            "expected accept log, got {:?}",
            actions[1]
        );
    }

    #[test]
    fn standard_8x8_ask_types_are_auto_accepted() {
        for game_type in ["8", "8b", "8w"] {
            let mut s = new_session();
            let actions = s.on_event(ask_offer_with_game(".8", "bob", "alice", game_type, "U"));
            assert_eq!(
                actions[0],
                SessionAction::Send("tell /os accept .8".to_string()),
                "game type {game_type} should be accepted"
            );
        }
    }

    #[test]
    fn synchro_and_random_ask_types_are_auto_accepted() {
        for game_type in ["s8", "s8b", "s8w", "8r20", "s8r20", "8br20", "s8wr20"] {
            let mut s = new_session();
            let actions = s.on_event(ask_offer_with_game(".8", "bob", "alice", game_type, "U"));
            assert_eq!(
                actions[0],
                SessionAction::Send("tell /os accept .8".to_string()),
                "game type {game_type} should be accepted"
            );
        }
    }

    #[test]
    fn komi_and_anti_ask_types_are_declined() {
        // Komi requires opening-move negotiation we don't implement; anti
        // inverts scoring, which our eval network isn't trained for.
        for game_type in ["8k", "s8k", "8a", "8br20a", "s8ka", "10b"] {
            let mut s = new_session();
            let actions = s.on_event(ask_offer_with_game(".8", "bob", "alice", game_type, "U"));
            assert_eq!(
                actions[0],
                SessionAction::Send("tell /os decline .8".to_string()),
                "game type {game_type} should be declined"
            );
            assert!(
                s.pending_accept.is_none(),
                "declining unsupported {game_type} must not latch"
            );
        }
    }

    #[test]
    fn global_ask_offer_is_auto_accepted_when_idle() {
        let mut s = new_session();
        let actions = s.on_event(ask_offer_with_game(".8", "bob", "*", "8", "U"));
        assert_eq!(
            actions[0],
            SessionAction::Send("tell /os accept .8".to_string())
        );
    }

    #[test]
    fn ask_offer_to_me_is_declined_when_busy() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "carol");
        let actions = s.on_event(ask_offer(".8", "bob", "alice"));
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            SessionAction::Send("tell /os decline .8".to_string())
        );
        assert!(
            matches!(&actions[1], SessionAction::Log(msg) if msg.contains(".8") && msg.contains("already in a match")),
            "expected decline log, got {:?}",
            actions[1]
        );
    }

    #[test]
    fn unsupported_ask_offer_to_me_is_declined_without_latching() {
        let mut s = new_session();
        let actions = s.on_event(ask_offer_with_game(".8", "bob", "alice", "10b", "U"));
        assert_eq!(actions.len(), 2);
        assert_eq!(
            actions[0],
            SessionAction::Send("tell /os decline .8".to_string())
        );
        assert!(
            matches!(&actions[1], SessionAction::Log(msg) if msg.contains("unsupported game 10b U")),
            "expected unsupported-game decline log, got {:?}",
            actions[1]
        );
        assert!(s.pending_accept.is_none());

        let accepted = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(
            accepted[0],
            SessionAction::Send("tell /os accept .9".to_string())
        );
    }

    #[test]
    fn own_outgoing_ask_offer_latches_busy_without_self_accepting() {
        // If the offer echoes back with us as the challenger (from == me),
        // we MUST NOT accept — the server rejects self-accepts anyway.
        // But the outgoing ask still has to mark the session busy until it
        // resolves, or a different incoming offer could be auto-accepted
        // concurrently and strand one of the two matches.
        let mut s = new_session();
        let actions = s.on_event(ask_offer(".8", "alice", "bob"));
        assert!(actions.is_empty(), "expected no action, got {actions:?}");
        assert_eq!(pending_accept_id(&s), Some(".8"));

        let actions = s.on_event(ask_offer_with_game(".9", "alice", "*", "8", "U"));
        assert!(actions.is_empty(), "expected no action, got {actions:?}");
        assert_eq!(pending_accept_id(&s), Some(".8"));
    }

    #[test]
    fn incoming_offer_is_declined_while_own_outgoing_ask_is_pending() {
        let mut s = new_session();
        let _ = s.on_event(ask_offer(".8", "alice", "bob"));
        assert_eq!(pending_accept_id(&s), Some(".8"));

        let second = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(second.len(), 2);
        assert_eq!(
            second[0],
            SessionAction::Send("tell /os decline .9".to_string())
        );
        assert!(
            matches!(&second[1], SessionAction::Log(msg) if msg.contains(".9") && msg.contains(".8")),
            "expected decline log naming both ids, got {:?}",
            second[1]
        );
    }

    #[test]
    fn third_party_ask_offer_is_ignored() {
        // Broadcast ask between two other players.
        let mut s = new_session();
        let actions = s.on_event(ask_offer(".8", "bob", "carol"));
        assert!(actions.is_empty(), "expected no action, got {actions:?}");
    }

    #[test]
    fn second_ask_offer_is_declined_while_first_accept_is_pending() {
        // Two offers arrive back-to-back before `+ match` lands (realistic
        // with `ts open 3` and a queue of pending requests). We must accept
        // exactly the first and decline the second, or we'd leak an
        // accepted game the single-match session can't play.
        let mut s = new_session();
        let first = s.on_event(ask_offer(".8", "bob", "alice"));
        assert_eq!(
            first[0],
            SessionAction::Send("tell /os accept .8".to_string())
        );
        assert_eq!(pending_accept_id(&s), Some(".8"));

        let second = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(second.len(), 2);
        assert_eq!(
            second[0],
            SessionAction::Send("tell /os decline .9".to_string())
        );
        assert!(
            matches!(&second[1], SessionAction::Log(msg) if msg.contains(".9") && msg.contains(".8")),
            "expected decline log naming both ids, got {:?}",
            second[1]
        );
        // Latch must still hold .8 — the first accept is still the pending one.
        assert_eq!(pending_accept_id(&s), Some(".8"));
    }

    #[test]
    fn pending_accept_is_cleared_on_match_created() {
        let mut s = new_session();
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        assert_eq!(pending_accept_id(&s), Some(".8"));

        let _ = s.on_event(match_created(".8", Some("bob")));
        assert!(s.pending_accept.is_none());
        assert_eq!(s.current_match.as_ref().unwrap().id, ".8");
    }

    #[test]
    fn pending_accept_survives_matching_ask_withdrawal_until_match_created() {
        // `- .id` can be the server's post-accept cleanup before `+ match`, so
        // the busy latch must stay until match creation confirms the accept.
        let mut s = new_session();
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        assert_eq!(pending_accept_id(&s), Some(".8"));

        let _ = s.on_event(OsEvent::AskWithdrawn {
            id: ".8".to_string(),
        });
        assert_eq!(pending_accept_id(&s), Some(".8"));

        let actions = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(
            actions[0],
            SessionAction::Send("tell /os decline .9".to_string())
        );

        let _ = s.on_event(match_created(".8", Some("bob")));
        assert!(s.pending_accept.is_none());
    }

    #[test]
    fn expired_withdrawn_pending_accept_recovers_on_next_offer() {
        let mut s = new_session();
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(OsEvent::AskWithdrawn {
            id: ".8".to_string(),
        });
        let pending = s
            .pending_accept
            .as_mut()
            .expect("test setup should leave a pending accept");
        pending.withdrawn_at = Some(
            Instant::now()
                .checked_sub(PENDING_ACCEPT_RECOVERY_TIMEOUT + Duration::from_millis(1))
                .expect("duration should be representable"),
        );

        let actions = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(
            actions[0],
            SessionAction::Send("tell /os accept .9".to_string())
        );
        assert_eq!(pending_accept_id(&s), Some(".9"));
    }

    #[test]
    fn unrelated_ask_withdrawal_does_not_clear_pending_accept() {
        // Someone else's ask was withdrawn. Our own pending must stay.
        let mut s = new_session();
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(OsEvent::AskWithdrawn {
            id: ".7".to_string(),
        });
        assert_eq!(pending_accept_id(&s), Some(".8"));
    }

    #[test]
    fn spectator_match_created_does_not_clear_pending_accept() {
        // Regression guard: `+ match` notifications for matches we aren't in
        // (we're subscribed via `notify + *` in the init script) must not
        // accidentally free our latch.
        let mut s = new_session();
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(match_created(".42", None));
        assert_eq!(pending_accept_id(&s), Some(".8"));
    }
}
