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
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::square::Square;
use reversi_core::types::Scoref;

use crate::ggs::protocol::{AskOffer, MatchState, OsEvent};

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
    /// [`Session::on_move_ready`] with the best move and score (or `None` for pass).
    StartSearch {
        game_id: String,
        /// Identifies this worker's events and result; GGS recycles game ids.
        search_id: u64,
        board: Board,
        /// Our color on this board.
        color: Disc,
        search_limit: SearchLimit,
    },
    /// Warm the game's transposition table by searching the opponent-to-move
    /// board on the game's own engine. Produces no move; the caller discards
    /// the result and never reports it back.
    StartPonder {
        game_id: String,
        /// Identifies this worker's events, as for [`SessionAction::StartSearch`].
        search_id: u64,
        board: Board,
    },
    /// Abort the ponder search for this game. Always emitted before any
    /// [`SessionAction::StartSearch`] so the search lock is released promptly.
    StopPonder { game_id: String },
    /// Informational; the caller typically prints this to stderr prefixed with `[ggs]`.
    Log(String),
}

/// Private per-match bookkeeping.
struct ActiveMatch {
    id: String,
    /// Number of concurrently searchable boards. Synchro matches use two.
    game_slots: usize,
    /// Our authoritative color keyed by the game id that carried it. For
    /// regular matches this is a single `<parent_id> -> color` entry; for
    /// synchro matches the two child games (`<id>.0`, `<id>.1`) have
    /// *flipped* colors and each gets its own entry populated from the
    /// corresponding chunk's clock line. Looking up by the event's id (not
    /// the parent id) is what lets a single `Session` play both halves of a
    /// synchro match without confusing which side we are on.
    my_colors: HashMap<String, Disc>,
    opponent: String,
    /// Game ids with a search in flight, mapped to that search's id. Set on
    /// [`SessionAction::StartSearch`] emit; cleared in
    /// [`Session::on_move_ready`] when the corresponding result arrives.
    searching_games: HashMap<String, u64>,
    /// Games whose search failed. They remain suppressed until the server
    /// confirms resignation, but no worker is still using their slot.
    failed_games: HashSet<String>,
    /// Synchro children for which `/os: end` arrived before the parent
    /// `- match`; later state messages for them must not restart search.
    ended_games: HashSet<String>,
}

const PENDING_ACCEPT_RECOVERY_TIMEOUT: Duration = Duration::from_secs(2);

/// Main time a GGS search never spends, in milliseconds.
///
/// The engine's endgame bank spends a fixed share of what is left on every
/// move, so without a floor the clock can reach about a second just before
/// the exact solve, where server lag could forfeit the game. With less than
/// this left, the search aborts at once and plays the TT move.
const MAIN_TIME_RESERVE_MS: u64 = 2000;

struct PendingAccept {
    id: String,
    opponent: Option<String>,
    variant: String,
    withdrawn_at: Option<Instant>,
}

#[derive(Default)]
struct PendingResync {
    /// Known or child-ID-inferred board count, or `None` while still unknown.
    game_slots: Option<usize>,
    is_player: bool,
    ended_games: HashSet<String>,
}

impl PendingResync {
    fn update_game_slots(&mut self, game_slots: Option<usize>) {
        match (self.game_slots, game_slots) {
            (None, Some(game_slots)) => {
                self.game_slots = Some(game_slots);
                if game_slots == 1 {
                    self.ended_games.clear();
                }
            }
            (Some(old), Some(new)) if new > old => self.game_slots = Some(new),
            _ => {}
        }
    }
}

/// Pure-logic GGS session state machine.
pub struct Session {
    my_username: String,
    level: usize,
    max_matches: usize,
    active_matches: Vec<ActiveMatch>,
    spectator_matches: HashSet<String>,
    pending_accepts: Vec<PendingAccept>,
    /// Ids of searches whose game ended before their worker returned. Their
    /// results are dropped by search id, so a recycled GGS game id never
    /// receives them, whatever order the workers finish in.
    retired_searches: HashSet<u64>,
    /// Id for the next [`SessionAction::StartSearch`] or
    /// [`SessionAction::StartPonder`].
    next_search_id: u64,
    /// Game ids awaiting a verbose state and their capacity state until the
    /// response establishes whether we are a player or spectator.
    resync_requested: HashMap<String, PendingResync>,
    /// Whether to ponder (search the opponent-to-move board) while waiting.
    ponder: bool,
    /// Latest opponent-to-move board per game id, kept while the game waits.
    waiting_boards: HashMap<String, Board>,
    /// Game ids with a ponder search possibly running.
    pondering_games: HashSet<String>,
    /// Game id we most recently sent a move for. In synchro its opponent is
    /// not on the clock yet, so the sibling is the one worth pondering.
    last_played_game: Option<String>,
}

impl Session {
    /// Creates a new session. Infallible.
    pub fn new(my_username: String, level: usize, max_matches: usize) -> Self {
        Self {
            my_username,
            level,
            max_matches,
            active_matches: Vec::new(),
            spectator_matches: HashSet::new(),
            pending_accepts: Vec::new(),
            retired_searches: HashSet::new(),
            next_search_id: 0,
            resync_requested: HashMap::new(),
            ponder: false,
            waiting_boards: HashMap::new(),
            pondering_games: HashSet::new(),
            last_played_game: None,
        }
    }

    /// Enables pondering: once every live game has a server-confirmed
    /// opponent-to-move board, each waiting game's engine searches that board
    /// to warm its transposition table. Ponders never run alongside a real
    /// search — or while any game's turn is still unknown — so they cannot
    /// hold the shared search pool a searching game needs.
    pub fn enable_ponder(&mut self) {
        self.ponder = true;
    }

    fn active_match(&self, game_id: &str) -> Option<&ActiveMatch> {
        self.active_matches
            .iter()
            .find(|active| active.contains_game(game_id))
    }

    fn active_match_mut(&mut self, game_id: &str) -> Option<&mut ActiveMatch> {
        self.active_matches
            .iter_mut()
            .find(|active| active.contains_game(game_id))
    }

    fn any_search_in_flight(&self) -> bool {
        self.active_matches
            .iter()
            .any(|active| !active.searching_games.is_empty())
    }

    fn is_ponderable_game(&self, id: &str) -> bool {
        self.active_matches.iter().any(|active| {
            active.contains_game(id)
                && !active.failed_games.contains(id)
                && !active.ended_games.contains(id)
        })
    }

    /// Emits a [`SessionAction::StopPonder`] for every possibly running ponder.
    fn stop_all_ponders(&mut self, actions: &mut Vec<SessionAction>) {
        for game_id in std::mem::take(&mut self.pondering_games) {
            actions.push(SessionAction::StopPonder { game_id });
        }
    }

    /// Drops the stored waiting boards of every game belonging to
    /// `parent_id` (the id itself and its synchro children). Used when a
    /// match's state becomes uncertain: a retained board could later satisfy
    /// the ponder gate even though the game may meanwhile be ours to move.
    fn clear_waiting_boards_for_match(&mut self, parent_id: &str) {
        self.waiting_boards
            .retain(|game_id, _| !is_game_of(parent_id, game_id));
    }

    /// Drops ponder state for one game, stopping its ponder if running.
    fn clear_ponder_for_game(&mut self, id: &str, actions: &mut Vec<SessionAction>) {
        self.waiting_boards.remove(id);
        if self.pondering_games.remove(id) {
            actions.push(SessionAction::StopPonder {
                game_id: id.to_string(),
            });
        }
    }

    /// Starts ponders, but only while the whole session is provably idle: no
    /// search in flight, no retired search still holding an engine, no match
    /// awaiting resync, and a server-confirmed opponent-to-move board for
    /// every live game. A game whose state has not arrived yet may turn out
    /// to be ours to move, and a ponder started early would hold the shared
    /// search pool exactly when that real search materializes.
    fn start_ready_ponders(&mut self, actions: &mut Vec<SessionAction>) {
        // A retired search still occupies the shared search pool even
        // though its match is gone, so treat it like a search in flight.
        if !self.ponder
            || self.any_search_in_flight()
            || !self.retired_searches.is_empty()
            || !self.resync_requested.is_empty()
        {
            return;
        }
        let live_games: usize = self
            .active_matches
            .iter()
            .map(|active| {
                active
                    .game_slots
                    .saturating_sub(active.ended_games.len() + active.failed_games.len())
            })
            .sum();
        let waiting_live = self
            .waiting_boards
            .keys()
            .filter(|id| self.is_ponderable_game(id))
            .count();
        if live_games == 0 || waiting_live < live_games {
            return;
        }
        for (id, board) in &self.waiting_boards {
            if self.pondering_games.contains(id)
                || !self.is_ponderable_game(id)
                || (self.last_played_game.as_ref() == Some(id)
                    && synchro_sibling(id)
                        .is_some_and(|sib| self.waiting_boards.contains_key(&sib)))
            {
                continue;
            }
            self.pondering_games.insert(id.clone());
            actions.push(SessionAction::StartPonder {
                game_id: id.clone(),
                search_id: self.next_search_id,
                board: *board,
            });
            self.next_search_id += 1;
        }
    }

    /// Consume one [`OsEvent`] and return the actions that should be performed.
    pub fn on_event(&mut self, event: OsEvent) -> Vec<SessionAction> {
        match event {
            OsEvent::MatchCreated {
                id,
                opponent,
                variant,
            } => self.on_match_created(id, opponent, variant),
            OsEvent::MatchState(state) => self.on_match_state(state),
            OsEvent::MatchStateUnavailable {
                id,
                variant,
                is_player,
            } => self.on_match_state_unavailable(id, variant, is_player),
            OsEvent::SpectatedMatchState { id, variant } => {
                self.on_spectated_match_state(id, variant)
            }
            OsEvent::MatchEnd { id, result, .. } => self.on_match_end(id, result),
            OsEvent::Kibitz { id, from, message } => {
                vec![SessionAction::Log(format!("[{id}] {from}: {message}"))]
            }
            OsEvent::AskOffer(offer) => self.on_ask_offer(offer),
            OsEvent::AskWithdrawn { id } => self.on_ask_withdrawn(id),
            OsEvent::ServerError { id, message } => self.on_server_error(id, message),
            OsEvent::RequestError { id, message } => self.on_request_error(id, message),
            OsEvent::Unknown => Vec::new(),
        }
    }

    /// Handle an incoming match-request ("ask") offer.
    ///
    /// Policy:
    /// * If the offer is our own outgoing ask echoed back, do NOT self-accept
    ///   (the server rejects self-accepts anyway; see `EXE_Service::accept`).
    /// * Only offers addressed to us are accepted; global `*` and third-party
    ///   offers are ignored.
    /// * Unsupported direct offers are declined.
    /// * Pending auto-accepts reserve their match slots, so the client
    ///   enforces its own capacity.
    fn on_ask_offer(
        &mut self,
        AskOffer {
            id,
            from,
            to,
            game_type,
            clock,
            mode,
        }: AskOffer,
    ) -> Vec<SessionAction> {
        self.recover_expired_pending_accept();
        if let Some(index) = self
            .pending_accepts
            .iter()
            .position(|pending| pending.id == id && pending.withdrawn_at.is_some())
        {
            self.pending_accepts.swap_remove(index);
        }
        if from == self.my_username.as_str() {
            if !is_supported_ask_clock(&clock) {
                return vec![
                    SessionAction::Send(format!("tell /os cancel {id}")),
                    SessionAction::Log(format!("cancelling offer {id}; unsupported clock {clock}")),
                ];
            }
            if is_supported_ask_game(&game_type)
                && !self.pending_accepts.iter().any(|pending| pending.id == id)
                && self.reserved_match_slots() < self.max_matches
            {
                self.pending_accepts.push(PendingAccept::new(
                    id,
                    (to != "*").then_some(to),
                    game_type,
                ));
            }
            return Vec::new();
        }
        if to != self.my_username {
            return Vec::new();
        }
        let refuse = |reason| decline_offer(&id, &from, reason);
        if !is_supported_ask_game(&game_type) {
            return refuse(format!("unsupported game {game_type} {mode}"));
        }
        if !is_supported_ask_clock(&clock) {
            return refuse(format!("unsupported clock {clock}"));
        }

        if self.pending_accepts.iter().any(|pending| pending.id == id) {
            return Vec::new();
        }

        if self.reserved_match_slots() >= self.max_matches {
            return refuse(format!("maximum {} matches reached", self.max_matches));
        }

        self.pending_accepts.push(PendingAccept::new(
            id.clone(),
            Some(from.clone()),
            game_type.clone(),
        ));
        vec![
            SessionAction::Send(format!("tell /os accept {id}")),
            SessionAction::Log(format!(
                "accepting offer {id} from {from} ({game_type} {mode})"
            )),
        ]
    }

    /// A withdrawal is also emitted immediately before an accepted ask turns
    /// into a match. Keep the reservation briefly so that ordering cannot
    /// reopen its slot during match creation.
    fn on_ask_withdrawn(&mut self, id: String) -> Vec<SessionAction> {
        if let Some(pending) = self
            .pending_accepts
            .iter_mut()
            .find(|pending| pending.id == id)
        {
            pending.withdrawn_at.get_or_insert_with(Instant::now);
        }
        Vec::new()
    }

    /// Processes a completed search result and its elapsed turn time.
    ///
    /// `elapsed` includes engine queue time and the wait to process the result.
    /// `game_id` and `search_id` come from the [`SessionAction::StartSearch`];
    /// a result for a search that is no longer current (e.g. its match ended
    /// while we were searching) is logged and dropped.
    pub fn on_move_ready(
        &mut self,
        game_id: &str,
        search_id: u64,
        mv: Option<(Square, Scoref)>,
        elapsed: Duration,
    ) -> Vec<SessionAction> {
        if self.retired_searches.remove(&search_id) {
            let mut actions = vec![SessionAction::Log(format!(
                "dropping stale move for ended game {game_id}"
            ))];
            self.start_ready_ponders(&mut actions);
            return actions;
        }
        if !self.take_current_search(game_id, search_id) {
            return vec![SessionAction::Log(format!(
                "dropping stale move for game {game_id}"
            ))];
        }
        let line = match mv {
            Some((sq, score)) => format!(
                "tell /os play {game_id} {sq}/{}/{:.3}",
                score.round() as i32,
                elapsed.as_secs_f64()
            ),
            None => format!(
                "tell /os play {game_id} PASS/0/{:.3}",
                elapsed.as_secs_f64()
            ),
        };
        let mut actions = vec![SessionAction::Send(line)];
        // The board we just played on gets a fresh opponent-to-move
        // state from the server; any stored board for it is stale.
        self.waiting_boards.remove(game_id);
        self.last_played_game = Some(game_id.to_string());
        self.start_ready_ponders(&mut actions);
        actions
    }

    /// Resign a game whose search terminated without producing a move.
    /// A failure marker prevents a duplicate state from starting a failure loop
    /// before the server confirms the resignation.
    pub fn on_search_aborted(&mut self, game_id: &str, search_id: u64) -> Vec<SessionAction> {
        if self.retired_searches.remove(&search_id) {
            let mut actions = vec![SessionAction::Log(format!(
                "dropping stale search failure for ended game {game_id}"
            ))];
            self.start_ready_ponders(&mut actions);
            return actions;
        }
        self.waiting_boards.remove(game_id);
        if self.take_current_search(game_id, search_id)
            && let Some(cur) = self.active_match_mut(game_id)
        {
            cur.failed_games.insert(game_id.to_string());
            return vec![SessionAction::Send(format!("tell /os resign {game_id}"))];
        }
        vec![SessionAction::Log(format!(
            "dropping stale search failure for game {game_id}"
        ))]
    }

    fn on_match_created(
        &mut self,
        id: String,
        opponent: Option<String>,
        variant: Option<String>,
    ) -> Vec<SessionAction> {
        self.recover_expired_pending_accept();
        self.clear_resync_for_match(&id);
        // Spectator vs player is resolved by whether our username appears in
        // the `+ match` name list (populated by `parse_match_created`).
        let Some(opponent_name) = opponent else {
            self.spectator_matches.insert(id.clone());
            return vec![SessionAction::Log(format!("ignoring spectator match {id}"))];
        };
        self.spectator_matches.remove(&id);

        if self.active_matches.iter().any(|active| active.id == id) {
            return vec![SessionAction::Log(format!(
                "duplicate match-created for {id}"
            ))];
        }

        let Some(variant) = variant.filter(|variant| is_supported_ask_game(variant)) else {
            return vec![
                SessionAction::Send(format!("tell /os resign {id}")),
                SessionAction::Log(format!(
                    "resigning unsupported match {id}; variant cannot be played"
                )),
            ];
        };

        self.consume_pending_accept(Some(&opponent_name), &variant);
        let game_slots = required_game_slots(&variant);
        // The server already started this match; unrelated pending asks must
        // not displace it.
        if self.active_match_slots() >= self.max_matches {
            return vec![
                SessionAction::Send(format!("tell /os resign {id}")),
                SessionAction::Log(format!(
                    "resigning match {id}; maximum {} matches reached",
                    self.max_matches
                )),
            ];
        }
        let level = self.level;
        let kind = if is_synchro_variant(&variant) {
            "synchro "
        } else {
            ""
        };
        let log = SessionAction::Log(format!(
            "{kind}match {id} started vs {opponent_name} ({variant}, level {level})"
        ));
        self.active_matches.push(ActiveMatch {
            id,
            game_slots,
            my_colors: HashMap::new(),
            opponent: opponent_name,
            searching_games: HashMap::new(),
            failed_games: HashSet::new(),
            ended_games: HashSet::new(),
        });
        let mut actions = vec![log];
        // A live game whose turn is still unknown exists now, so the
        // all-games-waiting invariant no longer holds.
        self.stop_all_ponders(&mut actions);
        actions
    }

    fn on_match_state(
        &mut self,
        MatchState {
            id,
            variant,
            opponent,
            board,
            side_to_move,
            my_color,
            my_clock,
        }: MatchState,
    ) -> Vec<SessionAction> {
        self.recover_expired_pending_accept();
        let resync_id = resync_match_id(&id, variant.as_deref());
        if self.spectator_matches.contains(&resync_id) {
            return vec![SessionAction::Log(format!(
                "ignoring state for spectator match {id}"
            ))];
        }
        let is_active = self.active_match(&id).is_some();
        if !is_active
            && my_color.is_none()
            && let Some(resync) = self.resync_requested.get_mut(&resync_id)
            && !resync.is_player
        {
            resync.update_game_slots(variant.as_deref().map(required_game_slots));
            return vec![SessionAction::Log(format!(
                "waiting for player role for match {id}"
            ))];
        }
        let resync = self.take_resync_for_match(&resync_id);
        let had_resync = resync.is_some();
        let PendingResync {
            is_player: resync_is_player,
            mut ended_games,
            ..
        } = resync.unwrap_or_default();
        let mut actions = Vec::new();

        if !is_active {
            let consumed_pending = !had_resync
                && variant.as_deref().is_some_and(|variant| {
                    self.consume_pending_accept(opponent.as_deref(), variant)
                });
            if my_color.is_none() && !resync_is_player && !consumed_pending {
                return vec![SessionAction::Log(format!(
                    "ignoring state for match {id}; we are not a player"
                ))];
            }
            let Some(variant) = variant.filter(|variant| is_supported_ask_game(variant)) else {
                return vec![
                    SessionAction::Send(format!("tell /os resign {id}")),
                    SessionAction::Log(format!(
                        "resigning recovered match {id}; variant cannot be played"
                    )),
                ];
            };
            let game_slots = required_game_slots(&variant);
            if game_slots == 1 {
                ended_games.clear();
            }
            let opponent_name = opponent.unwrap_or_else(|| "unknown".to_string());
            if self.active_match_slots() >= self.max_matches {
                return vec![
                    SessionAction::Send(format!("tell /os resign {id}")),
                    SessionAction::Log(format!(
                        "resigning recovered match {id}; maximum {} matches reached",
                        self.max_matches
                    )),
                ];
            }
            self.active_matches.push(ActiveMatch {
                id: resync_id.clone(),
                game_slots,
                my_colors: HashMap::new(),
                opponent: opponent_name.clone(),
                searching_games: HashMap::new(),
                failed_games: HashSet::new(),
                ended_games,
            });
            actions.push(SessionAction::Log(format!(
                "recovered match {resync_id} vs {opponent_name} ({variant})"
            )));
            // The recovered match may hold games whose turn is still unknown
            // (e.g. a synchro sibling); re-establish the invariant from zero.
            self.stop_all_ponders(&mut actions);
        }

        let search_id = self.next_search_id;
        let cur = self
            .active_match_mut(&id)
            .expect("active match was found or recovered above");

        if cur.ended_games.contains(&id) {
            actions.push(SessionAction::Log(format!(
                "ignoring state for ended game {id}"
            )));
            return actions;
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
            actions.push(SessionAction::Log(format!(
                "ignoring state for match {id}; our color not yet known"
            )));
            return actions;
        };

        if side_to_move != my_color {
            if !cur.failed_games.contains(&id) && self.ponder {
                self.clear_ponder_for_game(&id, &mut actions);
                self.waiting_boards.insert(id.clone(), board);
                self.start_ready_ponders(&mut actions);
            }
            return actions;
        }

        if cur.failed_games.contains(&id) || cur.searching_games.contains_key(&id) {
            actions.push(SessionAction::Log(format!(
                "search already in flight for match {id}, skipping duplicate state"
            )));
            return actions;
        }
        cur.searching_games.insert(id.clone(), search_id);
        self.next_search_id += 1;

        self.waiting_boards.remove(&id);
        self.stop_all_ponders(&mut actions);

        let search_limit = match my_clock {
            Some(clock) => SearchLimit::Time(TimeControlMode::Fischer {
                main_time_ms: clock.main_ms.saturating_sub(MAIN_TIME_RESERVE_MS),
                increment_ms: clock.increment_ms,
            }),
            None => SearchLimit::Level { level: self.level },
        };
        actions.push(SessionAction::StartSearch {
            game_id: id,
            search_id,
            board,
            color: my_color,
            search_limit,
        });
        actions
    }

    fn on_spectated_match_state(
        &mut self,
        id: String,
        variant: Option<String>,
    ) -> Vec<SessionAction> {
        let resync_id = resync_match_id(&id, variant.as_deref());
        self.clear_resync_for_match(&resync_id);
        self.spectator_matches.insert(resync_id);
        let mut actions = vec![SessionAction::Log(format!(
            "ignoring state for spectator match {id}"
        ))];
        self.start_ready_ponders(&mut actions);
        actions
    }

    fn on_match_state_unavailable(
        &mut self,
        id: String,
        variant: Option<String>,
        is_player: bool,
    ) -> Vec<SessionAction> {
        let resync_id = resync_match_id(&id, variant.as_deref());
        if self.spectator_matches.contains(&resync_id) {
            return vec![SessionAction::Log(format!(
                "ignoring state for spectator match {id}"
            ))];
        }
        if is_player
            && variant
                .as_deref()
                .is_some_and(|variant| !is_supported_ask_game(variant))
        {
            self.clear_resync_for_match(&resync_id);
            self.clear_waiting_boards_for_match(&resync_id);
            let mut actions = Vec::new();
            self.stop_all_ponders(&mut actions);
            actions.push(SessionAction::Send(format!("tell /os resign {id}")));
            actions.push(SessionAction::Log(format!(
                "resigning unparseable unsupported match {id}"
            )));
            return actions;
        }
        self.clear_waiting_boards_for_match(&resync_id);
        let game_slots = variant
            .as_deref()
            .map(required_game_slots)
            .or_else(|| synchro_parent_id(&id).map(|_| 2));
        let first_request = !self.resync_requested.contains_key(&resync_id);
        let resync = self.resync_requested.entry(resync_id).or_default();
        resync.is_player |= is_player;
        resync.update_game_slots(game_slots);
        let mut actions = Vec::new();
        // A match is now awaiting resync, so no game's turn is certain;
        // pondering may not continue until the resync resolves.
        self.stop_all_ponders(&mut actions);
        if first_request {
            actions.push(SessionAction::Send(format!("tell /os moves {id}")));
            actions.push(SessionAction::Log(format!(
                "requesting verbose state for unparseable match {id}"
            )));
        } else {
            actions.push(SessionAction::Log(format!(
                "state for match {id} is still unparseable"
            )));
        }
        actions
    }

    fn on_match_end(&mut self, id: String, result: String) -> Vec<SessionAction> {
        if self.spectator_matches.remove(&id) {
            self.clear_resync_for_match(&id);
            return log_non_current_match_end(&id);
        }
        let Some(index) = self
            .active_matches
            .iter()
            .position(|active| active.contains_game(&id))
        else {
            if !self.record_unresolved_child_end(&id) {
                self.clear_resync_for_match(&id);
            }
            return log_non_current_match_end(&id);
        };
        self.clear_resync_for_match(&id);
        let opponent = self.active_matches[index].opponent.clone();
        let mut actions = Vec::new();
        if self.active_matches[index].id == id {
            let ended = self.active_matches.swap_remove(index);
            self.waiting_boards
                .retain(|game_id, _| !ended.contains_game(game_id));
            actions.extend(
                self.pondering_games
                    .extract_if(|game_id| ended.contains_game(game_id))
                    .map(|game_id| SessionAction::StopPonder { game_id }),
            );
            self.retired_searches
                .extend(ended.searching_games.into_values());
        } else {
            // Synchro child game (`.<parent>.<n>`): the parent match persists.
            let cur = &mut self.active_matches[index];
            cur.ended_games.insert(id.clone());
            cur.failed_games.remove(&id);
            if let Some(search_id) = cur.searching_games.remove(&id) {
                self.retired_searches.insert(search_id);
            }
            self.clear_ponder_for_game(&id, &mut actions);
        }
        actions.push(SessionAction::Log(format!(
            "match {id} ended vs {opponent}, result {result}"
        )));
        self.start_ready_ponders(&mut actions);
        actions
    }

    /// Handle a `/os: error <id> <message>` chunk. The server rejects moves
    /// and other commands with this payload. For an auto-play bot the most
    /// common trigger is an illegal move. GGS does not follow a rejected move
    /// with a board update, so request a fresh join state. Do not clear an
    /// in-flight search: the same error shape is also used by unrelated manual
    /// commands, and their failure must not invalidate a valid search result.
    fn on_server_error(&mut self, id: String, message: String) -> Vec<SessionAction> {
        let log = SessionAction::Log(format!("server error for {id}: {message}"));
        if self.active_match(&id).is_some() {
            vec![SessionAction::Send(format!("tell /os moves {id}")), log]
        } else {
            vec![log]
        }
    }

    fn on_request_error(&mut self, id: Option<String>, message: String) -> Vec<SessionAction> {
        let Some(id) = id else {
            return vec![SessionAction::Log(format!("request failed: {message}"))];
        };
        self.pending_accepts.retain(|pending| pending.id != id);
        vec![SessionAction::Log(format!(
            "request {id} failed: {message}"
        ))]
    }

    /// Clears `game_id`'s in-flight search if it is `search_id`.
    fn take_current_search(&mut self, game_id: &str, search_id: u64) -> bool {
        let Some(cur) = self.active_match_mut(game_id) else {
            return false;
        };
        let current = cur.searching_games.get(game_id) == Some(&search_id);
        if current {
            cur.searching_games.remove(game_id);
        }
        current
    }

    fn active_match_slots(&self) -> usize {
        // A synchro match occupies one slot until the parent match ends,
        // even when one or both child games have already ended.
        let active_slots = self.active_matches.len();
        let unresolved_slots = self
            .resync_requested
            .iter()
            .filter(|(id, _)| self.active_match(id).is_none())
            .count();
        active_slots + unresolved_slots
    }

    fn reserved_match_slots(&self) -> usize {
        self.active_match_slots() + self.pending_accepts.len()
    }

    fn recover_expired_pending_accept(&mut self) {
        self.pending_accepts.retain(|pending| !pending.is_expired());
    }

    fn consume_pending_accept(&mut self, opponent: Option<&str>, variant: &str) -> bool {
        let index = match opponent {
            Some(opponent) => {
                let exact = self.pending_accepts.iter().position(|pending| {
                    pending.variant == variant && pending.opponent.as_deref() == Some(opponent)
                });
                exact.or_else(|| {
                    self.pending_accepts
                        .iter()
                        .position(|pending| pending.matches(opponent, variant))
                })
            }
            None => {
                let mut compatible = self
                    .pending_accepts
                    .iter()
                    .enumerate()
                    .filter(|(_, pending)| pending.variant == variant)
                    .map(|(index, _)| index);
                let first = compatible.next();
                if compatible.next().is_none() {
                    first
                } else {
                    None
                }
            }
        };
        if let Some(index) = index {
            self.pending_accepts.swap_remove(index);
            true
        } else {
            false
        }
    }

    fn record_unresolved_child_end(&mut self, id: &str) -> bool {
        let Some(parent) = synchro_parent_id(id) else {
            return false;
        };
        let Some(resync) = self.resync_requested.get_mut(parent) else {
            return false;
        };
        if resync.game_slots == Some(1) {
            return false;
        }
        resync.game_slots.get_or_insert(2);
        resync.ended_games.insert(id.to_string());
        true
    }

    /// Removes the resyncs of match `id` and its synchro children, merged
    /// into one.
    fn take_resync_for_match(&mut self, id: &str) -> Option<PendingResync> {
        let mut taken: Option<PendingResync> = None;
        self.resync_requested.retain(|requested, resync| {
            if !is_game_of(id, requested) {
                return true;
            }
            let merged = taken.get_or_insert_with(PendingResync::default);
            merged.game_slots = merged.game_slots.max(resync.game_slots);
            merged.is_player |= resync.is_player;
            merged.ended_games.extend(resync.ended_games.drain());
            false
        });
        taken
    }

    fn clear_resync_for_match(&mut self, id: &str) {
        self.resync_requested
            .retain(|requested, _| !is_game_of(id, requested));
    }
}

impl PendingAccept {
    fn new(id: String, opponent: Option<String>, variant: String) -> Self {
        Self {
            id,
            opponent,
            variant,
            withdrawn_at: None,
        }
    }

    fn matches(&self, opponent: &str, variant: &str) -> bool {
        self.variant == variant
            && self
                .opponent
                .as_deref()
                .is_none_or(|expected| expected == opponent)
    }

    fn is_expired(&self) -> bool {
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

/// The search time manager supports a whole-game bank and an additive
/// increment awarded every move. Other GGS clock modes must not be accepted
/// and silently treated as Fischer clocks.
pub(super) fn is_supported_ask_clock(clock: &str) -> bool {
    let mut fields = clock.split('/');
    let initial = fields.next().unwrap_or("");
    let increment = fields.next().unwrap_or("");
    let _extension = fields.next().unwrap_or("");
    if fields.next().is_some() {
        return false;
    }

    supported_move_count(initial, 0) && supported_move_count(increment, 1)
}

fn supported_move_count(field: &str, expected: u32) -> bool {
    field
        .split_once(',')
        .is_none_or(|(_, moves)| moves.parse() == Ok(expected))
}

/// Whether `game_id` is match `match_id` itself or one of its synchro
/// children (`<match_id>.0` / `<match_id>.1`).
pub(super) fn is_game_of(match_id: &str, game_id: &str) -> bool {
    game_id == match_id
        || game_id
            .strip_prefix(match_id)
            .is_some_and(|suffix| matches!(suffix, ".0" | ".1"))
}

/// Returns the other child game id of a synchro game id (`.5.0` ↔ `.5.1`).
pub(super) fn synchro_sibling(id: &str) -> Option<String> {
    let parent = synchro_parent_id(id)?;
    let other = if id.ends_with(".0") { "1" } else { "0" };
    Some(format!("{parent}.{other}"))
}

/// Returns `true` when `variant` is a synchro variant (leading `s` qualifier,
/// e.g. `s8b`, `s8r20`): two child games with flipped colors, reported under
/// `.<id>.0` / `.<id>.1`.
fn is_synchro_variant(variant: &str) -> bool {
    variant.starts_with('s')
}

fn required_game_slots(variant: &str) -> usize {
    if is_synchro_variant(variant) { 2 } else { 1 }
}

fn resync_match_id(game_id: &str, variant: Option<&str>) -> String {
    synchro_parent_id(game_id)
        .filter(|_| variant.is_none_or(is_synchro_variant))
        .unwrap_or(game_id)
        .to_string()
}

fn synchro_parent_id(game_id: &str) -> Option<&str> {
    let (parent, child) = game_id.rsplit_once('.')?;
    (!parent.is_empty() && matches!(child, "0" | "1")).then_some(parent)
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
        if self.game_slots == 2 {
            is_game_of(&self.id, id)
        } else {
            id == self.id
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggs::protocol::PlayerClock;

    fn unavailable(id: &str, variant: Option<&str>, is_player: bool) -> OsEvent {
        OsEvent::MatchStateUnavailable {
            id: id.to_string(),
            variant: variant.map(str::to_string),
            is_player,
        }
    }

    fn match_end(id: &str, result: &str) -> OsEvent {
        OsEvent::MatchEnd {
            id: id.to_string(),
            result: result.to_string(),
            score: None,
        }
    }

    fn send(line: &str) -> SessionAction {
        SessionAction::Send(line.to_string())
    }

    /// Sorted game ids of the `StartPonder` actions.
    fn ponder_ids(actions: &[SessionAction]) -> Vec<&str> {
        let mut ids: Vec<&str> = actions
            .iter()
            .filter_map(|a| match a {
                SessionAction::StartPonder { game_id, .. } => Some(game_id.as_str()),
                _ => None,
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    fn player_clock(main_ms: u64) -> PlayerClock {
        PlayerClock {
            main_ms,
            increment_ms: 0,
        }
    }

    fn opponentless_state(id: &str, my_color: Option<Disc>) -> OsEvent {
        OsEvent::MatchState(MatchState {
            id: id.to_string(),
            variant: Some("8b".to_string()),
            opponent: None,
            board: Board::new(),
            side_to_move: Disc::White,
            my_color,
            my_clock: None,
        })
    }

    fn new_session() -> Session {
        session_with_max(4)
    }

    fn session_with_max(max_matches: usize) -> Session {
        Session::new("alice".to_string(), 21, max_matches)
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

    fn match_state(id: &str, side_to_move: Disc, my_time_ms: u64) -> OsEvent {
        match_state_with_color(id, side_to_move, None, Some(player_clock(my_time_ms)))
    }

    fn match_state_with_color(
        id: &str,
        side_to_move: Disc,
        my_color: Option<Disc>,
        my_clock: Option<PlayerClock>,
    ) -> OsEvent {
        OsEvent::MatchState(MatchState {
            id: id.to_string(),
            variant: Some("8b".to_string()),
            opponent: Some("bob".to_string()),
            board: Board::new(),
            side_to_move,
            my_color,
            my_clock,
        })
    }

    /// Drive `MatchCreated` and then — simulating the authoritative clock
    /// line of the first state chunk — resolve our color for the parent
    /// match id. This mirrors the production flow where color is only known
    /// after the first `join` chunk is parsed, but keeps existing non-synchro
    /// tests concise.
    /// Id of `game_id`'s in-flight search.
    fn search_id(s: &Session, game_id: &str) -> u64 {
        s.active_match(game_id).unwrap().searching_games[game_id]
    }

    fn enter_match(session: &mut Session, id: &str, color: Disc, opponent: &str) {
        let _ = session.on_event(match_created(id, Some(opponent)));
        if let Some(cur) = session
            .active_matches
            .iter_mut()
            .find(|active| active.id == id)
        {
            cur.my_colors.insert(id.to_string(), color);
        }
    }

    #[test]
    fn ponder_starts_on_opponent_turn_only_when_enabled() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        assert_eq!(s.on_event(match_state(".5", Disc::White, 56_000)), vec![]);

        s.enable_ponder();
        let actions = s.on_event(match_state(".5", Disc::White, 56_000));
        assert!(
            matches!(&actions[0], SessionAction::StartPonder { game_id, .. } if game_id == ".5"),
            "expected StartPonder, got {actions:?}"
        );

        // A fresh opponent-to-move board restarts the ponder.
        let actions = s.on_event(match_state(".5", Disc::White, 55_000));
        assert_eq!(
            actions,
            vec![
                SessionAction::StopPonder {
                    game_id: ".5".to_string()
                },
                SessionAction::StartPonder {
                    game_id: ".5".to_string(),
                    search_id: 1,
                    board: Board::new()
                },
            ]
        );
    }

    #[test]
    fn ponder_stops_before_own_search_starts() {
        let mut s = new_session();
        s.enable_ponder();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::White, 56_000));

        let actions = s.on_event(match_state(".5", Disc::Black, 56_000));
        let stop = actions
            .iter()
            .position(|a| matches!(a, SessionAction::StopPonder { game_id } if game_id == ".5"));
        let start = actions.iter().position(
            |a| matches!(a, SessionAction::StartSearch { game_id, .. } if game_id == ".5"),
        );
        assert!(
            stop.is_some() && start.is_some() && stop < start,
            "expected StopPonder before StartSearch, got {actions:?}"
        );
    }

    #[test]
    fn ponder_waits_for_every_live_game_to_be_waiting() {
        let mut s = new_session();
        s.enable_ponder();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        enter_match(&mut s, ".7", Disc::White, "carol");

        let searching = s.on_event(match_state(".7", Disc::White, 56_000));
        assert!(matches!(&searching[0], SessionAction::StartSearch { .. }));

        // .5 goes to the opponent while .7 is still searching: no ponder yet.
        let waiting = s.on_event(match_state(".5", Disc::White, 56_000));
        assert!(
            ponder_ids(&waiting).is_empty(),
            "no ponder may start while a search is in flight, got {waiting:?}"
        );

        // .7's move is sent, but its post-move state has not come back yet —
        // it could still be our turn again (opponent pass), so no ponder.
        let actions = s.on_move_ready(
            ".7",
            search_id(&s, ".7"),
            Some((Square::F5, 0.0)),
            Duration::ZERO,
        );
        assert_eq!(actions, vec![send("tell /os play .7 f5/0/0.000")]);

        // Once the server confirms .7 is on the opponent's turn, every live
        // game is waiting and both start pondering.
        let actions = s.on_event(match_state(".7", Disc::Black, 55_000));
        assert_eq!(ponder_ids(&actions), [".5", ".7"], "got {actions:?}");
    }

    #[test]
    fn ponder_waits_for_synchro_sibling_state() {
        let mut s = new_session();
        s.enable_ponder();
        let _ = s.on_event(match_created_with_variant(".5", Some("bob"), Some("s8r20")));

        // Only one of the two synchro boards is known to be waiting; the
        // sibling's pending state could be our turn, so no ponder yet.
        let first = s.on_event(match_state_with_color(
            ".5.0",
            Disc::White,
            Some(Disc::Black),
            None,
        ));
        assert!(
            ponder_ids(&first).is_empty(),
            "must wait for the synchro sibling, got {first:?}"
        );

        let second = s.on_event(match_state_with_color(
            ".5.1",
            Disc::Black,
            Some(Disc::White),
            None,
        ));
        assert_eq!(ponder_ids(&second), [".5.0", ".5.1"], "got {second:?}");
    }

    #[test]
    fn synchro_ponders_only_the_game_the_opponent_is_thinking_in() {
        let mut s = new_session();
        s.enable_ponder();
        let _ = s.on_event(match_created_with_variant(".5", Some("bob"), Some("s8r20")));
        let _ = s.on_event(match_state_with_color(
            ".5.0",
            Disc::Black,
            Some(Disc::Black),
            None,
        ));
        let _ = s.on_event(match_state_with_color(
            ".5.1",
            Disc::Black,
            Some(Disc::White),
            None,
        ));
        let _ = s.on_move_ready(
            ".5.0",
            search_id(&s, ".5.0"),
            Some((Square::F5, 0.0)),
            Duration::ZERO,
        );

        // Both boards now wait, but only in `.5.1` is the opponent on the
        // clock; `.5.0` resumes after the next reveal.
        let actions = s.on_event(match_state_with_color(
            ".5.0",
            Disc::White,
            Some(Disc::Black),
            None,
        ));
        assert_eq!(ponder_ids(&actions), [".5.1"], "got {actions:?}");
    }

    #[test]
    fn new_match_stops_running_ponder() {
        let mut s = new_session();
        s.enable_ponder();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let started = s.on_event(match_state(".5", Disc::White, 56_000));
        assert!(matches!(&started[0], SessionAction::StartPonder { .. }));

        // A new match introduces a game whose turn is unknown: ponder stops.
        let actions = s.on_event(match_created(".7", Some("carol")));
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, SessionAction::StopPonder { game_id } if game_id == ".5")),
            "expected StopPonder on new match, got {actions:?}"
        );
    }

    #[test]
    fn resync_request_stops_running_ponder() {
        let mut s = new_session();
        s.enable_ponder();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let started = s.on_event(match_state(".5", Disc::White, 56_000));
        assert!(matches!(&started[0], SessionAction::StartPonder { .. }));

        let actions = s.on_event(unavailable(".9", None, false));
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, SessionAction::StopPonder { game_id } if game_id == ".5")),
            "expected StopPonder on resync request, got {actions:?}"
        );
        // And no restart while the resync is pending.
        let quiet = s.on_event(match_state(".5", Disc::White, 55_000));
        assert!(
            ponder_ids(&quiet).is_empty(),
            "no ponder may start during resync, got {quiet:?}"
        );
    }

    fn synchro_state(id: &str, side_to_move: Disc, my_color: Disc) -> OsEvent {
        OsEvent::MatchState(MatchState {
            id: id.to_string(),
            variant: Some("s8r20".to_string()),
            opponent: Some("bob".to_string()),
            board: Board::new(),
            side_to_move,
            my_color: Some(my_color),
            my_clock: None,
        })
    }

    #[test]
    fn resync_drops_stale_waiting_boards_of_the_match() {
        let mut s = new_session();
        s.enable_ponder();
        let _ = s.on_event(match_created_with_variant(".5", Some("bob"), Some("s8r20")));
        let _ = s.on_event(synchro_state(".5.0", Disc::White, Disc::Black));
        let pondering = s.on_event(synchro_state(".5.1", Disc::Black, Disc::White));
        assert!(
            !ponder_ids(&pondering).is_empty(),
            "setup: both boards should ponder, got {pondering:?}"
        );

        // `.5.0` becomes unparseable: its stored board is no longer trusted.
        let _ = s.on_event(unavailable(".5.0", None, true));

        // `.5.1`'s next state clears the parent resync, but the stale `.5.0`
        // board must not satisfy the ponder gate.
        let quiet = s.on_event(synchro_state(".5.1", Disc::Black, Disc::White));
        assert!(
            ponder_ids(&quiet).is_empty(),
            "stale .5.0 board must not restart pondering, got {quiet:?}"
        );

        // A fresh `.5.0` state restores the invariant and pondering resumes.
        let resumed = s.on_event(synchro_state(".5.0", Disc::White, Disc::Black));
        assert_eq!(ponder_ids(&resumed), [".5.0", ".5.1"], "got {resumed:?}");
    }

    #[test]
    fn recycled_id_plays_while_the_retired_search_winds_down() {
        let mut s = session_with_max(1);
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::Black, 56_000));
        let _ = s.on_event(match_end(".5", "0.00"));

        enter_match(&mut s, ".5", Disc::Black, "bob");
        assert_eq!(
            s.active_matches.len(),
            1,
            "the recycled match must not be resigned"
        );
        let search = s.on_event(match_state(".5", Disc::Black, 56_000));
        assert!(matches!(&search[0], SessionAction::StartSearch { .. }));

        // Workers may finish in either order; the new search (id 1) plays and
        // the retired one (id 0) is dropped without resigning the new match.
        let played = s.on_move_ready(".5", 1, Some((Square::F5, 0.0)), Duration::ZERO);
        assert_eq!(played[0], send("tell /os play .5 f5/0/0.000"));
        let stale = s.on_search_aborted(".5", 0);
        assert!(!stale.iter().any(|a| matches!(a, SessionAction::Send(_))));
    }

    #[test]
    fn match_end_stops_ponder() {
        let mut s = new_session();
        s.enable_ponder();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::White, 56_000));

        let actions = s.on_event(match_end(".5", "0.00"));
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, SessionAction::StopPonder { game_id } if game_id == ".5")),
            "expected StopPonder on match end, got {actions:?}"
        );
        assert!(s.waiting_boards.is_empty());
        assert!(s.pondering_games.is_empty());
    }

    #[test]
    fn multiple_matches_search_and_finish_independently() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        enter_match(&mut s, ".7", Disc::White, "carol");
        assert_eq!(s.active_matches.len(), 2);
        assert!(s.active_matches.iter().any(|active| active.id == ".5"));
        assert!(s.active_matches.iter().any(|active| active.id == ".7"));

        let search = s.on_event(match_state(".7", Disc::White, 56_000));
        assert!(
            matches!(&search[0], SessionAction::StartSearch { game_id, .. } if game_id == ".7")
        );

        let _ = s.on_event(match_end(".5", "0.00"));
        assert_eq!(s.active_matches.len(), 1);
        assert_eq!(s.active_matches[0].id, ".7");
        assert_eq!(
            s.on_move_ready(
                ".7",
                search_id(&s, ".7"),
                Some((Square::F5, 0.0)),
                Duration::ZERO
            ),
            vec![send("tell /os play .7 f5/0/0.000")]
        );
    }

    #[test]
    fn self_play_searches_both_colors() {
        let mut s = new_session();
        let _ = s.on_event(match_created(".5", Some("alice")));

        let black = s.on_event(match_state_with_color(
            ".5",
            Disc::Black,
            Some(Disc::Black),
            None,
        ));
        assert!(matches!(&black[0], SessionAction::StartSearch { .. }));
        let _ = s.on_move_ready(
            ".5",
            search_id(&s, ".5"),
            Some((Square::F5, 0.0)),
            Duration::ZERO,
        );

        let white = s.on_event(match_state_with_color(
            ".5",
            Disc::White,
            Some(Disc::White),
            None,
        ));
        assert!(matches!(&white[0], SessionAction::StartSearch { .. }));
    }

    #[test]
    fn unsupported_match_is_resigned_without_becoming_active() {
        for variant in [None, Some("10b"), Some("8a"), Some("8k")] {
            let mut s = new_session();
            let actions = s.on_event(match_created_with_variant(".5", Some("bob"), variant));
            assert_eq!(actions[0], send("tell /os resign .5"));
            assert!(s.active_matches.is_empty());
        }
    }

    #[test]
    fn match_state_on_my_turn_emits_start_search() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000));
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::StartSearch {
                game_id,
                search_limit,
                ..
            } => {
                assert_eq!(game_id, ".5");
                assert_eq!(
                    *search_limit,
                    SearchLimit::Time(TimeControlMode::Fischer {
                        main_time_ms: 56_000 - MAIN_TIME_RESERVE_MS,
                        increment_ms: 0,
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
        let actions = s.on_event(match_state_with_color(".5", Disc::Black, None, None));
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::StartSearch {
                game_id,
                search_limit,
                ..
            } => {
                assert_eq!(game_id, ".5");
                assert_eq!(*search_limit, SearchLimit::Level { level: 21 });
            }
            other => panic!("expected StartSearch, got {other:?}"),
        }
    }

    #[test]
    fn reconnect_state_recovers_match_without_created_event() {
        let mut s = session_with_max(1);
        let actions = s.on_event(match_state_with_color(
            ".5",
            Disc::Black,
            Some(Disc::Black),
            Some(player_clock(56_000)),
        ));

        assert_eq!(s.active_matches.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(msg) if msg.contains("recovered")));
        assert!(
            matches!(&actions[1], SessionAction::StartSearch { game_id, .. } if game_id == ".5")
        );
    }

    #[test]
    fn spectated_state_releases_its_resync_reservation() {
        let mut s = session_with_max(1);
        let _ = s.on_event(unavailable(".9", None, false));
        let _ = s.on_event(OsEvent::SpectatedMatchState {
            id: ".9".to_string(),
            variant: Some("10".to_string()),
        });

        assert!(s.resync_requested.is_empty());
        assert_eq!(
            s.on_event(ask_offer(".3", "bob", "alice"))[0],
            send("tell /os accept .3")
        );
    }

    #[test]
    fn compact_reconnect_of_low_numbered_match_recovers() {
        // `.1` is an ordinary match id, not a synchro child of "".
        let mut s = session_with_max(1);
        let _ = s.on_event(unavailable(".1", None, false));
        let actions = s.on_event(match_state_with_color(
            ".1",
            Disc::Black,
            Some(Disc::Black),
            Some(player_clock(56_000)),
        ));

        assert!(s.resync_requested.is_empty());
        assert!(
            matches!(&actions[1], SessionAction::StartSearch { game_id, .. } if game_id == ".1")
        );
    }

    #[test]
    fn unparseable_reconnect_state_is_resynced_once() {
        let mut s = session_with_max(1);
        let first = s.on_event(unavailable(".5", None, false));
        assert_eq!(first[0], send("tell /os moves .5"));
        let second = s.on_event(unavailable(".5", None, false));
        assert!(
            !second
                .iter()
                .any(|action| matches!(action, SessionAction::Send(_)))
        );

        let _ = s.on_event(match_end(".5", "aborted"));
        let reused = s.on_event(unavailable(".5", None, false));
        assert_eq!(reused[0], send("tell /os moves .5"));
    }

    #[test]
    fn unavailable_state_reserves_its_game_slot_until_role_is_known() {
        let mut s = session_with_max(1);
        // Compact GGF reconnects omit clock lines, so the parser cannot
        // know yet that this state belongs to a player.
        let _ = s.on_event(unavailable(".5", Some("8b"), false));

        let unresolved = s.on_event(opponentless_state(".5", None));
        assert!(
            matches!(&unresolved[0], SessionAction::Log(message) if message.contains("waiting"))
        );
        assert_eq!(s.resync_requested.len(), 1);

        let offer = s.on_event(ask_offer(".8", "carol", "alice"));
        assert_eq!(offer[0], send("tell /os decline .8"));

        let recovered = s.on_event(match_state_with_color(
            ".5",
            Disc::White,
            Some(Disc::Black),
            None,
        ));
        assert!(
            matches!(&recovered[0], SessionAction::Log(message) if message.contains("recovered"))
        );
        assert_eq!(s.active_matches[0].id, ".5");
    }

    #[test]
    fn resync_reserves_one_match_regardless_of_variant() {
        let mut s = session_with_max(2);
        let _ = s.on_event(unavailable(".5", None, false));
        assert_eq!(s.active_match_slots(), 1);

        let _ = s.on_event(opponentless_state(".5", None));
        assert_eq!(s.active_match_slots(), 1);

        let _ = s.on_event(unavailable(".5", Some("s8"), false));
        let _ = s.on_event(unavailable(".5", Some("8b"), false));
        assert_eq!(s.active_match_slots(), 1);
    }

    #[test]
    fn known_player_resync_survives_state_without_color() {
        let mut s = session_with_max(1);
        let _ = s.on_event(unavailable(".5", Some("8b"), true));
        let actions = s.on_event(match_state_with_color(".5", Disc::White, None, None));

        assert!(
            matches!(&actions[0], SessionAction::Log(message) if message.contains("recovered"))
        );
        assert_eq!(s.active_matches[0].id, ".5");
        assert_eq!(
            s.on_event(ask_offer(".8", "carol", "alice"))[0],
            send("tell /os decline .8")
        );
    }

    #[test]
    fn resync_recovery_does_not_consume_an_ambiguous_pending_accept() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(unavailable(".5", Some("8b"), true));
        let actions = s.on_event(opponentless_state(".5", Some(Disc::Black)));

        assert!(!actions.contains(&send("tell /os resign .5")));
        assert_eq!(s.pending_accepts.len(), 1);
        assert_eq!(s.active_matches[0].id, ".5");
    }

    #[test]
    fn synchro_child_resyncs_share_one_reservation() {
        let mut s = session_with_max(1);
        let first = s.on_event(unavailable(".5.0", None, true));
        let second = s.on_event(unavailable(".5.1", Some("s8"), true));

        assert_eq!(first[0], send("tell /os moves .5.0"));
        assert!(
            !second
                .iter()
                .any(|action| matches!(action, SessionAction::Send(_)))
        );
        assert_eq!(s.resync_requested.len(), 1);
        assert_eq!(s.active_match_slots(), 1);
    }

    #[test]
    fn ended_unresolved_synchro_child_releases_and_stays_ended() {
        let mut s = session_with_max(2);
        let _ = s.on_event(unavailable(".5.0", None, true));
        let _ = s.on_event(match_end(".5.0", "0.00"));
        assert_eq!(s.active_match_slots(), 1);

        let _ = s.on_event(OsEvent::MatchState(MatchState {
            id: ".5.1".to_string(),
            variant: Some("s8".to_string()),
            opponent: Some("bob".to_string()),
            board: Board::new(),
            side_to_move: Disc::White,
            my_color: Some(Disc::Black),
            my_clock: None,
        }));
        assert_eq!(s.active_match_slots(), 1);
        assert!(s.active_matches[0].ended_games.contains(".5.0"));
    }

    #[test]
    fn child_end_identifies_an_unknown_parent_as_synchro() {
        let mut s = session_with_max(4);
        let _ = s.on_event(unavailable(".5", None, true));
        assert_eq!(s.resync_requested[".5"].game_slots, None);
        let _ = s.on_event(match_end(".5.0", "0.00"));
        assert_eq!(s.resync_requested[".5"].game_slots, Some(2));
        assert_eq!(
            s.resync_requested[".5"].ended_games,
            HashSet::from([".5.0".to_string()])
        );
        assert_eq!(s.active_match_slots(), 1);
        let _ = s.on_event(OsEvent::MatchState(MatchState {
            id: ".5.1".to_string(),
            variant: Some("s8".to_string()),
            opponent: Some("bob".to_string()),
            board: Board::new(),
            side_to_move: Disc::White,
            my_color: Some(Disc::Black),
            my_clock: None,
        }));
        assert!(s.resync_requested.is_empty());
        assert_eq!(s.active_matches[0].game_slots, 2);
        assert_eq!(
            s.active_matches[0].ended_games,
            HashSet::from([".5.0".to_string()])
        );
    }

    #[test]
    fn recovered_synchro_fits_one_match_slot() {
        let mut s = session_with_max(1);
        let _ = s.on_event(unavailable(".5", None, true));
        let actions = s.on_event(OsEvent::MatchState(MatchState {
            id: ".5.0".to_string(),
            variant: Some("s8".to_string()),
            opponent: Some("bob".to_string()),
            board: Board::new(),
            side_to_move: Disc::White,
            my_color: Some(Disc::Black),
            my_clock: None,
        }));

        assert!(!actions.contains(&send("tell /os resign .5.0")));
        assert_eq!(s.active_matches.len(), 1);
        assert_eq!(s.active_matches[0].game_slots, 2);
        assert_eq!(s.active_match_slots(), 1);
    }

    #[test]
    fn unsupported_unparseable_player_match_is_resigned() {
        let mut s = session_with_max(1);
        let actions = s.on_event(unavailable(".5", Some("10b"), true));
        assert_eq!(actions[0], send("tell /os resign .5"));
    }

    #[test]
    fn child_match_end_invalidates_child_search_and_drops_stale_move() {
        let mut s = new_session();
        let _ = s.on_event(match_created_with_variant(".5", Some("bob"), Some("s8")));
        let search = s.on_event(match_state_with_color(
            ".5.1",
            Disc::White,
            Some(Disc::White),
            Some(player_clock(56_000)),
        ));
        assert!(matches!(&search[0], SessionAction::StartSearch { .. }));

        let ended = s.on_event(match_end(".5.1", "0.00"));
        assert_eq!(s.active_matches.len(), 1);
        assert!(!s.active_matches[0].searching_games.contains_key(".5.1"));
        assert!(matches!(&ended[0], SessionAction::Log(msg) if msg.contains(".5.1")));

        let duplicate = s.on_event(match_state_with_color(
            ".5.1",
            Disc::White,
            Some(Disc::White),
            None,
        ));
        assert!(matches!(&duplicate[0], SessionAction::Log(msg) if msg.contains("ended")));

        let stale = s.on_move_ready(".5.1", 0, Some((Square::F5, 0.0)), Duration::ZERO);
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
            Some(player_clock(60_000)),
        ));
        match &first[0] {
            SessionAction::StartSearch { game_id, .. } => assert_eq!(game_id, ".5.0"),
            other => panic!("expected StartSearch for .5.0, got {other:?}"),
        }

        // `.5.1`: we are White. Same side to move → search it without
        // clobbering the `.5.0` color.
        let second = s.on_event(match_state_with_color(
            ".5.1",
            Disc::White,
            Some(Disc::White),
            Some(player_clock(60_000)),
        ));
        match &second[0] {
            SessionAction::StartSearch { game_id, .. } => assert_eq!(game_id, ".5.1"),
            other => panic!("expected StartSearch for .5.1, got {other:?}"),
        }

        let cur = &s.active_matches[0];
        assert_eq!(cur.my_colors.get(".5.0"), Some(&Disc::Black));
        assert_eq!(cur.my_colors.get(".5.1"), Some(&Disc::White));
        assert!(cur.searching_games.contains_key(".5.0"));
        assert!(cur.searching_games.contains_key(".5.1"));

        // Per-game send should use the child id, not the parent.
        let send0 = s.on_move_ready(
            ".5.0",
            search_id(&s, ".5.0"),
            Some((Square::F5, 0.0)),
            Duration::from_millis(1234),
        );
        assert_eq!(send0, vec![send("tell /os play .5.0 f5/0/1.234")]);
        let send1 = s.on_move_ready(
            ".5.1",
            search_id(&s, ".5.1"),
            Some((Square::F5, 0.0)),
            Duration::from_millis(5678),
        );
        assert_eq!(send1, vec![send("tell /os play .5.1 f5/0/5.678")]);
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
            Some(player_clock(60_000)),
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
        let actions = s.on_event(match_state(".7", Disc::Black, 56_000));
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(msg) if msg.contains(".7")));
    }

    #[test]
    fn match_end_for_other_match_is_logged_only() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_end(".7", "12.00"));
        assert_eq!(s.active_matches[0].id, ".5");
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(_)));
    }

    #[test]
    fn interrupted_match_end_removes_active_match() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let actions = s.on_event(match_end(".5", "alice left"));

        assert!(s.active_matches.is_empty());
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(msg) if msg.contains("alice left")));
    }

    #[test]
    fn spectator_match_created_is_ignored() {
        let mut s = session_with_max(1);
        let actions = s.on_event(match_created(".9", None));
        assert!(s.active_matches.is_empty());
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::Log(msg) if msg.contains(".9")));

        let _ = s.on_event(unavailable(".9", Some("8b"), false));
        assert!(s.resync_requested.is_empty());
        assert_eq!(
            s.on_event(ask_offer(".8", "bob", "alice"))[0],
            send("tell /os accept .8")
        );
    }

    #[test]
    fn move_ready_pass_emits_tell_os_play_pass() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::Black, 56_000));
        let actions = s.on_move_ready(".5", search_id(&s, ".5"), None, Duration::from_millis(1234));
        assert_eq!(actions, vec![send("tell /os play .5 PASS/0/1.234")]);
    }

    #[test]
    fn stale_move_ready_is_logged_and_dropped() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_end(".5", "-64.00"));
        let actions = s.on_move_ready(".5", 0, Some((Square::F5, 0.0)), Duration::ZERO);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            SessionAction::Log(msg) => assert!(msg.contains(".5")),
            other => panic!("expected Log, got {other:?}"),
        }
        assert!(!actions.iter().any(|a| matches!(a, SessionAction::Send(_))));
    }

    #[test]
    fn duplicate_match_state_while_searching_does_not_re_emit_start_search() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");

        // First MatchState on our turn emits StartSearch exactly once.
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000));
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::StartSearch { .. }));

        // A second identical MatchState arrives before on_move_ready fires.
        // We must NOT re-emit StartSearch.
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000));
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
        let sent = s.on_move_ready(
            ".5",
            search_id(&s, ".5"),
            Some((Square::F5, 0.0)),
            Duration::ZERO,
        );
        assert!(matches!(&sent[0], SessionAction::Send(_)));
        let actions = s.on_event(match_state(".5", Disc::Black, 56_000));
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], SessionAction::StartSearch { .. }),
            "expected fresh StartSearch after flag reset, got {:?}",
            actions[0]
        );
    }

    #[test]
    fn aborted_search_resigns_without_retrying() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");

        let actions = s.on_event(match_state(".5", Disc::Black, 56_000));
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SessionAction::StartSearch { .. }));

        let aborted = s.on_search_aborted(".5", search_id(&s, ".5"));
        assert_eq!(aborted, vec![send("tell /os resign .5")]);

        let actions = s.on_event(match_state(".5", Disc::Black, 56_000));
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(&actions[0], SessionAction::Log(msg) if msg.contains("already in flight")),
            "expected duplicate state to stay suppressed after abort, got {:?}",
            actions[0]
        );

        let _ = s.on_event(match_end(".5", "aborted"));
        assert!(s.retired_searches.is_empty());
    }

    #[test]
    fn ended_search_failure_releases_capacity() {
        let mut s = session_with_max(1);
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::Black, 56_000));
        let _ = s.on_event(match_end(".5", "bob left"));

        let stale = s.on_search_aborted(".5", 0);
        assert!(matches!(&stale[0], SessionAction::Log(message) if message.contains("ended")));
        assert_eq!(
            s.on_event(ask_offer(".8", "carol", "alice"))[0],
            send("tell /os accept .8")
        );
    }

    #[test]
    fn server_error_during_search_resyncs_without_dropping_result() {
        // Regression: `/os: error .5 illegal move` used to map to Unknown,
        // leaving no diagnostic trace and no way to recover — we'd wait to
        // time out. The generic error must not discard a search that may be
        // unrelated to it.
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let first = s.on_event(match_state(".5", Disc::Black, 56_000));
        assert!(matches!(&first[0], SessionAction::StartSearch { .. }));

        let err = s.on_event(OsEvent::ServerError {
            id: ".5".to_string(),
            message: "illegal move".to_string(),
        });
        assert_eq!(err[0], send("tell /os moves .5"));
        match &err[1] {
            SessionAction::Log(msg) => {
                assert!(msg.contains(".5"), "log missing id: {msg}");
                assert!(msg.contains("illegal move"), "log missing message: {msg}");
            }
            other => panic!("expected Log, got {other:?}"),
        }

        assert_eq!(
            s.on_move_ready(
                ".5",
                search_id(&s, ".5"),
                Some((Square::F5, 0.0)),
                Duration::ZERO
            ),
            vec![send("tell /os play .5 f5/0/0.000")]
        );
    }

    #[test]
    fn rejected_move_resync_restarts_search() {
        let mut s = new_session();
        enter_match(&mut s, ".5", Disc::Black, "bob");
        let _ = s.on_event(match_state(".5", Disc::Black, 56_000));
        let _ = s.on_move_ready(
            ".5",
            search_id(&s, ".5"),
            Some((Square::F5, 0.0)),
            Duration::ZERO,
        );

        let err = s.on_event(OsEvent::ServerError {
            id: ".5".to_string(),
            message: "illegal move".to_string(),
        });
        assert_eq!(err[0], send("tell /os moves .5"));

        let retry = s.on_event(match_state(".5", Disc::Black, 56_000));
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
    fn match_error_does_not_release_same_numbered_ask() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(OsEvent::ServerError {
            id: ".8".to_string(),
            message: "illegal move".to_string(),
        });

        let next = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(next[0], send("tell /os decline .9"));
    }

    fn ask_offer(id: &str, from: &str, to: &str) -> OsEvent {
        ask_offer_with_game(id, from, to, "8b", "U")
    }

    fn ask_offer_with_game(id: &str, from: &str, to: &str, game_type: &str, mode: &str) -> OsEvent {
        OsEvent::AskOffer(AskOffer {
            id: id.to_string(),
            from: from.to_string(),
            to: to.to_string(),
            game_type: game_type.to_string(),
            clock: "01:00//02:00".to_string(),
            mode: mode.to_string(),
        })
    }

    #[test]
    fn unsupported_clock_offer_is_declined() {
        let mut s = new_session();
        let actions = s.on_event(ask_offer_with_clock(
            ".8",
            "bob",
            "alice",
            "01:00,N0//02:00",
        ));
        assert_eq!(actions[0], send("tell /os decline .8"));
    }

    #[test]
    fn own_unsupported_clock_offer_is_cancelled() {
        let mut s = new_session();
        let actions = s.on_event(ask_offer_with_clock(
            ".8",
            "alice",
            "bob",
            "01:00,N0//02:00",
        ));
        assert_eq!(actions[0], send("tell /os cancel .8"));
    }

    #[test]
    fn supported_clock_defaults_and_numeric_move_counts_are_accepted() {
        for clock in ["/5", "//", "00/01", "01:00,00/00:05,01/02:00"] {
            assert!(is_supported_ask_clock(clock), "rejected {clock}");
        }
    }

    fn ask_offer_with_clock(id: &str, from: &str, to: &str, clock: &str) -> OsEvent {
        let mut offer = ask_offer(id, from, to);
        let OsEvent::AskOffer(AskOffer { clock: value, .. }) = &mut offer else {
            unreachable!()
        };
        *value = clock.to_string();
        offer
    }

    #[test]
    fn standard_8x8_ask_types_are_auto_accepted() {
        for game_type in ["8", "8b", "8w"] {
            let mut s = new_session();
            let actions = s.on_event(ask_offer_with_game(".8", "bob", "alice", game_type, "U"));
            assert_eq!(
                actions[0],
                send("tell /os accept .8"),
                "game type {game_type} should be accepted"
            );
        }
    }

    #[test]
    fn synchro_and_random_ask_types_are_auto_accepted() {
        // GGS requires random starts to be synchro (or komi, unsupported here),
        // and rejects combining random starts with a requested color.
        for game_type in ["s8", "s8b", "s8w", "s8r20"] {
            let mut s = new_session();
            let actions = s.on_event(ask_offer_with_game(".8", "bob", "alice", game_type, "U"));
            assert_eq!(
                actions[0],
                send("tell /os accept .8"),
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
                send("tell /os decline .8"),
                "game type {game_type} should be declined"
            );
        }
    }

    #[test]
    fn global_ask_offer_is_not_accepted() {
        let mut s = new_session();
        let actions = s.on_event(ask_offer_with_game(".8", "bob", "*", "8", "U"));
        assert!(actions.is_empty());
    }

    #[test]
    fn full_session_declines_direct_and_ignores_global_offers() {
        let mut s = session_with_max(1);
        enter_match(&mut s, ".5", Disc::Black, "carol");
        assert_eq!(
            s.on_event(ask_offer(".8", "bob", "alice"))[0],
            send("tell /os decline .8")
        );
        assert!(
            s.on_event(ask_offer_with_game(".9", "bob", "*", "8", "U"))
                .is_empty()
        );
    }

    #[test]
    fn synchro_offer_reserves_one_match_slot() {
        let mut one_slot = session_with_max(1);
        assert_eq!(
            one_slot.on_event(ask_offer_with_game(".8", "bob", "alice", "s8", "U"))[0],
            send("tell /os accept .8")
        );
        assert_eq!(one_slot.reserved_match_slots(), 1);
        assert_eq!(
            one_slot.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os decline .9")
        );
        let _ = one_slot.on_event(OsEvent::MatchCreated {
            id: ".5".to_string(),
            opponent: Some("bob".to_string()),
            variant: Some("s8".to_string()),
        });
        assert_eq!(one_slot.active_matches[0].game_slots, 2);
        for id in [".5.0", ".5.1"] {
            let _ = one_slot.on_event(match_end(id, "0.00"));
            assert_eq!(one_slot.active_match_slots(), 1);
            assert_eq!(
                one_slot.on_event(ask_offer(".9", "carol", "alice"))[0],
                send("tell /os decline .9")
            );
        }
        let _ = one_slot.on_event(match_end(".5", "0.00"));
        assert_eq!(
            one_slot.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os accept .9")
        );
    }

    #[test]
    fn ordinary_and_synchro_matches_share_the_same_limit() {
        let mut s = session_with_max(2);
        enter_match(&mut s, ".5", Disc::Black, "carol");
        assert_eq!(
            s.on_event(ask_offer_with_game(".8", "bob", "alice", "s8", "U"))[0],
            send("tell /os accept .8")
        );
        let _ = s.on_event(OsEvent::MatchCreated {
            id: ".7".to_string(),
            opponent: Some("bob".to_string()),
            variant: Some("s8".to_string()),
        });
        assert_eq!(s.active_match_slots(), 2);
        assert_eq!(
            s.on_event(ask_offer(".9", "dave", "alice"))[0],
            send("tell /os decline .9")
        );
    }

    #[test]
    fn unexpected_match_over_capacity_is_resigned() {
        let mut s = session_with_max(1);
        enter_match(&mut s, ".5", Disc::Black, "bob");
        assert_eq!(
            s.on_event(match_created(".7", Some("carol")))[0],
            send("tell /os resign .7")
        );
        assert_eq!(s.active_matches.len(), 1);
    }

    #[test]
    fn accepted_offer_reservation_becomes_active_match() {
        let mut s = session_with_max(2);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(match_created(".42", Some("bob")));
        assert!(s.pending_accepts.is_empty());
        assert_eq!(s.active_matches.len(), 1);
        assert_eq!(
            s.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os accept .9")
        );
    }

    #[test]
    fn opponentless_state_consumes_unique_compatible_accept() {
        for my_color in [Some(Disc::Black), None] {
            let mut s = session_with_max(2);
            let _ = s.on_event(ask_offer(".8", "bob", "alice"));
            let _ = s.on_event(ask_offer_with_game(".9", "carol", "alice", "8w", "U"));

            let actions = s.on_event(opponentless_state(".42", my_color));

            assert!(!actions.contains(&send("tell /os resign .42")));
            assert_eq!(s.pending_accepts.len(), 1);
            assert_eq!(s.pending_accepts[0].variant, "8w");
            assert_eq!(s.active_matches[0].id, ".42");
        }
    }

    #[test]
    fn opponentless_state_does_not_guess_between_compatible_accepts() {
        let mut s = session_with_max(2);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(ask_offer(".9", "carol", "alice"));

        let actions = s.on_event(opponentless_state(".42", Some(Disc::Black)));

        assert!(!actions.contains(&send("tell /os resign .42")));
        assert_eq!(s.pending_accepts.len(), 2);
        assert_eq!(s.active_matches[0].id, ".42");
    }

    #[test]
    fn exact_offer_reservation_is_consumed_before_global_ask() {
        for recovered_from_state in [false, true] {
            let mut s = session_with_max(2);
            let _ = s.on_event(ask_offer_with_game(".8", "alice", "*", "8b", "U"));
            let _ = s.on_event(ask_offer(".9", "bob", "alice"));

            if recovered_from_state {
                let _ = s.on_event(match_state_with_color(
                    ".42",
                    Disc::White,
                    Some(Disc::Black),
                    None,
                ));
            } else {
                let _ = s.on_event(match_created(".42", Some("bob")));
            }

            assert_eq!(s.pending_accepts.len(), 1);
            assert!(s.pending_accepts[0].opponent.is_none());
            let second = s.on_event(match_created(".43", Some("carol")));
            assert!(
                matches!(&second[0], SessionAction::Log(message) if message.contains("started"))
            );
            assert_eq!(s.active_matches.len(), 2);
            assert!(s.pending_accepts.is_empty());
        }
    }

    #[test]
    fn failed_accept_releases_reservation() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));

        let failed = s.on_event(OsEvent::RequestError {
            id: Some(".8".to_string()),
            message: "request '.8' not found.".to_string(),
        });
        assert!(
            matches!(&failed[0], SessionAction::Log(message) if message.contains("request") && message.contains("failed"))
        );
        assert_eq!(
            s.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os accept .9")
        );
    }

    #[test]
    fn uncorrelated_request_error_keeps_reservation() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(OsEvent::RequestError {
            id: None,
            message: "your request doesn't fit the opponent's formula.".to_string(),
        });
        assert_eq!(
            s.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os decline .9")
        );
    }

    #[test]
    fn accepted_ask_stays_reserved_until_match_created() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));

        assert_eq!(
            s.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os decline .9")
        );
        let created = s.on_event(match_created(".42", Some("bob")));
        assert!(matches!(&created[0], SessionAction::Log(message) if message.contains("started")));
        assert_eq!(s.active_matches.len(), 1);
        assert!(s.pending_accepts.is_empty());
    }

    #[test]
    fn withdrawn_accept_does_not_block_unrelated_created_match() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "alice", "bob"));
        let _ = s.on_event(OsEvent::AskWithdrawn {
            id: ".8".to_string(),
        });

        let created = s.on_event(match_created(".42", Some("carol")));
        assert!(matches!(&created[0], SessionAction::Log(message) if message.contains("started")));
        assert_eq!(s.active_matches.len(), 1);
    }

    #[test]
    fn unsupported_ask_offer_to_me_is_declined_without_latching() {
        let mut s = new_session();
        let actions = s.on_event(ask_offer_with_game(".8", "bob", "alice", "10b", "U"));
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0], send("tell /os decline .8"));
        assert!(
            matches!(&actions[1], SessionAction::Log(msg) if msg.contains("unsupported game 10b U")),
            "expected unsupported-game decline log, got {:?}",
            actions[1]
        );
        let accepted = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(accepted[0], send("tell /os accept .9"));
    }

    #[test]
    fn own_outgoing_ask_offer_is_not_self_accepted() {
        // If the offer echoes back with us as the challenger (from == me),
        // we MUST NOT accept — the server rejects self-accepts anyway.
        let mut s = new_session();
        let actions = s.on_event(ask_offer(".8", "alice", "bob"));
        assert!(actions.is_empty(), "expected no action, got {actions:?}");

        let actions = s.on_event(ask_offer_with_game(".9", "alice", "*", "8", "U"));
        assert!(actions.is_empty(), "expected no action, got {actions:?}");
    }

    #[test]
    fn incoming_offer_is_declined_while_own_outgoing_ask_is_pending() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "alice", "bob"));

        let second = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(second.len(), 2);
        assert_eq!(second[0], send("tell /os decline .9"));
    }

    #[test]
    fn third_party_ask_offer_is_ignored() {
        // Broadcast ask between two other players.
        let mut s = new_session();
        let actions = s.on_event(ask_offer(".8", "bob", "carol"));
        assert!(actions.is_empty(), "expected no action, got {actions:?}");
    }

    #[test]
    fn back_to_back_ask_offers_fill_available_slots() {
        let mut s = session_with_max(2);
        let first = s.on_event(ask_offer(".8", "bob", "alice"));
        assert_eq!(first[0], send("tell /os accept .8"));

        let second = s.on_event(ask_offer(".9", "carol", "alice"));
        assert_eq!(second[0], send("tell /os accept .9"));
    }

    #[test]
    fn withdrawn_pending_offer_recovers_after_timeout() {
        let mut s = session_with_max(1);
        assert_eq!(
            s.on_event(ask_offer(".8", "bob", "alice"))[0],
            send("tell /os accept .8")
        );
        let _ = s.on_event(OsEvent::AskWithdrawn {
            id: ".8".to_string(),
        });
        assert_eq!(s.reserved_match_slots(), 1);
        assert_eq!(
            s.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os decline .9")
        );
        assert_eq!(s.pending_accepts[0].id, ".8");
        s.pending_accepts[0].withdrawn_at = Some(
            Instant::now()
                .checked_sub(PENDING_ACCEPT_RECOVERY_TIMEOUT + Duration::from_millis(1))
                .unwrap(),
        );

        assert_eq!(
            s.on_event(ask_offer(".9", "carol", "alice"))[0],
            send("tell /os accept .9")
        );
        assert_eq!(s.pending_accepts.len(), 1);
        assert_eq!(s.pending_accepts[0].id, ".9");
    }

    #[test]
    fn reused_withdrawn_request_id_replaces_old_reservation() {
        let mut s = session_with_max(1);
        let _ = s.on_event(ask_offer(".8", "bob", "alice"));
        let _ = s.on_event(OsEvent::AskWithdrawn {
            id: ".8".to_string(),
        });

        assert_eq!(
            s.on_event(ask_offer(".8", "carol", "alice"))[0],
            send("tell /os accept .8")
        );
        assert_eq!(s.pending_accepts[0].opponent.as_deref(), Some("carol"));
    }
}
