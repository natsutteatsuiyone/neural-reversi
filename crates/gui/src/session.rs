//! Game flow independent of the window: AI moves, hints, the solver, and game
//! analysis over one background [`Engine`].
//!
//! Callers poll [`Session::tick`] and pass the current [`Settings`] to every
//! command. Changes the presenter must surface beyond redrawing are queued as
//! [`Update`]s and collected with [`Session::take_updates`].

use std::collections::HashMap;
use std::time::{Duration, Instant};

use reversi_core::{analysis::GameAnalysisProgress, disc::Disc, square::Square};

use crate::engine::{Engine, EventKind, Progress, Request};
use crate::game::{Game, Position};
use crate::settings::Settings;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activity {
    Initializing,
    Move,
    Hint,
    Solver,
    Analysis,
}

#[derive(Debug, PartialEq)]
pub enum Update {
    /// A move was appended to the history; earlier notices are obsolete.
    Played,
    GameOver {
        black: u32,
        white: u32,
    },
    /// `Disc` had to pass.
    Passed(Disc),
    /// The history cursor moved; earlier notices are obsolete.
    Navigated,
    /// A search iteration was appended to [`Session::log`].
    Logged,
    /// The engine finished an AI move without choosing one.
    NoMove,
    Error(String),
}

struct SavedGame {
    game: Game,
    log: Vec<Progress>,
    analyses: HashMap<usize, GameAnalysisProgress>,
}

enum Mode {
    Live {
        /// Whether AI moves wait for the user to resume play.
        paused: bool,
    },
    /// Solving positions with the live game set aside.
    Solver {
        live: SavedGame,
        stopped: bool,
        cache: HashMap<Position, Vec<Progress>>,
    },
}

pub struct Session {
    engine: Engine,
    game: Game,
    mode: Mode,
    ready: bool,
    initialization_failed: bool,
    activity: Option<Activity>,
    run_id: u64,
    remaining_ms: u64,
    started: Option<Instant>,
    automatic_after: Option<Instant>,
    candidates: Vec<Progress>,
    /// Whether the search behind `candidates` ran to completion.
    candidates_complete: bool,
    log: Vec<Progress>,
    analyses: HashMap<usize, GameAnalysisProgress>,
    search_error: Option<String>,
    updates: Vec<Update>,
}

impl Session {
    /// Starts initializing the engine.
    pub fn new(settings: &Settings) -> Self {
        Self {
            engine: Engine::new(settings.hash_size),
            game: Game::new(Position::initial()),
            mode: Mode::Live { paused: false },
            ready: false,
            initialization_failed: false,
            activity: Some(Activity::Initializing),
            run_id: 0,
            remaining_ms: settings.game_time_limit * 1000,
            started: None,
            automatic_after: None,
            candidates: Vec::new(),
            candidates_complete: false,
            log: Vec::new(),
            analyses: HashMap::new(),
            search_error: None,
            updates: Vec::new(),
        }
    }

    /// The solver's game while solving, otherwise the live game.
    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn activity(&self) -> Option<Activity> {
        self.activity
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn initialization_failed(&self) -> bool {
        self.initialization_failed
    }

    pub fn is_solving(&self) -> bool {
        matches!(self.mode, Mode::Solver { .. })
    }

    /// The live game's position, even while solving.
    pub fn live_position(&self) -> Position {
        match &self.mode {
            Mode::Live { .. } => self.game.position(),
            Mode::Solver { live, .. } => live.game.position(),
        }
    }

    /// Whether AI moves in the live game wait for the user to resume play.
    pub fn is_paused(&self) -> bool {
        matches!(self.mode, Mode::Live { paused: true })
    }

    pub fn is_solver_stopped(&self) -> bool {
        matches!(self.mode, Mode::Solver { stopped: true, .. })
    }

    pub fn search_error(&self) -> Option<&str> {
        self.search_error.as_deref()
    }

    /// The AI's remaining clock, net of the move it is thinking about.
    pub fn clock_ms(&self) -> u64 {
        let elapsed = if self.activity == Some(Activity::Move) {
            self.started
                .map_or(0, |time| time.elapsed().as_millis() as u64)
        } else {
            0
        };
        self.remaining_ms.saturating_sub(elapsed)
    }

    /// How long the current search has been running.
    pub fn search_elapsed(&self) -> Option<Duration> {
        self.started.map(|time| time.elapsed())
    }

    /// Latest result per move of the current search, best first.
    pub fn candidates(&self) -> &[Progress] {
        &self.candidates
    }

    /// Whether the candidates are final rather than a running or stopped search.
    pub fn candidates_complete(&self) -> bool {
        self.candidates_complete
    }

    /// Iterations of the current or last AI move search.
    pub fn log(&self) -> &[Progress] {
        &self.log
    }

    /// Game analysis results keyed by history index.
    pub fn analyses(&self) -> &HashMap<usize, GameAnalysisProgress> {
        &self.analyses
    }

    /// The cursor one step back; the solver also steps over an automatic pass.
    pub fn previous_cursor(&self) -> usize {
        let mut cursor = self.game.cursor.saturating_sub(1);
        if self.is_solving()
            && self
                .game
                .history
                .get(cursor)
                .is_some_and(|m| m.square.is_none())
        {
            cursor = cursor.saturating_sub(1);
        }
        cursor
    }

    pub fn take_updates(&mut self) -> Vec<Update> {
        std::mem::take(&mut self.updates)
    }

    /// Handles engine events and automation due at `now`; returns whether
    /// anything visible may have changed.
    pub fn tick(&mut self, settings: &Settings, now: Instant) -> bool {
        let mut changed = (self.activity == Some(Activity::Move) && settings.timed)
            || self.activity == Some(Activity::Solver);
        if self.automatic_after.is_some_and(|after| now >= after) {
            self.automatic_after = None;
            self.automate(settings);
            changed = true;
        }
        while let Some(event) = self.engine.try_recv() {
            if event.id != self.run_id {
                continue;
            }
            changed = true;
            match event.kind {
                EventKind::Ready => {
                    self.ready = true;
                    self.initialization_failed = false;
                    self.activity = None;
                    self.started = None;
                    self.automate(settings);
                }
                EventKind::Error(error) => {
                    self.search_error = Some(error.clone());
                    self.initialization_failed = !self.ready;
                    self.updates.push(Update::Error(error));
                    self.activity = None;
                    self.started = None;
                    match &mut self.mode {
                        Mode::Live { paused } => *paused = true,
                        Mode::Solver { stopped, .. } => *stopped = true,
                    }
                }
                EventKind::Progress(progress) => {
                    // A best-only solve reports one move; a new best replaces the old.
                    if self.activity == Some(Activity::Solver) && !settings.solver_multi_pv {
                        self.candidates = vec![progress.clone()];
                    } else if let Some(candidate) = self
                        .candidates
                        .iter_mut()
                        .find(|p| p.square == progress.square)
                    {
                        *candidate = progress.clone();
                    } else {
                        self.candidates.push(progress.clone());
                    }
                    self.candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
                    let duplicate = self.log.last().is_some_and(|last| {
                        last.depth == progress.depth
                            && last.probability == progress.probability
                            && last.score == progress.score
                            && last.pv == progress.pv
                    });
                    if self.activity == Some(Activity::Move) {
                        // A repeated iteration (e.g. the report on stop) still
                        // carries newer node counts, so it replaces the row.
                        if duplicate {
                            self.log.pop();
                        } else if self.log.len() == 256 {
                            self.log.remove(0);
                        }
                        self.log.push(progress);
                        self.updates.push(Update::Logged);
                    }
                }
                EventKind::Analysis(analysis) => {
                    self.analyses.insert(analysis.move_index, analysis);
                }
                EventKind::Finished(outcome) => {
                    let activity = self.activity.take();
                    self.started = None;
                    match activity {
                        Some(Activity::Move) => {
                            if settings.timed {
                                self.remaining_ms =
                                    self.remaining_ms.saturating_sub(outcome.elapsed_ms);
                            }
                            if let Some(square) = outcome.best_move {
                                self.play(
                                    square,
                                    outcome.score,
                                    settings.timed.then_some(self.remaining_ms),
                                );
                            } else if !self.game.position().game_over() {
                                self.updates.push(Update::NoMove);
                                self.set_paused(true);
                            }
                            self.automate(settings);
                        }
                        Some(Activity::Solver) => {
                            self.candidates_complete = true;
                            if let Mode::Solver { cache, .. } = &mut self.mode {
                                cache.insert(self.game.position(), self.candidates.clone());
                            }
                        }
                        Some(Activity::Hint) => self.candidates_complete = true,
                        _ => {}
                    }
                }
            }
        }
        changed
    }

    /// Starts the search the current position calls for, if idle.
    fn automate(&mut self, settings: &Settings) {
        if !self.ready
            || self.activity.is_some()
            || self.game.position().game_over()
            || self
                .automatic_after
                .is_some_and(|after| Instant::now() < after)
        {
            return;
        }
        let position = self.game.position();
        match self.mode {
            Mode::Solver { stopped: false, .. } => self.solve(settings),
            Mode::Solver { .. } => {}
            Mode::Live { paused } => {
                if !paused && settings.ai_color() == Some(position.turn) {
                    self.submit(
                        settings,
                        Activity::Move,
                        Request::Move {
                            board: position.board,
                            level: settings.ai_level,
                            remaining_ms: settings.timed.then_some(self.remaining_ms),
                        },
                    );
                } else if settings.hints && settings.ai_color() != Some(position.turn) {
                    self.submit(
                        settings,
                        Activity::Hint,
                        Request::Hint {
                            board: position.board,
                            level: settings.hint_level,
                        },
                    );
                }
            }
        }
    }

    /// Replaces the live game and reinitializes the engine.
    pub fn new_game(&mut self, position: Position, settings: &Settings) {
        self.cancel(settings);
        self.game = Game::new(position);
        self.mode = Mode::Live { paused: false };
        self.analyses.clear();
        self.candidates.clear();
        self.log.clear();
        self.remaining_ms = settings.game_time_limit * 1000;
        self.reinitialize(settings);
    }

    /// Reinitializes the engine, e.g. for a new hash size.
    pub fn reinitialize(&mut self, settings: &Settings) {
        self.submit(
            settings,
            Activity::Initializing,
            Request::Initialize {
                hash_size: settings.hash_size,
            },
        );
    }

    /// Solves `position`, setting the live game aside unless already solving.
    pub fn enter_solver(&mut self, position: Position, settings: &Settings) {
        self.cancel(settings);
        let game = Game::new(position);
        let live = match std::mem::replace(&mut self.mode, Mode::Live { paused: false }) {
            Mode::Live { .. } => SavedGame {
                game: std::mem::replace(&mut self.game, game),
                log: std::mem::take(&mut self.log),
                analyses: std::mem::take(&mut self.analyses),
            },
            Mode::Solver { live, .. } => {
                self.game = game;
                live
            }
        };
        self.mode = Mode::Solver {
            live,
            stopped: false,
            cache: HashMap::new(),
        };
        self.candidates.clear();
        self.automate(settings);
    }

    /// Restores the live game, paused.
    pub fn exit_solver(&mut self, settings: &Settings) {
        self.cancel(settings);
        if let Mode::Solver { live, .. } =
            std::mem::replace(&mut self.mode, Mode::Live { paused: true })
        {
            self.game = live.game;
            self.log = live.log;
            self.analyses = live.analyses;
        }
        self.candidates.clear();
        self.automate(settings);
    }

    /// Discards solver results computed with other solver options.
    pub fn solver_options_changed(&mut self, settings: &Settings) {
        self.cancel(settings);
        if let Mode::Solver { stopped, cache, .. } = &mut self.mode {
            *stopped = false;
            cache.clear();
        }
        self.automate(settings);
    }

    pub fn hints_toggled(&mut self, settings: &Settings) {
        // The solver owns the candidates; hints apply again after exiting it.
        if self.is_solving() {
            return;
        }
        if !settings.hints {
            self.candidates.clear();
        }
        if self.activity == Some(Activity::Hint) {
            self.cancel(settings);
            self.candidates.clear();
        }
        self.automate(settings);
    }

    pub fn hint_level_changed(&mut self, settings: &Settings) {
        if self.activity == Some(Activity::Hint) {
            self.cancel(settings);
        }
        self.automate(settings);
    }

    /// Analyzes the whole game, pausing play, or aborts a running analysis.
    pub fn toggle_analysis(&mut self, settings: &Settings) {
        if self.activity == Some(Activity::Analysis) {
            self.cancel(settings);
        } else if self.ready
            && self.activity != Some(Activity::Move)
            && !self.game.history.is_empty()
        {
            self.set_paused(true);
            self.analyses.clear();
            self.submit(
                settings,
                Activity::Analysis,
                Request::AnalyzeGame {
                    board: self.game.initial.board,
                    moves: self.game.analysis_moves(),
                    level: settings.analysis_level,
                },
            );
        }
    }

    /// Stop or resume. Stopping an AI move commits its best move so far;
    /// stopping anything else pauses play (or the solver) until resumed.
    pub fn toggle_pause(&mut self, settings: &Settings) {
        if let Mode::Solver { stopped, .. } = &mut self.mode {
            *stopped = self.activity == Some(Activity::Solver);
            if *stopped {
                self.cancel(settings);
            } else {
                self.automate(settings);
            }
        } else if self.activity == Some(Activity::Move) {
            self.engine.stop();
        } else if self.activity.is_some() {
            self.cancel(settings);
            self.set_paused(true);
        } else {
            self.set_paused(false);
            self.automate(settings);
        }
    }

    /// Plays the user's move; ignored unless legal and the user's turn.
    pub fn play_move(&mut self, square: Square, settings: &Settings) {
        let position = self.game.position();
        if !self.ready
            || matches!(
                self.activity,
                Some(Activity::Initializing | Activity::Analysis)
            )
            || !position.board.is_legal_move(square)
            || (!self.is_solving() && settings.ai_color() == Some(position.turn))
        {
            return;
        }
        self.cancel(settings);
        self.play(square, None, None);
        self.set_paused(false);
        self.automate(settings);
    }

    /// Moves the history cursor, pausing a live game and restoring its clock.
    /// Ignored while a live game's AI move or analysis is running.
    pub fn navigate(&mut self, cursor: usize, settings: &Settings) {
        let solving = self.is_solving();
        if !solving && matches!(self.activity, Some(Activity::Move | Activity::Analysis)) {
            return;
        }
        self.cancel(settings);
        // Solver breadcrumbs collapse an automatic pass into the preceding move.
        let cursor = if solving
            && self
                .game
                .history
                .get(cursor)
                .is_some_and(|record| record.square.is_none())
        {
            cursor + 1
        } else {
            cursor
        };
        self.game.navigate(cursor);
        self.candidates.clear();
        self.updates.push(Update::Navigated);
        match &mut self.mode {
            Mode::Solver { stopped, .. } => *stopped = false,
            Mode::Live { paused } => {
                *paused = true;
                self.remaining_ms = self.game.history[..self.game.cursor]
                    .iter()
                    .rev()
                    .find_map(|m| m.remaining_ms)
                    .unwrap_or(settings.game_time_limit * 1000);
            }
        }
        self.automate(settings);
    }

    /// Pauses or resumes the live game; the solver has its own stop state.
    fn set_paused(&mut self, value: bool) {
        if let Mode::Live { paused } = &mut self.mode {
            *paused = value;
        }
    }

    fn cancel(&mut self, settings: &Settings) {
        self.search_error = None;
        if self.activity == Some(Activity::Move)
            && settings.timed
            && let Some(started) = self.started.take()
        {
            self.remaining_ms = self
                .remaining_ms
                .saturating_sub(started.elapsed().as_millis() as u64);
        }
        self.engine.cancel();
        self.activity = None;
        self.started = None;
        self.automatic_after = None;
    }

    fn submit(&mut self, settings: &Settings, activity: Activity, request: Request) {
        self.cancel(settings);
        self.run_id = self.engine.submit(request);
        self.activity = Some(activity);
        self.started = Some(Instant::now());
        if activity != Activity::Analysis {
            self.candidates.clear();
            self.candidates_complete = false;
        }
        if activity == Activity::Move {
            self.log.clear();
        }
    }

    fn solve(&mut self, settings: &Settings) {
        let position = self.game.position();
        if let Mode::Solver { cache, .. } = &self.mode
            && let Some(cached) = cache.get(&position)
        {
            self.candidates = cached.clone();
            self.candidates_complete = true;
            return;
        }
        self.submit(
            settings,
            Activity::Solver,
            Request::Solver {
                board: position.board,
                selectivity: settings.solver_selectivity,
                multi_pv: settings.solver_multi_pv,
            },
        );
    }

    fn play(&mut self, square: Square, score: Option<f32>, remaining: Option<u64>) {
        match self.game.play(square, score, remaining) {
            Ok(()) => {
                self.analyses.clear();
                self.updates.push(Update::Played);
                if self.is_solving() {
                    self.candidates.clear();
                    return;
                }
                let now = Instant::now();
                let position = self.game.position();
                if position.game_over() {
                    let (black, white) = position.scores();
                    self.updates.push(Update::GameOver { black, white });
                } else {
                    let passed = self
                        .game
                        .history
                        .last()
                        .filter(|record| record.square.is_none())
                        .map(|record| record.player);
                    let delay = if let Some(player) = passed {
                        self.updates.push(Update::Passed(player));
                        Duration::from_millis(1500)
                    } else {
                        Duration::from_millis(400)
                    };
                    self.automatic_after = Some(now + delay);
                }
            }
            Err(error) => {
                self.updates.push(Update::Error(error));
                self.set_paused(true);
            }
        }
        self.candidates.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::settings::GameMode;

    /// Ticks until `done`, firing pending automation without waiting out its delay.
    fn run_until(session: &mut Session, settings: &Settings, done: impl Fn(&Session) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !done(session) {
            assert!(Instant::now() < deadline, "session did not settle");
            session.tick(settings, Instant::now() + Duration::from_secs(2));
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn ai_replies_to_the_users_move_and_navigation_pauses_play() {
        let settings = Settings {
            game_mode: GameMode::AiWhite,
            timed: false,
            ai_level: 1,
            hash_size: 16,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);

        session.play_move(Square::D3, &settings);
        assert_eq!(session.take_updates(), [Update::Played]);
        run_until(&mut session, &settings, |s| s.game().history.len() == 2);
        let reply = &session.game().history[1];
        assert_eq!(reply.player, Disc::White);
        assert!(reply.score.is_some());

        session.navigate(0, &settings);
        assert!(session.is_paused());
        assert_eq!(session.game().cursor, 0);
        assert_eq!(session.activity(), None);
    }

    #[test]
    fn solver_reuses_cached_results_and_restores_the_live_game() {
        let settings = Settings {
            game_mode: GameMode::Human,
            hash_size: 16,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);
        session.play_move(Square::D3, &settings);

        // 54 moves of a complete game leave six empties for an instant exact solve.
        let game = "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8b6e8b5g2d8b7a6h2e7d7c8a8a7b8c7h1b2b1";
        let position = Position::from_transcript(&game[..108]).unwrap();
        session.enter_solver(position, &settings);
        run_until(&mut session, &settings, |s| s.activity().is_none());
        let best = session.candidates()[0].square.unwrap();

        session.play_move(best, &settings);
        run_until(&mut session, &settings, |s| s.activity().is_none());
        session.navigate(session.previous_cursor(), &settings);
        assert_eq!(session.activity(), None, "a cached result needs no search");
        assert_eq!(session.candidates()[0].square, Some(best));

        session.exit_solver(&settings);
        assert!(!session.is_solving());
        assert!(session.is_paused());
        assert_eq!(session.game().history.len(), 1);
    }

    #[test]
    fn best_only_solver_keeps_only_the_latest_best_move() {
        let settings = Settings {
            game_mode: GameMode::Human,
            hash_size: 16,
            solver_multi_pv: false,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);

        // Inject successive engine reports through the same queue tick consumes.
        // The best move changes explicitly, independent of weights or search timing.
        session.activity = Some(Activity::Solver);
        for (square, score, probability) in [(Square::D3, 2.0, 73), (Square::C4, 4.0, 100)] {
            let progress = Progress {
                square: Some(square),
                score,
                depth: 18,
                target_depth: 18,
                probability,
                nodes: 123,
                nps: 456.0,
                pv: square.to_string(),
                is_endgame: true,
            };
            session
                .engine
                .emit_for_test(session.run_id, EventKind::Progress(progress.clone()));
            session.tick(&settings, Instant::now());
            assert_eq!(session.candidates(), &[progress]);
        }
    }

    #[test]
    fn solver_below_full_selectivity_completes_at_its_target() {
        let settings = Settings {
            game_mode: GameMode::Human,
            hash_size: 16,
            solver_selectivity: 0,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);

        let game =
            "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8";
        session.enter_solver(Position::from_transcript(&game[..84]).unwrap(), &settings);
        let legal: std::collections::HashSet<_> =
            session.game().position().board.get_moves().iter().collect();
        assert!(!legal.is_empty());
        run_until(&mut session, &settings, |s| s.activity().is_none());
        let candidates: std::collections::HashSet<_> = session
            .candidates()
            .iter()
            .map(|c| c.square.expect("legal solver move"))
            .collect();
        assert_eq!(candidates, legal);
        assert_eq!(session.candidates().len(), legal.len());
        assert!(session.candidates().iter().all(|c| c.probability == 73));
        assert!(session.candidates_complete());
    }

    #[test]
    fn toggling_hints_keeps_stopped_solver_results() {
        let mut settings = Settings {
            game_mode: GameMode::Human,
            hash_size: 16,
            hints: true,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);

        let game =
            "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8";
        session.enter_solver(Position::from_transcript(&game[..84]).unwrap(), &settings);
        run_until(&mut session, &settings, |s| !s.candidates().is_empty());
        assert_eq!(session.activity(), Some(Activity::Solver));
        session.toggle_pause(&settings);
        assert!(!session.candidates_complete());
        let stopped = session.candidates().to_vec();

        settings.hints = false;
        session.hints_toggled(&settings);
        assert_eq!(session.candidates(), stopped);
        assert_eq!(session.activity(), None);
    }

    #[test]
    fn stopping_solver_before_results_preserves_legal_moves_and_browsed_game() {
        let settings = Settings {
            game_mode: GameMode::Human,
            hash_size: 16,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);
        session.play_move(Square::D3, &settings);
        session.play_move(Square::C5, &settings);
        session.navigate(1, &settings);
        let browsed = session.game().position();
        session.enter_solver(browsed, &settings);
        session.toggle_pause(&settings);
        assert!(session.is_solver_stopped());
        assert!(session.candidates().is_empty());
        assert!(session.game().position().board.has_legal_moves());
        assert_eq!(session.game().initial, browsed);
        session.exit_solver(&settings);
        assert_eq!(session.game().cursor, 1);
        assert_eq!(session.game().history.len(), 2);
        assert_eq!(session.game().position(), browsed);
        session.navigate(2, &settings);
        assert_eq!(session.game().history.len(), 2);
        assert!(session.is_paused());
    }
}
