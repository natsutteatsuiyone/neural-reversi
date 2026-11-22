use std::collections::HashMap;
use std::time::{Duration, Instant};

use reversi_core::{
    analysis::GameAnalysisProgress, player_color::PlayerColor, probcut::Selectivity, score::Score,
    square::Square,
};

use crate::engine::{Engine, EngineError, EventKind, Progress, Request};
use crate::game::{Game, MoveRecord, PlayError, Position};
use crate::settings::Settings;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    Engine(EngineError),
    Play(PlayError),
}

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
    Played,
    GameOver { black: u32, white: u32 },
    Passed(PlayerColor),
    Navigated,
    Logged,
    NoMove,
    Error(SessionError),
}

#[derive(Default)]
struct Candidates {
    items: Vec<Progress>,
    complete: bool,
}

impl Candidates {
    fn clear(&mut self) {
        self.items.clear();
        self.complete = false;
    }

    fn record(&mut self, progress: Progress, best_only: bool) {
        if best_only {
            self.items = vec![progress];
        } else if let Some(candidate) = self
            .items
            .iter_mut()
            .find(|candidate| candidate.square == progress.square)
        {
            *candidate = progress;
        } else {
            self.items.push(progress);
        }
        self.items
            .sort_by_key(|candidate| std::cmp::Reverse(candidate.score));
    }
}

struct SavedGame {
    game: Game,
    log: Vec<Progress>,
    analyses: HashMap<usize, GameAnalysisProgress>,
}

enum Mode {
    Live {
        paused: bool,
        candidates: Candidates,
    },
    Solver {
        live: SavedGame,
        stopped: bool,
        cache: HashMap<Position, Vec<Progress>>,
        candidates: Candidates,
    },
}

impl Mode {
    fn candidates(&self) -> &Candidates {
        match self {
            Self::Live { candidates, .. } | Self::Solver { candidates, .. } => candidates,
        }
    }

    fn candidates_mut(&mut self) -> &mut Candidates {
        match self {
            Self::Live { candidates, .. } | Self::Solver { candidates, .. } => candidates,
        }
    }
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
    displayed_second: Option<u64>,
    automatic_after: Option<Instant>,
    log: Vec<Progress>,
    analyses: HashMap<usize, GameAnalysisProgress>,
    search_error: Option<SessionError>,
    updates: Vec<Update>,
}

impl Session {
    pub fn new(settings: &Settings) -> Self {
        Self {
            engine: Engine::new(settings.hash_size),
            game: Game::new(Position::initial()),
            mode: Mode::Live {
                paused: false,
                candidates: Candidates::default(),
            },
            ready: false,
            initialization_failed: false,
            activity: Some(Activity::Initializing),
            run_id: 0,
            remaining_ms: settings.game_time_limit * 1000,
            started: None,
            displayed_second: None,
            automatic_after: None,
            log: Vec::new(),
            analyses: HashMap::new(),
            search_error: None,
            updates: Vec::new(),
        }
    }

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

    pub fn live_position(&self) -> Position {
        match &self.mode {
            Mode::Live { .. } => self.game.position(),
            Mode::Solver { live, .. } => live.game.position(),
        }
    }

    pub fn is_paused(&self) -> bool {
        matches!(self.mode, Mode::Live { paused: true, .. })
    }

    pub fn is_solver_stopped(&self) -> bool {
        matches!(self.mode, Mode::Solver { stopped: true, .. })
    }

    pub fn search_error(&self) -> Option<&SessionError> {
        self.search_error.as_ref()
    }

    pub fn clock_ms(&self) -> u64 {
        let elapsed = if self.activity == Some(Activity::Move) {
            self.started
                .map_or(0, |time| time.elapsed().as_millis() as u64)
        } else {
            0
        };
        self.remaining_ms.saturating_sub(elapsed)
    }

    pub fn search_elapsed(&self) -> Option<Duration> {
        self.started.map(|time| time.elapsed())
    }

    pub fn candidates(&self) -> &[Progress] {
        &self.mode.candidates().items
    }

    pub fn candidates_complete(&self) -> bool {
        self.mode.candidates().complete
    }

    pub fn log(&self) -> &[Progress] {
        &self.log
    }

    pub fn analyses(&self) -> &HashMap<usize, GameAnalysisProgress> {
        &self.analyses
    }

    pub fn previous_cursor(&self) -> usize {
        let mut cursor = self.game.cursor.saturating_sub(1);
        if self.game.history.get(cursor).is_some_and(is_pass) {
            cursor = cursor.saturating_sub(1);
        }
        cursor
    }

    pub fn take_updates(&mut self) -> Vec<Update> {
        std::mem::take(&mut self.updates)
    }

    pub fn tick(&mut self, settings: &Settings, now: Instant) -> bool {
        let elapsed = self.started.map_or(Duration::ZERO, |started| {
            now.saturating_duration_since(started)
        });
        let second = match self.activity {
            Some(Activity::Move) if settings.timed => {
                Some(self.remaining_ms.saturating_sub(elapsed.as_millis() as u64) / 1000)
            }
            Some(Activity::Solver) => Some(elapsed.as_secs()),
            _ => None,
        };
        let mut changed = second.is_some() && second != self.displayed_second;
        self.displayed_second = second;
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
                    let error = SessionError::Engine(error);
                    self.search_error = Some(error.clone());
                    self.initialization_failed = !self.ready;
                    self.updates.push(Update::Error(error));
                    self.activity = None;
                    self.started = None;
                    match &mut self.mode {
                        Mode::Live { paused, .. } => *paused = true,
                        Mode::Solver { stopped, .. } => *stopped = true,
                    }
                }
                EventKind::Progress(progress) => {
                    let best_only =
                        self.activity == Some(Activity::Solver) && !settings.solver_multi_pv;
                    self.mode
                        .candidates_mut()
                        .record(progress.clone(), best_only);
                    if self.activity == Some(Activity::Move) {
                        self.record_log_iteration(progress);
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
                            if let Mode::Solver {
                                cache, candidates, ..
                            } = &mut self.mode
                            {
                                candidates.complete = true;
                                cache.insert(self.game.position(), candidates.items.clone());
                            }
                        }
                        Some(Activity::Hint) => self.mode.candidates_mut().complete = true,
                        Some(Activity::Analysis)
                            if matches!(
                                &self.mode,
                                Mode::Live { candidates, .. } if !candidates.complete
                            ) =>
                        {
                            self.automate(settings);
                        }
                        _ => {}
                    }
                }
            }
        }
        changed
    }

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
            Mode::Live { paused, .. } => {
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

    pub fn new_game(&mut self, position: Position, settings: &Settings) {
        self.cancel(settings);
        self.game = Game::new(position);
        self.mode = Mode::Live {
            paused: false,
            candidates: Candidates::default(),
        };
        self.analyses.clear();
        self.log.clear();
        self.remaining_ms = settings.game_time_limit * 1000;
        self.reinitialize(settings);
    }

    pub fn reinitialize(&mut self, settings: &Settings) {
        self.submit(
            settings,
            Activity::Initializing,
            Request::Initialize {
                hash_size: settings.hash_size,
            },
        );
    }

    pub fn enter_solver(&mut self, position: Position, settings: &Settings) {
        self.cancel(settings);
        let game = Game::new(position);
        let live = match std::mem::replace(
            &mut self.mode,
            Mode::Live {
                paused: false,
                candidates: Candidates::default(),
            },
        ) {
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
            candidates: Candidates::default(),
        };
        self.automate(settings);
    }

    pub fn exit_solver(&mut self, settings: &Settings) {
        self.cancel(settings);
        if let Mode::Solver { live, .. } = std::mem::replace(
            &mut self.mode,
            Mode::Live {
                paused: true,
                candidates: Candidates::default(),
            },
        ) {
            self.game = live.game;
            self.log = live.log;
            self.analyses = live.analyses;
        }
        self.automate(settings);
    }

    pub fn solver_options_changed(&mut self, settings: &Settings) {
        self.cancel(settings);
        if let Mode::Solver { stopped, cache, .. } = &mut self.mode {
            *stopped = false;
            cache.clear();
        }
        self.automate(settings);
    }

    pub fn hints_toggled(&mut self, settings: &Settings) {
        let Mode::Live { candidates, .. } = &mut self.mode else {
            return;
        };
        if !settings.hints {
            candidates.clear();
        }
        if self.activity == Some(Activity::Hint) {
            self.cancel(settings);
            self.mode.candidates_mut().clear();
        }
        self.automate(settings);
    }

    pub fn hint_level_changed(&mut self, settings: &Settings) {
        let Mode::Live { candidates, .. } = &mut self.mode else {
            return;
        };
        candidates.complete = false;
        if self.activity == Some(Activity::Hint) {
            self.cancel(settings);
        }
        self.automate(settings);
    }

    pub fn toggle_analysis(&mut self, settings: &Settings) {
        if self.activity == Some(Activity::Analysis) {
            self.cancel(settings);
            if matches!(&self.mode, Mode::Live { candidates, .. } if !candidates.complete) {
                self.automate(settings);
            }
        } else if self.ready
            && !matches!(self.activity, Some(Activity::Initializing | Activity::Move))
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
        match &mut self.mode {
            Mode::Live { paused, .. } => *paused = false,
            Mode::Solver { stopped, .. } => *stopped = false,
        }
        self.automate(settings);
    }

    pub fn navigate(&mut self, cursor: usize, settings: &Settings) {
        let solving = self.is_solving();
        if !solving && matches!(self.activity, Some(Activity::Move | Activity::Analysis)) {
            return;
        }
        self.cancel(settings);
        let cursor = if self.game.history.get(cursor).is_some_and(is_pass) {
            cursor + 1
        } else {
            cursor
        };
        self.game.navigate(cursor);
        self.mode.candidates_mut().clear();
        self.updates.push(Update::Navigated);
        match &mut self.mode {
            Mode::Solver { stopped, .. } => *stopped = false,
            Mode::Live { paused, .. } => {
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

    fn set_paused(&mut self, value: bool) {
        if let Mode::Live { paused, .. } = &mut self.mode {
            *paused = value;
        }
    }

    fn record_log_iteration(&mut self, progress: Progress) {
        if let Some(last) = self.log.last_mut().filter(|last| {
            last.depth == progress.depth
                && last.probability == progress.probability
                && last.score == progress.score
                && last.pv == progress.pv
        }) {
            *last = progress;
            return;
        }
        if self.log.len() == 256 {
            self.log.remove(0);
        }
        self.log.push(progress);
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
            self.mode.candidates_mut().clear();
        }
        if activity == Activity::Move {
            self.log.clear();
        }
    }

    fn solve(&mut self, settings: &Settings) {
        let position = self.game.position();
        if let Mode::Solver {
            cache, candidates, ..
        } = &mut self.mode
            && let Some(cached) = cache.get(&position)
        {
            candidates.items = cached.clone();
            candidates.complete = true;
            return;
        }
        self.submit(
            settings,
            Activity::Solver,
            Request::Solver {
                board: position.board,
                selectivity: Selectivity::ENDGAME_LADDER[settings.solver_selectivity],
                multi_pv: settings.solver_multi_pv,
            },
        );
    }

    fn play(&mut self, square: Square, score: Option<Score>, remaining: Option<u64>) {
        self.mode.candidates_mut().clear();
        match self.game.play(square, score, remaining) {
            Ok(()) => {
                self.analyses.clear();
                self.updates.push(Update::Played);
                if self.is_solving() {
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
                        .filter(|record| is_pass(record))
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
                self.updates.push(Update::Error(SessionError::Play(error)));
                self.set_paused(true);
            }
        }
    }
}

fn is_pass(record: &MoveRecord) -> bool {
    record.square.is_none()
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use reversi_core::score::DiscScore;

    use crate::settings::GameMode;

    fn run_until(session: &mut Session, settings: &Settings, done: impl Fn(&Session) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !done(session) {
            assert!(Instant::now() < deadline, "session did not settle");
            session.tick(settings, Instant::now() + Duration::from_secs(2));
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn live_history_skips_forced_pass_and_resumes_play() {
        for game_mode in [GameMode::Human, GameMode::AiWhite, GameMode::AiBlack] {
            let mut settings = Settings {
                game_mode: GameMode::Human,
                timed: false,
                hints: false,
                ai_level: 1,
                hash_size: 16,
                ..Settings::default()
            };
            let mut session = Session::new(&settings);
            run_until(&mut session, &settings, Session::is_ready);
            let initial = Position::from_board_string(
                "XO------\n--------\n--------\n--------\n--------\n--------\n--------\nXO------",
                PlayerColor::Black,
            )
            .unwrap();
            session.game = Game::new(initial);
            session.game.play(Square::C1, None, None).unwrap();
            settings.game_mode = game_mode;

            session.navigate(1, &settings);
            assert_eq!(session.game.cursor, 2);
            assert_eq!(session.game.position().turn, PlayerColor::Black);
            assert!(session.game.position().board.is_legal_move(Square::C8));
            assert!(session.is_paused());
            session.navigate(session.previous_cursor(), &settings);
            assert_eq!(session.game.cursor, 0);
            session.navigate(session.game.cursor + 1, &settings);
            assert_eq!(session.game.cursor, 2);

            if game_mode == GameMode::AiBlack {
                session.toggle_pause(&settings);
                run_until(&mut session, &settings, |s| s.activity().is_none());
            } else {
                session.play_move(Square::C8, &settings);
            }
            assert_eq!(session.game.cursor, 3);
            assert_eq!(session.game.history[1].square, None);
            assert_eq!(session.game.history[2].square, Some(Square::C8));
            assert_eq!(session.game.history[2].player, PlayerColor::Black);
            assert!(session.game.position().game_over());
            assert!(!session.take_updates().contains(&Update::NoMove));
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
        assert_eq!(reply.player, PlayerColor::White);
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

        session.activity = Some(Activity::Solver);
        for (square, discs, probability) in [(Square::D3, 2, 73), (Square::C4, 4, 100)] {
            let progress = Progress {
                square: Some(square),
                score: Score::from_disc_diff(DiscScore::new(discs)),
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
        let legal: std::collections::HashSet<_> = session
            .game()
            .position()
            .board
            .legal_moves()
            .iter()
            .collect();
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
    fn analysis_resumes_interrupted_hints() {
        for (waiting_for_hint, abort) in
            [(true, false), (true, true), (false, false), (false, true)]
        {
            let settings = Settings {
                game_mode: GameMode::Human,
                hints: true,
                hint_level: 4,
                analysis_level: 1,
                hash_size: 16,
                ..Settings::default()
            };
            let mut session = Session::new(&settings);
            run_until(&mut session, &settings, Session::is_ready);
            if waiting_for_hint {
                run_until(&mut session, &settings, |s| s.candidates_complete());
            }
            session.play_move(Square::D3, &settings);
            if waiting_for_hint {
                assert_eq!(session.activity(), None);
            } else {
                session.navigate(1, &settings);
                assert_eq!(session.activity(), Some(Activity::Hint));
            }
            session.toggle_analysis(&settings);
            assert_eq!(session.activity(), Some(Activity::Analysis));
            if abort {
                session.toggle_analysis(&settings);
            } else {
                run_until(&mut session, &settings, |s| {
                    s.activity() != Some(Activity::Analysis)
                });
            }
            assert!(
                session.activity() == Some(Activity::Hint) || session.candidates_complete(),
                "hints were not resumed after analysis (abort={abort})"
            );
            run_until(&mut session, &settings, |s| s.activity().is_none());
            assert!(session.candidates_complete());
            assert_eq!(
                session.candidates().len(),
                session.game().position().board.legal_moves().count() as usize
            );
            assert!(session.is_paused());
        }
    }

    #[test]
    fn hints_reenabled_during_analysis_are_recomputed() {
        for abort in [false, true] {
            let mut settings = Settings {
                game_mode: GameMode::Human,
                hints: true,
                hint_level: 1,
                analysis_level: 1,
                hash_size: 16,
                ..Settings::default()
            };
            let mut session = Session::new(&settings);
            run_until(&mut session, &settings, Session::is_ready);
            session.play_move(Square::D3, &settings);
            run_until(&mut session, &settings, |s| s.candidates_complete());
            session.toggle_analysis(&settings);
            settings.hints = false;
            session.hints_toggled(&settings);
            settings.hints = true;
            session.hints_toggled(&settings);
            if abort {
                session.toggle_analysis(&settings);
            } else {
                run_until(&mut session, &settings, |s| {
                    s.activity() != Some(Activity::Analysis)
                });
            }
            run_until(&mut session, &settings, |s| s.activity().is_none());
            assert!(session.candidates_complete());
            assert_eq!(
                session.candidates().len(),
                session.game().position().board.legal_moves().count() as usize
            );
        }
    }

    #[test]
    fn analysis_keeps_finished_hints_and_does_not_start_unrequested_searches() {
        for abort in [false, true] {
            for case in ["complete", "disabled", "ai_turn", "solver"] {
                let mut settings = Settings {
                    game_mode: GameMode::Human,
                    hints: true,
                    hint_level: 1,
                    analysis_level: 1,
                    hash_size: 16,
                    ..Settings::default()
                };
                let mut session = Session::new(&settings);
                run_until(&mut session, &settings, Session::is_ready);
                session.play_move(Square::D3, &settings);
                session.navigate(1, &settings);
                match case {
                    "complete" => {
                        run_until(&mut session, &settings, |s| s.activity().is_none());
                        assert!(session.candidates_complete());
                    }
                    "disabled" => {
                        settings.hints = false;
                        session.hints_toggled(&settings);
                    }
                    "ai_turn" => settings.game_mode = GameMode::AiWhite,
                    "solver" => {
                        let game = "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8b6e8b5g2d8b7a6h2e7d7c8a8";
                        session.enter_solver(Position::from_transcript(game).unwrap(), &settings);
                        let square = session
                            .game()
                            .position()
                            .board
                            .legal_moves()
                            .iter()
                            .next()
                            .unwrap();
                        session.play_move(square, &settings);
                    }
                    _ => unreachable!(),
                }
                let candidates = session.candidates().to_vec();
                session.toggle_analysis(&settings);
                assert_eq!(session.activity(), Some(Activity::Analysis));
                if abort {
                    session.toggle_analysis(&settings);
                } else {
                    run_until(&mut session, &settings, |s| {
                        s.activity() != Some(Activity::Analysis)
                    });
                }
                assert_eq!(session.activity(), None, "case={case}, abort={abort}");
                assert_eq!(session.candidates(), candidates);
            }
        }
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

    #[test]
    fn tick_reports_running_clocks_only_when_their_displayed_second_changes() {
        let settings = Settings {
            game_mode: GameMode::Human,
            hash_size: 16,
            timed: true,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);

        let started = Instant::now();
        let at = |ms| started + Duration::from_millis(ms);
        session.activity = Some(Activity::Move);
        session.remaining_ms = 10_000;
        session.started = Some(started);
        assert!(session.tick(&settings, at(0)));
        assert!(session.tick(&settings, at(1)));
        assert!(!session.tick(&settings, at(50)));
        assert!(!session.tick(&settings, at(1_000)));
        assert!(session.tick(&settings, at(1_001)));

        session.activity = Some(Activity::Solver);
        assert!(session.tick(&settings, at(1_100)));
        assert!(!session.tick(&settings, at(1_900)));
        assert!(session.tick(&settings, at(2_000)));
        assert!(!session.tick(&settings, at(2_200)));
    }

    #[test]
    fn hint_level_change_during_analysis_and_navigation_clear_candidates_complete() {
        let mut settings = Settings {
            game_mode: GameMode::Human,
            hints: true,
            hint_level: 1,
            analysis_level: 1,
            hash_size: 16,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);
        session.play_move(Square::D3, &settings);
        run_until(&mut session, &settings, Session::candidates_complete);

        session.toggle_analysis(&settings);
        assert_eq!(session.activity(), Some(Activity::Analysis));
        settings.hint_level = 2;
        session.hint_level_changed(&settings);
        assert!(!session.candidates_complete());
        session.toggle_analysis(&settings);
        assert_eq!(session.activity(), Some(Activity::Hint));
        run_until(&mut session, &settings, Session::candidates_complete);

        settings.hints = false;
        session.navigate(0, &settings);
        assert!(!session.candidates_complete());
        assert!(session.candidates().is_empty());
    }

    #[test]
    fn playing_a_move_while_solver_is_stopped_resumes_solving() {
        let settings = Settings {
            game_mode: GameMode::Human,
            hash_size: 16,
            ..Settings::default()
        };
        let mut session = Session::new(&settings);
        run_until(&mut session, &settings, Session::is_ready);

        let game = "e6f4c3c4d3d6e3d2f3f5c1c2b4b3a3e2c5c6f6g5g4a2a1a4f2h5g3f7h6h3f8f1e1d1h4h7a5g7h8g6g1g8b6e8b5g2d8b7a6h2e7d7c8a8a7b8c7h1b2b1";
        session.enter_solver(Position::from_transcript(&game[..108]).unwrap(), &settings);
        session.toggle_pause(&settings);
        assert!(session.is_solver_stopped());

        let legal = session
            .game()
            .position()
            .board
            .legal_moves()
            .iter()
            .next()
            .unwrap();
        session.play_move(legal, &settings);
        assert!(!session.is_solver_stopped());
        run_until(&mut session, &settings, |s| s.activity().is_none());
        assert!(session.candidates_complete());
        assert!(!session.candidates().is_empty());
    }
}
