use std::time::{Duration, Instant};

use nsh_core::prelude::*;
use reversi_core::disc::Disc;
use reversi_core::square::Square;

use super::setup::SetupTab;
use super::{Action, Modal, ReversiApp, Ui};
use crate::game::Position;
use crate::session::Update;
use crate::settings::Settings;

enum ScrollUpdate {
    HistoryEnd,
    HistoryAt(usize),
    LogEnd,
}

impl Ui {
    pub(super) fn tick(self) {
        let (changed, updates) = self.app.with(|app| {
            let now = Instant::now();
            let mut changed = app.notification.tick(now);
            if app.copied_until.is_some_and(|until| now >= until) {
                app.copied_until = None;
                changed = true;
            }
            changed |= app.session.tick(&app.settings, now);
            (changed, app.present_updates())
        });
        if changed || !updates.is_empty() {
            self.changed
                .update(|revision| *revision = revision.wrapping_add(1));
        }
        self.scroll_updates(updates);
    }

    pub(super) fn action(self, action: Action) {
        let (previous, previous_locale) = self.read(|app| (app.modal, app.settings.locale()));
        let return_to = self.focused.get_untracked();
        let (copy, updates, next) = self.app.with(|app| {
            let copy = app.apply_action(action);
            (copy, app.present_updates(), app.modal)
        });
        if self.read(|app| app.settings.locale()) != previous_locale {
            self.locale_revision
                .update(|revision| *revision = revision.wrapping_add(1));
        }
        if previous.is_none() && next.is_some() {
            self.focus_registry
                .with(|registry| registry.return_to = return_to);
        }
        self.changed
            .update(|revision| *revision = revision.wrapping_add(1));
        if let Some(copy) = copy {
            drop(set_clipboard(copy));
        }
        self.scroll_updates(updates);
        if next != previous {
            if next.is_some() {
                set_timeout(Duration::from_millis(1), move || {
                    if self.read(|app| app.modal) == next {
                        self.focused.set(None);
                        self.focus_step(false);
                    }
                });
            } else if let Some(id) = self
                .focus_registry
                .with(|registry| registry.return_to.take())
            {
                self.focus(&id);
            }
        }
    }

    fn scroll_updates(self, updates: Vec<ScrollUpdate>) {
        if updates.is_empty() {
            return;
        }
        set_timeout(Duration::from_millis(1), move || {
            let (history, log) = self.read(|app| (app.history_scroll, app.log_scroll));
            for update in &updates {
                match *update {
                    ScrollUpdate::HistoryEnd => history.scroll_to_end(false),
                    ScrollUpdate::HistoryAt(index) => {
                        if let Some(row) = self.focus_registry.with(|registry| {
                            registry
                                .entries
                                .iter()
                                .find(|entry| {
                                    entry.id == format!("history-{index}") && entry.scope.is_none()
                                })
                                .cloned()
                        }) {
                            row.element.reveal(
                                nsh_core::RevealOptions::default()
                                    .vertical(nsh_core::ScrollAlignment::Center)
                                    .animated(false),
                            );
                        }
                    }
                    ScrollUpdate::LogEnd => log.scroll_to_end(false),
                }
            }
        });
    }
}

impl ReversiApp {
    fn present_updates(&mut self) -> Vec<ScrollUpdate> {
        let mut effects = Vec::new();
        for update in self.session.take_updates() {
            match update {
                Update::Played => {
                    self.notification.clear();
                    self.analysis_view.hover = None;
                    self.analysis_view.pointer = None;
                    effects.push(ScrollUpdate::HistoryEnd);
                }
                Update::GameOver { black, white } => {
                    let (message, description) = self.game_result(black, white);
                    self.notification.show(
                        message,
                        description,
                        "trophy.svg",
                        Instant::now() + Duration::from_secs(5),
                    );
                }
                Update::Passed(player) => self.notification.show(
                    tr!("player_passed", player = self.color(player)).to_string(),
                    String::new(),
                    "info.svg",
                    Instant::now() + Duration::from_millis(1500),
                ),
                Update::Navigated => {
                    self.notification.clear();
                    effects.push(ScrollUpdate::HistoryAt(
                        self.session.game().cursor.saturating_sub(1),
                    ));
                }
                Update::Logged => effects.push(ScrollUpdate::LogEnd),
                Update::NoMove => self
                    .notification
                    .show_error(tr!("engine_no_move").to_string(), String::new()),
                Update::Error(error) => {
                    let (message, description) = self.format_error(&error);
                    self.notification.show_error(message, description);
                }
            }
        }
        effects
    }

    fn apply_action(&mut self, action: Action) -> Option<String> {
        if self.modal.is_some()
            && matches!(
                action,
                Action::NewGame
                    | Action::Solver
                    | Action::Settings
                    | Action::Hint
                    | Action::Copy
                    | Action::Navigate(_)
                    | Action::Previous
                    | Action::Next
                    | Action::Latest
                    | Action::Pause
                    | Action::AnalyzeAndChart
                    | Action::AnalysisPanel
                    | Action::ExitSolver
                    | Action::Chart(_)
            )
        {
            return None;
        }
        match action {
            Action::NewGame | Action::Solver => {
                let solver = matches!(action, Action::Solver);
                self.analysis_view.level_menu = false;
                self.draft = self.settings.clone();
                self.setup.reset();
                if solver {
                    self.setup.set_position(self.session.game().position());
                }
                self.modal = Some(if solver {
                    Modal::SolverSetup
                } else {
                    Modal::NewGame
                });
                self.notification.clear();
            }
            Action::Setup => self.modal = Some(Modal::Setup),
            Action::Back => self.modal = Some(Modal::NewGame),
            Action::Start => {
                let solver = self.modal == Some(Modal::SolverSetup);
                let position = if matches!(self.modal, Some(Modal::Setup | Modal::SolverSetup)) {
                    self.setup.position()
                } else {
                    Ok(Position::initial())
                };
                match position {
                    Ok(position) => {
                        self.modal = None;
                        if solver {
                            self.analysis_view.close();
                            self.settings.solver_selectivity = self.draft.solver_selectivity;
                            self.settings.solver_multi_pv = self.draft.solver_multi_pv;
                            self.notification.clear();
                            self.save();
                            self.session.enter_solver(position, &self.settings);
                        } else {
                            self.settings = self.draft.clone();
                            rust_i18n::set_locale(self.settings.locale());
                            self.notification.clear();
                            self.save();
                            self.analysis_view.hover = None;
                            self.analysis_view.pointer = None;
                            self.session.new_game(position, &self.settings);
                        }
                    }
                    Err(error) => self.setup.error = Some(error),
                }
            }
            Action::Close => {
                self.modal = None;
                self.license.end_drag();
                self.menu_section = None;
                self.notification.clear();
            }
            Action::Settings => {
                self.analysis_view.level_menu = false;
                self.menu_section = None;
                self.modal = Some(Modal::Settings);
            }
            Action::MenuSection(section) => self.menu_section = section,
            Action::About => {
                self.analysis_view.level_menu = false;
                self.modal = Some(Modal::About);
                self.menu_section = None;
                self.license.load(false);
            }
            Action::License(third_party) => self.license.load(third_party),
            Action::Mode(mode) => self.draft.game_mode = mode,
            Action::Timed(timed) => self.draft.timed = timed,
            Action::Hint => {
                self.settings.hints = !self.settings.hints;
                self.save();
                self.session.hints_toggled(&self.settings);
            }
            Action::AnalyzeAndChart => {
                self.analysis_view.level_menu = false;
                if self.session.activity() != Some(crate::session::Activity::Analysis) {
                    self.analysis_view.chart = true;
                }
                self.session.toggle_analysis(&self.settings);
            }
            Action::AnalysisPanel => {
                if self.compact_layout {
                    self.compact_analysis_open = !self.compact_analysis_open;
                } else {
                    self.settings.analysis_open = !self.settings.analysis_open;
                    self.save();
                }
                if !self.analysis_open() {
                    self.analysis_view.close();
                }
            }
            Action::Chart(chart) => {
                self.analysis_view.chart = chart;
                if !chart {
                    self.analysis_view.hover = None;
                    self.analysis_view.pointer = None;
                }
            }
            Action::Hash(hash_size) => {
                if hash_size != self.settings.hash_size {
                    self.settings.hash_size = hash_size;
                    self.save();
                    self.session.reinitialize(&self.settings);
                }
            }
            Action::Language(language) => {
                self.settings.language = language;
                rust_i18n::set_locale(self.settings.locale());
                self.save();
            }
            Action::Navigate(cursor) => self.session.navigate(cursor, &self.settings),
            Action::Previous => self
                .session
                .navigate(self.session.previous_cursor(), &self.settings),
            Action::Next => self.session.navigate(
                (self.session.game().cursor + 1).min(self.session.game().history.len()),
                &self.settings,
            ),
            Action::Latest => self
                .session
                .navigate(self.session.game().history.len(), &self.settings),
            Action::Copy => {
                self.copied_until = Some(Instant::now() + Duration::from_millis(1500));
                return Some(self.session.game().transcript());
            }
            Action::Pause => self.session.toggle_pause(&self.settings),
            Action::ExitSolver => {
                self.session.exit_solver(&self.settings);
                self.analysis_view.close();
            }
            Action::Selectivity(index) => {
                self.set_solver_option(|settings| settings.solver_selectivity = index)
            }
            Action::MultiPv(multi_pv) => {
                self.set_solver_option(|settings| settings.solver_multi_pv = multi_pv)
            }
            Action::SetupTab(tab) => {
                self.setup.tab = tab;
                self.setup.refresh();
            }
            Action::Turn(turn) => {
                self.setup.turn = turn;
                self.setup.refresh();
            }
            Action::ClearBoard => {
                self.setup.cells.fill(Disc::Empty);
                self.setup.error = None;
            }
            Action::InitialBoard => self.setup.show(Position::initial()),
            Action::Cell(index) => {
                if matches!(self.modal, Some(Modal::Setup | Modal::SolverSetup))
                    && self.setup.tab == SetupTab::Manual
                {
                    self.setup.cells[index] = match self.setup.cells[index] {
                        Disc::Empty => Disc::Black,
                        Disc::Black => Disc::White,
                        Disc::White => Disc::Empty,
                    };
                    self.setup.error = None;
                } else if self.modal.is_none()
                    && let Some(square) = Square::from_u8(index as u8)
                {
                    self.session.play_move(square, &self.settings);
                }
            }
            Action::Slider(kind, value) => kind.set_value(self, value),
            Action::SetupTranscript(text) => {
                self.setup.transcript = text;
                self.setup.refresh();
            }
            Action::SetupBoard(text) => {
                self.setup.board = text;
                self.setup.refresh();
            }
            Action::AnalysisLevelMenu(open) => self.analysis_view.level_menu = open,
        }
        None
    }

    fn set_solver_option(&mut self, apply: impl FnOnce(&mut Settings)) {
        if self.modal == Some(Modal::SolverSetup) {
            apply(&mut self.draft);
        } else {
            let previous = (
                self.settings.solver_selectivity,
                self.settings.solver_multi_pv,
            );
            apply(&mut self.settings);
            if (
                self.settings.solver_selectivity,
                self.settings.solver_multi_pv,
            ) != previous
            {
                self.save();
                self.session.solver_options_changed(&self.settings);
            }
        }
    }
}
