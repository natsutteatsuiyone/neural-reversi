use std::time::{Duration, Instant};

use gpui_kit::{ClipboardItem, Context, Window};
use reversi_core::{disc::Disc, square::Square};

use super::setup::SetupTab;
use super::{Action, Modal, ReversiApp};
use crate::game::Position;
use crate::session::Update;
use crate::settings::Settings;

impl ReversiApp {
    pub(super) fn tick(&mut self, cx: &mut Context<Self>) {
        let now = Instant::now();
        let mut changed = self.notification.tick(now);
        if self.copied_until.is_some_and(|until| now >= until) {
            self.copied_until = None;
            changed = true;
        }
        changed |= self.session.tick(&self.settings, now);
        self.present_updates();
        if changed {
            cx.notify();
        }
    }

    fn present_updates(&mut self) {
        for update in self.session.take_updates() {
            match update {
                Update::Played => {
                    self.notification.clear();
                    self.history_scroll.scroll_to_bottom();
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
                Update::Passed(player) => {
                    let message = tr!("player_passed", player = self.color(player)).into();
                    self.notification.show(
                        message,
                        String::new(),
                        "info.svg",
                        Instant::now() + Duration::from_millis(1500),
                    );
                }
                Update::Navigated => {
                    self.history_scroll
                        .scroll_to_item(self.session.game().cursor.saturating_sub(1));
                    self.notification.clear();
                }
                Update::Logged => self
                    .log_scroll
                    .scroll_to_item(self.session.log().len().saturating_sub(1)),
                Update::NoMove => self.notification.show_error(tr!("engine_no_move").into()),
                Update::Error(error) => self.notification.show_error(error),
            }
        }
    }

    pub(super) fn action(&mut self, action: Action, window: &mut Window, cx: &mut Context<Self>) {
        let previous_modal = self.modal;
        if previous_modal.is_some()
            && matches!(
                action,
                Action::NewGame
                    | Action::Solver
                    | Action::Settings
                    | Action::Hint
                    | Action::Copy
                    | Action::Navigate(_)
                    | Action::Pause
                    | Action::Analyze
                    | Action::AnalysisPanel
                    | Action::ExitSolver
                    | Action::Chart(_)
            )
        {
            return;
        }
        if previous_modal.is_none() {
            self.return_focus = window.focused(cx);
        }
        match action {
            Action::NewGame | Action::Solver => {
                let solver = matches!(action, Action::Solver);
                self.draft = self.settings.clone();
                self.setup.reset(window, cx);
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
            Action::Setup => {
                self.modal = Some(Modal::Setup);
            }
            Action::Back => self.modal = Some(Modal::NewGame),
            Action::Start => {
                let solver = self.modal == Some(Modal::SolverSetup);
                let position = if matches!(self.modal, Some(Modal::Setup | Modal::SolverSetup)) {
                    self.setup.position(cx)
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
                            self.save();
                            self.analysis_view.hover = None;
                            self.analysis_view.pointer = None;
                            self.notification.clear();
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
                self.menu_section = None;
                self.modal = Some(Modal::Settings);
            }
            Action::MenuSection(section) => self.menu_section = section,
            Action::About => {
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
            Action::HintLevel(level) => {
                self.settings.hint_level = level;
                self.save();
                self.session.hint_level_changed(&self.settings);
            }
            Action::Analyze => self.session.toggle_analysis(&self.settings),
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
            Action::Chart(chart) => self.analysis_view.chart = chart,
            Action::Hash(hash_size) => {
                self.settings.hash_size = hash_size;
                self.save();
                self.session.reinitialize(&self.settings);
            }
            Action::Language(language) => {
                self.settings.language = language;
                rust_i18n::set_locale(self.settings.locale());
                self.setup.update_placeholders(window, cx);
                self.save();
            }
            Action::Navigate(cursor) => self.session.navigate(cursor, &self.settings),
            Action::Copy => {
                cx.write_to_clipboard(ClipboardItem::new_string(self.session.game().transcript()));
                self.copied_until = Some(Instant::now() + Duration::from_millis(1500));
            }
            Action::Pause => self.session.toggle_pause(&self.settings),
            Action::ExitSolver => {
                self.session.exit_solver(&self.settings);
                self.analysis_view.close();
            }
            Action::Selectivity(index) => {
                self.set_solver_option(|settings| settings.solver_selectivity = index);
            }
            Action::MultiPv(multi_pv) => {
                self.set_solver_option(|settings| settings.solver_multi_pv = multi_pv);
            }
            Action::SetupTab(tab) => {
                self.setup.tab = tab;
                self.setup.refresh(cx);
            }
            Action::Turn(turn) => {
                self.setup.turn = turn;
                self.setup.error = None;
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
                } else if self.modal.is_none() {
                    let Some(square) =
                        Square::from_u8(index as u8).filter(|sq| *sq != Square::None)
                    else {
                        return;
                    };
                    self.session.play_move(square, &self.settings);
                }
            }
        }
        self.present_updates();
        if self.modal != previous_modal {
            if self.modal.is_some() {
                self.modal_entry_focus.focus(window, cx);
            } else if let Some(focus) = self.return_focus.take() {
                focus.focus(window, cx);
            } else {
                self.focus.focus(window, cx);
            }
        }
        cx.notify();
    }

    fn set_solver_option(&mut self, apply: impl FnOnce(&mut Settings)) {
        if self.modal == Some(Modal::SolverSetup) {
            apply(&mut self.draft);
        } else {
            apply(&mut self.settings);
            self.save();
            self.session.solver_options_changed(&self.settings);
        }
    }
}
