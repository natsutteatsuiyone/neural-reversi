use std::rc::Rc;
use std::time::{Duration, Instant};

use nsh_core::keymap::When;
use nsh_core::prelude::*;
use reversi_core::player_color::PlayerColor;

use crate::session::Session;
use crate::settings::{GameMode, Language, Settings};

macro_rules! tr {
    ($($args:tt)*) => {{
        if let Some(locale) = nsh_core::element::use_context::<crate::ui::LocaleRevision>() {
            locale.0.get();
        }
        nsh_core::element::SharedString::from(rust_i18n::t!($($args)*).to_string())
    }};
}

mod about;
mod analysis;
mod assets;
mod board;
mod chart;
mod controller;
mod dialogs;
mod format;
mod header;
mod layout;
mod new_game;
mod notification;
mod setup;
mod sidebar;
mod slider;
mod solver;
mod theme;
mod widgets;

use about::LicenseViewer;
use analysis::AnalysisView;
use notification::Notification;
use setup::{SetupEditor, SetupTab};
use slider::SliderKind;
use theme::{ACCENT, AI_COLOR, BACKGROUND, BLUE, GOLD, PANEL, RED, SECONDARY, color, rgba};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelDrag {
    Right,
    Bottom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Modal {
    NewGame,
    Setup,
    SolverSetup,
    Settings,
    About,
}

#[derive(Clone, Debug)]
enum Action {
    NewGame,
    Setup,
    Back,
    Start,
    Close,
    Settings,
    About,
    MenuSection(Option<usize>),
    License(bool),
    Mode(GameMode),
    Timed(bool),
    Hint,
    AnalyzeAndChart,
    AnalysisPanel,
    Hash(usize),
    Language(Language),
    Navigate(usize),
    Previous,
    Next,
    Latest,
    Copy,
    Pause,
    Solver,
    ExitSolver,
    Selectivity(usize),
    MultiPv(bool),
    SetupTab(SetupTab),
    Turn(PlayerColor),
    ClearBoard,
    InitialBoard,
    Cell(usize),
    Chart(bool),
    Slider(SliderKind, f32),
    SetupTranscript(String),
    SetupBoard(String),
    AnalysisLevelMenu(bool),
}

struct ReversiApp {
    settings: Settings,
    settings_writable: bool,
    save_settings: fn(&Settings) -> Result<(), String>,
    unwritable_notified: bool,
    draft: Settings,
    session: Session,
    history_scroll: ElementRef,
    log_scroll: ElementRef,
    panel_drag: Option<PanelDrag>,
    analysis_view: AnalysisView,
    copied_until: Option<Instant>,
    modal: Option<Modal>,
    menu_section: Option<usize>,
    menu_hover_generation: u64,
    setup: SetupEditor,
    license: LicenseViewer,
    notification: Notification,
    board_cursor: usize,
    setup_cursor: usize,
    compact_layout: bool,
    compact_analysis_open: bool,
    viewport: Size<Points>,
}

#[derive(Clone)]
struct FocusEntry {
    id: String,
    scope: Option<Modal>,
    element: ElementRef,
    enabled: Rc<dyn Fn() -> bool>,
    activate: Rc<dyn Fn()>,
}

#[derive(Default)]
struct FocusRegistry {
    entries: Vec<FocusEntry>,
    return_to: Option<String>,
}

#[derive(Clone, Copy)]
struct LocaleRevision(Signal<u64>);

#[derive(Clone, Copy)]
struct Ui {
    app: Stored<ReversiApp>,
    changed: Signal<u64>,
    locale_revision: Signal<u64>,
    focused: Signal<Option<String>>,
    keyboard_focus: Signal<bool>,
    mouse_position: Signal<Point<Points>>,
    focus_registry: Stored<FocusRegistry>,
    scope: Option<Modal>,
}

impl Ui {
    fn new() -> Self {
        let (settings, load_error) = match Settings::load() {
            Ok(settings) => (settings, None),
            Err(error) => (Settings::default(), Some(error)),
        };
        Self::with_settings(settings, load_error, Settings::save)
    }

    fn with_settings(
        mut settings: Settings,
        load_error: Option<String>,
        save_settings: fn(&Settings) -> Result<(), String>,
    ) -> Self {
        let locale_revision = signal(0);
        provide_context(LocaleRevision(locale_revision));
        let settings_writable = load_error.is_none();
        // The user always plays Black at startup.
        if settings.game_mode == GameMode::AiBlack {
            settings.game_mode = GameMode::AiWhite;
        }
        rust_i18n::set_locale(settings.locale());
        let mut notification = Notification::default();
        if let Some(error) = load_error {
            notification.show_error(tr!("settings_load_failed").to_string(), error);
        }
        let session = Session::new(&settings);
        let viewport = window();
        Self {
            app: stored(ReversiApp {
                draft: settings.clone(),
                settings,
                settings_writable,
                save_settings,
                unwritable_notified: false,
                session,
                history_scroll: element_ref(),
                log_scroll: element_ref(),
                panel_drag: None,
                analysis_view: AnalysisView::default(),
                copied_until: None,
                modal: None,
                menu_section: None,
                menu_hover_generation: 0,
                setup: SetupEditor::new(),
                license: LicenseViewer::new(),
                notification,
                board_cursor: 19,
                setup_cursor: 0,
                compact_layout: viewport.width.0 < 1024.,
                compact_analysis_open: false,
                viewport,
            }),
            changed: signal(0),
            locale_revision,
            focused: signal(None),
            keyboard_focus: signal(false),
            mouse_position: signal(point(px(0.), px(0.))),
            focus_registry: stored(FocusRegistry::default()),
            scope: None,
        }
    }

    fn scoped(self, scope: Option<Modal>) -> Self {
        Self { scope, ..self }
    }

    fn read<R>(self, read: impl FnOnce(&ReversiApp) -> R) -> R {
        self.changed.get();
        self.app.with(|app| read(app))
    }

    fn update<R>(self, update: impl FnOnce(&mut ReversiApp) -> R) -> R {
        let result = self.app.with(update);
        self.changed
            .update(|revision| *revision = revision.wrapping_add(1));
        result
    }

    fn is_focused(self, id: &str) -> bool {
        self.focused.with(|focused| focused.as_deref() == Some(id))
    }

    fn is_focus_visible(self, id: &str) -> bool {
        self.keyboard_focus.get() && self.is_focused(id)
    }

    fn focus(self, id: &str) {
        let entries = self
            .focus_registry
            .with(|registry| registry.entries.clone());
        let scope = self.read(|app| app.modal);
        let Some(entry) = entries
            .iter()
            .find(|entry| entry.id == id && entry.scope == scope && (entry.enabled)())
        else {
            return;
        };
        if let Some(previous) = self.focused.get_untracked()
            && previous != id
            && let Some(previous) = entries.iter().find(|entry| entry.id == previous)
        {
            previous.element.blur();
        }
        self.focused.set(Some(id.to_owned()));
        entry.element.focus();
        entry.element.set_accessibility_focus();
    }

    fn focus_step(self, backwards: bool) {
        let scope = self.read(|app| app.modal);
        let entries = self
            .focus_registry
            .with(|registry| registry.entries.clone());
        let entries: Vec<_> = entries
            .into_iter()
            .filter(|entry| entry.scope == scope && (entry.enabled)())
            .collect();
        if entries.is_empty() {
            return;
        }
        let current = self.focused.get_untracked();
        let index = entries
            .iter()
            .position(|entry| Some(&entry.id) == current.as_ref());
        let next = match (index, backwards) {
            (Some(index), true) => (index + entries.len() - 1) % entries.len(),
            (Some(index), false) => (index + 1) % entries.len(),
            (None, true) => entries.len() - 1,
            (None, false) => 0,
        };
        self.focus(&entries[next].id);
    }

    fn activate_focused(self) {
        let current = self.focused.get_untracked();
        let entry = self.focus_registry.with(|registry| {
            registry
                .entries
                .iter()
                .find(|entry| Some(&entry.id) == current.as_ref())
                .cloned()
        });
        if let Some(entry) = entry
            && entry.scope == self.read(|app| app.modal)
            && (entry.enabled)()
        {
            (entry.activate)();
        }
    }

    fn bind_keys(self) {
        for (key, backwards) in [("tab", false), ("shift+tab", true)] {
            on_key_with(
                key,
                KeyOptions {
                    when: When::Always,
                    ..Default::default()
                },
                move |_| {
                    self.keyboard_focus.set(true);
                    if self.read(|app| app.analysis_view.level_menu) {
                        self.action(Action::AnalysisLevelMenu(false));
                    } else {
                        self.focus_step(backwards);
                    }
                },
            );
        }
        for key in ["enter", "space"] {
            on_key_with(
                key,
                KeyOptions {
                    when: When::NotEditing,
                    ..Default::default()
                },
                move |_| {
                    self.keyboard_focus.set(true);
                    if self.read(|app| app.analysis_view.level_menu) {
                        self.action(Action::AnalysisLevelMenu(false));
                    } else {
                        self.activate_focused();
                    }
                },
            );
        }
        on_key_with(
            "escape",
            KeyOptions {
                when: When::Always,
                ..Default::default()
            },
            move |_| {
                self.keyboard_focus.set(true);
                if self.read(|app| app.analysis_view.level_menu) {
                    self.action(Action::AnalysisLevelMenu(false));
                } else if self.read(|app| app.modal.is_some()) {
                    self.action(Action::Close);
                }
            },
        );
        for key in [
            "left",
            "right",
            "up",
            "down",
            "home",
            "end",
            "pageup",
            "pagedown",
            "shift+left",
            "shift+right",
            "shift+up",
            "shift+down",
        ] {
            on_key_with(
                key,
                KeyOptions {
                    when: When::NotEditing,
                    repeat: true,
                    ..Default::default()
                },
                move |key| {
                    self.keyboard_focus.set(true);
                    let (shift, key) = key
                        .strip_prefix("shift+")
                        .map_or((false, key), |key| (true, key));
                    if self.read(|app| app.analysis_view.level_menu && app.modal.is_none()) {
                        self.update(|app| SliderKind::AnalysisLevel.key(app, key, shift));
                        return;
                    }
                    if slider::focused_key(self, key, shift)
                        || analysis::focused_key(self, key)
                        || solver::focused_key(self, key)
                        || header::focused_key(self, key)
                        || dialogs::focused_key(self, key, shift)
                        || setup::focused_key(self, key)
                    {
                        return;
                    }
                    if self.read(|app| app.modal.is_none()) {
                        if self.is_focused("board-grid") {
                            board::board_key(self, key);
                        } else {
                            let cursor = self.read(|app| match key {
                                "left" if app.session.game().cursor > 0 => {
                                    Some(app.session.previous_cursor())
                                }
                                "right"
                                    if app.session.game().cursor
                                        < app.session.game().history.len() =>
                                {
                                    Some(app.session.game().cursor + 1)
                                }
                                _ => None,
                            });
                            if let Some(cursor) = cursor {
                                self.action(Action::Navigate(cursor));
                            }
                        }
                    }
                },
            );
        }
    }
}

#[component]
pub fn App() -> impl IntoElement {
    Ui::new().render()
}

impl Ui {
    fn render(self) -> impl IntoElement {
        let ui = self;
        ui.bind_keys();
        set_interval(Duration::from_millis(40), move || ui.tick());
        effect(move || {
            let viewport = window();
            ui.update(|app| {
                app.viewport = viewport;
                app.compact_layout = viewport.width.0 < 1024.;
                if !app.compact_layout {
                    app.compact_analysis_open = false;
                }
            });
        });
        layout::render(ui)
    }
}

pub fn run() -> impl std::process::Termination {
    #[cfg(target_os = "linux")]
    use nsh_platform_linux as platform;
    #[cfg(target_os = "macos")]
    use nsh_platform_macos as platform;
    #[cfg(target_os = "windows")]
    use nsh_platform_windows as platform;

    let root = platform::Root::of::<App>()
        .font(include_bytes!("../assets/fonts/NotoSans-Regular.ttf").as_slice())
        .tokens(
            nsh_base::Theme::from_json(theme::TOKENS)
                .expect("valid GUI theme")
                .into_tokens(),
        );
    platform::run_app(
        platform::AppSource::Rust(root),
        platform::AppOptions {
            title: "Neural Reversi".to_owned(),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            app_id: "neural-reversi".to_owned(),
            size: (1100, 920),
            ..Default::default()
        },
    )
}

impl ReversiApp {
    fn color(&self, player: PlayerColor) -> SharedString {
        match player {
            PlayerColor::Black => tr!("black"),
            PlayerColor::White => tr!("white"),
        }
    }

    fn game_result(&self, black: u32, white: u32) -> (String, String) {
        let message = if black == white {
            tr!("game_draw")
        } else {
            tr!(
                "game_win",
                winner = self.color(if black > white {
                    PlayerColor::Black
                } else {
                    PlayerColor::White
                })
            )
        };
        (
            message.to_string(),
            format!(
                "{}: {black} - {}: {white}",
                self.color(PlayerColor::Black),
                self.color(PlayerColor::White)
            ),
        )
    }

    fn save(&mut self) {
        if !self.settings_writable {
            if !self.unwritable_notified {
                self.unwritable_notified = true;
                self.notification
                    .show_error(tr!("settings_unwritable").to_string(), String::new());
            }
            return;
        }
        if let Err(error) = (self.save_settings)(&self.settings) {
            self.notification
                .show_error(tr!("settings_save_failed").to_string(), error);
        }
    }
}

#[cfg(test)]
mod tests;
