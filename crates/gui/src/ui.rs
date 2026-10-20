use std::time::{Duration, Instant};

use gpui_kit::component::{Root, Theme, ThemeMode};
use gpui_kit::{prelude::*, *};
use reversi_core::disc::Disc;

use crate::session::Session;
use crate::settings::{GameMode, Language, Settings};

/// Looks up a key in `locales/app.yml` for the current locale.
macro_rules! tr {
    ($($args:tt)*) => {
        gpui_kit::SharedString::from(rust_i18n::t!($($args)*))
    };
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
mod solver;
mod widgets;

use about::LicenseViewer;
use analysis::AnalysisView;
use assets::GuiAssets;
use notification::Notification;
use setup::{SetupEditor, SetupTab};

const BACKGROUND: u32 = 0x1c1c21;
const PANEL: u32 = 0x252529;
const TEXT: u32 = 0xededed;
const SECONDARY: u32 = 0xa8a8aa;
const ACCENT: u32 = 0x43a575;
const GOLD: u32 = 0xb8956c;
const BLUE: u32 = 0x7ba0c8;
const RED: u32 = 0xc45c5c;
const AI_COLOR: u32 = 0x5fb3c4;

const LEVEL_CHOICES: [usize; 9] = [1, 4, 8, 12, 16, 20, 24, 28, 30];

#[derive(Clone, Copy)]
enum PanelDrag {
    Right,
    Bottom,
}

#[derive(Clone, Copy, PartialEq)]
enum Modal {
    NewGame,
    Setup,
    SolverSetup,
    Settings,
    About,
}
#[derive(Clone, Copy)]
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
    HintLevel(usize),
    Analyze,
    AnalysisPanel,
    Hash(usize),
    Language(Language),
    Navigate(usize),
    Copy,
    Pause,
    Solver,
    ExitSolver,
    Selectivity(usize),
    MultiPv(bool),
    SetupTab(SetupTab),
    Turn(Disc),
    ClearBoard,
    InitialBoard,
    Cell(usize),
    Chart(bool),
}

struct ReversiApp {
    settings: Settings,
    settings_writable: bool,
    draft: Settings,
    session: Session,
    history_scroll: ScrollHandle,
    log_scroll: ScrollHandle,
    panel_drag: Option<PanelDrag>,
    analysis_view: AnalysisView,
    copied_until: Option<Instant>,
    modal: Option<Modal>,
    menu_section: Option<usize>,
    /// Invalidates pending hover-intent submenu switches.
    menu_hover_generation: u64,
    slider_layout: Option<(bool, Bounds<Pixels>)>,
    slider_focus: FocusHandle,
    setup: SetupEditor,
    license: LicenseViewer,
    notification: Notification,
    focus: FocusHandle,
    modal_focus: FocusHandle,
    modal_entry_focus: FocusHandle,
    return_focus: Option<FocusHandle>,
    board_focus: FocusHandle,
    board_cursor: usize,
    setup_focus: FocusHandle,
    setup_cursor: usize,
    compact_layout: bool,
    compact_analysis_open: bool,
    control_focus: std::cell::RefCell<std::collections::HashMap<ElementId, FocusHandle>>,
}

pub fn run() {
    gpui_kit::application()
        .with_assets(GuiAssets)
        .run(|cx: &mut App| {
            cx.text_system()
                .add_fonts(vec![std::borrow::Cow::Borrowed(include_bytes!(
                    "../assets/fonts/NotoSans-Regular.ttf"
                ))])
                .expect("load original board label font");
            gpui_kit::component::init(cx);
            Theme::change(ThemeMode::Dark, None, cx);
            let theme = Theme::global_mut(cx);
            theme.font_family = if cfg!(target_os = "windows") {
                "Segoe UI".into()
            } else {
                ".SystemUIFont".into()
            };
            theme.font_size = px(16.);
            theme.radius = px(8.);
            theme.radius_lg = px(14.);
            theme.background = rgb(BACKGROUND).into();
            theme.foreground = hsla(0., 0., 1., 0.92);
            theme.border = hsla(0., 0., 1., 0.06);
            theme.input = hsla(0., 0., 1., 0.06);
            theme.muted = hsla(0., 0., 1., 0.04);
            theme.muted_foreground = rgb(SECONDARY).into();
            theme.primary = rgb(ACCENT).into();
            theme.accent = Hsla::from(rgb(ACCENT)).opacity(0.12);
            theme.accent_foreground = rgb(0x52b885).into();
            theme.ring = rgb(ACCENT).into();
            theme.caret = rgb(TEXT).into();
            theme.selection = rgba(0x43a57555).into();
            theme.popover = rgb(0x2a2a2f).into();
            theme.popover_foreground = hsla(0., 0., 1., 0.92);
            cx.on_window_closed(|cx, _| {
                if cx.windows().is_empty() {
                    cx.quit();
                }
            })
            .detach();
            let bounds = Bounds::centered(None, size(px(1100.), px(920.)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(480.), px(620.))),
                    titlebar: Some(TitlebarOptions {
                        title: Some("Neural Reversi".into()),
                        ..Default::default()
                    }),
                    app_id: Some("neural-reversi".into()),
                    ..Default::default()
                },
                |window, cx| {
                    let view = cx.new(|cx| ReversiApp::new(window, cx));
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .expect("open Neural Reversi window");
            cx.activate(true);
        });
}

impl ReversiApp {
    fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let mut notification = Notification::default();
        let mut settings = Settings::load().unwrap_or_else(|error| {
            notification.show_error(error);
            Settings::default()
        });
        // The user always plays Black at startup.
        if settings.game_mode == GameMode::AiBlack {
            settings.game_mode = GameMode::AiWhite;
        }
        rust_i18n::set_locale(settings.locale());
        let settings_writable = !notification.is_visible();
        let session = Session::new(&settings);
        let focus = cx.focus_handle();
        focus.focus(window, cx);
        cx.spawn(async move |view, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(40))
                    .await;
                if view.update(cx, |view, cx| view.tick(cx)).is_err() {
                    break;
                }
            }
        })
        .detach();
        let setup = SetupEditor::new(window, cx);
        Self {
            draft: settings.clone(),
            settings,
            settings_writable,
            session,
            history_scroll: ScrollHandle::new(),
            log_scroll: ScrollHandle::new(),
            panel_drag: None,
            analysis_view: AnalysisView::default(),
            copied_until: None,
            modal: None,
            menu_section: None,
            menu_hover_generation: 0,
            slider_layout: None,
            slider_focus: cx.focus_handle(),
            setup,
            license: LicenseViewer::new(),
            notification,
            focus,
            modal_focus: cx.focus_handle(),
            modal_entry_focus: cx.focus_handle().tab_stop(true),
            return_focus: None,
            board_focus: cx.focus_handle(),
            board_cursor: 19,
            setup_focus: cx.focus_handle(),
            setup_cursor: 0,
            compact_layout: false,
            compact_analysis_open: false,
            control_focus: Default::default(),
        }
    }

    fn color(&self, color: Disc) -> SharedString {
        match color {
            Disc::Black => tr!("black"),
            Disc::White => tr!("white"),
            Disc::Empty => "—".into(),
        }
    }

    /// The winner line and the final score of a finished game.
    fn game_result(&self, black: u32, white: u32) -> (String, String) {
        let message = if black == white {
            tr!("game_draw").into()
        } else {
            let winner = self.color(if black > white {
                Disc::Black
            } else {
                Disc::White
            });
            tr!("game_win", winner = winner).into()
        };
        let score = format!(
            "{}: {black} - {}: {white}",
            self.color(Disc::Black),
            self.color(Disc::White),
        );
        (message, score)
    }

    fn save(&mut self) {
        if !self.settings_writable {
            self.notification
                .show_error(tr!("settings_unwritable").into());
            return;
        }
        if let Err(error) = self.settings.save() {
            self.notification.show_error(error);
        }
    }
}
