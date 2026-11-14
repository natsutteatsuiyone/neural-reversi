use nsh_base::{SliderValue, slider};
use nsh_core::prelude::*;
use reversi_core::level::MAX_LEVEL;

use super::widgets::focusable;
use super::{ACCENT, Action, ReversiApp, SECONDARY, Ui, color, rgba};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SliderKind {
    GameTime,
    AiLevel,
    HintLevel,
    AnalysisLevel,
}

impl SliderKind {
    pub(super) fn range(self) -> (f32, f32, f32) {
        match self {
            Self::GameTime => (30., 600., 30.),
            _ => (1., MAX_LEVEL as f32, 1.),
        }
    }

    pub(super) fn id(self) -> &'static str {
        match self {
            Self::GameTime => "slider-game-time",
            Self::AiLevel => "slider-ai-level",
            Self::HintLevel => "slider-hint-level",
            Self::AnalysisLevel => "slider-analysis-level",
        }
    }

    pub(super) fn label(self) -> SharedString {
        match self {
            Self::GameTime => tr!("time_per_game"),
            Self::AiLevel => tr!("ai_level"),
            Self::HintLevel => tr!("hint_level"),
            Self::AnalysisLevel => tr!("analysis_level"),
        }
    }

    fn enabled(self, app: &ReversiApp) -> bool {
        match self {
            Self::GameTime => {
                app.modal == Some(super::Modal::NewGame)
                    && app.draft.timed
                    && app.draft.game_mode != crate::settings::GameMode::Human
            }
            Self::AiLevel => {
                app.modal == Some(super::Modal::NewGame)
                    && !app.draft.timed
                    && app.draft.game_mode != crate::settings::GameMode::Human
            }
            Self::HintLevel => {
                app.modal == Some(super::Modal::Settings) && app.menu_section == Some(0)
            }
            Self::AnalysisLevel => {
                app.modal.is_none()
                    && app.analysis_view.level_menu
                    && app.analysis_open()
                    && app.session.activity() != Some(crate::session::Activity::Analysis)
            }
        }
    }

    pub(super) fn value(self, app: &ReversiApp) -> f32 {
        match self {
            Self::GameTime => app.draft.game_time_limit as f32,
            Self::AiLevel => app.draft.ai_level as f32,
            Self::HintLevel => app.settings.hint_level as f32,
            Self::AnalysisLevel => app.settings.analysis_level as f32,
        }
    }

    pub(super) fn set_value(self, app: &mut ReversiApp, value: f32) {
        let (min, max, step) = self.range();
        let value = (min + ((value - min) / step).round() * step).clamp(min, max) as usize;
        match self {
            Self::GameTime => app.draft.game_time_limit = value as u64,
            Self::AiLevel => app.draft.ai_level = value,
            Self::HintLevel if app.settings.hint_level != value => {
                app.settings.hint_level = value;
                app.save();
                app.session.hint_level_changed(&app.settings);
            }
            Self::AnalysisLevel if app.settings.analysis_level != value => {
                app.settings.analysis_level = value;
                app.save();
            }
            Self::HintLevel | Self::AnalysisLevel => {}
        }
    }

    pub(super) fn key(self, app: &mut ReversiApp, key: &str, shift: bool) -> bool {
        let (min, max, step) = self.range();
        let rounded = min + ((self.value(app) - min) / step).round() * step;
        let increment = if shift { 10. * step } else { step };
        let next = match key {
            "right" | "up" => rounded + increment,
            "left" | "down" => rounded - increment,
            "pageup" => rounded + 10. * step,
            "pagedown" => rounded - 10. * step,
            "home" => min,
            "end" => max,
            _ => return false,
        };
        self.set_value(app, next.clamp(min, max));
        true
    }
}

pub(super) fn focused_key(ui: Ui, key: &str, shift: bool) -> bool {
    let kind = [
        SliderKind::GameTime,
        SliderKind::AiLevel,
        SliderKind::HintLevel,
        SliderKind::AnalysisLevel,
    ]
    .into_iter()
    .find(|kind| ui.is_focused(kind.id()) && ui.read(|app| kind.enabled(app)));
    kind.is_some_and(|kind| ui.update(|app| kind.key(app, key, shift)))
}

pub(super) fn render_level_slider(ui: Ui, kind: SliderKind) -> View {
    view()
        .w(px(208.))
        .on_touch(move |_: &nsh_core::TouchEvent| ui.keyboard_focus.set(false))
        .gap_3()
        .px_3()
        .pt_2()
        .pb_1()
        .child(
            text(move || ui.read(|app| format!("Lv.{}", kind.value(app))))
                .font_family("Consolas")
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(color(ACCENT)),
        )
        .child(render_slider(ui, kind))
        .child(
            view()
                .flex_row()
                .justify_between()
                .text_size(px(12.))
                .line_height(px(18.))
                .text_color(color(SECONDARY))
                .child(text("1"))
                .child(text(MAX_LEVEL.to_string())),
        )
}

pub(super) fn render_slider(ui: Ui, kind: SliderKind) -> impl IntoElement {
    let (min, max, step) = kind.range();
    let reference = focusable(
        ui,
        kind.id(),
        move || ui.read(|app| kind.enabled(app)),
        || {},
    );
    view()
        .w_full()
        .h(px(4.))
        .flex_shrink_0()
        .on_touch(move |event: &nsh_core::TouchEvent| {
            if event.phase == nsh_core::TouchPhase::Start {
                ui.keyboard_focus.set(false);
                ui.focus(kind.id());
            }
        })
        .child(
            slider()
                .element_ref(&reference)
                .accessibility_label(move || Some(kind.label()))
                .min(min as f64)
                .max(max as f64)
                .step(step as f64)
                .value(move || Some(SliderValue::Single(ui.read(|app| kind.value(app)) as f64)))
                .on_change(move |value: &SliderValue| {
                    ui.focus(kind.id());
                    ui.action(Action::Slider(kind, value.start() as f32));
                })
                .w_full()
                .h(px(4.))
                .relative()
                .track_style(|s| {
                    s.absolute()
                        .left(px(6.))
                        .right(px(6.))
                        .top(px(-12.))
                        .h(px(28.))
                        .justify_center()
                })
                .indicator_style(|s| s.w_full().h(px(4.)).rounded_full().bg(rgba(0xffffff0a)))
                .range_style(|s| s.h_full().rounded_full().bg(color(ACCENT)))
                .thumb_style(move |s| {
                    s.size(px(12.))
                        .rounded_full()
                        .border_1()
                        .border_color(color(ACCENT))
                        .bg(color(0xffffff))
                        .group_hover(|s| s.ring(px(3.)).ring_color(rgba(0x43a57580)))
                        .group_active(|s| s.ring(px(3.)).ring_color(rgba(0x43a57580)))
                        .style_when(
                            move || ui.is_focus_visible(kind.id()),
                            |s| s.ring(px(3.)).ring_color(rgba(0x43a57580)),
                        )
                }),
        )
}
