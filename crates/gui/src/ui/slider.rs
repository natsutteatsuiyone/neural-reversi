use gpui_kit::component::StyledExt;
use gpui_kit::{prelude::*, *};

use super::dialogs::DIALOG_SMALL_LINE_HEIGHT;
use super::widgets::shadow;
use super::{ACCENT, ReversiApp, SECONDARY};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SliderKind {
    GameTime,
    AiLevel,
    HintLevel,
    AnalysisLevel,
}

impl SliderKind {
    /// Returns `(min, max, step)`.
    fn range(self) -> (f32, f32, f32) {
        match self {
            Self::GameTime => (30., 600., 30.),
            Self::AiLevel | Self::HintLevel | Self::AnalysisLevel => (1., 30., 1.),
        }
    }

    fn label(self) -> SharedString {
        match self {
            Self::GameTime => tr!("time_per_game"),
            Self::AiLevel => tr!("ai_level"),
            Self::HintLevel => tr!("hint_level"),
            Self::AnalysisLevel => tr!("analysis_level"),
        }
    }
}

#[derive(Clone)]
pub(super) struct SliderDrag(pub(super) SliderKind);

impl Render for SliderDrag {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        Empty
    }
}

impl ReversiApp {
    fn slider_value(&self, kind: SliderKind) -> f32 {
        match kind {
            SliderKind::GameTime => self.draft.game_time_limit as f32,
            SliderKind::AiLevel => self.draft.ai_level as f32,
            SliderKind::HintLevel => self.settings.hint_level as f32,
            SliderKind::AnalysisLevel => self.settings.analysis_level as f32,
        }
    }

    /// The current level above a 1–30 slider, for the hint and analysis levels.
    pub(super) fn render_level_slider(
        &self,
        kind: SliderKind,
        window: &Window,
        cx: &Context<Self>,
    ) -> Div {
        div()
            .w(px(208.))
            .flex()
            .flex_col()
            .gap_3()
            .px_3()
            .pt_2()
            .pb_1()
            .child(
                div()
                    .font_family("Consolas")
                    .font_semibold()
                    .text_color(rgb(ACCENT))
                    .child(format!("Lv.{}", self.slider_value(kind))),
            )
            .child(self.render_slider(kind, window, cx))
            .child(
                div()
                    .flex()
                    .justify_between()
                    .text_size(px(12.))
                    .line_height(px(DIALOG_SMALL_LINE_HEIGHT))
                    .text_color(rgb(SECONDARY))
                    .child("1")
                    .child("30"),
            )
    }

    pub(super) fn render_slider(
        &self,
        kind: SliderKind,
        window: &Window,
        cx: &Context<Self>,
    ) -> Div {
        let (min, max, step) = kind.range();
        let value = self.slider_value(kind);
        let fraction = (value - min) / (max - min);
        let view = cx.entity();
        div()
            .relative()
            .w_full()
            .h(px(4.))
            .flex_shrink_0()
            .child(
                div()
                    .absolute()
                    .top_0()
                    .w_full()
                    .h(px(4.))
                    .rounded_full()
                    .overflow_hidden()
                    .bg(rgba(0xffffff0a))
                    .child(
                        div().absolute().left(px(6.)).right(px(6.)).h_full().child(
                            div().h_full().w(relative(fraction)).bg(rgb(ACCENT)).child(
                                div()
                                    .absolute()
                                    .left(px(-6.))
                                    .w(px(6.))
                                    .h_full()
                                    .bg(rgb(ACCENT)),
                            ),
                        ),
                    ),
            )
            .child(
                div()
                    .id("setting-slider")
                    .role(Role::Slider)
                    .aria_label(kind.label())
                    .aria_numeric_value(value as f64)
                    .aria_min_numeric_value(min as f64)
                    .aria_max_numeric_value(max as f64)
                    .aria_numeric_value_step(step as f64)
                    .aria_orientation(Orientation::Horizontal)
                    .track_focus(&self.slider_focus.clone().tab_stop(true))
                    .on_a11y_action(AccessibleAction::Increment, {
                        let view = view.clone();
                        move |_, _, cx| {
                            view.update(cx, |this, cx| {
                                let value = (this.slider_value(kind) + step).min(max);
                                this.set_slider_value(kind, value, cx);
                            })
                        }
                    })
                    .on_a11y_action(AccessibleAction::Decrement, {
                        let view = view.clone();
                        move |_, _, cx| {
                            view.update(cx, |this, cx| {
                                let value = (this.slider_value(kind) - step).max(min);
                                this.set_slider_value(kind, value, cx);
                            })
                        }
                    })
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                        if this.slider_key(kind, &event.keystroke, cx) {
                            cx.stop_propagation();
                        }
                    }))
                    .absolute()
                    .top(px(-12.))
                    .bottom(px(-12.))
                    .w_full()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            // The analysis-level menu leaves focus on its trigger, whose
                            // window-level key handler drives the slider and closes the menu.
                            if kind != SliderKind::AnalysisLevel {
                                this.slider_focus.focus(window, cx);
                            }
                            this.set_slider(kind, event.position.x, cx);
                            cx.stop_propagation();
                        }),
                    )
                    .on_drag(SliderDrag(kind), |drag, _, _, cx| {
                        cx.stop_propagation();
                        cx.new(|_| drag.clone())
                    })
                    .on_drag_move(
                        cx.listener(|this, event: &DragMoveEvent<SliderDrag>, _, cx| {
                            let kind = event.drag(cx).0;
                            this.set_slider(kind, event.event.position.x, cx);
                        }),
                    )
                    .child(
                        div()
                            .id("setting-slider-hit")
                            .group("setting-slider-thumb")
                            .absolute()
                            .top_0()
                            .left(relative(fraction))
                            .ml(px(-8. - 12. * fraction))
                            .size(px(28.))
                            .cursor_pointer()
                            .child(
                                div()
                                    .id("setting-slider-thumb")
                                    .m_2()
                                    .size(px(12.))
                                    .rounded_full()
                                    .border_1()
                                    .border_color(rgb(ACCENT))
                                    .bg(rgb(0xffffff))
                                    .group_hover("setting-slider-thumb", |thumb| {
                                        thumb.shadow(slider_ring())
                                    })
                                    .group_active("setting-slider-thumb", |thumb| {
                                        thumb.shadow(slider_ring())
                                    })
                                    .when(
                                        cx.has_active_drag()
                                            || self.slider_focus.is_focused(window),
                                        |thumb| thumb.shadow(slider_ring()),
                                    ),
                            ),
                    ),
            )
            .child(
                canvas(
                    move |bounds, _, cx| {
                        view.update(cx, |view, _| {
                            view.slider_layout = Some((kind, bounds));
                        });
                    },
                    |_, _, _, _| {},
                )
                .absolute()
                .size_full(),
            )
    }

    /// Applies a slider key; returns whether the key was consumed.
    pub(super) fn slider_key(
        &mut self,
        kind: SliderKind,
        keystroke: &Keystroke,
        cx: &mut Context<Self>,
    ) -> bool {
        let (min, max, step) = kind.range();
        let rounded = min + ((self.slider_value(kind) - min) / step).round() * step;
        let large_step = 10. * step;
        let increment = if keystroke.modifiers.shift {
            large_step
        } else {
            step
        };
        let value = match keystroke.key.as_str() {
            "right" | "up" => rounded + increment,
            "left" | "down" => rounded - increment,
            "pageup" => rounded + large_step,
            "pagedown" => rounded - large_step,
            "home" => min,
            "end" => max,
            _ => return false,
        };
        self.set_slider_value(kind, value.clamp(min, max), cx);
        true
    }

    /// Sets the slider from a pointer x position within its last laid-out bounds.
    pub(super) fn set_slider(&mut self, kind: SliderKind, x: Pixels, cx: &mut Context<Self>) {
        let Some((layout_kind, bounds)) = self.slider_layout else {
            return;
        };
        if layout_kind != kind || bounds.size.width <= px(12.) {
            return;
        }
        let (min, max, step) = kind.range();
        let fraction = ((x - bounds.left() - px(6.)) / (bounds.size.width - px(12.))).clamp(0., 1.);
        let value = min + (fraction * (max - min) / step).round() * step;
        self.set_slider_value(kind, value, cx);
    }

    fn set_slider_value(&mut self, kind: SliderKind, value: f32, cx: &mut Context<Self>) {
        let value = value as usize;
        match kind {
            SliderKind::GameTime => self.draft.game_time_limit = value as u64,
            SliderKind::AiLevel => self.draft.ai_level = value,
            SliderKind::HintLevel if self.settings.hint_level != value => {
                self.settings.hint_level = value;
                self.save();
                self.session.hint_level_changed(&self.settings);
            }
            SliderKind::AnalysisLevel if self.settings.analysis_level != value => {
                self.settings.analysis_level = value;
                self.save();
            }
            SliderKind::HintLevel | SliderKind::AnalysisLevel => return,
        }
        cx.notify();
    }
}

fn slider_ring() -> Vec<BoxShadow> {
    vec![shadow(rgba(0x43a57580), 0., 0., 3.)]
}
