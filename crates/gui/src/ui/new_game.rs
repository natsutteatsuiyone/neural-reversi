use gpui_kit::component::StyledExt;
use gpui_kit::{prelude::*, *};
use reversi_core::disc::Disc;

use super::dialogs::{DIALOG_SMALL_LINE_HEIGHT, dialog_label, dialog_tab, dialog_tabs};
use super::widgets::{arrow_keys, flat_stone, interactive, shadow};
use super::{ACCENT, Action, ReversiApp, SECONDARY};
use crate::settings::GameMode;

#[derive(Clone)]
pub(super) struct DialogSliderDrag {
    pub(super) timed: bool,
}

impl Render for DialogSliderDrag {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        Empty
    }
}

impl ReversiApp {
    pub(super) fn render_new_game_dialog(&self, window: &Window, cx: &Context<Self>) -> Div {
        let human = self.draft.game_mode == GameMode::Human;
        div()
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_6()
            .py_4()
            .child(
                dialog_tabs()
                    .child(
                        dialog_tab(
                            "mode-ai",
                            tr!("vs_ai"),
                            Some("bot.svg"),
                            Action::Mode(GameMode::AiWhite),
                            !human,
                            cx,
                        )
                        .track_focus(&self.modal_entry_focus),
                    )
                    .child(dialog_tab(
                        "mode-human",
                        tr!("vs_human"),
                        Some("users.svg"),
                        Action::Mode(GameMode::Human),
                        human,
                        cx,
                    )),
            )
            .when(!human, |body| {
                body.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(dialog_label(tr!("you_play")))
                        .child(
                            div().flex().gap_3().children(
                                [
                                    (Disc::Black, GameMode::AiWhite),
                                    (Disc::White, GameMode::AiBlack),
                                ]
                                .into_iter()
                                .enumerate()
                                .map(|(index, (disc, mode))| {
                                    let selected = self.draft.game_mode == mode;
                                    interactive(
                                        div().id(("game-color", index)),
                                        self.color(disc),
                                        true,
                                    )
                                    .role(Role::RadioButton)
                                    .aria_toggled(selected.into())
                                    .on_key_down(arrow_keys(
                                        index,
                                        2,
                                        move |_, _| {
                                            Action::Mode(if mode == GameMode::AiWhite {
                                                GameMode::AiBlack
                                            } else {
                                                GameMode::AiWhite
                                            })
                                        },
                                        cx,
                                    ))
                                    .flex_1()
                                    .min_w_0()
                                    .flex()
                                    .flex_col()
                                    .items_center()
                                    .gap_2()
                                    .p_5()
                                    .rounded(px(14.))
                                    .border_2()
                                    .border_color(if selected {
                                        rgb(ACCENT)
                                    } else {
                                        rgba(0xffffff33)
                                    })
                                    .when(selected, |card| card.bg(rgba(0x43a5751a)))
                                    .cursor_pointer()
                                    .when(!selected, |card| {
                                        card.hover(|style| {
                                            style
                                                .border_color(rgba(0xffffff4d))
                                                .bg(rgba(0xffffff0d))
                                        })
                                    })
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.action(Action::Mode(mode), window, cx)
                                    }))
                                    .child(flat_stone(disc, 40.))
                                    .child(div().font_semibold().child(self.color(disc)))
                                }),
                            ),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(dialog_label(tr!("ai_mode")))
                        .child(
                            dialog_tabs()
                                .child(dialog_tab(
                                    "timed-mode",
                                    tr!("timed"),
                                    Some("timer.svg"),
                                    Action::Timed(true),
                                    self.draft.timed,
                                    cx,
                                ))
                                .child(dialog_tab(
                                    "level-mode",
                                    tr!("ai_level"),
                                    Some("zap.svg"),
                                    Action::Timed(false),
                                    !self.draft.timed,
                                    cx,
                                )),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .child(dialog_label(if self.draft.timed {
                                    tr!("time_per_game")
                                } else {
                                    tr!("ai_level")
                                }))
                                .child(
                                    div()
                                        .font_family("Consolas")
                                        .font_semibold()
                                        .text_color(rgb(ACCENT))
                                        .child(if self.draft.timed {
                                            let time = self.draft.game_time_limit;
                                            if time.is_multiple_of(60) {
                                                format!("{}m", time / 60)
                                            } else {
                                                format!("{}m {}s", time / 60, time % 60)
                                            }
                                        } else {
                                            self.draft.ai_level.to_string()
                                        }),
                                ),
                        )
                        .child(self.render_setting_slider(window, cx))
                        .child(
                            div()
                                .relative()
                                .w_full()
                                .when(self.draft.timed, |labels| labels.h(px(16.)))
                                .flex()
                                .justify_between()
                                .text_size(px(12.))
                                .line_height(px(DIALOG_SMALL_LINE_HEIGHT))
                                .text_color(rgb(SECONDARY))
                                .child(if self.draft.timed {
                                    "30s".into()
                                } else {
                                    tr!("easy")
                                })
                                .when(self.draft.timed, |labels| {
                                    labels.child(
                                        div()
                                            .absolute()
                                            .left(relative(0.474))
                                            .w_0()
                                            .flex()
                                            .justify_center()
                                            .child(
                                                div()
                                                    .flex_shrink_0()
                                                    .whitespace_nowrap()
                                                    .child("5m"),
                                            ),
                                    )
                                })
                                .child(if self.draft.timed {
                                    "10m".into()
                                } else {
                                    tr!("hard")
                                }),
                        ),
                )
            })
    }

    fn render_setting_slider(&self, window: &Window, cx: &Context<Self>) -> Div {
        let timed = self.draft.timed;
        let fraction = if timed {
            (self.draft.game_time_limit as f32 - 30.) / 570.
        } else {
            (self.draft.ai_level as f32 - 1.) / 29.
        };
        let view = cx.entity();
        let drag = DialogSliderDrag { timed };
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
                    .aria_label(if timed {
                        tr!("time_per_game")
                    } else {
                        tr!("ai_level")
                    })
                    .aria_numeric_value(if timed {
                        self.draft.game_time_limit as f64
                    } else {
                        self.draft.ai_level as f64
                    })
                    .aria_min_numeric_value(if timed { 30. } else { 1. })
                    .aria_max_numeric_value(if timed { 600. } else { 30. })
                    .aria_numeric_value_step(if timed { 30. } else { 1. })
                    .aria_orientation(Orientation::Horizontal)
                    .track_focus(&self.slider_focus.clone().tab_stop(true))
                    .on_a11y_action(AccessibleAction::Increment, {
                        let view = view.clone();
                        move |_, _, cx| {
                            view.update(cx, |this, cx| {
                                let value = if timed {
                                    (this.draft.game_time_limit + 30).min(600) as f32
                                } else {
                                    (this.draft.ai_level + 1).min(30) as f32
                                };
                                this.set_dialog_slider_value(timed, value, cx);
                            })
                        }
                    })
                    .on_a11y_action(AccessibleAction::Decrement, {
                        let view = view.clone();
                        move |_, _, cx| {
                            view.update(cx, |this, cx| {
                                let value = if timed {
                                    this.draft.game_time_limit.saturating_sub(30).max(30) as f32
                                } else {
                                    this.draft.ai_level.saturating_sub(1).max(1) as f32
                                };
                                this.set_dialog_slider_value(timed, value, cx);
                            })
                        }
                    })
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, _, cx| {
                        let (value, min, max, step) = if timed {
                            (this.draft.game_time_limit as f32, 30., 600., 30.)
                        } else {
                            (this.draft.ai_level as f32, 1., 30., 1.)
                        };
                        let rounded = min + ((value - min) / step).round() * step;
                        let increment = if event.keystroke.modifiers.shift {
                            10.
                        } else {
                            step
                        };
                        let value = match event.keystroke.key.as_str() {
                            "right" | "up" => rounded + increment,
                            "left" | "down" => rounded - increment,
                            "pageup" => rounded + 10.,
                            "pagedown" => rounded - 10.,
                            "home" => min,
                            "end" => max,
                            _ => return,
                        };
                        this.set_dialog_slider_value(timed, value.clamp(min, max), cx);
                        cx.stop_propagation();
                    }))
                    .absolute()
                    .top(px(-12.))
                    .bottom(px(-12.))
                    .w_full()
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            this.slider_focus.focus(window, cx);
                            this.set_dialog_slider(timed, event.position.x, cx);
                            cx.stop_propagation();
                        }),
                    )
                    .on_drag(drag, |drag, _, _, cx| {
                        cx.stop_propagation();
                        cx.new(|_| drag.clone())
                    })
                    .on_drag_move(cx.listener(
                        |this, event: &DragMoveEvent<DialogSliderDrag>, _, cx| {
                            let drag = event.drag(cx);
                            this.set_dialog_slider(drag.timed, event.event.position.x, cx);
                        },
                    ))
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
                            view.slider_layout = Some((timed, bounds));
                        });
                    },
                    |_, _, _, _| {},
                )
                .absolute()
                .size_full(),
            )
    }

    /// Sets the slider from a pointer x position within its last laid-out bounds.
    pub(super) fn set_dialog_slider(&mut self, timed: bool, x: Pixels, cx: &mut Context<Self>) {
        let Some((layout_timed, bounds)) = self.slider_layout else {
            return;
        };
        if layout_timed != timed || bounds.size.width <= px(12.) {
            return;
        }
        let fraction = ((x - bounds.left() - px(6.)) / (bounds.size.width - px(12.))).clamp(0., 1.);
        let value = if timed {
            (1. + fraction * 19.).round() * 30.
        } else {
            (1. + fraction * 29.).round()
        };
        self.set_dialog_slider_value(timed, value, cx);
    }

    fn set_dialog_slider_value(&mut self, timed: bool, value: f32, cx: &mut Context<Self>) {
        if timed {
            self.draft.game_time_limit = value as u64;
        } else {
            self.draft.ai_level = value as usize;
        }
        cx.notify();
    }
}

fn slider_ring() -> Vec<BoxShadow> {
    vec![shadow(rgba(0x43a57580), 0., 0., 3.)]
}
