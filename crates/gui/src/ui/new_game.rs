use gpui_kit::component::StyledExt;
use gpui_kit::{prelude::*, *};
use reversi_core::disc::Disc;

use super::dialogs::{DIALOG_SMALL_LINE_HEIGHT, dialog_label, dialog_tab, dialog_tabs};
use super::slider::SliderKind;
use super::widgets::{arrow_keys, flat_stone, interactive};
use super::{ACCENT, Action, ReversiApp, SECONDARY};
use crate::settings::GameMode;

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
                        .child(self.render_slider(
                            if self.draft.timed {
                                SliderKind::GameTime
                            } else {
                                SliderKind::AiLevel
                            },
                            window,
                            cx,
                        ))
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
}
