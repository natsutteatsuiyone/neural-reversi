use nsh_base::{radio, radio_group};
use nsh_core::prelude::*;
use reversi_core::player_color::PlayerColor;

use super::dialogs::{DIALOG_LINE_HEIGHT, dialog_label, dialog_tab, dialog_tabs};
use super::slider::{SliderKind, render_slider};
use super::widgets::{flat_stone, focusable, single_line_text};
use super::{ACCENT, Action, Ui, color, rgba};
use crate::settings::GameMode;

pub(super) fn render_new_game_dialog(ui: Ui) -> View {
    view()
        .w_full()
        .min_w_0()
        .gap_6()
        .py_4()
        .child(
            dialog_tabs()
                .child(dialog_tab(
                    ui,
                    "mode-ai",
                    move || tr!("vs_ai"),
                    Some("bot.svg"),
                    Action::Mode(GameMode::AiWhite),
                    move || ui.read(|a| a.draft.game_mode != GameMode::Human),
                ))
                .child(dialog_tab(
                    ui,
                    "mode-human",
                    move || tr!("vs_human"),
                    Some("users.svg"),
                    Action::Mode(GameMode::Human),
                    move || ui.read(|a| a.draft.game_mode == GameMode::Human),
                )),
        )
        .child(when(
            move || ui.read(|a| a.draft.game_mode != GameMode::Human),
            move || {
                view()
                    .gap_6()
                    .child(
                        view()
                            .gap_3()
                            .child(dialog_label(move || tr!("you_play")))
                            .child(
                                radio_group()
                                    .flex_row()
                                    .gap_3()
                                    .child(player_color(
                                        ui,
                                        PlayerColor::Black,
                                        GameMode::AiWhite,
                                        "game-color-black",
                                    ))
                                    .child(player_color(
                                        ui,
                                        PlayerColor::White,
                                        GameMode::AiBlack,
                                        "game-color-white",
                                    )),
                            ),
                    )
                    .child(
                        view()
                            .gap_3()
                            .child(dialog_label(move || tr!("ai_mode")))
                            .child(
                                dialog_tabs()
                                    .child(dialog_tab(
                                        ui,
                                        "timed-mode",
                                        move || tr!("timed"),
                                        Some("timer.svg"),
                                        Action::Timed(true),
                                        move || ui.read(|a| a.draft.timed),
                                    ))
                                    .child(dialog_tab(
                                        ui,
                                        "level-mode",
                                        move || tr!("ai_level"),
                                        Some("zap.svg"),
                                        Action::Timed(false),
                                        move || ui.read(|a| !a.draft.timed),
                                    )),
                            ),
                    )
                    .child(
                        view()
                            .gap_3()
                            .child(
                                view()
                                    .flex_row()
                                    .items_center()
                                    .justify_between()
                                    .child(dialog_label(move || {
                                        ui.read(|a| {
                                            if a.draft.timed {
                                                tr!("time_per_game")
                                            } else {
                                                tr!("ai_level")
                                            }
                                        })
                                    }))
                                    .child(
                                        single_line_text(
                                            move || {
                                                ui.read(|a| {
                                                    if a.draft.timed {
                                                        let time = a.draft.game_time_limit;
                                                        if time < 60 {
                                                            format!("{time}s")
                                                        } else if time.is_multiple_of(60) {
                                                            format!("{}m", time / 60)
                                                        } else {
                                                            format!("{}m {}s", time / 60, time % 60)
                                                        }
                                                    } else {
                                                        a.draft.ai_level.to_string()
                                                    }
                                                })
                                            },
                                            DIALOG_LINE_HEIGHT,
                                        )
                                        .font_family("Consolas")
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(color(ACCENT)),
                                    ),
                            )
                            .child(
                                when(
                                    move || ui.read(|a| a.draft.timed),
                                    move || render_slider(ui, SliderKind::GameTime),
                                )
                                .otherwise(move || render_slider(ui, SliderKind::AiLevel)),
                            )
                            .child(
                                view()
                                    .relative()
                                    .w_full()
                                    .flex_row()
                                    .justify_between()
                                    .text_size(px(12.))
                                    .line_height(px(18.))
                                    .style_when(
                                        move || ui.read(|app| app.draft.timed),
                                        |style| style.h(px(16.)),
                                    )
                                    .text_color(rgba(0xffffff99))
                                    .child(single_line_text(
                                        move || {
                                            ui.read(|a| {
                                                if a.draft.timed {
                                                    "30s".into()
                                                } else {
                                                    tr!("easy")
                                                }
                                            })
                                        },
                                        18.,
                                    ))
                                    .child(when(
                                        move || ui.read(|a| a.draft.timed),
                                        || {
                                            single_line_text("5m", 18.)
                                                .absolute()
                                                .left(relative(0.474))
                                                .translate_x(relative(-0.5))
                                        },
                                    ))
                                    .child(single_line_text(
                                        move || {
                                            ui.read(|a| {
                                                if a.draft.timed {
                                                    "10m".into()
                                                } else {
                                                    tr!("hard")
                                                }
                                            })
                                        },
                                        18.,
                                    )),
                            ),
                    )
            },
        ))
}

fn player_color(ui: Ui, disc: PlayerColor, mode: GameMode, id: &'static str) -> impl IntoElement {
    let reference = focusable(ui, id, || true, move || ui.action(Action::Mode(mode)));
    radio()
        .element_ref(&reference)
        .checked(move || ui.read(|a| a.draft.game_mode == mode))
        .accessibility_label(move || Some(ui.read(|a| a.color(disc))))
        .on_change(move |checked: &bool| {
            if *checked {
                ui.focus(id);
                ui.action(Action::Mode(mode));
            }
        })
        .flex_1()
        .min_w_0()
        .items_center()
        .gap_2()
        .p_5()
        .rounded(px(14.))
        .border(move || super::theme::stroke_px(2.))
        .border_color(move || {
            ui.read(|a| {
                if a.draft.game_mode == mode {
                    color(ACCENT)
                } else {
                    rgba(0xffffff33)
                }
            })
        })
        .bg(move || {
            ui.read(|a| {
                if a.draft.game_mode == mode {
                    rgba(0x43a5751a)
                } else {
                    rgba(0)
                }
            })
        })
        .hover(move |s| {
            s.border_color(move || {
                ui.read(|a| {
                    if a.draft.game_mode == mode {
                        color(ACCENT)
                    } else {
                        rgba(0xffffff4d)
                    }
                })
            })
            .bg(move || {
                ui.read(|a| {
                    if a.draft.game_mode == mode {
                        rgba(0x43a5751a)
                    } else {
                        rgba(0xffffff0d)
                    }
                })
            })
        })
        .style_when(
            move || ui.is_focus_visible(id),
            |s| {
                s.border_color(color(ACCENT))
                    .inset_ring(px(2.))
                    .inset_ring_color(color(ACCENT))
            },
        )
        .child(flat_stone(disc == PlayerColor::Black, 40.))
        .child(
            single_line_text(move || ui.read(|a| a.color(disc)), DIALOG_LINE_HEIGHT)
                .font_weight(FontWeight::SEMIBOLD),
        )
}
