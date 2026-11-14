use std::time::Duration;

use nsh_base::button;
use nsh_core::prelude::*;
use reversi_core::player_color::PlayerColor;

use super::analysis::{animated_icon, shadow};
use super::assets::icon;
use super::format::{ScorePrecision, display_score};
use super::widgets::{control, flat_stone, focusable, single_line_text};
use super::{
    ACCENT, Action, BACKGROUND, BLUE, GOLD, PANEL, RED, ReversiApp, SECONDARY, Ui, color, rgba,
};
use crate::session::Activity;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum HistoryRow {
    Move(usize),
    Resume,
    Result,
}

pub(super) fn render_sidebar(ui: Ui) -> impl IntoElement {
    let desktop = computed(move || ui.read(|app| app.viewport.width.0 >= 1024.));
    let players = view()
        .flex_col()
        .flex_shrink_0()
        .gap(move || px(if desktop.get() { 12. } else { 10. }))
        .p(move || px(if desktop.get() { 16. } else { 12. }))
        .border_b_1()
        .border_color(rgba(0xffffff1a))
        .children(
            [PlayerColor::Black, PlayerColor::White]
                .into_iter()
                .map(move |player| player_card(ui, player)),
        );
    view()
        .flex_col()
        .w_full()
        .h_full()
        .min_h_0()
        .bg(color(PANEL))
        .text_size(px(12.))
        .line_height(nsh_core::LineHeight::Multiple(1.4285714))
        .child(players)
        .child(
            view()
                .flex_1()
                .min_h_0()
                .flex_col()
                .child(
                    view()
                        .flex_row()
                        .flex_shrink_0()
                        .items_center()
                        .justify_end()
                        .gap_1()
                        .px_2()
                        .pt_2()
                        .child(
                            view()
                                .relative()
                                .child(control(
                                    ui,
                                    "copy",
                                    "copy.svg",
                                    "",
                                    Action::Copy,
                                    move || ui.read(|app| app.session.game().cursor > 0),
                                ))
                                .child(when(
                                    move || ui.read(|app| app.copied_until.is_some()),
                                    move || {
                                        view()
                                            .absolute()
                                            .inset_0()
                                            .pointer_events(PointerEvents::None)
                                            .items_center()
                                            .justify_center()
                                            .bg(color(PANEL))
                                            .child(icon("check.svg", 14., ACCENT))
                                    },
                                )),
                        )
                        .child(control(
                            ui,
                            "history-first",
                            "chevron-first.svg",
                            "",
                            Action::Navigate(0),
                            move || ui.read(|app| navigation(app) && app.session.game().cursor > 0),
                        ))
                        .child(control(
                            ui,
                            "undo",
                            "chevron-left.svg",
                            "",
                            Action::Previous,
                            move || ui.read(|app| navigation(app) && app.session.game().cursor > 0),
                        ))
                        .child(control(
                            ui,
                            "redo",
                            "chevron-right.svg",
                            "",
                            Action::Next,
                            move || {
                                ui.read(|app| {
                                    navigation(app)
                                        && app.session.game().cursor
                                            < app.session.game().history.len()
                                })
                            },
                        ))
                        .child(control(
                            ui,
                            "history-last",
                            "chevron-last.svg",
                            "",
                            Action::Latest,
                            move || {
                                ui.read(|app| {
                                    navigation(app)
                                        && app.session.game().cursor
                                            < app.session.game().history.len()
                                })
                            },
                        )),
                )
                .child(
                    scroll()
                        .element_ref(&ui.read(|app| app.history_scroll))
                        .flex_1()
                        .min_h_0()
                        .w_full()
                        .child(
                            view()
                                .p_2()
                                .min_h(percentage(1_f32))
                                .child(when(
                                    move || {
                                        ui.read(|app| {
                                            app.session.game().history.is_empty()
                                                && app.ai_resume_label().is_none()
                                        })
                                    },
                                    move || {
                                        view()
                                            .flex_1()
                                            .min_h(percentage(1_f32))
                                            .items_center()
                                            .justify_center()
                                            .gap_2()
                                            .text_color(color(SECONDARY))
                                            .child(icon("list.svg", 20., SECONDARY))
                                            .child(text(move || tr!("no_moves_yet")))
                                    },
                                ))
                                .child(each(
                                    move || {
                                        ui.read(|app| {
                                            let game = app.session.game();
                                            let mut rows =
                                                Vec::with_capacity(game.history.len() + 2);
                                            rows.extend((0..game.cursor).map(HistoryRow::Move));
                                            if app.ai_resume_label().is_some() {
                                                rows.push(HistoryRow::Resume);
                                            }
                                            rows.extend(
                                                (game.cursor..game.history.len())
                                                    .map(HistoryRow::Move),
                                            );
                                            let last = game
                                                .history
                                                .last()
                                                .map_or(game.initial, |record| record.position);
                                            if last.game_over() {
                                                rows.push(HistoryRow::Result);
                                            }
                                            rows
                                        })
                                    },
                                    |row| *row,
                                    move |row, _| match row.get_untracked() {
                                        HistoryRow::Move(index) => {
                                            history_row(ui, index).into_element()
                                        }
                                        HistoryRow::Resume => view()
                                            .flex_row()
                                            .items_center()
                                            .gap_1()
                                            .mb(px(2.))
                                            .child(view().w_8())
                                            .child(
                                                view().flex_1().min_w_0().px_2().child(
                                                    control(
                                                        ui,
                                                        "ai-resume",
                                                        "play.svg",
                                                        move || {
                                                            ui.read(|app| {
                                                                app.ai_resume_label()
                                                                    .unwrap_or_default()
                                                            })
                                                        },
                                                        Action::Pause,
                                                        || true,
                                                    )
                                                    .px_3()
                                                    .gap(px(6.))
                                                    .border_1()
                                                    .border_color(rgba(0x43a57599))
                                                    .bg(rgba(0x43a57526))
                                                    .text_color(color(ACCENT)),
                                                ),
                                            )
                                            .into_element(),
                                        HistoryRow::Result => result_row(ui).into_element(),
                                    },
                                )),
                        ),
                ),
        )
}

fn navigation(app: &ReversiApp) -> bool {
    !matches!(
        app.session.activity(),
        Some(Activity::Move | Activity::Analysis)
    )
}

fn player_card(ui: Ui, player: PlayerColor) -> impl IntoElement {
    let ai = computed(move || ui.read(|app| app.settings.ai_color() == Some(player)));
    let active = computed(move || {
        ui.read(|app| {
            let position = app.session.game().position();
            player == position.turn && !position.game_over()
        })
    });
    let thinking = computed(move || {
        ai.get() && active.get() && ui.read(|app| app.session.activity() == Some(Activity::Move))
    });
    view()
        .min_w_0()
        .overflow_hidden()
        .flex_row()
        .items_center()
        .gap_3()
        .p_3()
        .rounded(px(14.))
        .border_2()
        .border_color(move || {
            if active.get() {
                color(ACCENT)
            } else {
                rgba(0xffffff1a)
            }
        })
        .bg(move || {
            color(PANEL).blend(if active.get() {
                rgba(0x43a57526)
            } else {
                rgba(0xffffff0d)
            })
        })
        .shadow([shadow(rgba(0x00000066), 1., 2., 0.)])
        .style_when(
            move || active.get(),
            |s| {
                s.shadow([
                    shadow(rgba(0x43a57533), 4., 8., -2.),
                    shadow(rgba(0x43a57533), 0., 0., 1.),
                ])
            },
        )
        .child(flat_stone(player == PlayerColor::Black, 40.))
        .child(
            text(move || {
                ui.read(|app| {
                    let (black, white) = app.session.game().position().scores();
                    if player == PlayerColor::Black {
                        black
                    } else {
                        white
                    }
                    .to_string()
                })
            })
            .flex_shrink_0()
            .text_size(px(24.))
            .line_height(nsh_core::LineHeight::Multiple(1_f32))
            .font_weight(FontWeight::BOLD),
        )
        .child(
            view()
                .flex_1()
                .min_w_0()
                .flex_row()
                .items_center()
                .gap(px(6.))
                .text_size(px(12.))
                .line_height(nsh_core::LineHeight::Multiple(1.4285714))
                .font_weight(FontWeight::MEDIUM)
                .text_color(move || {
                    if thinking.get() {
                        color(BLUE)
                    } else {
                        rgba(0xffffff99)
                    }
                })
                .child(
                    when(
                        move || thinking.get(),
                        move || {
                            animated_icon(
                                "loader-circle.svg",
                                16.,
                                BLUE,
                                Duration::from_secs(1),
                                true,
                            )
                        },
                    )
                    .otherwise(move || {
                        when(move || ai.get(), move || icon("bot.svg", 16., BLUE))
                            .otherwise(move || icon("user.svg", 16., SECONDARY))
                    }),
                )
                .child(
                    single_line_text(
                        move || {
                            let is_thinking = thinking.get();
                            let is_ai = ai.get();
                            ui.read(|app| {
                                if is_thinking {
                                    tr!("thinking")
                                } else if is_ai {
                                    if app.settings.timed {
                                        "AI".into()
                                    } else {
                                        format!("AI Lv.{}", app.settings.ai_level).into()
                                    }
                                } else if app.settings.ai_color().is_none() {
                                    app.color(player)
                                } else {
                                    tr!("you")
                                }
                            })
                        },
                        12. * 20. / 14.,
                    )
                    .truncate(),
                ),
        )
        .child(
            view()
                .flex_row()
                .min_w_0()
                .flex_shrink_0()
                .items_center()
                .gap(px(6.))
                .child(when(
                    move || thinking.get(),
                    move || {
                        control(
                            ui,
                            if player == PlayerColor::Black {
                                "ai-control-black"
                            } else {
                                "ai-control-white"
                            },
                            "stop-circle.svg",
                            "",
                            Action::Pause,
                            || true,
                        )
                        .h_7()
                        .gap(px(6.))
                        .border_1()
                        .border_color(rgba(0xc45c5c66))
                        .bg(color(BACKGROUND))
                        .text_color(color(RED))
                    },
                ))
                .child(when(
                    move || ai.get() && ui.read(|app| app.settings.timed),
                    move || {
                        view()
                            .flex_row()
                            .items_center()
                            .gap(px(6.))
                            .px_2()
                            .py(px(2.))
                            .rounded(px(4.))
                            .border_1()
                            .border_color(rgba(0xffffff1a))
                            .bg(rgba(0xffffff0d))
                            .text_size(px(12.))
                            .line_height(nsh_core::LineHeight::Multiple(1.5))
                            .font_family("Consolas")
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(color(SECONDARY))
                            .child(icon("clock.svg", 14., SECONDARY))
                            .child(text(move || {
                                ui.read(|app| {
                                    let seconds = app.session.clock_ms() / 1000;
                                    format!("{}:{:02}", seconds / 60, seconds % 60)
                                })
                            }))
                    },
                )),
        )
}

fn history_row(ui: Ui, index: usize) -> impl IntoElement {
    let id = format!("history-{index}");
    let enabled = move || ui.read(navigation);
    let activate = move || {
        ui.focus(&format!("history-{index}"));
        ui.action(Action::Navigate(index + 1));
    };
    let focus = focusable(ui, &id, enabled, activate);
    let focus_id = id;
    button()
        .element_ref(&focus)
        .disabled(move || !enabled())
        .accessibility_label(move || {
            ui.read(|app| {
                app.session
                    .game()
                    .history
                    .get(index)
                    .map_or_else(SharedString::default, |record| {
                        let square = record
                            .square
                            .map_or_else(|| tr!("pass").to_string(), |square| square.to_string());
                        format!("{}. {} {square}", index + 1, app.color(record.player)).into()
                    })
            })
        })
        .selected(move || ui.read(|app| app.session.game().cursor == index + 1))
        .flex_row()
        .items_center()
        .gap_1()
        .mb(px(2.))
        .rounded(px(8.))
        .bg(move || {
            ui.read(|app| {
                if app.session.game().cursor == index + 1 {
                    rgba(0x43a5751a)
                } else {
                    rgba(0)
                }
            })
        })
        .opacity(move || {
            ui.read(|app| {
                if index >= app.session.game().cursor {
                    0.4_f32
                } else {
                    1_f32
                }
            })
        })
        .style_when(
            move || ui.is_focus_visible(&focus_id),
            |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
        )
        .hover(|s| s.bg(rgba(0xffffff14)))
        .on_press(activate)
        .child(
            text(format!("{}.", index + 1))
                .w_8()
                .text_center()
                .font_family("Consolas")
                .text_size(px(12.))
                .line_height(nsh_core::LineHeight::Multiple(1.5))
                .text_color(color(SECONDARY)),
        )
        .child(
            view()
                .flex_1()
                .min_w_0()
                .flex_row()
                .items_center()
                .gap(px(6.))
                .px_2()
                .py_1()
                .child(
                    when(
                        move || {
                            ui.read(|app| {
                                app.session
                                    .game()
                                    .history
                                    .get(index)
                                    .is_some_and(|r| r.player == PlayerColor::Black)
                            })
                        },
                        || flat_stone(true, 16.),
                    )
                    .otherwise(|| flat_stone(false, 16.)),
                )
                .child(
                    single_line_text(
                        move || {
                            ui.read(|app| {
                                app.session.game().history.get(index).map_or_else(
                                    SharedString::default,
                                    |r| {
                                        r.square
                                            .map_or_else(|| tr!("pass"), |s| s.to_string().into())
                                    },
                                )
                            })
                        },
                        12. * 20. / 14.,
                    )
                    .font_weight(FontWeight::MEDIUM),
                )
                .child(when(
                    move || {
                        ui.read(|app| {
                            app.session
                                .game()
                                .history
                                .get(index)
                                .is_some_and(|r| r.score.is_some())
                        })
                    },
                    move || {
                        view()
                            .flex_row()
                            .items_center()
                            .gap(px(6.))
                            .ml_2()
                            .child(icon("bot.svg", 12., BLUE))
                            .child(
                                text(move || {
                                    ui.read(|app| {
                                        app.session
                                            .game()
                                            .history
                                            .get(index)
                                            .and_then(|r| r.score)
                                            .map_or_else(String::new, |score| {
                                                display_score(score, ScorePrecision::Tenth)
                                            })
                                    })
                                })
                                .text_size(px(12.))
                                .line_height(nsh_core::LineHeight::Multiple(1.5))
                                .font_family("Consolas")
                                .text_color(color(SECONDARY)),
                            )
                    },
                )),
        )
}

fn result_row(ui: Ui) -> impl IntoElement {
    view()
        .flex_row()
        .items_center()
        .gap_1()
        .mb(px(2.))
        .opacity(move || {
            ui.read(|app| {
                if app.session.game().cursor < app.session.game().history.len() {
                    0.4_f32
                } else {
                    1_f32
                }
            })
        })
        .child(view().w_8())
        .child(
            view()
                .flex_1()
                .min_w_0()
                .flex_row()
                .items_center()
                .gap(px(6.))
                .px_2()
                .py_1()
                .child(icon("trophy.svg", 16., GOLD))
                .child(
                    text(move || {
                        ui.read(|app| {
                            let game = app.session.game();
                            let position = game.history.last().map_or(game.initial, |r| r.position);
                            let (black, white) = position.scores();
                            let (message, score) = app.game_result(black, white);
                            format!("{message} {score}")
                        })
                    })
                    .truncate()
                    .font_weight(FontWeight::MEDIUM),
                ),
        )
}

impl ReversiApp {
    pub(super) fn ai_resume_label(&self) -> Option<SharedString> {
        let position = self.session.game().position();
        let waiting = !self.session.is_solving()
            && self.session.is_paused()
            && self.session.activity().is_none()
            && self.settings.ai_color() == Some(position.turn)
            && !position.game_over();
        if !waiting {
            return None;
        }
        let game = self.session.game();
        Some(if game.history.is_empty() {
            tr!("start")
        } else if game.cursor < game.history.len() {
            tr!("resume_position")
        } else {
            tr!("resume")
        })
    }
}
