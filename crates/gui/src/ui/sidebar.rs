use std::time::Duration;

use gpui_kit::{prelude::*, *};
use reversi_core::disc::Disc;

use super::assets::icon;
use super::format::{ScorePrecision, display_score};
use super::widgets::{flat_stone, interactive, shadow};
use super::{ACCENT, Action, BACKGROUND, BLUE, GOLD, PANEL, RED, ReversiApp, SECONDARY};
use crate::game::MoveRecord;
use crate::session::Activity;

impl ReversiApp {
    pub(super) fn render_sidebar(&self, desktop: bool, cx: &Context<Self>) -> Div {
        let position = self.session.game().position();
        let (black, white) = position.scores();
        let mut players = div()
            .flex()
            .flex_col()
            .flex_shrink_0()
            .gap(px(if desktop { 12. } else { 10. }))
            .p(px(if desktop { 16. } else { 12. }))
            .border_b_1()
            .border_color(rgba(0xffffff1a));
        for (index, color, score) in [(0_usize, Disc::Black, black), (1, Disc::White, white)] {
            let ai = self.settings.ai_color() == Some(color);
            let active = color == position.turn && !position.game_over();
            let thinking = ai && active && self.session.activity() == Some(Activity::Move);
            let seconds = self.session.clock_ms() / 1000;
            players = players.child(
                div()
                    .min_w_0()
                    .overflow_hidden()
                    .flex()
                    .items_center()
                    .gap_3()
                    .p_3()
                    .rounded(px(14.))
                    .border_2()
                    .border_color(if active {
                        rgb(ACCENT)
                    } else {
                        rgba(0xffffff1a)
                    })
                    .shadow(if active {
                        vec![
                            shadow(rgba(0x43a57533), 4., 8., -2.),
                            shadow(rgba(0x43a57533), 0., 0., 1.),
                        ]
                    } else {
                        vec![shadow(rgba(0x00000066), 1., 2., 0.)]
                    })
                    // CSS clips outer shadows away from the card interior.
                    .bg(rgb(PANEL).blend(if active {
                        rgba(0x43a57526)
                    } else {
                        rgba(0xffffff0d)
                    }))
                    .child(flat_stone(color, 40.))
                    .child(
                        div()
                            .flex_shrink_0()
                            .text_size(px(24.))
                            .line_height(relative(1.))
                            .font_weight(FontWeight::BOLD)
                            .child(score.to_string()),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .flex()
                            .items_center()
                            .gap(px(6.))
                            .text_size(px(12.))
                            .line_height(relative(1.4285714))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(if thinking {
                                rgb(BLUE)
                            } else {
                                rgba(0xffffff99)
                            })
                            .child(if thinking {
                                icon("loader-circle.svg", 16., BLUE)
                                    .with_animation(
                                        ("player-thinking", index),
                                        Animation::new(Duration::from_secs(1)).repeat(),
                                        |icon, delta| {
                                            icon.with_transformation(
                                                gpui_kit::Transformation::rotate(
                                                    gpui_kit::radians(
                                                        delta * std::f32::consts::TAU,
                                                    ),
                                                ),
                                            )
                                        },
                                    )
                                    .into_any_element()
                            } else {
                                icon(
                                    if ai { "bot.svg" } else { "user.svg" },
                                    16.,
                                    if ai { BLUE } else { SECONDARY },
                                )
                                .into_any_element()
                            })
                            .child(div().truncate().child(if thinking {
                                tr!("thinking").into()
                            } else if ai {
                                if self.settings.timed {
                                    "AI".to_owned()
                                } else {
                                    format!("AI Lv.{}", self.settings.ai_level)
                                }
                            } else if self.settings.ai_color().is_none() {
                                self.color(color).into()
                            } else {
                                tr!("you").into()
                            })),
                    )
                    .child(
                        div()
                            .flex()
                            .min_w_0()
                            .flex_shrink_0()
                            .items_center()
                            .gap(px(6.))
                            .when(thinking, |card| {
                                card.child(
                                    self.control(
                                        ("ai-control", index),
                                        "stop-circle.svg",
                                        "",
                                        Action::Pause,
                                        true,
                                        cx,
                                    )
                                    .h_7()
                                    .gap(px(6.))
                                    .border_color(rgba(0xc45c5c66))
                                    .bg(rgb(BACKGROUND))
                                    .text_color(rgb(RED)),
                                )
                            })
                            .when(ai && self.settings.timed, |card| {
                                card.child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap(px(6.))
                                        .px_2()
                                        .py(px(2.))
                                        .rounded(px(4.))
                                        .border_1()
                                        .border_color(rgba(0xffffff1a))
                                        .bg(rgba(0xffffff0d))
                                        .text_size(px(12.))
                                        .line_height(relative(1.5))
                                        .font_family("Consolas")
                                        .font_weight(FontWeight::MEDIUM)
                                        .text_color(rgb(SECONDARY))
                                        .child(icon("clock.svg", 14., SECONDARY))
                                        .child(format!("{}:{:02}", seconds / 60, seconds % 60)),
                                )
                            }),
                    ),
            );
        }
        let game = self.session.game();
        let resume_label = self.ai_resume_label();
        let navigation = !matches!(
            self.session.activity(),
            Some(Activity::Move | Activity::Analysis)
        );
        let can_back = navigation && game.cursor > 0;
        let can_forward = navigation && game.cursor < game.history.len();
        let number = |number: usize| {
            div()
                .w_8()
                .text_center()
                .font_family("Consolas")
                .text_size(px(12.))
                .line_height(relative(1.5))
                .text_color(rgb(SECONDARY))
                .child(format!("{number}."))
        };
        let history_row = |(index, record): (usize, &MoveRecord)| {
            let square = record
                .square
                .map_or_else(|| tr!("pass").to_string(), |square| square.to_string());
            interactive(
                div().id(("history", index)),
                format!("{}. {} {square}", index + 1, self.color(record.player)),
                navigation,
            )
            .aria_selected(game.cursor == index + 1)
            .flex()
            .items_center()
            .gap_1()
            .mb(px(2.))
            .rounded(px(8.))
            .when(game.cursor == index + 1, |row| row.bg(rgba(0x43a5751a)))
            .when(index >= game.cursor, |row| row.opacity(0.4))
            .when(navigation, |row| {
                row.cursor_pointer()
                    .hover(|style| style.bg(rgba(0xffffff14)))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.action(Action::Navigate(index + 1), window, cx)
                    }))
            })
            .child(number(index + 1))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .items_center()
                    .gap(px(6.))
                    .px_2()
                    .py_1()
                    .child(flat_stone(record.player, 16.))
                    .child(div().font_weight(FontWeight::MEDIUM).child(square))
                    .when_some(record.score, |row, score| {
                        row.child(icon("bot.svg", 12., BLUE).ml_2()).child(
                            div()
                                .text_size(px(12.))
                                .line_height(relative(1.5))
                                .font_family("Consolas")
                                .text_color(rgb(SECONDARY))
                                .child(display_score(score, ScorePrecision::Tenth)),
                        )
                    }),
            )
        };
        // A paused AI's resume button sits right after the position it resumes
        // from, unnumbered so it does not read as a move.
        let resume_row = resume_label.map(|label| {
            div()
                .flex()
                .items_center()
                .gap_1()
                .mb(px(2.))
                .child(div().w_8())
                .child(
                    div().flex_1().min_w_0().flex().items_center().px_2().child(
                        self.control("ai-resume", "play.svg", label, Action::Pause, true, cx)
                            .px(px(12.))
                            .gap(px(6.))
                            .border_color(rgba(0x43a57599))
                            .bg(rgba(0x43a57526))
                            .text_color(rgb(ACCENT)),
                    ),
                )
        });
        let last = game
            .history
            .last()
            .map_or(game.initial, |record| record.position);
        let result_row = last.game_over().then(|| {
            let (black, white) = last.scores();
            let (message, score) = self.game_result(black, white);
            div()
                .flex()
                .items_center()
                .gap_1()
                .mb(px(2.))
                .when(game.cursor < game.history.len(), |row| row.opacity(0.4))
                .child(div().w_8())
                .child(
                    div()
                        .flex_1()
                        .min_w_0()
                        .flex()
                        .items_center()
                        .gap(px(6.))
                        .px_2()
                        .py_1()
                        .child(icon("trophy.svg", 16., GOLD))
                        .child(
                            div()
                                .truncate()
                                .font_weight(FontWeight::MEDIUM)
                                .child(format!("{message} {score}")),
                        ),
                )
        });
        div()
            .flex()
            .flex_col()
            .size_full()
            .min_h_0()
            .bg(rgb(PANEL))
            .text_size(px(12.))
            .line_height(relative(1.4285714))
            .child(players)
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .flex_shrink_0()
                            .flex()
                            .items_center()
                            .justify_end()
                            .gap_1()
                            .px_2()
                            .pt_2()
                            .child(self.control(
                                "copy",
                                if self.copied_until.is_some() {
                                    "check.svg"
                                } else {
                                    "copy.svg"
                                },
                                "",
                                Action::Copy,
                                game.cursor > 0,
                                cx,
                            ))
                            .child(self.control(
                                "history-first",
                                "chevron-first.svg",
                                "",
                                Action::Navigate(0),
                                can_back,
                                cx,
                            ))
                            .child(self.control(
                                "undo",
                                "chevron-left.svg",
                                "",
                                Action::Navigate(game.cursor.saturating_sub(1)),
                                can_back,
                                cx,
                            ))
                            .child(self.control(
                                "redo",
                                "chevron-right.svg",
                                "",
                                Action::Navigate(game.cursor + 1),
                                can_forward,
                                cx,
                            ))
                            .child(self.control(
                                "history-last",
                                "chevron-last.svg",
                                "",
                                Action::Navigate(game.history.len()),
                                can_forward,
                                cx,
                            )),
                    )
                    .child(
                        div()
                            .id("history-list")
                            .track_scroll(&self.history_scroll)
                            .flex_1()
                            .min_h_0()
                            .w_full()
                            .overflow_y_scroll()
                            .p_2()
                            .when(game.history.is_empty() && resume_row.is_none(), |list| {
                                list.child(
                                    div()
                                        .size_full()
                                        .flex()
                                        .flex_col()
                                        .items_center()
                                        .justify_center()
                                        .gap_2()
                                        .text_color(rgb(SECONDARY))
                                        .child(icon("list.svg", 20., SECONDARY).opacity(0.5))
                                        .child(tr!("no_moves_yet")),
                                )
                            })
                            .children(
                                game.history
                                    .iter()
                                    .enumerate()
                                    .take(game.cursor)
                                    .map(&history_row),
                            )
                            .children(resume_row)
                            .children(
                                game.history
                                    .iter()
                                    .enumerate()
                                    .skip(game.cursor)
                                    .map(&history_row),
                            )
                            .children(result_row),
                    ),
            )
    }
    /// Label of the button that lets a paused AI move, when the AI is waiting on its turn.
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
