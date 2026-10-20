use gpui_kit::{prelude::*, *};
use reversi_core::level::{ENDGAME_STEPS, Level};

use super::format::{ScorePrecision, display_score, whole_score};
use super::widgets::{activity_icon, arrow_keys, flat_stone, interactive};
use super::{ACCENT, Action, BACKGROUND, BLUE, PANEL, RED, ReversiApp, SECONDARY};
use crate::session::Activity;
use crate::settings::Settings;

impl ReversiApp {
    pub(super) fn render_solver(&self, cx: &Context<Self>) -> Div {
        let searching = self.session.activity() == Some(Activity::Solver);
        let caption = tr!("candidates");
        div()
            .flex()
            .flex_col()
            .size_full()
            .min_h_0()
            .bg(rgb(PANEL))
            .text_size(px(12.))
            .line_height(relative(1.4285714))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .px_4()
                    .py(px(10.))
                    .border_b_1()
                    .border_color(rgba(0xffffff1a))
                    .child(activity_icon("solver-activity", searching))
                    .child(div().font_weight(FontWeight::MEDIUM).child(tr!("solver")))
                    .when(searching, |header| {
                        header.child(
                            div()
                                .rounded_full()
                                .px_2()
                                .py(px(2.))
                                .text_size(px(12.))
                                .line_height(relative(1.5))
                                .text_color(rgb(BLUE))
                                .bg(rgba(0x7ba0c833))
                                .child({
                                    let seconds =
                                        self.session.search_elapsed().unwrap_or_default().as_secs();
                                    format!(
                                        "{} {}:{:02}",
                                        tr!("solving"),
                                        seconds / 60,
                                        seconds % 60
                                    )
                                }),
                        )
                    })
                    .when(self.session.is_solver_stopped(), |header| {
                        header.child(div().text_size(px(12.)).child(tr!("solver_stopped")))
                    })
                    .child(div().flex_1())
                    .child(self.control(
                        "solver-exit",
                        "x.svg",
                        tr!("exit_solver"),
                        Action::ExitSolver,
                        true,
                        cx,
                    )),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .px_4()
                    .py_3()
                    .border_b_1()
                    .border_color(rgba(0xffffff1a))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                self.control(
                                    "solver-back",
                                    "arrow-left.svg",
                                    tr!("undo"),
                                    Action::Navigate(self.session.previous_cursor()),
                                    self.session.game().cursor > 0,
                                    cx,
                                )
                                .border_color(rgba(0xffffff0f))
                                .bg(rgb(BACKGROUND)),
                            )
                            .child(div().flex_1())
                            .when(searching || self.session.is_solver_stopped(), |row| {
                                row.child(
                                    self.control(
                                        "solver-pause",
                                        if searching { "square.svg" } else { "play.svg" },
                                        if searching {
                                            tr!("stop")
                                        } else {
                                            tr!("resume")
                                        },
                                        Action::Pause,
                                        true,
                                        cx,
                                    )
                                    .border_color(if searching {
                                        rgba(0x00000000)
                                    } else {
                                        rgba(0xffffff0f)
                                    })
                                    .bg(rgb(BACKGROUND))
                                    .when(searching, |button| {
                                        button.bg(rgba(0xc45c5c33)).text_color(rgb(RED))
                                    }),
                                )
                            }),
                    )
                    .child(solver_options(&self.settings, cx)),
            )
            .child(
                div()
                    .id("solver-candidates")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_3()
                            .p_4()
                            .child(
                                div()
                                    .flex()
                                    .gap(px(0.275))
                                    .pr(px(0.275))
                                    .text_size(px(12.))
                                    .line_height(relative(1.5))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(rgb(SECONDARY))
                                    .children(
                                        caption
                                            .chars()
                                            .map(|character| div().child(character.to_string())),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(flat_stone(self.session.game().position().turn, 16.))
                                    .child(
                                        div()
                                            .font_weight(FontWeight::MEDIUM)
                                            .child(self.color(self.session.game().position().turn)),
                                    ),
                            ),
                    )
                    .when(self.session.candidates().is_empty(), |list| {
                        list.child(
                            div()
                                .px_4()
                                .py_6()
                                .text_center()
                                .text_color(rgb(SECONDARY))
                                .child(self.session.search_error().map_or_else(
                                    || {
                                        if !self.session.game().position().board.has_legal_moves() {
                                            tr!("solver_no_legal_moves")
                                        } else if searching {
                                            tr!("solving")
                                        } else if self.session.is_solver_stopped() {
                                            tr!("solver_no_results")
                                        } else {
                                            tr!("solver_pending")
                                        }
                                    },
                                    |error| error.to_owned().into(),
                                )),
                        )
                    })
                    .child(div().flex().flex_col().gap_1().px_2().pb_3().children(
                        self.session.candidates().iter().filter_map(|candidate| {
                            let square = candidate.square?;
                            let best = self.session.candidates().first().is_some_and(|best| {
                                whole_score(best.score) == whole_score(candidate.score)
                            });
                            let pv_end = candidate
                                .pv
                                .match_indices(' ')
                                .nth(7)
                                .map_or(candidate.pv.len(), |(offset, _)| offset);
                            Some(
                                interactive(
                                    // Keyed by move so focus follows it when scores reorder the list.
                                    div().id(("candidate", square.index())),
                                    format!(
                                        "{} {}",
                                        square,
                                        display_score(candidate.score, ScorePrecision::Whole)
                                    ),
                                    true,
                                )
                                .flex()
                                .items_center()
                                .gap_3()
                                .px_3()
                                .py_2()
                                .rounded(px(8.))
                                .text_size(px(15.))
                                .cursor_pointer()
                                .when(best, |row| row.bg(rgba(0x43a5751a)))
                                .hover(|style| style.bg(rgba(0xffffff0d)))
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.action(Action::Cell(square.index()), window, cx)
                                }))
                                .child(
                                    div()
                                        .w_12()
                                        .flex_shrink_0()
                                        .font_family("Consolas")
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(if best {
                                            rgb(ACCENT)
                                        } else {
                                            rgba(0xffffffeb)
                                        })
                                        .child(square.to_string()),
                                )
                                .child(
                                    div()
                                        .w(px(56.))
                                        .flex_shrink_0()
                                        .font_family("Consolas")
                                        .text_right()
                                        .text_color(if best {
                                            rgb(ACCENT)
                                        } else {
                                            rgba(0xffffff99)
                                        })
                                        .child(display_score(
                                            candidate.score,
                                            ScorePrecision::Whole,
                                        )),
                                )
                                .child(
                                    div()
                                        .w_10()
                                        .flex_shrink_0()
                                        .text_right()
                                        .text_size(px(13.))
                                        .line_height(relative(1.5))
                                        .text_color(rgb(SECONDARY))
                                        .child(format!("{}%", candidate.probability)),
                                )
                                .child(
                                    div()
                                        .flex_1()
                                        .min_w_0()
                                        .truncate()
                                        .text_size(px(13.))
                                        .line_height(relative(1.5))
                                        .font_family("Consolas")
                                        .text_color(rgb(SECONDARY))
                                        .child(candidate.pv[..pv_end].to_owned()),
                                ),
                            )
                        }),
                    )),
            )
    }
}

/// Selectivity and mode radios showing `options`, the settings or the dialog draft.
pub(super) fn solver_options(options: &Settings, cx: &Context<ReversiApp>) -> Div {
    div()
        .flex()
        .flex_wrap()
        .gap_6()
        .child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .child(
                    div()
                        .text_size(px(12.))
                        .line_height(relative(1.5))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(rgb(SECONDARY))
                        .child(tr!("selectivity")),
                )
                .child(div().flex().flex_wrap().gap_3().children(
                    Level::ENDGAME_SELECTIVITY.into_iter().enumerate().map(
                        |(index, selectivity)| {
                            radio(
                                ("selectivity", index),
                                format!("{}%", selectivity.probability()),
                                options.solver_selectivity == index,
                                Action::Selectivity(index),
                                cx,
                            )
                        },
                    ),
                )),
        )
        .child(
            div()
                .flex()
                .flex_col()
                .gap_2()
                .child(
                    div()
                        .text_size(px(12.))
                        .line_height(relative(1.5))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(rgb(SECONDARY))
                        .child(tr!("mode")),
                )
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_3()
                        .child(radio(
                            "best-only",
                            tr!("best_move_only"),
                            !options.solver_multi_pv,
                            Action::MultiPv(false),
                            cx,
                        ))
                        .child(radio(
                            "multi-pv",
                            tr!("all_moves"),
                            options.solver_multi_pv,
                            Action::MultiPv(true),
                            cx,
                        )),
                ),
        )
}

fn radio(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    selected: bool,
    action: Action,
    cx: &Context<ReversiApp>,
) -> Stateful<Div> {
    let label = label.into();
    let (index, count) = match action {
        Action::Selectivity(index) => (index, ENDGAME_STEPS),
        Action::MultiPv(value) => (usize::from(value), 2),
        _ => (0, 1),
    };
    interactive(div().id(id), label.clone(), true)
        .role(Role::RadioButton)
        .aria_toggled(selected.into())
        .on_key_down(arrow_keys(
            index,
            count,
            move |_, neighbor| match action {
                Action::MultiPv(value) => Action::MultiPv(!value),
                _ => Action::Selectivity(neighbor),
            },
            cx,
        ))
        .flex()
        .items_center()
        .gap(px(6.))
        .text_size(px(12.))
        .line_height(relative(1.4285714))
        .text_color(rgba(0xffffffeb))
        .cursor_pointer()
        .on_click(cx.listener(move |this, _, window, cx| this.action(action, window, cx)))
        .child(
            div()
                .size_4()
                .flex_shrink_0()
                .rounded_full()
                .border_1()
                .border_color(if selected {
                    rgb(ACCENT)
                } else {
                    rgba(0xffffff0f)
                })
                .when(selected, |radio| radio.bg(rgb(ACCENT)))
                .flex()
                .items_center()
                .justify_center()
                .when(selected, |radio| {
                    radio.child(div().size_2().rounded_full().bg(rgb(0xffffff)))
                }),
        )
        .child(label)
}
