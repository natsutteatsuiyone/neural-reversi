use gpui_kit::component::Icon;
use gpui_kit::{prelude::*, *};
use reversi_core::score::Score;

use super::assets::icon;
use super::format::{ScorePrecision, compact_number, display_depth, display_score, tenth_score};
use super::slider::SliderKind;
use super::widgets::{activity_icon, interactive, shadow};
use super::{ACCENT, Action, BACKGROUND, BLUE, PANEL, RED, ReversiApp, SECONDARY};
use crate::session::Activity;

/// Transient state of the analysis panel: the selected tab, the level menu,
/// and the evaluation chart's pointer.
#[derive(Default)]
pub(super) struct AnalysisView {
    pub(super) chart: bool,
    pub(super) level_menu: bool,
    pub(super) hover: Option<usize>,
    pub(super) bounds: Option<Bounds<Pixels>>,
    pub(super) pointer: Option<Point<Pixels>>,
}

impl AnalysisView {
    /// Returns to the thinking log with no menu or chart hover.
    pub(super) fn close(&mut self) {
        self.chart = false;
        self.level_menu = false;
        self.hover = None;
        self.pointer = None;
    }
}

impl ReversiApp {
    /// Handles a key while the level menu is open; returns whether it was consumed.
    pub(super) fn level_menu_key(&mut self, keystroke: &Keystroke, cx: &mut Context<Self>) -> bool {
        let view = &mut self.analysis_view;
        match keystroke.key.as_str() {
            "escape" | "tab" | "enter" | "space" => view.level_menu = false,
            _ => return self.slider_key(SliderKind::AnalysisLevel, keystroke, cx),
        }
        true
    }

    /// Full-window layer that closes the open level menu on click.
    pub(super) fn render_level_menu_dismiss(&self, cx: &Context<Self>) -> Deferred {
        deferred(
            div()
                .id("analysis-level-dismiss")
                .absolute()
                .inset_0()
                .occlude()
                .on_mouse_down(
                    MouseButton::Left,
                    cx.listener(|this, _, _, cx| {
                        this.analysis_view.level_menu = false;
                        cx.notify();
                        cx.stop_propagation();
                    }),
                ),
        )
        .with_priority(1)
    }

    pub(super) fn render_analysis_header(&self, cx: &Context<Self>) -> Stateful<Div> {
        let thinking = self.session.activity() == Some(Activity::Move);
        let analyzing = self.session.activity() == Some(Activity::Analysis);
        interactive(div().id("analysis-panel"), tr!("ai_analysis"), true)
            .aria_expanded(self.analysis_open())
            .h_10()
            .flex_shrink_0()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .overflow_hidden()
            .border_t_1()
            .border_color(rgba(0xffffff1a))
            .bg(rgb(PANEL))
            .cursor_pointer()
            .hover(|style| style.bg(rgba(0xffffff0d)))
            .on_click(
                cx.listener(|this, _, window, cx| this.action(Action::AnalysisPanel, window, cx)),
            )
            .child(activity_icon("analysis-activity", thinking))
            .child(
                div()
                    .flex_shrink_0()
                    .text_size(px(12.))
                    .line_height(relative(1.4285714))
                    .font_weight(FontWeight::MEDIUM)
                    .child(tr!("ai_analysis")),
            )
            .when(analyzing, |bar| {
                bar.child(
                    div()
                        .flex_shrink_0()
                        .rounded_full()
                        .px_2()
                        .py(px(2.))
                        .text_size(px(12.))
                        .line_height(relative(1.5))
                        .text_color(rgb(0xc8924d))
                        .bg(rgba(0xc8924d33))
                        .child(format!(
                            "{}... ({}/{})",
                            tr!("analyzing"),
                            self.session.analyses().len(),
                            self.session
                                .game()
                                .history
                                .iter()
                                .filter(|record| record.square.is_some())
                                .count()
                        )),
                )
            })
            .when(!thinking && !analyzing, |bar| {
                bar.when_some(self.session.log().last(), |bar, progress| {
                    bar.child(
                        div()
                            .flex()
                            .min_w_0()
                            .items_center()
                            .gap_2()
                            .text_size(px(12.))
                            .line_height(relative(1.5))
                            .font_family("Consolas")
                            .whitespace_nowrap()
                            .text_color(rgb(SECONDARY))
                            .child(tr!("best"))
                            .child(
                                div()
                                    .flex_shrink_0()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(rgb(ACCENT))
                                    .child(
                                        progress.square.map_or_else(
                                            || "—".into(),
                                            |square| square.to_string(),
                                        ),
                                    ),
                            )
                            .child(
                                div()
                                    .h_3()
                                    .mx_1()
                                    .border_l_1()
                                    .border_color(rgba(0xffffff1a)),
                            )
                            .child(
                                div()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(score_color(progress.score))
                                    .child(display_score(progress.score, ScorePrecision::Raw)),
                            )
                            .child(
                                div()
                                    .h_3()
                                    .mx_1()
                                    .border_l_1()
                                    .border_color(rgba(0xffffff1a)),
                            )
                            .child(
                                div()
                                    .truncate()
                                    .child(display_depth(progress.depth, progress.probability)),
                            ),
                    )
                })
            })
            .child(div().flex_1())
            .child(icon(
                if self.analysis_open() {
                    "chevron-down.svg"
                } else {
                    "chevron-up.svg"
                },
                16.,
                SECONDARY,
            ))
    }

    pub(super) fn render_analysis(&self, height: f32, window: &Window, cx: &Context<Self>) -> Div {
        let analyzing = self.session.activity() == Some(Activity::Analysis);
        let can_analyze = !self.session.game().history.is_empty()
            && self.session.activity() != Some(Activity::Move);
        let (label, image, color, background, hover) = if analyzing {
            (
                tr!("abort"),
                "square.svg",
                RED,
                rgba(0xc45c5c33),
                rgba(0xc45c5c4d),
            )
        } else {
            (
                tr!("analyze"),
                "search.svg",
                BLUE,
                rgba(0x7ba0c833),
                rgba(0x7ba0c84d),
            )
        };
        let mut level = div().relative().child(
            interactive(
                div().id("analysis-level"),
                tr!("analysis_level"),
                !analyzing,
            )
            .aria_expanded(self.analysis_view.level_menu)
            .h_6()
            .min_w(px(48.))
            .px_2()
            .flex()
            .items_center()
            .justify_between()
            .gap(px(6.))
            .rounded(px(10.))
            .border_1()
            .border_color(rgba(0xffffff1a))
            .bg(rgba(0xffffff0d))
            .text_size(px(12.))
            .line_height(relative(1.5))
            .text_color(rgba(0xffffff99))
            .when(analyzing, |button| button.opacity(0.5))
            .when(!analyzing, |button| {
                button
                    .cursor_pointer()
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.analysis_view.level_menu = !this.analysis_view.level_menu;
                        cx.notify();
                    }))
            })
            .child(self.settings.analysis_level.to_string())
            .child(icon("chevron-down.svg", 16., SECONDARY)),
        );
        if self.analysis_view.level_menu && self.modal.is_none() {
            level = level.child(
                deferred(
                    div()
                        // Keep a press on the slider from reaching the dismiss layer
                        // below, which would close the menu before the drag starts.
                        .occlude()
                        .on_any_mouse_down(|_, _, cx| cx.stop_propagation())
                        .absolute()
                        .top(px(28.))
                        .right_0()
                        .p_1()
                        .rounded(px(10.))
                        .border_1()
                        .border_color(rgba(0xffffff1a))
                        .bg(rgb(0x2a2a2f))
                        .shadow(vec![
                            shadow(rgba(0x00000080), 4., 8., -2.),
                            shadow(rgba(0xffffff0f), 0., 0., 1.),
                        ])
                        .text_size(px(12.))
                        .child(self.render_level_slider(SliderKind::AnalysisLevel, window, cx)),
                )
                .with_priority(2),
            );
        }
        div()
            .h(px(height))
            .flex_shrink_0()
            .min_h_0()
            .flex()
            .flex_col()
            .p_4()
            .gap_4()
            .bg(rgb(PANEL))
            .child(
                div()
                    .flex()
                    .flex_shrink_0()
                    .items_center()
                    .justify_between()
                    .gap_2()
                    .child(
                        div()
                            .flex()
                            .h_9()
                            .p_1()
                            .rounded(px(10.))
                            .bg(rgba(0xffffff0a))
                            .children(
                                [
                                    (false, "activity.svg", tr!("thinking_log")),
                                    (true, "chart.svg", tr!("evaluation")),
                                ]
                                .into_iter()
                                .map(|(chart, image, title)| {
                                    interactive(
                                        div().id(if chart { "chart-tab" } else { "log-tab" }),
                                        title.clone(),
                                        true,
                                    )
                                    .role(Role::Tab)
                                    .aria_selected(self.analysis_view.chart == chart)
                                    .on_key_down(cx.listener(
                                        move |this, event: &KeyDownEvent, window, cx| {
                                            if matches!(
                                                event.keystroke.key.as_str(),
                                                "left" | "right"
                                            ) {
                                                cx.stop_propagation();
                                                if chart == (event.keystroke.key == "right") {
                                                    return;
                                                }
                                                this.action(Action::Chart(!chart), window, cx);
                                                if event.keystroke.key == "right" {
                                                    window.focus_next(cx);
                                                } else {
                                                    window.focus_prev(cx);
                                                }
                                            }
                                        },
                                    ))
                                    .h_7()
                                    .px(px(6.))
                                    .border_1()
                                    .border_color(gpui_kit::transparent_black())
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .rounded(px(8.))
                                    .text_size(px(12.))
                                    .line_height(relative(1.4285714))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(if self.analysis_view.chart == chart {
                                        rgba(0xffffffeb)
                                    } else {
                                        rgba(0xffffff99)
                                    })
                                    .when(self.analysis_view.chart == chart, |tab| {
                                        tab.bg(rgb(BACKGROUND))
                                    })
                                    .cursor_pointer()
                                    .hover(|tab| tab.text_color(rgba(0xffffffeb)))
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        this.analysis_view.chart = chart;
                                        cx.notify();
                                    }))
                                    .child(Icon::empty().path(image).size_4())
                                    .child(title)
                                }),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(
                                        div()
                                            .text_size(px(12.))
                                            .line_height(relative(1.5))
                                            .text_color(rgb(SECONDARY))
                                            .child(tr!("analysis_level")),
                                    )
                                    .child(level),
                            )
                            .child(
                                interactive(div().id("analyze"), label.clone(), can_analyze)
                                    .h_7()
                                    .px_3()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .rounded(px(8.))
                                    .border_1()
                                    .border_color(gpui_kit::transparent_black())
                                    .text_size(px(12.))
                                    .line_height(relative(1.5))
                                    .font_weight(FontWeight::MEDIUM)
                                    .bg(background)
                                    .text_color(rgb(color))
                                    .when(!can_analyze, |button| button.opacity(0.5))
                                    .when(can_analyze, |button| {
                                        button
                                            .cursor_pointer()
                                            .hover(move |style| style.bg(hover))
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.analysis_view.chart = true;
                                                this.action(Action::Analyze, window, cx);
                                            }))
                                    })
                                    .child(icon(image, 14., color))
                                    .child(label),
                            ),
                    ),
            )
            .child(if self.analysis_view.chart {
                self.render_chart(cx).into_any_element()
            } else {
                self.render_thinking_log().into_any_element()
            })
    }

    fn render_thinking_log(&self) -> Div {
        div()
            .flex_1()
            .min_h_0()
            .flex()
            .flex_col()
            .rounded(px(10.))
            .overflow_hidden()
            .border_1()
            .border_color(rgba(0xffffff1a))
            .bg(rgba(0xffffff0d))
            .text_size(px(13.))
            .line_height(relative(1.5))
            .when_some(self.session.log().last(), |table, latest| {
                table.child(
                    div()
                        .flex()
                        .flex_shrink_0()
                        .gap_4()
                        .px_3()
                        .py_2()
                        .border_b_1()
                        .border_color(rgba(0xffffff1a))
                        .bg(rgba(0xffffff0d))
                        .children(
                            [
                                (tr!("nodes"), compact_number(latest.nodes as f64, 2)),
                                (
                                    "NPS:".into(),
                                    if latest.nps <= 0. {
                                        "-".into()
                                    } else {
                                        compact_number(latest.nps, 1)
                                    },
                                ),
                            ]
                            .into_iter()
                            .map(|(label, value)| {
                                div()
                                    .flex()
                                    .items_center()
                                    .gap(px(6.))
                                    .child(div().text_color(rgb(SECONDARY)).child(label))
                                    .child(div().font_family("Consolas").child(value))
                            }),
                        ),
                )
            })
            .child(
                div()
                    .flex()
                    .flex_shrink_0()
                    .py_2()
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(rgb(SECONDARY))
                    .bg(rgb(PANEL))
                    .border_b_1()
                    .border_color(rgba(0xffffff1a))
                    .child(div().w(px(108.)).flex_shrink_0().px_3().child(tr!("depth")))
                    .child(div().w(px(96.)).flex_shrink_0().px_3().child(tr!("score")))
                    .child(div().flex_1().px_3().child(tr!("pv_line"))),
            )
            .child(
                div()
                    .id("thinking-log")
                    .track_scroll(&self.log_scroll)
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .when(self.session.log().is_empty(), |list| {
                        list.child(
                            div()
                                .w_full()
                                .py_8()
                                .text_center()
                                .text_color(rgb(SECONDARY))
                                .child(if self.session.activity() == Some(Activity::Move) {
                                    tr!("thinking")
                                } else {
                                    tr!("thinking_log_empty")
                                }),
                        )
                    })
                    .children(
                        self.session
                            .log()
                            .iter()
                            .enumerate()
                            .map(|(index, progress)| {
                                div()
                                    .flex()
                                    .py(px(6.))
                                    .font_family("Consolas")
                                    .border_b_1()
                                    .border_color(rgba(0xffffff0d))
                                    .when(index + 1 == self.session.log().len(), |row| {
                                        row.bg(rgba(0x43a5751a))
                                    })
                                    .child(
                                        div()
                                            .w(px(108.))
                                            .flex_shrink_0()
                                            .px_3()
                                            .whitespace_nowrap()
                                            .child(display_depth(
                                                progress.depth,
                                                progress.probability,
                                            )),
                                    )
                                    .child(
                                        div()
                                            .w(px(96.))
                                            .flex_shrink_0()
                                            .px_3()
                                            .truncate()
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(score_color(progress.score))
                                            .child(display_score(
                                                progress.score,
                                                ScorePrecision::Raw,
                                            )),
                                    )
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w_0()
                                            .px_3()
                                            .truncate()
                                            .text_color(rgb(SECONDARY))
                                            .child(progress.pv.clone()),
                                    )
                            }),
                    ),
            )
    }
}

fn score_color(score: Score) -> gpui_kit::Rgba {
    let tenth = tenth_score(score);
    if tenth > 0. {
        rgb(ACCENT)
    } else if tenth < 0. {
        rgb(RED)
    } else {
        rgba(0xffffffeb)
    }
}
