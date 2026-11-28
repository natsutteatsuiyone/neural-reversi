use std::time::Duration;

use nsh_base::{PopoverEvent, button, popover};
use nsh_core::prelude::*;
use reversi_core::score::Score;

use super::assets::icon;
use super::chart::render_chart;
use super::format::{ScorePrecision, compact_number, display_depth, display_score, tenth_score};
use super::slider::{SliderKind, render_level_slider};
use super::widgets::{focusable, single_line_text};
use super::{ACCENT, Action, BACKGROUND, BLUE, PANEL, RED, ReversiApp, SECONDARY, Ui, color, rgba};
use crate::session::Activity;

#[derive(Default)]
pub(super) struct AnalysisView {
    pub(super) chart: bool,
    pub(super) level_menu: bool,
    pub(super) hover: Option<usize>,
    pub(super) pointer: Option<Point<Points>>,
}

impl AnalysisView {
    pub(super) fn close(&mut self) {
        self.chart = false;
        self.level_menu = false;
        self.hover = None;
        self.pointer = None;
    }
}

pub(super) fn animated_icon(
    name: &'static str,
    side: f32,
    tint: u32,
    duration: Duration,
    rotate: bool,
) -> impl IntoElement {
    let value = animated_value(if rotate { 0. } else { 1. });
    let animation = if rotate {
        Animation::timing(
            &value,
            360.,
            Timing::new(duration.as_millis() as f32, Easing::Linear),
        )
        .looped(None)
    } else {
        Animation::sequence([
            Animation::timing(
                &value,
                0.5,
                Timing::new(
                    duration.as_millis() as f32 * 0.5,
                    Easing::Bezier(0.4, 0., 0.6, 1.),
                ),
            ),
            Animation::timing(
                &value,
                1.,
                Timing::new(
                    duration.as_millis() as f32 * 0.5,
                    Easing::Bezier(0.4, 0., 0.6, 1.),
                ),
            ),
        ])
        .looped(None)
    };
    let running = animation.clone();
    effect(move || {
        running.stop();
        if !reduce_motion() {
            drop(running.start());
        } else {
            value.set(if rotate { 0. } else { 1. });
        }
    });
    cleanup(move || animation.stop());
    view()
        .size(px(side))
        .flex_shrink_0()
        .when(rotate, |v| v.bind_rotate(&value))
        .when(!rotate, |v| v.bind_opacity(&value))
        .child(icon(name, side, tint))
}

pub(super) fn activity_icon(ui: Ui, activity: Activity) -> impl IntoElement {
    when(
        move || ui.read(|app| app.session.activity() == Some(activity)),
        move || animated_icon("activity.svg", 16., BLUE, Duration::from_secs(2), false),
    )
    .otherwise(|| icon("activity.svg", 16., SECONDARY))
}

pub(super) fn render_analysis_header(ui: Ui) -> impl IntoElement {
    let focus = focusable(
        ui,
        "analysis-panel",
        || true,
        move || ui.action(Action::AnalysisPanel),
    );
    view()
        .element_ref(&focus)
        .role(AccessibilityRole::Button)
        .label(move || tr!("ai_analysis"))
        .accessibility_state(move || AccessibilityState {
            expanded: Some(ui.read(|app| app.analysis_open())),
            ..AccessibilityState::default()
        })
        .cursor_pointer()
        .h_10()
        .flex_shrink_0()
        .flex_row()
        .items_center()
        .gap_3()
        .px_4()
        .overflow_hidden()
        .border_t_1()
        .border_color(rgba(0xffffff1a))
        .bg(color(PANEL))
        .hover(|s| s.bg(rgba(0xffffff0d)))
        .style_when(
            move || ui.is_focus_visible("analysis-panel"),
            |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
        )
        .on_press(move || {
            ui.focus("analysis-panel");
            ui.action(Action::AnalysisPanel);
        })
        .child(activity_icon(ui, Activity::Move))
        .child(
            single_line_text(move || tr!("ai_analysis"), 12. * 20. / 14.)
                .flex_shrink_0()
                .text_size(px(12.))
                .font_weight(FontWeight::MEDIUM),
        )
        .child(when(
            move || ui.read(|app| app.session.activity() == Some(Activity::Analysis)),
            move || {
                text(move || {
                    ui.read(|app| {
                        format!(
                            "{}... ({}/{})",
                            tr!("analyzing"),
                            app.session.analyses().len(),
                            app.session
                                .game()
                                .history
                                .iter()
                                .filter(|record| record.square.is_some())
                                .count()
                        )
                    })
                })
                .flex_shrink_0()
                .rounded_full()
                .px_2()
                .py(px(2.))
                .text_size(px(12.))
                .line_height(nsh_core::LineHeight::Multiple(1.5))
                .text_color(color(0xc8924d))
                .bg(rgba(0xc8924d33))
            },
        ))
        .child(when(
            move || {
                ui.read(|app| {
                    !matches!(
                        app.session.activity(),
                        Some(Activity::Move | Activity::Analysis)
                    ) && !app.session.log().is_empty()
                })
            },
            move || {
                view()
                    .flex_row()
                    .min_w_0()
                    .items_center()
                    .gap_2()
                    .text_size(px(12.))
                    .line_height(nsh_core::LineHeight::Multiple(1.5))
                    .font_family("Consolas")
                    .text_color(color(SECONDARY))
                    .child(text(move || tr!("best")))
                    .child(
                        text(move || {
                            ui.read(|app| {
                                app.session
                                    .log()
                                    .last()
                                    .and_then(|p| p.square)
                                    .map_or_else(|| "—".to_owned(), |s| s.to_string())
                            })
                        })
                        .flex_shrink_0()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(color(ACCENT)),
                    )
                    .child(separator())
                    .child(
                        text(move || {
                            ui.read(|app| {
                                app.session.log().last().map_or_else(String::new, |p| {
                                    display_score(p.score, ScorePrecision::Raw)
                                })
                            })
                        })
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(move || {
                            ui.read(|app| {
                                app.session
                                    .log()
                                    .last()
                                    .map_or(rgba(0xffffffeb), |p| score_color(p.score))
                            })
                        }),
                    )
                    .child(separator())
                    .child(
                        text(move || {
                            ui.read(|app| {
                                app.session.log().last().map_or_else(String::new, |p| {
                                    display_depth(p.depth, p.selectivity)
                                })
                            })
                        })
                        .truncate(),
                    )
            },
        ))
        .child(view().flex_1())
        .child(
            when(
                move || ui.read(|app| app.analysis_open()),
                || icon("chevron-down.svg", 16., SECONDARY),
            )
            .otherwise(|| icon("chevron-up.svg", 16., SECONDARY)),
        )
}

fn separator() -> impl IntoElement {
    view()
        .h_3()
        .mx_1()
        .border_l_1()
        .border_color(rgba(0xffffff1a))
}

pub(super) fn render_analysis(ui: Ui, height: Memo<f32>) -> impl IntoElement {
    let analyzing =
        computed(move || ui.read(|app| app.session.activity() == Some(Activity::Analysis)));
    let can_analyze = move || {
        ui.read(|app| {
            !app.session.game().history.is_empty() && app.session.activity() != Some(Activity::Move)
        })
    };
    view()
        .h(move || px(height.get()))
        .flex_shrink_0()
        .min_h_0()
        .flex_col()
        .p_4()
        .gap_4()
        .bg(color(PANEL))
        .child(
            view()
                .flex_row()
                .flex_shrink_0()
                .items_center()
                .justify_between()
                .gap_2()
                .child(
                    view()
                        .flex_row()
                        .h_9()
                        .p_1()
                        .rounded(px(10.))
                        .bg(rgba(0xffffff0a))
                        .children(
                            [(false, "activity.svg"), (true, "chart.svg")]
                                .into_iter()
                                .map(move |(chart, image)| {
                                    let focus = focusable(
                                        ui,
                                        if chart { "chart-tab" } else { "log-tab" },
                                        || true,
                                        move || ui.action(Action::Chart(chart)),
                                    );
                                    button()
                                        .element_ref(&focus)
                                        .role(AccessibilityRole::Tab)
                                        .selected(move || {
                                            ui.read(|app| app.analysis_view.chart == chart)
                                        })
                                        .h_7()
                                        .px(px(6.))
                                        .border_1()
                                        .border_color(rgba(0))
                                        .flex_row()
                                        .items_center()
                                        .gap(px(6.))
                                        .rounded(px(8.))
                                        .text_size(px(12.))
                                        .line_height(nsh_core::LineHeight::Multiple(1.4285714))
                                        .font_weight(FontWeight::MEDIUM)
                                        .text_color(move || {
                                            ui.read(|app| {
                                                if app.analysis_view.chart == chart {
                                                    rgba(0xffffffeb)
                                                } else {
                                                    rgba(0xffffff99)
                                                }
                                            })
                                        })
                                        .bg(move || {
                                            ui.read(|app| {
                                                if app.analysis_view.chart == chart {
                                                    color(BACKGROUND)
                                                } else {
                                                    rgba(0)
                                                }
                                            })
                                        })
                                        .hover(|s| s.text_color(rgba(0xffffffeb)))
                                        .on_press(move || {
                                            ui.focus(if chart { "chart-tab" } else { "log-tab" });
                                            ui.action(Action::Chart(chart));
                                        })
                                        .style_when(
                                            move || {
                                                ui.is_focus_visible(if chart {
                                                    "chart-tab"
                                                } else {
                                                    "log-tab"
                                                })
                                            },
                                            |s| {
                                                s.inset_ring(px(2.)).inset_ring_color(color(ACCENT))
                                            },
                                        )
                                        .child(icon(image, 16., SECONDARY))
                                        .child(single_line_text(
                                            move || {
                                                if chart {
                                                    tr!("evaluation")
                                                } else {
                                                    tr!("thinking_log")
                                                }
                                            },
                                            12. * 20. / 14.,
                                        ))
                                }),
                        ),
                )
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .child(
                            view()
                                .flex_row()
                                .items_center()
                                .gap(px(6.))
                                .text_size(px(12.))
                                .text_color(color(SECONDARY))
                                .child(text(move || tr!("analysis_level")))
                                .child({
                                    let level_focus = focusable(
                                        ui,
                                        "analysis-level",
                                        move || !analyzing.get(),
                                        move || {
                                            ui.action(Action::AnalysisLevelMenu(
                                                !ui.read(|app| app.analysis_view.level_menu),
                                            ))
                                        },
                                    );
                                    popover()
                                        .open(move || {
                                            Some(ui.read(ReversiApp::analysis_level_menu_open))
                                        })
                                        .on_open_change(move |event: &PopoverEvent| {
                                            if let PopoverEvent::OpenChange(open, _) = event {
                                                if *open {
                                                    ui.focus("analysis-level");
                                                }
                                                ui.action(Action::AnalysisLevelMenu(*open));
                                            }
                                        })
                                        .placement(Placement::Bottom)
                                        .align(Align::End)
                                        .offset(px(4.))
                                        .popup_style(|s| {
                                            s.p_1()
                                                .rounded(px(10.))
                                                .border_1()
                                                .border_color(rgba(0xffffff1a))
                                                .bg(color(0x2a2a2f))
                                                .shadow([
                                                    shadow(rgba(0x00000080), 4., 8., -2.),
                                                    shadow(rgba(0xffffff0f), 0., 0., 1.),
                                                ])
                                                .text_size(px(12.))
                                        })
                                        .trigger(
                                            button()
                                                .element_ref(&level_focus)
                                                .on_press(move || ui.focus("analysis-level"))
                                                .accessibility_label(move || tr!("analysis_level"))
                                                .disabled(analyzing)
                                                .style_when(
                                                    move || ui.is_focus_visible("analysis-level"),
                                                    |s| {
                                                        s.inset_ring(px(2.))
                                                            .inset_ring_color(color(ACCENT))
                                                    },
                                                )
                                                .h_6()
                                                .min_w(px(48.))
                                                .px_2()
                                                .flex_row()
                                                .items_center()
                                                .justify_between()
                                                .gap(px(6.))
                                                .rounded(px(10.))
                                                .border_1()
                                                .border_color(rgba(0xffffff1a))
                                                .bg(rgba(0xffffff0d))
                                                .text_size(px(12.))
                                                .line_height(nsh_core::LineHeight::Multiple(1.5))
                                                .text_color(rgba(0xffffff99))
                                                .child(text(move || {
                                                    ui.read(|app| {
                                                        app.settings.analysis_level.to_string()
                                                    })
                                                }))
                                                .child(icon("chevron-down.svg", 16., SECONDARY)),
                                        )
                                        .child(when(
                                            move || ui.read(ReversiApp::analysis_level_menu_open),
                                            move || {
                                                render_level_slider(ui, SliderKind::AnalysisLevel)
                                            },
                                        ))
                                }),
                        )
                        .child(analysis_button(ui, analyzing, can_analyze)),
                ),
        )
        .child(
            when(
                move || ui.read(|app| app.analysis_view.chart),
                move || render_chart(ui),
            )
            .otherwise(move || render_thinking_log(ui)),
        )
}

fn render_thinking_log(ui: Ui) -> impl IntoElement {
    view()
        .flex_1()
        .min_h_0()
        .flex_col()
        .rounded(px(10.))
        .overflow_hidden()
        .border_1()
        .border_color(rgba(0xffffff1a))
        .bg(rgba(0xffffff0d))
        .text_size(px(13.))
        .line_height(nsh_core::LineHeight::Multiple(1.5))
        .child(when(
            move || ui.read(|app| !app.session.log().is_empty()),
            move || {
                view()
                    .flex_row()
                    .flex_shrink_0()
                    .gap_4()
                    .px_3()
                    .py_2()
                    .border_b_1()
                    .border_color(rgba(0xffffff1a))
                    .bg(rgba(0xffffff0d))
                    .child(
                        view()
                            .flex_row()
                            .items_center()
                            .gap(px(6.))
                            .child(
                                single_line_text(move || tr!("nodes"), 13. * 1.5)
                                    .text_color(color(SECONDARY)),
                            )
                            .child(
                                single_line_text(
                                    move || {
                                        ui.read(|app| {
                                            app.session.log().last().map_or_else(String::new, |p| {
                                                compact_number(p.nodes as f64, 2)
                                            })
                                        })
                                    },
                                    13. * 1.5,
                                )
                                .font_family("Consolas"),
                            ),
                    )
                    .child(
                        view()
                            .flex_row()
                            .items_center()
                            .gap(px(6.))
                            .child(single_line_text("NPS:", 13. * 1.5).text_color(color(SECONDARY)))
                            .child(
                                single_line_text(
                                    move || {
                                        ui.read(|app| {
                                            app.session.log().last().map_or_else(String::new, |p| {
                                                if p.nps <= 0. {
                                                    "-".into()
                                                } else {
                                                    compact_number(p.nps, 1)
                                                }
                                            })
                                        })
                                    },
                                    13. * 1.5,
                                )
                                .font_family("Consolas"),
                            ),
                    )
            },
        ))
        .child(
            view()
                .flex_row()
                .flex_shrink_0()
                .py_2()
                .font_weight(FontWeight::MEDIUM)
                .text_color(color(SECONDARY))
                .bg(color(PANEL))
                .border_b_1()
                .border_color(rgba(0xffffff1a))
                .child(
                    single_line_text(move || tr!("depth"), 13. * 1.5)
                        .w(px(108.))
                        .flex_shrink_0()
                        .px_3(),
                )
                .child(
                    single_line_text(move || tr!("score"), 13. * 1.5)
                        .w(px(96.))
                        .flex_shrink_0()
                        .px_3(),
                )
                .child(
                    single_line_text(move || tr!("pv_line"), 13. * 1.5)
                        .flex_1()
                        .px_3(),
                ),
        )
        .child(
            scroll()
                .element_ref(&ui.app.with(|app| app.log_scroll))
                .flex_1()
                .min_h_0()
                .child(
                    view()
                        .flex_col()
                        .child(when(
                            move || ui.read(|app| app.session.log().is_empty()),
                            move || {
                                text(move || {
                                    ui.read(|app| {
                                        if app.session.activity() == Some(Activity::Move) {
                                            tr!("thinking")
                                        } else {
                                            tr!("thinking_log_empty")
                                        }
                                    })
                                })
                                .w_full()
                                .py_8()
                                .text_center()
                                .text_color(color(SECONDARY))
                            },
                        ))
                        .child(each(
                            move || ui.read(|app| (0..app.session.log().len()).collect::<Vec<_>>()),
                            |index| *index,
                            move |index, _| {
                                let index = index.get_untracked();
                                view()
                                    .flex_row()
                                    .py(px(6.))
                                    .font_family("Consolas")
                                    .border_b_1()
                                    .border_color(rgba(0xffffff0d))
                                    .bg(move || {
                                        ui.read(|app| {
                                            if index + 1 == app.session.log().len() {
                                                rgba(0x43a5751a)
                                            } else {
                                                rgba(0)
                                            }
                                        })
                                    })
                                    .child(
                                        single_line_text(
                                            move || {
                                                ui.read(|app| {
                                                    app.session
                                                        .log()
                                                        .get(index)
                                                        .map_or_else(String::new, |p| {
                                                            display_depth(p.depth, p.selectivity)
                                                        })
                                                })
                                            },
                                            13. * 1.5,
                                        )
                                        .w(px(108.))
                                        .flex_shrink_0()
                                        .app_class(Some("whitespace-nowrap".into()))
                                        .px_3(),
                                    )
                                    .child(
                                        single_line_text(
                                            move || {
                                                ui.read(|app| {
                                                    app.session.log().get(index).map_or_else(
                                                        String::new,
                                                        |p| {
                                                            display_score(
                                                                p.score,
                                                                ScorePrecision::Raw,
                                                            )
                                                        },
                                                    )
                                                })
                                            },
                                            13. * 1.5,
                                        )
                                        .w(px(96.))
                                        .flex_shrink_0()
                                        .px_3()
                                        .truncate()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(
                                            move || {
                                                ui.read(|app| {
                                                    app.session
                                                        .log()
                                                        .get(index)
                                                        .map_or(rgba(0xffffffeb), |p| {
                                                            score_color(p.score)
                                                        })
                                                })
                                            },
                                        ),
                                    )
                                    .child(
                                        single_line_text(
                                            move || {
                                                ui.read(|app| {
                                                    app.session
                                                        .log()
                                                        .get(index)
                                                        .map_or_else(String::new, |p| p.pv.clone())
                                                })
                                            },
                                            13. * 1.5,
                                        )
                                        .flex_1()
                                        .min_w_0()
                                        .px_3()
                                        .truncate()
                                        .text_color(color(SECONDARY)),
                                    )
                            },
                        )),
                ),
        )
}

pub(super) fn score_color(score: Score) -> Hsla {
    let tenth = tenth_score(score);
    if tenth > 0. {
        color(ACCENT)
    } else if tenth < 0. {
        color(RED)
    } else {
        rgba(0xffffffeb)
    }
}

impl ReversiApp {
    pub(super) fn analysis_open(&self) -> bool {
        if self.compact_layout {
            self.compact_analysis_open
        } else {
            self.settings.analysis_open
        }
    }
}

fn analysis_button(
    ui: Ui,
    analyzing: Memo<bool>,
    enabled: impl Fn() -> bool + Copy + 'static,
) -> impl IntoElement {
    let focus = focusable(ui, "analyze", enabled, move || {
        ui.action(Action::AnalyzeAndChart)
    });
    button()
        .element_ref(&focus)
        .disabled(move || !enabled())
        .accessibility_label(move || {
            if analyzing.get() {
                tr!("abort")
            } else {
                tr!("analyze")
            }
        })
        .h_7()
        .px_3()
        .flex_row()
        .items_center()
        .gap(px(6.))
        .rounded(px(8.))
        .border_1()
        .border_color(rgba(0))
        .text_size(px(12.))
        .line_height(nsh_core::LineHeight::Multiple(1.5))
        .font_weight(FontWeight::MEDIUM)
        .bg(move || {
            if analyzing.get() {
                rgba(0xc45c5c33)
            } else {
                rgba(0x7ba0c833)
            }
        })
        .text_color(move || {
            if analyzing.get() {
                color(RED)
            } else {
                color(BLUE)
            }
        })
        .hover(move |s| {
            s.bg(move || {
                if analyzing.get() {
                    rgba(0xc45c5c4d)
                } else {
                    rgba(0x7ba0c84d)
                }
            })
        })
        .disabled_style(|s| s.opacity(0.5_f32))
        .style_when(
            move || ui.is_focus_visible("analyze"),
            |s| s.border_color(color(ACCENT)),
        )
        .on_press(move || {
            ui.focus("analyze");
            ui.action(Action::AnalyzeAndChart);
        })
        .child(
            when(move || analyzing.get(), || icon("square.svg", 14., RED))
                .otherwise(|| icon("search.svg", 14., BLUE)),
        )
        .child(text(move || {
            if analyzing.get() {
                tr!("abort")
            } else {
                tr!("analyze")
            }
        }))
}

pub(super) fn shadow(tint: Hsla, y: f32, blur: f32, spread: f32) -> BoxShadow {
    BoxShadow {
        offset_y: px(y),
        blur: px(blur),
        spread: px(spread),
        color: tint.into(),
        ..BoxShadow::default()
    }
}

pub(super) fn focused_key(ui: Ui, key: &str) -> bool {
    if !matches!(key, "left" | "right") {
        return false;
    }
    if ui.is_focused("log-tab") || ui.is_focused("chart-tab") {
        let chart = key == "right";
        ui.action(Action::Chart(chart));
        ui.focus(if chart { "chart-tab" } else { "log-tab" });
        return true;
    }
    false
}
