use nsh_base::{radio, radio_group};
use nsh_core::prelude::*;
use reversi_core::level::{ENDGAME_STEPS, Level};

use super::analysis::activity_icon;
use super::assets::icon;
use super::format::{ScorePrecision, display_score, whole_score};
use super::widgets::{control, flat_stone, focusable};
use super::{ACCENT, Action, BACKGROUND, BLUE, PANEL, RED, SECONDARY, Ui, color, rgba};
use crate::session::Activity;

pub(super) fn render_solver(ui: Ui) -> impl IntoElement {
    let searching =
        computed(move || ui.read(|app| app.session.activity() == Some(Activity::Solver)));
    view().flex_col().w_full().h_full().min_h_0().bg(color(PANEL)).text_size(px(12.)).line_height(nsh_core::LineHeight::Multiple(1.4285714))
        .child(view().flex_row().items_center().gap_3().px_4().py(px(10.)).border_b_1().border_color(rgba(0xffffff1a))
            .child(activity_icon(ui,Activity::Solver))
            .child(text(move||tr!("solver")).font_weight(FontWeight::MEDIUM))
            .child(when(move||searching.get(),move||text(move||ui.read(|app| {
                let seconds=app.session.search_elapsed().unwrap_or_default().as_secs();
                format!("{} {}:{:02}",tr!("solving"),seconds/60,seconds%60)
            })).rounded_full().px_2().py(px(2.)).text_size(px(12.)).line_height(nsh_core::LineHeight::Multiple(1.5)).text_color(color(BLUE)).bg(rgba(0x7ba0c833))))
            .child(when(move||ui.read(|app|app.session.is_solver_stopped()),||text(move||tr!("solver_stopped")).text_size(px(12.))))
            .child(view().flex_1())
            .child(control(ui,"solver-exit","x.svg",move||tr!("exit_solver"),Action::ExitSolver,||true)))
        .child(view().flex_col().gap_3().px_4().py_3().border_b_1().border_color(rgba(0xffffff1a))
            .child(view().flex_row().items_center().gap_2()
                .child(control(ui,"solver-back","arrow-left.svg",move||tr!("undo"),Action::Previous,
                    move||ui.read(|app|app.session.game().cursor>0)).border_1().border_color(rgba(0xffffff0f)).bg(color(BACKGROUND)))
                .child(view().flex_1())
                .child(when(move||searching.get()||ui.read(|app|app.session.is_solver_stopped()),move||
                    pause_button(ui,searching))))
            .child(solver_options(ui,false)))
        .child(scroll().flex_1().min_h_0()
            .child(view().flex_col()
                .child(view().flex_row().items_center().gap_3().p_4()
                    .child(text(move||tr!("candidates")).text_size(px(12.)).line_height(nsh_core::LineHeight::Multiple(1.5)).font_weight(FontWeight::MEDIUM).text_color(color(SECONDARY)))
                    .child(view().flex_row().items_center().gap_2()
                        .child(when(move||ui.read(|app|app.session.game().position().turn==reversi_core::player_color::PlayerColor::Black),||flat_stone(true,16.))
                            .otherwise(||flat_stone(false,16.)))
                        .child(text(move||ui.read(|app|app.color(app.session.game().position().turn))).font_weight(FontWeight::MEDIUM))))
                .child(when(move||ui.read(|app|app.session.candidates().is_empty()),move||
                    text(move||ui.read(|app| app.session.search_error().map_or_else(|| {
                        if !app.session.game().position().board.has_legal_moves(){tr!("solver_no_legal_moves")}
                        else if app.session.activity()==Some(Activity::Solver){tr!("solving")}
                        else if app.session.is_solver_stopped(){tr!("solver_no_results")}
                        else{tr!("solver_pending")}
                    },|error|error.to_owned().into())))
                        .px_4().py_6().text_center().text_color(color(SECONDARY))))
                .child(view().flex_col().gap_1().px_2().pb_3()
                    .child(each(move||ui.read(|app|app.session.candidates().iter().filter_map(|p|p.square.map(|s|s.index())).collect::<Vec<_>>()),|square|*square,move|square,_|candidate_row(ui,square.get_untracked()))))))
}

fn candidate_row(ui: Ui, square: usize) -> impl IntoElement {
    let id = format!("candidate-{square}");
    let focus = focusable(ui, &id, || true, move || ui.action(Action::Cell(square)));
    let focus_id = id;
    let best = computed(move || {
        ui.read(|app| {
            let candidate = app
                .session
                .candidates()
                .iter()
                .find(|p| p.square.is_some_and(|s| s.index() == square));
            candidate.is_some_and(|candidate| {
                app.session
                    .candidates()
                    .first()
                    .is_some_and(|first| whole_score(first.score) == whole_score(candidate.score))
            })
        })
    });
    nsh_base::button()
        .element_ref(&focus)
        .accessibility_label(move || {
            ui.read(|app| {
                app.session
                    .candidates()
                    .iter()
                    .find(|p| p.square.is_some_and(|s| s.index() == square))
                    .map_or_else(SharedString::default, |p| {
                        format!(
                            "{} {}",
                            p.square.expect("candidate has its square"),
                            display_score(p.score, ScorePrecision::Whole)
                        )
                        .into()
                    })
            })
        })
        .flex_row()
        .items_center()
        .gap_3()
        .px_3()
        .py_2()
        .rounded(px(8.))
        .text_size(px(15.))
        .bg(move || {
            if best.get() {
                rgba(0x43a5751a)
            } else {
                rgba(0)
            }
        })
        .style_when(
            move || ui.is_focus_visible(&focus_id),
            |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
        )
        .hover(|s| s.bg(rgba(0xffffff0d)))
        .on_press(move || {
            ui.focus(&format!("candidate-{square}"));
            ui.action(Action::Cell(square));
        })
        .child(
            text(
                reversi_core::square::Square::from_u8(square as u8)
                    .expect("candidate square")
                    .to_string(),
            )
            .w_12()
            .flex_shrink_0()
            .font_family("Consolas")
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(move || {
                if best.get() {
                    color(ACCENT)
                } else {
                    rgba(0xffffffeb)
                }
            }),
        )
        .child(
            text(move || {
                ui.read(|app| {
                    app.session
                        .candidates()
                        .iter()
                        .find(|p| p.square.is_some_and(|s| s.index() == square))
                        .map_or_else(String::new, |p| {
                            display_score(p.score, ScorePrecision::Whole)
                        })
                })
            })
            .w(px(56.))
            .flex_shrink_0()
            .font_family("Consolas")
            .text_right()
            .text_color(move || {
                if best.get() {
                    color(ACCENT)
                } else {
                    rgba(0xffffff99)
                }
            }),
        )
        .child(
            text(move || {
                ui.read(|app| {
                    app.session
                        .candidates()
                        .iter()
                        .find(|p| p.square.is_some_and(|s| s.index() == square))
                        .map_or_else(String::new, |p| format!("{}%", p.probability))
                })
            })
            .w_10()
            .flex_shrink_0()
            .text_right()
            .text_size(px(13.))
            .line_height(nsh_core::LineHeight::Multiple(1.5))
            .text_color(color(SECONDARY)),
        )
        .child(
            text(move || {
                ui.read(|app| {
                    app.session
                        .candidates()
                        .iter()
                        .find(|p| p.square.is_some_and(|s| s.index() == square))
                        .map_or_else(String::new, |p| {
                            let end =
                                p.pv.match_indices(' ')
                                    .nth(7)
                                    .map_or(p.pv.len(), |(offset, _)| offset);
                            p.pv[..end].to_owned()
                        })
                })
            })
            .flex_1()
            .min_w_0()
            .truncate()
            .text_size(px(13.))
            .line_height(nsh_core::LineHeight::Multiple(1.5))
            .font_family("Consolas")
            .text_color(color(SECONDARY)),
        )
}

pub(super) fn solver_options(ui: Ui, draft: bool) -> impl IntoElement {
    let options = move |app: &super::ReversiApp| {
        if draft {
            app.draft.clone()
        } else {
            app.settings.clone()
        }
    };
    let prefix = if draft { "draft" } else { "solver" };
    view()
        .flex_row()
        .flex_wrap()
        .gap_6()
        .child(
            view()
                .flex_col()
                .gap_2()
                .child(
                    text(move || tr!("selectivity"))
                        .text_size(px(12.))
                        .line_height(nsh_core::LineHeight::Multiple(1.5))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(color(SECONDARY)),
                )
                .child(radio_group().flex_row().flex_wrap().gap_3().children(
                    Level::ENDGAME_SELECTIVITY.into_iter().enumerate().map(
                        move |(index, selectivity)| {
                            let id = format!("{prefix}-selectivity-{index}");
                            let focus_id = id.clone();
                            let focus = focusable(
                                ui,
                                &id,
                                || true,
                                move || ui.action(Action::Selectivity(index)),
                            );
                            let selected =
                                move || ui.read(|app| options(app).solver_selectivity == index);
                            radio()
                                .element_ref(&focus)
                                .checked(selected)
                                .style_when(
                                    move || ui.is_focus_visible(&focus_id),
                                    |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
                                )
                                .on_change(move |_: &bool| {
                                    ui.focus(&id);
                                    ui.action(Action::Selectivity(index));
                                })
                                .flex_row()
                                .items_center()
                                .gap(px(6.))
                                .text_size(px(12.))
                                .line_height(nsh_core::LineHeight::Multiple(1.4285714))
                                .text_color(rgba(0xffffffeb))
                                .child(radio_dot(selected))
                                .child(text(format!("{}%", selectivity.probability())))
                        },
                    ),
                )),
        )
        .child(
            view()
                .flex_col()
                .gap_2()
                .child(
                    text(move || tr!("mode"))
                        .text_size(px(12.))
                        .line_height(nsh_core::LineHeight::Multiple(1.5))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(color(SECONDARY)),
                )
                .child(radio_group().flex_row().flex_wrap().gap_3().children(
                    [false, true].into_iter().map(move |multi| {
                        let id = format!("{prefix}-multi-{multi}");
                        let focus_id = id.clone();
                        let focus =
                            focusable(ui, &id, || true, move || ui.action(Action::MultiPv(multi)));
                        let selected = move || ui.read(|app| options(app).solver_multi_pv == multi);
                        radio()
                            .element_ref(&focus)
                            .checked(selected)
                            .style_when(
                                move || ui.is_focus_visible(&focus_id),
                                |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
                            )
                            .on_change(move |_: &bool| {
                                ui.focus(&id);
                                ui.action(Action::MultiPv(multi));
                            })
                            .flex_row()
                            .items_center()
                            .gap(px(6.))
                            .text_size(px(12.))
                            .line_height(nsh_core::LineHeight::Multiple(1.4285714))
                            .text_color(rgba(0xffffffeb))
                            .child(radio_dot(selected))
                            .child(text(move || {
                                if multi {
                                    tr!("all_moves")
                                } else {
                                    tr!("best_move_only")
                                }
                            }))
                    }),
                )),
        )
}

fn radio_dot(selected: impl Fn() -> bool + Copy + 'static) -> impl IntoElement {
    view()
        .size_4()
        .flex_shrink_0()
        .rounded_full()
        .border_1()
        .border_color(move || {
            if selected() {
                color(ACCENT)
            } else {
                rgba(0xffffff0f)
            }
        })
        .bg(move || if selected() { color(ACCENT) } else { rgba(0) })
        .items_center()
        .justify_center()
        .child(when(selected, || {
            view().size_2().rounded_full().bg(color(0xffffff))
        }))
}

fn pause_button(ui: Ui, searching: Memo<bool>) -> impl IntoElement {
    let focus = focusable(
        ui,
        "solver-pause",
        || true,
        move || ui.action(Action::Pause),
    );
    nsh_base::button()
        .element_ref(&focus)
        .accessibility_label(move || {
            if searching.get() {
                tr!("stop")
            } else {
                tr!("resume")
            }
        })
        .h_7()
        .px_2()
        .flex_row()
        .items_center()
        .justify_center()
        .gap_1()
        .rounded(px(8.))
        .border_1()
        .border_color(move || {
            if searching.get() {
                rgba(0)
            } else {
                rgba(0xffffff0f)
            }
        })
        .bg(move || {
            if searching.get() {
                rgba(0xc45c5c33)
            } else {
                color(BACKGROUND)
            }
        })
        .text_size(px(12.8))
        .font_weight(FontWeight::MEDIUM)
        .text_color(move || {
            if searching.get() {
                color(RED)
            } else {
                rgba(0xffffff99)
            }
        })
        .style_when(
            move || ui.is_focus_visible("solver-pause"),
            |s| s.border_color(color(ACCENT)),
        )
        .hover(|s| s.bg(rgba(0xc45c5c4d)))
        .on_press(move || {
            ui.focus("solver-pause");
            ui.action(Action::Pause);
        })
        .child(
            when(move || searching.get(), || icon("square.svg", 14., RED))
                .otherwise(|| icon("play.svg", 14., SECONDARY)),
        )
        .child(text(move || {
            if searching.get() {
                tr!("stop")
            } else {
                tr!("resume")
            }
        }))
}

pub(super) fn focused_key(ui: Ui, key: &str) -> bool {
    let forward = match key {
        "right" | "down" => true,
        "left" | "up" => false,
        _ => return false,
    };
    for prefix in ["solver", "draft"] {
        for index in 0..ENDGAME_STEPS {
            if ui.is_focused(&format!("{prefix}-selectivity-{index}")) {
                let next = if forward {
                    (index + 1).min(ENDGAME_STEPS - 1)
                } else {
                    index.saturating_sub(1)
                };
                ui.action(Action::Selectivity(next));
                ui.focus(&format!("{prefix}-selectivity-{next}"));
                return true;
            }
        }
        for multi in [false, true] {
            if ui.is_focused(&format!("{prefix}-multi-{multi}")) {
                let next = forward;
                ui.action(Action::MultiPv(next));
                ui.focus(&format!("{prefix}-multi-{next}"));
                return true;
            }
        }
    }
    false
}
