use nsh_core::desktop::WindowAttributes;
use nsh_core::element::View;
use nsh_core::element::desktop::current_window;
use nsh_core::prelude::*;

use super::assets::icon;
use super::{BACKGROUND, PANEL, PanelDrag, SECONDARY, Ui, color, rgba};
use super::{analysis, board, dialogs, header, notification, sidebar, solver};

#[derive(Clone, Copy, PartialEq)]
struct Geometry {
    desktop: bool,
    solver: bool,
    board_side: f32,
    top_height: f32,
    analysis_height: f32,
    sidebar_width: f32,
    sidebar_height: f32,
    padding: f32,
}

fn geometry(ui: Ui) -> Geometry {
    ui.read(|app| {
        let width = app.viewport.width.0;
        let height = app.viewport.height.0;
        let desktop = width >= 1024.;
        let solver = app.session.is_solving();
        let body_height = (height - 48. - if solver { 0. } else { 40. }).max(0.);
        let analysis_height = if app.analysis_open() && !solver {
            if desktop {
                ((body_height - 1.).max(0.) * app.settings.bottom_panel_size / 100.).max(260.)
            } else {
                280.
            }
        } else {
            0.
        };
        let sidebar_width = (width - 1.) * app.settings.right_panel_size / 100.;
        let sidebar_height = if width >= 640. { 320. } else { 288. };
        let padding = if desktop {
            48.
        } else if width >= 640. {
            32.
        } else {
            24.
        };
        let available_width = if desktop {
            width - sidebar_width - 49.
        } else {
            width - padding
        };
        let available_height = if desktop {
            body_height - analysis_height - 1. - padding
        } else {
            body_height - sidebar_height - padding
        };
        let board_side = available_width.min(available_height).max(304.);
        let top_height = if desktop {
            (body_height - analysis_height - if analysis_height > 0. { 1. } else { 0. })
                .max(board_side + padding)
        } else {
            board_side + padding + sidebar_height
        };
        Geometry {
            desktop,
            solver,
            board_side,
            top_height,
            analysis_height,
            sidebar_width,
            sidebar_height,
            padding,
        }
    })
}

pub(super) fn render(ui: Ui) -> impl IntoElement {
    let mut attributes = WindowAttributes::default();
    attributes.min_size = Some(Some(size(px(480.), px(620.))));
    current_window().set(attributes);
    let geometry = computed(move || geometry(ui));
    let board_side = computed(move || geometry.get().board_side);
    let analysis_height = computed(move || geometry.get().analysis_height);
    view()
        .relative()
        .flex_1()
        .size_full()
        .flex_col()
        .overflow_hidden()
        .on_touch(move |_: &nsh_core::TouchEvent| ui.keyboard_focus.set(false))
        .on_mouse_move(move |event: &MouseEvent| ui.mouse_position.set(event.page_position))
        .font_family(if cfg!(target_os = "windows") {
            "Segoe UI"
        } else {
            "sans-serif"
        })
        .text_size(px(13.))
        .line_height(nsh_core::LineHeight::Multiple(1.5))
        .text_color(rgba(0xffffffeb))
        .bg(color(BACKGROUND))
        .child(
            when(
                move || ui.read(|app| app.session.is_ready()),
                move || {
                    view()
                        .flex_1()
                        .min_h_0()
                        .flex_col()
                        .child(header::render_header(ui))
                        .child(
                            scroll().flex_1().min_h_0().child(
                                view()
                                    .flex_col()
                                    .child(
                                        view()
                                            .h(move || px(geometry.get().top_height))
                                            .flex_shrink_0()
                                            .flex_row()
                                            .style_when(
                                                move || !geometry.get().desktop,
                                                |s| s.flex_col(),
                                            )
                                            .child(
                                                view()
                                                    .flex_1()
                                                    .basis(move || {
                                                        if geometry.get().desktop {
                                                            px(0.).into()
                                                        } else {
                                                            Length::Auto
                                                        }
                                                    })
                                                    .min_w_0()
                                                    .items_center()
                                                    .justify_center()
                                                    .style_when(
                                                        move || !geometry.get().desktop,
                                                        |s| s.flex_none(),
                                                    )
                                                    .h(move || {
                                                        px(if geometry.get().desktop {
                                                            geometry.get().top_height
                                                        } else {
                                                            geometry.get().board_side
                                                                + geometry.get().padding
                                                        })
                                                    })
                                                    .child(board::render_board(ui, board_side)),
                                            )
                                            .child(when(
                                                move || geometry.get().desktop,
                                                move || splitter(ui, PanelDrag::Right),
                                            ))
                                            .child(
                                                view()
                                                    .flex_shrink_0()
                                                    .min_h_0()
                                                    .overflow_hidden()
                                                    .w(move || {
                                                        px(if geometry.get().desktop {
                                                            geometry.get().sidebar_width
                                                        } else {
                                                            ui.read(|app| app.viewport.width.0)
                                                        })
                                                    })
                                                    .h(move || {
                                                        px(if geometry.get().desktop {
                                                            geometry.get().top_height
                                                        } else {
                                                            geometry.get().sidebar_height
                                                        })
                                                    })
                                                    .child(
                                                        when(
                                                            move || geometry.get().solver,
                                                            move || solver::render_solver(ui),
                                                        )
                                                        .otherwise(move || {
                                                            sidebar::render_sidebar(ui)
                                                        }),
                                                    ),
                                            ),
                                    )
                                    .child(when(
                                        move || geometry.get().analysis_height > 0.,
                                        move || {
                                            view()
                                                .flex_col()
                                                .child(when(
                                                    move || geometry.get().desktop,
                                                    move || splitter(ui, PanelDrag::Bottom),
                                                ))
                                                .child(analysis::render_analysis(
                                                    ui,
                                                    analysis_height,
                                                ))
                                        },
                                    )),
                            ),
                        )
                        .child(when(
                            move || !geometry.get().solver,
                            move || analysis::render_analysis_header(ui),
                        ))
                },
            )
            .otherwise(move || initialization(ui)),
        )
        .child(notification::render_notification(ui))
        .child(dialogs::render_modal(ui))
}

fn initialization(ui: Ui) -> impl IntoElement {
    view()
        .flex_1()
        .items_center()
        .justify_center()
        .px_6()
        .child(
            when(
                move || ui.read(|app| app.session.initialization_failed()),
                move || {
                    view()
                        .max_w(px(448.))
                        .p_6()
                        .rounded(px(14.))
                        .border_1()
                        .border_color(rgba(0xffffff1a))
                        .bg(color(PANEL))
                        .gap_2()
                        .child(
                            text(move || tr!("engine_init_failed"))
                                .text_size(px(18.))
                                .line_height(nsh_core::LineHeight::Multiple(1.4))
                                .font_weight(FontWeight::SEMIBOLD),
                        )
                        .child(
                            text(move || tr!("unexpected_error"))
                                .text_size(px(12.))
                                .line_height(nsh_core::LineHeight::Multiple(1.4285714))
                                .text_color(color(SECONDARY)),
                        )
                },
            )
            .otherwise(move || {
                let angle = animated_value(0.);
                let spin = Animation::timing(&angle, 360., Timing::new(1000., Easing::Linear))
                    .looped(None);
                let running = spin.clone();
                effect(move || {
                    if reduce_motion() {
                        running.stop();
                        angle.set(0.);
                    } else {
                        running.start();
                    }
                });
                cleanup(move || spin.stop());
                view()
                    .items_center()
                    .gap_4()
                    .child(icon("loader-circle.svg", 48., SECONDARY).bind_rotate(&angle))
                    .child(text(move || tr!("loading")).text_size(px(16.)))
            }),
        )
}

fn splitter(ui: Ui, axis: PanelDrag) -> View {
    let horizontal = axis == PanelDrag::Bottom;
    let pressed = move |event: &ResponderEvent| {
        resize(ui, axis, event.page);
    };
    view()
        .relative()
        .z_index(1)
        .flex_shrink_0()
        .bg(rgba(0xffffff1a))
        .w(move || -> Length {
            if horizontal {
                relative(1.).into()
            } else {
                px(1.).into()
            }
        })
        .h(move || -> Length {
            if horizontal {
                px(1.).into()
            } else {
                relative(1.).into()
            }
        })
        .child(
            view()
                .absolute()
                .left(px(if horizontal { 0. } else { -4. }))
                .right(px(if horizontal { 0. } else { -4. }))
                .top(px(if horizontal { -4. } else { 0. }))
                .bottom(px(if horizontal { -4. } else { 0. }))
                .cursor(if horizontal {
                    Cursor::RowResize
                } else {
                    Cursor::ColResize
                })
                .claim_touches(TouchClaim::on_start())
                .on_responder_grant(move |event: &ResponderEvent| {
                    ui.update(|app| app.panel_drag = Some(axis));
                    pressed(event);
                })
                .on_responder_move(pressed)
                .on_responder_release(move |_: &ResponderEvent| {
                    ui.update(|app| {
                        app.panel_drag = None;
                        app.save();
                    })
                })
                .on_responder_terminate(move |_: &ResponderEvent| {
                    ui.update(|app| {
                        app.panel_drag = None;
                        app.save();
                    })
                }),
        )
        .child(when(
            move || ui.read(|app| app.modal.is_none()),
            move || {
                view()
                    .absolute()
                    .left(relative(0.5))
                    .top(relative(0.5))
                    .w(px(if horizontal { 20. } else { 12. }))
                    .h(px(if horizontal { 12. } else { 20. }))
                    .ml(px(if horizontal { -10. } else { -6. }))
                    .mt(px(if horizontal { -6. } else { -10. }))
                    .rounded(px(6.))
                    .border_1()
                    .border_color(rgba(0xffffff26))
                    .bg(rgba(0xffffff33))
                    .items_center()
                    .justify_center()
                    .child(icon("grip.svg", 10., SECONDARY).when(horizontal, |icon| {
                        icon.rotate(radians(std::f32::consts::FRAC_PI_2))
                    }))
            },
        ))
}

fn resize(ui: Ui, axis: PanelDrag, position: Point<Points>) {
    ui.update(|app| {
        let width = app.viewport.width.0;
        let height = app.viewport.height.0;
        match axis {
            PanelDrag::Right => {
                app.settings.right_panel_size =
                    ((width - position.x.0) / (width - 1.).max(1.) * 100.).clamp(18., 45.)
            }
            PanelDrag::Bottom => {
                let body = (height - 48. - if app.session.is_solving() { 0. } else { 40. }).max(0.);
                app.settings.bottom_panel_size =
                    ((height - 40. - position.y.0) / (body - 1.).max(1.) * 100.).clamp(18., 70.);
            }
        }
    });
}
