use std::time::Duration;

use gpui_kit::component::WindowExt;
use gpui_kit::{prelude::*, *};

use super::assets::icon;
use super::widgets::shadow;
use super::{ACCENT, Action, BACKGROUND, PANEL, PanelDrag, ReversiApp, SECONDARY};

impl Render for ReversiApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let viewport = window.viewport_size();
        let width = f32::from(viewport.width);
        let height = f32::from(viewport.height);
        if !self.session.is_ready() {
            return self.render_initialization(width, window, cx);
        }
        let desktop = width >= 1024.;
        self.compact_layout = !desktop;
        if desktop {
            self.compact_analysis_open = false;
        }
        let solver = self.session.is_solving();
        let body_height = (height - 48. - if solver { 0. } else { 40. }).max(0.);
        let analysis_height = if self.analysis_open() && !solver {
            if desktop {
                ((body_height - 1.).max(0.) * self.settings.bottom_panel_size / 100.).max(260.)
            } else {
                280.
            }
        } else {
            0.
        };
        let sidebar_width = (width - 1.) * self.settings.right_panel_size / 100.;
        let mobile_sidebar_height = if width >= 640. { 320. } else { 288. };
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
            body_height - mobile_sidebar_height - padding
        };
        // The board's 9.5-unit frame contains an 8-unit grid: 304px guarantees 256px of cells.
        let board_side = available_width.min(available_height).max(304.);
        let top_height = if desktop {
            (body_height - analysis_height - if analysis_height > 0. { 1. } else { 0. })
                .max(board_side + padding)
        } else {
            board_side + padding + mobile_sidebar_height
        };
        let board = div()
            .flex_1()
            .min_w_0()
            .when(!desktop, |area| {
                area.h(px(board_side + padding)).flex_shrink_0()
            })
            .flex()
            .items_center()
            .justify_center()
            .child(self.render_board(board_side, window, cx));
        let sidebar = div()
            .flex_shrink_0()
            .min_h_0()
            .overflow_hidden()
            .when(desktop, |panel| panel.w(px(sidebar_width)).h_full())
            .when(!desktop, |panel| {
                panel
                    .w_full()
                    .h(px(mobile_sidebar_height))
                    .border_t_1()
                    .border_color(rgba(0xffffff1a))
            })
            .child(if solver {
                self.render_solver(cx).into_any_element()
            } else {
                self.render_sidebar(desktop, cx).into_any_element()
            });
        let end_drag =
            |this: &mut Self, _: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>| {
                if this.panel_drag.take().is_some() {
                    this.save();
                    cx.notify();
                }
            };
        let mut root = self
            .app_shell("reversi-app")
            .flex_col()
            .overflow_hidden()
            .text_size(px(13.))
            .on_mouse_move(cx.listener(move |this, event: &MouseMoveEvent, _, cx| {
                if let Some(drag) = this.panel_drag {
                    match drag {
                        PanelDrag::Right => {
                            this.settings.right_panel_size = ((width - f32::from(event.position.x))
                                / (width - 1.).max(1.)
                                * 100.)
                                .clamp(18., 45.)
                        }
                        PanelDrag::Bottom => {
                            this.settings.bottom_panel_size =
                                ((height - 40. - f32::from(event.position.y))
                                    / (body_height - 1.).max(1.)
                                    * 100.)
                                    .clamp(18., 70.)
                        }
                    }
                    cx.notify();
                }
            }))
            .on_mouse_up(MouseButton::Left, cx.listener(end_drag))
            .on_mouse_up_out(MouseButton::Left, cx.listener(end_drag))
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                if event.keystroke.key == "escape" && this.modal.is_some() {
                    this.action(Action::Close, window, cx);
                    cx.stop_propagation();
                    return;
                }
                if this.analysis_view.level_menu {
                    if this.level_menu_key(&event.keystroke.key) {
                        cx.notify();
                        cx.stop_propagation();
                    }
                    return;
                }
                if window.has_focused_input(cx) || this.modal.is_some() {
                    return;
                }
                let action = match event.keystroke.key.as_str() {
                    "left" if this.session.game().cursor > 0 => {
                        Some(Action::Navigate(this.session.previous_cursor()))
                    }
                    "right" if this.session.game().cursor < this.session.game().history.len() => {
                        Some(Action::Navigate(this.session.game().cursor + 1))
                    }
                    _ => None,
                };
                if let Some(action) = action {
                    this.action(action, window, cx);
                    cx.stop_propagation();
                }
            }))
            .child(self.render_header(width, cx))
            .child(
                div()
                    .id("workspace-scroll")
                    .flex_1()
                    .min_h_0()
                    .overflow_y_scroll()
                    .child(
                        div()
                            .h(px(top_height))
                            .flex_shrink_0()
                            .flex()
                            .when(!desktop, |area| area.flex_col())
                            .child(board)
                            .when(desktop, |area| {
                                area.child(self.render_splitter(PanelDrag::Right, cx))
                            })
                            .child(sidebar),
                    )
                    .when(analysis_height > 0., |area| {
                        area.when(desktop, |area| {
                            area.child(self.render_splitter(PanelDrag::Bottom, cx))
                        })
                        .child(
                            self.render_analysis(analysis_height, cx)
                                .when(!desktop, |panel| {
                                    panel.border_t_1().border_color(rgba(0xffffff1a))
                                }),
                        )
                    }),
            )
            .when(!solver, |root| root.child(self.render_analysis_header(cx)));
        if self.analysis_view.level_menu && self.modal.is_none() {
            root = root.child(self.render_level_menu_dismiss(cx));
        }
        if self.notification.is_visible() {
            root =
                root.child(deferred(self.render_notification(width, window, cx)).with_priority(3));
        }
        if self.modal.is_some() {
            root = root.child(self.render_modal(window, cx));
        }
        root
    }
}

impl ReversiApp {
    fn render_initialization(
        &self,
        width: f32,
        window: &Window,
        cx: &Context<Self>,
    ) -> Stateful<Div> {
        let mut root = self
            .app_shell("initialization")
            .items_center()
            .justify_center()
            .text_size(px(16.));
        if self.session.initialization_failed() {
            root = root.px_6().child(
                div()
                    .max_w(px(448.))
                    .min_w_0()
                    .p_6()
                    .rounded(px(14.))
                    .border_1()
                    .border_color(rgba(0xffffff1a))
                    .bg(rgb(PANEL))
                    .text_center()
                    .shadow(vec![
                        shadow(rgba(0x0000008c), 8., 16., -4.),
                        shadow(rgba(0xffffff14), 0., 0., 1.),
                    ])
                    .child(
                        div()
                            .text_size(px(18.))
                            .line_height(relative(1.4))
                            .font_weight(FontWeight::SEMIBOLD)
                            .child(tr!("engine_init_failed")),
                    )
                    .child(
                        div()
                            .mt_2()
                            .text_size(px(12.))
                            .line_height(relative(1.4285714))
                            .text_color(rgb(SECONDARY))
                            .child(tr!("unexpected_error")),
                    ),
            );
        } else {
            root = root.child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap_4()
                    .child(div().size_12().with_animation(
                        "initial-loading-spinner",
                        Animation::new(Duration::from_secs(1)).repeat(),
                        |spinner, delta| {
                            spinner.child(
                                canvas(
                                    |_, _, _| (),
                                    move |bounds, _, window, _| {
                                        let angle = delta * std::f32::consts::TAU
                                            - std::f32::consts::FRAC_PI_4;
                                        let center = bounds.center();
                                        let at = |angle: f32| {
                                            center
                                                + point(
                                                    px(angle.cos() * 22.),
                                                    px(angle.sin() * 22.),
                                                )
                                        };
                                        let mut path = PathBuilder::stroke(px(4.));
                                        path.move_to(at(angle));
                                        path.arc_to(
                                            point(px(22.), px(22.)),
                                            px(0.),
                                            true,
                                            true,
                                            at(angle + std::f32::consts::PI * 1.5),
                                        );
                                        if let Ok(path) = path.build() {
                                            window.paint_path(path, rgb(ACCENT));
                                        }
                                    },
                                )
                                .size_full(),
                            )
                        },
                    ))
                    .child(div().text_color(rgb(SECONDARY)).child(tr!("loading"))),
            );
        }
        if self.notification.is_visible() {
            root =
                root.child(deferred(self.render_notification(width, window, cx)).with_priority(3));
        }
        root
    }

    fn app_shell(&self, id: &'static str) -> Stateful<Div> {
        div()
            .id(id)
            .track_focus(&self.focus)
            .relative()
            .size_full()
            .flex()
            .bg(rgb(BACKGROUND))
            .text_color(rgba(0xffffffeb))
            .font_family(if cfg!(target_os = "windows") {
                "Segoe UI"
            } else {
                "sans-serif"
            })
            .line_height(relative(1.5))
    }

    fn render_splitter(&self, axis: PanelDrag, cx: &Context<Self>) -> Stateful<Div> {
        let horizontal = matches!(axis, PanelDrag::Bottom);
        let start_drag =
            move |this: &mut Self, _: &MouseDownEvent, _: &mut Window, cx: &mut Context<Self>| {
                this.panel_drag = Some(axis);
                cx.stop_propagation();
            };
        div()
            .id(if horizontal {
                "bottom-splitter"
            } else {
                "right-splitter"
            })
            .relative()
            .flex_shrink_0()
            .bg(rgba(0xffffff1a))
            .hover(|line| line.bg(rgba(0x7ba0c899)))
            .when(horizontal, |line| {
                line.h(px(1.)).w_full().cursor(CursorStyle::ResizeRow)
            })
            .when(!horizontal, |line| {
                line.w(px(1.)).h_full().cursor(CursorStyle::ResizeColumn)
            })
            .on_mouse_down(MouseButton::Left, cx.listener(start_drag))
            .child(
                div()
                    .id(("splitter-hit", usize::from(horizontal)))
                    .absolute()
                    .when(horizontal, |hit| {
                        hit.left_0()
                            .right_0()
                            .top(px(-4.))
                            .bottom(px(-4.))
                            .cursor(CursorStyle::ResizeRow)
                    })
                    .when(!horizontal, |hit| {
                        hit.top_0()
                            .bottom_0()
                            .left(px(-4.))
                            .right(px(-4.))
                            .cursor(CursorStyle::ResizeColumn)
                    })
                    .on_mouse_down(MouseButton::Left, cx.listener(start_drag)),
            )
            .when(self.modal.is_none(), |line| {
                line.child(deferred(
                    div()
                        .id(("splitter-grip", usize::from(horizontal)))
                        .absolute()
                        .flex()
                        .items_center()
                        .justify_center()
                        .rounded(px(6.))
                        .border_1()
                        .border_color(rgba(0xffffff26))
                        .bg(rgba(0xffffff33))
                        .cursor(if horizontal {
                            CursorStyle::ResizeRow
                        } else {
                            CursorStyle::ResizeColumn
                        })
                        .when(horizontal, |grip| {
                            grip.left(relative(0.5))
                                .ml(px(-10.))
                                .top(px(-5.5))
                                .w(px(20.))
                                .h(px(12.))
                        })
                        .when(!horizontal, |grip| {
                            grip.top(relative(0.5))
                                .mt(px(-10.))
                                .left(px(-5.5))
                                .w(px(12.))
                                .h(px(20.))
                        })
                        .on_mouse_down(MouseButton::Left, cx.listener(start_drag))
                        .child(icon("grip.svg", 10., SECONDARY).when(horizontal, |grip| {
                            grip.with_transformation(gpui_kit::Transformation::rotate(
                                gpui_kit::radians(std::f32::consts::FRAC_PI_2),
                            ))
                        })),
                ))
            })
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
