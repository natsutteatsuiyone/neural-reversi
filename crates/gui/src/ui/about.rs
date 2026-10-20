use gpui_kit::{prelude::*, *};

use super::dialogs::{DIALOG_SMALL_LINE_HEIGHT, dialog_tab, dialog_tabs};
use super::{Action, PANEL, ReversiApp};

/// License text shown by the About dialog, with a native-style scrollbar.
pub(super) struct LicenseViewer {
    third_party: bool,
    lines: Vec<SharedString>,
    list: ListState,
    scroll_drag: Option<(Pixels, Pixels)>,
}

impl LicenseViewer {
    pub(super) fn new() -> Self {
        Self {
            third_party: false,
            lines: Vec::new(),
            list: ListState::new(0, ListAlignment::Top, px(0.)).measure_all(),
            scroll_drag: None,
        }
    }

    pub(super) fn load(&mut self, third_party: bool) {
        self.third_party = third_party;
        self.scroll_drag = None;
        self.lines = license_text(third_party)
            .lines()
            .map(|line| SharedString::from(line.to_owned()))
            .collect();
        self.list.reset(self.lines.len());
    }

    pub(super) fn end_drag(&mut self) {
        self.scroll_drag = None;
        self.list.scrollbar_drag_ended();
    }

    /// Follows a thumb drag; returns whether the list scrolled.
    pub(super) fn drag(&mut self, event: &MouseMoveEvent) -> bool {
        let Some((start_y, start_offset)) = self.scroll_drag else {
            return false;
        };
        if event.pressed_button != Some(MouseButton::Left) {
            self.end_drag();
            return false;
        }
        let bounds = scrollbar_bounds(&self.list);
        let thumb = scrollbar_thumb(&self.list, bounds);
        let travel = bounds.size.height - px(36.) - thumb.size.height;
        let maximum = self.list.max_offset_for_scrollbar().y;
        if travel <= px(0.) {
            return false;
        }
        let offset = (start_offset - (event.position.y - start_y) / travel * maximum)
            .clamp(-maximum, px(0.));
        self.list.set_offset_from_scrollbar(point(px(0.), offset));
        true
    }

    /// Starts a thumb drag, or steps by an arrow or a page on the track.
    fn press(&mut self, position: Point<Pixels>) {
        let bounds = scrollbar_bounds(&self.list);
        let thumb = scrollbar_thumb(&self.list, bounds);
        let offset = self.list.scroll_px_offset_for_scrollbar().y;
        if thumb.contains(&position) {
            self.list.scrollbar_drag_started();
            self.scroll_drag = Some((position.y, offset));
        } else {
            let step = if position.y < bounds.top() + px(15.) {
                px(40.)
            } else if position.y > bounds.bottom() - px(15.) {
                px(-40.)
            } else if position.y < thumb.top() {
                self.list.viewport_bounds().size.height
            } else {
                -self.list.viewport_bounds().size.height
            };
            let maximum = self.list.max_offset_for_scrollbar().y;
            self.list
                .set_offset_from_scrollbar(point(px(0.), (offset + step).clamp(-maximum, px(0.))));
        }
    }
}

fn license_text(third_party: bool) -> &'static str {
    if third_party {
        concat!(
            "Lucide icons\n\n",
            include_str!("../../assets/icons/LICENSE"),
            "\n\n",
            "Noto Sans board labels\n\n",
            include_str!("../../assets/fonts/LICENSE"),
            "\n\n",
            include_str!("../../THIRD_PARTY_LICENSES.txt")
        )
    } else {
        concat!(
            include_str!("../../../../NOTICE"),
            "\n",
            include_str!("../../../../LICENSE")
        )
    }
}

fn scrollbar_bounds(state: &ListState) -> Bounds<Pixels> {
    let viewport = state.viewport_bounds();
    Bounds::new(
        point(viewport.right() + px(12.), viewport.top() - px(12.)),
        size(px(15.), viewport.size.height + px(24.)),
    )
}

fn scrollbar_thumb(state: &ListState, bounds: Bounds<Pixels>) -> Bounds<Pixels> {
    let maximum = state.max_offset_for_scrollbar().y;
    let track_height = (bounds.size.height - px(36.)).max(px(0.));
    let thumb_height = (track_height * (bounds.size.height / (bounds.size.height + maximum)))
        .max(px(18.))
        .min(track_height);
    let fraction = if maximum > px(0.) {
        (-state.scroll_px_offset_for_scrollbar().y / maximum).clamp(0., 1.)
    } else {
        0.
    };
    Bounds::new(
        point(
            bounds.left() + px(3.),
            bounds.top() + px(18.) + (track_height - thumb_height) * fraction,
        ),
        size(px(9.), thumb_height),
    )
}

impl ReversiApp {
    pub(super) fn render_about_dialog(&self, list_height: Pixels, cx: &Context<Self>) -> Div {
        div()
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_6()
            .py_2()
            .child(
                dialog_tabs()
                    .child(
                        dialog_tab(
                            "license-main",
                            tr!("license"),
                            None,
                            Action::License(false),
                            !self.license.third_party,
                            cx,
                        )
                        .track_focus(&self.modal_entry_focus),
                    )
                    .child(dialog_tab(
                        "license-third-party",
                        tr!("third_party"),
                        None,
                        Action::License(true),
                        self.license.third_party,
                        cx,
                    )),
            )
            .child(
                div()
                    .relative()
                    .w_full()
                    .min_w_0()
                    .h(list_height)
                    .flex_shrink_0()
                    .border_1()
                    .border_color(rgba(0xffffff1a))
                    .rounded(px(8.))
                    .bg(rgb(PANEL))
                    .py_3()
                    .pl_3()
                    .pr(px(27.))
                    .overflow_hidden()
                    .child(
                        list(
                            self.license.list.clone(),
                            cx.processor(|this, index: usize, _, _| {
                                div()
                                    .w_full()
                                    .min_w_0()
                                    .min_h(px(DIALOG_SMALL_LINE_HEIGHT))
                                    .text_size(px(12.))
                                    .text_color(rgba(0xffffff99))
                                    .font_family("Consolas")
                                    .line_height(px(DIALOG_SMALL_LINE_HEIGHT))
                                    .whitespace_normal()
                                    .child(this.license.lines[index].clone())
                                    .into_any_element()
                            }),
                        )
                        .size_full(),
                    )
                    .child(self.render_license_scrollbar(cx)),
            )
    }

    fn render_license_scrollbar(&self, cx: &Context<Self>) -> Stateful<Div> {
        let state = self.license.list.clone();
        div()
            .id("license-scrollbar")
            .absolute()
            .right_0()
            .top_0()
            .bottom_0()
            .w(px(15.))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, _, cx| {
                    this.license.press(event.position);
                    cx.notify();
                    cx.stop_propagation();
                }),
            )
            .child(
                canvas(
                    |bounds, _, _| bounds,
                    move |_, bounds, window, _| {
                        window.paint_quad(fill(bounds, rgb(0xfcfcfc)));
                        let thumb = scrollbar_thumb(&state, bounds);
                        window.paint_quad(fill(thumb, rgb(0x8b8b8b)).corner_radii(px(4.5)));
                        let center_x = bounds.left() + px(7.5);
                        for (center_y, direction) in [
                            (bounds.top() + px(7.5), -1.),
                            (bounds.bottom() - px(7.5), 1.),
                        ] {
                            let mut arrow = PathBuilder::fill();
                            arrow.move_to(point(center_x, center_y + px(3. * direction)));
                            arrow.line_to(point(center_x - px(4.5), center_y - px(3. * direction)));
                            arrow.line_to(point(center_x + px(4.5), center_y - px(3. * direction)));
                            arrow.close();
                            if let Ok(path) = arrow.build() {
                                window.paint_path(path, rgb(0x8b8b8b));
                            }
                        }
                    },
                )
                .size_full(),
            )
    }
}
