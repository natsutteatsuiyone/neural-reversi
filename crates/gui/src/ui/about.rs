use nsh_core::prelude::*;

use super::dialogs::{dialog_tab, dialog_tabs};
use super::{Action, PANEL, Ui, color, rgba};

pub(super) struct LicenseViewer {
    pub(super) third_party: bool,
    pub(super) lines: Vec<SharedString>,
    pub(super) scroll: ElementRef,
    metrics: ScrollEvent,
    scroll_drag: Option<(f32, f32)>,
}

impl Default for LicenseViewer {
    fn default() -> Self {
        Self::new()
    }
}

impl LicenseViewer {
    pub(super) fn new() -> Self {
        Self {
            third_party: false,
            lines: Vec::new(),
            scroll: element_ref(),
            metrics: ScrollEvent::default(),
            scroll_drag: None,
        }
    }

    pub(super) fn load(&mut self, third_party: bool) {
        self.third_party = third_party;
        self.lines = license_text(third_party)
            .lines()
            .map(SharedString::from)
            .collect();
        self.scroll_drag = None;
        self.metrics.offset = point(px(0.), px(0.));
        self.scroll.scroll_to_offset(px(0.), false);
    }

    pub(super) fn end_drag(&mut self) {
        self.scroll_drag = None;
    }

    fn maximum(&self) -> f32 {
        (self.metrics.content_size.height.0 - self.metrics.size.height.0).max(0.)
    }

    fn thumb(&self, height: f32) -> (f32, f32) {
        let maximum = self.maximum();
        let track = (height - 36.).max(0.);
        let length = (track * (height / (height + maximum).max(1.)))
            .max(18.)
            .min(track);
        let fraction = if maximum > 0. {
            (self.metrics.offset.y.0 / maximum).clamp(0., 1.)
        } else {
            0.
        };
        (18. + (track - length) * fraction, length)
    }

    fn press(&mut self, local_y: f32, page_y: f32, height: f32) -> Option<f32> {
        let (top, length) = self.thumb(height);
        let offset = self.metrics.offset.y.0;
        if (top..=top + length).contains(&local_y) {
            self.scroll_drag = Some((page_y, offset));
            None
        } else {
            let delta = if local_y < 15. {
                -40.
            } else if local_y > height - 15. {
                40.
            } else if local_y < top {
                -self.metrics.size.height.0
            } else {
                self.metrics.size.height.0
            };
            Some((offset + delta).clamp(0., self.maximum()))
        }
    }

    fn drag(&mut self, page_y: f32, height: f32) -> Option<f32> {
        let (start, offset) = self.scroll_drag?;
        let (_, length) = self.thumb(height);
        let travel = height - 36. - length;
        if travel <= 0. {
            return None;
        }
        Some((offset + (page_y - start) / travel * self.maximum()).clamp(0., self.maximum()))
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

pub(super) fn render_about_dialog(ui: Ui) -> View {
    let scroll_reference = ui.read(|a| a.license.scroll);
    view()
        .w_full()
        .min_w_0()
        .gap_6()
        .py_2()
        .child(
            dialog_tabs()
                .child(dialog_tab(
                    ui,
                    "license-main",
                    move || tr!("license"),
                    None,
                    Action::License(false),
                    move || ui.read(|a| !a.license.third_party),
                ))
                .child(dialog_tab(
                    ui,
                    "license-third-party",
                    move || tr!("third_party"),
                    None,
                    Action::License(true),
                    move || ui.read(|a| a.license.third_party),
                )),
        )
        .child(
            view()
                .relative()
                .w_full()
                .min_w_0()
                .h(move || ui.read(|a| px(a.viewport.height.0 * 0.6)))
                .flex_shrink_0()
                .border_1()
                .border_color(rgba(0xffffff1a))
                .rounded(px(8.))
                .bg(color(PANEL))
                .py_3()
                .pl_3()
                .pr(px(27.))
                .overflow_hidden()
                .child(
                    virtual_list(
                        move || ui.read(|a| a.license.lines.len()),
                        move |at| {
                            text(move || {
                                ui.read(|a| {
                                    a.license.lines.get(at.index()).cloned().unwrap_or_default()
                                })
                            })
                            .w_full()
                            .min_w_0()
                            .min_h(px(18.))
                            .text_size(px(12.))
                            .text_color(rgba(0xffffff99))
                            .font_family("Consolas")
                            .line_height(px(18.))
                        },
                    )
                    .element_ref(&scroll_reference)
                    .flex_1()
                    .w_full()
                    .scrollbar(false)
                    .estimated_item_size(px(18.))
                    .keys(move || ui.read(|a| (0..a.license.lines.len()).collect::<Vec<_>>()))
                    .on_scroll(move |event: &ScrollEvent| {
                        if ui.read(|a| a.license.metrics != *event) {
                            ui.update(|a| a.license.metrics = *event);
                        }
                    }),
                )
                .child(render_license_scrollbar(ui)),
        )
}

fn render_license_scrollbar(ui: Ui) -> View {
    let height = signal(0_f32);
    let scroll_reference = ui.read(|a| a.license.scroll);
    let scroll_to = move |offset: Option<f32>| {
        if let Some(offset) = offset {
            scroll_reference.scroll_to_offset(px(offset), false);
        }
    };
    view()
        .absolute()
        .right(px(0.))
        .top(px(0.))
        .bottom(px(0.))
        .w(px(15.))
        .on_layout(move |bounds: &Bounds<Points>| height.set(bounds.size.height.0))
        .claim_touches(TouchClaim::on_start().ask(|request: &ClaimRequest| request.claim()))
        .keep_touches(true)
        .on_responder_grant(move |event: &ResponderEvent| {
            let offset = ui.update(|a| {
                a.license
                    .press(event.location.y.0, event.page.y.0, height.get_untracked())
            });
            scroll_to(offset);
        })
        .on_responder_move(move |event: &ResponderEvent| {
            let offset = ui.update(|a| a.license.drag(event.page.y.0, height.get_untracked()));
            scroll_to(offset);
        })
        .on_responder_release(move |_: &ResponderEvent| ui.update(|a| a.license.end_drag()))
        .on_responder_terminate(move |_: &ResponderEvent| ui.update(|a| a.license.end_drag()))
        .child(
            canvas(move || {
                let height = height.get();
                let (top, length) = ui.read(|a| a.license.thumb(height));
                let mut painter = Painter::new()
                    .fill_color(color(0xfcfcfc))
                    .fill_rect(0., 0., 15., height)
                    .fill_color(color(0x8b8b8b))
                    .begin_path()
                    .round_rect(
                        Bounds::new(point(px(3.), px(top)), size(px(9.), px(length))),
                        px(4.5),
                    )
                    .fill();
                for (y, direction) in [(7.5, -1.), (height - 7.5, 1.)] {
                    painter = painter
                        .begin_path()
                        .move_to(7.5, y + 3. * direction)
                        .line_to(3., y - 3. * direction)
                        .line_to(12., y - 3. * direction)
                        .close_path()
                        .fill();
                }
                painter
            })
            .size_full(),
        )
}
