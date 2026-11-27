use nsh_core::prelude::*;

use super::dialogs::{dialog_tab, dialog_tabs};
use super::{Action, PANEL, Ui, color, rgba};

pub(super) struct LicenseViewer {
    pub(super) third_party: bool,
    pub(super) lines: Vec<SharedString>,
    pub(super) scroll: ElementRef,
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
        self.scroll.scroll_to_offset(px(0.), false);
    }

    pub(super) fn end_drag(&mut self) {
        self.scroll_drag = None;
    }

    fn viewport(metrics: &ScrollEvent, height: f32) -> f32 {
        if metrics.size.height.0 > 0. {
            metrics.size.height.0
        } else {
            (height - 24.).max(0.)
        }
    }

    fn maximum(metrics: &ScrollEvent, height: f32) -> f32 {
        (metrics.content_size.height.0 - Self::viewport(metrics, height)).max(0.)
    }

    fn thumb(metrics: &ScrollEvent, height: f32) -> (f32, f32) {
        let maximum = Self::maximum(metrics, height);
        let track = (height - 36.).max(0.);
        let length = (track * (height / (height + maximum).max(1.)))
            .max(18.)
            .min(track);
        let fraction = if maximum > 0. {
            (metrics.offset.y.0 / maximum).clamp(0., 1.)
        } else {
            0.
        };
        (18. + (track - length) * fraction, length)
    }

    fn press(
        &mut self,
        metrics: &ScrollEvent,
        local_y: f32,
        page_y: f32,
        height: f32,
    ) -> Option<f32> {
        let (top, length) = Self::thumb(metrics, height);
        let offset = metrics.offset.y.0;
        if (top..=top + length).contains(&local_y) {
            self.scroll_drag = Some((page_y, offset));
            None
        } else {
            let page = Self::viewport(metrics, height);
            let delta = if local_y < 15. {
                -40.
            } else if local_y > height - 15. {
                40.
            } else if local_y < top {
                -page
            } else {
                page
            };
            Some((offset + delta).clamp(0., Self::maximum(metrics, height)))
        }
    }

    fn drag(&mut self, metrics: &ScrollEvent, page_y: f32, height: f32) -> Option<f32> {
        let (start, offset) = self.scroll_drag?;
        let (_, length) = Self::thumb(metrics, height);
        let travel = height - 36. - length;
        if travel <= 0. {
            return None;
        }
        let maximum = Self::maximum(metrics, height);
        Some((offset + (page_y - start) / travel * maximum).clamp(0., maximum))
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
    let scroll_reference = ui.app.with(|a| a.license.scroll);
    let license_tab = computed(move || ui.read(|a| (a.license.third_party, a.license.lines.len())));
    let metrics = signal(ScrollEvent::default());
    effect(move || {
        let (_, lines) = license_tab.get();
        metrics.set(ScrollEvent {
            offset: point(px(0.), px(0.)),
            content_size: size(px(0.), px(lines as f32 * 18.)),
            ..ScrollEvent::default()
        });
    });
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
                        move || license_tab.get().1,
                        move |at| {
                            text(move || {
                                license_tab.get();
                                ui.app.with(|a| {
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
                    .on_scroll(move |event: &ScrollEvent| {
                        if metrics.get_untracked() != *event {
                            metrics.set(*event);
                        }
                    }),
                )
                .child(render_license_scrollbar(ui, metrics)),
        )
}

fn render_license_scrollbar(ui: Ui, metrics: Signal<ScrollEvent>) -> View {
    let height = signal(0_f32);
    let scroll_reference = ui.app.with(|a| a.license.scroll);
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
            let current = metrics.get_untracked();
            let offset = ui.app.with(|a| {
                a.license.press(
                    &current,
                    event.location.y.0,
                    event.page.y.0,
                    height.get_untracked(),
                )
            });
            scroll_to(offset);
        })
        .on_responder_move(move |event: &ResponderEvent| {
            let current = metrics.get_untracked();
            let offset = ui.app.with(|a| {
                a.license
                    .drag(&current, event.page.y.0, height.get_untracked())
            });
            scroll_to(offset);
        })
        .on_responder_release(move |_: &ResponderEvent| ui.app.with(|a| a.license.end_drag()))
        .on_responder_terminate(move |_: &ResponderEvent| ui.app.with(|a| a.license.end_drag()))
        .child(
            canvas(move || {
                let height = height.get();
                let current = metrics.get();
                let (top, length) = LicenseViewer::thumb(&current, height);
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
