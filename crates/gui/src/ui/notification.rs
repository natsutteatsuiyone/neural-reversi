use std::time::{Duration, Instant};

use nsh_base::button;
use nsh_core::prelude::*;

use super::assets::icon;
use super::widgets::focusable;
use super::{ACCENT, BLUE, GOLD, SECONDARY, Ui, color, rgba};

pub(super) struct Notification {
    message: String,
    description: String,
    icon: &'static str,
    until: Option<Instant>,
    hover_remaining: Option<Duration>,
    height: Points,
    remove_at: Option<Instant>,
    generation: u64,
}

impl Default for Notification {
    fn default() -> Self {
        Self {
            message: String::new(),
            description: String::new(),
            icon: "info.svg",
            until: None,
            hover_remaining: None,
            height: px(0.),
            remove_at: None,
            generation: 0,
        }
    }
}

impl Notification {
    pub(super) fn is_visible(&self) -> bool {
        !self.message.is_empty()
    }

    pub(super) fn is_dismissible(&self) -> bool {
        self.is_visible()
            && self.until.is_none()
            && self.hover_remaining.is_none()
            && self.remove_at.is_none()
    }

    pub(super) fn clear(&mut self) {
        let generation = self.generation;
        *self = Self {
            generation,
            ..Self::default()
        };
    }

    pub(super) fn show_error(&mut self, message: String, description: String) {
        let height = if self.is_visible() {
            self.height
        } else {
            px(0.)
        };
        *self = Self {
            message,
            description,
            height,
            generation: self.generation.wrapping_add(1),
            ..Self::default()
        };
    }

    pub(super) fn show(
        &mut self,
        message: String,
        description: String,
        icon: &'static str,
        until: Instant,
    ) {
        let height = if self.is_visible() {
            self.height
        } else {
            px(0.)
        };
        *self = Self {
            message,
            description,
            icon,
            until: Some(until),
            height,
            generation: self.generation.wrapping_add(1),
            ..Self::default()
        };
    }

    pub(super) fn tick(&mut self, now: Instant) -> bool {
        let mut changed = false;
        if self.until.is_some_and(|until| now >= until) {
            self.until = None;
            self.remove_at = Some(now + Duration::from_millis(200));
            changed = true;
        }
        if self.remove_at.is_some_and(|remove_at| now >= remove_at) {
            self.message.clear();
            self.remove_at = None;
            changed = true;
        }
        changed
    }

    fn hover(&mut self, hovered: bool) {
        if self.remove_at.is_some() {
            return;
        }
        if hovered {
            if let Some(until) = self.until.take() {
                self.hover_remaining = Some(until.saturating_duration_since(Instant::now()));
            }
        } else if let Some(remaining) = self.hover_remaining.take() {
            self.until = Some(Instant::now() + remaining);
        }
    }
}

pub(super) fn render_notification(ui: Ui) -> impl IntoElement {
    when(
        move || ui.read(|app| app.notification.is_visible()),
        move || {
            let visible = animated_value(0.);
            let offset = animated_value(0.);
            let generation = stored(None::<u64>);
            let width = computed(move || {
                ui.read(|app| {
                    let width = app.viewport.width.0;
                    if width <= 600. {
                        (width - 32.).max(0.)
                    } else {
                        356.
                    }
                })
            });
            let transition = computed(move || {
                ui.read(|app| {
                    (
                        app.notification.generation,
                        app.notification.height.0,
                        app.notification.remove_at.is_some(),
                    )
                })
            });
            effect(move || {
                let (next, height, exiting) = transition.get();
                if height <= 0. {
                    return;
                }
                let first = generation.get() != Some(next);
                if first {
                    generation.replace(Some(next));
                    visible.set(0.);
                    offset.set(-height);
                }
                let to = if exiting { 0. } else { 1. };
                if reduce_motion() {
                    visible.set(to);
                    offset.set(0.);
                } else {
                    let duration = if exiting { 200. } else { 400. };
                    drop(visible.timing(to, duration, Easing::EASE));
                    drop(offset.timing(if exiting { -height } else { 0. }, duration, Easing::EASE));
                }
            });
            view()
                .absolute()
                .left(percentage(0.5))
                .ml(move || px(-width.get() * 0.5))
                .w(move || px(width.get()))
                .top(move || {
                    px(ui.read(|app| {
                        if app.viewport.width.0 <= 600. {
                            16.
                        } else {
                            24.
                        }
                    }))
                })
                .bind_opacity(&visible)
                .bind_translate_y(&offset)
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap(px(6.))
                        .p_4()
                        .rounded(px(8.))
                        .border_1()
                        .border_color(move || color(if dark() { 0x333333 } else { 0xededed }))
                        .bg(move || color(if dark() { 0x000000 } else { 0xffffff }))
                        .text_color(move || color(if dark() { 0xfcfcfc } else { 0x171717 }))
                        .text_size(px(13.))
                        .line_height(nsh_core::LineHeight::Multiple(1.5))
                        .shadow([BoxShadow {
                            offset_y: px(4.),
                            blur: px(12.),
                            color: rgba(0x0000001a).into(),
                            ..BoxShadow::default()
                        }])
                        .on_layout(move |bounds: &Bounds<Points>| {
                            if ui.read(|app| app.notification.height) != bounds.size.height {
                                ui.update(|app| app.notification.height = bounds.size.height);
                            }
                        })
                        .on_mouse_enter(move |_: &MouseEvent| {
                            ui.update(|app| app.notification.hover(true))
                        })
                        .on_mouse_leave(move |_: &MouseEvent| {
                            ui.update(|app| app.notification.hover(false))
                        })
                        .child(
                            view()
                                .size_4()
                                .flex_shrink_0()
                                .flex_row()
                                .items_center()
                                .ml(px(-3.))
                                .mr_1()
                                .child(
                                    when(
                                        move || {
                                            ui.read(|app| app.notification.icon == "trophy.svg")
                                        },
                                        || icon("trophy.svg", 16., GOLD),
                                    )
                                    .otherwise(move || {
                                        when(
                                            move || {
                                                ui.read(|app| app.notification.icon == "bot.svg")
                                            },
                                            || icon("bot.svg", 16., BLUE),
                                        )
                                        .otherwise(|| icon("info.svg", 16., BLUE))
                                    }),
                                ),
                        )
                        .child(
                            view()
                                .flex_1()
                                .min_w_0()
                                .flex_col()
                                .gap(px(2.))
                                .child(
                                    text(move || ui.read(|app| app.notification.message.clone()))
                                        .font_weight(FontWeight::MEDIUM),
                                )
                                .child(when(
                                    move || ui.read(|app| !app.notification.description.is_empty()),
                                    move || {
                                        text(move || {
                                            ui.read(|app| app.notification.description.clone())
                                        })
                                        .font_weight(FontWeight::NORMAL)
                                        .line_height(nsh_core::LineHeight::Multiple(1.4))
                                        .text_color(
                                            move || color(if dark() { 0xe8e8e8 } else { 0x3f3f3f }),
                                        )
                                    },
                                )),
                        )
                        .child(when(
                            move || ui.read(|app| app.notification.is_dismissible()),
                            move || {
                                let focus = focusable(
                                    ui,
                                    "notification-close",
                                    move || ui.read(|app| app.notification.is_dismissible()),
                                    move || ui.update(|app| app.notification.clear()),
                                );
                                button()
                                    .element_ref(&focus)
                                    .accessibility_label(move || Some(tr!("close")))
                                    .size_6()
                                    .flex_shrink_0()
                                    .items_center()
                                    .justify_center()
                                    .rounded(px(6.))
                                    .bg(rgba(0))
                                    .hover(|s| {
                                        s.bg(move || {
                                            if dark() {
                                                rgba(0xffffff1a)
                                            } else {
                                                rgba(0x00000014)
                                            }
                                        })
                                    })
                                    .style_when(
                                        move || ui.is_focus_visible("notification-close"),
                                        |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
                                    )
                                    .on_press(move || ui.update(|app| app.notification.clear()))
                                    .child(icon("x.svg", 14., SECONDARY))
                            },
                        )),
                )
        },
    )
}
