use std::time::{Duration, Instant};

use gpui_kit::{prelude::*, *};

use super::assets::icon;
use super::widgets::shadow;
use super::{BLUE, GOLD, ReversiApp};

/// Toast state: an optional timeout, pausing while hovered, and a 200ms exit.
pub(super) struct Notification {
    message: String,
    description: String,
    icon: &'static str,
    until: Option<Instant>,
    hover_remaining: Option<Duration>,
    height: Pixels,
    remove_at: Option<Instant>,
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
        }
    }
}

impl Notification {
    pub(super) fn is_visible(&self) -> bool {
        !self.message.is_empty()
    }

    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Shows `message` until it is cleared or replaced.
    pub(super) fn show_error(&mut self, message: String) {
        *self = Self {
            message,
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
        *self = Self {
            message,
            description,
            icon,
            until: Some(until),
            ..Self::default()
        };
    }

    /// Advances the timeout and exit animation; returns whether a redraw is needed.
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
            self.hover_remaining = self
                .until
                .take()
                .map(|until| until.saturating_duration_since(Instant::now()));
        } else if let Some(remaining) = self.hover_remaining.take() {
            self.until = Some(Instant::now() + remaining);
        }
    }
}

impl ReversiApp {
    pub(super) fn render_notification(
        &self,
        width: f32,
        window: &Window,
        cx: &Context<Self>,
    ) -> AnyElement {
        let notification = &self.notification;
        let dark = matches!(
            window.appearance(),
            WindowAppearance::Dark | WindowAppearance::VibrantDark
        );
        let mobile = width <= 600.;
        let toast_width = if mobile { (width - 32.).max(0.) } else { 356. };
        let message: SharedString = notification.message.clone().into();
        let exiting = notification.remove_at.is_some();
        let height = notification.height;
        let view = cx.entity().downgrade();
        let toast = div()
            .id("notification")
            .flex()
            .items_center()
            .gap(px(6.))
            .p_4()
            .rounded(px(8.))
            .border_1()
            .border_color(rgb(if dark { 0x333333 } else { 0xededed }))
            .bg(rgb(if dark { 0x000000 } else { 0xffffff }))
            .text_color(rgb(if dark { 0xfcfcfc } else { 0x171717 }))
            .text_size(px(13.))
            .line_height(relative(1.5))
            .shadow(vec![shadow(rgba(0x0000001a), 4., 12., 0.)])
            .on_hover(cx.listener(|this, hovered, _, _| {
                this.notification.hover(*hovered);
            }))
            .child(
                div()
                    .size_4()
                    .flex_shrink_0()
                    .flex()
                    .items_center()
                    .ml(px(-3.))
                    .mr_1()
                    .child(
                        icon(
                            notification.icon,
                            16.,
                            if notification.icon == "trophy.svg" {
                                GOLD
                            } else {
                                BLUE
                            },
                        )
                        .ml(px(-1.)),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap(px(2.))
                    .child(
                        div()
                            .font_weight(FontWeight::MEDIUM)
                            .whitespace_normal()
                            .child(message.clone()),
                    )
                    .when(!notification.description.is_empty(), |content| {
                        content.child(
                            div()
                                .font_weight(FontWeight::NORMAL)
                                .line_height(relative(1.4))
                                .text_color(rgb(if dark { 0xe8e8e8 } else { 0x3f3f3f }))
                                .whitespace_normal()
                                .child(notification.description.clone()),
                        )
                    }),
            );
        div()
            .absolute()
            .left(relative(0.5))
            .ml(px(-toast_width * 0.5))
            .w(px(toast_width))
            .child(toast)
            .child(
                canvas(
                    move |bounds, _, cx| {
                        let _ = view.update(cx, |this, cx| {
                            if this.notification.height != bounds.size.height {
                                this.notification.height = bounds.size.height;
                                cx.notify();
                            }
                        });
                    },
                    |_, _, _, _| (),
                )
                .absolute()
                .size_full(),
            )
            .with_animation(
                (message, usize::from(exiting)),
                Animation::new(Duration::from_millis(400)).with_easing(toast_ease),
                move |toast, delta| {
                    let visibility = if exiting { 1. - delta } else { delta };
                    toast
                        .top(px(if mobile { 16. } else { 24. }) - height * (1. - visibility))
                        .opacity(visibility)
                },
            )
            .into_any_element()
    }
}

fn toast_ease(progress: f32) -> f32 {
    let mut t = progress;
    // CSS ease, Sonner's default 400ms transform/opacity transition.
    for _ in 0..6 {
        t -= (0.75 * t - 0.75 * t * t + t * t * t - progress) / (0.75 - 1.5 * t + 3. * t * t);
    }
    (0.3 * t + 2.4 * t * t - 1.7 * t * t * t).clamp(0., 1.)
}
