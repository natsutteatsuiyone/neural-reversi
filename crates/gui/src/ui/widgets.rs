use std::time::Duration;

use gpui_kit::component::{Icon, tooltip::Tooltip};
use gpui_kit::{prelude::*, *};
use reversi_core::disc::Disc;

use super::assets::icon;
use super::{ACCENT, Action, BLUE, RED, ReversiApp, SECONDARY};
use crate::session::Activity;

impl ReversiApp {
    pub(super) fn control(
        &self,
        id: impl Into<ElementId>,
        name: &'static str,
        label: impl Into<SharedString>,
        action: Action,
        enabled: bool,
        cx: &Context<Self>,
    ) -> Stateful<Div> {
        let label = label.into();
        let id = id.into();
        let focus = self
            .control_focus
            .borrow_mut()
            .entry(id.clone())
            .or_insert_with(|| cx.focus_handle())
            .clone()
            .tab_stop(enabled && (self.modal.is_none() || matches!(action, Action::Close)));
        let title = if !label.is_empty() {
            label.clone()
        } else {
            match action {
                Action::NewGame => tr!("new_game"),
                Action::Solver => tr!("solver"),
                Action::Settings => tr!("settings"),
                Action::Copy => tr!("copy_transcript"),
                Action::Navigate(_) if name == "chevron-first.svg" => tr!("first_position"),
                Action::Navigate(_) if name == "chevron-left.svg" => {
                    tr!("previous_move")
                }
                Action::Navigate(_) if name == "chevron-last.svg" => tr!("return_latest"),
                Action::Navigate(_) => tr!("next_move"),
                Action::Pause => tr!("stop_and_play"),
                Action::Close => tr!("close"),
                _ => label.clone(),
            }
        };
        let primary = matches!(action, Action::NewGame);
        let destructive = matches!(action, Action::Pause)
            && matches!(
                self.session.activity(),
                Some(Activity::Move | Activity::Solver)
            );
        let resume = matches!(action, Action::Pause) && !self.session.is_solving() && !destructive;
        let solver = self.session.is_solving();
        let icon_size = if matches!(action, Action::Pause) && !self.session.is_solving() {
            14.
        } else if label.is_empty() && !matches!(action, Action::NewGame | Action::Solver) {
            16.
        } else {
            14.
        };
        interactive(div().id(id), title.clone(), enabled)
            .track_focus(&focus)
            .relative()
            .h_7()
            .flex_shrink_0()
            .px(px(if label.is_empty() { 0. } else { 10. }))
            .when(label.is_empty(), move |button| {
                button
                    .w_7()
                    .tooltip(move |window, cx| Tooltip::new(title.clone()).build(window, cx))
            })
            .flex()
            .items_center()
            .justify_center()
            .gap_1()
            .rounded(px(8.))
            .border_1()
            .border_color(gpui_kit::transparent_black())
            .text_size(px(12.8))
            .line_height(relative(1.4285714))
            .font_weight(FontWeight::MEDIUM)
            .text_color(rgba(0xffffff99))
            .when(!enabled, |button| button.opacity(0.5))
            .when(enabled, |button| {
                button
                    .cursor_pointer()
                    .hover(move |style| {
                        style
                            .bg(if primary {
                                rgb(0x52b885)
                            } else if destructive {
                                if solver {
                                    rgba(0xc45c5c4d)
                                } else {
                                    rgba(0xc45c5c1a)
                                }
                            } else if resume {
                                rgba(0x43a5751a)
                            } else {
                                rgba(0xffffff1a)
                            })
                            .text_color(if destructive {
                                rgb(RED)
                            } else if resume {
                                rgb(ACCENT)
                            } else {
                                rgba(0xffffffeb)
                            })
                    })
                    .on_click(
                        cx.listener(move |this, _, window, cx| this.action(action, window, cx)),
                    )
            })
            .child(
                Icon::empty()
                    .path(name)
                    .size(px(icon_size))
                    .when(name == "check.svg", |icon| icon.text_color(rgb(ACCENT))),
            )
            .when(!label.is_empty(), |button| button.child(label))
    }
}

/// Adds keyboard and assistive-technology behavior to the existing controls.
pub(super) fn interactive(
    element: Stateful<Div>,
    label: impl Into<SharedString>,
    enabled: bool,
) -> Stateful<Div> {
    element
        .role(Role::Button)
        .aria_label(label)
        .when(!enabled, |element| {
            element.a11y_synthetic_children(|tree| tree.parent_node().set_disabled())
        })
        .when(enabled, |element| {
            element.tab_index(0).focus_visible(|style| {
                style
                    .border_color(rgb(ACCENT))
                    .shadow(vec![shadow(rgb(ACCENT), 0., 0., 2.).inset()])
            })
        })
}

/// Arrow keys on item `index` of a `count`-item radio group or tab list
/// dispatch `select(app, neighbor)` and move focus to that neighbor.
pub(super) fn arrow_keys(
    index: usize,
    count: usize,
    select: impl Fn(&ReversiApp, usize) -> Action + 'static,
    cx: &Context<ReversiApp>,
) -> impl Fn(&KeyDownEvent, &mut Window, &mut App) + 'static {
    cx.listener(move |this, event: &KeyDownEvent, window, cx| {
        let forward = match event.keystroke.key.as_str() {
            "right" | "down" => true,
            "left" | "up" => false,
            _ => return,
        };
        cx.stop_propagation();
        if (forward && index + 1 == count) || (!forward && index == 0) {
            return;
        }
        let next = select(this, if forward { index + 1 } else { index - 1 });
        this.action(next, window, cx);
        if forward {
            window.focus_next(cx);
        } else {
            window.focus_prev(cx);
        }
    })
}

pub(super) fn move_board_cursor(index: usize, key: &str) -> Option<usize> {
    let row = index / 8;
    let column = index % 8;
    Some(match key {
        "left" => row * 8 + column.saturating_sub(1),
        "right" => row * 8 + (column + 1).min(7),
        "up" => row.saturating_sub(1) * 8 + column,
        "down" => (row + 1).min(7) * 8 + column,
        "home" => row * 8,
        "end" => row * 8 + 7,
        _ => return None,
    })
}

/// An outer shadow offset `y` pixels downward.
pub(super) fn shadow(color: impl Into<Hsla>, y: f32, blur: f32, spread: f32) -> BoxShadow {
    BoxShadow {
        inset: false,
        color: color.into(),
        offset: point(px(0.), px(y)),
        blur_radius: px(blur),
        spread_radius: px(spread),
    }
}

/// Flat stone used outside the main board; `diameter` must be 16, 21.6, or 40.
pub(super) fn flat_stone(disc: Disc, diameter: f32) -> Div {
    // Baked from Stone.tsx and the original App.css at DPR 3. Each size
    // retains the CSS's fixed-pixel inset shadows and 12px transparent padding.
    let images = match diameter {
        16. => ["stones/black-16.png", "stones/white-16.png"],
        21.6 => ["stones/black-21_6.png", "stones/white-21_6.png"],
        40. => ["stones/black-40.png", "stones/white-40.png"],
        _ => unreachable!("unsupported UI stone diameter"),
    };
    div().relative().size(px(diameter)).flex_shrink_0().child(
        img(images[usize::from(disc != Disc::Black)])
            .absolute()
            .left(px(-12.))
            .top(px(-12.))
            .size(px(diameter + 24.)),
    )
}

pub(super) fn activity_icon(id: &'static str, active: bool) -> AnyElement {
    if !active {
        return icon("activity.svg", 16., SECONDARY).into_any_element();
    }
    icon("activity.svg", 16., BLUE)
        .with_animation(
            id,
            Animation::new(Duration::from_secs(2)).repeat(),
            |icon, delta| {
                let phase = if delta <= 0.5 {
                    delta * 2.
                } else {
                    (1. - delta) * 2.
                };
                let mut t = phase;
                // Tailwind animate-pulse: cubic-bezier(.4, 0, .6, 1), opacity 1 → .5 → 1.
                for _ in 0..4 {
                    t -= (1.2 * t - 0.6 * t * t + 0.4 * t * t * t - phase)
                        / (1.2 - 1.2 * t + 1.2 * t * t);
                }
                icon.opacity(1. - 0.5 * (3. * t * t - 2. * t * t * t))
            },
        )
        .into_any_element()
}

#[cfg(test)]
mod tests {
    use super::move_board_cursor;
    #[test]
    fn board_navigation_stays_inside_rows_and_board() {
        assert_eq!(move_board_cursor(7, "right"), Some(7));
        assert_eq!(move_board_cursor(8, "left"), Some(8));
        assert_eq!(move_board_cursor(0, "up"), Some(0));
        assert_eq!(move_board_cursor(63, "down"), Some(63));
        assert_eq!(move_board_cursor(19, "down"), Some(27));
        assert_eq!(move_board_cursor(19, "enter"), None);
    }
}
