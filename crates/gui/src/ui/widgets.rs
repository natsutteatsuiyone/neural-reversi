use std::rc::Rc;

use nsh_base::{ButtonElement, OpenChangeReason, TooltipEvent, button, tooltip};
use nsh_core::prelude::*;

use super::assets::{self, icon};
use super::theme::layout_px;
use super::{ACCENT, Action, FocusEntry, RED, SECONDARY, Ui, color, rgba};

pub(super) fn single_line_text(value: impl IntoReactive<SharedString>, line_height: f32) -> Text {
    let height = computed(move || layout_px(line_height * font_scale()));
    // Native text measurement rounds up to whole points, not device pixels.
    text(value)
        .h(move || height.get())
        .line_height(move || height.get() / font_scale())
}

pub(super) fn focusable(
    ui: Ui,
    id: &str,
    enabled: impl Fn() -> bool + 'static,
    activate: impl Fn() + 'static,
) -> ElementRef {
    let element = element_ref();
    let id = id.to_owned();
    let scope = ui.scope;
    ui.focus_registry.with(|registry| {
        registry.entries.push(FocusEntry {
            id: id.clone(),
            scope,
            element,
            enabled: Rc::new(enabled),
            activate: Rc::new(activate),
        })
    });
    cleanup(move || {
        ui.focus_registry.try_with(|registry| {
            registry
                .entries
                .retain(|entry| entry.id != id || entry.scope != scope)
        });
    });
    element
}

pub(super) fn control(
    ui: Ui,
    id: &str,
    name: &'static str,
    label: impl IntoReactive<SharedString>,
    action: Action,
    enabled: impl Fn() -> bool + 'static,
) -> ButtonElement {
    let label = label.into_reactive();
    let label = computed(move || label.get());
    let solver_action = matches!(action, Action::Solver);
    let title_action = action.clone();
    let title = computed(move || {
        let label = label.get();
        if !label.is_empty() {
            return label;
        }
        match title_action {
            Action::NewGame => tr!("new_game"),
            Action::Solver => tr!("solver"),
            Action::Settings => tr!("settings"),
            Action::Copy => tr!("copy_transcript"),
            Action::Previous => tr!("previous_move"),
            Action::Next => tr!("next_move"),
            Action::Latest => tr!("return_latest"),
            Action::Navigate(_) if name == "chevron-first.svg" => tr!("first_position"),
            Action::Navigate(_) if name == "chevron-left.svg" => tr!("previous_move"),
            Action::Navigate(_) if name == "chevron-last.svg" => tr!("return_latest"),
            Action::Navigate(_) => tr!("next_move"),
            Action::Pause => tr!("stop_and_play"),
            Action::Close => tr!("close"),
            _ => label,
        }
    });
    let enabled = Rc::new(enabled);
    let can_focus = enabled.clone();
    let keyboard_action = action.clone();
    let element = focusable(
        ui,
        id,
        move || can_focus(),
        move || ui.action(keyboard_action.clone()),
    );
    let id = id.to_owned();
    let press_id = id.clone();
    let primary = matches!(action, Action::NewGame | Action::Start);
    let pause = matches!(action, Action::Pause);
    button()
        .element_ref(&element)
        .accessibility_label(move || Some(title.get()))
        .disabled(move || !enabled())
        .h(px(28.))
        .flex_shrink_0()
        .w(move || {
            if label.get().is_empty() {
                px(28.).into()
            } else {
                Length::Auto
            }
        })
        .px(move || px(if label.get().is_empty() { 0. } else { 10. }))
        .gap_1()
        .flex_row()
        .items_center()
        .justify_center()
        .rounded(px(8.))
        .border_1()
        .border_color(rgba(0x00000000))
        .text_size(px(12.8))
        .line_height(nsh_core::LineHeight::Multiple(1.4285714))
        .font_weight(FontWeight::MEDIUM)
        .text_color(move || {
            if primary {
                color(0xffffff)
            } else {
                rgba(0xffffff99)
            }
        })
        .bg(move || {
            if primary {
                color(ACCENT)
            } else {
                rgba(0x00000000)
            }
        })
        .hover(move |s| {
            s.bg(if primary {
                color(0x52b885)
            } else if pause {
                rgba(0xc45c5c1a)
            } else {
                rgba(0xffffff1a)
            })
            .text_color(if pause { color(RED) } else { rgba(0xffffffeb) })
        })
        .disabled_style(|s| s.opacity(0.5))
        .style_when(
            move || ui.is_focus_visible(&id),
            |s| s.border_color(color(ACCENT)),
        )
        .on_press(move || {
            ui.focus(&press_id);
            ui.action(action.clone());
        })
        .child(
            icon(name, 14., SECONDARY)
                .size(move || {
                    px(if pause && !ui.read(|app| app.session.is_solving()) {
                        14.
                    } else if label.get().is_empty() && !primary && !solver_action {
                        16.
                    } else {
                        14.
                    })
                })
                .text_color(move || {
                    if primary {
                        color(0xffffff)
                    } else {
                        rgba(0xffffff99)
                    }
                }),
        )
        .child(when(
            move || !label.get().is_empty(),
            move || single_line_text(label, 12.8 * 20. / 14.),
        ))
        .child(when(
            move || label.get().is_empty(),
            move || control_tooltip(ui, title),
        ))
}

fn control_tooltip(ui: Ui, title: Memo<SharedString>) -> impl IntoElement {
    let open = signal(false);
    let shown_at = signal(point(px(0.), px(0.)));
    let measured = signal(size(px(0.), px(0.)));
    tooltip()
        .open(move || Some(open.get()))
        .on_open_change(move |event: &TooltipEvent| {
            if let TooltipEvent::OpenChange(next, reason) = event {
                if *next && *reason == OpenChangeReason::Focus {
                    return;
                }
                if *next {
                    shown_at.set(ui.mouse_position.get_untracked());
                    measured.set(size(px(0.), px(0.)));
                }
                open.set(*next);
            }
        })
        .child(
            overlay()
                .anchor_point(move || {
                    let at = shown_at.get();
                    let extent = measured.get();
                    let viewport = window();
                    let coordinate = |at: f32, length: f32, limit: f32| {
                        if at + 1. + length > limit {
                            (at - length - 1.).max(0.)
                        } else {
                            at + 1.
                        }
                    };
                    point(
                        px(coordinate(at.x.0, extent.width.0, viewport.width.0)),
                        px(coordinate(at.y.0, extent.height.0, viewport.height.0)),
                    )
                })
                .placement(Placement::Bottom)
                .align(Align::Start)
                .offset(px(0.))
                .margin(px(0.))
                .pointer_events(PointerEvents::None)
                .on_position(move |position: &OverlayPosition| {
                    if measured.get_untracked() != position.bounds.size {
                        measured.set(position.bounds.size);
                    }
                })
                .opacity(move || -> f32 {
                    if measured.get().width.0 > 0. { 1. } else { 0. }
                })
                .child(
                    view()
                        .m(px(12.))
                        .bg(color(0x0a0a0a))
                        .text_color(rgba(0xffffffeb))
                        .border_1()
                        .border_color(rgba(0xffffff0f))
                        .shadow_md()
                        .rounded(px(8.))
                        .py(px(2.))
                        .px_2()
                        .text_size(px(14.))
                        .line_height(px(20.))
                        .child(text(title)),
                ),
        )
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

pub(super) fn flat_stone(black: bool, side: f32) -> impl IntoElement {
    let names = match side {
        16. => ["stones/black-16.png", "stones/white-16.png"],
        21.6 => ["stones/black-21_6.png", "stones/white-21_6.png"],
        40. => ["stones/black-40.png", "stones/white-40.png"],
        _ => unreachable!("unsupported UI stone diameter"),
    };
    view().relative().size(px(side)).flex_shrink_0().child(
        image(assets::image(names[usize::from(!black)]))
            .absolute()
            .left(px(-12.))
            .top(px(-12.))
            .size(px(side + 24.)),
    )
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
