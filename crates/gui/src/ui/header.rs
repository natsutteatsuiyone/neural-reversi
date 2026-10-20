use std::time::Duration;

use gpui_kit::base::FocusTrapElement;
use gpui_kit::component::Icon;
use gpui_kit::{prelude::*, *};

use super::assets::icon;
use super::dialogs::{DIALOG_LINE_HEIGHT, dialog_small_shadow};
use super::widgets::{interactive, shadow};
use super::{ACCENT, Action, BACKGROUND, GOLD, LEVEL_CHOICES, Modal, PANEL, ReversiApp, SECONDARY};
use crate::settings::Language;

/// Transposition table sizes offered by the settings menu, in MiB.
const HASH_SIZES: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

/// How long the pointer must rest on another menu row before its submenu
/// replaces the open one, so a diagonal move into the submenu can cross rows.
const MENU_HOVER_DELAY: Duration = Duration::from_millis(300);

impl ReversiApp {
    pub(super) fn render_header(&self, width: f32, cx: &Context<Self>) -> Div {
        let hint_enabled = !self.session.live_position().game_over();
        let compact = width <= 420.;
        div()
            .min_h_12()
            .h_12()
            .flex_shrink_0()
            .flex()
            .items_center()
            .justify_between()
            .px(px(if width >= 640. { 16. } else { 12. }))
            .bg(rgb(PANEL))
            .border_b_1()
            .border_color(rgba(0xffffff1a))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(if width >= 640. { 8. } else { 6. }))
                    .child(
                        self.control(
                            "new-game",
                            "play.svg",
                            if compact { "".into() } else { tr!("new_game") },
                            Action::NewGame,
                            true,
                            cx,
                        )
                        .gap_2()
                        .when(compact, |button| button.w(px(32.)).px_2())
                        .bg(rgb(ACCENT))
                        .text_color(rgb(0xffffff))
                        .shadow(dialog_small_shadow()),
                    )
                    .child(
                        self.control(
                            "solver",
                            "calculator.svg",
                            if compact { "".into() } else { tr!("solver") },
                            Action::Solver,
                            true,
                            cx,
                        )
                        .gap_2()
                        .when(compact, |button| button.w(px(32.)).px_2())
                        .border_color(rgba(0xffffff1a))
                        .bg(rgb(BACKGROUND)),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(if width >= 640. { 12. } else { 6. }))
                    .child(
                        div()
                            .id("hint-switch")
                            .flex()
                            .items_center()
                            .gap_2()
                            .px_2()
                            .py_1()
                            .rounded(px(10.))
                            .border_1()
                            .border_color(if self.settings.hints {
                                rgba(0xb8956c4d)
                            } else {
                                rgba(0xffffff1a)
                            })
                            .bg(if self.settings.hints {
                                rgba(0xb8956c1a)
                            } else {
                                rgba(0xffffff0d)
                            })
                            .child(icon(
                                "lightbulb.svg",
                                16.,
                                if self.settings.hints { GOLD } else { SECONDARY },
                            ))
                            .child(
                                interactive(div().id("hint-control"), tr!("hint"), hint_enabled)
                                    .role(Role::Switch)
                                    .aria_toggled(self.settings.hints.into())
                                    .when(hint_enabled, |switch| {
                                        switch.cursor_pointer().on_click(cx.listener(
                                            |this, _, window, cx| {
                                                this.action(Action::Hint, window, cx)
                                            },
                                        ))
                                    })
                                    .w(px(32.))
                                    .h(px(18.4))
                                    .flex_shrink_0()
                                    .flex()
                                    .items_center()
                                    .rounded_full()
                                    .border_1()
                                    .border_color(gpui_kit::transparent_black())
                                    .bg(if self.settings.hints {
                                        rgb(ACCENT)
                                    } else {
                                        rgba(0xffffff0f)
                                    })
                                    .when(!hint_enabled, |switch| switch.opacity(0.5))
                                    .child(
                                        div()
                                            .size_4()
                                            .rounded_full()
                                            .bg(rgb(BACKGROUND))
                                            .when(self.settings.hints, |thumb| thumb.ml(px(14.))),
                                    ),
                            )
                            .when(width > 520., |control| {
                                control.child(
                                    div()
                                        .id("hint-label")
                                        .cursor_pointer()
                                        .when(hint_enabled, |label| {
                                            label.on_click(cx.listener(|this, _, window, cx| {
                                                this.action(Action::Hint, window, cx)
                                            }))
                                        })
                                        .text_size(px(12.))
                                        .line_height(relative(1.4285714))
                                        .font_weight(FontWeight::MEDIUM)
                                        .text_color(rgba(0xffffff99))
                                        .child(tr!("hint")),
                                )
                            }),
                    )
                    .child(self.control("settings", "menu.svg", "", Action::Settings, true, cx)),
            )
    }

    pub(super) fn render_header_menu(&self, cx: &Context<Self>) -> Div {
        let menu_row =
            |id: &'static str, label: SharedString, image: &'static str, section: usize| {
                menu_item(id, label.clone())
                    .when(section == 0, |row| row.track_focus(&self.modal_entry_focus))
                    .aria_expanded(self.menu_section == Some(section))
                    .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                        match event.keystroke.key.as_str() {
                            "right" | "left" => {
                                this.action(
                                    Action::MenuSection(if event.keystroke.key == "right" {
                                        Some(section)
                                    } else {
                                        None
                                    }),
                                    window,
                                    cx,
                                );
                                cx.stop_propagation();
                            }
                            _ => {}
                        }
                    }))
                    .px_1p5()
                    .when(self.menu_section == Some(section), |row| {
                        row.bg(rgba(0x43a5751f)).text_color(rgb(0x52b885))
                    })
                    .on_hover(cx.listener(move |this, hovered, _, cx| {
                        if *hovered {
                            this.hover_menu_section(Some(section), cx);
                        }
                    }))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.action(Action::MenuSection(Some(section)), window, cx)
                    }))
                    .child(
                        Icon::empty()
                            .path(image)
                            .size(px(16.))
                            .mr_2()
                            .flex_shrink_0(),
                    )
                    .child(div().flex_1().child(label))
                    .child(
                        Icon::empty()
                            .path("chevron-right.svg")
                            .size(px(16.))
                            .flex_shrink_0(),
                    )
            };
        let mut menu = div()
            .id("settings-menu")
            .role(Role::Menu)
            .aria_label(tr!("settings"))
            .track_focus(&self.modal_focus)
            .tab_group()
            .on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                let key = event.keystroke.key.as_str();
                if !matches!(key, "up" | "down") {
                    return;
                }
                let before = window.focused(cx);
                loop {
                    if key == "up" {
                        window.focus_prev(cx);
                    } else {
                        window.focus_next(cx);
                    }
                    if this.modal_focus.contains_focused(window, cx) || window.focused(cx) == before
                    {
                        break;
                    }
                }
                cx.stop_propagation();
            }))
            .absolute()
            .top(px(42.))
            .right(px(16.))
            .w(px(192.))
            .p_1()
            .rounded(px(10.))
            .bg(rgb(0x2a2a2f))
            .text_size(px(12.))
            .line_height(px(DIALOG_LINE_HEIGHT))
            .text_color(rgba(0xffffffeb))
            .shadow(menu_shadow(false))
            .on_any_mouse_down(|_, _, cx| cx.stop_propagation())
            .child(menu_row("menu-hint", tr!("hint_level"), "lightbulb.svg", 0))
            .child(menu_row("menu-hash", tr!("hash_size"), "hard-drive.svg", 1))
            .child(menu_row("menu-language", tr!("language"), "globe.svg", 2))
            .child(div().h(px(1.)).mx(px(-4.)).my_1().bg(rgba(0xffffff0f)))
            .child(
                menu_item("menu-about", tr!("about"))
                    .px_1p5()
                    .on_hover(cx.listener(|this, hovered, _, cx| {
                        if *hovered {
                            this.hover_menu_section(None, cx);
                        }
                    }))
                    .on_click(
                        cx.listener(|this, _, window, cx| this.action(Action::About, window, cx)),
                    )
                    .child(
                        Icon::empty()
                            .path("info.svg")
                            .size(px(16.))
                            .mr_2()
                            .flex_shrink_0(),
                    )
                    .child(tr!("about")),
            );
        if let Some(section) = self.menu_section {
            let mut submenu = div()
                .id("menu-submenu")
                // Reaching the submenu cancels a switch started by rows crossed on the way.
                .on_hover(cx.listener(|this, hovered, _, _| {
                    if *hovered {
                        this.menu_hover_generation += 1;
                    }
                }))
                // The submenu lies outside the menu's bounds, so the menu's own
                // mouse-down guard does not cover it; without this, pressing an
                // item closes the menu before the click completes.
                .on_any_mouse_down(|_, _, cx| cx.stop_propagation())
                .absolute()
                .right(px(188.))
                .top(px(1. + section as f32 * (DIALOG_LINE_HEIGHT + 8.)))
                .w(px(96.))
                .p_1()
                .rounded(px(10.))
                .bg(rgb(0x2a2a2f))
                .shadow(menu_shadow(true));
            match section {
                0 => {
                    submenu = submenu.children(LEVEL_CHOICES.into_iter().map(|level| {
                        menu_choice(
                            ("hint-level", level),
                            format!("Lv.{level}"),
                            Action::HintLevel(level),
                            self.settings.hint_level == level,
                            cx,
                        )
                    }));
                }
                1 => {
                    submenu = submenu.children(HASH_SIZES.into_iter().map(|size| {
                        menu_choice(
                            ("hash-size", size),
                            format!("{size} MB"),
                            Action::Hash(size),
                            self.settings.hash_size == size,
                            cx,
                        )
                    }));
                }
                2 => {
                    submenu = submenu.children(
                        [
                            (Language::Auto, tr!("language_auto")),
                            (Language::English, "English".into()),
                            (Language::Japanese, "日本語".into()),
                        ]
                        .into_iter()
                        .enumerate()
                        .map(|(index, (language, label))| {
                            menu_choice(
                                ("menu-language-choice", index),
                                label,
                                Action::Language(language),
                                self.settings.language == language,
                                cx,
                            )
                        }),
                    );
                }
                _ => unreachable!(),
            }
            menu = menu.child(submenu);
        }
        div()
            .absolute()
            .inset_0()
            .size_full()
            .occlude()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, window, cx| this.action(Action::Close, window, cx)),
            )
            .on_mouse_up(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .on_scroll_wheel(|_, _, cx| cx.stop_propagation())
            .child(menu.focus_trap("settings-focus-trap", &self.modal_focus))
    }
}

fn menu_choice(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    action: Action,
    selected: bool,
    cx: &Context<ReversiApp>,
) -> Stateful<Div> {
    let label = label.into();
    menu_item(id, label.clone())
        .aria_selected(selected)
        .relative()
        .pl(px(6.))
        .pr(px(32.))
        .text_color(rgba(0xffffffeb))
        .on_click(cx.listener(move |this, _, window, cx| {
            this.action(action, window, cx);
            this.action(Action::Close, window, cx);
        }))
        .child(label)
        .when(selected, |item| {
            item.child(
                Icon::empty()
                    .path("check.svg")
                    .absolute()
                    .right(px(8.))
                    .size(px(16.)),
            )
        })
}

fn menu_item(id: impl Into<ElementId>, label: SharedString) -> Stateful<Div> {
    interactive(div().id(id), label, true)
        .role(Role::MenuItem)
        .h(px(DIALOG_LINE_HEIGHT + 8.))
        .flex()
        .items_center()
        .gap_1p5()
        .rounded(px(8.))
        .cursor_pointer()
        .hover(|style| style.bg(rgba(0x43a5751f)).text_color(rgb(0x52b885)))
}

fn menu_shadow(large: bool) -> Vec<BoxShadow> {
    vec![
        shadow(rgba(0xffffff17), 0., 0., 1.),
        if large {
            shadow(rgba(0x0000008c), 8., 16., -4.)
        } else {
            shadow(rgba(0x00000080), 4., 8., -2.)
        },
        shadow(
            rgba(if large { 0xffffff14 } else { 0xffffff0f }),
            0.,
            0.,
            1.,
        ),
    ]
}

impl ReversiApp {
    /// Opens the hovered row's submenu, immediately when none is open and after
    /// [`MENU_HOVER_DELAY`] when it would replace the open one.
    fn hover_menu_section(&mut self, section: Option<usize>, cx: &mut Context<Self>) {
        self.menu_hover_generation += 1;
        if self.menu_section.is_none() || self.menu_section == section {
            self.menu_section = section;
            cx.notify();
            return;
        }
        let generation = self.menu_hover_generation;
        cx.spawn(async move |view, cx| {
            cx.background_executor().timer(MENU_HOVER_DELAY).await;
            let _ = view.update(cx, |view, cx| {
                if view.menu_hover_generation == generation && view.modal == Some(Modal::Settings) {
                    view.menu_section = section;
                    cx.notify();
                }
            });
        })
        .detach();
    }
}
