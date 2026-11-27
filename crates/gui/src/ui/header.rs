use std::time::Duration;

use nsh_base::{MenuEvent, MenuItemElement, menu, menu_item, switch};
use nsh_core::prelude::*;

use super::assets::icon;
use super::dialogs::DIALOG_LINE_HEIGHT;
use super::slider::{self, SliderKind};
use super::theme::{layout_px, stroke_px};
use super::widgets::{control, focusable, single_line_text};
use super::{ACCENT, Action, BACKGROUND, GOLD, Modal, PANEL, SECONDARY, Ui, color, rgba};
use crate::settings::Language;

pub(super) fn render_header(ui: Ui) -> impl IntoElement {
    view()
        .h(px(48.))
        .min_h(px(48.))
        .flex_shrink_0()
        .flex_row()
        .items_center()
        .justify_between()
        .px(move || {
            px(if ui.read(|app| app.viewport.width.0) >= 640. {
                16.
            } else {
                12.
            })
        })
        .bg(color(PANEL))
        .border_b_1()
        .border_color(rgba(0xffffff1a))
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(
                    control(
                        ui,
                        "new-game",
                        "play.svg",
                        move || {
                            if ui.read(|app| app.viewport.width.0) <= 420. {
                                "".into()
                            } else {
                                tr!("new_game")
                            }
                        },
                        Action::NewGame,
                        || true,
                    )
                    .gap_2(),
                )
                .child(
                    control(
                        ui,
                        "solver",
                        "calculator.svg",
                        move || {
                            if ui.read(|app| app.viewport.width.0) <= 420. {
                                "".into()
                            } else {
                                tr!("solver")
                            }
                        },
                        Action::Solver,
                        || true,
                    )
                    .gap_2()
                    .bg(color(BACKGROUND))
                    .border_color(rgba(0xffffff1a)),
                ),
        )
        .child({
            let hint = focusable(
                ui,
                "hint-control",
                move || ui.read(|app| !app.session.live_position().game_over()),
                move || ui.action(Action::Hint),
            );
            view()
                .flex_row()
                .items_center()
                .gap_3()
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .px_2()
                        .py_1()
                        .rounded(px(10.))
                        .border(|| stroke_px(1.))
                        .border_color(move || {
                            if ui.read(|app| app.settings.hints) {
                                rgba(0xb8956c4d)
                            } else {
                                rgba(0xffffff1a)
                            }
                        })
                        .bg(move || {
                            if ui.read(|app| app.settings.hints) {
                                rgba(0xb8956c1a)
                            } else {
                                rgba(0xffffff0d)
                            }
                        })
                        .child(icon("lightbulb.svg", 16., GOLD).text_color(move || {
                            color(if ui.read(|app| app.settings.hints) {
                                GOLD
                            } else {
                                SECONDARY
                            })
                        }))
                        .child(
                            switch()
                                .flex_row()
                                .items_center()
                                .gap_2()
                                .element_ref(&hint)
                                .accessibility_label(move || Some(tr!("hint")))
                                .checked(move || ui.read(|app| app.settings.hints))
                                .disabled(move || {
                                    ui.read(|app| app.session.live_position().game_over())
                                })
                                .on_change(move |_: &bool| {
                                    ui.focus("hint-control");
                                    ui.action(Action::Hint);
                                })
                                .track_style(|s| {
                                    s.w(|| layout_px(32.))
                                        .h(|| layout_px(18.4))
                                        .flex_shrink_0()
                                        .flex_row()
                                        .items_center()
                                        .rounded_full()
                                        .border(|| stroke_px(1.))
                                        .border_color(rgba(0))
                                        .bg(rgba(0xffffff0f))
                                        .checked(|s| s.bg(color(ACCENT)))
                                        .style_when(
                                            move || ui.is_focus_visible("hint-control"),
                                            |s| {
                                                s.inset_ring(px(2.)).inset_ring_color(color(ACCENT))
                                            },
                                        )
                                })
                                .thumb_style(|s| {
                                    s.size(px(16.))
                                        .rounded_full()
                                        .bg(color(BACKGROUND))
                                        .checked(|s| s.translate_x(px(14.)))
                                })
                                .child(when(
                                    move || ui.read(|app| app.viewport.width.0) > 520.,
                                    move || {
                                        single_line_text(move || tr!("hint"), DIALOG_LINE_HEIGHT)
                                            .text_size(px(12.))
                                            .font_weight(FontWeight::MEDIUM)
                                            .text_color(rgba(0xffffff99))
                                    },
                                )),
                        ),
                )
                .child(settings_menu(ui))
        })
}

fn settings_menu(ui: Ui) -> impl IntoElement {
    menu()
        .open(move || Some(ui.read(|app| app.modal == Some(Modal::Settings))))
        .align(Align::End)
        .offset(px(4.))
        .popup_style(|s| {
            s.w(px(192.))
                .p_1()
                .rounded(px(10.))
                .bg(color(0x2a2a2f))
                .text_size(px(12.))
                .line_height(px(DIALOG_LINE_HEIGHT))
                .text_color(rgba(0xffffffeb))
                .shadow_lg()
        })
        .on_open_change(move |event: &MenuEvent| {
            if let MenuEvent::OpenChange(open, _) = event {
                if *open {
                    ui.focus("settings");
                    ui.action(Action::Settings);
                } else if ui.read(|app| app.modal == Some(Modal::Settings)) {
                    ui.action(Action::Close);
                }
            }
        })
        .trigger(
            control(ui, "settings", "menu.svg", "", Action::Settings, || true)
                .accessibility_label(move || Some(tr!("settings"))),
        )
        .child(when(
            move || ui.read(|app| app.modal == Some(Modal::Settings)),
            move || settings_contents(ui),
        ))
}

fn settings_contents(ui: Ui) -> impl IntoElement {
    let ui = ui.scoped(Some(Modal::Settings));
    let main = element_ref();
    let rows = [element_ref(), element_ref(), element_ref()];
    let selected = computed(move || ui.read(|app| app.menu_section));
    view()
        .element_ref(&main)
        .relative()
        .on_touch(move |_: &nsh_core::TouchEvent| ui.keyboard_focus.set(false))
        .child(section(ui, "menu-hint", "lightbulb.svg", 0, rows[0]))
        .child(section(ui, "menu-hash", "hard-drive.svg", 1, rows[1]))
        .child(section(ui, "menu-language", "globe.svg", 2, rows[2]))
        .child(view().h(px(1.)).my_1().bg(rgba(0xffffff0f)))
        .child({
            let reference = focusable(ui, "menu-about", || true, move || ui.action(Action::About));
            menu_item()
                .element_ref(&reference)
                .close_on_select(false)
                .h(px(DIALOG_LINE_HEIGHT + 8.))
                .px_1p5()
                .rounded(px(8.))
                .on_select(move || ui.action(Action::About))
                .style_when(
                    move || ui.is_focus_visible("menu-about"),
                    |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
                )
                .hover(|s| s.bg(rgba(0x43a5751f)).text_color(color(0x52b885)))
                .child(
                    view()
                        .on_mouse_enter(move |_: &MouseEvent| hover_section(ui, None))
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .child(icon("info.svg", 16., SECONDARY))
                        .child(text(move || tr!("about"))),
                )
        })
        .child(dynamic(move || {
            let section = selected.get();
            untracked(move || {
                let Some(section) = section else {
                    return view().into_element();
                };
                overlay()
                    .anchor_beside(&rows[section], &main)
                    .placement(Placement::Left)
                    .align(Align::Start)
                    .offset(px(-4.))
                    .child(
                        view()
                            .p_1()
                            .rounded(px(10.))
                            .bg(color(0x2a2a2f))
                            .shadow_lg()
                            .text_size(px(12.))
                            .line_height(px(DIALOG_LINE_HEIGHT))
                            .text_color(rgba(0xffffffeb))
                            .on_touch(move |_: &nsh_core::TouchEvent| ui.keyboard_focus.set(false))
                            .on_mouse_enter(move |_: &MouseEvent| {
                                ui.app.with(|app| app.menu_hover_generation += 1)
                            })
                            .child(match section {
                                0 => slider::render_level_slider(ui, SliderKind::HintLevel)
                                    .into_element(),
                                1 => view()
                                    .w(px(88.))
                                    .child([64, 128, 256, 512, 1024, 2048, 4096].into_iter().map(
                                        move |size| {
                                            choice(
                                                ui,
                                                &format!("hash-size-{size}"),
                                                format!("{size} MB"),
                                                Action::Hash(size),
                                                move || {
                                                    ui.read(|app| app.settings.hash_size == size)
                                                },
                                            )
                                        },
                                    ))
                                    .into_element(),
                                _ => view()
                                    .w(px(88.))
                                    .child([
                                        choice(
                                            ui,
                                            "menu-language-auto",
                                            move || tr!("language_auto"),
                                            Action::Language(Language::Auto),
                                            move || {
                                                ui.read(|app| {
                                                    app.settings.language == Language::Auto
                                                })
                                            },
                                        )
                                        .into_element(),
                                        choice(
                                            ui,
                                            "menu-language-english",
                                            "English",
                                            Action::Language(Language::English),
                                            move || {
                                                ui.read(|app| {
                                                    app.settings.language == Language::English
                                                })
                                            },
                                        )
                                        .into_element(),
                                        choice(
                                            ui,
                                            "menu-language-japanese",
                                            "日本語",
                                            Action::Language(Language::Japanese),
                                            move || {
                                                ui.read(|app| {
                                                    app.settings.language == Language::Japanese
                                                })
                                            },
                                        )
                                        .into_element(),
                                    ])
                                    .into_element(),
                            }),
                    )
                    .into_element()
            })
        }))
}

fn section(
    ui: Ui,
    id: &str,
    image: &'static str,
    section: usize,
    row: ElementRef,
) -> MenuItemElement {
    let reference = focusable(
        ui,
        id,
        || true,
        move || ui.action(Action::MenuSection(Some(section))),
    );
    let press_id = id.to_owned();
    let ring_id = press_id.clone();
    menu_item()
        .element_ref(&reference)
        .close_on_select(false)
        .h(px(DIALOG_LINE_HEIGHT + 8.))
        .px_1p5()
        .rounded(px(8.))
        .on_select(move || {
            ui.focus(&press_id);
            ui.action(Action::MenuSection(Some(section)));
        })
        .style_when(
            move || ui.is_focus_visible(&ring_id),
            |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
        )
        .style_when(
            move || ui.read(|app| app.menu_section == Some(section)),
            |s| s.bg(rgba(0x43a5751f)).text_color(color(0x52b885)),
        )
        .hover(|s| s.bg(rgba(0x43a5751f)).text_color(color(0x52b885)))
        .child(
            view()
                .element_ref(&row)
                .on_mouse_enter(move |_: &MouseEvent| hover_section(ui, Some(section)))
                .flex_row()
                .items_center()
                .gap_2()
                .child(icon(image, 16., SECONDARY))
                .child(
                    text(move || match section {
                        0 => tr!("hint_level"),
                        1 => tr!("hash_size"),
                        _ => tr!("language"),
                    })
                    .flex_1(),
                )
                .child(icon("chevron-right.svg", 16., SECONDARY)),
        )
}

fn choice(
    ui: Ui,
    id: &str,
    label: impl IntoReactive<SharedString>,
    action: Action,
    selected: impl Fn() -> bool + 'static,
) -> MenuItemElement {
    let keyboard = action.clone();
    let reference = focusable(
        ui,
        id,
        || true,
        move || {
            ui.action(Action::Close);
            ui.action(keyboard.clone());
        },
    );
    let ring_id = id.to_owned();
    menu_item()
        .element_ref(&reference)
        .h(px(DIALOG_LINE_HEIGHT + 8.))
        .relative()
        .flex_row()
        .items_center()
        .pl(px(6.))
        .pr(px(32.))
        .rounded(px(8.))
        .checked(move || Some(selected()))
        .indicator(icon("check.svg", 16., SECONDARY))
        .indicator_style(|s| {
            s.absolute()
                .right(px(8.))
                .top(px((DIALOG_LINE_HEIGHT + 8. - 16.) * 0.5))
                .size(px(16.))
                .opacity(0.)
                .checked(|s| s.opacity(1.))
        })
        .on_select(move || {
            ui.action(Action::Close);
            ui.action(action.clone());
        })
        .hover(|s| s.bg(rgba(0x43a5751f)).text_color(color(0x52b885)))
        .style_when(
            move || ui.is_focus_visible(&ring_id),
            |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
        )
        .child(text(label))
}

fn hover_section(ui: Ui, section: Option<usize>) {
    let generation = ui.update(|app| {
        app.menu_hover_generation += 1;
        if app.menu_section.is_none() || app.menu_section == section {
            app.menu_section = section;
            return None;
        }
        Some(app.menu_hover_generation)
    });
    if let Some(generation) = generation {
        set_timeout(Duration::from_millis(300), move || {
            if ui.app.with(|app| {
                app.menu_hover_generation == generation && app.modal == Some(Modal::Settings)
            }) {
                ui.update(|app| app.menu_section = section);
            }
        });
    }
}

pub(super) fn focused_key(ui: Ui, key: &str) -> bool {
    if !ui.read(|app| app.modal == Some(Modal::Settings)) {
        return false;
    }
    const SECTIONS: [&str; 3] = ["menu-hint", "menu-hash", "menu-language"];
    match key {
        "up" => ui.focus_step(true),
        "down" => ui.focus_step(false),
        "left" => {
            let section = ui.read(|app| app.menu_section);
            ui.action(Action::MenuSection(None));
            if let Some(id) = section.and_then(|index| SECTIONS.get(index)) {
                ui.focus(id);
            }
        }
        "right" => {
            const FIRST: [&str; 3] = ["slider-hint-level", "hash-size-64", "menu-language-auto"];
            for (index, id) in SECTIONS.into_iter().enumerate() {
                if ui.is_focused(id) {
                    ui.action(Action::MenuSection(Some(index)));
                    let target = FIRST[index];
                    set_timeout(Duration::from_millis(1), move || {
                        if ui.read(|app| {
                            app.modal == Some(Modal::Settings) && app.menu_section == Some(index)
                        }) {
                            ui.focus(target);
                        }
                    });
                    break;
                }
            }
        }
        _ => return false,
    }
    true
}
