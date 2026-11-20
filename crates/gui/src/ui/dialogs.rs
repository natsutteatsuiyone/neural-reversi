use nsh_base::{ButtonElement, TabElement, TabsElement, button, dialog, tab, tabs};
use nsh_core::prelude::*;

use super::assets::icon;
use super::setup::SetupTab;
use super::theme::stroke_px;
use super::widgets::{control, focusable, single_line_text};
use super::{ACCENT, Action, BACKGROUND, BLUE, Modal, PANEL, RED, Ui, color, rgba};
use crate::settings::GameMode;

pub(super) const DIALOG_LINE_HEIGHT: f32 = 12. * 20. / 14.;

pub(super) fn render_modal(ui: Ui) -> impl IntoElement {
    let current = computed(move || ui.read(|a| a.modal));
    dynamic(move || {
        let modal = current.get();
        untracked(move || match modal {
            Some(modal @ (Modal::NewGame | Modal::Setup | Modal::SolverSetup | Modal::About)) => {
                modal_card(ui, modal).into_element()
            }
            _ => view().into_element(),
        })
    })
}

fn modal_title(modal: Modal) -> SharedString {
    match modal {
        Modal::NewGame => tr!("new_game"),
        Modal::Setup => tr!("board_setup"),
        Modal::SolverSetup => tr!("solver"),
        Modal::About => "Neural Reversi".into(),
        Modal::Settings => tr!("settings"),
    }
}

fn modal_card(ui: Ui, modal: Modal) -> impl IntoElement {
    let ui = ui.scoped(Some(modal));
    let body = match modal {
        Modal::NewGame => super::new_game::render_new_game_dialog(ui),
        Modal::Setup | Modal::SolverSetup => super::setup::render_setup_dialog(ui),
        Modal::About => super::about::render_about_dialog(ui),
        Modal::Settings => view(),
    };
    let preferred: f32 = match modal {
        Modal::NewGame => 448.,
        Modal::About => 672.,
        _ => 512.,
    };
    let header = view()
        .flex_row()
        .items_baseline()
        .gap_2()
        .child((modal == Modal::About).then(|| icon("info.svg", 20., BLUE)))
        .child(
            single_line_text(move || modal_title(modal), 25.2)
                .text_size(px(18.))
                .font_weight(FontWeight::MEDIUM),
        )
        .child((modal == Modal::About).then(|| {
            single_line_text(format!("v{}", env!("CARGO_PKG_VERSION")), 19.5)
                .text_size(px(13.))
                .text_color(rgba(0xffffff99))
        }));
    let mut card = view()
        .relative()
        .flex_shrink_0()
        .overflow_hidden()
        .rounded(px(14.))
        .w(move || {
            ui.read(|a| {
                let width = a.viewport.width.0;
                px(if width < 640. {
                    (width - 32.).max(0.)
                } else {
                    preferred.min(width)
                })
            })
        })
        .max_h(move || ui.read(|a| px((a.viewport.height.0 - 32.).max(0.))))
        .bg(color(PANEL))
        .shadow([BoxShadow {
            offset_x: px(0.),
            offset_y: px(0.),
            blur: px(0.),
            spread: px(1.),
            color: rgba(0xffffff17).into(),
            inset: false,
        }])
        .text_color(rgba(0xffffffeb))
        .text_size(px(12.))
        .line_height(px(DIALOG_LINE_HEIGHT))
        .child(view().px_4().pt_4().flex_shrink_0().child(header))
        .child(
            scroll()
                .w_full()
                .min_w_0()
                .min_h_0()
                .flex_shrink(1.)
                .child(view().px_4().py_4().child(body)),
        );
    if modal == Modal::About {
        card = card.child(
            control(
                ui,
                "modal-close",
                "x.svg",
                SharedString::default(),
                Action::Close,
                || true,
            )
            .absolute()
            .top(px(8.))
            .right(px(8.)),
        );
    } else {
        let mut footer = view()
            .flex_row()
            .items_center()
            .gap_2()
            .p_4()
            .flex_shrink_0()
            .border_t(|| stroke_px(1.))
            .border_color(rgba(0xffffff0f))
            .bg(rgba(0xffffff05))
            .child(dialog_button(
                ui,
                "modal-cancel",
                move || tr!("cancel"),
                None,
                Some(Action::Close),
                0,
            ))
            .child(view().flex_1());
        if modal == Modal::NewGame {
            footer = footer.child(
                dialog_button(
                    ui,
                    "new-position",
                    move || tr!("board_setup"),
                    None,
                    Some(Action::Setup),
                    1,
                )
                .gap_1()
                .child(icon("chevron-right.svg", 16., 0xffffff)),
            );
        } else if modal == Modal::Setup {
            footer = footer.child(
                dialog_button(
                    ui,
                    "setup-back",
                    move || tr!("back"),
                    Some("chevron-left.svg"),
                    Some(Action::Back),
                    0,
                )
                .gap_1(),
            );
        }
        footer = footer.child(dialog_button(
            ui,
            "modal-start",
            move || {
                if modal == Modal::SolverSetup {
                    tr!("start_solver")
                } else {
                    tr!("start_game")
                }
            },
            Some("play.svg"),
            Some(Action::Start),
            2,
        ));
        card = card.child(footer);
    }
    dialog()
        .label(move || modal_title(modal))
        .animation(ModalAnimation::None)
        .on_cancel(move || ui.action(Action::Close))
        .child(
            view()
                .flex_1()
                .items_center()
                .justify_center()
                .bg(rgba(0x00000080))
                .style_when(
                    move || modal == Modal::NewGame,
                    |style| {
                        style
                            .pr(move || px(1. / pixel_ratio()))
                            .pb(move || px(1. / pixel_ratio()))
                    },
                )
                .on_touch(move |_: &nsh_core::TouchEvent| ui.keyboard_focus.set(false))
                .on_mouse_move(move |event: &MouseEvent| ui.mouse_position.set(event.page_position))
                .child(
                    view()
                        .absolute()
                        .inset_0()
                        .on_press(move || ui.action(Action::Close)),
                )
                .child(card),
        )
}

pub(super) fn dialog_tab(
    ui: Ui,
    id: &'static str,
    label: impl Fn() -> SharedString + 'static,
    image: Option<&'static str>,
    action: Action,
    selected: impl Fn() -> bool + 'static,
) -> TabElement {
    let label = computed(label);
    let selected = computed(selected);
    let press = action.clone();
    let reference = focusable(
        ui,
        id,
        || true,
        move || {
            if !selected.get_untracked() {
                ui.action(press.clone());
            }
        },
    );
    tab()
        .element_ref(&reference)
        .accessibility_label(move || Some(label.get()))
        .selected(selected)
        .group()
        .on_press(move || {
            ui.focus(id);
            if !selected.get_untracked() {
                ui.action(action.clone());
            }
        })
        .flex_1()
        .min_w_0()
        .w(px(0.))
        .h_full()
        .flex_row()
        .items_center()
        .justify_center()
        .gap_2()
        .px(px(6.))
        .rounded(px(8.))
        .border_1()
        .border_color(rgba(0))
        .text_size(px(12.))
        .font_weight(FontWeight::MEDIUM)
        .line_height(px(DIALOG_LINE_HEIGHT))
        .text_color(move || {
            rgba(if selected.get() {
                0xffffffeb
            } else {
                0xffffff8d
            })
        })
        .bg(move || {
            if selected.get() {
                color(BACKGROUND)
            } else {
                rgba(0)
            }
        })
        .hover(|s| s.text_color(rgba(0xffffffeb)))
        .style_when(
            move || ui.is_focus_visible(id),
            |s| {
                s.border_color(color(ACCENT))
                    .inset_ring(px(2.))
                    .inset_ring_color(color(ACCENT))
            },
        )
        .child(image.map(move |name| {
            icon(name, 16., 0xffffff)
                .text_color(move || {
                    rgba(if selected.get() {
                        0xffffffeb
                    } else {
                        0xffffff8d
                    })
                })
                .group_hover(|s| s.text_color(rgba(0xffffffeb)))
        }))
        .child(single_line_text(label, DIALOG_LINE_HEIGHT))
}

pub(super) fn dialog_button(
    ui: Ui,
    id: &'static str,
    label: impl Fn() -> SharedString + 'static,
    image: Option<&'static str>,
    action: Option<Action>,
    variant: u8,
) -> ButtonElement {
    let label = computed(label);
    let action = stored(action);
    let enabled = computed(move || {
        action.with(|action| match action {
            None => false,
            Some(Action::Start) => {
                ui.read(|a| a.modal == Some(Modal::NewGame) || a.setup.error.is_none())
            }
            _ => true,
        })
    });
    let activate = move || {
        if enabled.get_untracked()
            && let Some(action) = action.get()
        {
            ui.focus(id);
            ui.action(action);
        }
    };
    let reference = focusable(ui, id, move || enabled.get(), activate);
    button()
        .element_ref(&reference)
        .accessibility_label(move || Some(label.get()))
        .disabled(move || !enabled.get())
        .group()
        .on_press(activate)
        .h(px(32.))
        .px(px(10.))
        .flex_row()
        .items_center()
        .justify_center()
        .gap_2()
        .rounded(px(10.))
        .border_1()
        .border_color(if variant == 1 {
            rgba(0xffffff33)
        } else {
            rgba(0)
        })
        .text_size(px(12.))
        .line_height(px(DIALOG_LINE_HEIGHT))
        .font_weight(FontWeight::MEDIUM)
        .flex_shrink_0()
        .text_color(rgba(if variant == 2 { 0xffffffff } else { 0xffffff99 }))
        .bg(if variant == 2 {
            color(ACCENT)
        } else if variant == 1 {
            color(BACKGROUND)
        } else {
            rgba(0)
        })
        .opacity(move || if enabled.get() { 1_f32 } else { 0.5_f32 })
        .hover(move |s| {
            let s = s
                .bg(if variant == 2 {
                    color(0x52b885)
                } else {
                    rgba(0xffffff1a)
                })
                .text_color(rgba(if variant == 2 { 0xffffffff } else { 0xffffffeb }));
            if variant == 0 {
                s.shadow(dialog_small_shadow())
            } else {
                s
            }
        })
        .style_when(
            move || ui.is_focus_visible(id),
            |s| {
                s.border_color(color(ACCENT))
                    .inset_ring(px(2.))
                    .inset_ring_color(color(ACCENT))
            },
        )
        .child(image.map(move |name| {
            icon(name, 16., 0xffffff)
                .text_color(rgba(if variant == 2 { 0xffffffff } else { 0xffffff99 }))
                .group_hover(move |s| {
                    s.text_color(rgba(if variant == 2 { 0xffffffff } else { 0xffffffeb }))
                })
        }))
        .child(text(label))
}

pub(super) fn dialog_tabs() -> TabsElement {
    tabs()
        .h(px(36.))
        .w_full()
        .min_w_0()
        .flex_shrink_0()
        .p_1()
        .flex_row()
        .items_center()
        .rounded(px(10.))
        .bg(rgba(0xffffff0a))
}

pub(super) fn dialog_label(label: impl IntoReactive<SharedString>) -> impl IntoElement {
    single_line_text(label, DIALOG_LINE_HEIGHT)
        .text_size(px(12.))
        .font_weight(FontWeight::MEDIUM)
        .text_color(rgba(0xffffff99))
}

pub(super) fn dialog_error(error: impl IntoReactive<SharedString>) -> impl IntoElement {
    text(error)
        .w_full()
        .min_w_0()
        .text_size(px(12.))
        .line_height(px(DIALOG_LINE_HEIGHT))
        .text_color(color(RED))
}

pub(super) fn focused_key(ui: Ui, key: &str, _shift: bool) -> bool {
    let forward = match key {
        "right" | "down" => true,
        "left" | "up" => false,
        _ => return false,
    };
    let groups: &[&[(&str, Action)]] = &[
        &[
            ("mode-ai", Action::Mode(GameMode::AiWhite)),
            ("mode-human", Action::Mode(GameMode::Human)),
        ],
        &[
            ("game-color-black", Action::Mode(GameMode::AiWhite)),
            ("game-color-white", Action::Mode(GameMode::AiBlack)),
        ],
        &[
            ("timed-mode", Action::Timed(true)),
            ("level-mode", Action::Timed(false)),
        ],
        &[
            ("license-main", Action::License(false)),
            ("license-third-party", Action::License(true)),
        ],
        &[
            ("setup-tab-manual", Action::SetupTab(SetupTab::Manual)),
            (
                "setup-tab-transcript",
                Action::SetupTab(SetupTab::Transcript),
            ),
            ("setup-tab-board", Action::SetupTab(SetupTab::Board)),
        ],
        &[
            (
                "setup-turn-black",
                Action::Turn(reversi_core::player_color::PlayerColor::Black),
            ),
            (
                "setup-turn-white",
                Action::Turn(reversi_core::player_color::PlayerColor::White),
            ),
        ],
    ];
    for group in groups {
        if let Some(index) = group.iter().position(|(id, _)| ui.is_focused(id)) {
            let next = if forward {
                (index + 1).min(group.len() - 1)
            } else {
                index.saturating_sub(1)
            };
            if next != index {
                ui.action(group[next].1.clone());
                ui.focus(group[next].0);
            }
            return true;
        }
    }
    false
}

pub(super) fn dialog_small_shadow() -> Vec<BoxShadow> {
    vec![
        BoxShadow {
            offset_x: px(0.),
            offset_y: px(1.),
            blur: px(3.),
            spread: px(-1.),
            color: rgba(0x00000080).into(),
            inset: false,
        },
        BoxShadow {
            offset_x: px(0.),
            offset_y: px(0.),
            blur: px(0.),
            spread: px(1.),
            color: rgba(0xffffff0a).into(),
            inset: false,
        },
    ]
}
