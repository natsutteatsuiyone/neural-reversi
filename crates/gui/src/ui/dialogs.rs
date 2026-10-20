use gpui_kit::base::FocusTrapElement;
use gpui_kit::component::Icon;
use gpui_kit::{prelude::*, *};

use super::assets::icon;
use super::new_game::DialogSliderDrag;
use super::setup::SetupTab;
use super::widgets::{arrow_keys, interactive, shadow};
use super::{ACCENT, Action, BACKGROUND, BLUE, Modal, PANEL, RED, ReversiApp, SECONDARY};
use crate::settings::GameMode;

pub(super) const DIALOG_LINE_HEIGHT: f32 = 12. * 20. / 14.;
pub(super) const DIALOG_SMALL_LINE_HEIGHT: f32 = 18.;

impl ReversiApp {
    pub(super) fn render_modal(&self, window: &mut Window, cx: &Context<Self>) -> Div {
        let Some(modal) = self.modal else {
            return div();
        };
        let viewport = window.viewport_size();
        let (width, title, body) = match modal {
            Modal::Settings => return self.render_header_menu(cx),
            Modal::NewGame => (
                448.,
                tr!("new_game"),
                self.render_new_game_dialog(window, cx),
            ),
            Modal::Setup => (
                512.,
                tr!("board_setup"),
                self.render_setup_dialog(window, cx),
            ),
            Modal::SolverSetup => (512., tr!("solver"), self.render_setup_dialog(window, cx)),
            Modal::About => (
                672.,
                "Neural Reversi".into(),
                self.render_about_dialog(viewport.height * 0.6, cx),
            ),
        };
        let width = if viewport.width < px(640.) {
            viewport.width - px(32.)
        } else {
            px(width).min(viewport.width)
        };
        let header = div()
            .flex()
            .items_baseline()
            .gap_2()
            .flex_shrink_0()
            .when(modal == Modal::About, |header| {
                header.child(icon("info.svg", 20., BLUE).my_auto())
            })
            .child(
                div()
                    .text_size(px(18.))
                    .line_height(px(25.2))
                    .font_weight(FontWeight::MEDIUM)
                    .child(title.clone()),
            )
            .when(modal == Modal::About, |header| {
                header.child(
                    div()
                        .text_size(px(13.))
                        .line_height(px(19.5))
                        .text_color(rgb(SECONDARY))
                        .child(format!("v{}", env!("CARGO_PKG_VERSION"))),
                )
            });
        let mut card = div()
            .id("modal-dialog")
            .role(Role::Dialog)
            .aria_label(title.clone())
            .track_focus(&self.modal_focus)
            .tab_group()
            .relative()
            .max_h(viewport.height - px(32.))
            .w(width)
            .flex_shrink_0()
            .flex()
            .flex_col()
            .overflow_hidden()
            .rounded(px(14.))
            .shadow(vec![shadow(rgba(0xffffff17), 0., 0., 1.)])
            .bg(rgb(PANEL))
            .text_color(rgba(0xffffffeb))
            .text_size(px(12.))
            .line_height(px(DIALOG_LINE_HEIGHT))
            .on_any_mouse_down(|_, _, cx| cx.stop_propagation())
            .child(div().px_4().pt_4().flex_shrink_0().child(header))
            .child(
                div()
                    .id(("modal-content", modal as usize))
                    .w_full()
                    .min_w_0()
                    .min_h_0()
                    .overflow_y_scroll()
                    .px_4()
                    .py_4()
                    .child(body),
            );
        if modal != Modal::About {
            let valid = modal == Modal::NewGame || self.setup.error.is_none();
            let mut footer = div()
                .flex()
                .items_center()
                .gap_2()
                .p_4()
                .flex_shrink_0()
                .border_t_1()
                .border_color(rgba(0xffffff0f))
                .bg(rgba(0xffffff05))
                .child(dialog_button(
                    "modal-cancel",
                    tr!("cancel"),
                    None,
                    Some(Action::Close),
                    0,
                    cx,
                ))
                .child(div().flex_1());
            if modal == Modal::NewGame {
                footer = footer.child(
                    dialog_button(
                        "new-position",
                        tr!("board_setup"),
                        None,
                        Some(Action::Setup),
                        1,
                        cx,
                    )
                    .gap_1()
                    .child(icon("chevron-right.svg", 16., 0xffffff).text_color(rgba(0xffffff99))),
                );
            } else if modal == Modal::Setup {
                footer = footer.child(
                    dialog_button(
                        "setup-back",
                        tr!("back"),
                        Some("chevron-left.svg"),
                        Some(Action::Back),
                        0,
                        cx,
                    )
                    .gap_1(),
                );
            }
            footer = footer.child(dialog_button(
                "modal-start",
                if modal == Modal::SolverSetup {
                    tr!("start_solver")
                } else {
                    tr!("start_game")
                },
                Some("play.svg"),
                valid.then_some(Action::Start),
                2,
                cx,
            ));
            card = card.child(footer);
        } else {
            card = card.child(
                self.control("modal-close", "x.svg", "", Action::Close, true, cx)
                    .absolute()
                    .top(px(8.))
                    .right(px(8.)),
            );
        }
        div()
            .absolute()
            .inset_0()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(rgba(0x00000080))
            .occlude()
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, window, cx| this.action(Action::Close, window, cx)),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                if this.license.drag(event) {
                    cx.notify();
                }
            }))
            .on_drop(cx.listener(|this, drag: &DialogSliderDrag, window, cx| {
                this.set_dialog_slider(drag.timed, window.mouse_position().x, cx);
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    this.license.end_drag();
                    cx.stop_propagation();
                }),
            )
            .on_scroll_wheel(|_, _, cx| cx.stop_propagation())
            .child(card.focus_trap("modal-focus-trap", &self.modal_focus))
    }
}

pub(super) fn dialog_tab(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    image: Option<&'static str>,
    action: Action,
    selected: bool,
    cx: &Context<ReversiApp>,
) -> Stateful<Div> {
    let label = label.into();
    let (index, count) = match action {
        Action::Mode(mode) => (usize::from(mode == GameMode::Human), 2),
        Action::Timed(timed) => (usize::from(!timed), 2),
        Action::License(value) => (usize::from(value), 2),
        Action::SetupTab(tab) => (tab as usize, 3),
        _ => (0, 1),
    };
    interactive(div().id(id), label.clone(), true)
        .role(Role::Tab)
        .aria_selected(selected)
        .on_key_down(arrow_keys(
            index,
            count,
            move |this, neighbor| match action {
                Action::Mode(_) => Action::Mode(if this.draft.game_mode == GameMode::Human {
                    GameMode::AiWhite
                } else {
                    GameMode::Human
                }),
                Action::Timed(_) => Action::Timed(!this.draft.timed),
                Action::License(value) => Action::License(!value),
                _ => Action::SetupTab(
                    [SetupTab::Manual, SetupTab::Transcript, SetupTab::Board][neighbor],
                ),
            },
            cx,
        ))
        .flex_1()
        .min_w_0()
        .w_0()
        .h_full()
        .flex()
        .items_center()
        .justify_center()
        .gap_2()
        .px_1p5()
        .rounded(px(8.))
        .border_1()
        .border_color(rgba(0))
        .text_size(px(12.))
        .font_weight(FontWeight::MEDIUM)
        .line_height(px(DIALOG_LINE_HEIGHT))
        .whitespace_nowrap()
        .text_color(rgba(if selected { 0xffffffeb } else { 0xffffff8d }))
        .when(selected, |tab| tab.bg(rgb(BACKGROUND)))
        .cursor_pointer()
        .hover(|style| style.text_color(rgba(0xffffffeb)))
        .on_click(cx.listener(move |this, _, window, cx| this.action(action, window, cx)))
        .when_some(image, |tab, image| {
            tab.child(Icon::empty().path(image).size(px(16.)).flex_shrink_0())
        })
        .child(label)
}

/// `variant`: 0 = ghost, 1 = outlined, 2 = primary. `None` renders it disabled.
pub(super) fn dialog_button(
    id: impl Into<ElementId>,
    label: impl Into<SharedString>,
    image: Option<&'static str>,
    action: Option<Action>,
    variant: u8,
    cx: &Context<ReversiApp>,
) -> Stateful<Div> {
    let label = label.into();
    interactive(div().id(id), label.clone(), action.is_some())
        .h(px(32.))
        .px(px(10.))
        .flex()
        .items_center()
        .justify_center()
        .gap_2()
        .rounded(px(10.))
        .border_1()
        .border_color(rgba(0))
        .text_size(px(12.))
        .line_height(px(DIALOG_LINE_HEIGHT))
        .whitespace_nowrap()
        .font_weight(FontWeight::MEDIUM)
        .flex_shrink_0()
        .text_color(rgba(if variant == 2 { 0xffffffff } else { 0xffffff99 }))
        .when(variant == 1, |button| {
            button.border_color(rgba(0xffffff33)).bg(rgb(BACKGROUND))
        })
        .when(variant == 2, |button| button.bg(rgb(ACCENT)))
        .when(action.is_none(), |button| button.opacity(0.5))
        .when_some(action, |button, action| {
            button
                .cursor_pointer()
                .hover(move |style| {
                    let style = style
                        .bg(if variant == 2 {
                            rgb(0x52b885)
                        } else {
                            rgba(0xffffff1a)
                        })
                        .text_color(rgba(if variant == 2 { 0xffffffff } else { 0xffffffeb }));
                    if variant == 0 {
                        style.shadow(dialog_small_shadow())
                    } else {
                        style
                    }
                })
                .on_click(cx.listener(move |this, _, window, cx| this.action(action, window, cx)))
        })
        .when_some(image, |button, image| {
            button.child(Icon::empty().path(image).size(px(16.)).flex_shrink_0())
        })
        .child(label)
}

pub(super) fn dialog_tabs() -> Div {
    div()
        .h(px(36.))
        .w_full()
        .min_w_0()
        .flex_shrink_0()
        .p_1()
        .flex()
        .items_center()
        .rounded(px(10.))
        .bg(rgba(0xffffff0a))
}

pub(super) fn dialog_label(label: impl Into<SharedString>) -> Div {
    div()
        .text_size(px(12.))
        .line_height(px(DIALOG_LINE_HEIGHT))
        .font_weight(FontWeight::MEDIUM)
        .text_color(rgba(0xffffff99))
        .child(label.into())
}

pub(super) fn dialog_error(error: &str) -> Div {
    div()
        .w_full()
        .min_w_0()
        .text_size(px(12.))
        .line_height(px(DIALOG_LINE_HEIGHT))
        .whitespace_normal()
        .text_color(rgb(RED))
        .child(error.to_owned())
}

pub(super) fn dialog_small_shadow() -> Vec<BoxShadow> {
    vec![
        shadow(rgba(0x00000080), 1., 3., -1.),
        shadow(rgba(0xffffff0a), 0., 0., 1.),
    ]
}
