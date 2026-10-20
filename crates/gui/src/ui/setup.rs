use gpui_kit::component::input::{Input, InputEvent, InputState, Textarea, TextareaState};
use gpui_kit::{prelude::*, *};
use reversi_core::disc::Disc;

use super::board::board_label;
use super::dialogs::{
    DIALOG_LINE_HEIGHT, dialog_button, dialog_error, dialog_label, dialog_small_shadow, dialog_tab,
    dialog_tabs,
};
use super::solver::solver_options;
use super::widgets::{arrow_keys, flat_stone, interactive, move_board_cursor, shadow};
use super::{ACCENT, Action, Modal, PANEL, ReversiApp, SECONDARY};
use crate::game::{Position, SetupError};

#[derive(Clone, Copy, PartialEq)]
pub(super) enum SetupTab {
    Manual,
    Transcript,
    Board,
}

/// Board setup shared by the New Game and Solver dialogs. The preview always
/// shows the last valid input, and an empty transcript or board string keeps
/// it; `error` describes the current input, if invalid.
pub(super) struct SetupEditor {
    pub(super) tab: SetupTab,
    pub(super) cells: [Disc; 64],
    pub(super) turn: Disc,
    pub(super) error: Option<SetupError>,
    transcript_input: Entity<InputState>,
    board_input: Entity<TextareaState>,
}

fn setup_input<E: Styled>(input: E, transcript: bool, narrow: bool, focused: bool) -> E {
    input
        .w_full()
        .min_w_0()
        .rounded(px(8.))
        .text_size(px(if narrow { 13. } else { 12. }))
        .line_height(px(if narrow { 19.5 } else { DIALOG_LINE_HEIGHT }))
        .font_family("Consolas")
        .text_color(rgba(0xffffffeb))
        .bg(rgb(PANEL).blend(rgba(0xffffff0d)))
        .border_1()
        .border_color(rgba(0xffffff33))
        .shadow(if focused {
            vec![shadow(rgb(ACCENT), 0., 0., 1.)]
        } else if transcript {
            dialog_small_shadow()
        } else {
            vec![]
        })
}

impl SetupEditor {
    pub(super) fn set_position(&mut self, position: Position) {
        self.tab = SetupTab::Manual;
        self.show(position);
    }

    /// Shows `position` in the preview and clears the error.
    pub(super) fn show(&mut self, position: Position) {
        self.cells = std::array::from_fn(|index| position.cell(index));
        self.turn = position.turn;
        self.error = None;
    }

    pub(super) fn new(window: &mut Window, cx: &mut Context<ReversiApp>) -> Self {
        let transcript_input =
            cx.new(|cx| InputState::new(window, cx).placeholder(tr!("transcript_placeholder")));
        let board_input = cx.new(|cx| {
            TextareaState::new(window, cx)
                .placeholder(tr!("board_placeholder"))
                .rows(3)
        });
        cx.subscribe(&transcript_input, |this, _, event: &InputEvent, cx| {
            if matches!(event, InputEvent::Change) && this.setup.tab == SetupTab::Transcript {
                this.setup.refresh(cx);
                cx.notify();
            }
        })
        .detach();
        cx.subscribe(&board_input, |this, _, event: &InputEvent, cx| {
            if matches!(event, InputEvent::Change) && this.setup.tab == SetupTab::Board {
                this.setup.refresh(cx);
                cx.notify();
            }
        })
        .detach();
        Self {
            tab: SetupTab::Manual,
            cells: std::array::from_fn(|index| Position::initial().cell(index)),
            turn: Disc::Black,
            error: None,
            transcript_input,
            board_input,
        }
    }

    /// Restores the initial position on the Manual tab and empties both inputs.
    pub(super) fn reset(&mut self, window: &mut Window, cx: &mut App) {
        self.set_position(Position::initial());
        self.transcript_input
            .update(cx, |input, cx| input.set_value("", window, cx));
        self.board_input
            .update(cx, |input, cx| input.set_value("", window, cx));
    }

    pub(super) fn update_placeholders(&self, window: &mut Window, cx: &mut App) {
        self.transcript_input.update(cx, |input, cx| {
            input.set_placeholder(tr!("transcript_placeholder"), window, cx);
        });
        self.board_input.update(cx, |input, cx| {
            input.set_placeholder(tr!("board_placeholder"), window, cx);
        });
    }

    fn resolve(&self, cx: &App) -> Result<Position, SetupError> {
        let transcript = self.transcript_input.read(cx).value();
        let board = self.board_input.read(cx).value();
        match self.tab {
            SetupTab::Transcript if !transcript.trim().is_empty() => {
                Position::from_transcript(transcript.as_str())
            }
            SetupTab::Board if !board.trim().is_empty() => {
                Position::from_board_string(board.as_str(), self.turn)
            }
            _ => {
                let mut player = 0;
                let mut opponent = 0;
                for (index, disc) in self.cells.iter().enumerate() {
                    if *disc == self.turn {
                        player |= 1_u64 << index;
                    } else if *disc != Disc::Empty {
                        opponent |= 1_u64 << index;
                    }
                }
                Ok(Position {
                    board: reversi_core::board::Board::from_bitboards(player, opponent),
                    turn: self.turn,
                })
            }
        }
    }

    /// Re-reads the active tab's input into the preview, or records its error.
    pub(super) fn refresh(&mut self, cx: &App) {
        match self.resolve(cx) {
            Ok(position) => self.show(position),
            Err(error) => self.error = Some(error),
        }
    }

    /// The active tab's position, if it is a playable setup.
    pub(super) fn position(&self, cx: &App) -> Result<Position, SetupError> {
        let position = self.resolve(cx)?;
        position.validate_setup()?;
        Ok(position)
    }
}

impl ReversiApp {
    fn setup_error_text(&self, error: &SetupError) -> String {
        match error {
            SetupError::TranscriptLength => tr!("error_transcript_length"),
            SetupError::InvalidNotation => tr!("error_invalid_notation"),
            SetupError::IllegalMove(square) => {
                tr!(
                    "error_illegal_move",
                    square = square.to_string().to_uppercase()
                )
            }
            SetupError::GameOver => tr!("error_game_over"),
            SetupError::BoardLength => tr!("error_board_length"),
            SetupError::BoardCharacter => tr!("error_board_character"),
            SetupError::MissingColors => tr!("error_missing_colors"),
            SetupError::TooFewDiscs => tr!("error_too_few_discs"),
            SetupError::NoMoves => tr!("error_no_moves"),
            SetupError::TurnHasNoMoves => tr!("error_turn_has_no_moves"),
            SetupError::MissingTurn => tr!("error_missing_turn"),
        }
        .into()
    }

    fn render_setup_controls(&self, tab: SetupTab, window: &Window, cx: &Context<Self>) -> Div {
        let setup = &self.setup;
        let error = setup
            .error
            .as_ref()
            .map(|error| self.setup_error_text(error));
        let turn = setup.turn;
        let mut controls = div().w_full().min_w_0().flex().flex_col().gap_4();
        if tab != SetupTab::Manual {
            let transcript = tab == SetupTab::Transcript;
            let focused = if transcript {
                setup
                    .transcript_input
                    .read(cx)
                    .focus_handle(cx)
                    .is_focused(window)
            } else {
                setup
                    .board_input
                    .read(cx)
                    .focus_handle(cx)
                    .is_focused(window)
            };
            let narrow = transcript && window.viewport_size().width < px(768.);
            controls = controls.child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(dialog_label(if transcript {
                        tr!("transcript")
                    } else {
                        tr!("board_string")
                    }))
                    .child(if transcript {
                        setup_input(
                            Styled::h(Input::new(&setup.transcript_input), px(36.))
                                .aria_label(tr!("transcript"))
                                .focus_bordered(false)
                                .px_3()
                                .py(px(4.)),
                            transcript,
                            narrow,
                            focused,
                        )
                        .into_any_element()
                    } else {
                        setup_input(
                            Textarea::new(&setup.board_input)
                                .aria_label(tr!("board_string"))
                                .h(px(80.))
                                .bordered(false),
                            transcript,
                            narrow,
                            focused,
                        )
                        .into_any_element()
                    })
                    .when_some(error.as_deref(), |input, error| {
                        input.child(dialog_error(error))
                    }),
            );
        }
        controls = controls.child(
            div()
                .w_full()
                .min_w_0()
                .flex()
                .flex_col()
                .gap_2()
                .child(dialog_label(tr!("turn")))
                .child(
                    div().flex().gap_2().children(
                        [Disc::Black, Disc::White]
                            .into_iter()
                            .enumerate()
                            .map(|(index, disc)| {
                                let selected = turn == disc;
                                let read_only = tab == SetupTab::Transcript;
                                interactive(
                                    div().id(("setup-turn", tab as usize * 2 + index)),
                                    self.color(disc),
                                    !read_only,
                                )
                                .role(Role::RadioButton)
                                .aria_toggled(selected.into())
                                .when(!read_only, |button| {
                                    button.on_key_down(arrow_keys(
                                        index,
                                        2,
                                        move |_, _| {
                                            Action::Turn(if disc == Disc::Black {
                                                Disc::White
                                            } else {
                                                Disc::Black
                                            })
                                        },
                                        cx,
                                    ))
                                })
                                .flex_shrink_0()
                                .flex()
                                .items_center()
                                .gap_2()
                                .px_3()
                                .py_2()
                                .rounded(px(10.))
                                .border_1()
                                .border_color(if selected && !read_only {
                                    rgb(ACCENT)
                                } else if read_only && !selected {
                                    rgba(0xffffff1a)
                                } else {
                                    rgba(0xffffff26)
                                })
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(rgba(if selected {
                                    0xffffffeb
                                } else if read_only {
                                    0xffffff4d
                                } else {
                                    0xffffff99
                                }))
                                .when(selected, |button| {
                                    button.bg(if read_only {
                                        rgba(0xffffff0d)
                                    } else {
                                        rgba(0x43a57526)
                                    })
                                })
                                .when(read_only && !selected, |button| button.opacity(0.5))
                                .when(!read_only, |button| {
                                    button
                                        .cursor_pointer()
                                        .when(!selected, |button| {
                                            button.hover(|style| {
                                                style
                                                    .border_color(rgba(0xffffff40))
                                                    .bg(rgba(0xffffff0d))
                                            })
                                        })
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            this.action(Action::Turn(disc), window, cx)
                                        }))
                                })
                                .child(flat_stone(disc, 16.))
                                .child(self.color(disc))
                            }),
                    ),
                ),
        );
        if tab == SetupTab::Manual {
            controls = controls
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            dialog_button(
                                "setup-clear",
                                tr!("clear"),
                                None,
                                Some(Action::ClearBoard),
                                1,
                                cx,
                            )
                            .h(px(28.))
                            .rounded(px(8.))
                            .line_height(px(12.8 * 20. / 14.))
                            .text_size(px(12.8)),
                        )
                        .child(
                            dialog_button(
                                "setup-initial",
                                tr!("initial"),
                                None,
                                Some(Action::InitialBoard),
                                1,
                                cx,
                            )
                            .h(px(28.))
                            .rounded(px(8.))
                            .line_height(px(12.8 * 20. / 14.))
                            .text_size(px(12.8)),
                        ),
                )
                .when_some(error.as_deref(), |controls, error| {
                    controls.child(dialog_error(error))
                });
        }
        controls
    }

    pub(super) fn render_setup_dialog(&self, window: &Window, cx: &Context<Self>) -> Div {
        let controls =
            div()
                .flex_1()
                .min_w_0()
                .child(self.render_setup_controls(self.setup.tab, window, cx));
        let mut body = div()
            .w_full()
            .min_w_0()
            .flex()
            .flex_col()
            .gap_6()
            .py_4()
            .child(
                dialog_tabs().children(
                    [
                        (SetupTab::Manual, tr!("manual")),
                        (SetupTab::Transcript, tr!("transcript")),
                        (SetupTab::Board, tr!("board_string")),
                    ]
                    .into_iter()
                    .enumerate()
                    .map(|(index, (tab, label))| {
                        dialog_tab(
                            ("setup-tab", index),
                            label,
                            None,
                            Action::SetupTab(tab),
                            self.setup.tab == tab,
                            cx,
                        )
                        .when(index == 0, |tab| tab.track_focus(&self.modal_entry_focus))
                    }),
                ),
            )
            .child(
                div()
                    .w_full()
                    .min_w_0()
                    .flex()
                    .items_start()
                    .gap_4()
                    .child(
                        div()
                            .flex_shrink_0()
                            .child(self.render_setup_board(window, cx)),
                    )
                    .child(controls),
            );
        if self.modal == Some(Modal::SolverSetup) {
            body = body.pb_0().child(solver_options(&self.draft, cx).mt_2());
        }
        body
    }

    fn render_setup_board(&self, window: &Window, cx: &Context<Self>) -> Stateful<Div> {
        let manual = self.setup.tab == SetupTab::Manual;
        // CSS borders snap down to whole device pixels at fractional scaling.
        let border = window.scale_factor().floor().max(1.) / window.scale_factor();
        let side = 228. + 2. * border;
        let label_step = side / 8.;
        let mut frame = div()
            .absolute()
            .left(px(20.))
            .top(px(16.))
            .size(px(side))
            .rounded_lg()
            .border(px(border))
            .border_color(rgb(0x143d2e))
            .bg(rgb(0x1a4d3a));
        for (index, &disc) in self.setup.cells.iter().enumerate() {
            frame = frame.child(
                div()
                    .id(("setup-cell", index))
                    .role(Role::Cell)
                    .aria_label(format!(
                        "{}{} {}",
                        (b'A' + (index % 8) as u8) as char,
                        index / 8 + 1,
                        if disc == Disc::Empty {
                            tr!("empty")
                        } else {
                            self.color(disc)
                        }
                    ))
                    .when(
                        manual && self.setup_focus.is_focused(window) && self.setup_cursor == index,
                        |cell| cell.aria_active_descendant(),
                    )
                    .absolute()
                    .left(px(6. + (index % 8) as f32 * 27.))
                    .top(px(6. + (index / 8) as f32 * 27.))
                    .size(px(27.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(rgb(0x236b4e))
                    .border_r(px(border))
                    .border_b(px(border))
                    .border_color(rgb(0x1a4d3a))
                    .when(manual, |cell| {
                        cell.cursor_pointer()
                            .hover(|style| style.bg(rgb(0x2a7d5c)))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.setup_cursor = index;
                                this.setup_focus.focus(window, cx);
                                this.action(Action::Cell(index), window, cx);
                            }))
                    })
                    .when(disc != Disc::Empty, |cell| {
                        cell.child(flat_stone(disc, 21.6))
                    }),
            );
        }
        let mut board = div()
            .id("setup-grid")
            .role(Role::Grid)
            .aria_label(tr!("board_setup"))
            .aria_description(tr!("manual_help"))
            .aria_row_count(8)
            .aria_column_count(8)
            .track_focus(&self.setup_focus.clone().tab_stop(manual))
            .when(manual, |board| {
                board.on_key_down(cx.listener(|this, event: &KeyDownEvent, window, cx| {
                    if let Some(index) = move_board_cursor(this.setup_cursor, &event.keystroke.key)
                    {
                        this.setup_cursor = index;
                        cx.notify();
                        cx.stop_propagation();
                    } else if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.action(Action::Cell(this.setup_cursor), window, cx);
                        cx.stop_propagation();
                    }
                }))
            })
            .relative()
            .w(px(20. + side))
            .h(px(16. + side))
            .flex_shrink_0()
            .child(frame);
        for (index, (column, row)) in ["A", "B", "C", "D", "E", "F", "G", "H"]
            .into_iter()
            .zip(["1", "2", "3", "4", "5", "6", "7", "8"])
            .enumerate()
        {
            board = board
                .child(
                    board_label(column, label_step, 16., 11.)
                        .left(px(20. + index as f32 * label_step))
                        .top_0()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(rgb(SECONDARY)),
                )
                .child(
                    board_label(row, 20., label_step, 11.)
                        .left_0()
                        .top(px(16. + index as f32 * label_step))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(rgb(SECONDARY)),
                );
        }
        board
    }
}
