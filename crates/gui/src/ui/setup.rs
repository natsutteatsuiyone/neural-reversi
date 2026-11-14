use nsh_base::{radio, radio_group};
use nsh_core::element::input;
use nsh_core::prelude::*;
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::player_color::PlayerColor;

use super::dialogs::{dialog_button, dialog_error, dialog_label, dialog_tab, dialog_tabs};
use super::widgets::{flat_stone, focusable, move_board_cursor};
use super::{ACCENT, Action, Modal, PANEL, SECONDARY, Ui, color, rgba};
use crate::game::{Position, SetupError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SetupTab {
    Manual,
    Transcript,
    Board,
}

pub(super) struct SetupEditor {
    pub(super) tab: SetupTab,
    pub(super) cells: [Disc; 64],
    pub(super) turn: PlayerColor,
    pub(super) error: Option<SetupError>,
    pub(super) transcript: String,
    pub(super) board: String,
}

impl Default for SetupEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl SetupEditor {
    pub(super) fn new() -> Self {
        Self {
            tab: SetupTab::Manual,
            cells: std::array::from_fn(|index| Position::initial().cell(index)),
            turn: PlayerColor::Black,
            error: None,
            transcript: String::new(),
            board: String::new(),
        }
    }

    pub(super) fn set_position(&mut self, position: Position) {
        self.tab = SetupTab::Manual;
        self.show(position);
    }

    pub(super) fn show(&mut self, position: Position) {
        self.cells = std::array::from_fn(|index| position.cell(index));
        self.turn = position.turn;
        self.error = None;
    }

    pub(super) fn reset(&mut self) {
        self.set_position(Position::initial());
        self.transcript.clear();
        self.board.clear();
    }

    fn resolve(&self) -> Result<Position, SetupError> {
        match self.tab {
            SetupTab::Transcript if !self.transcript.trim().is_empty() => {
                Position::from_transcript(&self.transcript)
            }
            SetupTab::Board if !self.board.trim().is_empty() => {
                Position::from_board_string(&self.board, self.turn)
            }
            _ => {
                let mut player = 0;
                let mut opponent = 0;
                for (index, disc) in self.cells.iter().enumerate() {
                    if *disc == self.turn.into() {
                        player |= 1_u64 << index;
                    } else if *disc != Disc::Empty {
                        opponent |= 1_u64 << index;
                    }
                }
                Ok(Position {
                    board: Board::from_bitboards(player, opponent),
                    turn: self.turn,
                })
            }
        }
    }

    pub(super) fn refresh(&mut self) {
        match self.resolve() {
            Ok(position) => self.show(position),
            Err(error) => self.error = Some(error),
        }
    }

    pub(super) fn position(&self) -> Result<Position, SetupError> {
        let position = self.resolve()?;
        position.validate_setup()?;
        Ok(position)
    }
}

pub(super) fn setup_error_text(error: &SetupError) -> SharedString {
    match error {
        SetupError::TranscriptLength => tr!("error_transcript_length"),
        SetupError::InvalidNotation => tr!("error_invalid_notation"),
        SetupError::IllegalMove(square) => tr!(
            "error_illegal_move",
            square = square.to_string().to_uppercase()
        ),
        SetupError::GameOver => tr!("error_game_over"),
        SetupError::BoardLength => tr!("error_board_length"),
        SetupError::BoardCharacter => tr!("error_board_character"),
        SetupError::MissingColors => tr!("error_missing_colors"),
        SetupError::TooFewDiscs => tr!("error_too_few_discs"),
        SetupError::NoMoves => tr!("error_no_moves"),
        SetupError::TurnHasNoMoves => tr!("error_turn_has_no_moves"),
    }
}

pub(super) fn render_setup_dialog(ui: Ui) -> View {
    let solver = ui.read(|a| a.modal == Some(Modal::SolverSetup));
    view()
        .w_full()
        .min_w_0()
        .gap_6()
        .py_4()
        .child(
            dialog_tabs()
                .child(dialog_tab(
                    ui,
                    "setup-tab-manual",
                    move || tr!("manual"),
                    None,
                    Action::SetupTab(SetupTab::Manual),
                    move || ui.read(|a| a.setup.tab == SetupTab::Manual),
                ))
                .child(dialog_tab(
                    ui,
                    "setup-tab-transcript",
                    move || tr!("transcript"),
                    None,
                    Action::SetupTab(SetupTab::Transcript),
                    move || ui.read(|a| a.setup.tab == SetupTab::Transcript),
                ))
                .child(dialog_tab(
                    ui,
                    "setup-tab-board",
                    move || tr!("board_string"),
                    None,
                    Action::SetupTab(SetupTab::Board),
                    move || ui.read(|a| a.setup.tab == SetupTab::Board),
                )),
        )
        .child(
            view()
                .w_full()
                .min_w_0()
                .flex_row()
                .items_start()
                .gap_4()
                .child(render_setup_board(ui))
                .child(view().flex_1().min_w_0().child(render_setup_controls(ui))),
        )
        .child(solver.then(|| super::solver::solver_options(ui, true)))
        .when(solver, |v| v.pb(px(0.)))
}

fn render_setup_controls(ui: Ui) -> View {
    view()
        .w_full()
        .min_w_0()
        .gap_4()
        .child(setup_input(ui, SetupTab::Transcript).style_when(
            move || ui.read(|a| a.setup.tab != SetupTab::Transcript),
            |s| s.hidden(),
        ))
        .child(setup_input(ui, SetupTab::Board).style_when(
            move || ui.read(|a| a.setup.tab != SetupTab::Board),
            |s| s.hidden(),
        ))
        .child(
            view()
                .gap_2()
                .child(dialog_label(move || tr!("turn")))
                .child(
                    radio_group()
                        .flex_row()
                        .gap_2()
                        .child(turn_control(ui, PlayerColor::Black, "setup-turn-black"))
                        .child(turn_control(ui, PlayerColor::White, "setup-turn-white")),
                ),
        )
        .child(when(
            move || ui.read(|a| a.setup.tab == SetupTab::Manual),
            move || {
                view()
                    .gap_2()
                    .child(
                        dialog_button(
                            ui,
                            "setup-clear",
                            move || tr!("clear"),
                            None,
                            Some(Action::ClearBoard),
                            1,
                        )
                        .h(px(28.))
                        .rounded(px(8.))
                        .text_size(px(12.8))
                        .line_height(px(12.8 * 20. / 14.)),
                    )
                    .child(
                        dialog_button(
                            ui,
                            "setup-initial",
                            move || tr!("initial"),
                            None,
                            Some(Action::InitialBoard),
                            1,
                        )
                        .h(px(28.))
                        .rounded(px(8.))
                        .text_size(px(12.8))
                        .line_height(px(12.8 * 20. / 14.)),
                    )
            },
        ))
        .child(when(
            move || ui.read(|a| a.setup.error.is_some()),
            move || {
                dialog_error(move || {
                    ui.read(|a| {
                        a.setup
                            .error
                            .as_ref()
                            .map(setup_error_text)
                            .unwrap_or_default()
                    })
                })
            },
        ))
}

fn setup_input(ui: Ui, tab: SetupTab) -> View {
    let transcript = tab == SetupTab::Transcript;
    let id = if transcript {
        "setup-transcript-input"
    } else {
        "setup-board-input"
    };
    let reference = focusable(ui, id, move || ui.read(|a| a.setup.tab == tab), || {});
    view()
        .w_full()
        .min_w_0()
        .gap_2()
        .child(dialog_label(move || {
            if transcript {
                tr!("transcript")
            } else {
                tr!("board_string")
            }
        }))
        .child(
            input()
                .element_ref(&reference)
                .label(move || {
                    if transcript {
                        tr!("transcript")
                    } else {
                        tr!("board_string")
                    }
                })
                .placeholder(move || {
                    if transcript {
                        tr!("transcript_placeholder").to_string()
                    } else {
                        tr!("board_placeholder").to_string()
                    }
                })
                .value(move || {
                    ui.read(|a| {
                        if transcript {
                            a.setup.transcript.clone()
                        } else {
                            a.setup.board.clone()
                        }
                    })
                })
                .on_focus(move || ui.focus(id))
                .on_change(move |value: &str| {
                    ui.action(if transcript {
                        Action::SetupTranscript(value.to_owned())
                    } else {
                        Action::SetupBoard(value.to_owned())
                    })
                })
                .multiline(!transcript)
                .w_full()
                .min_w_0()
                .h(px(if transcript { 36. } else { 80. }))
                .px_3()
                .py(px(4.))
                .rounded(px(8.))
                .font_family("Consolas")
                .text_size(move || {
                    ui.read(|a| {
                        px(if transcript && a.viewport.width.0 < 768. {
                            13.
                        } else {
                            12.
                        })
                    })
                })
                .line_height(move || {
                    ui.read(|a| {
                        px(if transcript && a.viewport.width.0 < 768. {
                            19.5
                        } else {
                            12. * 20. / 14.
                        })
                    })
                })
                .text_color(rgba(0xffffffeb))
                .bg(color(PANEL).blend(rgba(0xffffff0d)))
                .border_1()
                .border_color(rgba(0xffffff33))
                .style_when(
                    move || ui.is_focused(id),
                    |s| s.ring(px(1.)).ring_color(color(ACCENT)),
                ),
        )
}

fn turn_control(ui: Ui, disc: PlayerColor, id: &'static str) -> impl IntoElement {
    let read_only = computed(move || ui.read(|a| a.setup.tab == SetupTab::Transcript));
    let selected = computed(move || ui.read(|a| a.setup.turn == disc));
    let reference = focusable(
        ui,
        id,
        move || !read_only.get(),
        move || ui.action(Action::Turn(disc)),
    );
    radio()
        .element_ref(&reference)
        .checked(selected)
        .disabled(read_only)
        .accessibility_label(move || Some(ui.read(|a| a.color(disc))))
        .on_change(move |checked: &bool| {
            if *checked {
                ui.focus(id);
                ui.action(Action::Turn(disc));
            }
        })
        .flex_row()
        .flex_shrink_0()
        .items_center()
        .gap_2()
        .px_3()
        .py_2()
        .rounded(px(10.))
        .border_1()
        .border_color(move || {
            if selected.get() && !read_only.get() {
                color(ACCENT)
            } else if read_only.get() && !selected.get() {
                rgba(0xffffff1a)
            } else {
                rgba(0xffffff26)
            }
        })
        .font_weight(FontWeight::MEDIUM)
        .text_color(move || {
            rgba(if selected.get() {
                0xffffffeb
            } else if read_only.get() {
                0xffffff4d
            } else {
                0xffffff99
            })
        })
        .bg(move || {
            if selected.get() {
                if read_only.get() {
                    rgba(0xffffff0d)
                } else {
                    rgba(0x43a57526)
                }
            } else {
                rgba(0)
            }
        })
        .opacity(move || -> f32 {
            if read_only.get() && !selected.get() {
                0.5
            } else {
                1.
            }
        })
        .hover(move |s| {
            s.border_color(move || {
                if selected.get() {
                    color(ACCENT)
                } else {
                    rgba(0xffffff40)
                }
            })
            .bg(move || {
                if selected.get() {
                    rgba(0x43a57526)
                } else {
                    rgba(0xffffff0d)
                }
            })
        })
        .style_when(
            move || ui.is_focus_visible(id),
            |s| {
                s.border_color(color(ACCENT))
                    .inset_ring(px(2.))
                    .inset_ring_color(color(ACCENT))
            },
        )
        .child(flat_stone(disc == PlayerColor::Black, 16.))
        .child(text(move || ui.read(|a| a.color(disc))))
}

fn render_setup_board(ui: Ui) -> View {
    let scale = pixel_ratio();
    let border = scale.floor().max(1.) / scale;
    let side = 228. + 2. * border;
    let step = side / 8.;
    let reference = focusable(
        ui,
        "setup-grid",
        move || ui.read(|a| a.setup.tab == SetupTab::Manual),
        move || ui.action(Action::Cell(ui.read(|a| a.setup_cursor))),
    );
    let mut frame = view()
        .absolute()
        .left(px(20.))
        .top(px(16.))
        .size(px(side))
        .rounded(px(8.))
        .border(px(border))
        .border_color(color(0x143d2e))
        .bg(color(0x1a4d3a));
    for index in 0..64 {
        let manual = move || ui.read(|a| a.setup.tab == SetupTab::Manual);
        frame = frame.child(
            view()
                .role(AccessibilityRole::Cell)
                .label(move || {
                    ui.read(|a| {
                        format!(
                            "{}{} {}",
                            (b'A' + (index % 8) as u8) as char,
                            index / 8 + 1,
                            PlayerColor::try_from(a.setup.cells[index])
                                .map(|c| a.color(c))
                                .unwrap_or_else(|_| tr!("empty"))
                        )
                    })
                })
                .absolute()
                .left(px(6. + (index % 8) as f32 * 27.))
                .top(px(6. + (index / 8) as f32 * 27.))
                .size(px(27.))
                .items_center()
                .justify_center()
                .bg(color(0x236b4e))
                .border_r(px(border))
                .border_b(px(border))
                .border_color(color(0x1a4d3a))
                .style_when(
                    move || {
                        ui.is_focus_visible("setup-grid") && ui.read(|a| a.setup_cursor == index)
                    },
                    |s| s.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
                )
                .hover(move |s| {
                    s.bg(move || {
                        if manual() {
                            color(0x2a7d5c)
                        } else {
                            color(0x236b4e)
                        }
                    })
                })
                .on_press(move || {
                    if manual() {
                        ui.update(|a| a.setup_cursor = index);
                        ui.focus("setup-grid");
                        ui.action(Action::Cell(index));
                    }
                })
                .child(when(
                    move || ui.read(|a| a.setup.cells[index] == Disc::Black),
                    || flat_stone(true, 21.6),
                ))
                .child(when(
                    move || ui.read(|a| a.setup.cells[index] == Disc::White),
                    || flat_stone(false, 21.6),
                )),
        );
    }
    let mut board = view()
        .element_ref(&reference)
        .role(AccessibilityRole::Grid)
        .label(move || tr!("board_setup"))
        .accessibility_description(move || tr!("manual_help"))
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
        let label = |label: &'static str, width: f32, height: f32| {
            view()
                .absolute()
                .w(px(width))
                .h(px(height))
                .items_center()
                .justify_center()
                .child(
                    text(label)
                        .font_family("Noto Sans")
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_size(px(11.))
                        .text_color(color(SECONDARY)),
                )
        };
        board = board
            .child(
                label(column, step, 16.)
                    .left(px(20. + index as f32 * step))
                    .top(px(0.)),
            )
            .child(
                label(row, 20., step)
                    .left(px(0.))
                    .top(px(16. + index as f32 * step)),
            );
    }
    board
}

pub(super) fn focused_key(ui: Ui, key: &str) -> bool {
    if !ui.is_focused("setup-grid") || !ui.read(|a| a.setup.tab == SetupTab::Manual) {
        return false;
    }
    if let Some(index) = move_board_cursor(ui.read(|a| a.setup_cursor), key) {
        ui.update(|a| a.setup_cursor = index);
        true
    } else if matches!(key, "enter" | "space") {
        ui.action(Action::Cell(ui.read(|a| a.setup_cursor)));
        true
    } else {
        false
    }
}
