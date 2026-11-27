use super::theme::{layout_px, stroke_px};
use std::time::{Duration, Instant};

use nsh_core::canvas::{CanvasOp, PathOp};
use nsh_core::element::{View, canvas, image};
use nsh_core::prelude::*;
use reversi_core::disc::Disc;
use reversi_core::player_color::PlayerColor;
use reversi_core::square::Square;

use super::assets;
use super::format::{ScorePrecision, display_score, whole_score};
use super::widgets::{focusable, move_board_cursor};
use super::{ACCENT, AI_COLOR, Action, BLUE, GOLD, ReversiApp, Ui, color};
use crate::engine::Progress;
use crate::game::Position;
use crate::session::Activity;

const TOTAL_SIZE: f32 = 9.5;
const BOARD_SIZE: f32 = 8.;
const FRAME_WIDTH: f32 = 0.15;
const BOARD_GRID: u32 = 0x042c1a;

#[derive(Clone, Debug, PartialEq)]
struct Candidate {
    label: String,
    progress: f32,
    best: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct Cell {
    disc: Disc,
    candidate: Option<Candidate>,
    legal: bool,
    last: bool,
    thinking: bool,
    trail: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
struct BoardPresentation {
    cells: [Cell; 64],
    playable: bool,
    ai_waiting: bool,
    show_legal: bool,
    show_waiting: bool,
    complete: bool,
    last_key: Option<(Position, usize)>,
    motion: Option<(Position, usize, Position)>,
}

fn presentation(app: &ReversiApp) -> BoardPresentation {
    let game = app.session.game();
    let position = game.position();
    let solver = app.session.is_solving();
    let human_turn = app.settings.ai_color() != Some(position.turn);
    let legal = position.board.legal_moves().bits();
    let idle = !matches!(
        app.session.activity(),
        Some(Activity::Initializing | Activity::Analysis | Activity::Move)
    );
    let show_candidates = (solver || (app.settings.hints && human_turn)) && idle;
    let mut candidates = [None; 64];
    let mut best_score = None;
    if show_candidates {
        for candidate in app.session.candidates() {
            if let Some(square) = candidate.square
                && legal & (1_u64 << square.index()) != 0
            {
                candidates[square.index()] = Some(candidate);
                let score = whole_score(candidate.score);
                best_score = Some(best_score.map_or(score, |best: i32| best.max(score)));
            }
        }
    }
    let thinking = (app.session.activity() == Some(Activity::Move))
        .then(|| {
            app.session
                .log()
                .iter()
                .rev()
                .find_map(|progress| progress.square)
        })
        .flatten();
    let mut trails = [None; 3];
    if let Some(current) = thinking {
        let mut count = 0;
        for square in app
            .session
            .log()
            .iter()
            .rev()
            .filter_map(|progress| progress.square)
        {
            if square != current && !trails.contains(&Some(square)) {
                trails[count] = Some(square);
                count += 1;
                if count == trails.len() {
                    break;
                }
            }
        }
    }
    let last_move = (!solver)
        .then(|| {
            game.history[..game.cursor]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, record)| record.square.is_some())
        })
        .flatten();
    let animate = !solver && !app.session.is_paused() && game.cursor == game.history.len();
    let last_key = last_move.map(|_| (position, game.cursor));
    let motion = last_move.filter(|_| animate).map(|(index, _)| {
        let before = index
            .checked_sub(1)
            .map_or(game.initial, |index| game.history[index].position);
        (position, game.cursor, before)
    });
    BoardPresentation {
        cells: std::array::from_fn(|index| {
            let square = Square::from_u8(index as u8).expect("board cell index");
            Cell {
                disc: position.cell(index),
                candidate: candidates[index].map(|candidate| Candidate {
                    label: display_score(candidate.score, ScorePrecision::Whole),
                    progress: candidate_progress(candidate),
                    best: Some(whole_score(candidate.score)) == best_score,
                }),
                legal: legal & (1_u64 << index) != 0,
                last: last_move.is_some_and(|(_, record)| record.square == Some(square)),
                thinking: thinking == Some(square),
                trail: trails.iter().position(|previous| *previous == Some(square)),
            }
        }),
        playable: app.modal.is_none() && idle && (solver || human_turn),
        ai_waiting: app.modal.is_none() && app.ai_resume_label().is_some(),
        show_legal: solver || human_turn,
        show_waiting: show_candidates
            && matches!(
                app.session.activity(),
                Some(Activity::Hint | Activity::Solver)
            )
            && !app.session.candidates().is_empty()
            && (!solver || app.settings.solver_multi_pv),
        complete: app.session.candidates_complete(),
        last_key,
        motion,
    }
}

pub(super) fn render_board(ui: Ui, side: Memo<f32>) -> View {
    let state = computed(move || ui.read(presentation));
    let cell_side = computed(move || side.get() / TOTAL_SIZE);
    let rim = computed(move || (side.get() - cell_side.get() * BOARD_SIZE) / 2.);
    let groove =
        computed(move || (cell_side.get() * 0.03 * pixel_ratio()).round().max(1.) / pixel_ratio());
    let diameter =
        computed(move || (cell_side.get() * 0.84 * pixel_ratio()).round() / pixel_ratio());
    let ring = computed(move || (cell_side.get() * 0.05 * pixel_ratio()).round() / pixel_ratio());
    let progress = signal(1_f32);
    let motion_trigger =
        computed(move || state.with(|state| (state.last_key, state.motion.is_some())));
    let identity = stored(None::<(Position, usize)>);
    // Shadow blur and spread are not channels of NSH's native animation graph.
    effect(move || {
        let (key, animate) = motion_trigger.get();
        if identity.get() != key {
            identity.replace(key);
            if animate && key.is_some() && !reduce_motion() {
                let begin = nsh_core::time::now();
                progress.set(0.);
                spawn(async move {
                    loop {
                        next_frame().await;
                        if reduce_motion() {
                            progress.set(1.);
                            break;
                        }
                        let elapsed = (nsh_core::time::now()
                            .saturating_duration_since(begin)
                            .as_secs_f32()
                            / 0.4)
                            .min(1.);
                        progress.set(elapsed);
                        if elapsed >= 1. {
                            break;
                        }
                    }
                });
            } else {
                progress.set(1.);
            }
        } else if !animate || reduce_motion() {
            progress.set(1.);
        }
    });
    let reference = focusable(
        ui,
        "board-grid",
        move || ui.read(|app| app.modal.is_none()),
        move || {
            board_key(ui, "enter");
        },
    );
    let mut board = view()
        .relative()
        .flex_shrink_0()
        .element_ref(&reference)
        .test_id("board-grid")
        .role(AccessibilityRole::Grid)
        .label(move || tr!("board"))
        .accessibility_description(move || tr!("board_keyboard_help"))
        .w(move || layout_px(side.get()))
        .h(move || layout_px(side.get()));
    board = board.child(
        view()
            .absolute()
            .left(move || layout_px(rim.get() - cell_side.get() * FRAME_WIDTH))
            .top(move || layout_px(rim.get() - cell_side.get() * FRAME_WIDTH))
            .w(move || layout_px(cell_side.get() * (BOARD_SIZE + FRAME_WIDTH * 2.)))
            .h(move || layout_px(cell_side.get() * (BOARD_SIZE + FRAME_WIDTH * 2.)))
            .bg(color(0x010102))
            .border(move || stroke_px(cell_side.get() * 0.01))
            .border_color(color(0x040407)),
    );
    let mut cells = view()
        .absolute()
        .left(move || layout_px(rim.get()))
        .top(move || layout_px(rim.get()))
        .w(move || layout_px(cell_side.get() * BOARD_SIZE))
        .h(move || layout_px(cell_side.get() * BOARD_SIZE))
        .child(
            image(assets::image("board/felt.png"))
                .absolute()
                .inset_0()
                .size_full()
                .object_fit(ResizeMode::Stretch),
        );
    for line in 0..=8 {
        cells = cells
            .child(
                view()
                    .absolute()
                    .left(move || layout_px(line as f32 * cell_side.get() - groove.get() / 2.))
                    .top_0()
                    .w(move || layout_px(groove.get()))
                    .h_full()
                    .bg(color(BOARD_GRID)),
            )
            .child(
                view()
                    .absolute()
                    .left_0()
                    .top(move || layout_px(line as f32 * cell_side.get() - groove.get() / 2.))
                    .w_full()
                    .h(move || layout_px(groove.get()))
                    .bg(color(BOARD_GRID)),
            );
    }
    for row in [2., 6.] {
        for column in [2., 6.] {
            cells = cells.child(
                view()
                    .absolute()
                    .left(move || layout_px((column - 0.06) * cell_side.get()))
                    .top(move || layout_px((row - 0.06) * cell_side.get()))
                    .w(move || layout_px(cell_side.get() * 0.12))
                    .h(move || layout_px(cell_side.get() * 0.12))
                    .rounded_full()
                    .bg(color(0x1a1a1a)),
            );
        }
    }
    for index in 0..64 {
        let cell = computed(move || state.with(|state| state.cells[index].clone()));
        let clickable = move || state.with(|state| state.playable && state.cells[index].legal);
        let square = Square::from_u8(index as u8).expect("board cell index");
        let mut element = view()
            .absolute()
            .test_id(&format!("board-cell-{index}"))
            .role(AccessibilityRole::Cell)
            .label(move || {
                let (disc, legal) = cell.with(|cell| (cell.disc, cell.legal));
                let label = match PlayerColor::try_from(disc) {
                    Ok(player) => ui.read(|app| app.color(player)),
                    Err(_) => tr!("empty"),
                };
                if legal {
                    format!(
                        "{} {} {}",
                        square.to_string().to_uppercase(),
                        label,
                        tr!("legal_move")
                    )
                } else {
                    format!("{} {}", square.to_string().to_uppercase(), label)
                }
            })
            .accessibility_state(move || AccessibilityState {
                selected: ui.is_focused("board-grid") && ui.read(|app| app.board_cursor == index),
                ..AccessibilityState::default()
            })
            .left(move || layout_px((index % 8) as f32 * cell_side.get()))
            .top(move || layout_px((index / 8) as f32 * cell_side.get()))
            .w(move || layout_px(cell_side.get()))
            .h(move || layout_px(cell_side.get()))
            .items_center()
            .justify_center()
            .focusable(false)
            .style_when(clickable, |style| style.cursor_pointer())
            .style_when(
                move || {
                    ui.is_focus_visible("board-grid") && ui.read(|app| app.board_cursor == index)
                },
                |style| style.inset_ring(px(2.)).inset_ring_color(color(ACCENT)),
            )
            .on_press(move || {
                if clickable() {
                    ui.update(|app| app.board_cursor = index);
                    ui.focus("board-grid");
                    ui.action(Action::Cell(index));
                } else if state.with(|state| state.ai_waiting) {
                    ui.update(ReversiApp::show_ai_paused_hint);
                }
            });
        let disc = view()
            .relative()
            .flex_shrink_0()
            .rounded_full()
            .w(move || layout_px(diameter.get()))
            .h(move || {
                let after = cell.with(|cell| cell.disc);
                let before =
                    state.with(|state| state.motion.map(|(_, _, before)| before.cell(index)));
                let scale = before
                    .filter(|before| *before != Disc::Empty && *before != after)
                    .map_or(1., |_| {
                        (std::f32::consts::PI * progress.get()).cos().abs().max(0.1)
                    });
                layout_px(diameter.get() * scale)
            })
            .style_when(
                move || cell.with(|cell| cell.disc == Disc::Empty),
                |style| style.hidden(),
            )
            .style_with(move |style| {
                let side = cell_side.get();
                let dropping = state.with(|state| {
                    state
                        .motion
                        .is_some_and(|(_, _, before)| before.cell(index) == Disc::Empty)
                });
                let height = if dropping {
                    (1. - progress.get()).powi(3) * side * 0.12
                } else {
                    0.
                };
                style.shadow([BoxShadow {
                    offset_x: px(-side * 0.04 - height),
                    offset_y: px(-side * 0.03 - height),
                    blur: px(side * 0.035 + height * 0.4),
                    spread: px(-side * 0.025),
                    color: color(0x000000).opacity(77. / 255.).into(),
                    inset: false,
                }])
            })
            .child(
                image(move || {
                    let after = cell.with(|cell| cell.disc);
                    let disc = state
                        .with(|state| state.motion.map(|(_, _, before)| before.cell(index)))
                        .filter(|before| {
                            *before != Disc::Empty && *before != after && progress.get() < 0.5
                        })
                        .unwrap_or(after);
                    assets::image(if disc == Disc::Black {
                        "board/disc-black.png"
                    } else {
                        "board/disc-white.png"
                    })
                })
                .absolute()
                .inset_0()
                .size_full()
                .object_fit(ResizeMode::Stretch),
            );
        element = element
            .child(disc)
            .child(
                view()
                    .absolute()
                    .inset_0()
                    .items_center()
                    .justify_center()
                    .gap(px(2.))
                    .style_when(
                        move || cell.with(|cell| cell.candidate.is_none()),
                        |style| style.hidden(),
                    )
                    .child(
                        text(move || {
                            cell.with(|cell| {
                                cell.candidate
                                    .as_ref()
                                    .map(|candidate| candidate.label.clone())
                                    .unwrap_or_default()
                            })
                        })
                        .text_size(move || px((cell_side.get() * 0.34).clamp(11., 18.)))
                        .line_height(nsh_core::LineHeight::Multiple(1.4))
                        .font_weight(FontWeight::BOLD)
                        .text_color(move || {
                            color(
                                if cell.with(|cell| {
                                    cell.candidate
                                        .as_ref()
                                        .is_some_and(|candidate| candidate.best)
                                }) {
                                    0x52b885
                                } else {
                                    0xffffff
                                },
                            )
                        }),
                    )
                    .child(
                        search_bar(cell_side, move || {
                            cell.with(|cell| {
                                cell.candidate
                                    .as_ref()
                                    .map_or(0., |candidate| candidate.progress)
                            })
                        })
                        .style_when(
                            move || {
                                state.with(|state| state.complete)
                                    || cell.with(|cell| {
                                        cell.candidate
                                            .as_ref()
                                            .is_none_or(|candidate| candidate.progress >= 1.)
                                    })
                            },
                            |style| style.hidden(),
                        ),
                    ),
            )
            .child(
                view()
                    .rounded_full()
                    .bg(color(0xffffff).opacity(77. / 255.))
                    .w(move || layout_px(cell_side.get() * 0.26))
                    .h(move || layout_px(cell_side.get() * 0.26))
                    .style_when(
                        move || {
                            !state.with(|state| {
                                state.show_legal
                                    && !state.show_waiting
                                    && state.cells[index].legal
                                    && state.cells[index].disc == Disc::Empty
                                    && state.cells[index].candidate.is_none()
                            })
                        },
                        |style| style.hidden(),
                    ),
            )
            .child(
                view()
                    .absolute()
                    .mt(px(2.))
                    .style_when(
                        move || {
                            !state.with(|state| {
                                state.show_waiting
                                    && state.cells[index].legal
                                    && state.cells[index].disc == Disc::Empty
                                    && state.cells[index].candidate.is_none()
                            })
                        },
                        |style| style.hidden(),
                    )
                    .child(search_bar(cell_side, || 0.)),
            )
            .child(
                view()
                    .absolute()
                    .rounded_full()
                    .border_color(color(GOLD))
                    .w(move || layout_px(diameter.get() + 2. * ring.get()))
                    .h(move || layout_px(diameter.get() + 2. * ring.get()))
                    .border(move || stroke_px(ring.get()))
                    .style_when(move || !cell.with(|cell| cell.last), |style| style.hidden()),
            )
            .child(when(
                move || cell.with(|cell| cell.thinking && cell.legal),
                move || thinking(cell_side),
            ))
            .child(when(
                move || cell.with(|cell| cell.legal && !cell.thinking && cell.trail.is_some()),
                move || {
                    assets::icon("bot.svg", 18., AI_COLOR)
                        .w(move || {
                            layout_px(18. - cell.with(|cell| cell.trail.unwrap_or(0)) as f32 * 3.)
                        })
                        .h(move || {
                            layout_px(18. - cell.with(|cell| cell.trail.unwrap_or(0)) as f32 * 3.)
                        })
                        .opacity(move || {
                            0.7 - cell.with(|cell| cell.trail.unwrap_or(0)) as f32 * 0.15
                        })
                },
            ));
        cells = cells.child(element);
    }
    board = board.child(cells);
    for (index, (column, row)) in ["A", "B", "C", "D", "E", "F", "G", "H"]
        .into_iter()
        .zip(["1", "2", "3", "4", "5", "6", "7", "8"])
        .enumerate()
    {
        board = board
            .child(
                board_label()
                    .left(move || layout_px(rim.get() + index as f32 * cell_side.get()))
                    .top(move || layout_px(rim.get() - cell_side.get() * 0.65))
                    .w(move || layout_px(cell_side.get()))
                    .h(move || layout_px(cell_side.get() * 0.3))
                    .child(
                        text(column)
                            .font_family("Noto Sans")
                            .text_size(move || px(cell_side.get() * 0.3))
                            .line_height(move || px(cell_side.get() * 0.3)),
                    ),
            )
            .child(
                board_label()
                    .left(move || layout_px(rim.get() - cell_side.get() * 0.65))
                    .top(move || layout_px(rim.get() + index as f32 * cell_side.get()))
                    .w(move || layout_px(cell_side.get() * 0.3))
                    .h(move || layout_px(cell_side.get()))
                    .child(
                        text(row)
                            .font_family("Noto Sans")
                            .text_size(move || px(cell_side.get() * 0.3))
                            .line_height(move || px(cell_side.get())),
                    ),
            );
    }
    board
}

pub(super) fn board_key(ui: Ui, key: &str) -> bool {
    let index = ui.read(|app| app.board_cursor);
    if let Some(next) = move_board_cursor(index, key) {
        ui.update(|app| app.board_cursor = next);
        true
    } else if matches!(key, "enter" | "space") {
        let state = ui.read(presentation);
        if state.playable && state.cells[index].legal {
            ui.action(Action::Cell(index));
        } else if state.ai_waiting {
            ui.update(ReversiApp::show_ai_paused_hint);
        }
        true
    } else {
        false
    }
}

fn board_label() -> View {
    view()
        .absolute()
        .items_center()
        .justify_center()
        .text_color(color(0xa0a0b0))
}

fn thinking(cell_side: Memo<f32>) -> View {
    let phase = animated_value(0.);
    let wave = animated_value(0.);
    let ripple = Animation::timing(&phase, 1., Timing::new(1500., Easing::Linear)).looped(None);
    let wobble = Animation::keyframes(
        &wave,
        [
            Keyframe::at(0., 0.).ease(Easing::Curve(Curve::Sin, EaseMode::Out)),
            Keyframe::at(0.25, 1.).ease(Easing::Curve(Curve::Sin, EaseMode::In)),
            Keyframe::at(0.5, 0.).ease(Easing::Curve(Curve::Sin, EaseMode::Out)),
            Keyframe::at(0.75, -1.).ease(Easing::Curve(Curve::Sin, EaseMode::In)),
            Keyframe::at(1., 0.),
        ],
        KeyframeTiming::new(2000.),
    )
    .looped(None);
    let a = ripple.clone();
    let b = wobble.clone();
    effect(move || {
        if reduce_motion() {
            a.stop();
            b.stop();
            phase.set(0.);
            wave.set(0.);
        } else {
            a.start();
            b.start();
        }
    });
    cleanup(move || {
        ripple.stop();
        wobble.stop();
    });
    let mut layer = view()
        .absolute()
        .inset_0()
        .items_center()
        .justify_center()
        .pointer_events(PointerEvents::None);
    for index in 0..3 {
        let shifted = phase.add(index as f32 * 0.2).modulo(1.);
        let edge = shifted.mul(0.6).add(0.5);
        let origin = edge.mul(-0.5).add(0.5);
        let opacity = shifted.mul(-0.7).add(0.7);
        layer = layer.child(
            canvas(move || {
                let side = cell_side.get();
                let mut painter = Painter::new()
                    .save()
                    .scale(side, side)
                    .stroke_color(color(AI_COLOR))
                    .line_width(px(2. / side))
                    .global_alpha(&opacity)
                    .begin_path();
                let start = painter.num(&origin);
                let length = painter.num(&edge);
                let radius = painter.num(2. / side);
                painter
                    .op(CanvasOp::Path(PathOp::RoundRect(
                        [start, start, length, length],
                        [radius; 4],
                    )))
                    .stroke()
                    .restore()
            })
            .absolute()
            .inset_0()
            .size_full(),
        );
    }
    let rotation = wave.mul(10.);
    let scale = wave.interpolate(&[-1., 0., 1.], &[1.05, 1., 1.05]);
    layer.child(
        assets::icon("bot.svg", 22., AI_COLOR)
            .bind_rotate(&rotation)
            .bind_scale(&scale),
    )
}

fn candidate_progress(candidate: &Progress) -> f32 {
    if candidate.is_endgame {
        (candidate.probability as f32 / 100.).clamp(0., 1.)
    } else if candidate.target_depth == 0 {
        1.
    } else {
        (candidate.depth as f32 / candidate.target_depth as f32).min(1.)
    }
}

fn search_bar(cell_side: Memo<f32>, progress: impl Fn() -> f32 + 'static) -> View {
    view()
        .w(move || px((cell_side.get() - 8.).max(0.) * 0.7))
        .h(px(4.))
        .rounded_full()
        .overflow_hidden()
        .bg(color(0x000000).opacity(102. / 255.))
        .child(
            view()
                .h_full()
                .w(move || px((cell_side.get() - 8.).max(0.) * 0.7 * progress()))
                .rounded_full()
                .bg(color(BLUE)),
        )
}

impl ReversiApp {
    pub(super) fn show_ai_paused_hint(&mut self) {
        if let Some(label) = self.ai_resume_label() {
            self.notification.show(
                tr!("waiting_for_ai").to_string(),
                tr!("ai_paused_hint", action = label).to_string(),
                "bot.svg",
                Instant::now() + Duration::from_secs(3),
            );
        }
    }
}
