use std::time::Duration;

use gpui_kit::{prelude::*, *};
use reversi_core::{disc::Disc, square::Square};

use super::assets::icon;
use super::format::{ScorePrecision, display_score, whole_score};
use super::widgets::move_board_cursor;
use super::{AI_COLOR, Action, BLUE, GOLD, ReversiApp};
use crate::engine::Progress;
use crate::session::Activity;

// Orthographic world dimensions from the original Board3DScene and board3d-utils.
const TOTAL_SIZE: f32 = 9.5;
const BOARD_SIZE: f32 = 8.;
const FRAME_WIDTH: f32 = 0.15;
// These are the original materials after lighting and ACES/sRGB conversion.
const BOARD_GRID: u32 = 0x042c1a;

impl ReversiApp {
    pub(super) fn render_board(
        &self,
        side: f32,
        window: &Window,
        cx: &Context<Self>,
    ) -> Stateful<Div> {
        let position = self.session.game().position();
        let solver = self.session.is_solving();
        let human_turn = self.settings.ai_color() != Some(position.turn);
        let legal = position.board.get_moves().bits();
        let show_legal = solver || human_turn;
        let idle = !matches!(
            self.session.activity(),
            Some(Activity::Initializing | Activity::Analysis | Activity::Move)
        );
        let playable = self.modal.is_none() && idle && show_legal;
        // A click on the board while the AI waits explains how to continue.
        let ai_waiting = self.modal.is_none() && self.ai_resume_label().is_some();
        let show_candidates = (solver || (self.settings.hints && human_turn)) && idle;
        let show_waiting = show_candidates
            && !self.session.candidates().is_empty()
            && (!solver || self.settings.solver_multi_pv);
        let mut candidates = [None; 64];
        let mut best_score = None;
        if show_candidates {
            for candidate in self.session.candidates() {
                if let Some(square) = candidate.square.filter(|square| *square != Square::None)
                    && legal & (1_u64 << square.index()) != 0
                {
                    candidates[square.index()] = Some(candidate);
                    let score = whole_score(candidate.score);
                    best_score = Some(best_score.map_or(score, |best: i32| best.max(score)));
                }
            }
        }
        let thinking = if self.session.activity() == Some(Activity::Move) {
            self.session
                .log()
                .iter()
                .rev()
                .find_map(|progress| progress.square)
        } else {
            None
        };
        let mut thinking_trail = [None; 3];
        if let Some(current) = thinking {
            let mut count = 0;
            for square in self
                .session
                .log()
                .iter()
                .rev()
                .filter_map(|progress| progress.square)
            {
                if square != current
                    && square != Square::None
                    && !thinking_trail.contains(&Some(square))
                {
                    thinking_trail[count] = Some(square);
                    count += 1;
                    if count == thinking_trail.len() {
                        break;
                    }
                }
            }
        }
        let last_move = if !solver {
            self.session.game().history[..self.session.game().cursor]
                .iter()
                .enumerate()
                .rev()
                .find(|(_, record)| record.square.is_some())
        } else {
            None
        };
        // Navigation pauses normal play. Only the live end of a running game
        // animates; the same animation id survives search-progress re-renders.
        let animate = !solver
            && !self.session.is_paused()
            && self.session.game().cursor == self.session.game().history.len();
        let previous = last_move.filter(|_| animate).map(|(index, _)| {
            index
                .checked_sub(1)
                .map_or(self.session.game().initial, |index| {
                    self.session.game().history[index].position
                })
        });

        let cell_side = side / TOTAL_SIZE;
        let board_side = cell_side * BOARD_SIZE;
        let rim = (side - board_side) / 2.;
        let frame_width = cell_side * FRAME_WIDTH;
        // Layout rounds edges to device pixels, so only a whole-pixel width
        // renders every line equally thick.
        let scale = window.scale_factor();
        let groove_width = (cell_side * 0.03 * scale).round().max(1.) / scale;
        // For the same reason the last-move ring stays concentric with its
        // disc only when both diameters are whole pixels of equal parity.
        let disc_diameter = (cell_side * 0.84 * scale).round() / scale;
        let ring_width = (cell_side * 0.05 * scale).round() / scale;
        // PNGs are reusable, color-managed material bakes of the original scene:
        // felt fibers/normal map and the beveled CylinderGeometry disc materials.
        // GPUI caches them by asset path; no texture work occurs during rendering.
        let mut cells = div()
            .absolute()
            .left(px(rim))
            .top(px(rim))
            .size(px(board_side))
            .child(
                img("board/felt.png")
                    .absolute()
                    .inset_0()
                    .size_full()
                    .object_fit(ObjectFit::Fill),
            );
        for line in 0..=8 {
            let offset = line as f32 * cell_side - groove_width / 2.;
            cells = cells
                .child(
                    div()
                        .absolute()
                        .left(px(offset))
                        .top_0()
                        .w(px(groove_width))
                        .h(px(board_side))
                        .bg(rgb(BOARD_GRID)),
                )
                .child(
                    div()
                        .absolute()
                        .left_0()
                        .top(px(offset))
                        .w(px(board_side))
                        .h(px(groove_width))
                        .bg(rgb(BOARD_GRID)),
                );
        }
        for row in [2., 6.] {
            for column in [2., 6.] {
                cells = cells.child(
                    div()
                        .absolute()
                        .left(px((column - 0.06) * cell_side))
                        .top(px((row - 0.06) * cell_side))
                        .size(px(cell_side * 0.12))
                        .rounded_full()
                        .bg(rgb(0x1a1a1a)),
                );
            }
        }
        for (index, candidate) in candidates.into_iter().enumerate() {
            let square = Square::from_u8(index as u8).expect("board cell index");
            let disc = position.cell(index);
            let is_legal = legal & (1_u64 << index) != 0;
            let clickable = playable && is_legal;
            let is_best =
                candidate.is_some_and(|candidate| Some(whole_score(candidate.score)) == best_score);
            let mut cell = div()
                .id(("board-cell", index))
                .role(Role::Cell)
                .aria_label(format!(
                    "{} {} {}",
                    square.to_string().to_uppercase(),
                    if disc == Disc::Empty {
                        tr!("empty")
                    } else {
                        self.color(disc)
                    },
                    if is_legal {
                        tr!("legal_move")
                    } else {
                        "".into()
                    }
                ))
                .aria_row_index(index / 8)
                .aria_column_index(index % 8)
                .when(
                    self.board_focus.is_focused(window) && self.board_cursor == index,
                    |cell| cell.aria_active_descendant(),
                )
                .absolute()
                .left(px((index % 8) as f32 * cell_side))
                .top(px((index / 8) as f32 * cell_side))
                .size(px(cell_side))
                .flex()
                .items_center()
                .justify_center()
                .when(clickable, |cell| {
                    cell.cursor_pointer()
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.board_cursor = index;
                            this.board_focus.focus(window, cx);
                            this.action(Action::Cell(index), window, cx);
                        }))
                });

            if disc != Disc::Empty {
                // The material asset includes .02 world units of transparent
                // padding around the original radius-.4 beveled disc.
                let stone = board_disc(disc_diameter);
                if let Some(before) = previous
                    .map(|position| position.cell(index))
                    .filter(|before| *before != disc)
                {
                    let move_id = self.session.game().cursor * 64 + index;
                    cell = cell.child(stone.with_animation(
                        ("disc-move", move_id),
                        Animation::new(Duration::from_millis(400)),
                        move |stone, delta| {
                            if before == Disc::Empty {
                                // A top-down orthographic drop keeps the disc's
                                // apparent size; its falling shadow settles below.
                                let height = (1. - delta).powi(3) * cell_side * 0.12;
                                disc_finish(stone, disc, cell_side, height)
                            } else {
                                let height = (std::f32::consts::PI * delta).cos().abs().max(0.1);
                                disc_finish(
                                    stone.h(px(disc_diameter * height)),
                                    if delta < 0.5 { before } else { disc },
                                    cell_side,
                                    0.,
                                )
                            }
                        },
                    ));
                } else {
                    cell = cell.child(disc_finish(stone, disc, cell_side, 0.));
                }
            } else if let Some(candidate) = candidate {
                let label = display_score(candidate.score, ScorePrecision::Whole);
                let progress = candidate_progress(candidate);
                cell = cell.child(
                    div()
                        .absolute()
                        .inset_0()
                        .flex()
                        .flex_col()
                        .items_center()
                        .justify_center()
                        .gap(px(2.))
                        .text_size(px(18.))
                        .line_height(px(25.2))
                        .font_weight(FontWeight::BOLD)
                        .text_color(rgb(if is_best { 0x52b885 } else { 0xffffff }))
                        .child(label)
                        .when(
                            progress < 1. && !self.session.candidates_complete(),
                            |overlay| overlay.child(search_bar(cell_side, progress)),
                        ),
                );
            } else {
                if show_legal && is_legal {
                    cell = cell.child(
                        div()
                            .size(px(cell_side * 0.26))
                            .rounded_full()
                            .bg(rgba(0xffffff4d)),
                    );
                }
                if show_waiting && is_legal {
                    cell = cell.child(div().absolute().mt(px(2.)).child(search_bar(cell_side, 0.)));
                }
            }
            if last_move.is_some_and(|(_, record)| record.square == Some(square)) {
                cell = cell.child(
                    div()
                        .absolute()
                        .size(px(disc_diameter + 2. * ring_width))
                        .rounded_full()
                        .border(px(ring_width))
                        .border_color(rgb(GOLD)),
                );
            }
            if thinking == Some(square) && is_legal {
                for ripple in 0..3 {
                    cell = cell.child(
                        div()
                            .absolute()
                            .rounded(px(2.))
                            .border_2()
                            .border_color(rgb(AI_COLOR))
                            .with_animation(
                                ("thinking-ripple", index * 3 + ripple),
                                Animation::new(Duration::from_millis(1500)).repeat(),
                                move |ring, delta| {
                                    let phase = (delta + ripple as f32 * 0.2) % 1.;
                                    ring.size(px(cell_side * (0.5 + phase * 0.6)))
                                        .opacity(0.7 * (1. - phase))
                                },
                            ),
                    );
                }
                cell = cell.child(icon("bot.svg", 22., AI_COLOR).with_animation(
                    ("thinking-bot", index),
                    Animation::new(Duration::from_millis(2000)).repeat(),
                    |bot, delta| {
                        let wave = (delta * std::f32::consts::TAU).sin();
                        let scale = 1. + wave.abs() * 0.05;
                        bot.with_transformation(
                            Transformation::rotate(radians(wave * 10_f32.to_radians()))
                                .with_scaling(size(scale, scale)),
                        )
                    },
                ));
            } else if is_legal
                && let Some(trail_index) = thinking_trail
                    .iter()
                    .position(|previous| *previous == Some(square))
            {
                cell = cell.child(
                    icon("bot.svg", 18. - trail_index as f32 * 3., AI_COLOR)
                        .opacity(0.7 - trail_index as f32 * 0.15),
                );
            }
            cells = cells.child(cell);
        }
        let cells = cells.id("board-cells").when(ai_waiting, |cells| {
            cells.on_click(cx.listener(|this, _, _, cx| {
                this.show_ai_paused_hint();
                cx.notify();
            }))
        });

        let mut board = div()
            .id("board-grid")
            .role(Role::Grid)
            .aria_label(tr!("board"))
            .aria_description(tr!("board_keyboard_help"))
            .aria_row_count(8)
            .aria_column_count(8)
            .track_focus(&self.board_focus.clone().tab_stop(self.modal.is_none()))
            .on_key_down(cx.listener(move |this, event: &KeyDownEvent, window, cx| {
                if let Some(index) = move_board_cursor(this.board_cursor, &event.keystroke.key) {
                    this.board_cursor = index;
                    cx.notify();
                    cx.stop_propagation();
                } else if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                    if playable && legal & (1_u64 << this.board_cursor) != 0 {
                        this.action(Action::Cell(this.board_cursor), window, cx);
                    } else if ai_waiting {
                        this.show_ai_paused_hint();
                        cx.notify();
                    }
                    cx.stop_propagation();
                }
            }))
            .relative()
            .size(px(side))
            .flex_shrink_0()
            .child(
                div()
                    .absolute()
                    .left(px(rim - frame_width))
                    .top(px(rim - frame_width))
                    .size(px(board_side + frame_width * 2.))
                    .bg(rgb(0x010102))
                    .border(px(cell_side * 0.01))
                    .border_color(rgb(0x040407)),
            )
            .child(cells);
        let columns = ["A", "B", "C", "D", "E", "F", "G", "H"];
        let rows = ["1", "2", "3", "4", "5", "6", "7", "8"];
        let label_size = cell_side * 0.3;
        let label_edge = rim - cell_side * 0.5 - label_size / 2.;
        for index in 0..8 {
            let offset = rim + index as f32 * cell_side;
            board = board
                .child(
                    board_label(columns[index], cell_side, label_size, label_size)
                        .font_family("Noto Sans")
                        .left(px(offset))
                        .top(px(label_edge)),
                )
                .child(
                    board_label(rows[index], label_size, cell_side, label_size)
                        .font_family("Noto Sans")
                        .left(px(label_edge))
                        .top(px(offset)),
                );
        }
        board
    }
}

pub(super) fn board_label(label: &'static str, width: f32, height: f32, font_size: f32) -> Div {
    div()
        .absolute()
        .w(px(width))
        .h(px(height))
        .flex()
        .items_center()
        .justify_center()
        .text_size(px(font_size))
        .line_height(px(height))
        .font_weight(FontWeight::NORMAL)
        .text_color(rgb(0xa0a0b0))
        .child(label)
}

fn board_disc(diameter: f32) -> Div {
    div()
        .relative()
        .size(px(diameter))
        .flex_shrink_0()
        .rounded_full()
}

fn disc_shadow(cell_side: f32, height: f32) -> BoxShadow {
    BoxShadow {
        inset: false,
        color: rgba(0x0000004d).into(),
        offset: point(
            px(-cell_side * 0.04 - height),
            px(-cell_side * 0.03 - height),
        ),
        blur_radius: px(cell_side * 0.035 + height * 0.4),
        spread_radius: px(-cell_side * 0.025),
    }
}

fn disc_finish(stone: Div, disc: Disc, cell_side: f32, drop_height: f32) -> Div {
    stone
        .shadow(vec![disc_shadow(cell_side, drop_height)])
        .child(
            img(if disc == Disc::Black {
                "board/disc-black.png"
            } else {
                "board/disc-white.png"
            })
            // Resolve both dimensions against the animated frame, not the
            // image's intrinsic square aspect ratio in normal flow.
            .absolute()
            .inset_0()
            .size_full()
            .object_fit(ObjectFit::Fill),
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

fn search_bar(cell_side: f32, progress: f32) -> Div {
    let width = (cell_side - 8.).max(0.) * 0.7;
    div()
        .w(px(width))
        .h(px(4.))
        .rounded_full()
        .overflow_hidden()
        .bg(rgba(0x00000066))
        .child(
            div()
                .h_full()
                .w(px(width * progress))
                .rounded_full()
                .bg(rgb(BLUE)),
        )
}

impl ReversiApp {
    /// Tells the player that the AI waits for the start/resume button.
    pub(super) fn show_ai_paused_hint(&mut self) {
        if let Some(label) = self.ai_resume_label() {
            self.notification.show(
                tr!("waiting_for_ai").into(),
                tr!("ai_paused_hint", action = label).into(),
                "bot.svg",
                std::time::Instant::now() + Duration::from_secs(3),
            );
        }
    }
}
