use gpui_kit::{prelude::*, *};
use reversi_core::disc::Disc;

use super::format::{ScorePrecision, display_score};
use super::widgets::shadow;
use super::{ACCENT, Action, ReversiApp, SECONDARY};
use crate::engine::format_pv;
use crate::session::Activity;

impl ReversiApp {
    pub(super) fn render_chart(&self, cx: &Context<Self>) -> Div {
        let scores: Vec<_> = self
            .session
            .game()
            .history
            .iter()
            .enumerate()
            .filter(|(_, record)| record.square.is_some())
            .enumerate()
            .map(|(move_index, (index, record))| {
                let score = self
                    .session
                    .analyses()
                    .get(&index)
                    .map(|value| value.played_score)
                    .or(record.score)
                    .map(|score| {
                        if record.player == Disc::White {
                            -score
                        } else {
                            score
                        }
                    });
                (move_index + 1, index, record, score)
            })
            .collect();
        let (low, high) = y_domain(scores.iter().filter_map(|(_, _, _, score)| *score));
        // Vertical position of a score as a fraction of the plot height, top = 0.
        let y = move |score: f32| (high - score) / (high - low);
        let zero = y(0.);
        let plot_height = 140.;
        let ticks = y_ticks(low, high);
        let grid_ticks = ticks.clone();
        let points: Vec<_> = scores
            .iter()
            .filter_map(|(number, _, _, score)| score.map(|score| (move_x(*number), y(score))))
            .collect();
        let cursor = if self.session.game().cursor < self.session.game().history.len() {
            scores
                .iter()
                .rev()
                .find(|(_, index, _, _)| *index < self.session.game().cursor)
                .map(|(number, _, _, _)| move_x(*number))
        } else {
            None
        };
        let hovered = scores
            .iter()
            .find(|(_, index, _, _)| Some(*index) == self.analysis_view.hover);
        let hover_x = hovered.map(|(number, _, _, _)| move_x(*number));
        let view = cx.entity().downgrade();
        let mut plot = div()
            .id("evaluation-plot")
            .absolute()
            .left(px(52.))
            .right(px(18.))
            .top(px(13.))
            .h(px(plot_height))
            .when(
                !matches!(
                    self.session.activity(),
                    Some(Activity::Move | Activity::Analysis)
                ),
                |plot| plot.cursor_pointer(),
            )
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                if let Some(bounds) = this.analysis_view.bounds {
                    let number = ((f32::from(event.position.x - bounds.origin.x)
                        / f32::from(bounds.size.width).max(1.)
                        * MOVE_SPAN)
                        .round() as usize)
                        .saturating_add(1);
                    this.analysis_view.hover = this
                        .session
                        .game()
                        .history
                        .iter()
                        .enumerate()
                        .filter(|(_, record)| record.square.is_some())
                        .take(number)
                        .last()
                        .map(|(index, _)| index);
                    this.analysis_view.pointer = Some(event.position);
                    cx.notify();
                }
            }))
            .on_hover(cx.listener(|this, hovered, _, cx| {
                if !hovered {
                    this.analysis_view.hover = None;
                    this.analysis_view.pointer = None;
                    cx.notify();
                }
            }))
            .on_click(cx.listener(|this, _, window, cx| {
                if !matches!(
                    this.session.activity(),
                    Some(Activity::Move | Activity::Analysis)
                ) && let Some(index) = this.analysis_view.hover
                {
                    this.action(Action::Navigate(index + 1), window, cx);
                }
            }))
            .child(
                div()
                    .absolute()
                    .top_0()
                    .w_full()
                    .h(relative(zero))
                    .bg(rgba(0x00000017)),
            )
            .child(
                div()
                    .absolute()
                    .top(relative(zero))
                    .bottom_0()
                    .w_full()
                    .bg(rgba(0xffffff0a)),
            )
            .child(
                canvas(
                    move |bounds, _, cx| {
                        let _ = view.update(cx, |this, _| this.analysis_view.bounds = Some(bounds));
                    },
                    move |bounds, _, window, _| {
                        for tick in grid_ticks {
                            paint_chart_dash(
                                window,
                                bounds,
                                true,
                                y(tick),
                                if tick == 0. { 4. } else { 3. },
                                if tick == 0. {
                                    rgba(0xffffff4d)
                                } else {
                                    rgba(0xffffff1a)
                                },
                            );
                        }
                        if let Some(cursor) = cursor {
                            paint_chart_dash(window, bounds, false, cursor, 4., rgba(0xffffff80));
                        }
                        if let Some(hover_x) = hover_x {
                            paint_chart_dash(window, bounds, false, hover_x, 3., rgba(0xffffff33));
                        }
                        if points.len() > 1 {
                            let at = |x: f32, y: f32| {
                                point(
                                    bounds.origin.x + bounds.size.width * x,
                                    bounds.origin.y + bounds.size.height * y,
                                )
                            };
                            let mut path = PathBuilder::stroke(px(2.));
                            path.move_to(at(points[0].0, points[0].1));
                            for (index, pair) in points.windows(2).enumerate() {
                                let (x0, y0) = pair[0];
                                let (x1, y1) = pair[1];
                                let third = (x1 - x0) / 3.;
                                path.cubic_bezier_to(
                                    at(x1, y1),
                                    at(x0 + third, y0 + third * monotone_slope(&points, index)),
                                    at(x1 - third, y1 - third * monotone_slope(&points, index + 1)),
                                );
                            }
                            if let Ok(path) = path.build() {
                                window.paint_path(path, rgb(ACCENT));
                            }
                        }
                    },
                )
                .absolute()
                .size_full(),
            );
        for tick in ticks {
            plot = plot.child(
                div()
                    .absolute()
                    .top(relative(y(tick)))
                    .right_full()
                    .mr_1()
                    .mt(px(-7.3333335))
                    .text_size(px(12.))
                    .line_height(relative(1.5))
                    .text_color(rgba(0xffffff80))
                    .child(format!("{tick:.0}")),
            );
        }
        for number in [1, 10, 20, 30, 40, 50, 60] {
            plot = plot.child(
                div()
                    .absolute()
                    .left(relative(move_x(number)))
                    .top_full()
                    .mt_2()
                    .w(px(20.))
                    .ml(px(-10.))
                    .text_center()
                    .text_size(px(12.))
                    .line_height(relative(1.5))
                    .text_color(rgba(0xffffff80))
                    .child(number.to_string()),
            );
        }
        for (number, index, _, score) in &scores {
            if let Some(score) = score {
                let loss = self
                    .session
                    .analyses()
                    .get(index)
                    .map_or(0., |analysis| analysis.score_loss);
                let active = self.analysis_view.hover == Some(*index);
                let flagged = loss_color(loss);
                let radius = if active || flagged.is_some() { 5. } else { 3. };
                let stroke = if active {
                    2.
                } else if flagged.is_some() {
                    1.5
                } else {
                    1.
                };
                let outer_radius = radius + stroke * 0.5;
                let color = if active {
                    ACCENT
                } else {
                    flagged.unwrap_or(ACCENT)
                };
                plot = plot.child(
                    div()
                        .absolute()
                        .left(relative(move_x(*number)))
                        .top(px(y(*score) * plot_height))
                        .ml(px(-outer_radius))
                        .mt(px(-outer_radius))
                        .size(px(outer_radius * 2.))
                        .rounded_full()
                        .border(px(stroke))
                        .border_color(if active {
                            rgba(0xffffff80)
                        } else if flagged.is_some() {
                            rgba(0xffffff66)
                        } else {
                            rgba(0xffffff33)
                        })
                        .bg(rgb(color)),
                );
            }
        }
        if let Some((number, index, record, score)) = hovered
            && let Some(pointer) = self.analysis_view.pointer
            && score.is_some()
        {
            let analysis = self.session.analyses().get(index);
            let tooltip = div()
                .px_3()
                .py_2()
                .rounded(px(10.))
                .bg(rgb(0x2a2a2f))
                .border_1()
                .border_color(rgba(0xffffff33))
                .text_size(px(12.))
                .line_height(relative(1.4285714))
                .text_color(rgba(0xffffffeb))
                .shadow(vec![
                    shadow(rgba(0x0000008c), 8., 16., -4.),
                    shadow(rgba(0xffffff14), 0., 0., 1.),
                ])
                .child(
                    div()
                        .mb_1()
                        .text_size(px(12.))
                        .line_height(relative(1.5))
                        .text_color(rgb(SECONDARY))
                        .child(format!("#{number}")),
                )
                .when_some(*score, |tooltip, score| {
                    tooltip.child(div().font_weight(FontWeight::SEMIBOLD).child(format!(
                        "{}: {}",
                        record.square.unwrap(),
                        display_score(score, ScorePrecision::Tenth)
                    )))
                })
                .when_some(analysis, |tooltip, analysis| {
                    tooltip.child(
                        div()
                            .mt_1()
                            .flex()
                            .flex_col()
                            .gap(px(2.))
                            .text_size(px(12.))
                            .line_height(relative(1.5))
                            .text_color(rgb(SECONDARY))
                            .when(Some(analysis.best_move) != record.square, |details| {
                                details.child(format!(
                                    "{} {} ({})",
                                    tr!("best"),
                                    analysis.best_move,
                                    display_score(
                                        if record.player == Disc::White {
                                            -analysis.best_score
                                        } else {
                                            analysis.best_score
                                        },
                                        ScorePrecision::Raw
                                    )
                                ))
                            })
                            .when(!analysis.pv.is_empty(), |details| {
                                details.child(format!(
                                    "{}: {}",
                                    tr!("pv_line"),
                                    format_pv(&analysis.pv)
                                ))
                            })
                            .when_some(loss_color(analysis.score_loss), |details, color| {
                                details.child(div().text_color(rgb(color)).child(format!(
                                    "{} {:.1}",
                                    tr!("loss"),
                                    analysis.score_loss
                                )))
                            }),
                    )
                });
            plot = plot.child(deferred(
                gpui_kit::anchored()
                    .position(pointer + point(px(10.), px(10.)))
                    .position_mode(gpui_kit::AnchoredPositionMode::Window)
                    .child(tooltip),
            ));
        }
        div()
            .h(px(198.))
            .w_full()
            .flex_shrink_0()
            .relative()
            .rounded(px(10.))
            .border_1()
            .border_color(rgba(0xffffff1a))
            .bg(rgba(0xffffff0d))
            .child(
                div()
                    .absolute()
                    .left(px(12.))
                    .top(px(13.))
                    .w_2()
                    .h(px(plot_height * zero))
                    .rounded(px(2.))
                    .bg(rgb(0x1f1f22))
                    .border(px(0.5))
                    .border_color(rgba(0xffffff4d)),
            )
            .child(
                div()
                    .absolute()
                    .left(px(12.))
                    .top(px(13. + plot_height * zero))
                    .w_2()
                    .h(px(plot_height * (1. - zero)))
                    .rounded(px(2.))
                    .bg(rgb(0xd4d4d4))
                    .border(px(0.5))
                    .border_color(rgba(0xffffff26)),
            )
            .child(plot)
    }
}

/// The x axis spans moves 1 to 60.
const MOVE_SPAN: f32 = 59.;

/// Horizontal position of a move number as a fraction of the plot width.
fn move_x(number: usize) -> f32 {
    (number - 1) as f32 / MOVE_SPAN
}

/// Y-axis range in multiples of 4: padded around the scores, always including
/// zero, giving the losing side at least 30% of the dominant side's height,
/// and clamped to ±64. Small or missing scores use ±8.
fn y_domain(scores: impl Iterator<Item = f32>) -> (f32, f32) {
    let (low, high) = scores.fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), score| {
        (low.min(score), high.max(score))
    });
    if !low.is_finite() || low.abs().max(high.abs()) <= 4. {
        return (-8., 8.);
    }
    let padding = (((high - low).abs() * 0.08 / 4.).ceil() * 4.).max(4.);
    let mut low = (((low - padding) / 4.).floor() * 4.).min(0.);
    let mut high = (((high + padding) / 4.).ceil() * 4.).max(0.);
    let opposite = ((low.abs().max(high) * 0.3 / 4.).ceil() * 4.).max(8.);
    if low.abs() > high {
        high = high.max(opposite);
    } else if high > low.abs() {
        low = low.min(-opposite);
    }
    (low.max(-64.), high.min(64.))
}

/// Marker color for a move that lost more than 2 (yellow) or 6 (red) discs.
fn loss_color(loss: f32) -> Option<u32> {
    if loss > 6. {
        Some(0xef4444)
    } else if loss > 2. {
        Some(0xeab308)
    } else {
        None
    }
}

/// Y-axis ticks: multiples of the smallest step in 4, 8, 16, 32 that splits the
/// range into at most six parts, so labels never overlap in the 140 px plot.
fn y_ticks(low: f32, high: f32) -> Vec<f32> {
    let step = [4., 8., 16., 32.]
        .into_iter()
        .find(|step| (high - low) / step <= 6.)
        .unwrap_or(32.);
    let mut ticks = Vec::new();
    let mut tick = (low / step).ceil() * step;
    while tick <= high {
        ticks.push(tick);
        tick += step;
    }
    ticks
}

// Recharts' monotone line uses d3-shape's Steffen tangents, including one-sided endpoints.
fn monotone_slope(points: &[(f32, f32)], index: usize) -> f32 {
    if points.len() == 2 {
        return (points[1].1 - points[0].1) / (points[1].0 - points[0].0);
    }
    if index == 0 {
        let secant = (points[1].1 - points[0].1) / (points[1].0 - points[0].0);
        return (3. * secant - monotone_slope(points, 1)) * 0.5;
    }
    if index + 1 == points.len() {
        let secant =
            (points[index].1 - points[index - 1].1) / (points[index].0 - points[index - 1].0);
        return (3. * secant - monotone_slope(points, index - 1)) * 0.5;
    }
    let h0 = points[index].0 - points[index - 1].0;
    let h1 = points[index + 1].0 - points[index].0;
    let s0 = (points[index].1 - points[index - 1].1) / h0;
    let s1 = (points[index + 1].1 - points[index].1) / h1;
    let weighted = (s0 * h1 + s1 * h0) / (h0 + h1);
    let sign = |slope: f32| if slope < 0. { -1. } else { 1. };
    (sign(s0) + sign(s1)) * s0.abs().min(s1.abs()).min(0.5 * weighted.abs())
}

fn paint_chart_dash(
    window: &mut Window,
    bounds: Bounds<Pixels>,
    horizontal: bool,
    offset: f32,
    dash: f32,
    color: gpui_kit::Rgba,
) {
    let (start, end) = if horizontal {
        let y = bounds.origin.y + bounds.size.height * offset;
        (point(bounds.left(), y), point(bounds.right(), y))
    } else {
        let x = bounds.origin.x + bounds.size.width * offset;
        (point(x, bounds.top()), point(x, bounds.bottom()))
    };
    let mut path = PathBuilder::stroke(px(1.)).dash_array(&[px(dash), px(dash)]);
    path.move_to(start);
    path.line_to(end);
    if let Ok(path) = path.build() {
        window.paint_path(path, color);
    }
}

#[cfg(test)]
mod tests {
    use super::{y_domain, y_ticks};
    #[test]
    fn y_domain_pads_includes_zero_and_reserves_the_losing_side() {
        assert_eq!(y_domain([].into_iter()), (-8., 8.));
        assert_eq!(y_domain([0.1, 3.].into_iter()), (-8., 8.));
        assert_eq!(y_domain([0., -20., -60.].into_iter()), (-64., 24.));
        assert_eq!(y_domain([2., 10.].into_iter()), (-8., 16.));
    }
    #[test]
    fn y_ticks_stay_sparse_and_include_zero() {
        assert_eq!(y_ticks(-8., 8.), [-8., -4., 0., 4., 8.]);
        assert_eq!(y_ticks(-64., 24.), [-64., -48., -32., -16., 0., 16.]);
        assert_eq!(y_ticks(-12., 36.), [-8., 0., 8., 16., 24., 32.]);
        assert_eq!(y_ticks(-64., 64.), [-64., -32., 0., 32., 64.]);
    }
}
