use nsh_core::canvas::{CanvasTextAlign, CanvasTextBaseline};
use nsh_core::prelude::*;
use reversi_core::player_color::PlayerColor;
use reversi_core::score::Score;

use super::analysis::shadow;
use super::format::{ScorePrecision, display_score, tenth_score};
use super::{ACCENT, Action, SECONDARY, Ui, color, rgba};
use crate::engine::format_pv;
use crate::session::Activity;

#[derive(Clone)]
struct ChartPoint {
    number: usize,
    index: usize,
    score: Option<Score>,
    loss: f32,
}

fn scores(ui: Ui) -> Vec<ChartPoint> {
    ui.read(|app| {
        app.session
            .game()
            .history
            .iter()
            .enumerate()
            .filter(|(_, record)| record.square.is_some())
            .enumerate()
            .map(|(move_index, (index, record))| {
                let score = app
                    .session
                    .analyses()
                    .get(&index)
                    .map(|a| a.played_score)
                    .or(record.score)
                    .map(|score| {
                        if record.player == PlayerColor::White {
                            -score
                        } else {
                            score
                        }
                    });
                let loss = app
                    .session
                    .analyses()
                    .get(&index)
                    .map_or(0., |a| tenth_score(a.score_loss));
                ChartPoint {
                    number: move_index + 1,
                    index,
                    score,
                    loss,
                }
            })
            .collect()
    })
}

const CHART_FONT: &str = if cfg!(target_os = "windows") {
    "12px Segoe UI"
} else {
    "12px sans-serif"
};

pub(super) fn render_chart(ui: Ui) -> impl IntoElement {
    let chart_size = signal(size(px(0.), px(0.)));
    let hovered = move || ui.read(|app| app.analysis_view.hover);
    view()
        .flex_1()
        .min_h_0()
        .w_full()
        .relative()
        .rounded(px(10.))
        .border_1()
        .border_color(rgba(0xffffff1a))
        .bg(rgba(0xffffff0d))
        .on_layout(move |bounds: &Bounds<Points>| chart_size.set(bounds.size))
        .on_mouse_move(move |event: &MouseEvent| {
            let plot_width = (chart_size.get_untracked().width.0 - 70.).max(1.);
            let raw = ((event.position.x.0 - 52.) / plot_width * MOVE_SPAN).round();
            let next_hover = if event.position.x.0 < 44. || raw < 0. {
                None
            } else {
                ui.app.with(|app| {
                    app.session
                        .game()
                        .history
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.square.is_some())
                        .nth(raw as usize)
                        .map(|(index, _)| index)
                })
            };
            let next_pointer = next_hover.map(|_| event.page_position);
            if ui.app.with(|app| {
                app.analysis_view.hover != next_hover || app.analysis_view.pointer != next_pointer
            }) {
                ui.update(|app| {
                    app.analysis_view.hover = next_hover;
                    app.analysis_view.pointer = next_pointer;
                });
            }
        })
        .on_mouse_leave(move |_: &MouseEvent| {
            if ui.app.with(|app| {
                app.analysis_view.hover.is_some() || app.analysis_view.pointer.is_some()
            }) {
                ui.update(|app| {
                    app.analysis_view.hover = None;
                    app.analysis_view.pointer = None;
                });
            }
        })
        .on_press(move || {
            let index = ui.read(|app| {
                if matches!(
                    app.session.activity(),
                    Some(Activity::Move | Activity::Analysis)
                ) {
                    None
                } else {
                    app.analysis_view.hover
                }
            });
            if let Some(index) = index {
                ui.action(Action::Navigate(index + 1));
            }
        })
        .child(
            canvas(move || {
                let scores = scores(ui);
                let bounds = chart_size.get();
                let plot_width = (bounds.width.0 - 70.).max(0.);
                let (low, high) = y_domain(
                    scores
                        .iter()
                        .filter_map(|point| point.score.map(Score::to_disc_diff_f32)),
                );
                let y = |score: f32| (high - score) / (high - low);
                let zero = y(0.);
                let plot_height = (bounds.height.0 - 44.).max(40.);
                let at =
                    |x: f32, y: f32| point(px(52. + plot_width * x), px(13. + plot_height * y));
                let mut painter = Painter::new()
                    .fill_color(rgba(0x00000017))
                    .fill_rect(52., 13., plot_width, plot_height * zero)
                    .fill_color(rgba(0xffffff0a))
                    .fill_rect(
                        52.,
                        13. + plot_height * zero,
                        plot_width,
                        plot_height * (1. - zero),
                    );
                let border = super::theme::stroke_px(0.5).0;
                for (top, height, fill, edge) in [
                    (13., plot_height * zero, 0x1f1f22, 0xffffff4d),
                    (
                        13. + plot_height * zero,
                        plot_height * (1. - zero),
                        0xd4d4d4,
                        0xffffff26,
                    ),
                ] {
                    painter = painter
                        .fill_color(color(fill))
                        .begin_path()
                        .round_rect(
                            Bounds::new(point(px(12.), px(top)), size(px(8.), px(height))),
                            px(2.),
                        )
                        .fill()
                        .stroke_color(rgba(edge))
                        .line_width(px(border))
                        .begin_path()
                        .round_rect(
                            Bounds::new(
                                point(px(12. + border / 2.), px(top + border / 2.)),
                                size(px(8. - border), px((height - border).max(0.))),
                            ),
                            px((2. - border / 2.).max(0.)),
                        )
                        .stroke();
                }
                for tick in y_ticks(low, high) {
                    let dash = if tick == 0. { 4. } else { 3. };
                    painter = painter
                        .stroke_color(if tick == 0. {
                            rgba(0xffffff4d)
                        } else {
                            rgba(0xffffff1a)
                        })
                        .line_width(1.)
                        .line_dash(&[px(dash), px(dash)])
                        .begin_path()
                        .move_to(52., 13. + plot_height * y(tick))
                        .line_to(52. + plot_width, 13. + plot_height * y(tick))
                        .stroke()
                        .font(CHART_FONT)
                        .text_align(CanvasTextAlign::Right)
                        .text_baseline(CanvasTextBaseline::Middle)
                        .fill_color(rgba(0xffffff80))
                        .fill_text(&format!("{tick:.0}"), 48., 13. + plot_height * y(tick));
                }
                let cursor = ui.read(|app| {
                    if app.session.game().cursor < app.session.game().history.len() {
                        scores
                            .iter()
                            .rev()
                            .find(|p| p.index < app.session.game().cursor)
                            .map(|p| move_x(p.number))
                    } else {
                        None
                    }
                });
                for (guide, dash, tint) in [
                    (cursor, 4., rgba(0xffffff80)),
                    (
                        scores
                            .iter()
                            .find(|p| Some(p.index) == hovered())
                            .map(|p| move_x(p.number)),
                        3.,
                        rgba(0xffffff33),
                    ),
                ] {
                    if let Some(x) = guide {
                        painter = painter
                            .stroke_color(tint)
                            .line_width(1.)
                            .line_dash(&[px(dash), px(dash)])
                            .begin_path()
                            .move_to(52. + plot_width * x, 13.)
                            .line_to(52. + plot_width * x, 13. + plot_height)
                            .stroke();
                    }
                }
                let points: Vec<_> = scores
                    .iter()
                    .filter_map(|p| p.score.map(|s| (move_x(p.number), y(s.to_disc_diff_f32()))))
                    .collect();
                if points.len() > 1 {
                    let start = at(points[0].0, points[0].1);
                    painter = painter
                        .line_dash(&[])
                        .stroke_color(color(ACCENT))
                        .line_width(2.)
                        .begin_path()
                        .move_to(start.x, start.y);
                    for (index, pair) in points.windows(2).enumerate() {
                        let (x0, y0) = pair[0];
                        let (x1, y1) = pair[1];
                        let third = (x1 - x0) / 3.;
                        painter = painter.bezier_curve_to(
                            at(x0 + third, y0 + third * monotone_slope(&points, index)),
                            at(x1 - third, y1 - third * monotone_slope(&points, index + 1)),
                            at(x1, y1),
                        );
                    }
                    painter = painter.stroke();
                }
                for p in &scores {
                    if let Some(score) = p.score {
                        let active = Some(p.index) == hovered();
                        let flagged = loss_color(p.loss);
                        let radius = if active || flagged.is_some() { 5. } else { 3. };
                        let stroke = if active {
                            2.
                        } else if flagged.is_some() {
                            1.5
                        } else {
                            1.
                        };
                        let center = at(move_x(p.number), y(score.to_disc_diff_f32()));
                        painter = painter
                            .line_dash(&[])
                            .line_width(stroke)
                            .fill_color(color(if active {
                                ACCENT
                            } else {
                                flagged.unwrap_or(ACCENT)
                            }))
                            .stroke_color(if active {
                                rgba(0xffffff80)
                            } else if flagged.is_some() {
                                rgba(0xffffff66)
                            } else {
                                rgba(0xffffff33)
                            })
                            .begin_path()
                            .arc(center.x, center.y, radius, 0., std::f32::consts::TAU)
                            .fill()
                            .stroke();
                    }
                }
                painter = painter
                    .font(CHART_FONT)
                    .text_align(CanvasTextAlign::Center)
                    .text_baseline(CanvasTextBaseline::Top)
                    .fill_color(rgba(0xffffff80));
                for number in [1, 10, 20, 30, 40, 50, 60] {
                    painter = painter.fill_text(
                        &number.to_string(),
                        52. + plot_width * move_x(number),
                        13. + plot_height + 8.,
                    );
                }
                painter.finish()
            })
            .absolute()
            .inset_0()
            .w_full()
            .h_full(),
        )
        .child(when(
            move || {
                ui.read(|app| {
                    app.analysis_view.hover.is_some() && app.analysis_view.pointer.is_some()
                }) && scores(ui)
                    .iter()
                    .any(|p| Some(p.index) == hovered() && p.score.is_some())
            },
            move || {
                overlay()
                    .anchor_point(move || {
                        ui.read(|app| {
                            let pointer = app.analysis_view.pointer.unwrap_or_default();
                            point(pointer.x + px(10.), pointer.y + px(10.))
                        })
                    })
                    .align(nsh_core::overlay::Align::Start)
                    .offset(px(0.))
                    .pointer_events(PointerEvents::None)
                    .child(
                        view()
                            .px_3()
                            .py_2()
                            .rounded(px(10.))
                            .bg(color(0x2a2a2f))
                            .shadow([
                                shadow(rgba(0x0000008c), 8., 16., -4.),
                                shadow(rgba(0xffffff14), 0., 0., 1.),
                            ])
                            .border_1()
                            .border_color(rgba(0xffffff33))
                            .text_size(px(12.))
                            .line_height(nsh_core::LineHeight::Multiple(1.4285714))
                            .text_color(rgba(0xffffffeb))
                            .child(
                                text(move || {
                                    scores(ui)
                                        .iter()
                                        .find(|p| Some(p.index) == hovered())
                                        .map_or_else(String::new, |p| format!("#{}", p.number))
                                })
                                .mb_1()
                                .text_size(px(12.))
                                .line_height(nsh_core::LineHeight::Multiple(1.5))
                                .text_color(color(SECONDARY)),
                            )
                            .child(
                                text(move || {
                                    ui.read(|app| {
                                        let Some(index) = app.analysis_view.hover else {
                                            return String::new();
                                        };
                                        let Some(record) = app.session.game().history.get(index)
                                        else {
                                            return String::new();
                                        };
                                        let score = app
                                            .session
                                            .analyses()
                                            .get(&index)
                                            .map(|a| a.played_score)
                                            .or(record.score)
                                            .map(|s| {
                                                if record.player == PlayerColor::White {
                                                    -s
                                                } else {
                                                    s
                                                }
                                            });
                                        score.map_or_else(String::new, |score| {
                                            format!(
                                                "{}: {}",
                                                record.square.expect("chart only includes moves"),
                                                display_score(score, ScorePrecision::Tenth)
                                            )
                                        })
                                    })
                                })
                                .font_weight(FontWeight::SEMIBOLD),
                            )
                            .child(when(
                                move || {
                                    ui.read(|app| {
                                        app.analysis_view.hover.is_some_and(|index| {
                                            app.session.analyses().contains_key(&index)
                                        })
                                    })
                                },
                                move || {
                                    view()
                                        .mt_1()
                                        .gap(px(2.))
                                        .text_size(px(12.))
                                        .line_height(nsh_core::LineHeight::Multiple(1.5))
                                        .text_color(color(SECONDARY))
                                        .child(when(
                                            move || {
                                                ui.read(|app| {
                                                    let Some(index) = app.analysis_view.hover
                                                    else {
                                                        return false;
                                                    };
                                                    let (Some(analysis), Some(record)) = (
                                                        app.session.analyses().get(&index),
                                                        app.session.game().history.get(index),
                                                    ) else {
                                                        return false;
                                                    };
                                                    Some(analysis.best_move) != record.square
                                                })
                                            },
                                            move || {
                                                text(move || {
                                                    ui.read(|app| {
                                                        let Some(index) = app.analysis_view.hover
                                                        else {
                                                            return String::new();
                                                        };
                                                        let (Some(analysis), Some(record)) = (
                                                            app.session.analyses().get(&index),
                                                            app.session.game().history.get(index),
                                                        ) else {
                                                            return String::new();
                                                        };
                                                        let score = if record.player
                                                            == PlayerColor::White
                                                        {
                                                            -analysis.best_score
                                                        } else {
                                                            analysis.best_score
                                                        };
                                                        format!(
                                                            "{} {} ({})",
                                                            tr!("best"),
                                                            analysis.best_move,
                                                            display_score(
                                                                score,
                                                                ScorePrecision::Tenth
                                                            )
                                                        )
                                                    })
                                                })
                                            },
                                        ))
                                        .child(when(
                                            move || {
                                                ui.read(|app| {
                                                    app.analysis_view
                                                        .hover
                                                        .and_then(|index| {
                                                            app.session.analyses().get(&index)
                                                        })
                                                        .is_some_and(|a| !a.pv.is_empty())
                                                })
                                            },
                                            move || {
                                                text(move || {
                                                    ui.read(|app| {
                                                        app.analysis_view
                                                            .hover
                                                            .and_then(|index| {
                                                                app.session.analyses().get(&index)
                                                            })
                                                            .map_or_else(String::new, |a| {
                                                                format!(
                                                                    "{}: {}",
                                                                    tr!("pv_line"),
                                                                    format_pv(&a.pv)
                                                                )
                                                            })
                                                    })
                                                })
                                            },
                                        ))
                                        .child(when(
                                            move || {
                                                ui.read(|app| {
                                                    app.analysis_view
                                                        .hover
                                                        .and_then(|index| {
                                                            app.session.analyses().get(&index)
                                                        })
                                                        .is_some_and(|a| {
                                                            loss_color(tenth_score(a.score_loss))
                                                                .is_some()
                                                        })
                                                })
                                            },
                                            move || {
                                                text(move || {
                                                    ui.read(|app| {
                                                        app.analysis_view
                                                            .hover
                                                            .and_then(|index| {
                                                                app.session.analyses().get(&index)
                                                            })
                                                            .map_or_else(String::new, |a| {
                                                                format!(
                                                                    "{} {:.1}",
                                                                    tr!("loss"),
                                                                    tenth_score(a.score_loss)
                                                                )
                                                            })
                                                    })
                                                })
                                                .text_color(move || {
                                                    ui.read(|app| {
                                                        color(
                                                            app.analysis_view
                                                                .hover
                                                                .and_then(|index| {
                                                                    app.session
                                                                        .analyses()
                                                                        .get(&index)
                                                                })
                                                                .and_then(|a| {
                                                                    loss_color(tenth_score(
                                                                        a.score_loss,
                                                                    ))
                                                                })
                                                                .unwrap_or(SECONDARY),
                                                        )
                                                    })
                                                })
                                            },
                                        ))
                                },
                            )),
                    )
            },
        ))
}
const MOVE_SPAN: f32 = (60 - 1) as f32;

fn move_x(number: usize) -> f32 {
    (number - 1) as f32 / MOVE_SPAN
}

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

fn loss_color(loss: f32) -> Option<u32> {
    if loss > 6. {
        Some(0xef4444)
    } else if loss > 2. {
        Some(0xeab308)
    } else {
        None
    }
}

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
