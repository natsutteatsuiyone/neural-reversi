use std::io::{self, Write};

use crate::aspiration::{midgame_window, retry_midgame};
use crate::score::Score;

fn write_windows(out: &mut impl Write) -> io::Result<()> {
    let edge = Score::INF.value() - 1;
    let square = i64::from(edge).pow(2);
    let histories = [
        None,
        Some((-Score::INF.value(), 0)),
        Some((-edge, -square)),
        Some((edge, square)),
        Some((-Score::MAX.value(), -i64::from(Score::MAX.value()).pow(2))),
        Some((Score::MAX.value(), i64::from(Score::MAX.value()).pow(2))),
        Some((31, 10_192)),
        Some((-31, -10_193)),
        Some((31, 10_194)),
        Some((0, 0)),
    ];
    for depth in [0, 4, 5, 6, u32::MAX] {
        for helper in (0..16).chain([usize::MAX]) {
            for history in histories {
                let (alpha, beta, delta) = midgame_window(
                    depth,
                    history.map(|(avg, mss)| (Score::from_raw(avg), mss)),
                    helper,
                );
                let stats = history.map_or("none".into(), |(avg, mss)| format!("{avg},{mss}"));
                writeln!(
                    out,
                    "w {depth} {helper} {stats} {} {} {}",
                    alpha.value(),
                    beta.value(),
                    delta.value(),
                )?;
            }
        }
    }
    Ok(())
}

fn write_retries(out: &mut impl Write) -> io::Result<()> {
    let inf = Score::INF.value();
    for (lo, hi) in [
        (-inf, inf),
        (-inf, -1),
        (-40, 30),
        (100, 101),
        (inf - 1, inf),
        (-inf, -inf + 1),
        (-37, inf),
    ] {
        for width in [512, 513, 44286, 65536, 106288] {
            for style in 0..4 {
                for abort_at in [0, 1, 3, 12, 16] {
                    let mut windows = Vec::new();
                    let (mut alpha, mut beta) = (Score::from_raw(lo), Score::from_raw(hi));
                    let (result, clean) =
                        retry_midgame(&mut alpha, &mut beta, Score::ZERO + width, |a, b| {
                            let index = windows.len();
                            assert!(index < 16, "aspiration retry limit exceeded");
                            windows.push(format!("{},{}", a.value(), b.value()));
                            if index == abort_at {
                                return (Score::INF, true);
                            }
                            let low = a.max(-Score::INF + 1);
                            let high = b.min(Score::INF - 1);
                            let score = match style {
                                0 => Score::ZERO,
                                1 => low,
                                2 => high,
                                _ if index % 2 == 0 => high,
                                _ => low,
                            };
                            (score, false)
                        });
                    writeln!(
                        out,
                        "r {lo} {hi} {width} {style} {abort_at} {} {} {} {} {}",
                        alpha.value(),
                        beta.value(),
                        result.value(),
                        usize::from(clean),
                        windows.join(";"),
                    )?;
                }
            }
        }
    }
    Ok(())
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    write_windows(out)?;
    write_retries(out)
}
