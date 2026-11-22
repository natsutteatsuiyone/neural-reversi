use std::io::{self, Write};

use crate::probcut::{ENDGAME_LEVELS, EndgameWindow};
use crate::score::Score;

fn write_window(out: &mut impl Write, beta: Score, margins: (f64, f64)) -> io::Result<()> {
    let (verification, screen) = margins;
    let window = EndgameWindow::from_scaled_margins(beta, verification, screen);
    writeln!(
        out,
        "{} {} {} {} {} {} {}",
        beta.value(),
        verification.to_bits(),
        screen.to_bits(),
        verification.ceil() as i32,
        screen.floor() as i32,
        window.pc_beta.value(),
        window.screen_beta.value(),
    )
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let betas = [
        -Score::INF + 1,
        Score::MIN,
        Score::from_raw(-2049),
        Score::ZERO,
        Score::from_raw(1),
        Score::MAX - 1,
        Score::MAX,
        Score::INF,
    ];
    let evaluations = [
        Score::MIN,
        Score::from_raw(-4097),
        Score::ZERO,
        Score::from_raw(4097),
        Score::MAX,
    ];
    for empties in (0..=64).chain([u32::MAX]) {
        for level in ENDGAME_LEVELS {
            for cut_node in [false, true] {
                for beta in betas {
                    for evaluation in evaluations {
                        write_window(
                            out,
                            beta,
                            EndgameWindow::scaled_margins(
                                empties, level, cut_node, beta, evaluation,
                            ),
                        )?;
                    }
                }
            }
        }
    }

    let verification_margins = [
        -0.0,
        f64::from_bits(1),
        1.0f64.next_down(),
        1.0,
        1.0f64.next_up(),
        f64::from(i32::MAX).next_down(),
        f64::from(i32::MAX),
        f64::from(i32::MAX).next_up(),
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ];
    let screen_margins = [
        f64::NEG_INFINITY,
        f64::from(i32::MIN).next_down(),
        f64::from(i32::MIN),
        f64::from(i32::MIN).next_up(),
        (-1.0f64).next_down(),
        -1.0,
        (-1.0f64).next_up(),
        -f64::from_bits(1),
        -0.0,
        0.0,
        1.0f64.next_down(),
        1.0,
        1.0f64.next_up(),
        f64::from(i32::MAX).next_up(),
        f64::INFINITY,
        -f64::NAN,
    ];
    for beta in betas {
        for verification in verification_margins {
            for screen in screen_margins {
                write_window(out, beta, (verification, screen))?;
            }
        }
    }
    Ok(())
}
