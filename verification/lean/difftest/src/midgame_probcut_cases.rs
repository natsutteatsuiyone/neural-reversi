use std::io::{self, Write};

use crate::probcut::{self, ProbcutCoefficients, Selectivity};
use crate::score::Score;
use crate::types::Depth;

fn write_thresholds(
    out: &mut impl Write,
    depth: Depth,
    beta: Score,
    t: f64,
    cut_node: bool,
    coefficients: &ProbcutCoefficients,
) -> io::Result<()> {
    let shallow = probcut::probcut_depth(depth);
    let verification = probcut::compute_probcut_beta(beta, t, coefficients);
    let screen = probcut::compute_eval_beta(beta, t, coefficients, cut_node);
    writeln!(
        out,
        "{depth} {shallow} {} {} {} {} {} {} {} {} {}",
        beta.value(),
        t.to_bits(),
        u8::from(cut_node),
        coefficients.mean.to_bits(),
        coefficients.sigma.to_bits(),
        coefficients.mean0.to_bits(),
        coefficients.sigma0.to_bits(),
        verification.value(),
        screen.value(),
    )
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    for ply in 0..60 {
        for depth in 0..60 {
            let coefficients = probcut::midgame_coefficients(ply, depth);
            for (beta, selectivity, cut_node) in [
                (-Score::INF + 1, Selectivity::Mid, true),
                (Score::INF, Selectivity::Mid, false),
                (Score::from_raw(5473), Selectivity::Level1, true),
                (Score::from_raw(-8289), Selectivity::Level1, false),
                (Score::ZERO, Selectivity::Level4, false),
            ] {
                write_thresholds(
                    out,
                    depth,
                    beta,
                    selectivity.t_value(),
                    cut_node,
                    coefficients,
                )?;
            }
        }
    }

    for (value, t) in [
        (0.0, -0.0),
        (f64::from_bits(1), 0.5),
        (-f64::from_bits(3), 0.5),
        (f64::MIN_POSITIVE.next_down(), 0.5),
        (f64::MIN_POSITIVE, 0.5f64.next_up()),
        (1.0f64.next_down(), 1.0),
        (1.0f64.next_up(), 1.0),
        (f64::from(i32::MIN), 1.0),
        (f64::from(i32::MAX), 1.0),
    ] {
        let coefficients = ProbcutCoefficients {
            mean: value,
            sigma: value,
            mean0: -value,
            sigma0: value,
        };
        for beta in [Score::ZERO, Score::from_raw(1), -Score::INF, Score::INF] {
            for cut_node in [false, true] {
                write_thresholds(out, Depth::MAX, beta, t, cut_node, &coefficients)?;
            }
        }
    }
    Ok(())
}
