use crate::engine::node_type::{NodeType, NonPV, PV};
use crate::probcut::Selectivity;
use crate::score::Score;
use crate::types::Depth;
use std::io::{self, Write};

trait SearchStrategy {
    const IS_ENDGAME: bool;
}

struct Phase<const ENDGAME: bool>;

impl<const ENDGAME: bool> SearchStrategy for Phase<ENDGAME> {
    const IS_ENDGAME: bool = ENDGAME;
}

include!(concat!(env!("OUT_DIR"), "/lmr_source.rs"));

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    for (index, value) in LMR_LOG.into_iter().enumerate() {
        writeln!(out, "log {index} {value}")?;
    }
    for depth in 0..=62 {
        for moves in 0..=62 {
            writeln!(out, "curve {depth} {moves} {}", lmr_curve(depth, moves))?;
        }
    }
    for depth in [4, 5, 9, 16, 26, 60, 61, u32::MAX] {
        for ordinal in [2, 3, 4, 9, 35, 61] {
            for total in [3, 62] {
                for flags in 0..32 {
                    let selectivity = if flags & 1 != 0 {
                        Selectivity::Level1
                    } else {
                        Selectivity::None
                    };
                    let cut = flags & 2 != 0;
                    let tt_pv = flags & 4 != 0;
                    for (alpha, value) in [
                        (0, 0),
                        (0, 1),
                        (0, -1),
                        (0, 511),
                        (0, -511),
                        (0, -512),
                        (32_767, -39_294),
                        (-32_767, 1 << 30),
                    ] {
                        let score = Score::ZERO + alpha;
                        let reduction = match flags >> 3 {
                            0 => compute_lmr_reduction::<NonPV, Phase<false>>,
                            1 => compute_lmr_reduction::<PV, Phase<false>>,
                            2 => compute_lmr_reduction::<NonPV, Phase<true>>,
                            _ => compute_lmr_reduction::<PV, Phase<true>>,
                        };
                        let result =
                            reduction(selectivity, depth, ordinal, total, cut, tt_pv, value, score);
                        writeln!(
                            out,
                            "reduce {depth} {ordinal} {total} {flags} {alpha} {value} {result}"
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}
