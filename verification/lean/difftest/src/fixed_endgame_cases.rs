use std::io::{self, Write};

use flip_scalar_difftest::opt_or_dash;

use crate::level::Level;
use crate::probcut::{ENDGAME_STEPS, Selectivity};
use crate::types::Depth;

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let ladder = Selectivity::ENDGAME_LADDER;
    let [first, second, third, fourth, exact] = ladder.map(Selectivity::as_u8);
    writeln!(out, "{first} {second} {third} {fourth} {exact}")?;
    let caps = [Selectivity::Mid].into_iter().chain(ladder);

    for empties in [0, 1, 14, 30, 60, Depth::MAX] {
        // Enumerate the pass/fail patterns, including depth limits that rise
        // again after the first failing step.
        let patterns = if empties == 0 { 1 } else { 1 << ENDGAME_STEPS };
        for mask in 0..patterns {
            let depths = std::array::from_fn(|index| {
                if mask & (1 << index) == 0 {
                    empties
                } else {
                    empties - 1
                }
            });
            let level = Level::with_depths(8, depths);
            let [d1, d2, d3, d4, d5] = depths;
            for first_cap in caps.clone() {
                for second_cap in caps.clone() {
                    let result = level
                        .stopping_at(first_cap)
                        .stopping_at(second_cap)
                        .end_selectivity(empties);
                    let first_cap = first_cap.as_u8();
                    let second_cap = second_cap.as_u8();
                    writeln!(
                        out,
                        "{empties} {first_cap} {second_cap} {d1} {d2} {d3} {d4} {d5} {}",
                        opt_or_dash(result.map(Selectivity::as_u8))
                    )?;
                }
            }
        }
    }
    Ok(())
}
