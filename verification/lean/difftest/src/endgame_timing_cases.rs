use std::io::{self, Write};

use flip_scalar_difftest::SplitMix64;

#[allow(clippy::duplicate_mod)]
#[path = "../../../../crates/reversi-core/src/search/time_control/endgame_budget.rs"]
mod endgame_budget;

const CAST_BOUNDARY: u64 = 1 << 53;
const BUDGETS: [u64; 8] = [
    0,
    1,
    999,
    10_000,
    CAST_BOUNDARY - 1,
    CAST_BOUNDARY + 1,
    1 << 63,
    u64::MAX,
];
const CAST_CASES: [u64; 9] = [
    0,
    1,
    CAST_BOUNDARY - 1,
    CAST_BOUNDARY,
    CAST_BOUNDARY + 1,
    CAST_BOUNDARY + 2,
    CAST_BOUNDARY + 3,
    (1 << 63) + 1,
    u64::MAX,
];

fn write_case(out: &mut impl Write, elapsed: u64, budget: u64, factor: f64) -> io::Result<()> {
    writeln!(
        out,
        "{elapsed} {budget} {} {} {} {} {}",
        factor.to_bits(),
        (elapsed as f64).to_bits(),
        (budget as f64).to_bits(),
        endgame_budget::projected_time(elapsed, factor).to_bits(),
        u8::from(endgame_budget::permits(elapsed, budget, factor)),
    )
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let special_factors = [
        0.0,
        -0.0,
        f64::from_bits(1),
        f64::from_bits((1 << 52) - 1),
        f64::MIN_POSITIVE,
        0.5,
        1.0,
        1.0_f64.next_up(),
        1.5,
        f64::MAX,
        f64::NAN,
        -f64::NAN,
    ];
    let native_factors = (0..=60).flat_map(endgame_budget::factors);
    for factor in native_factors.chain(special_factors) {
        for budget in BUDGETS {
            let boundary = (budget as f64 / factor) as u64;
            for elapsed in [
                boundary.saturating_sub(2),
                boundary.saturating_sub(1),
                boundary,
                boundary.saturating_add(1),
                boundary.saturating_add(2),
            ] {
                write_case(out, elapsed, budget, factor)?;
            }
        }
        for elapsed in CAST_CASES {
            write_case(out, elapsed, elapsed, factor)?;
        }
    }

    let mut rng = SplitMix64(0x74a1_892f_a120_9125);
    for _ in 0..1024 {
        let factor = f64::from_bits(rng.next_u64() & 0x7fff_ffff_ffff_ffff);
        write_case(out, rng.next_u64(), rng.next_u64(), factor)?;
    }
    Ok(())
}
