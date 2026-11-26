#[allow(dead_code, clippy::enum_variant_names)]
#[path = "../../../../crates/reversi-core/src"]
mod engine {
    #[path = "eval/network/activations.rs"]
    pub mod activations;
    #[path = "util/align.rs"]
    pub mod align;
    pub mod bitboard;
    pub mod constants;
    pub mod level;
    #[path = "search/node_type.rs"]
    pub mod node_type;
    #[allow(unused_imports)]
    pub mod probcut;
    pub mod score;
    #[path = "search/stack.rs"]
    pub mod search_stack;
    pub mod square;
    pub mod types;
}
use engine::{
    activations, align, bitboard, constants, level, probcut, score, search_stack, square, types,
};

#[allow(dead_code)]
#[path = "../../../../crates/reversi-core/src/search/threading/abort.rs"]
mod abort;
#[path = "../../../../crates/reversi-core/src/search"]
mod search {
    #[allow(dead_code)]
    pub(crate) mod time_control;
    pub(crate) mod threading {
        pub(crate) use crate::abort::AbortState;
    }
}
use search::time_control;

mod util {
    pub use crate::engine::align;
}

mod activation_cases;
#[path = "../../../../crates/reversi-core/src/search/aspiration.rs"]
#[allow(dead_code)]
mod aspiration;
mod endgame_aspiration_cases;
mod endgame_probcut_cases;
mod endgame_timing_cases;
mod eval_cache_cases;
#[path = "../../../../crates/reversi-core/src/eval/eval_cache/policy.rs"]
mod eval_cache_policy;
mod fixed_endgame_cases;
mod lazy_smp_cases;
#[path = "../../../../crates/reversi-core/src/search/midgame/lazy_smp/policy.rs"]
mod lazy_smp_policy;
mod lmr_cases;
mod midgame_aspiration_cases;
mod midgame_probcut_cases;
mod move_cases;
mod pv_stack_cases;
mod split_report_cases;
mod time_settings_cases;

#[path = "."]
mod flip {
    #[path = "../../../../crates/reversi-core/src/flip/lrmask.rs"]
    mod lrmask;

    #[path = "../../../../crates/reversi-core/src/flip/flip_scalar.rs"]
    mod flip_scalar;

    pub fn masks(sq: usize) -> [u64; 8] {
        lrmask::LRMASK[sq].0
    }

    pub fn flip(sq: usize, player: u64, opponent: u64) -> u64 {
        flip_scalar::flip(
            crate::square::Square::from_u8(sq as u8).unwrap(),
            player,
            opponent,
        )
    }
}

use flip_scalar_difftest::{SplitMix64, run_buffered};

const RANDOM_CASES_PER_SQUARE: usize = 400;

fn main() -> std::io::Result<()> {
    run_buffered(|out| match std::env::args().nth(1).as_deref() {
        None => flip_cases(out),
        Some("activation-cases") => activation_cases::write_cases(out),
        Some("move-cases") => move_cases::write_cases(out),
        Some("endgame-timing-cases") => endgame_timing_cases::write_cases(out),
        Some("time-settings-cases") => time_settings_cases::write_cases(out),
        Some("fixed-endgame-cases") => fixed_endgame_cases::write_cases(out),
        Some("endgame-probcut-cases") => endgame_probcut_cases::write_cases(out),
        Some("endgame-aspiration-cases") => endgame_aspiration_cases::write_cases(out),
        Some("midgame-probcut-cases") => midgame_probcut_cases::write_cases(out),
        Some("midgame-aspiration-cases") => midgame_aspiration_cases::write_cases(out),
        Some("pv-stack-cases") => pv_stack_cases::write_cases(out),
        Some("lazy-smp-cases") => lazy_smp_cases::write_cases(out),
        Some("lmr-cases") => lmr_cases::write_cases(out),
        Some("split-report-cases") => split_report_cases::write_cases(out),
        Some("finite-score-cases") => write_finite_scores(out),
        Some("eval-cache-cases") => eval_cache_cases::write_cases(out),
        Some(arg) => panic!("unknown case generator: {arg}"),
    })
}

fn write_finite_scores(out: &mut impl std::io::Write) -> std::io::Result<()> {
    let outside_i16 = [
        i32::MIN,
        i32::MIN + 1,
        -65_536,
        -65_535,
        -32_769,
        32_768,
        32_769,
        65_534,
        65_535,
        65_536,
        i32::MAX - 1,
        i32::MAX,
    ];
    for raw in (i16::MIN..=i16::MAX).map(i32::from).chain(outside_i16) {
        writeln!(
            out,
            "{raw} {}",
            usize::from((score::Score::ZERO + raw).is_finite())
        )?;
    }
    Ok(())
}

fn flip_cases(out: &mut impl std::io::Write) -> std::io::Result<()> {
    let mut rng = SplitMix64(0x6e65_7572_616c_2d72);
    for sq in 0..64 {
        let bit = 1u64 << sq;
        let mut cases = vec![(0, 0), (!0, 0), (0, !0), (!0, !0)];
        for k in 0..RANDOM_CASES_PER_SQUARE {
            let (a, b, c) = (rng.next_u64(), rng.next_u64(), rng.next_u64());
            let (player, opponent) = match k % 4 {
                0 => (a & !b, b),
                1 => (a & b & c, !a & b),
                2 => (a, b),
                _ => (a & b, !(a & b) & !(c & b & a)),
            };
            let origin_empty = if k % 4 == 2 { !0 } else { !bit };
            cases.push((player & origin_empty, opponent & origin_empty));
        }
        for (player, opponent) in cases {
            let m = flip::masks(sq);
            let result = flip::flip(sq, player, opponent);
            writeln!(
                out,
                "{sq} {} {} {} {} {} {} {} {} {player} {opponent} {result}",
                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]
            )?;
        }
    }
    Ok(())
}
