use std::io::{self, Write};

use flip_scalar_difftest::csv;

#[allow(dead_code, clippy::enum_variant_names)]
#[path = "../../../../../crates/reversi-core/src"]
mod engine {
    pub mod bitboard;
    pub mod constants;
    #[path = "eval/pattern_feature.rs"]
    pub mod pattern_feature;
    #[path = "search/side_to_move.rs"]
    pub mod side_to_move;
    pub mod square;
}

mod search {
    pub use super::engine::side_to_move;
}

mod board {
    use super::bitboard::Bitboard;

    pub struct Board(pub Bitboard, pub Bitboard);

    impl Board {
        pub fn player(&self) -> Bitboard {
            self.0
        }

        pub fn opponent(&self) -> Bitboard {
            self.1
        }

        pub fn switch_players(&self) -> Board {
            Board(self.1, self.0)
        }
    }
}

use engine::bitboard::Bitboard;
use engine::pattern_feature::{EVAL_F2X, NUM_PATTERN_FEATURES, PatternFeature, PatternFeatures};
use engine::side_to_move::SideToMove;
use engine::square::Square;
use engine::{bitboard, constants, square};

fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn lanes(feature: &PatternFeature) -> String {
    csv((0..NUM_PATTERN_FEATURES).map(|i| feature[i]))
}

fn mover(side: usize) -> SideToMove {
    if side == 0 {
        SideToMove::Player
    } else {
        SideToMove::Opponent
    }
}

fn write_cases(out: &mut impl Write) -> io::Result<()> {
    for (index, pattern) in EVAL_F2X.iter().enumerate() {
        let squares = csv(pattern.squares[..pattern.n_square]
            .iter()
            .map(|sq| sq.index()));
        writeln!(out, "geometry {index} {squares}")?;
    }

    let mut state = 0x94D0_49BB_1331_11EB;
    for series in 0..32 {
        for sq in 0..64 {
            let placement = 1u64 << sq;
            let neighbor = 1u64 << ((sq + 1) % 64);
            let outside = !placement;
            let flips = match series {
                0 => neighbor,
                1 => outside,
                _ => (random(&mut state) & outside) | neighbor,
            };
            let free = outside & !flips;
            let actor = random(&mut state) & free;
            let victim = (random(&mut state) & free & !actor) | flips;
            let side = (series + sq) % 2;
            let (p, o) = if side == 0 {
                (actor, victim)
            } else {
                (victim, actor)
            };
            let ply = (series * 7 + sq) % (constants::MAX_PLY - 1);
            let mut features =
                PatternFeatures::new(&board::Board(Bitboard::new(p), Bitboard::new(o)), ply);
            features.update(
                Square::from_u8(sq as u8).unwrap(),
                Bitboard::new(flips),
                ply,
                mover(side),
            );
            writeln!(
                out,
                "board {p} {o} {sq} {flips} {side} {ply} {} {} {} {}",
                lanes(features.p_feature(ply)),
                lanes(features.o_feature(ply)),
                lanes(features.p_feature(ply + 1)),
                lanes(features.o_feature(ply + 1)),
            )?;
        }
    }

    for case in 0..512 {
        let sq = case % 64;
        let flips = random(&mut state) | (1 << (case / 8 % 64));
        let side = (case / 64 + sq) % 2;
        let ply = case % (constants::MAX_PLY - 1);
        let seed = board::Board(
            Bitboard::new(random(&mut state)),
            Bitboard::new(random(&mut state)),
        );
        let mut features = PatternFeatures::new(&seed, ply);
        let p = lanes(features.p_feature(ply));
        let o = lanes(features.o_feature(ply));
        features.update(
            Square::from_u8(sq as u8).unwrap(),
            Bitboard::new(flips),
            ply,
            mover(side),
        );
        assert_eq!(lanes(features.p_feature(ply)), p);
        assert_eq!(lanes(features.o_feature(ply)), o);
        writeln!(
            out,
            "raw {sq} {flips} {side} {ply} {p} {o} {} {}",
            lanes(features.p_feature(ply + 1)),
            lanes(features.o_feature(ply + 1)),
        )?;
    }
    Ok(())
}
#[cfg(not(target_arch = "wasm32"))]
fn main() -> io::Result<()> {
    flip_scalar_difftest::run_buffered(write_cases)
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use std::sync::OnceLock;

    fn cases() -> &'static [u8] {
        static CASES: OnceLock<Vec<u8>> = OnceLock::new();
        CASES.get_or_init(|| {
            let mut out = Vec::new();
            super::write_cases(&mut out).expect("writing to Vec cannot fail");
            out
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn fixture_ptr() -> *const u8 {
        cases().as_ptr()
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn fixture_len() -> usize {
        cases().len()
    }
}
