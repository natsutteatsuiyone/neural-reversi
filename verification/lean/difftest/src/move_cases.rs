use std::io::{self, Write};

use crate::SplitMix64;
use crate::bitboard::Bitboard;

const BACKEND: &str = cfg_select! {
    all(target_arch = "x86_64", target_feature = "avx512vl") => "avx512",
    all(target_arch = "x86_64", target_feature = "avx2") => "avx2",
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "sha3"
    ) => "neon-sha3",
    all(target_arch = "aarch64", target_feature = "neon") => "neon",
    _ => "portable",
};

const SHA3_OVERLAPPING_COUNTEREXAMPLE: (u64, u64) = (0x201, 0x40200);
const OPENING_POSITION: (u64, u64) = (0x0000_0008_1000_0000, 0x0000_0010_0800_0000);

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "{BACKEND}")?;
    for (p, o) in [
        (0, 0),
        (0, !0),
        (!0, 0),
        (!0, !0),
        SHA3_OVERLAPPING_COUNTEREXAMPLE,
        OPENING_POSITION,
        (0x0004_3e24_8068_0404, 0x006a_400a_4400_7a02),
        (0x0081_6600_041c_1020, 0x0066_804a_1a22_6a00),
    ] {
        write_case(out, p, o)?;
    }

    for rank in 0..8 {
        for file in 0..8 {
            for dr in -1..=1 {
                for df in -1..=1 {
                    if (dr, df) == (0, 0) {
                        continue;
                    }
                    let p = 1 << (rank * 8 + file);
                    let mut o = 0;
                    for distance in 1..=7 {
                        let r = rank + dr * distance;
                        let f = file + df * distance;
                        if !(0..8).contains(&r) || !(0..8).contains(&f) {
                            break;
                        }
                        if distance >= 2 {
                            write_case(out, p, o)?;
                        }
                        o |= 1 << (r * 8 + f);
                    }
                }
            }
        }
    }

    let mut rng = SplitMix64(0x7265_7665_7273_692d);
    for i in 0..4096 {
        let (a, b, c) = (rng.next(), rng.next(), rng.next());
        let (p, o) = match i % 4 {
            0 => (a & !b, b),
            1 => (a & b & c, !a & b),
            2 => (a, b),
            _ => (a & b, !(a & b) & c),
        };
        write_case(out, p, o)?;
    }
    Ok(())
}

fn write_case(out: &mut impl Write, p: u64, o: u64) -> io::Result<()> {
    let player = Bitboard::from(p);
    let opponent = Bitboard::from(o);
    let legal = player.legal_moves(opponent);
    let (combined, potential) = player.legal_moves_and_potential(opponent);
    writeln!(
        out,
        "{p} {o} {} {} {}",
        u64::from(legal),
        u64::from(combined),
        u64::from(potential)
    )
}
