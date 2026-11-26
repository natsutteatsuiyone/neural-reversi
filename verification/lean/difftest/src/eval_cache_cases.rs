use std::io::{self, Write};

use flip_scalar_difftest::{SplitMix64, opt_or_dash};

use crate::eval_cache_policy::{EMPTY_ENTRY, KEY_MASK, bucket_count, location, pack, select_way};
use crate::score::Score;

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let mut random = SplitMix64(0x7342_9204_9813_5871);
    let keys = [0, u64::MAX, 0x1234_5678_9ABC_DEF0]
        .into_iter()
        .chain((0..64).map(|bit| 1_u64 << bit))
        .chain((0..8).map(|_| random.next_u64()));
    for key in keys {
        for size in 0..usize::BITS {
            let count = bucket_count(size);
            let (bucket, victim) = location(key, count as u64 - 1);
            writeln!(
                out,
                "g {} {size} {key} {count} {bucket} {victim}",
                usize::BITS
            )?;
        }
        for raw in [-32767, -16384, -1, 0, 1, 16383, 32767] {
            let word = pack(key, Score::from_raw(raw));
            writeln!(out, "e {key} {raw} {word}")?;
        }
    }

    for key in [0_u64, u64::MAX, 0x4321_FEDC_BA98_7654] {
        let words = [
            EMPTY_ENTRY,
            pack(key, Score::from_raw(-32767)),
            pack(key, Score::from_raw(32767)),
            pack(key ^ 1, Score::from_raw(7)),
            (((key ^ 1) & KEY_MASK) << 16) | EMPTY_ENTRY,
        ];
        for mut layout in 0..5usize.pow(4) {
            let bucket = std::array::from_fn(|_| {
                let value = words[layout % words.len()];
                layout /= words.len();
                value
            });
            let [a, b, c, d] = bucket;
            for victim in 0..4 {
                let (score, way) = select_way(key, &bucket, victim);
                let answer = opt_or_dash(score.map(Score::value));
                writeln!(out, "p {key} {victim} {a} {b} {c} {d} {way} {answer}")?;
            }
        }
    }
    Ok(())
}
