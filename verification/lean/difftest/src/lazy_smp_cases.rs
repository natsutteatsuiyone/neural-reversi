use std::io::{self, Write};
use std::num::NonZeroUsize;

use crate::SplitMix64;
use crate::lazy_smp_policy::{Ballot, skips_depth, winner};
use crate::score::Score;
use crate::square::Square;

fn write_vote(out: &mut impl Write, ballots: &[Ballot]) -> io::Result<()> {
    let winner = winner(ballots, |b| *b).map_or("-".into(), |i| i.to_string());
    write!(out, "v {winner}")?;
    for ballot in ballots {
        write!(
            out,
            " {},{},{}",
            ballot.sq.index(),
            ballot.score.value(),
            ballot.depth
        )?;
    }
    writeln!(out)
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    for id in (1..=64).chain([usize::MAX]) {
        let helper = NonZeroUsize::new(id).unwrap();
        for depth_ply in 0..=60 {
            let skip = usize::from(skips_depth(helper, depth_ply));
            writeln!(out, "s {id} {depth_ply} {skip}")?;
        }
    }

    let ballot = |sq, score, depth| Ballot {
        sq: Square::from_u8(sq).unwrap(),
        score: Score::from_raw(score),
        depth,
    };
    for ballots in [
        vec![],
        vec![ballot(63, -32767, 1)],
        vec![ballot(63, 0, 10), ballot(0, 0, 10)],
        vec![ballot(63, 256, 10), ballot(0, 0, 20)],
        vec![ballot(63, 0, 10), ballot(0, 0, 6), ballot(0, 0, 6)],
        vec![ballot(0, -32767, 1), ballot(63, 32767, 60)],
        vec![ballot(63, 32767, 60); 64],
    ] {
        write_vote(out, &ballots)?;
    }
    let mut random = SplitMix64(0xf93d_5c11_a280_17d9);
    let anchors = [-32767, -16384, -256, -1, 0, 1, 256, 16384, 32767];
    for i in 0..1024 {
        let ballots: Vec<_> = (0..(i % 65))
            .map(|_| {
                let square = (random.next() % if i % 2 == 0 { 4 } else { 64 }) as u8;
                let score = if i % 3 == 0 {
                    anchors[(random.next() % (anchors.len() as u64)) as usize]
                } else {
                    (random.next() % 65535) as i32 - 32767
                };
                let depth = (random.next() % 60 + 1) as u32;
                ballot(square, score, depth)
            })
            .collect();
        write_vote(out, &ballots)?;
    }
    Ok(())
}
