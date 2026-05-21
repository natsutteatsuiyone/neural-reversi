use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::bitboard::Bitboard;
use reversi_core::flip::{flip, flip2, flip3, flip4};
use reversi_core::square::Square;
use std::hint::black_box;

const N: usize = 4096;

fn random_positions(seed: u64) -> Vec<(Square, u64, u64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N);
    while out.len() < N {
        let p: u64 = rng.random();
        let o: u64 = rng.random();
        let o = o & !p;
        let sq_idx: u32 = rng.random_range(0..64);
        let sq = unsafe { Square::from_u32_unchecked(sq_idx) };
        if (p | o) & (1u64 << sq.index()) != 0 {
            continue;
        }
        out.push((sq, p, o));
    }
    out
}

fn random_position_groups(seed: u64) -> Vec<([Square; 4], u64, u64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N);
    while out.len() < N {
        let p: u64 = rng.random();
        let o: u64 = rng.random::<u64>() & !p;
        let empty = !(p | o);
        if empty.count_ones() < 4 {
            continue;
        }

        let mut squares = [Square::A1; 4];
        let mut used = 0u64;
        for square in &mut squares {
            loop {
                let sq_idx: u32 = rng.random_range(0..64);
                let bit = 1u64 << sq_idx;
                if (empty & bit) != 0 && (used & bit) == 0 {
                    *square = unsafe { Square::from_u32_unchecked(sq_idx) };
                    used |= bit;
                    break;
                }
            }
        }

        out.push((squares, p, o));
    }
    out
}

fn criterion_benchmark(c: &mut Criterion) {
    let cases = random_positions(0xfeed_beef);
    let groups = random_position_groups(0xfeed_cafe);

    c.bench_function("flip::random_4096", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(sq, p, o) in &cases {
                let f = flip(sq, Bitboard::new(p), Bitboard::new(o));
                acc ^= f.bits();
            }
            black_box(acc)
        })
    });

    c.bench_function("flip2::random_4096", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(sqs, p, o) in &groups {
                let (f0, f1) = flip2(sqs[0], sqs[1], Bitboard::new(p), Bitboard::new(o));
                acc ^= f0.bits() ^ f1.bits();
            }
            black_box(acc)
        })
    });

    c.bench_function("flip3::random_4096", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(sqs, p, o) in &groups {
                let (f0, f1, f2) =
                    flip3(sqs[0], sqs[1], sqs[2], Bitboard::new(p), Bitboard::new(o));
                acc ^= f0.bits() ^ f1.bits() ^ f2.bits();
            }
            black_box(acc)
        })
    });

    c.bench_function("flip4::random_4096", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for &(sqs, p, o) in &groups {
                let (f0, f1, f2, f3) = flip4(
                    sqs[0],
                    sqs[1],
                    sqs[2],
                    sqs[3],
                    Bitboard::new(p),
                    Bitboard::new(o),
                );
                acc ^= f0.bits() ^ f1.bits() ^ f2.bits() ^ f3.bits();
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
