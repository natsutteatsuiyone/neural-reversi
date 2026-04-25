use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::bitboard::Bitboard;
use reversi_core::flip::flip;
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

fn criterion_benchmark(c: &mut Criterion) {
    let cases = random_positions(0xfeed_beef);

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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
