use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::bitboard::Bitboard;
use reversi_core::count_last_flip::count_last_flip;
use reversi_core::square::Square;
use std::hint::black_box;

const N: usize = 4096;

fn random_positions(seed: u64) -> Vec<(Square, u64)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N);
    while out.len() < N {
        let p: u64 = rng.random();
        let sq_idx: u32 = rng.random_range(0..64);
        let sq = unsafe { Square::from_u32_unchecked(sq_idx) };
        if p & (1u64 << sq.index()) != 0 {
            continue;
        }
        out.push((sq, p));
    }
    out
}

fn criterion_benchmark(c: &mut Criterion) {
    let cases = random_positions(0xc01d_cafe);

    c.bench_function("count_last_flip::random_4096", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for &(sq, p) in &cases {
                acc ^= count_last_flip(Bitboard::new(p), sq);
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
