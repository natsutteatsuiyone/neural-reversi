use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

mod common;
mod solve1;

use common::random_square;
use reversi_core::bitboard::Bitboard;
use reversi_core::count_last_flip::count_last_flip;
use reversi_core::square::Square;

const N_COUNT_CASES: usize = 4096;

#[derive(Clone, Copy)]
struct CountCase {
    sq: Square,
    player: u64,
}

fn random_count_cases(seed: u64) -> Vec<CountCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_COUNT_CASES);
    while out.len() < N_COUNT_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);
        out.push(CountCase { sq, player });
    }
    out
}

fn random_empty_square_case(rng: &mut StdRng) -> (Square, u64) {
    loop {
        let player: u64 = rng.random();
        let empty = !player;
        if empty == 0 {
            continue;
        }

        let sq = loop {
            let candidate = random_square(rng);
            if empty & (1u64 << candidate.index()) != 0 {
                break candidate;
            }
        };

        return (sq, player);
    }
}

fn bench_count_last_flip(c: &mut Criterion, cases: &[CountCase]) {
    c.bench_function("count_last_flip::count_last_flip", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for &case in cases {
                let player = Bitboard::new(black_box(case.player));
                acc ^= count_last_flip(player, black_box(case.sq));
            }
            black_box(acc)
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let count_cases = random_count_cases(0xc01d_cafe);

    bench_count_last_flip(c, &count_cases);
    solve1::bench_solve1(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(750))
        .measurement_time(Duration::from_secs(4))
        .sample_size(100);
    targets = criterion_benchmark
}
criterion_main!(benches);
