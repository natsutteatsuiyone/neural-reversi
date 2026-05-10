use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::bitboard::Bitboard;
use reversi_core::stability;

const N: usize = 4096;

#[derive(Clone, Copy)]
struct StabilityCase {
    player: u64,
    opponent: u64,
}

fn random_cases(seed: u64, occupied: usize) -> Vec<StabilityCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N);

    while out.len() < N {
        let mut player = 0u64;
        let mut opponent = 0u64;
        let mut n_discs = 0usize;

        while n_discs < occupied {
            let bit = 1u64 << rng.random_range(0..64);
            if ((player | opponent) & bit) != 0 {
                continue;
            }

            if rng.random::<bool>() {
                player |= bit;
            } else {
                opponent |= bit;
            }
            n_discs += 1;
        }

        out.push(StabilityCase { player, opponent });
    }

    out
}

fn bench_case_set(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, cases: &[StabilityCase]) {
    group.bench_with_input(BenchmarkId::from_parameter(name), cases, |b, cases| {
        b.iter(|| {
            let mut acc = 0u64;
            for case in cases {
                let player = Bitboard::new(black_box(case.player));
                let opponent = Bitboard::new(black_box(case.opponent));
                acc ^= stability::get_stable_discs(player, opponent).bits();
            }
            black_box(acc)
        })
    });
}

fn bench_get_stable_discs(c: &mut Criterion) {
    let midgame = random_cases(0x57ab_1e00, 24);
    let late = random_cases(0x57ab_1e01, 56);

    let mut group = c.benchmark_group("stability::get_stable_discs");
    bench_case_set(&mut group, "midgame_4096", &midgame);
    bench_case_set(&mut group, "late_4096", &late);
    group.finish();
}

fn bench_edge_table_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("stability::edge_table");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    group.bench_function("build", |b| {
        b.iter(|| black_box(stability::build_edge_stability_table_for_bench()))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_get_stable_discs,
    bench_edge_table_build
);
criterion_main!(benches);
