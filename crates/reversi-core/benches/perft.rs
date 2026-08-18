use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use reversi_core::perft::perft_root;

const BENCH_DEPTH: u32 = 9;
const EXPECTED_NODES: u64 = 3_005_320;

fn perft_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("perft");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    assert_eq!(
        perft_root(BENCH_DEPTH),
        EXPECTED_NODES,
        "reference node count mismatch at depth {BENCH_DEPTH}"
    );

    group.bench_function("9", |b| {
        b.iter(|| {
            let nodes = perft_root(black_box(BENCH_DEPTH));
            black_box(nodes)
        });
    });

    group.finish();
}

criterion_group!(benches, perft_benchmark);
criterion_main!(benches);
