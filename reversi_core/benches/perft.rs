use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use reversi_core::perft::perft_root;

const BENCH_DEPTHS: [u32; 1] = [9];
const REFERENCE_COUNTS: &[(u32, u64)] = &[
    (1, 4),
    (2, 12),
    (3, 56),
    (4, 244),
    (5, 1_396),
    (6, 8_200),
    (7, 55_092),
    (8, 390_216),
    (9, 3_005_320),
];

fn expected_nodes(depth: u32) -> Option<u64> {
    REFERENCE_COUNTS
        .iter()
        .find_map(|&(d, nodes)| (d == depth).then_some(nodes))
}

fn perft_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("perft_root");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    for &depth in &BENCH_DEPTHS {
        let expected = expected_nodes(depth).unwrap_or_else(|| {
            panic!(
                "no reference node count recorded for perft depth {depth}; update REFERENCE_COUNTS"
            );
        });

        assert_eq!(
            perft_root(depth),
            expected,
            "reference node count mismatch at depth {depth}"
        );

        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| {
                let nodes = perft_root(black_box(depth));
                black_box(nodes)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, perft_benchmark);
criterion_main!(benches);
