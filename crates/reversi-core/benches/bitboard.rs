use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

mod common;

use common::{bench_case_set, random_disjoint_bitboards};
use reversi_core::bitboard::Bitboard;

// A single fixed input lets the branch predictor and caches memoize the result,
// so it measures back-to-back throughput of one identical call rather than the
// realistic varied-position workload these functions see in search. Every bench
// below instead runs a fixed corpus of random positions and XOR-accumulates the
// results, matching the other movegen/flip/stability benches.
const N_CASES: usize = 4096;
const MOVE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const DISC_SEED: u64 = 0xD1B5_4A32_D192_ED03;

/// Random disjoint `(player, opponent)` positions for the move-generation benches.
fn move_cases() -> Vec<(Bitboard, Bitboard)> {
    let mut rng = StdRng::seed_from_u64(MOVE_SEED);
    (0..N_CASES)
        .map(|_| {
            let (player, opponent) = random_disjoint_bitboards(&mut rng);
            (Bitboard::new(player), Bitboard::new(opponent))
        })
        .collect()
}

/// Random disc sets for the corner-region benches (single-bitboard inputs).
fn disc_cases() -> Vec<Bitboard> {
    let mut rng = StdRng::seed_from_u64(DISC_SEED);
    (0..N_CASES).map(|_| Bitboard::new(rng.random())).collect()
}

fn get_moves_checksum(cases: &[(Bitboard, Bitboard)]) -> u64 {
    let mut acc = 0u64;
    for &(player, opponent) in cases {
        acc ^= black_box(player).get_moves(black_box(opponent)).bits();
    }
    acc
}

fn get_potential_checksum(cases: &[(Bitboard, Bitboard)]) -> u64 {
    let mut acc = 0u64;
    for &(player, opponent) in cases {
        acc ^= black_box(player)
            .get_potential_moves(black_box(opponent))
            .bits();
    }
    acc
}

fn separate_checksum(cases: &[(Bitboard, Bitboard)]) -> u64 {
    let mut acc = 0u64;
    for &(player, opponent) in cases {
        let player = black_box(player);
        let opponent = black_box(opponent);
        acc ^= player.get_moves(opponent).bits();
        acc ^= player.get_potential_moves(opponent).bits().rotate_left(1);
    }
    acc
}

fn combined_checksum(cases: &[(Bitboard, Bitboard)]) -> u64 {
    let mut acc = 0u64;
    for &(player, opponent) in cases {
        let (moves, potential) = black_box(player).get_moves_and_potential(black_box(opponent));
        acc ^= moves.bits();
        acc ^= potential.bits().rotate_left(1);
    }
    acc
}

fn corner_weighted_count_checksum(cases: &[Bitboard]) -> u64 {
    let mut acc = 0u64;
    for &discs in cases {
        acc ^= u64::from(black_box(discs).corner_weighted_count());
    }
    acc
}

fn corner_stability_checksum(cases: &[Bitboard]) -> u64 {
    let mut acc = 0u64;
    for &discs in cases {
        acc ^= u64::from(black_box(discs).corner_stability());
    }
    acc
}

fn bench_get_moves(c: &mut Criterion) {
    let cases = move_cases();
    let mut group = c.benchmark_group("bitboard::get_moves");
    bench_case_set(&mut group, "varied", &cases, get_moves_checksum);
    group.finish();
}

fn bench_get_potential_moves(c: &mut Criterion) {
    let cases = move_cases();
    let mut group = c.benchmark_group("bitboard::get_potential_moves");
    bench_case_set(&mut group, "varied", &cases, get_potential_checksum);
    group.finish();
}

fn bench_get_moves_and_potential(c: &mut Criterion) {
    let cases = move_cases();
    let mut group = c.benchmark_group("bitboard::moves_and_potential");
    bench_case_set(&mut group, "separate", &cases, separate_checksum);
    bench_case_set(&mut group, "combined", &cases, combined_checksum);
    group.finish();
}

fn bench_corner_weighted_count(c: &mut Criterion) {
    let cases = disc_cases();
    let mut group = c.benchmark_group("bitboard::corner_weighted_count");
    bench_case_set(&mut group, "varied", &cases, corner_weighted_count_checksum);
    group.finish();
}

fn bench_corner_stability(c: &mut Criterion) {
    let cases = disc_cases();
    let mut group = c.benchmark_group("bitboard::corner_stability");
    bench_case_set(&mut group, "varied", &cases, corner_stability_checksum);
    group.finish();
}

criterion_group!(
    benches,
    bench_get_moves,
    bench_get_potential_moves,
    bench_get_moves_and_potential,
    bench_corner_weighted_count,
    bench_corner_stability,
);
criterion_main!(benches);
