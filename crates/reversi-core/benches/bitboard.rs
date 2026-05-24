use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use reversi_core::bitboard::Bitboard;

const PLAYER: Bitboard = Bitboard::new(0x0000_3c00_003c_0000);
const OPPONENT: Bitboard = Bitboard::new(0x003c_003c_3c00_3c00);
const COUNT_INPUT: Bitboard = Bitboard::new(0x8100_3c3c_3c3c_0081);
const STABILITY_INPUT: Bitboard = Bitboard::new(0xc381_0000_0000_81c3);

fn bench_get_moves(c: &mut Criterion) {
    c.bench_function("bitboard::get_moves", |b| {
        b.iter(|| black_box(black_box(PLAYER).get_moves(black_box(OPPONENT))))
    });
}

fn bench_get_potential_moves(c: &mut Criterion) {
    c.bench_function("bitboard::get_potential_moves", |b| {
        b.iter(|| black_box(black_box(PLAYER).get_potential_moves(black_box(OPPONENT))))
    });
}

fn bench_get_moves_and_potential(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitboard::moves_and_potential");

    group.bench_function("separate", |b| {
        b.iter(|| {
            let player = black_box(PLAYER);
            let opponent = black_box(OPPONENT);
            black_box((
                player.get_moves(opponent),
                player.get_potential_moves(opponent),
            ))
        })
    });

    group.bench_function("combined", |b| {
        b.iter(|| black_box(black_box(PLAYER).get_moves_and_potential(black_box(OPPONENT))))
    });

    group.finish();
}

fn bench_corner_weighted_count(c: &mut Criterion) {
    c.bench_function("bitboard::corner_weighted_count", |b| {
        b.iter(|| black_box(black_box(COUNT_INPUT).corner_weighted_count()))
    });
}

fn bench_corner_stability(c: &mut Criterion) {
    c.bench_function("bitboard::corner_stability", |b| {
        b.iter(|| black_box(black_box(STABILITY_INPUT).corner_stability()))
    });
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
