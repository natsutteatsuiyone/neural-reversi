use criterion::{Criterion, criterion_group, criterion_main};
use reversi_core::bitboard;
use reversi_core::square::Square;
use std::hint::black_box;

fn bench_get_moves(c: &mut Criterion) {
    let p_initial = Square::D5.bitboard() | Square::E4.bitboard();
    let o_initial = Square::D4.bitboard() | Square::E5.bitboard();

    c.bench_function("bitboard_get_moves", |b| {
        b.iter(|| black_box(p_initial).get_moves(black_box(o_initial)))
    });
}

fn bench_corner_weighted_count(c: &mut Criterion) {
    let board = bitboard::Bitboard::new(0x8100000000000081 | 0x00FF00000000FF00);

    c.bench_function("bitboard_corner_weighted_count", |b| {
        b.iter(|| black_box(board).corner_weighted_count())
    });
}

fn bench_get_corner_stability(c: &mut Criterion) {
    let board = 0x8100000000000081 | 0x8000000000000000 | 0x0100000000000000;

    c.bench_function("bitboard_corner_stability", |b| {
        b.iter(|| bitboard::Bitboard::new(black_box(board)).corner_stability())
    });
}

fn bench_get_potential_moves(c: &mut Criterion) {
    let p = Square::D5.bitboard() | Square::E4.bitboard();
    let o = Square::D4.bitboard() | Square::E5.bitboard();

    c.bench_function("bitboard_potential_moves", |b| {
        b.iter(|| black_box(p).get_potential_moves(black_box(o)))
    });
}

fn bench_get_moves_and_potential(c: &mut Criterion) {
    let p = Square::D5.bitboard() | Square::E4.bitboard();
    let o = Square::D4.bitboard() | Square::E5.bitboard();

    c.bench_function("bitboard_moves_and_potential", |b| {
        b.iter(|| black_box(p).get_moves_and_potential(black_box(o)))
    });
}

criterion_group!(
    benches,
    bench_get_moves,
    bench_corner_weighted_count,
    bench_get_corner_stability,
    bench_get_potential_moves,
    bench_get_moves_and_potential
);
criterion_main!(benches);
