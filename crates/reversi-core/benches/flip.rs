use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

mod common;

use common::{bench_case_set, random_disjoint_bitboards, random_square};
use reversi_core::bitboard::Bitboard;
use reversi_core::flip::{flip, flip2, flip3, flip4};
use reversi_core::square::Square;

const N_CASES: usize = 4096;

#[derive(Clone, Copy)]
struct FlipCase {
    square: Square,
    player: u64,
    opponent: u64,
}

#[derive(Clone, Copy)]
struct FlipGroupCase {
    squares: [Square; 4],
    player: u64,
    opponent: u64,
}

fn random_positions_with_min_empty(seed: u64, min_empty: u32) -> Vec<FlipCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (player, opponent) = random_disjoint_bitboards(&mut rng);
        let empty = !(player | opponent);
        if empty.count_ones() < min_empty.max(1) {
            continue;
        }

        let square = loop {
            let candidate = random_square(&mut rng);
            if empty & (1u64 << candidate.index()) != 0 {
                break candidate;
            }
        };

        out.push(FlipCase {
            square,
            player,
            opponent,
        });
    }
    out
}

fn random_position_groups(seed: u64) -> Vec<FlipGroupCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (player, opponent) = random_disjoint_bitboards(&mut rng);
        let empty = !(player | opponent);
        // Fewer than four empty squares makes selecting four distinct candidates impossible.
        if empty.count_ones() < 4 {
            continue;
        }

        let mut squares = [Square::A1; 4];
        let mut used = 0u64;
        for square in &mut squares {
            loop {
                let candidate = random_square(&mut rng);
                let bit = 1u64 << candidate.index();
                if (empty & bit) != 0 && (used & bit) == 0 {
                    *square = candidate;
                    used |= bit;
                    break;
                }
            }
        }

        out.push(FlipGroupCase {
            squares,
            player,
            opponent,
        });
    }
    out
}

#[inline(always)]
fn bitboards(player: u64, opponent: u64) -> (Bitboard, Bitboard) {
    (
        Bitboard::new(black_box(player)),
        Bitboard::new(black_box(opponent)),
    )
}

fn flip_checksum(cases: &[FlipCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let (player, opponent) = bitboards(case.player, case.opponent);
        acc ^= flip(black_box(case.square), player, opponent).bits();
    }
    acc
}

fn flip2_checksum(cases: &[FlipGroupCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let [sq0, sq1, ..] = black_box(case.squares);
        let (player, opponent) = bitboards(case.player, case.opponent);
        let (f0, f1) = flip2(sq0, sq1, player, opponent);
        acc ^= f0.bits() ^ f1.bits();
    }
    acc
}

fn flip2_separate_checksum(cases: &[FlipGroupCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let [sq0, sq1, ..] = black_box(case.squares);
        let (player, opponent) = bitboards(case.player, case.opponent);
        let f0 = flip(sq0, player, opponent);
        let f1 = flip(sq1, player, opponent);
        acc ^= f0.bits() ^ f1.bits();
    }
    acc
}

fn flip3_checksum(cases: &[FlipGroupCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let [sq0, sq1, sq2, ..] = black_box(case.squares);
        let (player, opponent) = bitboards(case.player, case.opponent);
        let (f0, f1, f2) = flip3(sq0, sq1, sq2, player, opponent);
        acc ^= f0.bits() ^ f1.bits() ^ f2.bits();
    }
    acc
}

fn flip3_separate_checksum(cases: &[FlipGroupCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let [sq0, sq1, sq2, ..] = black_box(case.squares);
        let (player, opponent) = bitboards(case.player, case.opponent);
        let f0 = flip(sq0, player, opponent);
        let f1 = flip(sq1, player, opponent);
        let f2 = flip(sq2, player, opponent);
        acc ^= f0.bits() ^ f1.bits() ^ f2.bits();
    }
    acc
}

fn flip4_checksum(cases: &[FlipGroupCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let [sq0, sq1, sq2, sq3] = black_box(case.squares);
        let (player, opponent) = bitboards(case.player, case.opponent);
        let (f0, f1, f2, f3) = flip4(sq0, sq1, sq2, sq3, player, opponent);
        acc ^= f0.bits() ^ f1.bits() ^ f2.bits() ^ f3.bits();
    }
    acc
}

fn flip4_separate_checksum(cases: &[FlipGroupCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let [sq0, sq1, sq2, sq3] = black_box(case.squares);
        let (player, opponent) = bitboards(case.player, case.opponent);
        let f0 = flip(sq0, player, opponent);
        let f1 = flip(sq1, player, opponent);
        let f2 = flip(sq2, player, opponent);
        let f3 = flip(sq3, player, opponent);
        acc ^= f0.bits() ^ f1.bits() ^ f2.bits() ^ f3.bits();
    }
    acc
}

fn criterion_benchmark(c: &mut Criterion) {
    // This filter is not needed for single-square sampling; it mirrors the
    // grouped corpus so arity comparisons use comparable board density.
    let shared_position_cases = random_positions_with_min_empty(0xfeed_beef, 4);
    let groups = random_position_groups(0xfeed_cafe);

    let mut group = c.benchmark_group("flip");

    bench_case_set(&mut group, "flip", &shared_position_cases, flip_checksum);
    bench_case_set(&mut group, "flip2", &groups, flip2_checksum);
    bench_case_set(
        &mut group,
        "flip2_separate",
        &groups,
        flip2_separate_checksum,
    );
    bench_case_set(&mut group, "flip3", &groups, flip3_checksum);
    bench_case_set(
        &mut group,
        "flip3_separate",
        &groups,
        flip3_separate_checksum,
    );
    bench_case_set(&mut group, "flip4", &groups, flip4_checksum);
    bench_case_set(
        &mut group,
        "flip4_separate",
        &groups,
        flip4_separate_checksum,
    );

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
