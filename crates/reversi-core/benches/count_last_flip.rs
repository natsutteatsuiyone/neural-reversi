use std::hint::black_box;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

mod common;

use common::random_square;
use reversi_core::bitboard::Bitboard;
use reversi_core::constants::{SCORE_MAX, SCORE_MIN};
use reversi_core::count_last_flip::{count_last_flip, solve1};
use reversi_core::square::Square;
use reversi_core::types::Score;

const N_CASES: usize = 4096;

#[derive(Clone, Copy)]
struct CountCase {
    sq: Square,
    player: u64,
}

#[derive(Clone, Copy)]
struct SolveCase {
    sq: Square,
    player: u64,
    alpha: Score,
}

fn random_count_cases(seed: u64) -> Vec<CountCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);
        out.push(CountCase { sq, player });
    }
    out
}

fn random_solve_cases(seed: u64) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);

        out.push(SolveCase {
            sq,
            player,
            alpha: rng.random_range(SCORE_MIN..=SCORE_MAX),
        });
    }
    out
}

fn solve_cases_where_player_cannot_move(seed: u64, alpha: Score) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);
    while out.len() < N_CASES {
        let (sq, player) = random_empty_square_case(&mut rng);
        if count_last_flip(Bitboard::new(player), sq) != 0 {
            continue;
        }

        out.push(SolveCase { sq, player, alpha });
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

#[inline(always)]
fn solve1_separate_count_calls(player: Bitboard, alpha: Score, sq: Square) -> Score {
    let mut n_flipped = count_last_flip(player, sq);
    let mut score = 2 * player.count() as Score - SCORE_MAX + 2 + n_flipped;

    if n_flipped == 0 {
        let score_if_opp_passes = if score > 0 { score } else { score - 2 };
        if score_if_opp_passes > alpha {
            n_flipped = count_last_flip(!player, sq);
            score = if n_flipped > 0 {
                score - 2 - n_flipped
            } else {
                score_if_opp_passes
            };
        } else {
            score = score_if_opp_passes;
        }
    }

    score
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

fn bench_solve_case_set(group: &mut BenchmarkGroup<'_, WallTime>, name: &str, cases: &[SolveCase]) {
    group.bench_with_input(BenchmarkId::new("specialized", name), cases, |b, cases| {
        b.iter(|| {
            let mut acc = 0i32;
            for &case in cases {
                let player = Bitboard::new(black_box(case.player));
                acc ^= solve1(player, black_box(case.alpha), black_box(case.sq));
            }
            black_box(acc)
        })
    });

    group.bench_with_input(
        BenchmarkId::new("separate_count_calls", name),
        cases,
        |b, cases| {
            b.iter(|| {
                let mut acc = 0i32;
                for &case in cases {
                    let player = Bitboard::new(black_box(case.player));
                    acc ^= solve1_separate_count_calls(
                        player,
                        black_box(case.alpha),
                        black_box(case.sq),
                    );
                }
                black_box(acc)
            })
        },
    );
}

fn bench_solve1(
    c: &mut Criterion,
    mixed_cases: &[SolveCase],
    pass_pruned: &[SolveCase],
    opponent_checked: &[SolveCase],
) {
    let mut group = c.benchmark_group("count_last_flip::solve1");

    bench_solve_case_set(&mut group, "mixed", mixed_cases);
    bench_solve_case_set(&mut group, "player_pass_pruned", pass_pruned);
    bench_solve_case_set(&mut group, "opponent_checked", opponent_checked);
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    let count_cases = random_count_cases(0xc01d_cafe);
    let mixed_solve = random_solve_cases(0x51d3_f00d);
    // Above all legal scores, so the player-pass branch remains pruned even if the cutoff
    // comparison changes from strict to inclusive.
    let pass_pruned = solve_cases_where_player_cannot_move(0xdead_beef, SCORE_MAX + 1);
    let opponent_checked = solve_cases_where_player_cannot_move(0xfade_5eed, SCORE_MIN - 1);

    bench_count_last_flip(c, &count_cases);
    bench_solve1(c, &mixed_solve, &pass_pruned, &opponent_checked);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
