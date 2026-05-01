use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::bitboard::Bitboard;
use reversi_core::constants::{SCORE_MAX, SCORE_MIN};
use reversi_core::count_last_flip::{count_last_flip, solve1};
use reversi_core::square::Square;
use reversi_core::types::Score;
use std::hint::black_box;

const N: usize = 4096;

#[derive(Clone, Copy)]
struct SolveCase {
    sq: Square,
    player: u64,
    alpha: Score,
}

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

fn random_solve_cases(seed: u64) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N);
    while out.len() < N {
        let p: u64 = rng.random();
        let sq_idx: u32 = rng.random_range(0..64);
        let sq = unsafe { Square::from_u32_unchecked(sq_idx) };
        if p & (1u64 << sq.index()) != 0 {
            continue;
        }

        out.push(SolveCase {
            sq,
            player: p,
            alpha: rng.random_range((SCORE_MAX - 128)..=SCORE_MAX),
        });
    }
    out
}

fn solve_cases_where_player_cannot_move(seed: u64, alpha: Score) -> Vec<SolveCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N);
    while out.len() < N {
        let p: u64 = rng.random();
        let sq_idx: u32 = rng.random_range(0..64);
        let sq = unsafe { Square::from_u32_unchecked(sq_idx) };
        if p & (1u64 << sq.index()) != 0 {
            continue;
        }
        if count_last_flip(Bitboard::new(p), sq) != 0 {
            continue;
        }

        out.push(SolveCase {
            sq,
            player: p,
            alpha,
        });
    }
    out
}

#[inline(always)]
fn solve1_baseline(player: Bitboard, alpha: Score, sq: Square) -> Score {
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

fn bench_solve_case_set(c: &mut Criterion, name: &str, cases: &[SolveCase]) {
    let mut group = c.benchmark_group("solve1");

    group.bench_with_input(BenchmarkId::new("specialized", name), cases, |b, cases| {
        b.iter(|| {
            let mut acc = 0i32;
            for &case in cases {
                let player = Bitboard::new(case.player);
                acc ^= solve1(black_box(player), black_box(case.alpha), black_box(case.sq));
            }
            black_box(acc)
        })
    });

    group.bench_with_input(
        BenchmarkId::new("baseline_two_counts", name),
        cases,
        |b, cases| {
            b.iter(|| {
                let mut acc = 0i32;
                for &case in cases {
                    let player = Bitboard::new(case.player);
                    acc ^= solve1_baseline(
                        black_box(player),
                        black_box(case.alpha),
                        black_box(case.sq),
                    );
                }
                black_box(acc)
            })
        },
    );

    group.finish();
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

    let random_solve = random_solve_cases(0x51d3_f00d);
    let pass_pruned = solve_cases_where_player_cannot_move(0xdead_beef, SCORE_MAX);
    let opponent_checked = solve_cases_where_player_cannot_move(0xfade_5eed, SCORE_MIN - 128);

    bench_solve_case_set(c, "random_4096", &random_solve);
    bench_solve_case_set(c, "player_pass_pruned_4096", &pass_pruned);
    bench_solve_case_set(c, "opponent_checked_4096", &opponent_checked);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
