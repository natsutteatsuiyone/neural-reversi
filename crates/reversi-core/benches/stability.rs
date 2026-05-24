use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

mod common;

use common::{add_random_bit, bench_case_set};
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::constants::{SCORE_MAX, SCORE_MIN};
use reversi_core::stability;
use reversi_core::types::Score;

const N_CASES: usize = 4096;

#[derive(Clone, Copy)]
struct StabilityCase {
    player: u64,
    opponent: u64,
}

#[derive(Clone, Copy)]
struct CutoffCase {
    board: Board,
    n_empties: u32,
    alpha: Score,
}

fn random_case(rng: &mut StdRng, player_discs: usize, opponent_discs: usize) -> StabilityCase {
    let mut player = 0u64;
    let mut opponent = 0u64;
    let mut occupied = 0u64;

    for _ in 0..player_discs {
        player |= add_random_bit(rng, &mut occupied);
    }

    for _ in 0..opponent_discs {
        opponent |= add_random_bit(rng, &mut occupied);
    }

    StabilityCase { player, opponent }
}

fn random_cases(seed: u64, player_discs: usize, opponent_discs: usize) -> Vec<StabilityCase> {
    assert!(player_discs + opponent_discs <= 64);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);

    for _ in 0..N_CASES {
        out.push(random_case(&mut rng, player_discs, opponent_discs));
    }

    out
}

fn cutoff_cases(
    seed: u64,
    player_discs: usize,
    opponent_discs: usize,
    alpha: Score,
) -> Vec<CutoffCase> {
    assert!(player_discs + opponent_discs <= 64);

    let n_empties = (64 - player_discs - opponent_discs) as u32;

    random_cases(seed, player_discs, opponent_discs)
        .into_iter()
        .map(|case| CutoffCase {
            board: Board::from_bitboards(case.player, case.opponent),
            n_empties,
            alpha,
        })
        .collect()
}

fn cutoff_cases_by_empty_count(seed: u64, alpha: Score) -> Vec<CutoffCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);

    for case_index in 0..N_CASES {
        let n_empties = (case_index % 64) as u32;
        let occupied = 64 - n_empties as usize;
        let opponent_discs = occupied / 2;
        let player_discs = occupied - opponent_discs;
        let case = random_case(&mut rng, player_discs, opponent_discs);
        out.push(CutoffCase {
            board: Board::from_bitboards(case.player, case.opponent),
            n_empties,
            alpha,
        });
    }

    out
}

fn empty_alpha_sweep_cases(seed: u64) -> Vec<CutoffCase> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(64 * ((SCORE_MAX - SCORE_MIN + 1) as usize));

    for n_empties in 0..64 {
        let occupied = 64 - n_empties as usize;
        for alpha in SCORE_MIN..=SCORE_MAX {
            let opponent_discs = if occupied == 0 {
                0
            } else {
                rng.random_range(0..=occupied)
            };
            let player_discs = occupied - opponent_discs;
            let case = random_case(&mut rng, player_discs, opponent_discs);
            out.push(CutoffCase {
                board: Board::from_bitboards(case.player, case.opponent),
                n_empties,
                alpha,
            });
        }
    }

    out
}

fn stable_discs_checksum(cases: &[StabilityCase]) -> u64 {
    let mut acc = 0u64;
    for &case in cases {
        let player = Bitboard::new(black_box(case.player));
        let opponent = Bitboard::new(black_box(case.opponent));
        acc ^= stability::get_stable_discs(player, opponent).bits();
    }
    acc
}

fn stability_cutoff_checksum(cases: &[CutoffCase]) -> Score {
    let mut acc = 0;
    for &case in cases {
        let board = black_box(case.board);
        let score = stability::stability_cutoff(
            black_box(&board),
            black_box(case.n_empties),
            black_box(case.alpha),
        )
        .unwrap_or(SCORE_MAX + 1);
        acc ^= score;
    }
    acc
}

fn bench_get_stable_discs(c: &mut Criterion) {
    let sparse = random_cases(0x57ab_1e00, 12, 12);
    let dense = random_cases(0x57ab_1e01, 28, 28);

    let mut group = c.benchmark_group("stability::get_stable_discs");
    for (name, cases) in [("12v12", &sparse), ("28v28", &dense)] {
        bench_case_set(&mut group, name, cases, stable_discs_checksum);
    }
    group.finish();
}

fn bench_stability_cutoff(c: &mut Criterion) {
    let alpha_threshold_rejected = cutoff_cases_by_empty_count(0x57ab_1e02, SCORE_MIN);
    let disc_margin_rejected = cutoff_cases(0x57ab_1e03, 49, 7, SCORE_MAX);
    let stable_disc_checked = cutoff_cases(0x57ab_1e05, 28, 28, SCORE_MAX);
    let empty_alpha_sweep = empty_alpha_sweep_cases(0x57ab_1e04);

    let mut group = c.benchmark_group("stability::stability_cutoff");
    for (name, cases) in [
        ("alpha_threshold_rejected", &alpha_threshold_rejected),
        ("disc_margin_rejected", &disc_margin_rejected),
        ("stable_disc_checked", &stable_disc_checked),
        ("empty_alpha_sweep", &empty_alpha_sweep),
    ] {
        bench_case_set(&mut group, name, cases, stability_cutoff_checksum);
    }
    group.finish();
}

criterion_group!(benches, bench_get_stable_discs, bench_stability_cutoff);
criterion_main!(benches);
