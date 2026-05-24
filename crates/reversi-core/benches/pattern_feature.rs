use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};

mod common;

use common::{board_ply, choose_square};
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::eval::pattern_feature::{NUM_PATTERN_FEATURES, PatternFeatures};
use reversi_core::flip::flip;
use reversi_core::search::side_to_move::SideToMove;
use reversi_core::square::Square;

const N_CASES: usize = 512;
const MAX_PLAYOUT_MOVES: u32 = 48;

#[derive(Clone, Copy)]
struct InitCase {
    board: Board,
    ply: usize,
}

struct UpdateCase {
    pattern_features: PatternFeatures,
    sq: Square,
    flipped: Bitboard,
    ply: usize,
    side_to_move: SideToMove,
}

fn advance(board: Board, rng: &mut StdRng) -> Board {
    let moves = board.get_moves();
    if !moves.is_empty() {
        let sq = choose_square(moves, rng);
        let flipped = flip(sq, board.player(), board.opponent());
        return board.make_move_with_flipped(flipped, sq);
    }

    let passed = board.switch_players();
    if passed.has_legal_moves() {
        passed
    } else {
        Board::new()
    }
}

fn playout_position(rng: &mut StdRng) -> Board {
    let mut board = Board::new();
    let moves = rng.random_range(0..MAX_PLAYOUT_MOVES);
    for _ in 0..moves {
        board = advance(board, rng);
    }

    while !board.has_legal_moves() {
        let passed = board.switch_players();
        if !passed.has_legal_moves() {
            board = Board::new();
        } else {
            board = passed;
        }
    }

    board
}

fn cases(seed: u64) -> (Vec<InitCase>, Vec<UpdateCase>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut init_cases = Vec::with_capacity(N_CASES);
    let mut update_cases = Vec::with_capacity(N_CASES);

    for i in 0..N_CASES {
        let board = playout_position(&mut rng);
        let moves = board.get_moves();
        let sq = choose_square(moves, &mut rng);
        let flipped = flip(sq, board.player(), board.opponent());
        let ply = board_ply(&board);
        let side_to_move = if i % 2 == 0 {
            SideToMove::Player
        } else {
            SideToMove::Opponent
        };
        let feature_board = if side_to_move == SideToMove::Player {
            board
        } else {
            board.switch_players()
        };

        init_cases.push(InitCase { board, ply });
        update_cases.push(UpdateCase {
            pattern_features: PatternFeatures::new(&feature_board, ply),
            sq,
            flipped,
            ply,
            side_to_move,
        });
    }

    (init_cases, update_cases)
}

fn feature_sample(pattern_features: &PatternFeatures, ply: usize, sq: Square) -> u64 {
    let idx = sq.index() % NUM_PATTERN_FEATURES;
    let other_idx = (idx + NUM_PATTERN_FEATURES / 2) % NUM_PATTERN_FEATURES;
    let p_feature = pattern_features.p_feature(ply);
    let o_feature = pattern_features.o_feature(ply);

    (p_feature[idx] as u64)
        ^ ((o_feature[idx] as u64) << 16)
        ^ ((p_feature[other_idx] as u64) << 32)
        ^ ((o_feature[other_idx] as u64) << 48)
}

fn bench_pattern_features(c: &mut Criterion) {
    let (init_cases, mut update_cases) = cases(0xf3a7_0001);
    let mut group = c.benchmark_group("pattern_feature");
    group.throughput(Throughput::Elements(N_CASES as u64));

    group.bench_function("new", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for case in &init_cases {
                let pattern_features =
                    PatternFeatures::new(black_box(&case.board), black_box(case.ply));
                acc ^= feature_sample(&pattern_features, case.ply, Square::A1);
            }
            black_box(acc)
        })
    });

    group.bench_function("update", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for case in &mut update_cases {
                black_box(&mut case.pattern_features).update(
                    black_box(case.sq),
                    black_box(case.flipped),
                    black_box(case.ply),
                    black_box(case.side_to_move),
                );
                acc ^= feature_sample(&case.pattern_features, case.ply + 1, case.sq);
            }
            black_box(acc)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_pattern_features);
criterion_main!(benches);
