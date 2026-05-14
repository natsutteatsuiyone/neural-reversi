use std::hint::black_box;
use std::sync::Arc;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::eval::Eval;
use reversi_core::move_list::MoveList;
use reversi_core::probcut::Selectivity;
use reversi_core::search::search_context::SearchContext;
use reversi_core::square::Square;
use reversi_core::transposition_table::TranspositionTable;

const N_CASES: usize = 256;
const MIN_MOVES: u32 = 8;

struct Case {
    board: Board,
    moves: Bitboard,
}

fn new_context(board: &Board) -> SearchContext {
    let eval = Arc::new(Eval::new().expect("failed to load eval weights"));
    let tt = Arc::new(TranspositionTable::new(1));
    SearchContext::new(board, Selectivity::None, tt, eval)
}

fn random_cases(seed: u64) -> Vec<Case> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(N_CASES);

    while out.len() < N_CASES {
        let player: u64 = rng.random();
        let opponent: u64 = rng.random::<u64>() & !player;
        let board = Board::from_bitboards(player, opponent);
        let moves = board.get_moves();
        if moves.count() >= MIN_MOVES {
            out.push(Case { board, moves });
        }
    }

    out
}

fn evaluated_lists(cases: &[Case]) -> Vec<MoveList> {
    let mut ctx = new_context(&cases[0].board);

    cases
        .iter()
        .map(|case| {
            let mut moves = MoveList::with_moves(&case.board, case.moves);
            moves.evaluate_moves_fast(&mut ctx, &case.board, Square::None);
            moves
        })
        .collect()
}

fn checksum_move_list(moves: &MoveList) -> u64 {
    let mut acc = moves.count() as u64;
    if let Some(sq) = moves.wipeout_move() {
        acc ^= 0x9e37_79b9_7f4a_7c15u64.rotate_left(sq.index() as u32);
    }
    for mv in moves.iter() {
        acc = acc.rotate_left(7) ^ mv.flipped.bits() ^ (mv.sq.index() as u64);
    }
    acc
}

fn criterion_benchmark(c: &mut Criterion) {
    let cases = random_cases(0x5eed_f00d);
    let lists = evaluated_lists(&cases);

    c.bench_function("move_list::with_moves_256", |b| {
        b.iter(|| {
            let mut acc = 0u64;
            for case in &cases {
                let moves = MoveList::with_moves(black_box(&case.board), black_box(case.moves));
                acc ^= checksum_move_list(&moves);
            }
            black_box(acc)
        })
    });

    c.bench_function("move_list::evaluate_fast_256", |b| {
        let mut ctx = new_context(&cases[0].board);

        b.iter(|| {
            let mut acc = 0i32;
            for case in &cases {
                let mut moves = MoveList::with_moves(black_box(&case.board), black_box(case.moves));
                moves.evaluate_moves_fast(&mut ctx, black_box(&case.board), Square::None);
                acc ^= moves.iter().fold(0, |sum, mv| sum ^ mv.value);
            }
            black_box(acc)
        })
    });

    c.bench_function("move_list::sort_256", |b| {
        b.iter_batched(
            || lists.clone(),
            |mut lists| {
                let mut acc = 0i32;
                for moves in &mut lists {
                    moves.sort();
                    acc ^= moves.iter().next().map_or(0, |mv| mv.value);
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("move_list::best_first_256", |b| {
        b.iter_batched(
            || lists.clone(),
            |mut lists| {
                let mut acc = 0i32;
                for moves in &mut lists {
                    for mv in moves.best_first_iter() {
                        acc ^= mv.value;
                    }
                }
                black_box(acc)
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
