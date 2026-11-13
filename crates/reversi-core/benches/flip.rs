use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{SeedableRng, rngs::StdRng};

mod common;

use common::choose_square;
use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::flip::flip;
use reversi_core::obf::ObfPosition;
use reversi_core::square::Square;

const MOVE_CHAIN_COUNT: usize = 256;
const CORPUS_SEED: u64 = 0xf11f_5e4c_1a7e_2026;

const ENDGAME_SOURCES: &[&str] = &[
    include_str!("../../../problem/fforum-1-19.obf"),
    include_str!("../../../problem/fforum-20-39.obf"),
    include_str!("../../../problem/fforum-40-59.obf"),
    include_str!("../../../problem/fforum-60-79.obf"),
    include_str!("../../../problem/hard-20.obf"),
    include_str!("../../../problem/hard-25.obf"),
    include_str!("../../../problem/hard-30.obf"),
    include_str!("../../../problem/small-35.txt"),
];

#[derive(Clone, Copy)]
enum PlayoutStep {
    Move(Square),
    Pass,
}

struct MoveChain {
    start: Board,
    steps: Vec<PlayoutStep>,
    expected_final: Board,
}

#[derive(Clone, Copy)]
struct FlipCase {
    square: Square,
    player: Bitboard,
    opponent: Bitboard,
}

fn source_boards() -> Vec<Board> {
    let mut boards = Vec::new();
    for source in ENDGAME_SOURCES {
        for line in source.lines() {
            let Some(position) = ObfPosition::parse(line).expect("benchmark OBF line must parse")
            else {
                continue;
            };
            boards.push(position.board);
        }
    }
    assert!(!boards.is_empty(), "flip consumer corpus must not be empty");
    boards
}

fn move_chains(seed: u64) -> Vec<MoveChain> {
    let sources = source_boards();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut chains = Vec::with_capacity(MOVE_CHAIN_COUNT);
    let mut source_index = 0usize;

    while chains.len() < MOVE_CHAIN_COUNT {
        let start = sources[source_index % sources.len()];
        source_index += 1;
        let mut board = start;
        let mut steps = Vec::with_capacity(board.empty_count() as usize + 2);

        loop {
            let moves = board.legal_moves();
            if !moves.is_empty() {
                let sq = choose_square(moves, &mut rng);
                steps.push(PlayoutStep::Move(sq));
                board = board.make_move(sq);
                continue;
            }

            let passed = board.switch_players();
            if passed.legal_moves().is_empty() {
                break;
            }
            steps.push(PlayoutStep::Pass);
            board = passed;
        }

        if steps
            .iter()
            .any(|&step| matches!(step, PlayoutStep::Move(_)))
        {
            chains.push(MoveChain {
                start,
                steps,
                expected_final: board,
            });
        }
    }

    chains
}

#[inline(never)]
fn replay_move_chains(chains: &[MoveChain]) -> u64 {
    let mut checksum = 0u64;

    for chain in chains {
        let mut board = chain.start;
        for &step in &chain.steps {
            board = match step {
                PlayoutStep::Move(sq) => board.make_move(sq),
                PlayoutStep::Pass => board.switch_players(),
            };
        }
        debug_assert_eq!(board, chain.expected_final);
        checksum = checksum.rotate_left(11)
            ^ board.player().bits()
            ^ board.opponent().bits().rotate_left(1);
    }

    black_box(checksum)
}

fn flip_cases(chains: &[MoveChain]) -> Vec<FlipCase> {
    let case_count = chains
        .iter()
        .map(|chain| {
            chain
                .steps
                .iter()
                .filter(|step| matches!(step, PlayoutStep::Move(_)))
                .count()
        })
        .sum();
    let mut cases = Vec::with_capacity(case_count);

    for chain in chains {
        let mut board = chain.start;
        for &step in &chain.steps {
            board = match step {
                PlayoutStep::Move(square) => {
                    cases.push(FlipCase {
                        square,
                        player: board.player(),
                        opponent: board.opponent(),
                    });
                    board.make_move(square)
                }
                PlayoutStep::Pass => board.switch_players(),
            };
        }
        debug_assert_eq!(board, chain.expected_final);
    }

    assert!(!cases.is_empty(), "single-flip corpus must not be empty");
    cases
}

#[inline(never)]
fn flip_checksum(cases: &[FlipCase]) -> u64 {
    let mut checksum = 0u64;

    for &case in cases {
        let flipped = flip(
            black_box(case.square),
            black_box(case.player),
            black_box(case.opponent),
        );
        checksum = checksum.rotate_left(7) ^ flipped.bits();
    }

    black_box(checksum)
}

fn bench_single_flip(c: &mut Criterion, cases: &[FlipCase]) {
    let mut group = c.benchmark_group("flip");
    group.throughput(Throughput::Elements(cases.len() as u64));
    group.bench_function("single", |b| {
        b.iter(|| flip_checksum(black_box(cases)));
    });
    group.finish();
}

fn flip_consumer_benchmark(c: &mut Criterion) {
    let chains = move_chains(CORPUS_SEED);
    let total_steps: u64 = chains.iter().map(|chain| chain.steps.len() as u64).sum();
    let single_flip_cases = flip_cases(&chains);

    bench_single_flip(c, &single_flip_cases);

    let mut group = c.benchmark_group("flip/consumer");
    group.sample_size(50);
    group.warm_up_time(Duration::from_millis(750));
    group.measurement_time(Duration::from_secs(4));

    group.throughput(Throughput::Elements(total_steps));
    group.bench_with_input(
        BenchmarkId::new("make_move_chain", "problem_playouts"),
        &chains,
        |b, chains| b.iter(|| replay_move_chains(black_box(chains))),
    );

    group.finish();
}

criterion_group!(benches, flip_consumer_benchmark);
criterion_main!(benches);
