use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

mod common;

use common::{board_from_rows, board_ply, weights_path};
use reversi_core::board::Board;
use reversi_core::eval::pattern_feature::{PatternFeature, PatternFeatures};
use reversi_core::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME, Network, NetworkSmall};

struct BenchInput {
    board: Board,
    ply: usize,
    pattern_features: PatternFeatures,
}

impl BenchInput {
    fn new(board: Board) -> Self {
        let ply = board_ply(&board);
        let pattern_features = PatternFeatures::new(&board, ply);
        Self {
            board,
            ply,
            pattern_features,
        }
    }

    fn pattern_feature(&self) -> &PatternFeature {
        self.pattern_features.p_feature(self.ply)
    }
}

fn load_main_network() -> Network {
    let bytes = std::fs::read(weights_path(EVAL_FILE_NAME))
        .expect("failed to read main evaluation network weights file");
    Network::from_bytes(&bytes).expect("failed to load main evaluation network weights")
}

fn load_small_network() -> NetworkSmall {
    let bytes = std::fs::read(weights_path(EVAL_SM_FILE_NAME))
        .expect("failed to read small evaluation network weights file");
    NetworkSmall::from_bytes(&bytes).expect("failed to load small evaluation network weights")
}

fn midgame_input() -> BenchInput {
    BenchInput::new(board_from_rows([
        "--XOX---", "-OOXXX--", "--OXOO--", "--XXXO--", "---OXO--", "--XOO---", "----XX--",
        "---O----",
    ]))
}

fn endgame_input() -> BenchInput {
    BenchInput::new(board_from_rows([
        "XOXOXOXO", "OXOXOXOX", "XOXOXOXO", "OXOXOXOX", "XOXOXOXO", "OXOXOXOX", "--------",
        "--------",
    ]))
}

fn bench_main_network(c: &mut Criterion, network: &Network, input: &BenchInput) {
    c.bench_function("network::evaluate", |b| {
        b.iter(|| {
            let score = black_box(network).evaluate(
                black_box(&input.board),
                black_box(input.pattern_feature()),
                black_box(input.ply),
            );
            black_box(score.value())
        })
    });
}

fn bench_small_network(c: &mut Criterion, network: &NetworkSmall, input: &BenchInput) {
    assert!(
        input.ply >= 30,
        "network_small expects ply >= 30 for benchmarking"
    );

    c.bench_function("network_small::evaluate", |b| {
        b.iter(|| {
            let score = black_box(network)
                .evaluate(black_box(input.pattern_feature()), black_box(input.ply));
            black_box(score.value())
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    // Keep zstd decoding and weight layout setup outside the timed loops.
    let network = load_main_network();
    let network_small = load_small_network();

    let main_input = midgame_input();
    let small_input = endgame_input();

    bench_main_network(c, &network, &main_input);
    bench_small_network(c, &network_small, &small_input);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
