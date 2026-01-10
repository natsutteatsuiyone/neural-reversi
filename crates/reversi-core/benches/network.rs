use criterion::{Criterion, criterion_group, criterion_main};
use reversi_core::board::Board;
use reversi_core::disc::Disc;
use reversi_core::eval::pattern_feature::{PatternFeature, PatternFeatures};
use reversi_core::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use reversi_core::eval::{Network, NetworkSmall};
use std::hint::black_box;

struct BenchInput {
    board: Board,
    ply: usize,
    pattern_features: PatternFeatures,
}

impl BenchInput {
    fn new(board: Board) -> Self {
        let ply = (board.get_player_count() + board.get_opponent_count()) as usize;
        let pattern_features = PatternFeatures::new(&board, ply);
        Self {
            board,
            ply,
            pattern_features,
        }
    }

    fn pattern_feature(&self) -> &PatternFeature {
        &self.pattern_features.p_features[self.ply]
    }
}

fn board_from_rows(rows: [&str; 8]) -> Board {
    let mut board_string = String::with_capacity(64);
    for row in rows {
        board_string.push_str(row);
    }
    Board::from_string(&board_string, Disc::Black)
}

fn load_main_network() -> Network {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let eval_path = format!("{}/../{}", manifest_dir, EVAL_FILE_NAME);
    let bytes =
        std::fs::read(&eval_path).expect("failed to read main evaluation network weights file");
    Network::from_bytes(&bytes).expect("failed to load main evaluation network weights")
}

fn load_small_network() -> NetworkSmall {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let eval_path = format!("{}/../{}", manifest_dir, EVAL_SM_FILE_NAME);
    let bytes =
        std::fs::read(&eval_path).expect("failed to read small evaluation network weights file");
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

fn criterion_benchmark(c: &mut Criterion) {
    let network = load_main_network();
    let network_small = load_small_network();

    let midgame = midgame_input();
    let endgame = endgame_input();

    c.bench_function("network::evaluate_midgame", |b| {
        b.iter(|| {
            black_box(network.evaluate(
                black_box(&midgame.board),
                black_box(midgame.pattern_feature()),
                midgame.ply,
            ))
        });
    });

    c.bench_function("network_small::evaluate_endgame", |b| {
        assert!(
            endgame.ply >= 30,
            "network_small expects ply >= 30 for benchmarking"
        );
        b.iter(|| {
            black_box(network_small.evaluate(black_box(endgame.pattern_feature()), endgame.ply))
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
