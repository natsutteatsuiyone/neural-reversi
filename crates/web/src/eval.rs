mod network;

use std::io;

use reversi_core::{
    board::Board,
    eval::{eval_cache::EvalCache, pattern_feature::PatternFeature},
    score::Score,
};

use crate::{eval::network::Network, search::context::SearchContext};

macro_rules! eval_weights_literal {
    () => {
        "eval_wasm-9c7b9796.zst"
    };
}

/// Neural network evaluator with a set-associative evaluation cache.
pub struct Eval {
    network: Network,
    cache: EvalCache,
}

impl Eval {
    /// Creates a new evaluator by loading the embedded weight data.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the weight data cannot be decompressed or parsed.
    ///
    /// [`io::Error`]: std::io::Error
    pub fn new() -> io::Result<Self> {
        let network = Network::from_bytes(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../",
            eval_weights_literal!()
        )))?;
        Ok(Eval {
            network,
            cache: EvalCache::new(17),
        })
    }

    /// Returns the cached evaluation score, computing it via the network on a cache miss.
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> Score {
        let key = board.hash();
        self.cache.get_or_insert_with(key, || {
            self.network.evaluate(ctx.pattern_feature(), ctx.ply())
        })
    }

    /// Evaluates a precomputed pattern feature without cache or search context overhead.
    #[inline(always)]
    pub(crate) fn evaluate_network(&self, pattern_feature: &PatternFeature, ply: usize) -> Score {
        self.network.evaluate(pattern_feature, ply)
    }
}

#[cfg(test)]
mod tests {
    use byteorder::{LittleEndian, ReadBytesExt};
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use reversi_core::eval::pattern_feature::{
        INPUT_FEATURE_DIMS, PATTERN_FEATURE_OFFSETS, PatternFeatures,
    };
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    use super::*;

    const INPUT_DIMS: usize = 256;
    const HIDDEN_DIMS: usize = 16;

    struct ReferenceStack {
        l1_biases: Vec<i32>,
        l1_weights: Vec<i16>,
        l2_biases: Vec<i32>,
        l2_weights: Vec<i16>,
        output_bias: i32,
        output_weights: Vec<i16>,
    }

    /// The network in the order the weight file stores it, evaluated one
    /// value at a time.
    struct ReferenceNetwork {
        input_biases: Vec<i16>,
        input_weights: Vec<i8>,
        stacks: Vec<ReferenceStack>,
    }

    fn read_i16s(reader: &mut impl io::Read, len: usize) -> Vec<i16> {
        let mut values = vec![0; len];
        reader.read_i16_into::<LittleEndian>(&mut values).unwrap();
        values
    }

    fn read_i32s(reader: &mut impl io::Read, len: usize) -> Vec<i32> {
        let mut values = vec![0; len];
        reader.read_i32_into::<LittleEndian>(&mut values).unwrap();
        values
    }

    /// Returns `bias + inputs · weights` after the squared clipped activation.
    fn reference_neuron(bias: i32, inputs: &[i64], weights: &[i16]) -> i64 {
        let sum: i64 = inputs
            .iter()
            .zip(weights)
            .map(|(&input, &weight)| input * i64::from(weight))
            .sum();
        let clipped = ((i64::from(bias) + sum) >> 10).clamp(0, 1023);
        (clipped * clipped) >> 10
    }

    impl ReferenceNetwork {
        fn load(bytes: &[u8]) -> Self {
            let mut reader = zstd::stream::read::Decoder::new(bytes).unwrap();
            let input_biases = read_i16s(&mut reader, INPUT_DIMS);
            let mut input_weights = vec![0; INPUT_FEATURE_DIMS * INPUT_DIMS];
            reader.read_i8_into(&mut input_weights).unwrap();
            let stacks = (0..60)
                .map(|_| ReferenceStack {
                    l1_biases: read_i32s(&mut reader, HIDDEN_DIMS),
                    l1_weights: read_i16s(&mut reader, HIDDEN_DIMS * INPUT_DIMS),
                    l2_biases: read_i32s(&mut reader, HIDDEN_DIMS),
                    l2_weights: read_i16s(&mut reader, HIDDEN_DIMS * HIDDEN_DIMS),
                    output_bias: reader.read_i32::<LittleEndian>().unwrap(),
                    output_weights: read_i16s(&mut reader, HIDDEN_DIMS + INPUT_DIMS),
                })
                .collect();
            Self {
                input_biases,
                input_weights,
                stacks,
            }
        }

        fn evaluate(&self, pattern_feature: &PatternFeature, ply: usize) -> Score {
            let stack = &self.stacks[ply];

            let input: Vec<i64> = (0..INPUT_DIMS)
                .map(|i| {
                    let sum: i64 = PATTERN_FEATURE_OFFSETS
                        .iter()
                        .enumerate()
                        .map(|(idx, offset)| {
                            let row = offset + usize::from(pattern_feature[idx]);
                            i64::from(self.input_weights[row * INPUT_DIMS + i])
                        })
                        .sum();
                    let clipped = (i64::from(self.input_biases[i]) + sum).clamp(0, 255);
                    (clipped * clipped) >> 8
                })
                .collect();
            let l1: Vec<i64> = (0..HIDDEN_DIMS)
                .map(|j| {
                    let weights = &stack.l1_weights[j * INPUT_DIMS..(j + 1) * INPUT_DIMS];
                    reference_neuron(stack.l1_biases[j], &input, weights)
                })
                .collect();
            let l2: Vec<i64> = (0..HIDDEN_DIMS)
                .map(|j| {
                    let weights = &stack.l2_weights[j * HIDDEN_DIMS..(j + 1) * HIDDEN_DIMS];
                    reference_neuron(stack.l2_biases[j], &l1, weights)
                })
                .collect();

            let output: i64 = l2
                .iter()
                .chain(&input)
                .zip(&stack.output_weights)
                .map(|(&value, &weight)| value * i64::from(weight))
                .sum();
            let raw = (i64::from(stack.output_bias) + output) >> 7;
            Score::from_raw(raw as i32).clamp(Score::MIN + 1, Score::MAX - 1)
        }
    }

    #[test]
    fn network_matches_reference_on_random_games() {
        let eval = Eval::new().unwrap();
        let reference = ReferenceNetwork::load(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../",
            eval_weights_literal!()
        )));
        let mut rng = StdRng::seed_from_u64(0xE7A1);
        let mut n_positions = 0;

        for _ in 0..40 {
            let mut board = Board::new();
            loop {
                if !board.has_legal_moves() {
                    board = board.switch_players();
                    if !board.has_legal_moves() {
                        break;
                    }
                }
                let ply = 60 - board.empty_count() as usize;
                let features = PatternFeatures::new(&board, ply);
                for feature in [features.p_feature(ply), features.o_feature(ply)] {
                    assert_eq!(
                        eval.evaluate_network(feature, ply),
                        reference.evaluate(feature, ply),
                        "ply {ply} of {board:?}"
                    );
                }
                n_positions += 1;

                let moves = board.legal_moves();
                let choice = rng.random_range(0..moves.count() as usize);
                board = board.make_move(moves.iter().nth(choice).unwrap());
            }
        }

        assert!(
            n_positions > 2000,
            "only {n_positions} positions were checked"
        );
    }
}
