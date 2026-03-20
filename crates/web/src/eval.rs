mod network;

use std::{env, io};

use reversi_core::{board::Board, eval::eval_cache::EvalCache, types::ScaledScore};

use crate::{eval::network::Network, search::search_context::SearchContext};

/// Expands to the WASM-specific evaluation weight file name.
macro_rules! eval_weights_literal {
    () => {
        "eval_wasm-e6bbc4f6.zst"
    };
}

/// Neural network evaluator with an LRU-style cache.
pub struct Eval {
    network: Network,
    cache: EvalCache,
}

impl Eval {
    /// Creates a new evaluator by loading the embedded weight file.
    pub fn new() -> io::Result<Self> {
        Self::with_weight_files()
    }

    /// Creates a new evaluator from the compile-time-embedded weight data.
    pub fn with_weight_files() -> io::Result<Self> {
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
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> ScaledScore {
        let key = board.hash();
        if let Some(score_cache) = self.cache.probe(key) {
            return score_cache;
        }

        let score = self
            .network
            .evaluate(board, ctx.get_pattern_feature(), ctx.ply());

        self.cache.store(key, score);
        score
    }
}
