mod network;

use std::io;

use reversi_core::{board::Board, eval::eval_cache::EvalCache, types::ScaledScore};

use crate::{eval::network::Network, search::search_context::SearchContext};

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
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> ScaledScore {
        let key = board.hash();
        if let Some(score_cache) = self.cache.probe(key) {
            return score_cache;
        }

        let score = self.network.evaluate(ctx.get_pattern_feature(), ctx.ply());

        self.cache.store(key, score);
        score
    }
}
