mod network;

use std::{env, io};

use reversi_core::{board::Board, eval::eval_cache::EvalCache, types::ScaledScore};

use crate::{eval::network::Network, search::search_context::SearchContext};

/// Macro for the WASM-specific evaluation weight file name.
macro_rules! eval_weights_literal {
    () => {
        "eval_wasm-549b18e6.zst"
    };
}

pub struct Eval {
    network: Network,
    cache: EvalCache,
}

impl Eval {
    pub fn new() -> io::Result<Self> {
        Self::with_weight_files()
    }

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
