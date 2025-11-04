mod network;

use std::{env, io};

use reversi_core::board::Board;

use crate::{eval::network::Network, search::search_context::SearchContext};

/// Macro for the WASM-specific evaluation weight file name.
macro_rules! eval_weights_literal {
    () => {
        "eval_wasm-882dcae6.zst"
    };
}

pub struct Eval {
    network: Network,
}

impl Eval {
    pub fn new() -> io::Result<Self> {
        Self::with_weight_files()
    }

    pub fn with_weight_files() -> io::Result<Self> {
        let network = Network::from_bytes(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../",
            eval_weights_literal!()
        )))?;

        Ok(Eval { network })
    }

    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> i32 {
        self.network
            .evaluate(board, ctx.get_pattern_feature(), ctx.ply())
    }
}
