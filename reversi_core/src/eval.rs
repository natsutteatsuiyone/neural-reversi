mod base_input;
pub mod constants;
mod eval_cache;
mod linear_layer;
mod network;
mod network_small;
pub mod pattern_feature;
mod phase_adaptive_input;
mod relu;

use std::io;

use constants::*;
use eval_cache::EvalCache;
use network::Network;
use network_small::NetworkSmall;

use crate::board::Board;
use crate::search::search_context::{GamePhase, SearchContext};
use crate::types::Score;


pub struct Eval {
    network: Network,
    network_sm: NetworkSmall,
    pub cache: EvalCache,
    pub cache_sm: EvalCache,
}

impl Eval {
    pub fn new(file_path: &str, small_file_path: &str) -> io::Result<Self> {
        let network = Network::new(file_path)?;
        let network_sm = NetworkSmall::new(small_file_path)?;
        Ok(Eval {
            network,
            network_sm,
            cache: EvalCache::new(17),
            cache_sm: EvalCache::new(17),
        })
    }

    /// Evaluate the current position.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The search context.
    /// * `board` - The current board.
    ///
    /// # Returns
    ///
    /// The evaluation score of the current position.
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> Score {
        let key = board.hash();
        if ctx.game_phase == GamePhase::MidGame || ctx.empty_list.count > 30 {
            if let Some(score_cache) = self.cache.probe(key) {
                return score_cache;
            }

            let score = self.network.evaluate(ctx, board);
            self.cache.store(key, score);
            score
        } else {
            if let Some(score_cache) = self.cache_sm.probe(key) {
                return score_cache;
            }

            let score = self.network_sm.evaluate(ctx, board);
            self.cache_sm.store(key, score);
            score
        }
    }
}
