mod activations;
mod base_input;
pub mod constants;
mod eval_cache;
mod linear_layer;
mod network;
mod network_small;
pub mod pattern_feature;
mod phase_adaptive_input;

use std::io;
use std::env;

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
    pub fn new() -> io::Result<Self> {
        let exe_path = env::current_exe().map_err(|e| io::Error::other(format!("Failed to get current executable path: {e}")))?;
        let exe_dir = exe_path.parent().ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Failed to get parent directory of executable"))?;

        let eval_file_path = exe_dir.join(EVAL_FILE_NAME);
        let eval_sm_file_path = exe_dir.join(EVAL_SM_FILE_NAME);

        if !eval_file_path.exists() {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("\"{}\" not found: {}", EVAL_FILE_NAME, eval_file_path.display())));
        }
        if !eval_sm_file_path.exists() {
            return Err(io::Error::new(io::ErrorKind::NotFound, format!("\"{}\" not found: {}", EVAL_SM_FILE_NAME, eval_sm_file_path.display())));
        }

        let network = Network::new(eval_file_path.to_str().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Failed to convert eval_file_path to str"))?)?;
        let network_sm = NetworkSmall::new(eval_sm_file_path.to_str().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Failed to convert eval_sm_file_path to str"))?)?;
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
        if ctx.game_phase == GamePhase::MidGame {
            if let Some(score_cache) = self.cache.probe(key) {
                return score_cache;
            }

            let score = self.network.evaluate(board, ctx.get_pattern_feature(), ctx.ply());
            self.cache.store(key, score);
            score
        } else {
            if let Some(score_cache) = self.cache_sm.probe(key) {
                return score_cache;
            }

            let score = self.network_sm.evaluate(board, ctx.get_pattern_feature(), ctx.ply());
            self.cache_sm.store(key, score);
            score
        }
    }
}
