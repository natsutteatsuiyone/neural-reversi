mod activations;
pub mod constants;
mod eval_cache;
mod input_layer;
mod linear_layer;
mod network;
mod network_small;
pub mod pattern_feature;

use std::env;
use std::io;
use std::path::Path;

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

fn missing_weights_error(path: &Path) -> io::Error {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("weights file");
    io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "Missing weights \"{name}\".\nExpected to find the file at: {path}.",
            path = path.display(),
        ),
    )
}

impl Eval {
    pub fn new() -> io::Result<Self> {
        let exe_path = env::current_exe()?;
        let exe_dir = exe_path.parent().unwrap();

        let eval_file_path = exe_dir.join(EVAL_FILE_NAME);
        let eval_sm_file_path = exe_dir.join(EVAL_SM_FILE_NAME);

        let eval_override = eval_file_path.is_file().then_some(eval_file_path);
        let eval_sm_override = eval_sm_file_path.is_file().then_some(eval_sm_file_path);

        Self::with_weight_files(eval_override.as_deref(), eval_sm_override.as_deref())
    }

    pub fn with_weight_files(
        eval_path: Option<&Path>,
        eval_sm_path: Option<&Path>,
    ) -> io::Result<Self> {
        let network = match eval_path {
            Some(path) => match Network::new(path) {
                Err(err) if err.kind() == io::ErrorKind::NotFound => {
                    Err(missing_weights_error(path))
                }
                other => other,
            },
            None => Network::from_bytes(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../",
                constants::eval_main_weights_literal!()
            ))),
        }?;

        let network_sm = match eval_sm_path {
            Some(path) => match NetworkSmall::new(path) {
                Err(err) if err.kind() == io::ErrorKind::NotFound => {
                    Err(missing_weights_error(path))
                }
                other => other,
            },
            None => NetworkSmall::from_bytes(include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../",
                constants::eval_small_weights_literal!()
            ))),
        }?;

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
        if ctx.game_phase == GamePhase::MidGame || ctx.ply() < 30 {
            if let Some(score_cache) = self.cache.probe(key) {
                return score_cache;
            }

            let score = self
                .network
                .evaluate(board, ctx.get_pattern_feature(), ctx.ply());
            self.cache.store(key, score);
            score
        } else {
            if let Some(score_cache) = self.cache_sm.probe(key) {
                return score_cache;
            }

            let score = self
                .network_sm
                .evaluate(board, ctx.get_pattern_feature(), ctx.ply());
            self.cache_sm.store(key, score);
            score
        }
    }
}
