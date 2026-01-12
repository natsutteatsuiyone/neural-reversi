//! Neural network-based position evaluation.
//!
//! This module provides phase-adaptive evaluation using two neural networks:
//! - Main network: Used for midgame and early positions (ply < 30)
//! - Small network: Used for endgame positions

mod activations;
pub mod eval_cache;
mod input_layer;
mod linear_layer;
mod network;
mod network_small;
mod output_layer;
pub mod pattern_feature;
mod util;

use std::env;
use std::io;
use std::path::Path;

use eval_cache::EvalCache;
pub use network::Network;
pub use network_small::NetworkSmall;

use crate::board::Board;
use crate::search::search_context::{GamePhase, SearchContext};
use crate::types::{ScaledScore, Score};

macro_rules! eval_main_weights_literal {
    () => {
        "eval-882dcae6.zst"
    };
}

macro_rules! eval_small_weights_literal {
    () => {
        "eval_sm-8d62dc63.zst"
    };
}

/// Filename for the main neural network weights (zstd compressed).
pub const EVAL_FILE_NAME: &str = eval_main_weights_literal!();

/// Filename for the small neural network weights (zstd compressed).
pub const EVAL_SM_FILE_NAME: &str = eval_small_weights_literal!();

/// Neural network evaluator.
pub struct Eval {
    /// Main neural network for midgame evaluation.
    network: Network,
    /// Small neural network for endgame evaluation.
    network_sm: NetworkSmall,
    /// Evaluation cache to avoid redundant computation.
    pub cache: EvalCache,
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
    /// Creates a new `Eval` with default or override weight files.
    ///
    /// Looks for weight files in the executable's directory. If not found,
    /// uses embedded weights.
    pub fn new() -> io::Result<Self> {
        let exe_path = env::current_exe()?;
        let exe_dir = exe_path.parent().unwrap();

        let eval_file_path = exe_dir.join(EVAL_FILE_NAME);
        let eval_sm_file_path = exe_dir.join(EVAL_SM_FILE_NAME);

        let eval_override = eval_file_path.is_file().then_some(eval_file_path);
        let eval_sm_override = eval_sm_file_path.is_file().then_some(eval_sm_file_path);

        Self::with_weight_files(eval_override.as_deref(), eval_sm_override.as_deref())
    }

    /// Creates a new `Eval` with specified weight files.
    ///
    /// # Arguments
    ///
    /// * `eval_path` - Path to main network weights, or `None` for embedded.
    /// * `eval_sm_path` - Path to small network weights, or `None` for embedded.
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
                "/../../",
                eval_main_weights_literal!()
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
                "/../../",
                eval_small_weights_literal!()
            ))),
        }?;

        Ok(Eval {
            network,
            network_sm,
            cache: EvalCache::new(17),
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
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> ScaledScore {
        if ctx.game_phase == GamePhase::MidGame || ctx.ply() < 30 {
            let key = board.hash();
            if let Some(score_cache) = self.cache.probe(key) {
                return score_cache;
            }

            let score = self
                .network
                .evaluate(board, ctx.get_pattern_feature(), ctx.ply());
            self.cache.store(key, score);
            score
        } else {
            self.network_sm
                .evaluate(ctx.get_pattern_feature(), ctx.ply())
        }
    }

    /// Simple evaluation without SearchContext.
    ///
    /// This is a convenience method for quick move selection when
    /// there's no SearchContext available. It creates pattern features
    /// on the fly, so it's slower than the cached version.
    ///
    /// # Arguments
    ///
    /// * `board` - The current board position.
    ///
    /// # Returns
    ///
    /// The evaluation score of the current position.
    pub fn evaluate_simple(&self, board: &Board) -> ScaledScore {
        let n_empties = board.get_empty_count() as usize;
        if n_empties == 0 {
            let final_score = board.get_player_count() as Score * 2 - 64;
            return ScaledScore::from_disc_diff(final_score);
        }

        let ply = 60 - n_empties;
        let pattern_features = pattern_feature::PatternFeatures::new(board, ply);

        self.network
            .evaluate(board, &pattern_features.p_features[ply], ply)
    }
}
