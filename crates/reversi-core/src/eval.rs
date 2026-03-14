//! Neural network-based position evaluation.
//!
//! This module provides phase-adaptive evaluation using two neural networks:
//! - Main network: General-purpose network for all positions
//! - Small network: Optimized for endgame (ply >= 30 only)

use std::env;
use std::io;
use std::path::Path;

use eval_cache::EvalCache;
pub use network::Network;
pub use network_small::NetworkSmall;

use crate::board::Board;
use crate::constants::INITIAL_EMPTY_COUNT;
use crate::search::search_context::SearchContext;
use crate::types::ScaledScore;

use self::network_small::ENDGAME_START_PLY;

mod activations;
pub mod eval_cache;
mod input_layer;
mod linear_layer;
mod network;
mod network_small;
mod output_layer;
pub mod pattern_feature;
mod util;

/// Log2 of the number of evaluation cache entries.
const EVAL_CACHE_SIZE_LOG2: u32 = 17;

/// Selects which neural network to use for evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Use the main network for evaluation.
    Main,
    /// Use the small network for endgame (ply >= 30 only).
    Small,
}

macro_rules! eval_main_weights_literal {
    () => {
        "eval-dbdbed46.zst"
    };
}

macro_rules! eval_small_weights_literal {
    () => {
        "eval_sm-69fb3156.zst"
    };
}

/// Filename for the main neural network weights (zstd compressed).
pub const EVAL_FILE_NAME: &str = eval_main_weights_literal!();

/// Filename for the small neural network weights (zstd compressed).
pub const EVAL_SM_FILE_NAME: &str = eval_small_weights_literal!();

/// Evaluates board positions using dual neural networks.
pub struct Eval {
    /// Main neural network for early and midgame evaluation.
    network: Network,
    /// Small network optimized for endgame evaluation.
    network_sm: NetworkSmall,
    /// Evaluation cache to avoid redundant neural network computation.
    cache: EvalCache,
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
    /// Creates a new [`Eval`] using weight files from the executable's directory,
    /// falling back to embedded weights.
    pub fn new() -> io::Result<Self> {
        let exe_path = env::current_exe()?;
        let exe_dir = exe_path.parent().ok_or_else(|| {
            io::Error::other(format!(
                "Cannot determine executable directory: path '{}' has no parent",
                exe_path.display()
            ))
        })?;

        let eval_file_path = exe_dir.join(EVAL_FILE_NAME);
        let eval_sm_file_path = exe_dir.join(EVAL_SM_FILE_NAME);

        let eval_override = eval_file_path.is_file().then_some(eval_file_path);
        let eval_sm_override = eval_sm_file_path.is_file().then_some(eval_sm_file_path);

        Self::with_weight_files(eval_override.as_deref(), eval_sm_override.as_deref())
    }

    /// Creates a new [`Eval`] with specified weight file paths, or [`None`] for embedded weights.
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
            cache: EvalCache::new(EVAL_CACHE_SIZE_LOG2),
        })
    }

    /// Evaluates the current position.
    ///
    /// Network selection:
    /// - `ply < 30`: Always uses main network (small network does not support early/midgame)
    /// - `ply >= 30`: Uses small network when [`EvalMode::Small`], otherwise main network
    ///
    /// Only main network evaluations are cached; the small network is fast enough without caching.
    #[inline]
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> ScaledScore {
        if ctx.eval_mode == EvalMode::Main || ctx.ply() < ENDGAME_START_PLY {
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

    /// Evaluates a position without [`SearchContext`].
    ///
    /// Uses the main network with pattern features created on the fly, so it is
    /// slower than [`evaluate`](Self::evaluate). Returns the exact final score
    /// when the board has no empties.
    pub fn evaluate_simple(&self, board: &Board) -> ScaledScore {
        let n_empties = board.get_empty_count() as usize;
        if n_empties == 0 {
            return board.final_score_scaled();
        }

        let ply = INITIAL_EMPTY_COUNT - n_empties;
        let pattern_features = pattern_feature::PatternFeatures::new(board, ply);

        self.network
            .evaluate(board, pattern_features.p_feature(ply), ply)
    }

    /// Clears the evaluation cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}
