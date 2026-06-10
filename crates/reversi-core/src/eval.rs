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

pub mod eval_cache;
mod network;
mod network_small;
pub mod pattern_feature;
mod util;

/// Log2 of the number of evaluation cache entries.
const EVAL_CACHE_SIZE_LOG2: u32 = 18;

/// Which neural network to use for evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalMode {
    /// Use the main network for evaluation.
    Main,
    /// Use the small network for endgame (ply >= 30 only).
    Small,
}

macro_rules! eval_main_weights_literal {
    () => {
        "eval-e6bbc4f6.zst"
    };
}

macro_rules! eval_small_weights_literal {
    () => {
        "eval_sm-e6bbc4f6.zst"
    };
}

/// Filename for the main neural network weights (zstd compressed).
pub const EVAL_FILE_NAME: &str = eval_main_weights_literal!();

/// Filename for the small neural network weights (zstd compressed).
pub const EVAL_SM_FILE_NAME: &str = eval_small_weights_literal!();

/// A position evaluator backed by dual neural networks.
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
    #[inline(always)]
    pub fn evaluate(&self, ctx: &SearchContext, board: &Board) -> ScaledScore {
        if Self::should_use_main_network(ctx.eval_mode, ctx.ply()) {
            self.evaluate_main_with_key(ctx, board, board.hash())
        } else {
            self.evaluate_small(ctx)
        }
    }

    /// Returns whether the main network (and thus the eval cache) is used
    /// for the given `(eval_mode, ply)` pair.
    #[inline(always)]
    pub fn should_use_main_network(eval_mode: EvalMode, ply: usize) -> bool {
        eval_mode == EvalMode::Main || ply < ENDGAME_START_PLY
    }

    /// Evaluates the position with the main network and cache, using a precomputed `board.hash()`.
    ///
    /// Intended for the main-network path — see [`should_use_main_network`](Self::should_use_main_network).
    #[inline(always)]
    pub fn evaluate_main_with_key(
        &self,
        ctx: &SearchContext,
        board: &Board,
        key: u64,
    ) -> ScaledScore {
        if let Some(score_cache) = self.cache.probe(key) {
            return score_cache;
        }

        let score = self
            .network
            .evaluate(board, ctx.get_pattern_feature(), ctx.ply());
        self.cache.store(key, score);
        score
    }

    /// Evaluates the position with the small network (no cache).
    ///
    /// Intended for the small-network path — see [`should_use_main_network`](Self::should_use_main_network).
    #[inline(always)]
    pub fn evaluate_small(&self, ctx: &SearchContext) -> ScaledScore {
        self.network_sm
            .evaluate(ctx.get_pattern_feature(), ctx.ply())
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

    /// Software-prefetches the eval-cache line for `key`.
    ///
    /// Issue this between `make_move` and `evaluate` so the load overlaps
    /// with intervening SIMD work (e.g. pattern-feature updates). Only
    /// useful when the next eval call will probe the cache — see
    /// [`should_use_main_network`](Self::should_use_main_network).
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        self.cache.prefetch(key);
    }

    /// Clears the evaluation cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use std::path::Path;

    #[test]
    fn main_mode_uses_the_main_network_at_every_ply() {
        for ply in [
            0,
            1,
            ENDGAME_START_PLY - 1,
            ENDGAME_START_PLY,
            ENDGAME_START_PLY + 1,
            59,
        ] {
            assert!(
                Eval::should_use_main_network(EvalMode::Main, ply),
                "Main mode must use the main network at ply {ply}"
            );
        }
    }

    #[test]
    fn small_mode_switches_networks_at_the_endgame_boundary() {
        assert!(Eval::should_use_main_network(EvalMode::Small, 0));
        assert!(Eval::should_use_main_network(
            EvalMode::Small,
            ENDGAME_START_PLY - 1
        ));
        assert!(!Eval::should_use_main_network(
            EvalMode::Small,
            ENDGAME_START_PLY
        ));
        assert!(!Eval::should_use_main_network(
            EvalMode::Small,
            ENDGAME_START_PLY + 1
        ));
        assert!(!Eval::should_use_main_network(EvalMode::Small, 59));
    }

    #[test]
    fn missing_weights_error_is_not_found_and_names_the_file() {
        let err = missing_weights_error(Path::new("some/dir/eval-test.zst"));
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(
            err.to_string().contains("eval-test.zst"),
            "message should name the file: {err}"
        );
    }

    #[test]
    fn missing_weights_error_falls_back_when_path_has_no_file_name() {
        // `..` has no file name component, exercising the `unwrap_or` fallback.
        let err = missing_weights_error(Path::new(".."));
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        assert!(err.to_string().contains("weights file"));
    }

    #[test]
    fn evaluate_simple_returns_the_exact_terminal_score_with_no_empties() {
        let eval = Eval::with_weight_files(None, None).expect("embedded weights should load");
        // A full board has zero empties, so the network is bypassed for the
        // exact final score.
        let board = Board::from_bitboards(u64::MAX, 0);
        assert_eq!(board.get_empty_count(), 0);
        assert_eq!(eval.evaluate_simple(&board), board.final_score_scaled());
    }

    #[test]
    fn evaluate_simple_runs_the_network_for_a_non_terminal_position() {
        let eval = Eval::with_weight_files(None, None).expect("embedded weights should load");
        // The opening position has empty squares, so evaluate_simple takes the
        // network path (ply + pattern-feature construction) rather than the
        // terminal early return exercised above.
        let board = Board::new();
        assert!(board.get_empty_count() > 0);

        // evaluate_simple bypasses the eval cache, so repeated calls recompute
        // the same value deterministically.
        let score = eval.evaluate_simple(&board);
        assert_eq!(eval.evaluate_simple(&board), score);
    }

    #[test]
    fn evaluate_simple_is_stable_across_interleaved_positions() {
        use crate::square::Square;

        // The main network reuses a per-thread scratch buffer, so a
        // read-before-write would leak one position's leftover state into the
        // next. Interleaving two positions and re-checking each guards that.
        let eval = Eval::with_weight_files(None, None).expect("embedded weights should load");

        let board_a = Board::new();
        let board_b = board_a.make_move(Square::D3);

        let a1 = eval.evaluate_simple(&board_a);
        let b1 = eval.evaluate_simple(&board_b);
        // board_b dirtied the scratch buffer; re-evaluating board_a must be unaffected.
        let a2 = eval.evaluate_simple(&board_a);
        let b2 = eval.evaluate_simple(&board_b);

        assert_eq!(
            a1, a2,
            "A must evaluate identically after the buffer was dirtied by B"
        );
        assert_eq!(
            b1, b2,
            "B must evaluate identically after the buffer was dirtied by A"
        );
        assert_ne!(
            a1, b1,
            "the two positions must differ so the test actually exercises cross-position reuse"
        );
    }
}
