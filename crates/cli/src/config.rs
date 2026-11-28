//! Resolved engine configuration shared by every CLI mode.

use std::path::PathBuf;
use std::sync::Arc;

use reversi_core::engine::Engine;
use reversi_core::eval::Eval;

/// Engine parameters resolved from CLI arguments.
pub struct EngineConfig {
    pub hash_size: usize,
    pub level: usize,
    pub threads: Option<usize>,
    pub eval_file: Option<PathBuf>,
    pub eval_sm_file: Option<PathBuf>,
}

impl EngineConfig {
    /// Loads the evaluation networks and builds the engine.
    ///
    /// # Errors
    ///
    /// Returns an error naming the weight file if a configured one cannot be
    /// read or is not a network of the expected format.
    pub fn engine(&self) -> Result<Engine, String> {
        let eval = Eval::with_weight_files(self.eval_file.as_deref(), self.eval_sm_file.as_deref())
            .map_err(|e| e.to_string())?;
        Ok(Engine::from_eval(
            Arc::new(eval),
            self.hash_size,
            self.threads,
        ))
    }
}

#[cfg(test)]
pub(crate) fn test_engine(tt_mb_size: usize, n_threads: usize) -> Engine {
    use std::sync::OnceLock;

    static EVAL: OnceLock<Arc<Eval>> = OnceLock::new();
    let eval = EVAL
        .get_or_init(|| {
            Arc::new(
                Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
            )
        })
        .clone();
    Engine::from_eval(eval, tt_mb_size, Some(n_threads))
}
