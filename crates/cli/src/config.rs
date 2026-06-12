//! Resolved engine configuration shared by every CLI mode.

use std::path::PathBuf;

use reversi_core::probcut::Selectivity;

/// Engine parameters resolved from CLI arguments.
///
/// Passing this one struct to each mode replaces threading six positional
/// arguments through every entry point.
pub struct EngineConfig {
    pub hash_size: usize,
    pub level: usize,
    pub selectivity: Selectivity,
    pub threads: Option<usize>,
    pub eval_file: Option<PathBuf>,
    pub eval_sm_file: Option<PathBuf>,
}
