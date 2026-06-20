//! Resolved engine configuration shared by every CLI mode.

use std::path::PathBuf;

use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;

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

impl EngineConfig {
    /// Builds the [`SearchOptions`] for this configuration after verifying that
    /// any explicitly supplied weight file exists.
    ///
    /// If a configured weight file is missing, this prints a diagnostic and
    /// exits the process, giving every CLI mode the same early failure before a
    /// search starts.
    pub fn search_options(&self) -> SearchOptions {
        for path in [self.eval_file.as_deref(), self.eval_sm_file.as_deref()]
            .into_iter()
            .flatten()
        {
            if !path.exists() {
                eprintln!("Weight file does not exist: {}", path.display());
                std::process::exit(1);
            }
        }

        SearchOptions::new(self.hash_size)
            .with_threads(self.threads)
            .with_eval_paths(self.eval_file.as_deref(), self.eval_sm_file.as_deref())
    }
}
