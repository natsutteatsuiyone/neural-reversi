//! Search options and configuration.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::constants::MAX_THREADS;
use crate::eval::EvalMode;
use crate::level::Level;
use crate::probcut::Selectivity;

use super::SearchProgressCallback;
use super::time_control::TimeControlMode;

/// Configuration options for search initialization.
pub struct SearchOptions {
    pub tt_mb_size: usize,
    pub n_threads: usize,
    pub eval_path: Option<PathBuf>,
    pub eval_sm_path: Option<PathBuf>,
}

impl SearchOptions {
    /// Creates search options with the specified transposition table size and defaults for
    /// other parameters.
    #[must_use]
    pub fn new(tt_mb_size: usize) -> Self {
        SearchOptions {
            tt_mb_size,
            ..Default::default()
        }
    }

    /// Overrides the number of search threads.
    #[must_use]
    pub fn with_threads(mut self, n_threads: Option<usize>) -> Self {
        if let Some(value) = n_threads {
            self.n_threads = value;
        }
        self
    }

    /// Sets custom paths for the neural network weight files.
    #[must_use]
    pub fn with_eval_paths<P, Q>(mut self, eval_path: Option<P>, eval_sm_path: Option<Q>) -> Self
    where
        P: AsRef<Path>,
        Q: AsRef<Path>,
    {
        self.eval_path = eval_path.map(|p| p.as_ref().to_path_buf());
        self.eval_sm_path = eval_sm_path.map(|p| p.as_ref().to_path_buf());
        self
    }
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions {
            tt_mb_size: 64,
            n_threads: num_cpus::get().min(MAX_THREADS),
            eval_path: None,
            eval_sm_path: None,
        }
    }
}

/// Search constraint definition.
pub enum SearchConstraint {
    Level(Level),
    Time(TimeControlMode),
}

/// Options for a single search run.
pub struct SearchRunOptions {
    pub constraint: SearchConstraint,
    pub selectivity: Selectivity,
    pub multi_pv: bool,
    pub callback: Option<Arc<SearchProgressCallback>>,
    pub eval_mode: Option<EvalMode>,
}

impl SearchRunOptions {
    /// Creates search run options with a level constraint.
    #[must_use]
    pub fn with_level(level: Level, selectivity: Selectivity) -> Self {
        SearchRunOptions {
            constraint: SearchConstraint::Level(level),
            selectivity,
            multi_pv: false,
            callback: None,
            eval_mode: None,
        }
    }

    /// Creates search run options with a time constraint.
    #[must_use]
    pub fn with_time(mode: TimeControlMode, selectivity: Selectivity) -> Self {
        SearchRunOptions {
            constraint: SearchConstraint::Time(mode),
            selectivity,
            multi_pv: false,
            callback: None,
            eval_mode: None,
        }
    }

    /// Enables multi-PV mode.
    #[must_use]
    pub fn multi_pv(mut self, enabled: bool) -> Self {
        self.multi_pv = enabled;
        self
    }

    /// Sets the progress callback.
    #[must_use]
    pub fn callback<F>(mut self, f: F) -> Self
    where
        F: Fn(super::SearchProgress) + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(f));
        self
    }

    /// Forces a specific evaluation mode.
    #[must_use]
    pub fn with_eval_mode(mut self, mode: EvalMode) -> Self {
        self.eval_mode = Some(mode);
        self
    }
}
