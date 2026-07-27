//! Search options and configuration.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::constants::MAX_THREADS;
use crate::eval::EvalMode;
use crate::level::Level;

use super::SearchProgressCallback;
use super::time_control::TimeControlMode;

/// Number of CPUs available to this process, falling back to 1.
pub(crate) fn available_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

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
            tt_mb_size: 512,
            n_threads: available_cpus().min(MAX_THREADS),
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
    pub multi_pv: bool,
    pub callback: Option<Arc<SearchProgressCallback>>,
    pub eval_mode: Option<EvalMode>,
    pub probcut_disabled: bool,
}

impl SearchRunOptions {
    /// Creates search run options with a level constraint.
    #[must_use]
    pub fn with_level(level: Level) -> Self {
        SearchRunOptions {
            constraint: SearchConstraint::Level(level),
            multi_pv: false,
            callback: None,
            eval_mode: None,
            probcut_disabled: false,
        }
    }

    /// Creates search run options with a time constraint.
    #[must_use]
    pub fn with_time(mode: TimeControlMode) -> Self {
        SearchRunOptions {
            constraint: SearchConstraint::Time(mode),
            multi_pv: false,
            callback: None,
            eval_mode: None,
            probcut_disabled: false,
        }
    }

    /// Disables midgame ProbCut pruning and the reductions gated on it,
    /// yielding unpruned midgame scores for data generation.
    #[must_use]
    pub fn disable_probcut(mut self) -> Self {
        self.probcut_disabled = true;
        self
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_thread_count_stays_within_bounds() {
        // The default caps the machine's parallelism at MAX_THREADS.
        let opts = SearchOptions::default();
        assert!(opts.n_threads <= MAX_THREADS);
    }

    #[test]
    fn with_threads_overrides_only_when_some() {
        let default_threads = SearchOptions::default().n_threads;
        assert_eq!(
            SearchOptions::new(64).with_threads(None).n_threads,
            default_threads
        );
        assert_eq!(SearchOptions::new(64).with_threads(Some(3)).n_threads, 3);
    }

    #[test]
    fn with_eval_paths_maps_optional_paths() {
        let opts = SearchOptions::new(64).with_eval_paths(Some("a.zst"), Some("b.zst"));
        assert_eq!(opts.eval_path.as_deref(), Some(Path::new("a.zst")));
        assert_eq!(opts.eval_sm_path.as_deref(), Some(Path::new("b.zst")));

        let cleared = SearchOptions::new(64).with_eval_paths::<&str, &str>(None, None);
        assert!(cleared.eval_path.is_none());
        assert!(cleared.eval_sm_path.is_none());
    }

    #[test]
    fn run_options_with_level_sets_a_level_constraint() {
        let opts = SearchRunOptions::with_level(Level::unlimited());
        assert!(matches!(opts.constraint, SearchConstraint::Level(_)));
        assert!(!opts.probcut_disabled);
    }

    #[test]
    fn run_options_with_time_sets_a_time_constraint() {
        let opts = SearchRunOptions::with_time(TimeControlMode::Infinite);
        assert!(matches!(opts.constraint, SearchConstraint::Time(_)));
        assert!(!opts.probcut_disabled);
    }

    #[test]
    fn disable_probcut_sets_the_flag() {
        let opts = SearchRunOptions::with_level(Level::unlimited()).disable_probcut();
        assert!(opts.probcut_disabled);
    }
}
