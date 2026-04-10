//! Diagnostic counters collected during search.

/// Accumulated search statistics for diagnostic purposes.
///
/// Tracks how often various pruning and caching mechanisms fire during
/// a search, enabling performance analysis without affecting search behavior.
#[derive(Debug, Clone, Default)]
pub struct SearchCounters {
    /// Total number of nodes visited during search.
    pub n_nodes: u64,
    /// Number of transposition table probes.
    pub tt_probes: u64,
    /// Number of transposition table cutoffs (probe succeeded and caused a cutoff).
    pub tt_hits: u64,
    /// Number of ProbCut attempts (entered the ProbCut routine).
    pub probcut_attempts: u64,
    /// Number of ProbCut cutoffs (shallow search confirmed the cutoff).
    pub probcut_cuts: u64,
    /// Number of Enhanced Transposition Cutoff attempts.
    pub etc_attempts: u64,
    /// Number of ETC cutoffs.
    pub etc_cuts: u64,
    /// Number of stability cutoffs.
    pub stability_cuts: u64,
}

impl SearchCounters {
    /// Merges counters from another context (e.g. a parallel search thread).
    pub fn merge(&mut self, other: &SearchCounters) {
        self.n_nodes += other.n_nodes;
        self.tt_probes += other.tt_probes;
        self.tt_hits += other.tt_hits;
        self.probcut_attempts += other.probcut_attempts;
        self.probcut_cuts += other.probcut_cuts;
        self.etc_attempts += other.etc_attempts;
        self.etc_cuts += other.etc_cuts;
        self.stability_cuts += other.stability_cuts;
    }
}
