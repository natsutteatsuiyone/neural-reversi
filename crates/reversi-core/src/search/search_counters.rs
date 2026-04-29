//! Diagnostic counters collected during search.
//!
//! `n_nodes` is always live (used for NPS reporting). All other counters are
//! gated on the `search-stats` Cargo feature so the hot path stays cold in
//! production builds; only `evaltest` opts in. The `increment_*` methods
//! compile to a no-op when the feature is off.

macro_rules! search_stats {
    ($($field:ident => $inc:ident),* $(,)?) => {
        /// Accumulated search statistics for diagnostic purposes.
        ///
        /// Tracks how often various pruning and caching mechanisms fire during
        /// a search, enabling performance analysis without affecting search behavior.
        #[derive(Debug, Clone, Default)]
        pub struct SearchCounters {
            /// Total number of nodes visited during search.
            pub n_nodes: u64,
            $(
                #[cfg(feature = "search-stats")]
                pub $field: u64,
            )*
        }

        impl SearchCounters {
            /// Merges counters from another context (e.g. a parallel search thread).
            pub fn merge(&mut self, other: &SearchCounters) {
                self.n_nodes += other.n_nodes;
                $(
                    #[cfg(feature = "search-stats")]
                    { self.$field += other.$field; }
                )*
            }

            #[inline(always)]
            pub(crate) fn increment_nodes(&mut self) {
                self.n_nodes += 1;
            }

            $(
                #[inline(always)]
                pub(crate) fn $inc(&mut self) {
                    #[cfg(feature = "search-stats")]
                    { self.$field += 1; }
                }
            )*
        }
    };
}

search_stats! {
    tt_probes         => increment_tt_probe,
    tt_hits           => increment_tt_hit,
    probcut_attempts  => increment_probcut_attempt,
    probcut_cuts      => increment_probcut_cut,
    etc_attempts      => increment_etc_attempt,
    etc_cuts          => increment_etc_cut,
    stability_cuts    => increment_stability_cut,
}
