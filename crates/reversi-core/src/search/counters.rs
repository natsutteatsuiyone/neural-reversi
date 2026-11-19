//! Diagnostic counters collected during search.
//!
//! `n_nodes` is always live (used for NPS reporting). All other counters are
//! gated on the `search-stats` Cargo feature so the hot path stays cold in
//! production builds; only `search-test` opts in. The `increment_*` methods
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
            /// Returns the work recorded since an earlier progress snapshot.
            #[cfg(feature = "search-stats")]
            #[must_use]
            pub fn delta_since(&self, previous: &Self) -> Self {
                Self {
                    n_nodes: self.n_nodes.saturating_sub(previous.n_nodes),
                    $($field: self.$field.saturating_sub(previous.$field),)*
                }
            }

            /// Merges counters from another context (e.g. a parallel search thread).
            pub fn merge(&mut self, other: &SearchCounters) {
                self.n_nodes += other.n_nodes;
                $(
                    #[cfg(feature = "search-stats")]
                    { self.$field += other.$field; }
                )*
            }
        }

        /// The counters of [`SearchCounters`] other than `n_nodes`.
        ///
        /// `SearchContext` keeps them apart from `n_nodes` so that enabling
        /// `search-stats` does not move its hot fields.
        #[derive(Debug, Clone, Default)]
        pub(crate) struct SearchStats {
            $(
                #[cfg(feature = "search-stats")]
                $field: u64,
            )*
        }

        impl SearchStats {
            /// Returns these counters together with `n_nodes`.
            pub(crate) fn with_nodes(&self, n_nodes: u64) -> SearchCounters {
                SearchCounters {
                    n_nodes,
                    $(
                        #[cfg(feature = "search-stats")]
                        $field: self.$field,
                    )*
                }
            }

            /// Adds the counters of `other` other than `n_nodes`.
            #[cfg_attr(not(feature = "search-stats"), expect(unused_variables))]
            pub(crate) fn merge(&mut self, other: &SearchCounters) {
                $(
                    #[cfg(feature = "search-stats")]
                    { self.$field += other.$field; }
                )*
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
    tt_penalizes      => increment_tt_penalize,
    probcut_attempts  => increment_probcut_attempt,
    probcut_cuts      => increment_probcut_cut,
    tt_probcut_attempts => increment_tt_probcut_attempt,
    tt_probcut_cuts   => increment_tt_probcut_cut,
    etc_attempts      => increment_etc_attempt,
    etc_cuts          => increment_etc_cut,
    stability_cuts    => increment_stability_cut,
}
