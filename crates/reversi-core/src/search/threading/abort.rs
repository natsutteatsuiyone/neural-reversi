use std::sync::atomic::{AtomicU64, Ordering};

const SEARCH_GEN_MASK: u64 = 0xFFFF_FFFF_0000_0000;
const ABORT_GEN_MASK: u64 = 0x0000_0000_FFFF_FFFF;

/// Generation-tagged abort state shared by the thread pool, the timer thread,
/// and external abort callers.
///
/// High 32 bits: current search generation, bumped by [`Self::begin_search`].
/// Low 32 bits: an abort marker. Starting a search sets it to the previous
/// generation. Requesting an abort sets it to the current generation.
/// The current search is aborted iff the halves are equal.
///
/// An abort that happens strictly before `begin_search` targets the previous
/// generation. [`StopToken`](crate::search::StopToken) delivers a pending
/// stop with [`Self::request_abort_for`] after the new generation starts.
#[derive(Debug)]
pub(crate) struct AbortState(AtomicU64);

impl AbortState {
    /// Creates abort state with no active abort request.
    pub(crate) fn new() -> Self {
        Self(AtomicU64::new(ABORT_GEN_MASK))
    }

    /// Starts a new search generation, neutralizes stale abort requests, and
    /// returns the new generation.
    pub(crate) fn begin_search(&self) -> u32 {
        let previous = self.0.update(Ordering::AcqRel, Ordering::Acquire, |state| {
            let generation = (state >> 32) as u32;
            (u64::from(generation.wrapping_add(1)) << 32) | u64::from(generation)
        });
        ((previous >> 32) as u32).wrapping_add(1)
    }

    /// Records an abort request for the current search generation.
    pub(crate) fn request_abort(&self) {
        let _ = self
            .0
            .try_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                Some((state & SEARCH_GEN_MASK) | (state >> 32))
            });
    }

    /// Records an abort request for `generation`, or does nothing once a later
    /// search has begun.
    pub(crate) fn request_abort_for(&self, generation: u32) {
        let _ = self
            .0
            .try_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                ((state >> 32) as u32 == generation)
                    .then_some((state & SEARCH_GEN_MASK) | u64::from(generation))
            });
    }

    /// Returns whether the current search generation has an abort request.
    #[inline]
    pub(crate) fn is_aborted(&self) -> bool {
        let state = self.0.load(Ordering::Acquire);
        (state >> 32) == (state & ABORT_GEN_MASK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_aborted_initially() {
        let state = AbortState::new();

        assert!(!state.is_aborted());
    }

    #[test]
    fn abort_during_current_generation_is_observed() {
        let state = AbortState::new();

        state.begin_search();
        state.request_abort();

        assert!(state.is_aborted());
    }

    #[test]
    fn begin_search_neutralizes_prior_abort() {
        let state = AbortState::new();

        state.begin_search();
        state.request_abort();
        state.begin_search();

        assert!(!state.is_aborted());
    }

    #[test]
    fn begin_search_ignores_a_marker_matching_the_next_generation() {
        for (previous, next) in [(u32::MAX - 1, u32::MAX), (u32::MAX, 0)] {
            let word = (u64::from(previous) << 32) | u64::from(next);
            let state = AbortState(AtomicU64::new(word));

            assert_eq!(state.begin_search(), next);
            assert!(!state.is_aborted(), "generation {next} started aborted");
            state.request_abort_for(previous);
            assert!(!state.is_aborted());
            state.request_abort_for(next);
            assert!(state.is_aborted());
        }
    }

    #[test]
    fn abort_before_any_search_is_noop_after_begin() {
        let state = AbortState::new();

        state.request_abort();
        state.begin_search();

        assert!(!state.is_aborted());
    }

    #[test]
    fn abort_for_a_generation_reaches_only_that_generation() {
        let state = AbortState::new();

        let first = state.begin_search();
        state.request_abort_for(first);
        assert!(state.is_aborted());

        let second = state.begin_search();
        assert_eq!(second, first.wrapping_add(1));
        state.request_abort_for(first);
        assert!(!state.is_aborted());

        state.request_abort_for(second);
        assert!(state.is_aborted());
    }
}
