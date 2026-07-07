use std::sync::atomic::{AtomicU64, Ordering};

const SEARCH_GEN_INCREMENT: u64 = 1 << 32;
const SEARCH_GEN_MASK: u64 = 0xFFFF_FFFF_0000_0000;
const ABORT_GEN_MASK: u64 = 0x0000_0000_FFFF_FFFF;

/// Generation-tagged abort state shared by the thread pool, the timer thread,
/// and external abort callers.
///
/// High 32 bits: current search generation, bumped by [`Self::begin_search`].
/// Low 32 bits: the generation an abort was requested for.
/// The current search is aborted iff the halves are equal. Because an abort
/// is recorded as a generation match rather than a resettable flag, a racing
/// abort can never be erased by the start of a search; conversely a stale
/// abort from a finished generation never leaks into the next one.
///
/// An abort that happens strictly before `begin_search` targets the previous
/// generation and is a no-op; callers that need "abort the search I am about
/// to start" must serialize on their side.
///
/// The search generation is 32 bits wide. A stale abort can collide with a
/// recycled generation only after exactly 2^32 intervening searches without
/// another abort request.
#[derive(Debug)]
pub(crate) struct AbortState(AtomicU64);

impl AbortState {
    /// Creates abort state with no active abort request.
    pub(crate) fn new() -> Self {
        Self(AtomicU64::new(ABORT_GEN_MASK))
    }

    /// Starts a new search generation and neutralizes stale abort requests.
    pub(crate) fn begin_search(&self) {
        self.0.fetch_add(SEARCH_GEN_INCREMENT, Ordering::AcqRel);
    }

    /// Records an abort request for the current search generation.
    pub(crate) fn request_abort(&self) {
        let _ = self
            .0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                Some((state & SEARCH_GEN_MASK) | (state >> 32))
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
    fn abort_before_any_search_is_noop_after_begin() {
        let state = AbortState::new();

        state.request_abort();
        state.begin_search();

        assert!(!state.is_aborted());
    }

    #[test]
    fn request_abort_is_idempotent() {
        let state = AbortState::new();

        state.begin_search();
        state.request_abort();
        state.request_abort();

        assert!(state.is_aborted());

        state.begin_search();

        assert!(!state.is_aborted());
    }
}
