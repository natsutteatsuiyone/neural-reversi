//! Stop requests that end a search run early.

use std::sync::{Arc, Mutex};

use super::threading::AbortState;

/// A caller-held request to end a search run early.
///
/// Hand a clone to a run with [`SearchRunOptions::stop_token`] and call
/// [`StopToken::stop`] from any thread. A stopped run still returns its best
/// move so far, or a fallback legal move when no iteration completed. The
/// request reaches every run that carries the token, even one that has not
/// started yet, and never a run that does not.
///
/// [`SearchRunOptions::stop_token`]: super::SearchRunOptions::stop_token
#[derive(Clone, Debug, Default)]
pub struct StopToken(Arc<Mutex<StopState>>);

#[derive(Debug, Default)]
struct StopState {
    stopped: bool,
    /// Runs in progress that carry the token, as their pool's abort state
    /// and search generation.
    runs: Vec<(Arc<AbortState>, u32)>,
}

impl StopToken {
    /// Creates a token that has not been stopped.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stops every run that carries this token, now or later.
    ///
    /// Calling it again has no further effect.
    pub fn stop(&self) {
        let mut state = self.0.lock().unwrap();
        state.stopped = true;
        for (abort_state, generation) in &state.runs {
            abort_state.request_abort_for(*generation);
        }
    }

    /// Ties the token to search `generation` of `abort_state` until the
    /// returned guard drops. A token stopped earlier aborts that search at once.
    pub(crate) fn attach(&self, abort_state: &Arc<AbortState>, generation: u32) -> StopAttachment {
        let mut state = self.0.lock().unwrap();
        if state.stopped {
            abort_state.request_abort_for(generation);
        }
        state.runs.push((Arc::clone(abort_state), generation));
        StopAttachment {
            token: self.clone(),
            abort_state: Arc::clone(abort_state),
            generation,
        }
    }
}

/// Keeps a [`StopToken`] tied to one search until dropped.
pub(crate) struct StopAttachment {
    token: StopToken,
    abort_state: Arc<AbortState>,
    generation: u32,
}

impl Drop for StopAttachment {
    fn drop(&mut self) {
        let mut state = self.token.0.lock().unwrap();
        state.runs.retain(|(abort_state, generation)| {
            !(Arc::ptr_eq(abort_state, &self.abort_state) && *generation == self.generation)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_before_the_run_starts_aborts_that_run() {
        let abort_state = Arc::new(AbortState::new());
        let token = StopToken::new();
        token.stop();

        let _attachment = token.attach(&abort_state, abort_state.begin_search());

        assert!(abort_state.is_aborted());
    }

    #[test]
    fn stop_during_the_run_aborts_it() {
        let abort_state = Arc::new(AbortState::new());
        let token = StopToken::new();
        let _attachment = token.attach(&abort_state, abort_state.begin_search());
        assert!(!abort_state.is_aborted());

        token.stop();

        assert!(abort_state.is_aborted());
    }

    #[test]
    fn stop_after_the_run_ended_does_not_reach_the_next_run() {
        let abort_state = Arc::new(AbortState::new());
        let token = StopToken::new();
        drop(token.attach(&abort_state, abort_state.begin_search()));
        abort_state.begin_search();

        token.stop();

        assert!(!abort_state.is_aborted());
    }
}
