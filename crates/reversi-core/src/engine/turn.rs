//! The search turn of an engine.

use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::search::StopToken;

/// How often a waiter checks whether its [`StopToken`] was stopped.
const STOP_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Whether a search holds the thread pool of an engine.
#[derive(Default)]
pub(crate) struct TurnState {
    busy: Mutex<bool>,
    freed: Condvar,
}

impl TurnState {
    /// Waits for the turn.
    pub(crate) fn wait(self: &Arc<Self>) -> SearchTurn {
        let mut busy = self.busy.lock().unwrap();
        while *busy {
            busy = self.freed.wait(busy).unwrap();
        }
        *busy = true;
        SearchTurn(Arc::clone(self))
    }

    /// Waits for the turn, or returns [`None`] once `stop` is stopped.
    pub(super) fn wait_unless_stopped(self: &Arc<Self>, stop: &StopToken) -> Option<SearchTurn> {
        let mut busy = self.busy.lock().unwrap();
        loop {
            if stop.is_stopped() {
                return None;
            }
            if !*busy {
                break;
            }
            busy = self.freed.wait_timeout(busy, STOP_POLL_INTERVAL).unwrap().0;
        }
        *busy = true;
        Some(SearchTurn(Arc::clone(self)))
    }
}

/// Exclusive use of the thread pool of an [`Engine`], which runs one search
/// at a time.
///
/// Dropping it passes the turn on to a waiting search.
///
/// [`Engine`]: super::Engine
pub(crate) struct SearchTurn(Arc<TurnState>);

impl SearchTurn {
    /// Returns whether this is the turn of the engine that owns `state`.
    pub(crate) fn belongs_to(&self, state: &Arc<TurnState>) -> bool {
        Arc::ptr_eq(&self.0, state)
    }
}

impl Drop for SearchTurn {
    fn drop(&mut self) {
        *self.0.busy.lock().unwrap() = false;
        // Every waiter: one woken alone may be stopped and leave the turn free.
        self.0.freed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;

    use super::*;

    #[test]
    fn a_waiter_gets_the_turn_only_once_it_is_dropped() {
        let state = Arc::new(TurnState::default());
        let turn = state.wait();

        let (acquired_tx, acquired) = mpsc::channel();
        let waiter = thread::spawn({
            let state = Arc::clone(&state);
            move || {
                let turn = state.wait();
                acquired_tx.send(()).unwrap();
                turn
            }
        });
        assert!(acquired.recv_timeout(Duration::from_millis(100)).is_err());

        drop(turn);
        assert!(acquired.recv_timeout(Duration::from_secs(60)).is_ok());
        drop(waiter.join().unwrap());
        assert!(!*state.busy.lock().unwrap());
    }

    #[test]
    fn a_stopped_waiter_leaves_without_the_turn() {
        let state = Arc::new(TurnState::default());
        let turn = state.wait();

        let stop = StopToken::new();
        let waiter = thread::spawn({
            let state = Arc::clone(&state);
            let stop = stop.clone();
            move || state.wait_unless_stopped(&stop).is_some()
        });
        stop.stop();
        assert!(!waiter.join().unwrap());
        assert!(*state.busy.lock().unwrap());

        // A stopped token gets no turn, not even a free one.
        drop(turn);
        assert!(state.wait_unless_stopped(&stop).is_none());
        assert!(state.wait_unless_stopped(&StopToken::new()).is_some());
    }
}
