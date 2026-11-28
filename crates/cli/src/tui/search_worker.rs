//! Background AI moves and hints over one search engine.
//!
//! A cancelled hint keeps the worker busy until its search has ended. Its
//! result is discarded, but the engine remains available for the next search.

use std::fmt;
use std::sync::mpsc::{self, Receiver};
use std::time::Instant;

use reversi_core::board::Board;
use reversi_core::engine::{self, Engine, Finished, GameId};
use reversi_core::level::Level;
use reversi_core::search::SearchRunOptions;
use reversi_core::search::result::SearchResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SearchKind {
    Move,
    Hint,
}

#[derive(Debug)]
struct PendingSearch {
    receiver: Receiver<Option<Finished>>,
    kind: Option<SearchKind>,
}

pub(super) struct SearchWorker {
    engine: Engine,
    game: GameId,
    pending: Option<PendingSearch>,
}

impl fmt::Debug for SearchWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchWorker")
            .field("available", &self.is_available())
            .field("pending", &self.pending)
            .finish_non_exhaustive()
    }
}

impl SearchWorker {
    pub(super) fn new(engine: Engine) -> Self {
        let game = engine.open();
        Self {
            engine,
            game,
            pending: None,
        }
    }

    #[must_use]
    pub(super) fn is_available(&self) -> bool {
        self.pending.is_none()
    }

    pub(super) fn activity(&self) -> Option<SearchKind> {
        self.pending.as_ref().and_then(|pending| pending.kind)
    }

    /// Starts a search unless [`Self::poll`] has yet to collect an earlier one.
    pub(super) fn start(&mut self, board: Board, level: Level, kind: SearchKind) -> bool {
        if self.pending.is_some() {
            return false;
        }
        let (engine_kind, multi_pv) = match kind {
            SearchKind::Move => (engine::SearchKind::Move, false),
            SearchKind::Hint => (engine::SearchKind::Analysis, true),
        };
        let options = SearchRunOptions::with_level(level).multi_pv(multi_pv);
        let (sender, receiver) = mpsc::channel();
        let started = self.engine.start(
            self.game,
            engine_kind,
            board,
            Instant::now(),
            move |_| options,
            move |finished| {
                let _ = sender.send(finished);
            },
        );
        if started.is_err() {
            return false;
        }
        self.pending = Some(PendingSearch {
            receiver,
            kind: Some(kind),
        });
        true
    }

    /// Stops a hint and suppresses its result, even if it is already queued.
    pub(super) fn cancel_hint(&mut self) {
        if let Some(pending) = &mut self.pending
            && pending.kind == Some(SearchKind::Hint)
        {
            self.engine.stop(self.game, engine::SearchKind::Analysis);
            pending.kind = None;
        }
    }

    /// Collects a finished search and returns its result only if it is still
    /// wanted.
    pub(super) fn poll(&mut self) -> Option<(SearchKind, SearchResult)> {
        let pending = self.pending.as_ref()?;
        let finished = pending.receiver.try_recv().ok()?;
        let kind = pending.kind;
        self.pending = None;
        kind.zip(finished)
            .map(|(kind, finished)| (kind, finished.result))
    }

    /// Resets the engine for a new game unless [`Self::poll`] has yet to
    /// collect a search.
    pub(super) fn reset(&mut self) -> bool {
        if self.pending.is_some() {
            return false;
        }
        self.engine.reset(self.game);
        true
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::thread;
    use std::time::Duration;

    use reversi_core::square::Square;

    use super::*;

    fn worker() -> SearchWorker {
        SearchWorker::new(crate::config::test_engine(0, 1))
    }

    fn wait_until_available(worker: &mut SearchWorker) -> Option<(SearchKind, SearchResult)> {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let result = worker.poll();
            if worker.is_available() {
                return result;
            }
            assert!(
                Instant::now() < deadline,
                "search worker did not finish its search"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn completed_hint_returns_all_moves_and_reuses_the_engine_for_an_ai_move() {
        let mut worker = worker();
        let board = Board::new().make_move(Square::D3);
        let level = Level::uniform(3, 0);

        assert!(worker.start(board, level, SearchKind::Hint));
        let (kind, result) = wait_until_available(&mut worker).unwrap();
        assert_eq!(kind, SearchKind::Hint);
        let hints: HashSet<_> = result.pv_moves().iter().map(|pv| pv.sq).collect();
        let legal: HashSet<_> = board.legal_moves().iter().collect();
        assert_eq!(hints, legal);
        assert!(worker.poll().is_none());

        assert!(worker.reset());
        assert!(worker.start(board, level, SearchKind::Move));
        let (kind, result) = wait_until_available(&mut worker).unwrap();
        assert_eq!(kind, SearchKind::Move);
        assert!(board.is_legal_move(result.best_move().unwrap()));
        assert_eq!(result.depth(), level.mid_depth);
    }

    #[test]
    fn cancelled_hint_stays_busy_until_reclaimed_and_does_not_stop_the_next_move() {
        let mut worker = worker();
        let board = Board::new().make_move(Square::D3);
        let level = Level::uniform(3, 0);

        assert!(worker.start(board, Level::uniform(12, 0), SearchKind::Hint));
        worker.cancel_hint();
        assert_eq!(worker.activity(), None);
        assert!(!worker.is_available());
        assert!(!worker.start(board, level, SearchKind::Move));
        assert!(!worker.reset());
        assert!(wait_until_available(&mut worker).is_none());
        assert!(worker.poll().is_none());

        assert!(worker.start(board, level, SearchKind::Move));
        let (kind, result) = wait_until_available(&mut worker).unwrap();
        assert_eq!(kind, SearchKind::Move);
        assert!(board.is_legal_move(result.best_move().unwrap()));
        assert_eq!(result.depth(), level.mid_depth);
    }

    #[test]
    fn cancelling_a_queued_hint_discards_its_result_and_reclaims_the_engine() {
        let mut worker = worker();
        let board = Board::new().make_move(Square::D3);
        assert!(worker.start(board, Level::uniform(3, 0), SearchKind::Hint));

        // Queue a completed search without depending on worker timing.
        let pending = worker.pending.as_mut().unwrap();
        let completed = pending
            .receiver
            .recv_timeout(Duration::from_secs(60))
            .unwrap();
        let (sender, receiver) = mpsc::channel();
        pending.receiver = receiver;
        assert!(sender.send(completed).is_ok());

        worker.cancel_hint();
        assert!(!worker.is_available());
        assert!(worker.poll().is_none());
        assert!(worker.is_available());
    }
}
