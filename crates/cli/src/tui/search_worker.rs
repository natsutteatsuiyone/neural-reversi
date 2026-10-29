//! Background AI moves and hints over one search engine.
//!
//! A cancelled hint keeps the engine busy until its worker returns it. Its
//! result is discarded, but the engine remains available for the next search.

use std::fmt;
use std::sync::mpsc::{self, Receiver};
use std::thread;

use reversi_core::board::Board;
use reversi_core::level::Level;
use reversi_core::search::result::SearchResult;
use reversi_core::search::{Search, SearchRunOptions, StopToken};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SearchKind {
    Move,
    Hint,
}

#[derive(Debug)]
struct PendingSearch {
    receiver: Receiver<(Search, SearchResult)>,
    kind: Option<SearchKind>,
    stop: Option<StopToken>,
}

pub(super) struct SearchWorker {
    search: Option<Search>,
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
    pub(super) fn new(search: Search) -> Self {
        Self {
            search: Some(search),
            pending: None,
        }
    }

    #[must_use]
    pub(super) fn is_available(&self) -> bool {
        self.search.is_some()
    }

    pub(super) fn activity(&self) -> Option<SearchKind> {
        self.pending.as_ref().and_then(|pending| pending.kind)
    }

    /// Starts a search unless a worker still owns the engine.
    pub(super) fn start(&mut self, board: Board, level: Level, kind: SearchKind) -> bool {
        let Some(mut search) = self.search.take() else {
            return false;
        };
        let mut options = SearchRunOptions::with_level(level);
        let stop = if kind == SearchKind::Hint {
            let stop = StopToken::new();
            options = options.multi_pv(true).stop_token(stop.clone());
            Some(stop)
        } else {
            None
        };
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let result = search.run(&board, &options);
            let _ = sender.send((search, result));
        });
        self.pending = Some(PendingSearch {
            receiver,
            kind: Some(kind),
            stop,
        });
        true
    }

    /// Stops a hint and suppresses its result, even if it is already queued.
    pub(super) fn cancel_hint(&mut self) {
        if let Some(pending) = &mut self.pending
            && pending.kind == Some(SearchKind::Hint)
        {
            if let Some(stop) = pending.stop.take() {
                stop.stop();
            }
            pending.kind = None;
        }
    }

    /// Reclaims the engine and returns only a result that is still wanted.
    pub(super) fn poll(&mut self) -> Option<(SearchKind, SearchResult)> {
        let pending = self.pending.as_ref()?;
        let (search, result) = pending.receiver.try_recv().ok()?;
        let kind = pending.kind;
        self.search = Some(search);
        self.pending = None;
        kind.map(|kind| (kind, result))
    }

    /// Resets the engine for a new game unless a worker still owns it.
    pub(super) fn reset(&mut self) -> bool {
        let Some(search) = &mut self.search else {
            return false;
        };
        search.init();
        true
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::time::{Duration, Instant};

    use reversi_core::search::options::SearchOptions;
    use reversi_core::square::Square;

    use super::*;

    fn worker() -> SearchWorker {
        SearchWorker::new(Search::new(&SearchOptions::new(0).with_threads(Some(1))))
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
                "search worker did not return the engine"
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
        let legal: HashSet<_> = board.get_moves().iter().collect();
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
