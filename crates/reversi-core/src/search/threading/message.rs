use std::sync::Arc;
use std::sync::mpsc::Sender;

use crate::search::SearchTask;
use crate::search::search_result::SearchResult;

use super::thread::Thread;

/// Messages that can be sent to the main thread.
pub(super) enum Message {
    /// Starts a new search with the given task and returns results via the sender.
    StartThinking(SearchTask, Arc<Thread>, Sender<SearchResult>),

    /// Signals the thread to exit.
    Exit,
}
