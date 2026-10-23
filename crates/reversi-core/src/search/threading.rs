//! Search threads.
//!
//! Thread 0 is the main thread and runs the root search. In the midgame the
//! other threads run Lazy SMP helper searches on the same root: they share
//! results through the transposition table and mark the positions they are
//! inside with breadcrumbs. In the endgame they join the split points of YBWC
//! (Young Brothers Wait Concept).
//!
//! Reference: <https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/thread.cpp>

mod abort;
mod breadcrumbs;
mod pool;
mod split_point;
mod thread;

pub(crate) use abort::AbortState;
pub(in crate::search) use breadcrumbs::Breadcrumb;
pub use pool::ThreadPool;
pub(in crate::search) use split_point::SplitPoint;
pub(in crate::search) use thread::{HelperJob, SplitRequest, Thread};
