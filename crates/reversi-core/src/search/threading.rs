//! Parallel search infrastructure using YBWC (Young Brothers Wait Concept).
//!
//! Reference: <https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/thread.cpp>

mod message;
mod pool;
mod split_point;
mod thread;

pub use pool::ThreadPool;
pub use split_point::SplitPoint;
pub use thread::SplitRequest;
pub use thread::Thread;
