//! Synchronization primitives of the YBWC split-point protocol.
//!
//! Normal builds use `std`. Builds with `--cfg loom` swap in the instrumented
//! primitives of `loom`, so the loom tests in `search::threading` can explore
//! the protocol's interleavings and memory orderings on the real code.

#[cfg(not(loom))]
pub(crate) use std::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU64, AtomicUsize},
    sync::{Condvar, Mutex},
};

#[cfg(loom)]
pub(crate) use loom::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU64, AtomicUsize},
    sync::{Condvar, Mutex},
};

/// Thread spawning and scheduling hints.
pub(crate) mod thread {
    #[cfg(not(loom))]
    pub(crate) use std::thread::{JoinHandle, current, sleep, spawn, yield_now};

    #[cfg(loom)]
    pub(crate) use loom::thread::{JoinHandle, current, yield_now};

    /// Spawns a loom thread whose stack fits a `SearchContext`; loom's
    /// default stack is 32 KiB.
    #[cfg(loom)]
    pub(crate) fn spawn<F, T>(f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        loom::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(f)
            .expect("loom thread spawn")
    }

    /// Under loom a timed sleep is a yield: loom has no clock.
    #[cfg(loom)]
    pub(crate) fn sleep(_: std::time::Duration) {
        yield_now();
    }
}

/// [`std::cell::SyncUnsafeCell`] whose accesses loom can track.
///
/// Access goes through closures so that, under loom, every read and write is
/// checked for a happens-before relation with the conflicting accesses of
/// other threads.
pub(crate) struct SyncUnsafeCell<T>(
    #[cfg(not(loom))] std::cell::SyncUnsafeCell<T>,
    #[cfg(loom)] loom::cell::UnsafeCell<T>,
);

// SAFETY: same contract as `std::cell::SyncUnsafeCell`: callers serialize
// access to the value.
#[cfg(loom)]
unsafe impl<T: Sync> Sync for SyncUnsafeCell<T> {}

impl<T> SyncUnsafeCell<T> {
    /// Creates a cell holding `value`.
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        #[cfg(not(loom))]
        return Self(std::cell::SyncUnsafeCell::new(value));
        #[cfg(loom)]
        return Self(loom::cell::UnsafeCell::new(value));
    }

    /// Calls `f` with a pointer for reading the value.
    #[inline(always)]
    pub(crate) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        #[cfg(not(loom))]
        return f(self.0.get());
        #[cfg(loom)]
        return self.0.with(f);
    }

    /// Calls `f` with a pointer for writing the value.
    #[inline(always)]
    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        #[cfg(not(loom))]
        return f(self.0.get());
        #[cfg(loom)]
        return self.0.with_mut(f);
    }
}
