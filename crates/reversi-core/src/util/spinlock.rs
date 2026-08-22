//! Spinlock implementation with TTAS pattern and adaptive backoff.

use std::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, Ordering},
};

/// Maximum spin iterations before yielding to the OS scheduler.
const SPIN_LIMIT: u32 = 100;

/// Maximum exponent for exponential backoff (2^6 = 64 spins).
const MAX_BACKOFF_EXP: u32 = 6;

/// A cache-line-aligned, data-less spinlock.
#[repr(align(64))]
pub(crate) struct SpinLock {
    state: AtomicBool,
}

/// An RAII guard that releases a [`SpinLock`] when dropped.
#[must_use = "the lock is released immediately if the guard is unused"]
pub(crate) struct SpinLockGuard<'a>(&'a SpinLock);

impl SpinLock {
    /// Creates an unlocked spinlock.
    pub(crate) const fn new() -> Self {
        Self {
            state: AtomicBool::new(false),
        }
    }

    /// Acquires the lock and returns its guard.
    #[inline]
    pub(crate) fn lock(&self) -> SpinLockGuard<'_> {
        if !self.try_acquire() {
            self.lock_slow();
        }
        SpinLockGuard(self)
    }

    #[inline]
    fn try_acquire(&self) -> bool {
        !self.state.load(Ordering::Relaxed)
            && self
                .state
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
    }

    #[cold]
    fn lock_slow(&self) {
        let mut spin_count: u32 = 0;
        let mut backoff_exp: u32 = 0;

        loop {
            while self.state.load(Ordering::Relaxed) {
                spin_loop();
                spin_count += 1;

                if spin_count >= SPIN_LIMIT {
                    std::thread::yield_now();
                    spin_count = 0;
                    backoff_exp = 0;
                }
            }

            if !self.state.load(Ordering::Relaxed)
                && self
                    .state
                    .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                    .is_ok()
            {
                return;
            }

            for _ in 0..(1u32 << backoff_exp) {
                spin_loop();
            }

            if backoff_exp < MAX_BACKOFF_EXP {
                backoff_exp += 1;
            }
        }
    }

    #[inline]
    fn unlock(&self) {
        self.state.store(false, Ordering::Release);
    }
}

impl Drop for SpinLockGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.0.unlock();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
        thread,
    };

    use super::SpinLock;

    #[test]
    fn concurrent_increments_do_not_lose_updates() {
        const THREADS: u64 = 8;
        const ITERS: u64 = 10_000;

        let counter = Arc::new((SpinLock::new(), AtomicU64::new(0)));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..ITERS {
                        let _guard = counter.0.lock();
                        let value = counter.1.load(Ordering::Relaxed);
                        counter.1.store(value + 1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.1.load(Ordering::Relaxed), THREADS * ITERS);
    }
}
