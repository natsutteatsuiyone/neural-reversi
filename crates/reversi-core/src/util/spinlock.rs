//! Spinlock implementation with TTAS pattern and adaptive backoff.

use std::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, Ordering},
};

use lock_api::{GuardSend, RawMutex};

/// Maximum spin iterations before yielding to the OS scheduler.
const SPIN_LIMIT: u32 = 100;

/// Maximum exponent for exponential backoff (2^6 = 64 spins).
const MAX_BACKOFF_EXP: u32 = 6;

/// Raw spinlock with cache-line alignment to prevent false sharing.
#[repr(align(64))]
pub struct RawSpinLock {
    state: AtomicBool,
}

// SAFETY: This implementation satisfies RawMutex's contract:
// - `lock()` spins until the CAS succeeds, guaranteeing mutual exclusion.
// - `try_lock()` returns true only when CAS atomically transitions false→true.
// - `unlock()` uses Release store, pairing with Acquire CAS in lock paths to
//   establish a happens-before relationship for all data protected by the lock.
unsafe impl RawMutex for RawSpinLock {
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = RawSpinLock {
        state: AtomicBool::new(false),
    };

    type GuardMarker = GuardSend;

    #[inline]
    fn lock(&self) {
        if !self.try_lock() {
            self.lock_slow();
        }
    }

    #[inline]
    fn try_lock(&self) -> bool {
        !self.state.load(Ordering::Relaxed)
            && self
                .state
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
    }

    #[inline]
    unsafe fn unlock(&self) {
        self.state.store(false, Ordering::Release);
    }

    #[inline]
    fn is_locked(&self) -> bool {
        self.state.load(Ordering::Relaxed)
    }
}

impl RawSpinLock {
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use lock_api::Mutex;

    use super::RawSpinLock;

    type SpinMutex<T> = Mutex<RawSpinLock, T>;

    #[test]
    fn try_lock_is_exclusive_until_the_guard_drops() {
        let mutex: SpinMutex<i32> = Mutex::new(0);

        let guard = mutex.try_lock().expect("first try_lock should succeed");
        assert!(mutex.is_locked());
        assert!(
            mutex.try_lock().is_none(),
            "a second try_lock must fail while the lock is held"
        );

        drop(guard);
        assert!(!mutex.is_locked());
        assert!(
            mutex.try_lock().is_some(),
            "try_lock should succeed again after release"
        );
    }

    #[test]
    fn concurrent_increments_do_not_lose_updates() {
        const THREADS: u64 = 8;
        const ITERS: u64 = 10_000;

        let counter: Arc<SpinMutex<u64>> = Arc::new(Mutex::new(0));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..ITERS {
                        *counter.lock() += 1;
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*counter.lock(), THREADS * ITERS);
    }
}
