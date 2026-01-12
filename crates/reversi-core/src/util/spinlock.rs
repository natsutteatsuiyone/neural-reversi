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
