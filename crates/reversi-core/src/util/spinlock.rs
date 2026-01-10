use std::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, Ordering},
};

use lock_api::{GuardSend, RawMutex};

/// RawSpinLock
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
        if self.try_lock() {
            return;
        }

        self.lock_slow();
    }

    #[inline]
    fn try_lock(&self) -> bool {
        self.state
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
    #[inline]
    fn try_acquire_weak(&self) -> bool {
        self.state
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    #[cold]
    fn lock_slow(&self) {
        let mut yield_counter: u32 = 0;
        let mut backoff_exp: u32 = 0;

        loop {
            while self.state.load(Ordering::Relaxed) {
                spin_loop();

                yield_counter = yield_counter.wrapping_add(1);
                if yield_counter >= 1_000 {
                    std::thread::yield_now();
                    yield_counter = 0;
                    backoff_exp = 0;
                }
            }

            if self.try_acquire_weak() {
                return;
            }

            let limit = 1u32 << backoff_exp.min(6);
            for _ in 0..limit {
                spin_loop();
            }

            if backoff_exp < 10 {
                backoff_exp += 1;
            }
        }
    }
}
