use std::sync::atomic::AtomicBool;
use lock_api::RawMutex;
use lock_api::GuardSend;
use std::sync::atomic::Ordering;
use std::hint::spin_loop;

/// RawSpinLock is a simple spin lock based on an atomic flag.
///
/// This structure implements the `lock_api::RawMutex` trait
/// and can serve as a basis for higher-level lock wrappers (for example, `lock_api::Mutex`).
pub struct RawSpinLock {
    state: AtomicBool,
}

unsafe impl RawMutex for RawSpinLock {
    // The initial state is false (i.e., the lock is not acquired)
    #[allow(clippy::declare_interior_mutable_const)]
    const INIT: Self = RawSpinLock {
        state: AtomicBool::new(false),
    };

    // Indicates that the lock guard is Send.
    type GuardMarker = GuardSend;

    /// Acquires the lock.
    ///
    /// First, attempts to change the flag from false to true using compare_exchange.
    /// If it fails, busy-waits (spins) until the flag becomes false.
    #[inline]
    fn lock(&self) {
        // Attempt the Compare-And-Swap (CAS) initially.
        while self.state.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            // If the attempt fails, spin while the flag remains true.
            while self.state.load(Ordering::Relaxed) {
                spin_loop();
            }
        }
    }

    /// Releases the lock.
    ///
    /// Although defined as unsafe, the caller must ensure correct lifetime management
    /// through proper lock guard usage.
    #[inline]
    unsafe fn unlock(&self) {
        self.state.store(false, Ordering::Release);
    }

    /// Attempts to acquire the lock without blocking.
    ///
    /// Returns true if the lock was successfully acquired, or false if it was already held.
    #[inline]
    fn try_lock(&self) -> bool {
        self.state.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
    }

    /// Checks whether the lock is currently held.
    #[inline]
    fn is_locked(&self) -> bool {
        self.state.load(Ordering::Relaxed)
    }
}
