//! Thread-safe bitset implementation using atomic operations.

use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe 64-bit bitset using atomic operations.
#[derive(Default)]
pub struct AtomicBitSet {
    data: AtomicU64,
}

impl AtomicBitSet {
    /// Creates a new empty atomic bitset with all bits cleared.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            data: AtomicU64::new(0),
        }
    }

    /// Counts the bits set in one relaxed load.
    #[inline(always)]
    pub fn count(&self) -> u32 {
        self.data.load(Ordering::Relaxed).count_ones()
    }

    /// Sets the bit at `index`, which must be less than 64.
    #[inline(always)]
    pub fn set(&self, index: usize) {
        debug_assert!(index < 64);
        self.data.fetch_or(1 << index, Ordering::Relaxed);
    }

    /// Clears the bit at `index` with release ordering. The index must be less than 64.
    #[inline(always)]
    pub fn reset(&self, index: usize) {
        debug_assert!(index < 64);
        self.data.fetch_and(!(1 << index), Ordering::Release);
    }

    /// Tests the bit at `index` with a relaxed load. The index must be less than 64.
    #[inline(always)]
    pub fn test(&self, index: usize) -> bool {
        (self.data.load(Ordering::Relaxed) >> index) & 1 != 0
    }

    /// Checks whether an acquire load sees all bits clear.
    #[inline(always)]
    pub fn none(&self) -> bool {
        self.data.load(Ordering::Acquire) == 0
    }

    /// Clears all bits with a relaxed store, ending any release sequence on the bitset.
    #[inline(always)]
    pub fn clear(&self) {
        self.data.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_test_count_reset_and_clear() {
        let bits = AtomicBitSet::new();
        assert!(bits.none());
        assert_eq!(bits.count(), 0);
        assert!(!bits.test(0));
        assert!(!bits.test(5));

        bits.set(0);
        bits.set(5);
        bits.set(63);
        assert!(!bits.none());
        assert_eq!(bits.count(), 3);
        assert!(bits.test(0));
        assert!(bits.test(5));
        assert!(bits.test(63));
        assert!(!bits.test(1));

        bits.reset(5);
        assert_eq!(bits.count(), 2);
        assert!(!bits.test(5));

        bits.clear();
        assert!(bits.none());
        assert_eq!(bits.count(), 0);
        assert!(!bits.test(0));
        assert!(!bits.test(63));
    }

    #[test]
    fn empty_mask_observes_payloads_from_both_releases() {
        use std::time::{Duration, Instant};

        let bits = AtomicBitSet::new();
        let payloads = [AtomicU64::new(0), AtomicU64::new(0)];
        bits.set(0);
        bits.set(63);

        std::thread::scope(|scope| {
            for (bit, payload, expected) in [(0, &payloads[0], 31), (63, &payloads[1], 47)] {
                let bits = &bits;
                scope.spawn(move || {
                    payload.store(expected, Ordering::Relaxed);
                    bits.reset(bit);
                });
            }

            let deadline = Instant::now() + Duration::from_secs(10);
            while !bits.none() {
                assert!(Instant::now() < deadline, "helper bits stayed set");
                std::thread::yield_now();
            }
            assert_eq!(payloads[0].load(Ordering::Relaxed), 31);
            assert_eq!(payloads[1].load(Ordering::Relaxed), 47);
        });
    }
}
