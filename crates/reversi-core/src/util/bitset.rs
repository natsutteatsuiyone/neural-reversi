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

    /// Returns the number of bits currently set.
    #[inline(always)]
    pub fn count(&self) -> u32 {
        self.data.load(Ordering::Relaxed).count_ones()
    }

    /// Sets the bit at the specified index to 1.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index (0-based).
    #[inline(always)]
    pub fn set(&self, index: usize) {
        debug_assert!(index < 64);
        self.data.fetch_or(1 << index, Ordering::Relaxed);
    }

    /// Clears the bit at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index (0-based).
    #[inline(always)]
    pub fn reset(&self, index: usize) {
        debug_assert!(index < 64);
        self.data.fetch_and(!(1 << index), Ordering::Relaxed);
    }

    /// Tests whether the bit at the specified index is set.
    ///
    /// # Arguments
    ///
    /// * `index` - Bit index (0-based).
    ///
    /// # Returns
    ///
    /// `true` if the bit is set.
    #[inline(always)]
    pub fn test(&self, index: usize) -> bool {
        (self.data.load(Ordering::Relaxed) >> index) & 1 != 0
    }

    /// Checks if all bits are clear.
    ///
    /// # Returns
    ///
    /// `true` if no bits are set.
    #[inline(always)]
    pub fn none(&self) -> bool {
        self.data.load(Ordering::Relaxed) == 0
    }

    /// Clears all bits.
    #[inline(always)]
    pub fn clear(&self) {
        self.data.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let bitset = AtomicBitSet::new();
        assert_eq!(bitset.count(), 0);
        assert!(bitset.none());
    }

    #[test]
    fn test_set_and_test() {
        let bitset = AtomicBitSet::new();

        bitset.set(0);
        assert!(bitset.test(0));
        assert_eq!(bitset.count(), 1);

        bitset.set(5);
        assert!(bitset.test(5));
        assert_eq!(bitset.count(), 2);

        bitset.set(63);
        assert!(bitset.test(63));
        assert_eq!(bitset.count(), 3);
    }

    #[test]
    fn test_reset() {
        let bitset = AtomicBitSet::new();

        bitset.set(10);
        assert!(bitset.test(10));

        bitset.reset(10);
        assert!(!bitset.test(10));
        assert_eq!(bitset.count(), 0);
    }

    #[test]
    fn test_count() {
        let bitset = AtomicBitSet::new();
        assert_eq!(bitset.count(), 0);

        for i in 0..10 {
            bitset.set(i);
        }
        assert_eq!(bitset.count(), 10);
    }

    #[test]
    fn test_none() {
        let bitset = AtomicBitSet::new();
        assert!(bitset.none());

        bitset.set(0);
        assert!(!bitset.none());

        bitset.reset(0);
        assert!(bitset.none());
    }

    #[test]
    fn test_clear() {
        let bitset = AtomicBitSet::new();

        bitset.set(1);
        bitset.set(5);
        bitset.set(20);
        assert_eq!(bitset.count(), 3);

        bitset.clear();
        assert_eq!(bitset.count(), 0);
        assert!(bitset.none());
    }

    #[test]
    fn test_multiple_operations() {
        let bitset = AtomicBitSet::new();

        bitset.set(0);
        bitset.set(1);
        bitset.set(31);
        bitset.set(32);
        bitset.set(63);

        assert_eq!(bitset.count(), 5);
        assert!(bitset.test(0));
        assert!(bitset.test(1));
        assert!(bitset.test(31));
        assert!(bitset.test(32));
        assert!(bitset.test(63));

        bitset.reset(1);
        bitset.reset(32);

        assert_eq!(bitset.count(), 3);
        assert!(!bitset.test(1));
        assert!(!bitset.test(32));
        assert!(bitset.test(0));
        assert!(bitset.test(31));
        assert!(bitset.test(63));
    }
}
