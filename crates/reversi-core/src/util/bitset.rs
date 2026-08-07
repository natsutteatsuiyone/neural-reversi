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

    /// Sets the bit at the specified index.
    #[inline(always)]
    pub fn set(&self, index: usize) {
        debug_assert!(index < 64);
        self.data.fetch_or(1 << index, Ordering::Relaxed);
    }

    /// Clears the bit at the specified index.
    #[inline(always)]
    pub fn reset(&self, index: usize) {
        debug_assert!(index < 64);
        self.data.fetch_and(!(1 << index), Ordering::Release);
    }

    /// Tests whether the bit at the specified index is set.
    #[inline(always)]
    pub fn test(&self, index: usize) -> bool {
        (self.data.load(Ordering::Relaxed) >> index) & 1 != 0
    }

    /// Checks whether all bits are clear.
    #[inline(always)]
    pub fn none(&self) -> bool {
        self.data.load(Ordering::Acquire) == 0
    }

    /// Clears all bits.
    #[inline(always)]
    pub fn clear(&self) {
        self.data.store(0, Ordering::Relaxed);
    }
}
