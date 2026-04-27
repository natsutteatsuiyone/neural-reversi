//! Hash table for caching neural network evaluation results.

use std::hint::{Locality, prefetch_read};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::ScaledScore;

const KEY_MASK: u64 = 0xFFFFFFFFFFFF;
const SCORE_BITS: u32 = 16;

/// Hash table for caching neural network evaluation results.
///
/// Bit layout of each entry (`AtomicU64`):
/// - Bits 16-63 (48 bits): Truncated position hash key
/// - Bits 0-15 (16 bits): Evaluation score (2's complement signed integer)
pub struct EvalCache {
    table: Box<[AtomicU64]>,
    mask: u64,
}

impl EvalCache {
    /// Creates a new cache with `2^size_log2` entries.
    pub fn new(size_log2: u32) -> Self {
        let size = 1usize << size_log2;
        let mask = size as u64 - 1;

        let table = (0..size).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();

        EvalCache {
            table: table.into_boxed_slice(),
            mask,
        }
    }

    /// Stores an evaluation score in the cache.
    #[inline(always)]
    pub fn store(&self, key: u64, score: ScaledScore) {
        let index = self.index(key);
        let value = Self::pack(key, score.value());

        unsafe {
            self.table
                .get_unchecked(index)
                .store(value, Ordering::Relaxed);
        }
    }

    /// Returns the cached evaluation score for `key`, or [`None`] if not found.
    ///
    /// Empty slots store `0`. With a non-zero `key & KEY_MASK` (true for any
    /// real position hash) the empty entry naturally fails the key compare,
    /// so an explicit zero check is unnecessary.
    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<ScaledScore> {
        let index = self.index(key);
        let entry = unsafe { self.table.get_unchecked(index).load(Ordering::Relaxed) };

        let key_masked = key & KEY_MASK;
        let entry_key = entry >> SCORE_BITS;
        if entry_key != key_masked {
            return None;
        }

        let score = ScaledScore::from_raw(entry as i16 as i32);
        Some(score)
    }

    /// Prefetches the entry that `probe(key)` will read into L1.
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        let index = self.index(key);
        // SAFETY: `index()` returns a value in `0..self.table.len()`, so
        // `add(index)` stays within the same allocation.
        let addr = unsafe { self.table.as_ptr().add(index) };
        prefetch_read(addr, Locality::L1);
    }

    /// Calculates table index by rotating the key so high bits influence the bucket.
    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        (key.rotate_left(SCORE_BITS) & self.mask) as usize
    }

    /// Packs key and score into a single `u64`.
    #[inline(always)]
    fn pack(key: u64, score: i32) -> u64 {
        ((key & KEY_MASK) << SCORE_BITS) | (score as u16 as u64)
    }

    /// Clears all entries in the cache.
    pub fn clear(&self) {
        for entry in self.table.iter() {
            entry.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let cache = EvalCache::new(4);
        assert_eq!(cache.mask, 15);
        assert_eq!(cache.table.len(), 16);
    }

    #[test]
    fn test_store_and_probe() {
        let cache = EvalCache::new(4);
        let key = 0x123456789ABCDEF0;
        let score = ScaledScore::from_raw(42);

        cache.store(key, score);
        assert_eq!(cache.probe(key), Some(score));
    }

    #[test]
    fn test_probe_nonexistent() {
        let cache = EvalCache::new(4);
        assert_eq!(cache.probe(0x123456789ABCDEF0), None);
    }

    #[test]
    fn test_store_overwrite() {
        let cache = EvalCache::new(4);
        let key = 0x123456789ABCDEF0;

        cache.store(key, ScaledScore::from_raw(42));
        cache.store(key, ScaledScore::from_raw(84));
        assert_eq!(cache.probe(key), Some(ScaledScore::from_raw(84)));
    }

    #[test]
    fn test_different_keys() {
        let cache = EvalCache::new(10);
        let key1 = 0x123456789ABCDEF0;
        let key2 = 0xDEF0123456789ABC;

        cache.store(key1, ScaledScore::from_raw(42));
        cache.store(key2, ScaledScore::from_raw(84));

        assert_eq!(cache.probe(key1), Some(ScaledScore::from_raw(42)));
        assert_eq!(cache.probe(key2), Some(ScaledScore::from_raw(84)));
    }

    #[test]
    fn test_clear() {
        let cache = EvalCache::new(4);
        let key1 = 0x123456789ABCDEF0;
        let key2 = 0xDEF0123456789ABC;

        cache.store(key1, ScaledScore::from_raw(42));
        cache.store(key2, ScaledScore::from_raw(84));

        cache.clear();

        assert_eq!(cache.probe(key1), None);
        assert_eq!(cache.probe(key2), None);
    }
}
