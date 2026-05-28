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

    fn colliding_keys() -> (u64, u64) {
        // `index()` uses the high bits after `rotate_left(16)`, so these
        // share a bucket for a 16-entry cache while keeping distinct stored
        // 48-bit keys.
        (0x1234_0000_0000_0001, 0x1234_0000_0000_0002)
    }

    #[test]
    fn new_allocates_power_of_two_entries_and_masks_indices() {
        let one_entry = EvalCache::new(0);
        assert_eq!(one_entry.table.len(), 1);
        assert_eq!(one_entry.mask, 0);
        assert_eq!(one_entry.index(0), 0);
        assert_eq!(one_entry.index(u64::MAX), 0);

        let cache = EvalCache::new(4);
        assert_eq!(cache.table.len(), 16);
        assert_eq!(cache.mask, 15);
    }

    #[test]
    fn index_uses_rotated_high_key_bits() {
        let cache = EvalCache::new(4);

        for bucket in 0..16 {
            let key = (bucket as u64) << 48;
            assert_eq!(cache.index(key), bucket, "bucket {bucket}");
        }
    }

    #[test]
    fn pack_preserves_low_48_key_bits_and_signed_score_bits() {
        let key = 0xABCD_FEDC_BA98_7654;

        for raw_score in [
            -ScaledScore::INF.value(),
            -1,
            0,
            1,
            ScaledScore::INF.value(),
        ] {
            let packed = EvalCache::pack(key, raw_score);

            assert_eq!(packed >> SCORE_BITS, key & KEY_MASK, "score {raw_score}");
            assert_eq!(packed as i16 as i32, raw_score, "score {raw_score}");
        }
    }

    #[test]
    fn probe_returns_only_exact_stored_truncated_key() {
        let cache = EvalCache::new(4);
        let (stored_key, colliding_key) = colliding_keys();
        let score = ScaledScore::from_raw(-1234);

        assert_eq!(cache.index(stored_key), cache.index(colliding_key));
        cache.store(stored_key, score);

        assert_eq!(cache.probe(stored_key), Some(score));
        assert_eq!(cache.probe(colliding_key), None);
        assert_eq!(cache.probe(0x9876_0000_0000_0001), None);
    }

    #[test]
    fn store_overwrites_previous_value_for_same_key() {
        let cache = EvalCache::new(4);
        let key = 0x1234_5678_9ABC_DEF0;

        cache.store(key, ScaledScore::from_raw(42));
        cache.store(key, ScaledScore::from_raw(-84));

        assert_eq!(cache.probe(key), Some(ScaledScore::from_raw(-84)));
    }

    #[test]
    fn store_replaces_existing_entry_on_bucket_collision() {
        let cache = EvalCache::new(4);
        let (old_key, new_key) = colliding_keys();

        cache.store(old_key, ScaledScore::from_raw(11));
        cache.store(new_key, ScaledScore::from_raw(22));

        assert_eq!(cache.probe(old_key), None);
        assert_eq!(cache.probe(new_key), Some(ScaledScore::from_raw(22)));
    }

    #[test]
    fn score_round_trips_across_signed_16_bit_cache_encoding() {
        let cache = EvalCache::new(5);
        let cases = [
            (0x0001_0000_0000_0001, -ScaledScore::INF.value()),
            (0x0002_0000_0000_0002, -1),
            (0x0003_0000_0000_0003, 0),
            (0x0004_0000_0000_0004, 1),
            (0x0005_0000_0000_0005, ScaledScore::INF.value()),
        ];

        for (key, raw_score) in cases {
            let score = ScaledScore::from_raw(raw_score);
            cache.store(key, score);
            assert_eq!(cache.probe(key), Some(score), "key {key:#018x}");
        }
    }

    #[test]
    fn clear_removes_all_stored_entries_and_resets_backing_slots() {
        let cache = EvalCache::new(4);
        let keys = [
            0x0001_0000_0000_0001,
            0x0002_0000_0000_0002,
            0x0003_0000_0000_0003,
        ];

        for (idx, &key) in keys.iter().enumerate() {
            cache.store(key, ScaledScore::from_raw((idx as i32 + 1) * 100));
            assert!(cache.probe(key).is_some(), "precondition key {idx}");
        }

        cache.clear();

        for entry in cache.table.iter() {
            assert_eq!(entry.load(Ordering::Relaxed), 0);
        }
        for &key in &keys {
            assert_eq!(cache.probe(key), None, "key {key:#018x}");
        }
    }

    #[test]
    fn prefetch_targets_the_probe_bucket_without_changing_cache_contents() {
        let cache = EvalCache::new(4);
        let key = 0xF000_0000_0000_000F;
        let score = ScaledScore::from_raw(321);

        cache.store(key, score);
        let before = cache.table[cache.index(key)].load(Ordering::Relaxed);

        cache.prefetch(key);
        cache.prefetch(u64::MAX);

        assert_eq!(
            cache.table[cache.index(key)].load(Ordering::Relaxed),
            before
        );
        assert_eq!(cache.probe(key), Some(score));
    }
}
