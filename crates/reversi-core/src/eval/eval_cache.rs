//! Hash table for caching neural network evaluation results.

use std::hint::{Locality, prefetch_read};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::ScaledScore;

const KEY_MASK: u64 = 0xFFFFFFFFFFFF;
const SCORE_BITS: u32 = 16;
const EMPTY_SCORE_BITS: u16 = i16::MIN as u16;
const EMPTY_ENTRY: u64 = EMPTY_SCORE_BITS as u64;
const WAYS: usize = 4;
const WAY_BITS: u32 = WAYS.trailing_zeros();

/// One cache set. `align(32)` keeps every bucket inside a single 64-byte line,
/// so a probe of all `WAYS` entries touches exactly one line.
#[repr(C, align(32))]
struct Bucket {
    entries: [AtomicU64; WAYS],
}

/// A four-way set-associative cache for neural network evaluation results.
///
/// Bit layout of each entry (`AtomicU64`):
/// - Bits 16-63 (48 bits): Truncated position hash key
/// - Bits 0-15 (16 bits): Evaluation score as a signed 16-bit integer
///
/// The otherwise invalid score `i16::MIN` marks an empty entry.
///
/// The bucket index takes the low bits of the key rotated by 16, so a lookup
/// depends on all 64 key bits as long as the index keeps at least the 16 bits
/// that the stored 48-bit tag drops.
pub struct EvalCache {
    table: Box<[Bucket]>,
    mask: u64,
}

impl EvalCache {
    /// Creates a new cache with `2^size_log2` entries, rounded up to one bucket
    /// when `size_log2` is smaller than a bucket.
    pub fn new(size_log2: u32) -> Self {
        assert!(size_log2 < usize::BITS, "EvalCache size does not fit usize");
        let bucket_count = 1usize << size_log2.saturating_sub(WAY_BITS);
        let mask = bucket_count as u64 - 1;

        let table = (0..bucket_count)
            .map(|_| Bucket {
                entries: [const { AtomicU64::new(EMPTY_ENTRY) }; WAYS],
            })
            .collect::<Vec<_>>();

        EvalCache {
            table: table.into_boxed_slice(),
            mask,
        }
    }

    /// Returns the cached evaluation score for `key`, or [`None`] if not found.
    #[cfg(test)]
    fn probe(&self, key: u64) -> Option<ScaledScore> {
        self.probe_for_store(key).0
    }

    /// Returns a cached score or evaluates and stores it on a miss.
    #[inline(always)]
    pub fn get_or_insert_with(
        &self,
        key: u64,
        evaluate: impl FnOnce() -> ScaledScore,
    ) -> ScaledScore {
        let (cached, slot) = self.probe_for_store(key);
        if let Some(score) = cached {
            return score;
        }

        let score = evaluate();
        slot.store(Self::pack(key, score.value()), Ordering::Relaxed);
        score
    }

    /// Probes every way and returns the replacement slot on a miss.
    #[inline(always)]
    fn probe_for_store(&self, key: u64) -> (Option<ScaledScore>, &AtomicU64) {
        let (index, victim_way) = self.location(key);
        // SAFETY: `location()` masks `index` to `0..self.table.len()`.
        let bucket = unsafe { self.table.get_unchecked(index) };
        let key_masked = key & KEY_MASK;

        // All ways share one cache line, so loading them unconditionally is
        // cheaper than branching per way on a compare that rarely predicts.
        let entries: [u64; WAYS] = std::array::from_fn(|way| {
            // SAFETY: `way` is bounded by the array length.
            unsafe { bucket.entries.get_unchecked(way) }.load(Ordering::Relaxed)
        });

        let mut hit_ways = 0u32;
        let mut empty_ways = 0u32;
        for (way, &entry) in entries.iter().enumerate() {
            let matches = (entry >> SCORE_BITS == key_masked) & (entry as u16 != EMPTY_SCORE_BITS);
            hit_ways |= (matches as u32) << way;
            empty_ways |= ((entry == EMPTY_ENTRY) as u32) << way;
        }

        if hit_ways != 0 {
            // A key occupies at most one way, so the lowest match is the match.
            let way = hit_ways.trailing_zeros() as usize;
            // SAFETY: `hit_ways` only carries bits below `WAYS`, so `way` is a
            // valid index into both arrays.
            let (entry, slot) = unsafe {
                (
                    *entries.get_unchecked(way),
                    bucket.entries.get_unchecked(way),
                )
            };
            return (Some(Self::unpack_score(entry)), slot);
        }

        let way = if empty_ways != 0 {
            empty_ways.trailing_zeros() as usize
        } else {
            victim_way
        };
        // SAFETY: `empty_ways` only carries bits below `WAYS`, and `location()`
        // bounds `victim_way` to `0..WAYS`.
        (None, unsafe { bucket.entries.get_unchecked(way) })
    }

    /// Prefetches the bucket that `probe(key)` will read into L1.
    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        let (index, _) = self.location(key);
        // SAFETY: `location()` returns an index in `0..self.table.len()`, so
        // `add(index)` stays within the same allocation.
        let addr = unsafe { self.table.as_ptr().add(index) };
        prefetch_read(addr, Locality::L1);
    }

    /// Calculates the bucket and the way to evict from the position key.
    #[inline(always)]
    fn location(&self, key: u64) -> (usize, usize) {
        let rotated = key.rotate_left(SCORE_BITS);
        let index = (rotated & self.mask) as usize;
        // The index consumes the low end of `rotated`, so the top end supplies
        // an eviction way that does not correlate with it.
        let victim_way = (rotated >> (u64::BITS - WAY_BITS)) as usize;
        (index, victim_way)
    }

    /// Packs key and score into a single `u64`.
    #[inline(always)]
    fn pack(key: u64, score: i32) -> u64 {
        debug_assert!(
            (-ScaledScore::INF.value()..=ScaledScore::INF.value()).contains(&score),
            "cache score must be within the ScaledScore sentinel range"
        );
        ((key & KEY_MASK) << SCORE_BITS) | score as u16 as u64
    }

    /// Decodes a score from a non-empty packed entry.
    #[inline(always)]
    fn unpack_score(entry: u64) -> ScaledScore {
        ScaledScore::from_raw(entry as i16 as i32)
    }

    /// Clears all entries in the cache.
    pub fn clear(&self) {
        for bucket in self.table.iter() {
            for entry in &bucket.entries {
                entry.store(EMPTY_ENTRY, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colliding_keys() -> (u64, u64) {
        // `location()` uses the high bits after `rotate_left(16)`, so these
        // share a bucket for a 16-entry cache while keeping distinct stored
        // 48-bit keys.
        (0x1234_0000_0000_0001, 0x1234_0000_0000_0002)
    }

    #[test]
    fn new_allocates_power_of_two_entries_and_masks_indices() {
        let minimum = EvalCache::new(0);
        assert_eq!(minimum.table.len() * WAYS, WAYS);
        assert_eq!(minimum.mask, 0);

        let cache = EvalCache::new(4);
        assert_eq!(cache.table.len(), 4);
        assert_eq!(cache.table.len() * WAYS, 16);
        assert_eq!(cache.mask, 3);
        assert_eq!(
            cache.table.as_ptr().addr() % size_of::<Bucket>(),
            0,
            "a bucket must not straddle a cache line"
        );
    }

    #[test]
    #[should_panic(expected = "does not fit usize")]
    fn new_rejects_unrepresentable_size() {
        let _ = EvalCache::new(usize::BITS);
    }

    #[test]
    fn location_uses_rotated_high_key_bits() {
        let cache = EvalCache::new(4);

        for original_index in 0..16u64 {
            let key = original_index << 48;
            let (bucket, _) = cache.location(key);
            assert_eq!(bucket as u64, original_index & 3, "index {original_index}");
        }

        // The eviction way comes from the far end of the rotated key, so it
        // stays independent of the bucket index.
        for way in 0..WAYS as u64 {
            let key = way << (u64::BITS - WAY_BITS - SCORE_BITS);
            let (bucket, victim_way) = cache.location(key);
            assert_eq!(bucket, 0, "way {way}");
            assert_eq!(victim_way as u64, way, "way {way}");
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
            assert_ne!(packed as u16, EMPTY_SCORE_BITS, "score {raw_score}");
            assert_eq!(
                EvalCache::unpack_score(packed).value(),
                raw_score,
                "score {raw_score}"
            );
        }
    }

    #[test]
    fn empty_slots_do_not_match_zero_fingerprint_keys() {
        let cache = EvalCache::new(17);

        for key in [0, 0x1234_0000_0000_0000] {
            assert_eq!(cache.probe(key), None, "key {key:#018x}");
        }

        cache.get_or_insert_with(0, || ScaledScore::ZERO);
        assert_eq!(cache.probe(0), Some(ScaledScore::ZERO));

        cache.clear();
        assert_eq!(cache.probe(0), None);
    }

    #[test]
    fn production_sizes_distinguish_every_key_bit() {
        // A lookup depends on the whole key only while the bucket index carries
        // the key bits that the stored tag drops.
        const MIN_SIZE_LOG2: u32 = WAY_BITS + (u64::BITS - KEY_MASK.count_ones());
        const { assert!(crate::eval::EVAL_CACHE_SIZE_LOG2 >= MIN_SIZE_LOG2) };

        let key = 0x1234_5678_9ABC_DEF0;

        for size_log2 in [MIN_SIZE_LOG2, MIN_SIZE_LOG2 + 1] {
            let cache = EvalCache::new(size_log2);
            let (index, _) = cache.location(key);
            let fingerprint = key & KEY_MASK;

            for bit in 0..64 {
                let other = key ^ (1_u64 << bit);
                let (other_index, _) = cache.location(other);
                let other_fingerprint = other & KEY_MASK;
                assert!(
                    index != other_index || fingerprint != other_fingerprint,
                    "size {size_log2}, bit {bit}"
                );
            }
        }
    }

    #[test]
    fn probe_returns_only_exact_stored_truncated_key() {
        let cache = EvalCache::new(4);
        let (stored_key, colliding_key) = colliding_keys();
        let score = ScaledScore::from_raw(-1234);

        assert_eq!(
            cache.location(stored_key).0,
            cache.location(colliding_key).0
        );
        cache.get_or_insert_with(stored_key, || score);

        assert_eq!(cache.probe(stored_key), Some(score));
        assert_eq!(cache.probe(colliding_key), None);
        assert_eq!(cache.probe(0x9876_0000_0000_0001), None);
    }

    #[test]
    fn insert_retains_two_entries_on_bucket_collision() {
        let cache = EvalCache::new(4);
        let (old_key, new_key) = colliding_keys();

        cache.get_or_insert_with(old_key, || ScaledScore::from_raw(11));
        cache.get_or_insert_with(new_key, || ScaledScore::from_raw(22));

        assert_eq!(cache.probe(old_key), Some(ScaledScore::from_raw(11)));
        assert_eq!(cache.probe(new_key), Some(ScaledScore::from_raw(22)));
    }

    #[test]
    fn get_or_insert_with_evaluates_only_on_a_miss() {
        let cache = EvalCache::new(4);
        let calls = std::cell::Cell::new(0);
        let key = 0x1234_5678_9ABC_DEF0;

        let first = cache.get_or_insert_with(key, || {
            calls.set(calls.get() + 1);
            ScaledScore::from_raw(123)
        });
        let second = cache.get_or_insert_with(key, || {
            calls.set(calls.get() + 1);
            ScaledScore::from_raw(456)
        });

        assert_eq!(first, ScaledScore::from_raw(123));
        assert_eq!(second, first);
        assert_eq!(calls.get(), 1);
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
            cache.get_or_insert_with(key, || score);
            assert_eq!(cache.probe(key), Some(score), "key {key:#018x}");
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "cache score must be within the ScaledScore sentinel range")]
    fn insert_rejects_scores_outside_the_scaled_score_domain() {
        let cache = EvalCache::new(4);
        cache.get_or_insert_with(1, || -ScaledScore::INF - 1);
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
            cache.get_or_insert_with(key, || ScaledScore::from_raw((idx as i32 + 1) * 100));
            assert!(cache.probe(key).is_some(), "precondition key {idx}");
        }

        cache.clear();

        for bucket in cache.table.iter() {
            for entry in &bucket.entries {
                assert_eq!(entry.load(Ordering::Relaxed), EMPTY_ENTRY);
            }
        }
        for &key in &keys {
            assert_eq!(cache.probe(key), None, "key {key:#018x}");
        }
    }
}
