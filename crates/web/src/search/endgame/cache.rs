use reversi_core::types::Score;

const SCORE_BITS: u32 = 8;
const BOUND_BITS: u32 = 1;
const VALID_BITS: u32 = 1;

const SCORE_MASK: u64 = (1 << SCORE_BITS) - 1;
const BOUND_MASK: u64 = (1 << BOUND_BITS) - 1;
const VALID_MASK: u64 = 1;

const BOUND_SHIFT: u32 = SCORE_BITS;
const VALID_SHIFT: u32 = SCORE_BITS + BOUND_BITS;
const META_BITS: u32 = SCORE_BITS + BOUND_BITS + VALID_BITS;
const KEY_MASK: u64 = !((1u64 << META_BITS) - 1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EndGameCacheBound {
    Lower = 0,
    Upper = 1,
}

impl EndGameCacheBound {
    #[inline(always)]
    fn determine(score: Score, beta: Score) -> Self {
        if score < beta {
            Self::Upper
        } else {
            Self::Lower
        }
    }

    #[inline(always)]
    fn from_bit(bit: u64) -> Self {
        if bit == 0 { Self::Lower } else { Self::Upper }
    }
}

/// Decoded entry from the Web endgame cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct EndGameCacheEntry {
    pub(super) score: Score,
    bound: EndGameCacheBound,
}

impl EndGameCacheEntry {
    #[inline(always)]
    pub(super) fn can_cut(self, beta: Score) -> bool {
        match self.bound {
            EndGameCacheBound::Lower => self.score >= beta,
            EndGameCacheBound::Upper => self.score < beta,
        }
    }
}

/// Packed hash table for Web endgame null-window search bounds.
pub(super) struct EndGameCache {
    table: Box<[u64]>,
    mask: usize,
}

impl EndGameCache {
    /// Creates a new cache with the given memory budget in bytes.
    pub(super) fn new(memory_bytes: usize) -> Self {
        let entries = memory_bytes / std::mem::size_of::<u64>();
        assert!(
            entries > 0,
            "EndGameCache: memory_bytes ({memory_bytes}) too small for one entry"
        );
        assert!(
            entries.is_power_of_two(),
            "EndGameCache: entry count ({entries}) must be a power of two"
        );
        Self {
            table: vec![0; entries].into_boxed_slice(),
            mask: entries - 1,
        }
    }

    /// Computes the table index for a hash key.
    #[inline(always)]
    pub(super) fn index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    /// Probes the cache for an entry with a matching hash tag.
    #[inline(always)]
    pub(super) fn probe(&self, cache_idx: usize, key: u64) -> Option<EndGameCacheEntry> {
        // SAFETY: all cache indices come from `self.index`, which masks them to
        // the table length. Tests also exercise forced collisions through index.
        let entry = unsafe { *self.table.get_unchecked(cache_idx) };
        if ((entry >> VALID_SHIFT) & VALID_MASK) == 0 || (entry ^ key) & KEY_MASK != 0 {
            return None;
        }

        Some(EndGameCacheEntry {
            score: (entry & SCORE_MASK) as u8 as i8 as Score,
            bound: EndGameCacheBound::from_bit((entry >> BOUND_SHIFT) & BOUND_MASK),
        })
    }

    /// Stores a bound entry for `key`.
    #[inline(always)]
    pub(super) fn store(&mut self, cache_idx: usize, key: u64, score: Score, beta: Score) {
        debug_assert!(
            i8::try_from(score).is_ok(),
            "EndGameCache score out of i8 range: {score}"
        );
        // SAFETY: all cache indices come from `self.index`, which masks them to
        // the table length.
        unsafe {
            *self.table.get_unchecked_mut(cache_idx) =
                Self::pack(key, score, EndGameCacheBound::determine(score, beta));
        }
    }

    /// Clears all entries.
    #[allow(dead_code)]
    pub(super) fn clear(&mut self) {
        self.table.fill(0);
    }

    #[inline(always)]
    fn pack(key: u64, score: Score, bound: EndGameCacheBound) -> u64 {
        (key & KEY_MASK)
            | (VALID_MASK << VALID_SHIFT)
            | ((bound as u64) << BOUND_SHIFT)
            | ((score as i8 as u8) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_probe_hit() {
        let mut cache = EndGameCache::new(1024);
        let key = 0x1234_5678_9abc_def0;
        let cache_idx = cache.index(key);

        cache.store(cache_idx, key, 12, 12);

        let entry = cache.probe(cache_idx, key).unwrap();
        assert_eq!(entry.score, 12);
        assert!(entry.can_cut(12));
    }

    #[test]
    fn upper_bound_cuts_below_beta_only() {
        let mut cache = EndGameCache::new(1024);
        let key = 0xabcd_1234_5678_9abc;
        let cache_idx = cache.index(key);

        cache.store(cache_idx, key, 8, 12);

        let entry = cache.probe(cache_idx, key).unwrap();
        assert!(entry.can_cut(12));
        assert!(!entry.can_cut(8));
    }

    #[test]
    fn different_key_misses_on_forced_index_collision() {
        let mut cache = EndGameCache::new(std::mem::size_of::<u64>());
        let key1 = 0x1111_2222_3333_4444;
        let key2 = 0x5555_6666_7777_8888;
        let cache_idx1 = cache.index(key1);
        let cache_idx2 = cache.index(key2);

        cache.store(cache_idx1, key1, 12, 12);

        assert_eq!(cache_idx1, cache_idx2);
        assert!(cache.probe(cache_idx2, key2).is_none());
    }

    #[test]
    fn clear_removes_entries() {
        let mut cache = EndGameCache::new(1024);
        let key = 0x9876_5432_10fe_dcba;
        let cache_idx = cache.index(key);

        cache.store(cache_idx, key, -8, -7);
        assert!(cache.probe(cache_idx, key).is_some());

        cache.clear();

        assert!(cache.probe(cache_idx, key).is_none());
    }
}
