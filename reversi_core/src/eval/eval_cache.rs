use std::sync::atomic::{AtomicU64, Ordering};

use crate::types::Score;

const KEY_MASK: u64 = 0xFFFFFFFFFFFF;
const SCORE_MASK: u64 = 0xFFFF;
const SCORE_BITS: u32 = 16;

/// Hash table for caching neural network evaluation results
/// Bit layout of each entry (AtomicU64):
/// - key:        48 bits (bit position 16-63)
/// - score:      16 bits (bit position 0-15) 2's complement signed integer
pub struct EvalCache {
    table: Box<[AtomicU64]>,
    mask: u64,
}

impl EvalCache {
    /// Initialize cache with specified size (must be power of 2)
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the cache (log2 of the number of entries)
    ///
    /// # Returns
    ///
    /// A new EvalCache instance
    pub fn new(size: u32) -> Self {
        let size = 1 << size;
        let mask = size as u64 - 1;

        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(AtomicU64::new(0));
        }

        EvalCache {
            table: table.into_boxed_slice(),
            mask,
        }
    }

    /// Store an entry in the cache
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key
    /// * `score` - The evaluation score
    pub fn store(&self, key: u64, score: i32) {
        let index = self.index(key);
        let value = Self::pack(key, score);
        self.table[index].store(value, Ordering::Relaxed);
    }

    /// Retrieve an entry from the cache
    /// Returns Some(score) if the key matches exactly, None otherwise
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key
    ///
    /// # Returns
    ///
    /// The evaluation score if the key matches, None otherwise
    pub fn probe(&self, key: u64) -> Option<i32> {
        let index = self.index(key);
        let entry = self.table[index].load(Ordering::Relaxed);

        if entry == 0 {
            return None;
        }

        let (entry_key, score) = Self::unpack(entry);
        if (entry_key) == (key & KEY_MASK) {
            Some(score)
        } else {
            None
        }
    }

    /// Calculate index from key using multiplicative hashing
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key
    ///
    /// # Returns
    ///
    /// The index in the cache table
    #[inline]
    fn index(&self, key: u64) -> usize {
        (key.rotate_left(SCORE_BITS) & self.mask) as usize
    }

    /// Pack key and score into a single u64
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key
    /// * `score` - The evaluation score
    ///
    /// # Returns
    ///
    /// The packed u64 value
    #[inline]
    fn pack(key: u64, score: i32) -> u64 {
        let key_bits = (key & KEY_MASK) << SCORE_BITS;
        let score_bits = score as u64 & SCORE_MASK;

        key_bits | score_bits
    }

    /// Unpack key and score from a u64
    ///
    /// # Arguments
    ///
    /// * `value` - The packed u64 value
    ///
    /// # Returns
    ///
    /// A tuple containing the key and score
    #[inline]
    fn unpack(entry: u64) -> (u64, Score) {
        let key = (entry >> SCORE_BITS) & KEY_MASK;
        let score = ((entry & SCORE_MASK) as i16) as Score;

        (key, score)
    }

    /// Clear all entries in the cache
    pub fn clear(&self) {
        for entry in self.table.iter() {
            entry.store(0, Ordering::Relaxed);
        }
    }
}
