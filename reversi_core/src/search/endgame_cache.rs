use crate::{
    square::Square,
    transposition_table::Bound,
    types::{Depth, Score},
};

const VALUE_BITS: u32 = 8;
const EMPTIES_BITS: u32 = 3;
const BOUND_BITS: u32 = 2;
const BEST_MOVE_BITS: u32 = 6;

const EMPTIES_MASK: u64 = 0x7;
const BOUND_MASK: u64 = 0x3;
const BEST_MASK: u64 = 0x3F;

const EMPTIES_SHIFT: u32 = VALUE_BITS;
const BOUND_SHIFT: u32 = VALUE_BITS + EMPTIES_BITS;
const BEST_SHIFT: u32 = VALUE_BITS + EMPTIES_BITS + BOUND_BITS;
const META_BITS: u32 = VALUE_BITS + EMPTIES_BITS + BOUND_BITS + BEST_MOVE_BITS;
const KEY_MASK: u64 = !((1u64 << META_BITS) - 1);

const EMPTIES_SHIFTED_MASK: u64 = EMPTIES_MASK << EMPTIES_SHIFT;
const BOUND_SHIFTED_MASK: u64 = BOUND_MASK << BOUND_SHIFT;
const BEST_SHIFTED_MASK: u64 = BEST_MASK << BEST_SHIFT;
const EMPTIES_OFFSET: Depth = 5;
const MAX_EMPTIES: Depth = EMPTIES_OFFSET + EMPTIES_MASK as Depth;

/// Entry structure for endgame cache
pub struct EndGameCacheEntry {
    pub score: Score,
    pub best_move: Square,
    pub bound: Bound,
}

impl EndGameCacheEntry {
    /// Determine if the entry should cause a cut
    ///
    /// # Arguments
    ///
    /// * `beta` - The beta value for comparison
    ///
    /// # Returns
    ///
    /// * `bool` - True if the entry causes a cut, false otherwise
    pub fn should_cut(&self, beta: Score) -> bool {
        let bound = if self.score >= beta {
            Bound::Lower as u8
        } else {
            Bound::Upper as u8
        };
        (self.bound as u8 & bound) != 0
    }
}

/// Endgame cache structure
pub struct EndGameCache {
    table: Box<[u64]>,
    mask: u64,
}

impl EndGameCache {
    /// Create a new endgame cache with the specified size
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the cache in bits (e.g., 20 for 1M entries)
    ///
    /// # Returns
    ///
    /// * `EndGameCache` - The created endgame cache
    pub fn new(size: u32) -> Self {
        let entries = 1usize << size;
        let mask = (entries as u64) - 1;
        let table = vec![0u64; entries].into_boxed_slice();
        EndGameCache { table, mask }
    }

    /// Calculate the index in the cache table
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position
    ///
    /// # Returns
    ///
    /// * `usize` - The index in the cache table
    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        key as usize & self.mask as usize
    }

    /// Pack the entry into a u64
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position
    /// * `value` - The score to store
    /// * `n_empties` - The number of empty squares on the board
    /// * `bound` - The bound type (Exact, Lower, Upper)
    /// * `best_move` - The best move found in this position
    ///
    /// # Returns
    ///
    /// * `u64` - The packed cache entry
    #[inline(always)]
    fn pack(key: u64, value: Score, n_empties: Depth, bound: Bound, best_move: Square) -> u64 {
        debug_assert!(n_empties >= EMPTIES_OFFSET && n_empties <= MAX_EMPTIES);

        let v = (value as i8 as u8) as u64;
        let e = ((n_empties - EMPTIES_OFFSET) as u64) << EMPTIES_SHIFT;
        let b = (bound as u8 as u64) << BOUND_SHIFT;
        let bm = (best_move as u64) << BEST_SHIFT;

        (key & KEY_MASK) | bm | b | e | v
    }

    /// Unpack the value from the entry
    ///
    /// # Arguments
    ///
    /// * `entry` - The packed cache entry
    ///
    /// # Returns
    ///
    /// * `Score` - The unpacked score value
    #[inline(always)]
    fn unpack_value(entry: u64) -> Score {
        (entry as u8) as i8 as Score
    }

    /// Unpack the bound from the entry
    ///
    /// # Arguments
    ///
    /// * `entry` - The packed cache entry
    ///
    /// # Returns
    ///
    /// * `Bound` - The bound type
    #[inline(always)]
    fn unpack_bound(entry: u64) -> Bound {
        unsafe { std::mem::transmute(((entry & BOUND_SHIFTED_MASK) >> BOUND_SHIFT) as u8) }
    }

    /// Unpack the best move from the entry
    ///
    /// # Arguments
    ///
    /// * `entry` - The packed cache entry
    ///
    /// # Returns
    /// * `u8` - The best move as a u8
    #[inline(always)]
    fn unpack_best(entry: u64) -> u8 {
        ((entry & BEST_SHIFTED_MASK) >> BEST_SHIFT) as u8
    }

    /// Probe the cache for an entry
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position
    /// * `empties` - The number of empty squares on the board
    ///
    /// # Returns
    ///
    /// * `Option<EndGameCacheEntry>` - The cached entry if found
    #[inline(always)]
    pub fn probe(&self, key: u64, n_empties: Depth) -> Option<EndGameCacheEntry> {
        let idx = self.index(key);
        let entry = unsafe { *self.table.get_unchecked(idx) };

        if entry == 0 {
            return None;
        }

        let expected_key = key & KEY_MASK;
        let expected_empties = ((n_empties - EMPTIES_OFFSET) as u64) << EMPTIES_SHIFT;
        let mask = KEY_MASK | EMPTIES_SHIFTED_MASK;

        if (entry & mask) != (expected_key | expected_empties) {
            return None;
        }

        Some(EndGameCacheEntry {
            score: Self::unpack_value(entry),
            best_move: Square::from_u8_unchecked(Self::unpack_best(entry)),
            bound: Self::unpack_bound(entry),
        })
    }

    /// Store an entry
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position
    /// * `n_empties` - The number of empty squares on the board
    /// * `value` - The score to store
    /// * `bound` - The bound type (Exact, Lower, Upper)
    /// * `best_move` - The best move found in this position
    #[inline(always)]
    pub fn store(
        &mut self,
        key: u64,
        n_empties: Depth,
        value: Score,
        bound: Bound,
        best_move: Square,
    ) {
        let idx = self.index(key);
        let entry = Self::pack(key, value, n_empties, bound, best_move);
        unsafe {
            *self.table.get_unchecked_mut(idx) = entry;
        }
    }
}
