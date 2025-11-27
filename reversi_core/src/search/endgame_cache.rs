use crate::{square::Square, types::Score};

const SCORE_BITS: u32 = 8;
const BOUND_BITS: u32 = 1;
const BEST_MOVE_BITS: u32 = 7;

const BOUND_MASK: u64 = 0x1;
const BEST_MOVE_MASK: u64 = 0x7F;

const BOUND_SHIFT: u32 = SCORE_BITS;
const BEST_MOVE_SHIFT: u32 = SCORE_BITS + BOUND_BITS;
const META_BITS: u32 = BEST_MOVE_SHIFT + BEST_MOVE_BITS;
const KEY_MASK: u64 = !((1u64 << META_BITS) - 1);

#[derive(Clone, Copy)]
pub enum EndGameCacheBound {
    Lower = 0,
    Upper = 1,
}

impl EndGameCacheBound {
    #[inline(always)]
    pub fn determine_bound(score: Score, beta: Score) -> Self {
        if score < beta {
            EndGameCacheBound::Upper
        } else {
            EndGameCacheBound::Lower
        }
    }
}

/// Entry structure for endgame cache
#[derive(Clone, Copy)]
pub struct EndGameCacheEntry {
    pub score: Score,
    pub best_move: Square,
    pub bound: EndGameCacheBound,
}

impl EndGameCacheEntry {
    #[inline(always)]
    pub fn can_cut(&self, beta: Score) -> bool {
        match self.bound {
            EndGameCacheBound::Lower => self.score >= beta,
            EndGameCacheBound::Upper => self.score < beta,
        }
    }
}

/// Endgame cache structure
pub struct EndGameCache {
    table: Box<[u64]>,
    mask: usize,
}

impl EndGameCache {
    /// Create a new endgame cache with the specified size
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the cache in bits (e.g., 20 for 1M entries)
    pub fn new(size: u32) -> Self {
        let entries = 1usize << size;
        EndGameCache {
            table: vec![0u64; entries].into_boxed_slice(),
            mask: entries - 1,
        }
    }

    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    #[inline(always)]
    fn pack(key: u64, value: Score, bound: EndGameCacheBound, best_move: Square) -> u64 {
        (key & KEY_MASK)
            | ((best_move as u64) << BEST_MOVE_SHIFT)
            | ((bound as u64) << BOUND_SHIFT)
            | ((value as i8 as u8) as u64)
    }

    /// Probe the cache for an entry
    #[inline(always)]
    pub fn probe(&self, key: u64) -> Option<EndGameCacheEntry> {
        let idx = self.index(key);
        let entry = unsafe { *self.table.get_unchecked(idx) };

        if (entry ^ key) & KEY_MASK != 0 {
            return None;
        }

        Some(EndGameCacheEntry {
            score: (entry as u8) as i8 as Score,
            best_move: Square::from_u8_unchecked(
                ((entry >> BEST_MOVE_SHIFT) & BEST_MOVE_MASK) as u8,
            ),
            bound: unsafe { std::mem::transmute(((entry >> BOUND_SHIFT) & BOUND_MASK) as u8) },
        })
    }

    /// Store an entry
    #[inline(always)]
    pub fn store(&mut self, key: u64, value: Score, bound: EndGameCacheBound, best_move: Square) {
        let idx = self.index(key);
        unsafe {
            *self.table.get_unchecked_mut(idx) = Self::pack(key, value, bound, best_move);
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.table.fill(0);
    }
}
