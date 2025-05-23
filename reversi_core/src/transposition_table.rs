use crate::square::Square;
use crate::types::{Depth, NodeType, Score, Selectivity};
use std::{
    mem,
    sync::atomic::{AtomicU64, Ordering},
};

/// Size of each cluster in the transposition table.
const CLUSTER_SIZE: usize = 4;

/// Bound indicator for the transposition table entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Lower = 1,
    Upper = 2,
    Exact = 3,
}

impl Bound {
    #[inline]
    pub fn determine_bound<NT: NodeType>(best_score: Score, beta: Score) -> Bound {
        if best_score >= beta {
            Bound::Lower
        } else if NT::PV_NODE {
            Bound::Exact
        } else {
            Bound::Upper
        }
    }
}

/// Represents an entry in the transposition table.
#[derive(Default)]
struct TTEntry {
    data: AtomicU64,
}

/// Represents the transposition table.
pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    cluster_count: u64,
}

pub struct TTData {
    pub key: u16,
    pub score: Score,
    pub best_move: Square,
    pub bound: u8,
    pub depth: Depth,
    pub selectivity: Selectivity,
    pub generation: u8,
}

impl Default for TTData {
    fn default() -> Self {
        TTData {
            key: 0,
            score: 0,
            best_move: Square::None,
            bound: Bound::None as u8,
            depth: 0,
            selectivity: 0,
            generation: 0,
        }
    }
}

impl TTEntry {
    const KEY_SIZE: i32 = 16;
    const KEY_SHIFT: i32 = 0;
    const KEY_MASK: u64 = (1 << (Self::KEY_SIZE)) - 1;

    const SCORE_SIZE: i32 = 16;
    const SCORE_SHIFT: i32 = Self::KEY_SHIFT + Self::KEY_SIZE;
    const SCORE_MASK: u64 = (1 << (Self::SCORE_SIZE)) - 1;

    const BEST_MOVE_SIZE: i32 = 7;
    const BEST_MOVE_SHIFT: i32 = Self::SCORE_SHIFT + Self::SCORE_SIZE;
    const BEST_MOVE_MASK: u64 = (1 << (Self::BEST_MOVE_SIZE)) - 1;

    const BOUND_SIZE: i32 = 2;
    const BOUND_SHIFT: i32 = Self::BEST_MOVE_SHIFT + Self::BEST_MOVE_SIZE;
    const BOUND_MASK: u64 = (1 << (Self::BOUND_SIZE)) - 1;

    const DEPTH_SIZE: i32 = 6;
    const DEPTH_SHIFT: i32 = Self::BOUND_SHIFT + Self::BOUND_SIZE;
    const DEPTH_MASK: u64 = (1 << (Self::DEPTH_SIZE)) - 1;

    const SELECTIVITY_SIZE: i32 = 3;
    const SELECTIVITY_SHIFT: i32 = Self::DEPTH_SHIFT + Self::DEPTH_SIZE;
    const SELECTIVITY_MASK: u64 = (1 << (Self::SELECTIVITY_SIZE)) - 1;

    const GENERATION_SIZE: i32 = 7;
    const GENERATION_SHIFT: i32 = Self::SELECTIVITY_SHIFT + Self::SELECTIVITY_SIZE;
    const GENERATION_MASK: u64 = (1 << (Self::GENERATION_SIZE)) - 1;

    #[allow(clippy::too_many_arguments)]
    fn pack(
        &self,
        key: u16,
        score: Score,
        best_move: u8,
        bound: u8,
        depth: u8,
        selectivity: u8,
        generation: u8,
    ) {
        let data = key as u64
            | (((score as u16) as u64) << Self::SCORE_SHIFT)
            | ((best_move as u64) << Self::BEST_MOVE_SHIFT)
            | ((bound as u64) << Self::BOUND_SHIFT)
            | ((depth as u64) << Self::DEPTH_SHIFT)
            | ((selectivity as u64) << Self::SELECTIVITY_SHIFT)
            | ((generation as u64) << Self::GENERATION_SHIFT);
        self.data.store(data, Ordering::Relaxed);
    }

   #[inline]
   fn unpack_from_u64(data_u64: u64) -> TTData {
       let key = ((data_u64 >> Self::KEY_SHIFT) & Self::KEY_MASK) as u16;
       let score = ((data_u64 >> Self::SCORE_SHIFT) & Self::SCORE_MASK) as i16;
       let best_move = ((data_u64 >> Self::BEST_MOVE_SHIFT) & Self::BEST_MOVE_MASK) as u8;
       let bound = ((data_u64 >> Self::BOUND_SHIFT) & Self::BOUND_MASK) as u8;
       let depth = ((data_u64 >> Self::DEPTH_SHIFT) & Self::DEPTH_MASK) as u8;
       let selectivity = ((data_u64 >> Self::SELECTIVITY_SHIFT) & Self::SELECTIVITY_MASK) as u8;
       let generation = ((data_u64 >> Self::GENERATION_SHIFT) & Self::GENERATION_MASK) as u8;

       TTData {
           key,
           score: score as Score,
           best_move: Square::from_usize_unchecked(best_move as usize),
           bound,
           depth: depth as Depth,
           selectivity,
           generation,
       }
   }

   #[inline]
   fn unpack(&self) -> TTData {
       let data = self.data.load(Ordering::Relaxed);
       Self::unpack_from_u64(data)
   }

    /// Saves data into the transposition table entry.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position.
    /// * `score` - The evaluation score of the position.
    /// * `bound` - The bound type (none, lower, upper, exact).
    /// * `depth` - The search depth at which the position was evaluated.
    /// * `best_move` - The best move found for the position.
    /// * `selectivity` - The selectivity of the search.
    /// * `static_score` - The static evaluation score of the position.
    /// * `generation` - The current generation count.
    #[allow(clippy::too_many_arguments)]
    pub fn save(
        &self,
        key: u64,
        score: Score,
        bound: Bound,
        depth: Depth,
        best_move: Square,
        selectivity: u8,
        generation: u8,
    ) {
        let tt_data = self.unpack();
        let key16 = key as u16;
        let is_key_different = key16 != tt_data.key;

        if bound == Bound::Exact
            || is_key_different
            || depth >= tt_data.depth
            || selectivity > tt_data.selectivity
            || tt_data.relative_age(generation) > 0
        {
            self.pack(
                key16,
                score,
                best_move as u8,
                bound as u8,
                depth as u8,
                selectivity,
                generation,
            );
        }
    }
}

/// Data retrieved from the transposition table.
impl TTData {
    /// Determines whether a cutoff should occur based on the bound and beta value.
    ///
    /// # Arguments
    ///
    /// * `beta` - The beta value from alpha-beta pruning.
    ///
    /// # Returns
    ///
    /// `true` if a cutoff should occur, otherwise `false`.
    #[inline]
    pub fn should_cutoff(&self, beta: Score) -> bool {
        let bound = if self.score >= beta {
            Bound::Lower as u8
        } else {
            Bound::Upper as u8
        };
        (self.bound & bound) != 0
    }

    /// Checks if the transposition table entry is occupied.
    ///
    /// # Returns
    ///
    /// `true` if the entry is occupied, otherwise `false`.
    pub fn is_occupied(&self) -> bool {
        self.bound != Bound::None as u8
    }

    /// Calculates the relative age of the entry based on the current generation.
    ///
    /// # Arguments
    ///
    /// * `generation` - The current generation count.
    ///
    /// # Returns
    ///
    /// An `i32` representing the relative age.
    fn relative_age(&self, generation: u8) -> i32 {
        generation as i32 - self.generation as i32
    }
}

impl TranspositionTable {
    /// Initializes a new `TranspositionTable` with the specified memory size in megabytes.
    ///
    /// # Arguments
    ///
    /// * `mb_size` - The size of the transposition table in megabytes.
    ///
    /// # Panics
    ///
    /// Panics if `mb_size` is negative.
    pub fn new(mb_size: i32) -> Self {
        if mb_size < 0 {
            panic!("mb_size must be non-negative");
        }

        let cluster_count = if mb_size == 0 {
            16
        } else {
            let cluster_byte_size = mem::size_of::<TTEntry>() * CLUSTER_SIZE;
            (mb_size as u64 * 1024 * 1024) / cluster_byte_size as u64
        };
        let entries_size = cluster_count as usize * CLUSTER_SIZE + 1;

        TranspositionTable {
            entries: (0..entries_size).map(|_| TTEntry::default()).collect(),
            cluster_count,
        }
    }

    /// Clears the transposition table.
    pub fn clear(&self) {
        unsafe {
            let ptr = self.entries.as_ptr() as *mut TTEntry;
            std::ptr::write_bytes(ptr, 0, self.entries.len());
        }
    }

    /// Prefetches the transposition table entry corresponding to the given hash.
    ///
    /// This function uses the `_mm_prefetch` intrinsic to prefetch the memory address
    /// of the transposition table entry corresponding to the given hash key. This can
    /// improve performance by reducing cache misses during subsequent accesses.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position to prefetch.
    #[inline]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let index = self.get_cluster_idx(key);
            let addr = self.entries.as_ptr().add(index) as *const i8;
            std::arch::x86_64::_mm_prefetch(addr, std::arch::x86_64::_MM_HINT_T0);
        }
    }

    /// Probes the transposition table for an entry corresponding to the given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position to probe.
    /// * `generation` - The current generation count.
    ///
    /// # Returns
    ///
    /// A tuple where:
    /// - The first element is `true` if a valid entry was found, `false` otherwise.
    /// - The second element is the retrieved `TTData`.
    /// - The third element is the index of the entry in the transposition table.
    pub fn probe(&self, key: u64, generation: u8) -> (bool, TTData, usize) {
        if is_x86_feature_detected!("avx2") {
            return unsafe { self.probe_avx2(key, generation) };
        }

        let key16 = key as u16;
        let cluster_idx = self.get_cluster_idx(key);

        for i in 0..CLUSTER_SIZE {
            let idx = cluster_idx + i;
            let entry = unsafe { self.entries.get_unchecked(idx) };
            let tt_data = entry.unpack();

            if tt_data.key == key16 && tt_data.is_occupied() {
                return (true, tt_data, idx);
            }
        }

        let mut replace_idx = cluster_idx;
        let replace_data = unsafe { self.entries.get_unchecked(cluster_idx).unpack() };
        let mut replace_score =
            replace_data.depth as i32 - replace_data.relative_age(generation) * 8;

        for i in 1..CLUSTER_SIZE {
            let idx = cluster_idx + i;
            let entry = unsafe { self.entries.get_unchecked(idx) };
            let tt_data = entry.unpack();
            let score = tt_data.depth as i32 - tt_data.relative_age(generation) * 8;

            if score < replace_score {
                replace_score = score;
                replace_idx = idx;
            }
        }

        (false, TTData::default(), replace_idx)
    }

    #[inline]
    unsafe fn probe_avx2(&self, key: u64, generation: u8) -> (bool, TTData, usize) {
        use std::arch::x86_64::*;

        let key16 = key as u16;
        let base = self.get_cluster_idx(key);
        let ptr = self.entries.as_ptr().add(base) as *const __m256i;
        let v = _mm256_loadu_si256(ptr);

        let key_vec = _mm256_set1_epi16(key16 as i16);
        let cmp = _mm256_cmpeq_epi16(v, key_vec);
        let hit_mask = _mm256_movemask_epi8(cmp) as u32;

        const LANE_MASK: u32 = 0x0303_0303;
        let relevant_hits = hit_mask & LANE_MASK;

        if relevant_hits != 0 {
            let lane_idx = (relevant_hits.trailing_zeros() / 8) as usize;

            let entries_array: [u64; 4] = std::mem::transmute(v);
            let raw = entries_array[lane_idx];

            let data = TTEntry::unpack_from_u64(raw);
            if data.is_occupied() {
                return (true, data, base + lane_idx);
            }
        }

        let depth_mask_vec = _mm256_set1_epi64x(TTEntry::DEPTH_MASK as i64);
        let gen_mask_vec = _mm256_set1_epi64x(TTEntry::GENERATION_MASK as i64);

        let depth_vec = _mm256_and_si256(
            _mm256_srli_epi64(v, TTEntry::DEPTH_SHIFT),
            depth_mask_vec,
        );
        let gen_vec = _mm256_and_si256(
            _mm256_srli_epi64(v, TTEntry::GENERATION_SHIFT),
            gen_mask_vec,
        );

        let current_gen_vec = _mm256_set1_epi64x(generation as i64);
        let age_vec = _mm256_sub_epi64(current_gen_vec, gen_vec);
        let score_vec = _mm256_sub_epi64(depth_vec, _mm256_slli_epi64(age_vec, 3));
        let scores: [i64; 4] = core::mem::transmute(score_vec);

        let mut replace_idx = 0;
        let mut min_score = scores[0];

        if scores[1] < min_score {
            min_score = scores[1];
            replace_idx = 1;
        }
        if scores[2] < min_score {
            min_score = scores[2];
            replace_idx = 2;
        }
        if scores[3] < min_score {
            replace_idx = 3;
        }

        (false, TTData::default(), base + replace_idx)
    }

    /// Stores data in the transposition table at the specified entry index.
    ///
    /// # Arguments
    ///
    /// * `entry_index` - The index of the entry to store data in.
    /// * `key` - The hash key of the board position.
    /// * `score` - The evaluation score of the position.
    /// * `bound` - The bound type (none, lower, upper, exact).
    /// * `depth` - The search depth at which the position was evaluated.
    /// * `best_move` - The best move found for the position.
    /// * `generation` - The current generation count.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &self,
        entry_index: usize,
        key: u64,
        score: Score,
        bound: Bound,
        depth: Depth,
        best_move: Square,
        selectivity: u8,
        generation: u8,
    ) {
        let entry = unsafe { self.entries.get_unchecked(entry_index) };
        entry.save(key, score, bound, depth, best_move, selectivity, generation);
    }

    /// Calculates the cluster index based on the hash key.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position.
    ///
    /// # Returns
    ///
    /// The starting index of the cluster in the entries vector
    #[inline(always)]
    fn get_cluster_idx(&self, key: u64) -> usize {
        (Self::mul_hi64(key, self.cluster_count) as usize) * CLUSTER_SIZE
    }

    /// Multiplies two `u64` values and returns the high 64 bits of the product.
    ///
    /// # Arguments
    ///
    /// * `a` - The first `u64` value.
    /// * `b` - The second `u64` value.
    ///
    /// # Returns
    ///
    /// The high 64 bits of the product of `a` and `b`.
    #[inline(always)]
    fn mul_hi64(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        (product >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttentry_store_and_read() {
        let entry = TTEntry::default();
        let test_key: u64 = 42;
        let test_score: Score = -64;
        let test_bound = Bound::Exact;
        let test_depth: Depth = 60;
        let test_best_move = Square::from_usize_unchecked(3);
        let test_selectivity: u8 = 5;
        let test_generation: u8 = 127;

        entry.save(
            test_key,
            test_score,
            test_bound,
            test_depth,
            test_best_move,
            test_selectivity,
            test_generation,
        );

        let data = entry.unpack();
        assert_eq!(data.key, test_key as u16);
        assert_eq!(data.score, test_score);
        assert_eq!(data.bound, test_bound as u8);
        assert_eq!(data.depth, test_depth);
        assert_eq!(data.best_move, test_best_move);
        assert_eq!(data.selectivity, test_selectivity);
        assert_eq!(data.generation, test_generation);
    }
}
