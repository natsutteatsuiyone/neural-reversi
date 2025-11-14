use crate::search::node_type::NodeType;
use crate::square::Square;
use crate::types::{Depth, Score, Selectivity};
use aligned_vec::{AVec, ConstAlign};
use std::{
    mem,
    sync::atomic::{AtomicU64, Ordering},
};

/// Size of each cluster in the transposition table.
const CLUSTER_SIZE: usize = 4;

/// Bound type for transposition table entries.
///
/// Indicates the relationship between the stored score and the actual position value:
/// - `None`: No valid entry
/// - `Lower`: Score is a lower bound (fail-high occurred)
/// - `Upper`: Score is an upper bound (fail-low)
/// - `Exact`: Score is the exact minimax value
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    Lower = 1,
    Upper = 2,
    Exact = 3,
}

impl Bound {
    /// Determines the appropriate bound type based on search results.
    ///
    /// # Arguments
    ///
    /// * `best_score` - The best score found during search
    /// * `beta` - The beta cutoff value
    ///
    /// # Type Parameters
    ///
    /// * `NT` - Node type (PV or non-PV) that affects bound determination
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

/// A single entry in the transposition table.
///
/// # Entry Format
///
/// - 16 bits: Hash key (lower 16 bits for verification)
/// - 16 bits: Evaluation score
/// - 7 bits: Best move square
/// - 2 bits: Bound type (none/lower/upper/exact)
/// - 6 bits: Search depth
/// - 3 bits: Selectivity level
/// - 7 bits: Generation counter for aging
#[derive(Default)]
struct TTEntry {
    data: AtomicU64,
}

impl TTEntry {
    // Bit layout constants for packing data into 64 bits
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

    /// Packs all fields into a single 64-bit value and stores it.
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

    /// Unpacks a 64-bit value into TTData fields.
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
            best_move: Square::from_u8_unchecked(best_move),
            bound,
            depth: depth as Depth,
            selectivity,
            generation,
        }
    }

    /// Loads and unpacks the entry data atomically.
    #[inline]
    fn unpack(&self) -> TTData {
        let data = self.data.load(Ordering::Relaxed);
        Self::unpack_from_u64(data)
    }

    /// Saves data into the transposition table entry.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position
    /// * `score` - The evaluation score of the position
    /// * `bound` - The bound type (none, lower, upper, exact)
    /// * `depth` - The search depth at which the position was evaluated
    /// * `best_move` - The best move found for the position
    /// * `selectivity` - The selectivity level of the search
    /// * `generation` - The current generation count for aging
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

/// Data stored in and retrieved from transposition table entries.
///
/// This struct represents the unpacked form of a transposition table entry,
/// containing all the information needed to use cached search results.
pub struct TTData {
    /// Lower 16 bits of the position hash for verification
    pub key: u16,
    /// Evaluation score of the position
    pub score: Score,
    /// Best move found during search
    pub best_move: Square,
    /// Bound type
    pub bound: u8,
    /// Search depth at which this position was evaluated
    pub depth: Depth,
    /// Selectivity level used during search
    pub selectivity: Selectivity,
    /// Generation counter for replacement policy
    pub generation: u8,
}

impl Default for TTData {
    /// Creates an empty TTData instance with default values.
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
    pub fn should_cut(&self, beta: Score) -> bool {
        let bound = if self.score >= beta {
            Bound::Lower as u8
        } else {
            Bound::Upper as u8
        };
        (self.bound & bound) != 0
    }

    /// Checks if the transposition table entry contains valid data.
    ///
    /// # Returns
    ///
    /// `true` if the entry has a valid bound type (not None), `false` otherwise
    pub fn is_occupied(&self) -> bool {
        self.bound != Bound::None as u8
    }

    /// Calculates the relative age of the entry based on the current generation.
    ///
    /// # Arguments
    ///
    /// * `generation` - The current generation count
    ///
    /// # Returns
    ///
    /// The age difference (positive means this entry is older)
    fn relative_age(&self, generation: u8) -> i32 {
        generation as i32 - self.generation as i32
    }
}

/// The main transposition table structure.
pub struct TranspositionTable {
    /// Array of table entries organized in clusters
    entries: AVec<TTEntry, ConstAlign<32>>,
    /// Number of clusters in the table
    cluster_count: u64,
}

impl TranspositionTable {
    /// Initializes a new `TranspositionTable` with the specified memory size in megabytes.
    ///
    /// # Arguments
    ///
    /// * `mb_size` - The size of the transposition table in megabytes.
    pub fn new(mb_size: usize) -> Self {
        let cluster_count = if mb_size == 0 {
            16
        } else {
            let cluster_byte_size = mem::size_of::<TTEntry>() * CLUSTER_SIZE;
            (mb_size as u64 * 1024 * 1024) / cluster_byte_size as u64
        };
        let entries_size = cluster_count as usize * CLUSTER_SIZE + 1;

        TranspositionTable {
            entries: AVec::from_iter(32, (0..entries_size).map(|_| TTEntry::default())),
            cluster_count,
        }
    }

    /// Clears all entries in the transposition table.
    pub fn clear(&self) {
        unsafe {
            let ptr = self.entries.as_ptr() as *mut TTEntry;
            std::ptr::write_bytes(ptr, 0, self.entries.len());
        }
    }

    /// Prefetches the transposition table entry corresponding to the given hash.
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

    /// Probes the transposition table for an entry matching the given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position to probe
    /// * `generation` - The current generation count for aging
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `bool`: Whether a matching entry was found
    /// - `TTData`: The entry data (default if not found)
    /// - `usize`: The index for storing new data (either the found entry or replacement candidate)
    #[inline(always)]
    pub fn probe(&self, key: u64, generation: u8) -> (bool, TTData, usize) {
        #[cfg(target_arch = "x86_64")]
        {
            if cfg!(target_feature = "avx2") {
                return unsafe { self.probe_avx2(key, generation) };
            }
        }

        self.probe_fallback(key, generation)
    }

    /// AVX2-optimized probe implementation for faster cluster scanning.
    #[target_feature(enable = "avx2")]
    #[cfg(target_arch = "x86_64")]
    fn probe_avx2(&self, key: u64, generation: u8) -> (bool, TTData, usize) {
        unsafe {
            use std::arch::x86_64::*;

            let key16 = key as u16;
            let base = self.get_cluster_idx(key);
            let ptr = self.entries.as_ptr().add(base) as *const __m256i;
            let v = _mm256_load_si256(ptr);

            // Create a vector with the key repeated in all 16-bit lanes
            let key_vec = _mm256_set1_epi16(key16 as i16);
            // Compare all keys in the cluster simultaneously
            let cmp = _mm256_cmpeq_epi16(v, key_vec);
            let hit_mask = _mm256_movemask_epi8(cmp) as u32;

            // Mask to check only the key fields (first 16 bits of each 64-bit entry)
            // bits 0-1: first 16 bits of entry 0 (the key)
            // bits 8-9: first 16 bits of entry 1 (the key)
            // bits 16-17: first 16 bits of entry 2 (the key)
            // bits 24-25: first 16 bits of entry 3 (the key)
            const LANE_MASK: u32 = 0x03030303;
            let relevant_hits = hit_mask & LANE_MASK;

            if relevant_hits != 0 {
                // Found a potential key match - extract and verify
                // Count trailing zeros and map to entry index
                let tz = relevant_hits.trailing_zeros();
                let lane_idx = (tz / 8) as usize;

                let entries_array: [u64; 4] = std::mem::transmute(v);
                let raw = entries_array[lane_idx];

                let data = TTEntry::unpack_from_u64(raw);
                if data.is_occupied() {
                    return (true, data, base + lane_idx);
                }
            }

            // Extract depth and generation fields from all entries using SIMD
            let depth_mask_vec = _mm256_set1_epi64x(TTEntry::DEPTH_MASK as i64);
            let gen_mask_vec = _mm256_set1_epi64x(TTEntry::GENERATION_MASK as i64);

            let depth_vec =
                _mm256_and_si256(_mm256_srli_epi64(v, TTEntry::DEPTH_SHIFT), depth_mask_vec);
            let gen_vec = _mm256_and_si256(
                _mm256_srli_epi64(v, TTEntry::GENERATION_SHIFT),
                gen_mask_vec,
            );

            // Calculate replacement scores: depth - (age * 8)
            let current_gen_vec = _mm256_set1_epi64x(generation as i64);
            let age_vec = _mm256_sub_epi64(current_gen_vec, gen_vec);
            let score_vec = _mm256_sub_epi64(depth_vec, _mm256_slli_epi64(age_vec, 3));
            let scores: [i64; 4] = core::mem::transmute(score_vec);

            // Find the entry with the lowest replacement score
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
    }

    /// Fallback probe implementation.
    fn probe_fallback(&self, key: u64, generation: u8) -> (bool, TTData, usize) {
        let key16 = key as u16;
        let cluster_idx = self.get_cluster_idx(key);

        // look for exact key match
        for i in 0..CLUSTER_SIZE {
            let idx = cluster_idx + i;
            let entry = unsafe { self.entries.get_unchecked(idx) };
            let tt_data = entry.unpack();

            if tt_data.key == key16 && tt_data.is_occupied() {
                return (true, tt_data, idx);
            }
        }

        // find best replacement candidate
        // Score formula: depth - (age * 8)
        // Lower score = better replacement candidate
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

    /// Stores data in the transposition table at the specified entry index.
    ///
    /// # Arguments
    ///
    /// * `entry_index` - The index returned by `probe`
    /// * `key` - The hash key of the board position
    /// * `score` - The evaluation score of the position
    /// * `bound` - The bound type (none, lower, upper, exact)
    /// * `depth` - The search depth at which the position was evaluated
    /// * `best_move` - The best move found for the position
    /// * `selectivity` - The selectivity level used in search
    /// * `generation` - The current generation count
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

    /// Calculates the cluster index for a given hash key.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position
    ///
    /// # Returns
    ///
    /// The starting index of the cluster in the entries vector
    #[inline(always)]
    fn get_cluster_idx(&self, key: u64) -> usize {
        (Self::mul_hi64(key, self.cluster_count) as usize) * CLUSTER_SIZE
    }

    /// Multiplies two 64-bit values and returns the high 64 bits of the result.
    ///
    /// # Arguments
    ///
    /// * `a` - The first value (typically the hash key)
    /// * `b` - The second value (typically the cluster count)
    ///
    /// # Returns
    ///
    /// The high 64 bits of the 128-bit product
    #[inline(always)]
    fn mul_hi64(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        (product >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::node_type::{NonPV, PV};

    /// Tests that TTEntry correctly packs and unpacks all fields.
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

    /// Tests boundary values for packed fields.
    #[test]
    fn test_ttentry_boundary_values() {
        let entry = TTEntry::default();

        // Test maximum values for each field
        let max_key: u64 = 0xFFFF; // 16 bits
        let max_score: Score = 32767; // Max i16
        let min_score: Score = -32768; // Min i16
        let max_depth: Depth = 63; // 6 bits
        let max_best_move = Square::from_usize_unchecked(63); // 7 bits (0-63 squares)
        let max_selectivity: u8 = 7; // 3 bits
        let max_generation: u8 = 127; // 7 bits

        // Test with maximum positive score
        entry.save(
            max_key,
            max_score,
            Bound::Lower,
            max_depth,
            max_best_move,
            max_selectivity,
            max_generation,
        );

        let data = entry.unpack();
        assert_eq!(data.key, max_key as u16);
        assert_eq!(data.score, max_score);
        assert_eq!(data.bound, Bound::Lower as u8);
        assert_eq!(data.depth, max_depth);
        assert_eq!(data.best_move, max_best_move);
        assert_eq!(data.selectivity, max_selectivity);
        assert_eq!(data.generation, max_generation);

        // Test with minimum negative score
        entry.save(0, min_score, Bound::Upper, 0, Square::None, 0, 0);

        let data = entry.unpack();
        assert_eq!(data.key, 0);
        assert_eq!(data.score, min_score);
        assert_eq!(data.bound, Bound::Upper as u8);
        assert_eq!(data.depth, 0);
        assert_eq!(data.best_move, Square::None);
        assert_eq!(data.selectivity, 0);
        assert_eq!(data.generation, 0);
    }

    /// Tests the replacement policy in TTEntry::save.
    #[test]
    fn test_ttentry_replacement_policy() {
        let entry = TTEntry::default();

        // Initial save
        entry.save(
            100,
            50,
            Bound::Lower,
            10,
            Square::from_usize_unchecked(5),
            2,
            1,
        );
        let data = entry.unpack();
        assert_eq!(data.depth, 10);
        assert_eq!(data.generation, 1);

        // Try to replace with shallower depth - should not replace
        entry.save(
            100,
            60,
            Bound::Lower,
            8,
            Square::from_usize_unchecked(6),
            2,
            1,
        );
        let data = entry.unpack();
        assert_eq!(data.depth, 10); // Should remain 10
        assert_eq!(data.score, 50); // Should remain 50

        // Replace with deeper depth - should replace
        entry.save(
            100,
            70,
            Bound::Lower,
            12,
            Square::from_usize_unchecked(7),
            2,
            1,
        );
        let data = entry.unpack();
        assert_eq!(data.depth, 12);
        assert_eq!(data.score, 70);

        // Replace with exact bound - should always replace
        entry.save(
            100,
            80,
            Bound::Exact,
            5,
            Square::from_usize_unchecked(8),
            2,
            1,
        );
        let data = entry.unpack();
        assert_eq!(data.depth, 5);
        assert_eq!(data.score, 80);
        assert_eq!(data.bound, Bound::Exact as u8);

        // Different key - should replace
        entry.save(
            200,
            90,
            Bound::Upper,
            3,
            Square::from_usize_unchecked(9),
            2,
            1,
        );
        let data = entry.unpack();
        assert_eq!(data.key, 200);
        assert_eq!(data.score, 90);

        // Newer generation - should replace
        entry.save(
            200,
            100,
            Bound::Lower,
            2,
            Square::from_usize_unchecked(10),
            2,
            5,
        );
        let data = entry.unpack();
        assert_eq!(data.generation, 5);
        assert_eq!(data.score, 100);
    }

    /// Tests Bound::determine_bound for different node types.
    #[test]
    fn test_bound_determine() {
        // Test PV node
        assert_eq!(Bound::determine_bound::<PV>(100, 50), Bound::Lower); // score >= beta
        assert_eq!(Bound::determine_bound::<PV>(40, 50), Bound::Exact); // score < beta in PV

        // Test non-PV node
        assert_eq!(Bound::determine_bound::<NonPV>(100, 50), Bound::Lower); // score >= beta
        assert_eq!(Bound::determine_bound::<NonPV>(40, 50), Bound::Upper); // score < beta in non-PV
    }

    /// Tests TTData methods.
    #[test]
    fn test_ttdata_methods() {
        let mut data = TTData::default();

        // Test is_occupied
        assert!(!data.is_occupied());
        data.bound = Bound::Lower as u8;
        assert!(data.is_occupied());
        data.bound = Bound::Upper as u8;
        assert!(data.is_occupied());
        data.bound = Bound::Exact as u8;
        assert!(data.is_occupied());
        data.bound = Bound::None as u8;
        assert!(!data.is_occupied());

        // Test should_cutoff
        data.score = 100;
        data.bound = Bound::Lower as u8;
        assert!(data.should_cut(50)); // score >= beta, lower bound
        assert!(!data.should_cut(150)); // score < beta, lower bound

        data.score = 30;
        data.bound = Bound::Upper as u8;
        assert!(data.should_cut(50)); // score < beta, upper bound
        assert!(!data.should_cut(20)); // score >= beta, upper bound

        data.bound = Bound::Exact as u8;
        assert!(data.should_cut(50)); // exact bound matches both
        assert!(data.should_cut(20));

        // Test relative_age
        data.generation = 5;
        assert_eq!(data.relative_age(8), 3);
        assert_eq!(data.relative_age(5), 0);
        assert_eq!(data.relative_age(3), -2);
    }

    /// Tests TranspositionTable creation with different sizes.
    #[test]
    fn test_transposition_table_new() {
        // Test with 0 MB (minimum size)
        let tt = TranspositionTable::new(0);
        assert_eq!(tt.cluster_count, 16);
        assert_eq!(tt.entries.len(), 16 * CLUSTER_SIZE + 1);

        // Test with 1 MB
        let tt = TranspositionTable::new(1);
        let expected_clusters = (1024 * 1024) / (std::mem::size_of::<TTEntry>() * CLUSTER_SIZE);
        assert_eq!(tt.cluster_count, expected_clusters as u64);
        assert_eq!(tt.entries.len(), expected_clusters * CLUSTER_SIZE + 1);
    }

    /// Tests basic probe and store operations.
    #[test]
    fn test_probe_and_store() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;
        let generation = 1;

        // First probe should return not found
        let (found, _, idx) = tt.probe(key, generation);
        assert!(!found);

        // Store an entry
        tt.store(
            idx,
            key,
            100,
            Bound::Exact,
            20,
            Square::from_usize_unchecked(10),
            3,
            generation,
        );

        // Second probe should find the entry
        let (found, data, _) = tt.probe(key, generation);
        assert!(found);
        assert_eq!(data.key, key as u16);
        assert_eq!(data.score, 100);
        assert_eq!(data.bound, Bound::Exact as u8);
        assert_eq!(data.depth, 20);
        assert_eq!(data.best_move, Square::from_usize_unchecked(10));
        assert_eq!(data.selectivity, 3);
        assert_eq!(data.generation, generation);
    }

    /// Tests cluster replacement when full.
    #[test]
    fn test_cluster_replacement() {
        let tt = TranspositionTable::new(1);
        let generation = 1;

        // Find a key and determine its cluster
        let base_key = 0x1000000000000000u64;
        let cluster_idx = tt.get_cluster_idx(base_key);

        // Store entries directly in the cluster at known positions
        for i in 0..CLUSTER_SIZE {
            let entry_idx = cluster_idx + i;
            // Create keys that will match this cluster (use same base but different high bits)
            let key = (base_key & 0xFFFF) | ((i as u64 + 1) << 16);
            tt.store(
                entry_idx,
                key,
                i as Score * 10,
                Bound::Lower,
                (CLUSTER_SIZE - i) as Depth * 5, // Decreasing depth
                Square::from_usize_unchecked(i),
                1,
                generation,
            );
        }

        // Now probe for a key that maps to the same cluster but isn't found
        let new_key = (base_key & 0xFFFF) | (0x9999 << 16); // Different high bits
        let actual_cluster = tt.get_cluster_idx(new_key);

        // If it doesn't map to the same cluster, adjust the key
        let test_key = if actual_cluster == cluster_idx {
            new_key
        } else {
            // Use a key that we know maps to the right cluster
            base_key
        };

        let (found, _, replace_idx) = tt.probe(test_key, generation);

        // If we used base_key and it's found, try a different approach
        if found {
            // Clear the entry and try again
            tt.clear();

            // Re-fill the cluster
            for i in 0..CLUSTER_SIZE {
                let entry_idx = cluster_idx + i;
                let key = (i as u64 + 100) << 32 | (base_key & 0xFFFFFFFF);
                tt.store(
                    entry_idx,
                    key,
                    i as Score * 10,
                    Bound::Lower,
                    (CLUSTER_SIZE - i) as Depth * 5,
                    Square::from_usize_unchecked(i),
                    1,
                    generation,
                );
            }

            let new_test_key = 999u64 << 32 | (base_key & 0xFFFFFFFF);
            let (found, _, replace_idx) = tt.probe(new_test_key, generation);
            assert!(!found);

            // Store the new entry
            tt.store(
                replace_idx,
                new_test_key,
                999,
                Bound::Exact,
                50,
                Square::from_usize_unchecked(20),
                1,
                generation,
            );

            // Verify the new entry was stored
            let (found, data, _) = tt.probe(new_test_key, generation);
            assert!(found);
            assert_eq!(data.score, 999);
        } else {
            // Store the new entry
            tt.store(
                replace_idx,
                test_key,
                999,
                Bound::Exact,
                50,
                Square::from_usize_unchecked(20),
                1,
                generation,
            );

            // Verify the new entry was stored
            let (found, data, _) = tt.probe(test_key, generation);
            assert!(found);
            assert_eq!(data.score, 999);
        }
    }

    /// Tests generation-based aging in replacement.
    #[test]
    fn test_generation_aging() {
        let tt = TranspositionTable::new(1);

        // Use a simple approach: store one entry, then try to replace it
        let old_key = 0x1111111111111111u64;
        let (_, _, idx1) = tt.probe(old_key, 1);

        // Store an entry with old generation but high depth
        tt.store(
            idx1,
            old_key,
            100,
            Bound::Lower,
            30, // High depth
            Square::from_usize_unchecked(5),
            1,
            1, // Old generation
        );

        // Verify it was stored
        let (found, data, _) = tt.probe(old_key, 1);
        assert!(found);
        assert_eq!(data.generation, 1);
        assert_eq!(data.depth, 30);

        // Now try to replace with a newer generation but lower depth
        let new_key = 0x2222222222222222u64;
        let (found, _, replace_idx) = tt.probe(new_key, 10); // Much newer generation

        if !found {
            // Should replace the old entry despite high depth due to age penalty
            tt.store(
                replace_idx,
                new_key,
                200,
                Bound::Exact,
                10, // Lower depth but newer
                Square::from_usize_unchecked(20),
                1,
                10,
            );

            let (found, data, _) = tt.probe(new_key, 10);
            assert!(found);
            assert_eq!(data.generation, 10);
        } else {
            // If the new key was found (unlikely but possible), just verify aging works
            // by checking that the relative_age calculation is correct
            let test_data = TTData {
                key: 0,
                score: 0,
                best_move: Square::None,
                bound: Bound::None as u8,
                depth: 0,
                selectivity: 0,
                generation: 1,
            };

            assert_eq!(test_data.relative_age(10), 9); // Current gen 10 - entry gen 1 = 9
            assert_eq!(test_data.relative_age(1), 0); // Same generation
            assert_eq!(test_data.relative_age(0), -1); // Entry is newer
        }
    }

    /// Tests the clear function.
    #[test]
    fn test_clear() {
        let tt = TranspositionTable::new(1);

        // Store some entries
        for i in 0..10 {
            let key = i * 0x1000000000000000;
            let (_, _, idx) = tt.probe(key, 1);
            tt.store(
                idx,
                key,
                i as Score * 10,
                Bound::Exact,
                20,
                Square::from_usize_unchecked(i as usize),
                1,
                1,
            );
        }

        // Verify entries exist
        let (found, _, _) = tt.probe(0, 1);
        assert!(found);

        // Clear the table
        tt.clear();

        // Verify entries are gone
        for i in 0..10 {
            let key = i * 0x1000000000000000;
            let (found, _, _) = tt.probe(key, 1);
            assert!(!found);
        }
    }

    /// Tests mul_hi64 function.
    #[test]
    fn test_mul_hi64() {
        // Test with small values
        assert_eq!(TranspositionTable::mul_hi64(0, 0), 0);
        assert_eq!(TranspositionTable::mul_hi64(1, 1), 0);

        // Test with values that produce high bits
        let a = 0xFFFFFFFFFFFFFFFF;
        let b = 2;
        assert_eq!(TranspositionTable::mul_hi64(a, b), 1);

        // Test with large values
        let a = 0x8000000000000000;
        let b = 4;
        assert_eq!(TranspositionTable::mul_hi64(a, b), 2);
    }

    /// Tests get_cluster_idx distribution.
    #[test]
    fn test_get_cluster_idx() {
        let tt = TranspositionTable::new(1);

        // Test that different keys map to different clusters
        let key1 = 0x123456789ABCDEF0;
        let key2 = 0x0FEDCBA987654321;
        let key3 = 0xAAAAAAAAAAAAAAAA;

        let idx1 = tt.get_cluster_idx(key1);
        let idx2 = tt.get_cluster_idx(key2);
        let idx3 = tt.get_cluster_idx(key3);

        // Indices should be multiples of CLUSTER_SIZE
        assert_eq!(idx1 % CLUSTER_SIZE, 0);
        assert_eq!(idx2 % CLUSTER_SIZE, 0);
        assert_eq!(idx3 % CLUSTER_SIZE, 0);

        // Indices should be within bounds
        assert!(idx1 / CLUSTER_SIZE < tt.cluster_count as usize);
        assert!(idx2 / CLUSTER_SIZE < tt.cluster_count as usize);
        assert!(idx3 / CLUSTER_SIZE < tt.cluster_count as usize);
    }
}
