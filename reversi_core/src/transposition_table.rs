use crate::search::node_type::NodeType;
use crate::square::Square;
use crate::types::{Depth, Score, Selectivity};
use aligned_vec::{AVec, ConstAlign};
use cfg_if::cfg_if;
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
    /// * `alpha` - The alpha cutoff value
    /// * `beta` - The beta cutoff value
    ///
    /// # Type Parameters
    ///
    /// * `NT` - Node type (PV or non-PV) that affects bound determination
    #[inline]
    pub fn determine_bound<NT: NodeType>(best_score: Score, alpha: Score, beta: Score) -> Bound {
        if best_score >= beta {
            return Bound::Lower;
        }

        if NT::PV_NODE && best_score > alpha {
            return Bound::Exact;
        }

        Bound::Upper
    }

    /// Converts an 8-bit value to a `Bound` without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the value is a valid `Bound` variant.
    #[inline]
    pub unsafe fn from_u8_unchecked(value: u8) -> Bound {
        unsafe { std::mem::transmute(value) }
    }
}

/// Result of a transposition table probe operation.
///
/// This enum provides a type-safe representation of probe results,
/// distinguishing between cache hits and misses while avoiding
/// unnecessary `TTEntryData` initialization on misses.
#[derive(Debug)]
pub enum TTProbeResult {
    /// Found an entry with a matching key.
    Hit {
        /// The data of the entry.
        data: TTEntryData,
        /// The index of the entry in the table.
        index: usize,
    },
    /// No matching entry found, or collision.
    Miss {
        /// The index where a new entry should be stored.
        index: usize,
    },
}

impl TTProbeResult {
    /// Returns the entry index for storing data.
    #[inline(always)]
    pub fn index(&self) -> usize {
        match self {
            TTProbeResult::Hit { index, .. } => *index,
            TTProbeResult::Miss { index } => *index,
        }
    }

    /// Returns the cached data if hit, otherwise None.
    #[inline(always)]
    pub fn data(&self) -> Option<TTEntryData> {
        match self {
            TTProbeResult::Hit { data, .. } => Some(*data),
            TTProbeResult::Miss { .. } => None,
        }
    }

    /// Returns the best move if hit, otherwise Square::None.
    #[inline(always)]
    pub fn best_move(&self) -> Square {
        match self {
            TTProbeResult::Hit { data, .. } => data.best_move(),
            TTProbeResult::Miss { .. } => Square::None,
        }
    }

    /// Returns true if the probe was a hit.
    #[inline(always)]
    pub fn is_hit(&self) -> bool {
        matches!(self, TTProbeResult::Hit { .. })
    }
}

/// A lightweight wrapper around raw TT entry data for lazy unpacking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTEntryData {
    raw: u64,
}

impl TTEntryData {
    #[inline(always)]
    pub fn key(&self) -> u32 {
        ((self.raw >> TTEntry::KEY_SHIFT) & TTEntry::KEY_MASK) as u32
    }

    #[inline(always)]
    pub fn score(&self) -> Score {
        ((self.raw >> TTEntry::SCORE_SHIFT) & TTEntry::SCORE_MASK) as i16 as Score
    }

    #[inline(always)]
    pub fn best_move(&self) -> Square {
        let val = ((self.raw >> TTEntry::BEST_MOVE_SHIFT) & TTEntry::BEST_MOVE_MASK) as u8;
        Square::from_u8_unchecked(val)
    }

    #[inline(always)]
    pub fn bound(&self) -> Bound {
        let val = ((self.raw >> TTEntry::BOUND_SHIFT) & TTEntry::BOUND_MASK) as u8;
        unsafe { Bound::from_u8_unchecked(val) }
    }

    #[inline(always)]
    pub fn depth(&self) -> Depth {
        ((self.raw >> TTEntry::DEPTH_SHIFT) & TTEntry::DEPTH_MASK) as u8 as Depth
    }

    #[inline(always)]
    pub fn selectivity(&self) -> Selectivity {
        let val = ((self.raw >> TTEntry::SELECTIVITY_SHIFT) & TTEntry::SELECTIVITY_MASK) as u8;
        Selectivity::from_u8(val)
    }

    #[inline(always)]
    pub fn generation(&self) -> u8 {
        ((self.raw >> TTEntry::GENERATION_SHIFT) & TTEntry::GENERATION_MASK) as u8
    }

    #[inline(always)]
    pub fn is_endgame(&self) -> bool {
        ((self.raw >> TTEntry::IS_ENDGAME_SHIFT) & TTEntry::IS_ENDGAME_MASK) != 0
    }

    /// Determines whether a cutoff should occur based on the bound and beta value.
    #[inline(always)]
    pub fn can_cut(&self, beta: Score) -> bool {
        let score = self.score();
        let bound_raw = ((self.raw >> TTEntry::BOUND_SHIFT) & TTEntry::BOUND_MASK) as u8;

        let required = if score >= beta {
            Bound::Lower as u8
        } else {
            Bound::Upper as u8
        };

        (bound_raw & required) != 0
    }

    /// Checks if the transposition table entry contains valid data.
    ///
    /// # Returns
    ///
    /// `true` if the entry has a valid bound type (not None), `false` otherwise
    #[inline(always)]
    pub fn is_occupied(&self) -> bool {
        ((self.raw >> TTEntry::BOUND_SHIFT) & TTEntry::BOUND_MASK) != 0
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
    #[inline(always)]
    fn relative_age(&self, generation: u8) -> i32 {
        generation as i32 - self.generation() as i32
    }
}

/// A single entry in the transposition table.
///
/// # Entry Format
///
/// - 22 bits: Hash key (lower 22 bits for verification)
/// - 16 bits: Evaluation score
/// - 7 bits: Best move square
/// - 2 bits: Bound type (none/lower/upper/exact)
/// - 6 bits: Search depth
/// - 3 bits: Selectivity level
/// - 7 bits: Generation counter for aging
/// - 1 bit: Endgame flag
#[derive(Default)]
#[repr(transparent)]
pub struct TTEntry {
    data: AtomicU64,
}

impl TTEntry {
    // Bit layout constants for packing data into 64 bits
    pub(crate) const KEY_SIZE: i32 = 22;
    pub(crate) const KEY_SHIFT: i32 = 0;
    pub(crate) const KEY_MASK: u64 = (1 << (Self::KEY_SIZE)) - 1;

    pub(crate) const SCORE_SIZE: i32 = 16;
    pub(crate) const SCORE_SHIFT: i32 = Self::KEY_SHIFT + Self::KEY_SIZE;
    pub(crate) const SCORE_MASK: u64 = (1 << (Self::SCORE_SIZE)) - 1;

    pub(crate) const BEST_MOVE_SIZE: i32 = 7;
    pub(crate) const BEST_MOVE_SHIFT: i32 = Self::SCORE_SHIFT + Self::SCORE_SIZE;
    pub(crate) const BEST_MOVE_MASK: u64 = (1 << (Self::BEST_MOVE_SIZE)) - 1;

    pub(crate) const BOUND_SIZE: i32 = 2;
    pub(crate) const BOUND_SHIFT: i32 = Self::BEST_MOVE_SHIFT + Self::BEST_MOVE_SIZE;
    pub(crate) const BOUND_MASK: u64 = (1 << (Self::BOUND_SIZE)) - 1;

    pub(crate) const DEPTH_SIZE: i32 = 6;
    pub(crate) const DEPTH_SHIFT: i32 = Self::BOUND_SHIFT + Self::BOUND_SIZE;
    pub(crate) const DEPTH_MASK: u64 = (1 << (Self::DEPTH_SIZE)) - 1;

    pub(crate) const SELECTIVITY_SIZE: i32 = 3;
    pub(crate) const SELECTIVITY_SHIFT: i32 = Self::DEPTH_SHIFT + Self::DEPTH_SIZE;
    pub(crate) const SELECTIVITY_MASK: u64 = (1 << (Self::SELECTIVITY_SIZE)) - 1;

    pub(crate) const GENERATION_SIZE: i32 = 7;
    pub(crate) const GENERATION_SHIFT: i32 = Self::SELECTIVITY_SHIFT + Self::SELECTIVITY_SIZE;
    pub(crate) const GENERATION_MASK: u64 = (1 << (Self::GENERATION_SIZE)) - 1;

    pub(crate) const IS_ENDGAME_SHIFT: i32 = Self::GENERATION_SHIFT + Self::GENERATION_SIZE;
    pub(crate) const IS_ENDGAME_MASK: u64 = 1;

    /// Packs all fields into a single 64-bit value and stores it.
    #[allow(clippy::too_many_arguments)]
    fn pack(
        &self,
        key: u32,
        score: Score,
        best_move: u8,
        bound: u8,
        depth: u8,
        selectivity: u8,
        generation: u8,
        is_endgame: bool,
    ) {
        let data = key as u64
            | (((score as u16) as u64) << Self::SCORE_SHIFT)
            | ((best_move as u64) << Self::BEST_MOVE_SHIFT)
            | ((bound as u64) << Self::BOUND_SHIFT)
            | ((depth as u64) << Self::DEPTH_SHIFT)
            | ((selectivity as u64) << Self::SELECTIVITY_SHIFT)
            | ((generation as u64) << Self::GENERATION_SHIFT)
            | ((is_endgame as u64) << Self::IS_ENDGAME_SHIFT);
        self.data.store(data, Ordering::Relaxed);
    }

    /// Unpacks a 64-bit value into TTEntryData.
    #[inline(always)]
    fn unpack_from_u64(data_u64: u64) -> TTEntryData {
        TTEntryData { raw: data_u64 }
    }

    /// Loads and unpacks the entry data atomically.
    #[inline(always)]
    pub fn unpack(&self) -> TTEntryData {
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
    /// * `is_endgame` - Whether this entry is from endgame search
    #[allow(clippy::too_many_arguments)]
    pub fn save(
        &self,
        key: u64,
        score: Score,
        bound: Bound,
        depth: Depth,
        best_move: Square,
        selectivity: Selectivity,
        generation: u8,
        is_endgame: bool,
    ) {
        let key22 = (key & Self::KEY_MASK) as u32;

        // Fast path: Exact bound always replaces
        if bound == Bound::Exact {
            self.pack(
                key22,
                score,
                best_move as u8,
                bound as u8,
                depth as u8,
                selectivity.as_u8(),
                generation,
                is_endgame,
            );
            return;
        }

        // Load raw data once
        let raw_data = self.data.load(Ordering::Relaxed);
        let stored_key = (raw_data & Self::KEY_MASK) as u32;

        // Different key: always replace (don't need to preserve best_move)
        if key22 != stored_key {
            self.pack(
                key22,
                score,
                best_move as u8,
                bound as u8,
                depth as u8,
                selectivity.as_u8(),
                generation,
                is_endgame,
            );
            return;
        }

        // Same key: check replacement conditions using bit operations
        let stored_depth = ((raw_data >> Self::DEPTH_SHIFT) & Self::DEPTH_MASK) as i8;
        let stored_selectivity =
            ((raw_data >> Self::SELECTIVITY_SHIFT) & Self::SELECTIVITY_MASK) as u8;
        let stored_generation =
            ((raw_data >> Self::GENERATION_SHIFT) & Self::GENERATION_MASK) as u8;

        let should_replace = (depth as i8) >= stored_depth.saturating_sub(2)
            || selectivity.as_u8() > stored_selectivity
            || generation != stored_generation;

        if should_replace {
            // Need to preserve best_move if new one is Square::None
            let bm = if best_move != Square::None {
                best_move as u8
            } else {
                ((raw_data >> Self::BEST_MOVE_SHIFT) & Self::BEST_MOVE_MASK) as u8
            };

            self.pack(
                key22,
                score,
                bm,
                bound as u8,
                depth as u8,
                selectivity.as_u8(),
                generation,
                is_endgame,
            );
        }
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
        let entries_size = cluster_count as usize * CLUSTER_SIZE;

        TranspositionTable {
            entries: AVec::from_iter(32, (0..entries_size).map(|_| TTEntry::default())),
            cluster_count,
        }
    }

    /// Clears all entries in the transposition table.
    pub fn clear(&self) {
        for entry in &*self.entries {
            entry.data.store(0, Ordering::Relaxed);
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

        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = key;
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
    /// Probes the transposition table for an entry matching the given key.
    ///
    /// # Arguments
    ///
    /// * `key` - The hash key of the board position to probe
    /// * `generation` - The current generation count for aging
    ///
    /// # Returns
    ///
    /// A `ProbeResult` indicating whether a matching entry was found.
    #[inline(always)]
    pub fn probe(&self, key: u64, generation: u8) -> TTProbeResult {
        cfg_if! {
            if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
                unsafe { self.probe_avx2(key, generation) }
            } else {
                self.probe_fallback(key, generation)
            }
        }
    }

    /// AVX2-optimized probe implementation for faster cluster scanning.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    #[inline]
    fn probe_avx2(&self, key: u64, generation: u8) -> TTProbeResult {
        use std::arch::x86_64::*;

        let key22 = (key & TTEntry::KEY_MASK) as u32;
        let base = self.get_cluster_idx(key);
        let ptr = unsafe { self.entries.as_ptr().add(base) } as *const __m256i;

        let v = unsafe { _mm256_load_si256(ptr) };
        let key_mask = _mm256_set1_epi64x(TTEntry::KEY_MASK as i64);
        let keys_in_entries = _mm256_and_si256(v, key_mask);
        let key_broadcast = _mm256_set1_epi64x(key22 as i64);
        let cmp = _mm256_cmpeq_epi32(keys_in_entries, key_broadcast);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        const KEY_MATCH_MASK: u32 = 0x0F0F0F0F;
        let key_matches = mask & KEY_MATCH_MASK;

        let bound_mask = _mm256_set1_epi64x(TTEntry::BOUND_MASK as i64);
        let bounds = _mm256_and_si256(_mm256_srli_epi64(v, TTEntry::BOUND_SHIFT), bound_mask);
        let zero = _mm256_setzero_si256();
        let occupied_cmp = _mm256_cmpeq_epi32(bounds, zero);
        let occupied_mask = !(_mm256_movemask_epi8(occupied_cmp) as u32) & KEY_MATCH_MASK;

        let hits = key_matches & occupied_mask;

        if hits != 0 {
            let tz = hits.trailing_zeros();
            let lane_idx = (tz / 8) as usize;

            let entries_ptr = ptr as *const u64;
            let raw_data = unsafe { *entries_ptr.add(lane_idx) };
            let tt_data = TTEntryData { raw: raw_data };

            return TTProbeResult::Hit {
                data: tt_data,
                index: base + lane_idx,
            };
        }

        let depth_mask = _mm256_set1_epi64x(TTEntry::DEPTH_MASK as i64);
        let depths = _mm256_and_si256(_mm256_srli_epi64(v, TTEntry::DEPTH_SHIFT), depth_mask);

        let gen_mask = _mm256_set1_epi64x(TTEntry::GENERATION_MASK as i64);
        let gens = _mm256_and_si256(_mm256_srli_epi64(v, TTEntry::GENERATION_SHIFT), gen_mask);

        let current_gen = _mm256_set1_epi64x(generation as i64);
        let ages = _mm256_sub_epi64(current_gen, gens);

        let scores = _mm256_sub_epi64(depths, _mm256_slli_epi64(ages, 3));
        let scores_arr: [i64; 4] = unsafe { std::mem::transmute(scores) };

        let s0 = scores_arr[0];
        let s1 = scores_arr[1];
        let s2 = scores_arr[2];
        let s3 = scores_arr[3];

        let min01 = if s0 <= s1 { (s0, 0usize) } else { (s1, 1) };
        let min23 = if s2 <= s3 { (s2, 2usize) } else { (s3, 3) };
        let (_, replace_idx) = if min01.0 <= min23.0 { min01 } else { min23 };

        TTProbeResult::Miss {
            index: base + replace_idx,
        }
    }

    /// Fallback probe implementation.
    #[allow(dead_code)]
    fn probe_fallback(&self, key: u64, generation: u8) -> TTProbeResult {
        let key22 = (key & TTEntry::KEY_MASK) as u32;
        let cluster_idx = self.get_cluster_idx(key);

        // look for exact key match
        for i in 0..CLUSTER_SIZE {
            let idx = cluster_idx + i;
            let entry = unsafe { self.entries.get_unchecked(idx) };
            let data = entry.data.load(Ordering::Relaxed);
            let entry_key = ((data >> TTEntry::KEY_SHIFT) & TTEntry::KEY_MASK) as u32;
            let bound_val = (data >> TTEntry::BOUND_SHIFT) & TTEntry::BOUND_MASK;
            let is_occupied = bound_val != 0;

            if entry_key == key22 && is_occupied {
                let tt_data = TTEntryData { raw: data };
                return TTProbeResult::Hit {
                    data: tt_data,
                    index: idx,
                };
            }
        }

        // find best replacement candidate
        // Score formula: depth - (age * 8)
        // Lower score = better replacement candidate
        let mut replace_idx = cluster_idx;
        let replace_data = unsafe { self.entries.get_unchecked(cluster_idx).unpack() };
        let mut replace_score =
            replace_data.depth() as i32 - replace_data.relative_age(generation) * 8;

        for i in 1..CLUSTER_SIZE {
            let idx = cluster_idx + i;
            let entry = unsafe { self.entries.get_unchecked(idx) };
            let tt_data = entry.unpack();
            let score = tt_data.depth() as i32 - tt_data.relative_age(generation) * 8;

            if score < replace_score {
                replace_score = score;
                replace_idx = idx;
            }
        }

        TTProbeResult::Miss { index: replace_idx }
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
    /// * `is_endgame` - Whether this entry is from endgame search
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
        selectivity: Selectivity,
        generation: u8,
        is_endgame: bool,
    ) {
        let entry = unsafe { self.entries.get_unchecked(entry_index) };
        entry.save(
            key,
            score,
            bound,
            depth,
            best_move,
            selectivity,
            generation,
            is_endgame,
        );
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

    fn sq(idx: usize) -> Square {
        Square::from_usize_unchecked(idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn store_entry(
        tt: &TranspositionTable,
        key: u64,
        score: Score,
        bound: Bound,
        depth: Depth,
        best_move: usize,
        selectivity: Selectivity,
        generation: u8,
    ) -> usize {
        let idx = tt.probe(key, generation).index();
        tt.store(
            idx,
            key,
            score,
            bound,
            depth,
            sq(best_move),
            selectivity,
            generation,
            false, // default to midgame for tests
        );
        idx
    }

    fn cluster_keys(tt: &TranspositionTable, target_cluster: usize, count: usize) -> Vec<u64> {
        let mut keys = Vec::with_capacity(count);
        let mut candidate: u64 = 0;

        while keys.len() < count {
            if tt.get_cluster_idx(candidate) == target_cluster {
                keys.push(candidate);
            }
            candidate = candidate.wrapping_add(1);
        }

        keys
    }

    /// Tests that TTEntry correctly packs and unpacks all fields.
    #[test]
    fn test_ttentry_store_and_read() {
        let entry = TTEntry::default();
        let test_key: u64 = 42;
        let test_score: Score = -64;
        let test_bound = Bound::Exact;
        let test_depth: Depth = 60;
        let test_best_move = sq(3);
        let test_selectivity = Selectivity::Level5;
        let test_generation: u8 = 127;

        entry.save(
            test_key,
            test_score,
            test_bound,
            test_depth,
            test_best_move,
            test_selectivity,
            test_generation,
            false,
        );

        let data = entry.unpack();
        assert_eq!(data.key(), (test_key & TTEntry::KEY_MASK) as u32);
        assert_eq!(data.score(), test_score);
        assert_eq!(data.bound(), test_bound);
        assert_eq!(data.depth(), test_depth);
        assert_eq!(data.best_move(), test_best_move);
        assert_eq!(data.selectivity(), test_selectivity);
        assert_eq!(data.generation(), test_generation);
    }

    /// Tests boundary values for packed fields.
    #[test]
    fn test_ttentry_boundary_values() {
        let entry = TTEntry::default();

        // Test maximum values for each field
        let max_key: u64 = 0x3FFFFF; // 22 bits
        let max_score: Score = 32767; // Max i16
        let min_score: Score = -32768; // Min i16
        let max_depth: Depth = 63; // 6 bits
        let max_best_move = sq(63); // 7 bits (0-63 squares)
        let max_generation: u8 = 127; // 7 bits

        // Test with maximum positive score
        entry.save(
            max_key,
            max_score,
            Bound::Lower,
            max_depth,
            max_best_move,
            Selectivity::None,
            max_generation,
            false,
        );

        let data = entry.unpack();
        assert_eq!(data.key(), (max_key & TTEntry::KEY_MASK) as u32);
        assert_eq!(data.score(), max_score);
        assert_eq!(data.bound(), Bound::Lower);
        assert_eq!(data.depth(), max_depth);
        assert_eq!(data.best_move(), max_best_move);
        assert_eq!(data.selectivity(), Selectivity::None);
        assert_eq!(data.generation(), max_generation);

        // Test with minimum negative score
        entry.save(
            0,
            min_score,
            Bound::Upper,
            0,
            Square::None,
            Selectivity::Level0,
            0,
            false,
        );

        let data = entry.unpack();
        assert_eq!(data.key(), 0);
        assert_eq!(data.score(), min_score);
        assert_eq!(data.bound(), Bound::Upper);
        assert_eq!(data.depth(), 0);
        assert_eq!(data.best_move(), Square::None);
        assert_eq!(data.selectivity(), Selectivity::Level0);
        assert_eq!(data.generation(), 0);
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
            sq(5),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.unpack();
        assert_eq!(data.depth(), 10);
        assert_eq!(data.generation(), 1);

        // Try to replace with slightly shallower depth (within 2 plies) - should replace
        entry.save(
            100,
            60,
            Bound::Lower,
            8,
            sq(6),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.unpack();
        assert_eq!(data.depth(), 8); // Replacement allowed within 2 plies
        assert_eq!(data.score(), 60); // New value should be stored

        // Replace with deeper depth - should replace
        entry.save(
            100,
            70,
            Bound::Lower,
            12,
            sq(7),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.unpack();
        assert_eq!(data.depth(), 12);
        assert_eq!(data.score(), 70);

        // Replace with exact bound - should always replace
        entry.save(
            100,
            80,
            Bound::Exact,
            5,
            sq(8),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.unpack();
        assert_eq!(data.depth(), 5);
        assert_eq!(data.score(), 80);
        assert_eq!(data.bound(), Bound::Exact);

        // Different key - should replace
        entry.save(
            200,
            90,
            Bound::Upper,
            3,
            sq(9),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.unpack();
        assert_eq!(data.key(), 200);
        assert_eq!(data.score(), 90);

        // Newer generation - should replace
        entry.save(
            200,
            100,
            Bound::Lower,
            2,
            sq(10),
            Selectivity::Level2,
            5,
            false,
        );
        let data = entry.unpack();
        assert_eq!(data.generation(), 5);
        assert_eq!(data.score(), 100);
    }

    /// Tests Bound::determine_bound for different node types.
    #[test]
    fn test_bound_determine() {
        // Test PV node
        assert_eq!(Bound::determine_bound::<PV>(100, 30, 50), Bound::Lower); // score >= beta
        assert_eq!(Bound::determine_bound::<PV>(40, 30, 50), Bound::Exact); // alpha < score < beta in PV
        assert_eq!(Bound::determine_bound::<PV>(20, 30, 50), Bound::Upper); // score <= alpha in PV

        // Test non-PV node
        assert_eq!(Bound::determine_bound::<NonPV>(100, 30, 50), Bound::Lower); // score >= beta
        assert_eq!(Bound::determine_bound::<NonPV>(40, 30, 50), Bound::Upper); // score < beta in non-PV
        assert_eq!(Bound::determine_bound::<NonPV>(20, 30, 50), Bound::Upper); // score <= alpha in non-PV
    }

    /// Tests TTEntryData methods.
    #[test]
    fn test_ttentry_data_methods() {
        // Construct a TTEntry to test TTEntryData via unpack
        let entry = TTEntry::default();

        // Test is_occupied
        assert!(!entry.unpack().is_occupied());

        entry.save(
            0,
            0,
            Bound::Lower,
            0,
            Square::None,
            Selectivity::Level0,
            0,
            false,
        );
        assert!(entry.unpack().is_occupied());

        entry.save(
            0,
            0,
            Bound::Upper,
            0,
            Square::None,
            Selectivity::Level0,
            0,
            false,
        );
        assert!(entry.unpack().is_occupied());

        entry.save(
            0,
            0,
            Bound::Exact,
            0,
            Square::None,
            Selectivity::Level0,
            0,
            false,
        );
        assert!(entry.unpack().is_occupied());

        // can_cut and relative_age are tested implicitly or can be tested via helper if necessary
        // Recreating scenarios for can_cut

        // Test can_cut
        entry.save(
            0,
            100,
            Bound::Lower,
            0,
            Square::None,
            Selectivity::Level0,
            0,
            false,
        );
        assert!(entry.unpack().can_cut(50)); // score >= beta, lower bound
        assert!(!entry.unpack().can_cut(150)); // score < beta, lower bound

        entry.save(
            0,
            30,
            Bound::Upper,
            0,
            Square::None,
            Selectivity::Level0,
            0,
            false,
        );
        assert!(entry.unpack().can_cut(50)); // score < beta, upper bound
        assert!(!entry.unpack().can_cut(20)); // score >= beta, upper bound

        entry.save(
            0,
            30,
            Bound::Exact,
            0,
            Square::None,
            Selectivity::Level0,
            0,
            false,
        );
        assert!(entry.unpack().can_cut(50)); // exact bound matches both
        assert!(entry.unpack().can_cut(20));

        // Test relative_age
        entry.save(
            0,
            0,
            Bound::Exact,
            0,
            Square::None,
            Selectivity::Level0,
            5,
            false,
        );
        assert_eq!(entry.unpack().relative_age(8), 3);
        assert_eq!(entry.unpack().relative_age(5), 0);
        assert_eq!(entry.unpack().relative_age(3), -2);
    }

    /// Tests TranspositionTable creation with different sizes.
    #[test]
    fn test_transposition_table_new() {
        // Test with 0 MB (minimum size)
        let tt = TranspositionTable::new(0);
        assert_eq!(tt.cluster_count, 16);
        assert_eq!(tt.entries.len(), 16 * CLUSTER_SIZE);

        // Test with 1 MB
        let tt = TranspositionTable::new(1);
        let expected_clusters = (1024 * 1024) / (std::mem::size_of::<TTEntry>() * CLUSTER_SIZE);
        assert_eq!(tt.cluster_count, expected_clusters as u64);
        assert_eq!(tt.entries.len(), expected_clusters * CLUSTER_SIZE);
    }

    /// Tests basic probe and store operations.
    #[test]
    fn test_probe_and_store() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;
        let generation = 1;

        // First probe should return not found
        assert!(!tt.probe(key, generation).is_hit());

        // Store an entry
        let idx = tt.probe(key, generation).index();
        tt.store(
            idx,
            key,
            100,
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level3,
            generation,
            false,
        );

        // Second probe should find the entry
        let result = tt.probe(key, generation);
        assert!(result.is_hit());
        let d = result.data().unwrap();
        assert_eq!(d.key(), (key & TTEntry::KEY_MASK) as u32);
        assert_eq!(d.score(), 100);
        assert_eq!(d.bound(), Bound::Exact);
        assert_eq!(d.depth(), 20);
        assert_eq!(d.best_move(), sq(10));
        assert_eq!(d.selectivity(), Selectivity::Level3);
        assert_eq!(d.generation(), generation);
    }

    /// Tests collision handling and replacement strategies.
    #[test]
    fn test_cluster_replacement() {
        let tt = TranspositionTable::new(1);
        let target_cluster = tt.get_cluster_idx(0);

        // Generate keys that map to the same cluster
        let keys = cluster_keys(&tt, target_cluster, CLUSTER_SIZE + 1);

        // Fill the cluster
        for (i, key) in keys.iter().take(CLUSTER_SIZE).enumerate() {
            let depth: Depth = i as Depth * 10;
            tt.store(
                target_cluster + i,
                *key,
                i as Score * 10,
                Bound::Lower,
                depth,
                sq(i),
                Selectivity::Level1,
                1,
                false,
            );
        }

        let new_key = keys[CLUSTER_SIZE];
        let result = tt.probe(new_key, 1);
        assert!(!result.is_hit());
        let replace_idx = result.index();
        assert_eq!(replace_idx, target_cluster);

        tt.store(
            replace_idx,
            new_key,
            999,
            Bound::Exact,
            50,
            sq(20),
            Selectivity::Level1,
            1,
            false,
        );

        let result = tt.probe(new_key, 1);
        assert!(result.is_hit());
        assert_eq!(result.data().unwrap().score(), 999);
    }

    /// Tests generation-based aging in replacement.
    #[test]
    fn test_generation_aging() {
        let tt = TranspositionTable::new(1);
        let target_cluster = tt.get_cluster_idx(0);
        let keys = cluster_keys(&tt, target_cluster, CLUSTER_SIZE + 1);

        for (i, key) in keys.iter().take(CLUSTER_SIZE).enumerate() {
            let depth: Depth = 30u32 - i as Depth;
            tt.store(
                target_cluster + i,
                *key,
                100 + i as Score,
                Bound::Lower,
                depth,
                sq(i),
                Selectivity::Level1,
                1,
                false,
            );
        }

        let new_key = keys[CLUSTER_SIZE];
        let result = tt.probe(new_key, 10);
        assert!(!result.is_hit());
        let replace_idx = result.index();
        assert_eq!(replace_idx, target_cluster + (CLUSTER_SIZE - 1));

        tt.store(
            replace_idx,
            new_key,
            200,
            Bound::Exact,
            10,
            sq(20),
            Selectivity::Level1,
            10,
            false,
        );

        let result = tt.probe(new_key, 10);
        assert!(result.is_hit());
        assert_eq!(result.data().unwrap().generation(), 10);
    }

    /// Tests the clear function.
    #[test]
    fn test_clear() {
        let tt = TranspositionTable::new(1);

        // Store some entries
        for i in 0..10 {
            let key = i * 0x1000000000000000;
            store_entry(
                &tt,
                key,
                i as Score * 10,
                Bound::Exact,
                20,
                i as usize,
                Selectivity::Level1,
                1,
            );
        }

        // Verify entries exist
        assert!(tt.probe(0, 1).is_hit());

        // Clear the table
        tt.clear();

        // Verify entries are gone
        for i in 0..10 {
            let key = i * 0x1000000000000000;
            assert!(!tt.probe(key, 1).is_hit());
        }
    }
}
