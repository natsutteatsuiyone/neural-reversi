//! Lightweight 2-way transposition table for single-threaded WASM search.
//!
//! Each cluster contains two 16-byte entries. On probe miss, the replacement
//! candidate is chosen by preferring empty slots, then the entry with the
//! lowest replacement score (`depth - AGE_WEIGHT * relative_age`).

use std::cell::Cell;

use reversi_core::{
    probcut::Selectivity,
    search::node_type::NodeType,
    square::Square,
    types::{Depth, ScaledScore, Score},
};

/// Number of entries per cluster.
const CLUSTER_SIZE: usize = 2;

// ── Bound ────────────────────────────────────────────────────────────

/// Bound type stored in a transposition-table entry.
///
/// Bit patterns allow efficient cutoff checks via bitwise AND:
/// - [`Bound::Lower`] (01) and [`Bound::Exact`] (11) share bit 0.
/// - [`Bound::Upper`] (10) and [`Bound::Exact`] (11) share bit 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    #[default]
    None = 0,
    Lower = 1,
    Upper = 2,
    Exact = 3,
}

impl Bound {
    #[inline(always)]
    pub fn classify_scaled<NT: NodeType>(
        best_score: ScaledScore,
        alpha: ScaledScore,
        beta: ScaledScore,
    ) -> Bound {
        Self::classify_inner::<NT>(best_score.value(), alpha.value(), beta.value())
    }

    #[inline(always)]
    pub fn classify_score<NT: NodeType>(best_score: Score, alpha: Score, beta: Score) -> Bound {
        Self::classify_inner::<NT>(best_score, alpha, beta)
    }

    #[inline(always)]
    fn classify_inner<NT: NodeType>(best_score: i32, alpha: i32, beta: i32) -> Bound {
        if best_score >= beta {
            return Bound::Lower;
        }
        if NT::PV_NODE && best_score > alpha {
            return Bound::Exact;
        }
        Bound::Upper
    }
}

// ── TTData / TTEntryData ─────────────────────────────────────────────

/// Compact data stored inside each [`TTEntry`] slot (8 bytes).
///
/// All enum fields are `#[repr(u8)]`, so the struct fits in 8 bytes
/// and can be wrapped in a single [`Cell`].
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct TTData {
    score: i16,
    best_move: Square,
    bound: Bound,
    depth: u8,
    selectivity: Selectivity,
    is_endgame: bool,
    generation: u8,
}

const _: () = assert!(std::mem::size_of::<TTData>() == 8);

/// Metadata extracted from a transposition-table entry, with
/// ergonomic field types.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTEntryData {
    pub score: ScaledScore,
    pub best_move: Square,
    pub bound: Bound,
    pub depth: Depth,
    pub selectivity: Selectivity,
    pub is_endgame: bool,
}

impl TTEntryData {
    /// Returns whether this entry allows an immediate cutoff in a
    /// null-window search with target `beta`.
    #[inline(always)]
    pub fn can_cut(&self, beta: ScaledScore) -> bool {
        let required = if self.score >= beta {
            Bound::Lower as u8
        } else {
            Bound::Upper as u8
        };
        (self.bound as u8 & required) != 0
    }
}

impl From<TTData> for TTEntryData {
    #[inline(always)]
    fn from(d: TTData) -> Self {
        TTEntryData {
            score: ScaledScore::from_raw(d.score as i32),
            best_move: d.best_move,
            bound: d.bound,
            depth: d.depth as Depth,
            selectivity: d.selectivity,
            is_endgame: d.is_endgame,
        }
    }
}

// ── TTProbeResult ────────────────────────────────────────────────────

/// Result of probing the transposition table.
pub enum TTProbeResult {
    Hit { data: TTEntryData, index: usize },
    Miss { index: usize },
}

impl TTProbeResult {
    #[inline(always)]
    pub fn index(&self) -> usize {
        match self {
            TTProbeResult::Hit { index, .. } | TTProbeResult::Miss { index } => *index,
        }
    }

    #[inline(always)]
    pub fn data(&self) -> Option<TTEntryData> {
        match self {
            TTProbeResult::Hit { data, .. } => Some(*data),
            TTProbeResult::Miss { .. } => None,
        }
    }

    #[inline(always)]
    pub fn best_move(&self) -> Square {
        match self {
            TTProbeResult::Hit { data, .. } => data.best_move,
            TTProbeResult::Miss { .. } => Square::None,
        }
    }
}

// ── TTEntry ──────────────────────────────────────────────────────────

/// A single 16-byte transposition-table slot.
///
/// `key` (8 bytes) + `data` (8 bytes) = 16 bytes.  Both are wrapped in
/// [`Cell`] for interior mutability; `data` is read/written as a whole
/// [`TTData`] struct in one `get()`/`set()` call.
#[derive(Clone)]
#[repr(C)]
struct TTEntry {
    key: Cell<u64>,
    data: Cell<TTData>,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: Cell::new(0),
            data: Cell::new(TTData::default()),
        }
    }
}

// ── TranspositionTable ───────────────────────────────────────────────

/// Weight applied to the generation age difference in the replacement score.
const AGE_WEIGHT: i32 = 8;

/// Lightweight 2-way transposition table for single-threaded WASM search.
pub struct TranspositionTable {
    table: Vec<TTEntry>,
    /// Bitmask for cluster index: `cluster_count - 1`.
    mask: usize,
    /// Generation counter for entry aging (incremented each search).
    generation: Cell<u8>,
}

impl TranspositionTable {
    /// Creates a new table with the given size in MiB.
    pub fn new(mb_size: usize) -> Self {
        let cluster_count = if mb_size == 0 {
            16
        } else {
            let bytes = mb_size * 1024 * 1024;
            let total_entries = bytes / std::mem::size_of::<TTEntry>();
            let clusters = total_entries / CLUSTER_SIZE;
            1 << clusters.ilog2()
        };
        let entry_count = cluster_count * CLUSTER_SIZE;

        TranspositionTable {
            table: vec![TTEntry::default(); entry_count],
            mask: cluster_count - 1,
            generation: Cell::new(0),
        }
    }

    /// Increments the generation counter (wraps around u8).
    #[inline]
    pub fn increment_generation(&self) {
        self.generation.set(self.generation.get().wrapping_add(1));
    }

    /// Returns the entry index of the first slot in the cluster for `key`.
    #[inline(always)]
    fn cluster_index(&self, key: u64) -> usize {
        (((key >> 32) as usize) & self.mask) * CLUSTER_SIZE
    }

    /// Probes the cluster for a matching entry.
    ///
    /// On miss, returns the index of the best replacement candidate:
    /// an empty slot if available, otherwise the entry with the lowest
    /// replacement score (`depth - AGE_WEIGHT * relative_age`).
    #[inline(always)]
    pub fn probe(&self, key: u64) -> TTProbeResult {
        let base = self.cluster_index(key);
        let generation = self.generation.get();

        let mut replace_idx = base;
        let mut replace_score = i32::MAX;

        for i in 0..CLUSTER_SIZE {
            let idx = base + i;
            // SAFETY: cluster_index() returns a value in 0..cluster_count * CLUSTER_SIZE,
            // and i < CLUSTER_SIZE, so idx < table.len().
            let entry = unsafe { self.table.get_unchecked(idx) };

            let data = entry.data.get();
            if entry.key.get() == key && data.bound != Bound::None {
                return TTProbeResult::Hit {
                    data: TTEntryData::from(data),
                    index: idx,
                };
            }

            if data.bound == Bound::None {
                return TTProbeResult::Miss { index: idx };
            }

            let age = generation.wrapping_sub(data.generation) as i32;
            let score = data.depth as i32 - AGE_WEIGHT * age;
            if score < replace_score {
                replace_score = score;
                replace_idx = idx;
            }
        }

        TTProbeResult::Miss { index: replace_idx }
    }

    /// Read-only lookup without replacement candidate.
    #[inline(always)]
    pub fn lookup(&self, key: u64) -> Option<TTEntryData> {
        let base = self.cluster_index(key);

        for i in 0..CLUSTER_SIZE {
            // SAFETY: base + i < table.len() (see probe()).
            let entry = unsafe { self.table.get_unchecked(base + i) };

            if entry.key.get() == key {
                let data = entry.data.get();
                if data.bound != Bound::None {
                    return Some(TTEntryData::from(data));
                }
            }
        }
        None
    }

    /// Stores a result if it wins the replacement policy.
    ///
    /// Exact bounds always replace. A different key also replaces, since the
    /// caller already chose this slot as the cluster victim. For the same key,
    /// non-exact results replace when depth regression is at most two plies,
    /// selectivity is higher, or generation changed. If `best_move` is
    /// [`Square::None`], the existing move is preserved for same-key non-exact
    /// updates.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn store(
        &self,
        index: usize,
        key: u64,
        score: ScaledScore,
        bound: Bound,
        depth: Depth,
        best_move: Square,
        selectivity: Selectivity,
        is_endgame: bool,
    ) {
        let generation = self.generation.get();
        // SAFETY: index is produced by probe(), which guarantees index < table.len().
        let entry = unsafe { self.table.get_unchecked(index) };

        let same_key = entry.key.get() == key;
        let existing = entry.data.get();

        let should_replace = if bound == Bound::Exact || !same_key {
            true
        } else {
            (depth as i8) >= (existing.depth as i8).saturating_sub(2)
                || (selectivity as u8) > existing.selectivity as u8
                || generation != existing.generation
        };

        if !should_replace {
            return;
        }

        let write_best_move = if best_move != Square::None || bound == Bound::Exact || !same_key {
            best_move
        } else {
            existing.best_move
        };

        entry.key.set(key);
        entry.data.set(TTData {
            score: score.value() as i16,
            best_move: write_best_move,
            bound,
            depth: depth as u8,
            selectivity,
            is_endgame,
            generation,
        });
    }

    /// Clears all entries.
    pub fn clear(&self) {
        // SAFETY: TTEntry is #[repr(C)] and all-zero bytes represent the correct
        // default state (key=0, TTData fields all zero, bound=0 = Bound::None).
        // No concurrent access on single-threaded WASM.
        unsafe {
            std::ptr::write_bytes(
                self.table.as_ptr() as *mut u8,
                0,
                self.table.len() * std::mem::size_of::<TTEntry>(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_size() {
        assert_eq!(std::mem::size_of::<TTEntry>(), 16);
        assert_eq!(std::mem::size_of::<TTData>(), 8);
    }

    #[test]
    fn test_new_table_size() {
        // 1 MiB / 16 bytes = 65536 entries → 32768 clusters
        let tt = TranspositionTable::new(1);
        assert_eq!(tt.table.len(), 65536);
        assert_eq!(tt.mask, 32767);
    }

    #[test]
    fn test_new_zero_size() {
        let tt = TranspositionTable::new(0);
        assert_eq!(tt.table.len(), 16 * CLUSTER_SIZE);
    }

    #[test]
    fn test_store_and_probe_hit() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;
        let score = ScaledScore::from_raw(42);

        let probe = tt.probe(key);
        assert!(matches!(probe, TTProbeResult::Miss { .. }));

        tt.store(
            probe.index(),
            key,
            score,
            Bound::Exact,
            10,
            Square::D3,
            Selectivity::None,
            false,
        );

        let probe = tt.probe(key);
        assert!(matches!(probe, TTProbeResult::Hit { .. }));
        let data = probe.data().unwrap();
        assert_eq!(data.score, score);
        assert_eq!(data.best_move, Square::D3);
        assert_eq!(data.bound, Bound::Exact);
        assert_eq!(data.depth, 10);
        assert!(!data.is_endgame);
    }

    #[test]
    fn test_store_and_lookup() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0xDEADBEEFCAFEBABE;
        let score = ScaledScore::from_raw(-100);

        assert!(tt.lookup(key).is_none());

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            score,
            Bound::Lower,
            5,
            Square::E6,
            Selectivity::Level1,
            true,
        );

        let data = tt.lookup(key).unwrap();
        assert_eq!(data.score, score);
        assert_eq!(data.bound, Bound::Lower);
        assert!(data.is_endgame);
    }

    #[test]
    fn test_probe_miss_on_different_key() {
        let tt = TranspositionTable::new(1);
        let key1: u64 = 0x1111111111111111;
        let key2: u64 = 0x2222222222222222;

        let idx = tt.probe(key1).index();
        tt.store(
            idx,
            key1,
            ScaledScore::from_raw(10),
            Bound::Upper,
            3,
            Square::None,
            Selectivity::None,
            false,
        );

        // key2 may or may not map to the same cluster, but should not hit key1
        let probe = tt.probe(key2);
        if let TTProbeResult::Hit { data, .. } = probe {
            // If it's a hit, it must not be key1's data (different key)
            panic!(
                "Should not hit with different key, got depth={}",
                data.depth
            );
        }
    }

    #[test]
    fn test_two_entries_in_same_cluster() {
        let tt = TranspositionTable::new(1);
        // Two keys that map to the same cluster (same upper 32 bits masked)
        let base_key: u64 = 0x1234567800000001;
        let key2: u64 = 0x1234567800000002;

        let idx1 = tt.probe(base_key).index();
        tt.store(
            idx1,
            base_key,
            ScaledScore::from_raw(10),
            Bound::Exact,
            5,
            Square::A1,
            Selectivity::None,
            false,
        );

        let idx2 = tt.probe(key2).index();
        tt.store(
            idx2,
            key2,
            ScaledScore::from_raw(20),
            Bound::Lower,
            8,
            Square::B2,
            Selectivity::None,
            false,
        );

        // Both should be retrievable
        let data1 = tt.probe(base_key).data().unwrap();
        assert_eq!(data1.score, ScaledScore::from_raw(10));
        assert_eq!(data1.best_move, Square::A1);

        let data2 = tt.probe(key2).data().unwrap();
        assert_eq!(data2.score, ScaledScore::from_raw(20));
        assert_eq!(data2.best_move, Square::B2);
    }

    #[test]
    fn test_replacement_evicts_shallowest() {
        let tt = TranspositionTable::new(1);
        // Fill both cluster slots
        let key1: u64 = 0x1234567800000001;
        let key2: u64 = 0x1234567800000002;
        let key3: u64 = 0x1234567800000003;

        let idx1 = tt.probe(key1).index();
        tt.store(
            idx1,
            key1,
            ScaledScore::from_raw(10),
            Bound::Exact,
            3,
            Square::A1,
            Selectivity::None,
            false,
        );

        let idx2 = tt.probe(key2).index();
        tt.store(
            idx2,
            key2,
            ScaledScore::from_raw(20),
            Bound::Exact,
            10,
            Square::B2,
            Selectivity::None,
            false,
        );

        // Third key should evict key1 (depth 3) not key2 (depth 10)
        let idx3 = tt.probe(key3).index();
        tt.store(
            idx3,
            key3,
            ScaledScore::from_raw(30),
            Bound::Exact,
            7,
            Square::C3,
            Selectivity::None,
            false,
        );

        // key1 should be gone, key2 and key3 should remain
        assert!(tt.lookup(key1).is_none());
        assert!(tt.lookup(key2).is_some());
        let data3 = tt.lookup(key3).unwrap();
        assert_eq!(data3.score, ScaledScore::from_raw(30));
    }

    #[test]
    fn test_replacement_preserves_deep_same_key() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(100),
            Bound::Lower,
            15,
            Square::D3,
            Selectivity::None,
            false,
        );

        // Shallow non-exact update, same generation, same selectivity → rejected
        // (depth 5 < 15 - 2 = 13, selectivity not higher, same generation)
        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(50),
            Bound::Upper,
            5,
            Square::E4,
            Selectivity::None,
            false,
        );

        let data = tt.lookup(key).unwrap();
        assert_eq!(data.score, ScaledScore::from_raw(100));
        assert_eq!(data.depth, 15);
    }

    #[test]
    fn test_same_key_replaces_when_depth_close() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(100),
            Bound::Lower,
            10,
            Square::D3,
            Selectivity::None,
            false,
        );

        // depth 9 >= 10 - 2 = 8 → replaces (within 2-ply regression)
        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(50),
            Bound::Upper,
            9,
            Square::E4,
            Selectivity::None,
            false,
        );

        let data = tt.lookup(key).unwrap();
        assert_eq!(data.score, ScaledScore::from_raw(50));
        assert_eq!(data.depth, 9);
    }

    #[test]
    fn test_exact_bound_always_replaces() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(100),
            Bound::Lower,
            15,
            Square::D3,
            Selectivity::None,
            false,
        );

        // Exact bound should replace even at lower depth
        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(50),
            Bound::Exact,
            5,
            Square::E4,
            Selectivity::None,
            false,
        );

        let data = tt.lookup(key).unwrap();
        assert_eq!(data.score, ScaledScore::from_raw(50));
        assert_eq!(data.bound, Bound::Exact);
    }

    #[test]
    fn test_preserves_best_move_on_same_key_none_move() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(100),
            Bound::Lower,
            10,
            Square::D3,
            Selectivity::None,
            false,
        );

        // Same key, non-exact, best_move=None → preserves existing D3
        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(80),
            Bound::Upper,
            10,
            Square::None,
            Selectivity::None,
            false,
        );

        let data = tt.lookup(key).unwrap();
        assert_eq!(data.best_move, Square::D3);
        assert_eq!(data.score, ScaledScore::from_raw(80));
    }

    #[test]
    fn test_clear() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0xABCDABCDABCDABCD;

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(50),
            Bound::Exact,
            8,
            Square::A1,
            Selectivity::None,
            false,
        );
        assert!(tt.lookup(key).is_some());

        tt.clear();
        assert!(tt.lookup(key).is_none());
    }

    #[test]
    fn test_can_cut() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x5555555555555555;

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(100),
            Bound::Lower,
            10,
            Square::None,
            Selectivity::None,
            false,
        );

        let data = tt.probe(key).data().unwrap();
        assert!(data.can_cut(ScaledScore::from_raw(50)));
        assert!(!data.can_cut(ScaledScore::from_raw(200)));
    }

    #[test]
    fn test_age_based_eviction() {
        let tt = TranspositionTable::new(1);
        let key1: u64 = 0x1234567800000001;
        let key2: u64 = 0x1234567800000002;
        let key3: u64 = 0x1234567800000003;

        // Store key1 at generation 0, depth 10
        let idx1 = tt.probe(key1).index();
        tt.store(
            idx1,
            key1,
            ScaledScore::from_raw(10),
            Bound::Lower,
            10,
            Square::A1,
            Selectivity::None,
            false,
        );

        // Store key2 at generation 0, depth 3
        let idx2 = tt.probe(key2).index();
        tt.store(
            idx2,
            key2,
            ScaledScore::from_raw(20),
            Bound::Lower,
            3,
            Square::B2,
            Selectivity::None,
            false,
        );

        // Advance generation by 2
        tt.increment_generation();
        tt.increment_generation();

        // key3 should evict key2 (depth 3, age 2 → score = 3 - 8*2 = -13)
        // over key1 (depth 10, age 2 → score = 10 - 8*2 = -6)
        let idx3 = tt.probe(key3).index();
        tt.store(
            idx3,
            key3,
            ScaledScore::from_raw(30),
            Bound::Exact,
            5,
            Square::C3,
            Selectivity::None,
            false,
        );

        assert!(tt.lookup(key1).is_some());
        assert!(tt.lookup(key2).is_none());
        assert!(tt.lookup(key3).is_some());
    }

    #[test]
    fn test_old_generation_replaced_over_deep() {
        let tt = TranspositionTable::new(1);
        let key1: u64 = 0x1234567800000001;
        let key2: u64 = 0x1234567800000002;
        let key3: u64 = 0x1234567800000003;

        // Store key1 at generation 0, depth 20 (deep but old)
        let idx1 = tt.probe(key1).index();
        tt.store(
            idx1,
            key1,
            ScaledScore::from_raw(10),
            Bound::Lower,
            20,
            Square::A1,
            Selectivity::None,
            false,
        );

        // Advance generation by 3
        for _ in 0..3 {
            tt.increment_generation();
        }

        // Store key2 at generation 3, depth 5 (shallow but current)
        let idx2 = tt.probe(key2).index();
        tt.store(
            idx2,
            key2,
            ScaledScore::from_raw(20),
            Bound::Lower,
            5,
            Square::B2,
            Selectivity::None,
            false,
        );

        // key1: score = 20 - 8*3 = -4, key2: score = 5 - 8*0 = 5
        // key3 should evict key1 (lower replacement score)
        let idx3 = tt.probe(key3).index();
        tt.store(
            idx3,
            key3,
            ScaledScore::from_raw(30),
            Bound::Exact,
            1,
            Square::C3,
            Selectivity::None,
            false,
        );

        assert!(tt.lookup(key1).is_none());
        assert!(tt.lookup(key2).is_some());
        assert!(tt.lookup(key3).is_some());
    }

    #[test]
    fn test_different_generation_allows_same_key_overwrite() {
        let tt = TranspositionTable::new(1);
        let key: u64 = 0x123456789ABCDEF0;

        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(100),
            Bound::Lower,
            15,
            Square::D3,
            Selectivity::None,
            false,
        );

        // Advance generation
        tt.increment_generation();

        // Shallow non-exact update for same key should now succeed (different generation)
        let idx = tt.probe(key).index();
        tt.store(
            idx,
            key,
            ScaledScore::from_raw(50),
            Bound::Upper,
            5,
            Square::E4,
            Selectivity::None,
            false,
        );

        let data = tt.lookup(key).unwrap();
        assert_eq!(data.score, ScaledScore::from_raw(50));
        assert_eq!(data.depth, 5);
    }

    #[test]
    fn test_bound_classify() {
        use reversi_core::search::node_type::{NonPV, PV};

        assert_eq!(
            Bound::classify_scaled::<PV>(
                ScaledScore::from_raw(100),
                ScaledScore::from_raw(0),
                ScaledScore::from_raw(50)
            ),
            Bound::Lower
        );
        assert_eq!(
            Bound::classify_scaled::<PV>(
                ScaledScore::from_raw(30),
                ScaledScore::from_raw(0),
                ScaledScore::from_raw(50)
            ),
            Bound::Exact
        );
        assert_eq!(
            Bound::classify_scaled::<NonPV>(
                ScaledScore::from_raw(30),
                ScaledScore::from_raw(0),
                ScaledScore::from_raw(50)
            ),
            Bound::Upper
        );
        assert_eq!(
            Bound::classify_scaled::<PV>(
                ScaledScore::from_raw(-10),
                ScaledScore::from_raw(0),
                ScaledScore::from_raw(50)
            ),
            Bound::Upper
        );
    }
}
