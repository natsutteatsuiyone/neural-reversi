//! Transposition table for caching search results.
//!
//! Each entry stores the full board plus packed search metadata, so hash-index
//! collisions only affect replacement and never produce a false hit.

use crate::board::Board;
use crate::constants::CACHE_LINE_SIZE;
use crate::probcut::Selectivity;
use crate::search::node_type::NodeType;
use crate::square::Square;
use crate::types::{Depth, ScaledScore};
use aligned_vec::{AVec, ConstAlign};
use std::{
    hint::{Locality, prefetch_read},
    mem,
    sync::atomic::{AtomicU8, AtomicU64, Ordering, fence},
};

/// Size of each cluster in the transposition table.
const CLUSTER_SIZE: usize = 2;

/// Represents the bound type of a transposition table entry.
///
/// The variants are assigned specific bit patterns to allow efficient cutoff
/// checks using bitwise AND:
/// - [`Bound::Lower`] (1/01) and [`Bound::Exact`] (3/11) both have bit 0 set.
/// - [`Bound::Upper`] (2/10) and [`Bound::Exact`] (3/11) both have bit 1 set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    /// No valid entry.
    None = 0,
    /// Score is a lower bound (fail-high).
    Lower = 1,
    /// Score is an upper bound (fail-low).
    Upper = 2,
    /// Score is the exact minimax value.
    Exact = 3,
}

impl Bound {
    /// Classifies a midgame search result into a [`Bound`] type.
    ///
    /// - [`Bound::Lower`] if `best_score >= beta` (fail-high).
    /// - [`Bound::Exact`] if `alpha < best_score < beta` and node is PV.
    /// - [`Bound::Upper`] otherwise (fail-low or non-PV exact-ish value).
    #[inline(always)]
    pub fn classify<NT: NodeType>(
        best_score: ScaledScore,
        alpha: ScaledScore,
        beta: ScaledScore,
    ) -> Bound {
        if best_score >= beta {
            return Bound::Lower;
        }

        if NT::PV_NODE && best_score > alpha {
            return Bound::Exact;
        }

        Bound::Upper
    }

    /// Converts a `u8` value to a [`Bound`].
    ///
    /// # Safety
    ///
    /// `value` must be in `0..=3`.
    #[inline]
    pub(crate) unsafe fn from_u8_unchecked(value: u8) -> Bound {
        debug_assert!(
            value <= 3,
            "Bound::from_u8_unchecked called with out-of-range value: {value}"
        );
        // SAFETY: Bound is #[repr(u8)] with contiguous variants 0–3.
        unsafe { mem::transmute(value) }
    }
}

/// Represents the result of probing a transposition table cluster.
pub enum TTProbeResult {
    /// Found an entry whose stored board matches the probe.
    Hit {
        /// The data of the entry.
        data: TTEntryData,
        /// The index of the entry in the table.
        index: usize,
    },
    /// No matching board was found.
    Miss {
        /// Replacement candidate for a potential store.
        index: usize,
    },
}

impl TTProbeResult {
    /// Returns the slot index for a subsequent store.
    #[inline(always)]
    pub fn index(&self) -> usize {
        match self {
            TTProbeResult::Hit { index, .. } | TTProbeResult::Miss { index } => *index,
        }
    }

    /// Returns the cached data if hit, otherwise [`None`].
    #[inline(always)]
    pub fn data(&self) -> Option<TTEntryData> {
        match self {
            TTProbeResult::Hit { data, .. } => Some(*data),
            TTProbeResult::Miss { .. } => None,
        }
    }

    /// Returns the best move if hit, otherwise [`Square::None`].
    #[inline(always)]
    pub fn best_move(&self) -> Square {
        match self {
            TTProbeResult::Hit { data, .. } => data.best_move(),
            TTProbeResult::Miss { .. } => Square::None,
        }
    }

    /// Returns `true` if the probe was a hit.
    #[inline(always)]
    pub fn is_hit(&self) -> bool {
        matches!(self, TTProbeResult::Hit { .. })
    }
}

/// Byte-aligned fields packed into 64 bits for atomic load/store.
///
/// Each field occupies a full byte (or two for `score`), eliminating
/// bit-shift/mask operations on the hot read path.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TTDataFields {
    score: i16,
    best_move: u8,
    bound: u8,
    depth: u8,
    selectivity: u8,
    generation: u8,
    is_endgame: u8,
}

const _: () = assert!(mem::size_of::<TTDataFields>() == 8);

impl TTDataFields {
    #[inline(always)]
    fn to_u64(self) -> u64 {
        // SAFETY: TTDataFields is #[repr(C)], 8 bytes, no padding.
        unsafe { mem::transmute(self) }
    }

    #[inline(always)]
    fn from_u64(raw: u64) -> Self {
        // SAFETY: TTDataFields is #[repr(C)] with no padding (static-asserted
        // to be exactly 8 bytes), so every u64 bit pattern maps to a valid
        // instance whose individual integer fields accept all bit patterns.
        unsafe { mem::transmute(raw) }
    }

    #[inline(always)]
    fn new(
        score: ScaledScore,
        best_move: Square,
        bound: Bound,
        depth: Depth,
        selectivity: Selectivity,
        generation: u8,
        is_endgame: bool,
    ) -> Self {
        Self {
            score: score.value() as i16,
            best_move: best_move as u8,
            bound: bound as u8,
            depth: depth as u8,
            selectivity: selectivity.as_u8(),
            generation: generation & TTEntry::GENERATION_MASK,
            is_endgame: is_endgame as u8,
        }
    }

    /// Creates a new `TTDataFields` with a substituted best move.
    #[inline(always)]
    fn with_best_move(self, best_move: Square) -> Self {
        Self {
            best_move: best_move as u8,
            ..self
        }
    }
}

/// Packed metadata from a validated transposition-table entry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTEntryData {
    fields: TTDataFields,
    seq: u64,
}

impl TTEntryData {
    /// Returns the 64-bit SeqLock sequence counter.
    #[inline(always)]
    pub fn sequence(&self) -> u64 {
        self.seq
    }

    /// Returns the cached search score.
    #[inline(always)]
    pub fn score(&self) -> ScaledScore {
        ScaledScore::from_raw(self.fields.score as i32)
    }

    /// Returns the best move found for this position.
    #[inline(always)]
    pub fn best_move(&self) -> Square {
        // SAFETY: only valid Square indices (0..=64) are stored.
        unsafe { Square::from_u8_unchecked(self.fields.best_move) }
    }

    /// Returns the [`Bound`] type of this entry.
    #[inline(always)]
    pub fn bound(&self) -> Bound {
        // SAFETY: bound was stored from a valid Bound enum discriminant (0–3).
        unsafe { Bound::from_u8_unchecked(self.fields.bound) }
    }

    /// Returns the search depth at which this entry was computed.
    #[inline(always)]
    pub fn depth(&self) -> Depth {
        self.fields.depth as Depth
    }

    /// Returns the [`Selectivity`] level used during search.
    #[inline(always)]
    pub fn selectivity(&self) -> Selectivity {
        Selectivity::from_u8(self.fields.selectivity)
    }

    /// Returns the generation counter when this entry was stored.
    #[inline(always)]
    pub fn generation(&self) -> u8 {
        self.fields.generation
    }

    /// Returns `true` if this entry is from an endgame search.
    #[inline(always)]
    pub fn is_endgame(&self) -> bool {
        self.fields.is_endgame != 0
    }

    /// Returns whether this entry allows an immediate return in a null-window
    /// search with target `beta`.
    ///
    /// - If `score >= beta` (fail-high): Returns `true` if the entry contains
    ///   a lower or exact bound.
    /// - If `score < beta` (fail-low): Returns `true` if the entry contains
    ///   an upper or exact bound.
    #[inline(always)]
    pub fn can_cut(&self, beta: ScaledScore) -> bool {
        let required = if self.score() >= beta {
            Bound::Lower as u8
        } else {
            Bound::Upper as u8
        };
        (self.fields.bound & required) != 0
    }

    /// Returns `true` if the entry contains initialized metadata.
    #[inline(always)]
    pub fn is_occupied(&self) -> bool {
        self.fields.bound != 0
    }

    /// Returns the age distance in the 7-bit generation ring.
    #[inline(always)]
    fn relative_age(&self, generation: u8) -> i32 {
        (generation.wrapping_sub(self.fields.generation) & TTEntry::GENERATION_MASK) as i32
    }
}

/// One slot inside a transposition-table cluster.
///
/// Stores the full board, [`TTDataFields`] (transmuted through an AtomicU64),
/// and a 64-bit SeqLock sequence counter used to read the split payload
/// consistently.
#[repr(C)]
pub struct TTEntry {
    player: AtomicU64,
    opponent: AtomicU64,
    data: AtomicU64,
    seq: AtomicU64,
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            player: AtomicU64::new(0),
            opponent: AtomicU64::new(0),
            data: AtomicU64::new(0),
            seq: AtomicU64::new(0),
        }
    }
}

impl TTEntry {
    /// Weight applied to the generation age difference in the replacement score.
    const AGE_WEIGHT: i32 = 8;

    /// 7-bit generation mask for the generation ring.
    const GENERATION_MASK: u8 = 0x7F;

    /// Retry budget for reading a stable SeqLock snapshot when a writer is mid-update.
    const READ_RETRIES: u32 = 3;

    /// Retry budget for the save CAS loop when another writer wins the race.
    const SAVE_RETRIES: u32 = 4;

    /// Attempts to read a stable snapshot of the slot.
    #[inline(always)]
    fn try_load_snapshot(&self) -> Option<(u64, u64, TTEntryData)> {
        let seq1 = self.seq.load(Ordering::Acquire);
        if (seq1 & 1) != 0 {
            return None;
        }

        // `seq1` synchronizes with the writer's final release-store of the even
        // sequence value, so the split payload can be read with relaxed loads.
        let player = self.player.load(Ordering::Relaxed);
        let opponent = self.opponent.load(Ordering::Relaxed);
        let raw = self.data.load(Ordering::Relaxed);

        // Keep the payload loads before the validation read on weakly ordered
        // CPUs while preserving the initial acquire edge through `seq1`.
        fence(Ordering::Acquire);
        let seq2 = self.seq.load(Ordering::Relaxed);
        if seq1 != seq2 {
            return None;
        }

        Some((
            player,
            opponent,
            TTEntryData {
                fields: TTDataFields::from_u64(raw),
                seq: seq1,
            },
        ))
    }

    /// Reads the slot and returns metadata if it matches `board`.
    #[inline(always)]
    pub fn read(&self, board: &Board) -> Option<TTEntryData> {
        let board_player = board.player.bits();
        let board_opponent = board.opponent.bits();
        for _ in 0..Self::READ_RETRIES {
            if let Some((player, opponent, data)) = self.try_load_snapshot() {
                if player == board_player && opponent == board_opponent {
                    return Some(data);
                }
                // Board doesn't match — genuine miss, no retry needed.
                return None;
            }
            std::hint::spin_loop();
        }
        None
    }

    /// Stores a result if it wins the replacement policy.
    ///
    /// Exact bounds always replace. A different board also replaces, since the
    /// caller already chose this slot as the cluster victim. For the same board,
    /// non-exact results replace when depth regression is at most two plies, selectivity is
    /// higher, or generation changed. If `best_move` is [`Square::None`], the
    /// existing move is preserved for same-board non-exact updates.
    pub(crate) fn save(&self, board: &Board, data: TTDataFields) {
        let new_player = board.player.bits();
        let new_opponent = board.opponent.bits();
        // SAFETY: data was constructed by TTDataFields::new(), which stores
        // valid Bound and Square discriminants.
        let bound = unsafe { Bound::from_u8_unchecked(data.bound) };
        let best_move = unsafe { Square::from_u8_unchecked(data.best_move) };

        // Retry the full replacement policy if another writer wins the race
        // between snapshot acquisition and the claim CAS.
        for _ in 0..Self::SAVE_RETRIES {
            let Some((stored_player, stored_opponent, stored_data)) = self.try_load_snapshot()
            else {
                std::hint::spin_loop();
                continue;
            };

            let same_board = new_player == stored_player && new_opponent == stored_opponent;
            let should_replace = if bound == Bound::Exact || !same_board {
                true
            } else {
                // Tolerate up to 2 plies of depth regression so aspiration re-searches
                // and nearby-depth revisits can still overwrite the slot.
                data.depth as u32 + 2 >= stored_data.depth()
                    || data.selectivity > stored_data.selectivity().as_u8()
                    || data.generation != stored_data.generation()
            };

            if !should_replace {
                return;
            }

            let write_data = if best_move != Square::None || bound == Bound::Exact || !same_board {
                data
            } else {
                data.with_best_move(stored_data.best_move())
            };

            if self.seqlock_write(stored_data.seq, new_player, new_opponent, write_data) {
                return;
            }

            std::hint::spin_loop();
        }
    }

    /// Tries to claim the slot from a specific stable snapshot and publish a new value.
    #[inline(always)]
    fn seqlock_write(
        &self,
        expected_seq: u64,
        player: u64,
        opponent: u64,
        data: TTDataFields,
    ) -> bool {
        debug_assert_eq!(expected_seq & 1, 0, "expected stable snapshot");

        if self
            .seq
            .compare_exchange(
                expected_seq,
                expected_seq | 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return false;
        }

        // We now hold exclusive write access (sequence is odd).
        //
        // The release fence ensures that, on weakly ordered architectures,
        // the odd-sequence CAS store is visible to other cores before any
        // payload store below. Without it a reader could observe new payload
        // values while still seeing the old even sequence on both its
        // validation loads, accepting a torn snapshot as consistent.
        fence(Ordering::Release);

        self.player.store(player, Ordering::Relaxed);
        self.opponent.store(opponent, Ordering::Relaxed);
        self.data.store(data.to_u64(), Ordering::Relaxed);
        self.seq
            .store(expected_seq.wrapping_add(2), Ordering::Release);
        true
    }
}

/// Shared transposition table.
pub struct TranspositionTable {
    /// Flat array of [`TTEntry`] values grouped into fixed-size clusters.
    entries: AVec<TTEntry, ConstAlign<CACHE_LINE_SIZE>>,
    /// Number of [`TTEntry`] clusters in the table.
    cluster_count: u64,
    /// Generation counter for entry aging (incremented each search).
    generation: AtomicU8,
}

impl TranspositionTable {
    /// Retry budget for `probe` when every slot in the cluster is in-flight.
    const PROBE_RETRIES: u32 = 3;

    /// Creates a table with `mb_size` MiB of storage.
    ///
    /// The backing entry storage is exactly `mb_size` MiB when `mb_size > 0`.
    /// When `mb_size` is 0, creates a minimal 16-cluster table (1 KiB)
    /// suitable for testing.
    pub fn new(mb_size: usize) -> Self {
        let cluster_count = if mb_size == 0 {
            16
        } else {
            let cluster_byte_size = mem::size_of::<TTEntry>() * CLUSTER_SIZE;
            (mb_size as u64 * 1024 * 1024) / cluster_byte_size as u64
        };
        let entries_size = cluster_count as usize * CLUSTER_SIZE;

        TranspositionTable {
            entries: AVec::from_iter(
                CACHE_LINE_SIZE,
                (0..entries_size).map(|_| TTEntry::default()),
            ),
            cluster_count,
            generation: AtomicU8::new(0),
        }
    }

    /// Returns the table size in MiB.
    pub fn mb_size(&self) -> usize {
        let cluster_byte_size = mem::size_of::<TTEntry>() * CLUSTER_SIZE;
        ((self.cluster_count * cluster_byte_size as u64) / (1024 * 1024)) as usize
    }

    /// Clears all entries. Must be called with no concurrent readers or writers.
    pub fn clear(&self) {
        // SAFETY: All TTEntry fields are AtomicU64 (interior-mutable via UnsafeCell),
        // and zeroing bytes is equivalent to AtomicU64::new(0).
        unsafe {
            std::ptr::write_bytes(
                self.entries.as_ptr() as *mut u8,
                0,
                self.entries.len() * mem::size_of::<TTEntry>(),
            );
        }
    }

    /// Estimates the fraction of occupied entries by sampling.
    ///
    /// Samples up to 1024 clusters uniformly across the table and returns
    /// the ratio of occupied entries as a value in `[0.0, 1.0]`.
    pub fn usage_rate(&self) -> f64 {
        let cluster_count = self.cluster_count as usize;
        if cluster_count == 0 {
            return 0.0;
        }
        let sample_clusters = 1024.min(cluster_count);
        let step = cluster_count / sample_clusters;
        let mut occupied = 0u64;
        for i in 0..sample_clusters {
            let base = i * step * CLUSTER_SIZE;
            for j in 0..CLUSTER_SIZE {
                if let Some((_, _, data)) = self.entries[base + j].try_load_snapshot()
                    && data.is_occupied()
                {
                    occupied += 1;
                }
            }
        }
        occupied as f64 / (sample_clusters * CLUSTER_SIZE) as f64
    }

    /// Returns the current generation counter value (7-bit, `0..=127`).
    #[inline]
    pub fn generation(&self) -> u8 {
        self.generation.load(Ordering::Relaxed) & TTEntry::GENERATION_MASK
    }

    /// Increments the generation counter and returns the new value.
    ///
    /// The counter wraps at 128 (7-bit range) to match [`TTDataFields::generation`].
    #[inline]
    pub fn increment_generation(&self) -> u8 {
        let new =
            self.generation.load(Ordering::Relaxed).wrapping_add(1) & TTEntry::GENERATION_MASK;
        self.generation.store(new, Ordering::Relaxed);
        new
    }

    /// Resets the generation counter to zero.
    #[inline]
    pub fn reset_generation(&self) {
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Prefetches the cluster corresponding to the given hash key with a
    /// temporal-locality cache hint.
    ///
    /// Clusters are 64 bytes (2 x 32-byte entries) and the backing allocation is
    /// 64-byte aligned, so prefetching the cluster start covers the full cluster.
    #[inline]
    pub fn prefetch(&self, key: u64) {
        let index = self.get_cluster_idx(key);
        // SAFETY: `get_cluster_idx` returns an index within `entries`, so `.add(index)`
        // stays within the allocation.
        let addr = unsafe { self.entries.as_ptr().add(index) };
        prefetch_read(addr, Locality::L1);
    }

    /// Performs a read-only lookup without selecting a replacement slot.
    #[inline(always)]
    pub fn lookup(&self, board: &Board, key: u64) -> Option<TTEntryData> {
        let cluster_idx = self.get_cluster_idx(key);

        for i in 0..CLUSTER_SIZE {
            // SAFETY: `cluster_idx + i < cluster_count * CLUSTER_SIZE == entries.len()`.
            let entry = unsafe { self.entries.get_unchecked(cluster_idx + i) };
            if let Some(data) = entry.read(board) {
                return Some(data);
            }
        }

        None
    }

    /// Probes the cluster for `board`.
    ///
    /// On miss, returns the first unused slot if one exists; otherwise returns
    /// the slot with the lowest replacement score (`depth - AGE_WEIGHT * relative_age`).
    #[inline(always)]
    pub fn probe(&self, board: &Board, key: u64) -> TTProbeResult {
        let cluster_idx = self.get_cluster_idx(key);
        let board_player = board.player.bits();
        let board_opponent = board.opponent.bits();
        let generation = self.generation();
        // Keep the last stable victim so a persistently busy slot does not
        // force replacement back to the cluster head.
        let mut fallback_replace_idx = None;

        // Retry the cluster when a slot is in-flight so a just-finished hit
        // can still win over a replacement miss.
        for _ in 0..Self::PROBE_RETRIES {
            let mut replace_idx = cluster_idx;
            let mut replace_score = i32::MAX;
            let mut saw_in_flight_writer = false;

            for i in 0..CLUSTER_SIZE {
                let idx = cluster_idx + i;
                // SAFETY: `cluster_idx + i < cluster_count * CLUSTER_SIZE == entries.len()`.
                let entry = unsafe { self.entries.get_unchecked(idx) };
                let Some((player, opponent, tt_data)) = entry.try_load_snapshot() else {
                    saw_in_flight_writer = true;
                    continue;
                };

                if player == board_player && opponent == board_opponent {
                    return TTProbeResult::Hit {
                        data: tt_data,
                        index: idx,
                    };
                }

                if !tt_data.is_occupied() {
                    return TTProbeResult::Miss { index: idx };
                }

                let score =
                    tt_data.depth() as i32 - tt_data.relative_age(generation) * TTEntry::AGE_WEIGHT;
                if score < replace_score {
                    replace_score = score;
                    replace_idx = idx;
                }
            }

            if replace_score != i32::MAX {
                fallback_replace_idx = Some(replace_idx);
            }

            if !saw_in_flight_writer {
                return TTProbeResult::Miss { index: replace_idx };
            }

            std::hint::spin_loop();
        }

        TTProbeResult::Miss {
            index: fallback_replace_idx.unwrap_or(cluster_idx),
        }
    }

    /// Stores data into a slot previously returned by [`probe`](Self::probe).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `entry_index` is out of bounds.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn store(
        &self,
        entry_index: usize,
        board: &Board,
        score: ScaledScore,
        bound: Bound,
        depth: Depth,
        best_move: Square,
        selectivity: Selectivity,
        is_endgame: bool,
    ) {
        debug_assert!(
            entry_index < self.entries.len(),
            "TT store index {} out of bounds",
            entry_index
        );
        let data = TTDataFields::new(
            score,
            best_move,
            bound,
            depth,
            selectivity,
            self.generation(),
            is_endgame,
        );
        // SAFETY: `entry_index` originates from `probe`, which only returns
        // indices within `entries`.
        let entry = unsafe { self.entries.get_unchecked(entry_index) };
        entry.save(board, data);
    }

    /// Returns the first entry index of the cluster selected by `key`.
    #[inline(always)]
    fn get_cluster_idx(&self, key: u64) -> usize {
        (crate::util::mul_hi64(key, self.cluster_count) as usize) * CLUSTER_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::Bitboard;
    use crate::search::node_type::{NonPV, PV};

    fn sq(idx: usize) -> Square {
        Square::from_usize(idx).unwrap()
    }

    fn make_board(player: u64, opponent: u64) -> Board {
        Board {
            player: Bitboard::new(player),
            opponent: Bitboard::new(opponent),
        }
    }

    /// Tests that TTEntry correctly packs and unpacks all fields.
    #[test]
    fn test_ttentry_store_and_read() {
        let entry = TTEntry::default();
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let test_score = ScaledScore::from_disc_diff(-64);
        let test_bound = Bound::Exact;
        let test_depth: Depth = 60;
        let test_best_move = sq(3);
        let test_selectivity = Selectivity::Level5;
        let test_generation: u8 = 127;

        entry.save(
            &board,
            TTDataFields::new(
                test_score,
                test_best_move,
                test_bound,
                test_depth,
                test_selectivity,
                test_generation,
                false,
            ),
        );

        let data = entry.read(&board).expect("should hit");
        assert_eq!(data.score(), test_score);
        assert_eq!(data.bound(), test_bound);
        assert_eq!(data.depth(), test_depth);
        assert_eq!(data.best_move(), test_best_move);
        assert_eq!(data.selectivity(), test_selectivity);
        assert_eq!(data.generation(), test_generation);
    }

    /// Tests that reading with a different board returns None.
    #[test]
    fn test_ttentry_different_board_miss() {
        let entry = TTEntry::default();
        let board1 = make_board(1, 2);
        let board2 = make_board(3, 4);

        entry.save(
            &board1,
            TTDataFields::new(
                ScaledScore::from_raw(50),
                sq(5),
                Bound::Exact,
                10,
                Selectivity::Level2,
                1,
                false,
            ),
        );

        assert!(entry.read(&board1).is_some());
        assert!(entry.read(&board2).is_none());
    }

    /// Tests the replacement policy in TTEntry::save.
    #[test]
    fn test_ttentry_replacement_policy() {
        let entry = TTEntry::default();
        let board = make_board(100, 200);

        // Initial save
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(50),
                sq(5),
                Bound::Lower,
                10,
                Selectivity::Level2,
                1,
                false,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 10);

        // Slightly shallower (within 2 plies) - should replace
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(60),
                sq(6),
                Bound::Lower,
                8,
                Selectivity::Level2,
                1,
                false,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 8);
        assert_eq!(data.score().value(), 60);

        // Deeper - should replace
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(70),
                sq(7),
                Bound::Lower,
                12,
                Selectivity::Level2,
                1,
                false,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 12);

        // Exact bound - always replaces
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(80),
                sq(8),
                Bound::Exact,
                5,
                Selectivity::Level2,
                1,
                false,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 5);
        assert_eq!(data.bound(), Bound::Exact);

        // Different board - always replaces
        let board2 = make_board(300, 400);
        entry.save(
            &board2,
            TTDataFields::new(
                ScaledScore::from_raw(90),
                sq(9),
                Bound::Upper,
                3,
                Selectivity::Level2,
                1,
                false,
            ),
        );
        let data = entry.read(&board2).unwrap();
        assert_eq!(data.score().value(), 90);
        assert!(entry.read(&board).is_none()); // old board no longer matches

        // Newer generation - should replace
        entry.save(
            &board2,
            TTDataFields::new(
                ScaledScore::from_raw(100),
                sq(10),
                Bound::Lower,
                2,
                Selectivity::Level2,
                5,
                false,
            ),
        );
        let data = entry.read(&board2).unwrap();
        assert_eq!(data.generation(), 5);
    }

    /// Tests that save preserves the stored best move when the new one is Square::None.
    #[test]
    fn test_ttentry_preserves_best_move_when_new_move_is_none() {
        let entry = TTEntry::default();
        let board = make_board(123, 456);

        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(30),
                sq(11),
                Bound::Lower,
                12,
                Selectivity::Level2,
                1,
                false,
            ),
        );

        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(40),
                Square::None,
                Bound::Lower,
                11,
                Selectivity::Level2,
                1,
                false,
            ),
        );

        let data = entry.read(&board).unwrap();
        assert_eq!(data.best_move(), sq(11));
        assert_eq!(data.score().value(), 40);
    }

    /// Tests Bound::classify for different node types.
    #[test]
    fn test_bound_classify() {
        let s = |v| ScaledScore::from_raw(v);
        assert_eq!(Bound::classify::<PV>(s(100), s(30), s(50)), Bound::Lower);
        assert_eq!(Bound::classify::<PV>(s(40), s(30), s(50)), Bound::Exact);
        assert_eq!(Bound::classify::<PV>(s(20), s(30), s(50)), Bound::Upper);
        assert_eq!(Bound::classify::<NonPV>(s(100), s(30), s(50)), Bound::Lower);
        assert_eq!(Bound::classify::<NonPV>(s(40), s(30), s(50)), Bound::Upper);
        assert_eq!(Bound::classify::<NonPV>(s(20), s(30), s(50)), Bound::Upper);
    }

    /// Tests TTEntryData occupancy and cutoff helpers.
    #[test]
    fn test_ttentry_data_helpers() {
        let s = |v| ScaledScore::from_raw(v);
        let entry = TTEntry::default();
        let board = make_board(1, 2);

        // Test is_occupied on an empty entry.
        assert!(!entry.try_load_snapshot().unwrap().2.is_occupied());

        // After storing with Lower bound
        entry.save(
            &board,
            TTDataFields::new(
                s(0),
                Square::None,
                Bound::Lower,
                0,
                Selectivity::Level1,
                0,
                false,
            ),
        );
        assert!(entry.read(&board).unwrap().is_occupied());

        // can_cut tests
        entry.save(
            &board,
            TTDataFields::new(
                s(100),
                Square::None,
                Bound::Lower,
                0,
                Selectivity::Level1,
                0,
                false,
            ),
        );
        assert!(entry.read(&board).unwrap().can_cut(s(50)));
        assert!(!entry.read(&board).unwrap().can_cut(s(150)));

        entry.save(
            &board,
            TTDataFields::new(
                s(30),
                Square::None,
                Bound::Upper,
                0,
                Selectivity::Level1,
                0,
                false,
            ),
        );
        assert!(entry.read(&board).unwrap().can_cut(s(50)));
        assert!(!entry.read(&board).unwrap().can_cut(s(20)));

        entry.save(
            &board,
            TTDataFields::new(
                s(30),
                Square::None,
                Bound::Exact,
                0,
                Selectivity::Level1,
                0,
                false,
            ),
        );
        assert!(entry.read(&board).unwrap().can_cut(s(50)));
        assert!(entry.read(&board).unwrap().can_cut(s(20)));
    }

    /// Tests TranspositionTable creation with different sizes.
    #[test]
    fn test_transposition_table_new() {
        let tt = TranspositionTable::new(0);
        assert_eq!(std::mem::size_of::<TTEntry>(), 32);
        assert_eq!(tt.cluster_count, 16);
        assert_eq!(tt.entries.len(), 16 * CLUSTER_SIZE);

        let tt = TranspositionTable::new(1);
        let expected_clusters = (1024 * 1024) / (std::mem::size_of::<TTEntry>() * CLUSTER_SIZE);
        assert_eq!(tt.cluster_count, expected_clusters as u64);
    }

    /// Tests basic probe and store operations.
    #[test]
    fn test_probe_and_store() {
        let tt = TranspositionTable::new(1);
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();
        let generation = tt.increment_generation();

        assert!(!tt.probe(&board, key).is_hit());

        let idx = tt.probe(&board, key).index();
        tt.store(
            idx,
            &board,
            ScaledScore::from_raw(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level3,
            false,
        );

        let result = tt.probe(&board, key);
        assert!(result.is_hit());
        let d = result.data().unwrap();
        assert_eq!(d.score().value(), 100);
        assert_eq!(d.bound(), Bound::Exact);
        assert_eq!(d.depth(), 20);
        assert_eq!(d.best_move(), sq(10));
        assert_eq!(d.selectivity(), Selectivity::Level3);
        assert_eq!(d.generation(), generation);
    }

    /// Tests the clear function.
    #[test]
    fn test_clear() {
        let tt = TranspositionTable::new(1);
        tt.increment_generation();

        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();
        let idx = tt.probe(&board, key).index();
        tt.store(
            idx,
            &board,
            ScaledScore::from_raw(100),
            Bound::Exact,
            20,
            sq(10),
            Selectivity::Level1,
            false,
        );

        assert!(tt.probe(&board, key).is_hit());

        tt.clear();
        assert!(!tt.probe(&board, key).is_hit());
    }

    /// Tests that generation wraps at 128 (7-bit), not 256.
    #[test]
    fn test_generation_wraps_at_128() {
        let tt = TranspositionTable::new(0);
        for _ in 0..127 {
            tt.increment_generation();
        }
        assert_eq!(tt.generation(), 127);

        // Next increment should wrap to 0
        let g = tt.increment_generation();
        assert_eq!(g, 0);
        assert_eq!(tt.generation(), 0);

        // Verify it keeps counting from 0
        let g = tt.increment_generation();
        assert_eq!(g, 1);
    }

    /// Tests that probe prefers an unused slot even immediately after generation wrap.
    #[test]
    fn test_probe_prefers_unused_slot_after_generation_wrap() {
        let tt = TranspositionTable::new(0);

        for _ in 0..127 {
            tt.increment_generation();
        }
        assert_eq!(tt.generation(), 127);

        let live_board = make_board(0x0000000810000000, 0x0000001008000000);
        let cluster_idx = tt.get_cluster_idx(live_board.hash());

        tt.store(
            cluster_idx,
            &live_board,
            ScaledScore::from_raw(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
        );

        assert_eq!(
            tt.entries[cluster_idx]
                .read(&live_board)
                .unwrap()
                .generation(),
            127
        );
        assert!(
            !tt.entries[cluster_idx + 1]
                .try_load_snapshot()
                .unwrap()
                .2
                .is_occupied()
        );

        tt.increment_generation();
        assert_eq!(tt.generation(), 0);

        let miss_board = (1u64..)
            .map(|seed| make_board(seed, seed << 32))
            .find(|board| tt.get_cluster_idx(board.hash()) == cluster_idx)
            .expect("should find a second board in the same cluster");

        let probe = tt.probe(&miss_board, miss_board.hash());
        assert!(!probe.is_hit());
        assert_eq!(probe.index(), cluster_idx + 1);
    }

    /// Tests that probe does not treat an in-flight SeqLock write as an unused slot.
    #[test]
    fn test_probe_ignores_in_flight_slot_after_generation_wrap() {
        let tt = TranspositionTable::new(0);

        for _ in 0..127 {
            tt.increment_generation();
        }
        assert_eq!(tt.generation(), 127);

        let live_board = make_board(0x0000000810000000, 0x0000001008000000);
        let cluster_idx = tt.get_cluster_idx(live_board.hash());

        tt.store(
            cluster_idx,
            &live_board,
            ScaledScore::from_raw(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
        );

        let in_flight = unsafe { tt.entries.get_unchecked(cluster_idx + 1) };
        assert!(
            in_flight
                .seq
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        );
        // Odd sequence means try_load_snapshot fails (write in progress).
        assert!(in_flight.try_load_snapshot().is_none());

        tt.increment_generation();
        assert_eq!(tt.generation(), 0);

        let miss_board = (1u64..)
            .map(|seed| make_board(seed, seed << 32))
            .find(|board| tt.get_cluster_idx(board.hash()) == cluster_idx)
            .expect("should find a second board in the same cluster");

        let probe = tt.probe(&miss_board, miss_board.hash());
        assert!(!probe.is_hit());
        assert_eq!(probe.index(), cluster_idx);
    }

    /// Tests that probe keeps the last stable victim if the first slot stays busy.
    #[test]
    fn test_probe_keeps_stable_victim_when_first_slot_stays_busy() {
        let tt = TranspositionTable::new(0);
        let stable_board = make_board(0x0000000810000000, 0x0000001008000000);
        let cluster_idx = tt.get_cluster_idx(stable_board.hash());

        tt.store(
            cluster_idx + 1,
            &stable_board,
            ScaledScore::from_raw(100),
            Bound::Lower,
            4,
            sq(10),
            Selectivity::Level1,
            false,
        );

        let in_flight = unsafe { tt.entries.get_unchecked(cluster_idx) };
        assert!(
            in_flight
                .seq
                .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        );
        assert!(in_flight.try_load_snapshot().is_none());

        let miss_board = (1u64..)
            .map(|seed| make_board(seed, seed << 32))
            .find(|board| tt.get_cluster_idx(board.hash()) == cluster_idx && *board != stable_board)
            .expect("should find a second board in the same cluster");

        let probe = tt.probe(&miss_board, miss_board.hash());
        assert!(!probe.is_hit());
        assert_eq!(probe.index(), cluster_idx + 1);
    }

    /// Tests that probe falls back to cluster_idx when all slots are in-flight.
    #[test]
    fn test_probe_falls_back_to_cluster_idx_when_all_slots_busy() {
        let tt = TranspositionTable::new(0);
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let cluster_idx = tt.get_cluster_idx(board.hash());

        // Mark both slots as in-flight (odd sequence number).
        for i in 0..CLUSTER_SIZE {
            let entry = unsafe { tt.entries.get_unchecked(cluster_idx + i) };
            assert!(
                entry
                    .seq
                    .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            );
        }

        let miss_board = (1u64..)
            .map(|seed| make_board(seed, seed << 32))
            .find(|b| tt.get_cluster_idx(b.hash()) == cluster_idx)
            .expect("should find a board in the same cluster");

        let probe = tt.probe(&miss_board, miss_board.hash());
        assert!(!probe.is_hit());
        assert_eq!(probe.index(), cluster_idx);
    }

    /// Tests that relative_age works correctly across the 7-bit wrap boundary.
    #[test]
    fn test_relative_age_wrap_around() {
        let entry = TTEntry::default();
        let board = make_board(1, 2);

        // Store entry at generation 126
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(0),
                Square::None,
                Bound::Exact,
                0,
                Selectivity::Level1,
                126,
                false,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.generation(), 126);

        // Global generation wrapped to 2: age = (2 - 126) & 0x7F = 4
        assert_eq!(data.relative_age(2), 4);

        // Global generation at 127: age = (127 - 126) & 0x7F = 1
        assert_eq!(data.relative_age(127), 1);

        // Same generation: age = 0
        assert_eq!(data.relative_age(126), 0);
    }

    /// Tests that the is_endgame flag is not corrupted even with large generation values.
    #[test]
    fn test_no_endgame_corruption() {
        let entry = TTEntry::default();
        let board = make_board(42, 84);

        // Store with generation=127, is_endgame=false
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(50),
                sq(5),
                Bound::Exact,
                10,
                Selectivity::Level1,
                127,
                false,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.generation(), 127);
        assert!(!data.is_endgame(), "is_endgame should be false");

        // Store with generation=127, is_endgame=true
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(50),
                sq(5),
                Bound::Exact,
                10,
                Selectivity::Level1,
                127,
                true,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.generation(), 127);
        assert!(data.is_endgame(), "is_endgame should be true");

        // Defense-in-depth: even if generation=255 is passed (should be masked to 127)
        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(50),
                sq(5),
                Bound::Exact,
                10,
                Selectivity::Level1,
                255,
                false,
            ),
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.generation(), 127);
        assert!(
            !data.is_endgame(),
            "is_endgame must not be corrupted by generation overflow"
        );
    }

    /// Tests SeqLock sequence counter behavior.
    #[test]
    fn test_seqlock_version_increments() {
        let entry = TTEntry::default();
        let board = make_board(1, 2);

        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(10),
                sq(1),
                Bound::Lower,
                5,
                Selectivity::Level1,
                0,
                false,
            ),
        );
        let v1 = entry.read(&board).unwrap().sequence();
        assert!(v1 & 1 == 0, "sequence should be even after save");

        entry.save(
            &board,
            TTDataFields::new(
                ScaledScore::from_raw(20),
                sq(2),
                Bound::Exact,
                6,
                Selectivity::Level1,
                0,
                false,
            ),
        );
        let v2 = entry.read(&board).unwrap().sequence();
        assert!(v2 > v1, "sequence should increment");
        assert!(v2 & 1 == 0, "sequence should be even after save");
    }

    /// Tests that stale snapshots cannot claim the entry for writing.
    #[test]
    fn test_seqlock_write_rejects_stale_snapshot() {
        let entry = TTEntry::default();
        let board1 = make_board(1, 2);
        let board2 = make_board(3, 4);

        entry.save(
            &board1,
            TTDataFields::new(
                ScaledScore::from_raw(10),
                sq(1),
                Bound::Exact,
                5,
                Selectivity::Level1,
                0,
                false,
            ),
        );
        let (_, _, snapshot) = entry.try_load_snapshot().unwrap();

        entry.save(
            &board2,
            TTDataFields::new(
                ScaledScore::from_raw(20),
                sq(2),
                Bound::Exact,
                6,
                Selectivity::Level2,
                1,
                false,
            ),
        );

        assert!(!entry.seqlock_write(
            snapshot.seq,
            1,
            2,
            TTDataFields::new(
                ScaledScore::from_raw(30),
                sq(3),
                Bound::Exact,
                7,
                Selectivity::Level3,
                2,
                false,
            ),
        ));

        assert!(entry.read(&board1).is_none());
        let data = entry.read(&board2).unwrap();
        assert_eq!(data.score().value(), 20);
        assert_eq!(data.depth(), 6);
        assert_eq!(data.best_move(), sq(2));
        assert_eq!(data.generation(), 1);
    }
}
