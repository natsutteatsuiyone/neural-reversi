//! Transposition table for caching search results.
//!
//! Each entry stores the full board plus packed search metadata, so hash-index
//! collisions only affect replacement and never produce a false hit.

use crate::board::Board;
use crate::probcut::Selectivity;
use crate::search::node_type::NodeType;
use crate::square::Square;
use crate::types::{Depth, ScaledScore, Score};
use aligned_vec::{AVec, ConstAlign};
use std::{
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
    pub fn classify_scaled<NT: NodeType>(
        best_score: ScaledScore,
        alpha: ScaledScore,
        beta: ScaledScore,
    ) -> Bound {
        Self::classify_inner::<NT>(best_score.value(), alpha.value(), beta.value())
    }

    /// Classifies an endgame search result into a [`Bound`] type.
    ///
    /// - [`Bound::Lower`] if `best_score >= beta` (fail-high).
    /// - [`Bound::Exact`] if `alpha < best_score < beta` and node is PV.
    /// - [`Bound::Upper`] otherwise (fail-low or non-PV exact-ish value).
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

    /// Converts a 2-bit value to a [`Bound`].
    #[inline]
    pub fn from_u8(value: u8) -> Bound {
        match value {
            0 => Bound::None,
            1 => Bound::Lower,
            2 => Bound::Upper,
            _ => Bound::Exact,
        }
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
            TTProbeResult::Hit { index, .. } => *index,
            TTProbeResult::Miss { index } => *index,
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

/// Packed metadata from a validated transposition-table entry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TTEntryData {
    raw: u64,
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
        let raw = ((self.raw >> TTEntry::SCORE_SHIFT) & TTEntry::SCORE_MASK) as i16 as i32;
        ScaledScore::from_raw(raw)
    }

    /// Returns the best move found for this position.
    #[inline(always)]
    pub fn best_move(&self) -> Square {
        let val = ((self.raw >> TTEntry::BEST_MOVE_SHIFT) & TTEntry::BEST_MOVE_MASK) as u8;
        Square::from_u8_unchecked(val)
    }

    /// Returns the [`Bound`] type of this entry.
    #[inline(always)]
    pub fn bound(&self) -> Bound {
        let val = ((self.raw >> TTEntry::BOUND_SHIFT) & TTEntry::BOUND_MASK) as u8;
        Bound::from_u8(val)
    }

    /// Returns the search depth at which this entry was computed.
    #[inline(always)]
    pub fn depth(&self) -> Depth {
        ((self.raw >> TTEntry::DEPTH_SHIFT) & TTEntry::DEPTH_MASK) as u8 as Depth
    }

    /// Returns the [`Selectivity`] level used during search.
    #[inline(always)]
    pub fn selectivity(&self) -> Selectivity {
        let val = ((self.raw >> TTEntry::SELECTIVITY_SHIFT) & TTEntry::SELECTIVITY_MASK) as u8;
        Selectivity::from_u8(val)
    }

    /// Returns the generation counter when this entry was stored.
    #[inline(always)]
    pub fn generation(&self) -> u8 {
        ((self.raw >> TTEntry::GENERATION_SHIFT) & TTEntry::GENERATION_MASK) as u8
    }

    /// Returns `true` if this entry is from an endgame search.
    #[inline(always)]
    pub fn is_endgame(&self) -> bool {
        ((self.raw >> TTEntry::IS_ENDGAME_SHIFT) & TTEntry::IS_ENDGAME_MASK) != 0
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
        let score = self.score();
        let bound_raw = ((self.raw >> TTEntry::BOUND_SHIFT) & TTEntry::BOUND_MASK) as u8;

        let required = if score >= beta {
            Bound::Lower as u8
        } else {
            Bound::Upper as u8
        };

        (bound_raw & required) != 0
    }

    /// Returns `true` if the entry contains initialized metadata.
    #[inline(always)]
    pub fn is_occupied(&self) -> bool {
        ((self.raw >> TTEntry::BOUND_SHIFT) & TTEntry::BOUND_MASK) != 0
    }

    /// Returns the age distance in the 7-bit generation ring.
    #[inline(always)]
    fn relative_age(&self, generation: u8) -> i32 {
        ((generation.wrapping_sub(self.generation())) & (TTEntry::GENERATION_MASK as u8)) as i32
    }
}

/// One slot inside a transposition-table cluster.
///
/// Stores the full board, packed metadata, and a 64-bit SeqLock sequence
/// counter used to read the split payload consistently.
///
/// # Layout
///
/// - `player`: Player bitboard (AtomicU64)
/// - `opponent`: Opponent bitboard (AtomicU64)
/// - `data`: Packed data word with the following bit layout:
///   - 16 bits: Evaluation score
///   - 7 bits: Best move square
///   - 2 bits: Bound type (none/lower/upper/exact)
///   - 6 bits: Search depth
///   - 3 bits: Selectivity level
///   - 7 bits: Generation counter for aging
///   - 1 bit: Endgame flag
/// - `seq`: 64-bit SeqLock sequence number
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
    // Bit layout constants for packing data into 64 bits
    const SCORE_SIZE: i32 = 16;
    const SCORE_SHIFT: i32 = 0;
    const SCORE_MASK: u64 = (1 << Self::SCORE_SIZE) - 1;

    const BEST_MOVE_SIZE: i32 = 7;
    const BEST_MOVE_SHIFT: i32 = Self::SCORE_SHIFT + Self::SCORE_SIZE;
    const BEST_MOVE_MASK: u64 = (1 << Self::BEST_MOVE_SIZE) - 1;

    const BOUND_SIZE: i32 = 2;
    const BOUND_SHIFT: i32 = Self::BEST_MOVE_SHIFT + Self::BEST_MOVE_SIZE;
    const BOUND_MASK: u64 = (1 << Self::BOUND_SIZE) - 1;

    const DEPTH_SIZE: i32 = 6;
    const DEPTH_SHIFT: i32 = Self::BOUND_SHIFT + Self::BOUND_SIZE;
    const DEPTH_MASK: u64 = (1 << Self::DEPTH_SIZE) - 1;

    const SELECTIVITY_SIZE: i32 = 3;
    const SELECTIVITY_SHIFT: i32 = Self::DEPTH_SHIFT + Self::DEPTH_SIZE;
    const SELECTIVITY_MASK: u64 = (1 << Self::SELECTIVITY_SIZE) - 1;

    const GENERATION_SIZE: i32 = 7;
    const GENERATION_SHIFT: i32 = Self::SELECTIVITY_SHIFT + Self::SELECTIVITY_SIZE;
    const GENERATION_MASK: u64 = (1 << Self::GENERATION_SIZE) - 1;

    const IS_ENDGAME_SHIFT: i32 = Self::GENERATION_SHIFT + Self::GENERATION_SIZE;
    const IS_ENDGAME_MASK: u64 = 1;

    /// Weight applied to the generation age difference in the replacement score.
    const AGE_WEIGHT: i32 = 8;

    /// Builds the packed data word from all fields.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn pack_data(
        score: ScaledScore,
        best_move: u8,
        bound: u8,
        depth: u8,
        selectivity: u8,
        generation: u8,
        is_endgame: bool,
    ) -> u64 {
        (((score.value() as u16) as u64) << Self::SCORE_SHIFT)
            | ((best_move as u64) << Self::BEST_MOVE_SHIFT)
            | ((bound as u64) << Self::BOUND_SHIFT)
            | ((depth as u64) << Self::DEPTH_SHIFT)
            | ((selectivity as u64) << Self::SELECTIVITY_SHIFT)
            | (((generation & Self::GENERATION_MASK as u8) as u64) << Self::GENERATION_SHIFT)
            | ((is_endgame as u64) << Self::IS_ENDGAME_SHIFT)
    }

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

        Some((player, opponent, TTEntryData { raw, seq: seq1 }))
    }

    /// Reads the slot and returns metadata if it matches `board`.
    #[inline(always)]
    pub fn read(&self, board: &Board) -> Option<TTEntryData> {
        let board_player = board.player.bits();
        let board_opponent = board.opponent.bits();
        for _ in 0..3 {
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
    #[allow(clippy::too_many_arguments)]
    pub fn save(
        &self,
        board: &Board,
        score: ScaledScore,
        bound: Bound,
        depth: Depth,
        best_move: Square,
        selectivity: Selectivity,
        generation: u8,
        is_endgame: bool,
    ) {
        let new_player = board.player.bits();
        let new_opponent = board.opponent.bits();

        let best_move_u8 = best_move as u8;
        let bound_u8 = bound as u8;
        let depth_u8 = depth as u8;
        let selectivity_u8 = selectivity.as_u8();

        // Retry the full replacement policy if another writer wins the race
        // between snapshot acquisition and the claim CAS.
        for _ in 0..4 {
            let Some((stored_player, stored_opponent, stored_data)) = self.try_load_snapshot()
            else {
                std::hint::spin_loop();
                continue;
            };

            let same_board = new_player == stored_player && new_opponent == stored_opponent;
            let should_replace = if bound == Bound::Exact || !same_board {
                true
            } else {
                let stored_depth = stored_data.depth() as i8;
                let stored_selectivity = stored_data.selectivity().as_u8();
                let stored_generation = stored_data.generation();

                // Replace if ANY of these conditions hold:
                // 1. New depth is at most 2 plies below stored depth (allows slight depth regression)
                // 2. New selectivity is higher (more conservative search is more valuable)
                // 3. Different generation (prioritize current search's entries)
                (depth as i8) >= stored_depth.saturating_sub(2)
                    || selectivity_u8 > stored_selectivity
                    || generation != stored_generation
            };

            if !should_replace {
                return;
            }

            let write_best_move =
                if best_move != Square::None || bound == Bound::Exact || !same_board {
                    best_move_u8
                } else {
                    stored_data.best_move() as u8
                };

            if self.seqlock_write(
                stored_data.seq,
                new_player,
                new_opponent,
                write_best_move,
                score,
                bound_u8,
                depth_u8,
                selectivity_u8,
                generation,
                is_endgame,
            ) {
                return;
            }

            std::hint::spin_loop();
        }
    }

    /// Tries to claim the slot from a specific stable snapshot and publish a new value.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn seqlock_write(
        &self,
        expected_seq: u64,
        player: u64,
        opponent: u64,
        best_move: u8,
        score: ScaledScore,
        bound: u8,
        depth: u8,
        selectivity: u8,
        generation: u8,
        is_endgame: bool,
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
        // The sequence counter lives in a separate atomic from the payload, so
        // writers must publish the odd sequence value before any split payload
        // stores become visible. Otherwise, a reader that uses relaxed payload
        // loads plus a final acquire fence could still observe the old even
        // sequence around partially updated fields and accept a corrupt
        // snapshot on a weakly ordered CPU.
        fence(Ordering::Release);

        // Step 1: Store player and opponent
        self.player.store(player, Ordering::Relaxed);
        self.opponent.store(opponent, Ordering::Relaxed);

        // Step 2: Publish the new payload, then release the lock with the next even sequence.
        let final_data = Self::pack_data(
            score,
            best_move,
            bound,
            depth,
            selectivity,
            generation,
            is_endgame,
        );
        self.data.store(final_data, Ordering::Relaxed);
        self.seq
            .store(expected_seq.wrapping_add(2), Ordering::Release);
        true
    }

    #[inline(always)]
    fn clear(&self) {
        // `TranspositionTable::clear` is currently only used for quiescent
        // resets between searches/games, so bypass the SeqLock protocol and
        // zero the entry directly.
        self.player.store(0, Ordering::Relaxed);
        self.opponent.store(0, Ordering::Relaxed);
        self.data.store(0, Ordering::Relaxed);
        self.seq.store(0, Ordering::Relaxed);
    }
}

/// Shared transposition table.
pub struct TranspositionTable {
    /// Flat array of [`TTEntry`] values grouped into fixed-size clusters.
    entries: AVec<TTEntry, ConstAlign<64>>,
    /// Number of [`TTEntry`] clusters in the table.
    cluster_count: u64,
    /// Generation counter for entry aging (incremented each search).
    generation: AtomicU8,
}

impl TranspositionTable {
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
            entries: AVec::from_iter(64, (0..entries_size).map(|_| TTEntry::default())),
            cluster_count,
            generation: AtomicU8::new(0),
        }
    }

    /// Clears all entries.
    pub fn clear(&self) {
        for entry in &*self.entries {
            entry.clear();
        }
    }

    /// Returns the current generation counter value.
    #[inline]
    pub fn generation(&self) -> u8 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Increments the generation counter and returns the new value.
    ///
    /// The counter wraps at 128 (7-bit range) to match the packed storage width
    /// in [`TTEntry`], preventing overflow into the `is_endgame` flag.
    ///
    /// # Concurrency
    ///
    /// This performs a non-atomic read-modify-write. It is safe only because
    /// the sole call site (`Search::execute_search`) holds `&mut self`,
    /// guaranteeing no concurrent callers.
    #[inline]
    pub fn increment_generation(&self) -> u8 {
        let new = self.generation.load(Ordering::Relaxed).wrapping_add(1)
            & (TTEntry::GENERATION_MASK as u8);
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

    /// Performs a read-only lookup without selecting a replacement slot.
    #[inline(always)]
    pub fn lookup(&self, board: &Board, key: u64) -> Option<TTEntryData> {
        let cluster_idx = self.get_cluster_idx(key);

        for i in 0..CLUSTER_SIZE {
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
        for _ in 0..3 {
            let mut replace_idx = cluster_idx;
            let mut replace_score = i32::MAX;
            let mut saw_in_flight_writer = false;

            for i in 0..CLUSTER_SIZE {
                let idx = cluster_idx + i;
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
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
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
        let entry = unsafe { self.entries.get_unchecked(entry_index) };
        entry.save(
            board,
            score,
            bound,
            depth,
            best_move,
            selectivity,
            self.generation(),
            is_endgame,
        );
    }

    /// Returns the first entry index of the cluster selected by `key`.
    #[inline(always)]
    fn get_cluster_idx(&self, key: u64) -> usize {
        (Self::mul_hi64(key, self.cluster_count) as usize) * CLUSTER_SIZE
    }

    /// Returns the high 64 bits of `a * b`.
    #[inline(always)]
    fn mul_hi64(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        (product >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::Bitboard;
    use crate::search::node_type::{NonPV, PV};

    fn sq(idx: usize) -> Square {
        Square::from_usize_unchecked(idx)
    }

    fn make_board(player: u64, opponent: u64) -> Board {
        Board {
            player: Bitboard::new(player),
            opponent: Bitboard::new(opponent),
        }
    }

    fn store_entry(
        tt: &TranspositionTable,
        board: &Board,
        score: ScaledScore,
        bound: Bound,
        depth: Depth,
        best_move: usize,
        selectivity: Selectivity,
    ) -> usize {
        let key = board.hash();
        let idx = tt.probe(board, key).index();
        tt.store(
            idx,
            board,
            score,
            bound,
            depth,
            sq(best_move),
            selectivity,
            false,
        );
        idx
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
            test_score,
            test_bound,
            test_depth,
            test_best_move,
            test_selectivity,
            test_generation,
            false,
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
            ScaledScore::from_raw(50),
            Bound::Exact,
            10,
            sq(5),
            Selectivity::Level2,
            1,
            false,
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
            ScaledScore::from_raw(50),
            Bound::Lower,
            10,
            sq(5),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 10);

        // Slightly shallower (within 2 plies) - should replace
        entry.save(
            &board,
            ScaledScore::from_raw(60),
            Bound::Lower,
            8,
            sq(6),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 8);
        assert_eq!(data.score().value(), 60);

        // Deeper - should replace
        entry.save(
            &board,
            ScaledScore::from_raw(70),
            Bound::Lower,
            12,
            sq(7),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 12);

        // Exact bound - always replaces
        entry.save(
            &board,
            ScaledScore::from_raw(80),
            Bound::Exact,
            5,
            sq(8),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.depth(), 5);
        assert_eq!(data.bound(), Bound::Exact);

        // Different board - always replaces
        let board2 = make_board(300, 400);
        entry.save(
            &board2,
            ScaledScore::from_raw(90),
            Bound::Upper,
            3,
            sq(9),
            Selectivity::Level2,
            1,
            false,
        );
        let data = entry.read(&board2).unwrap();
        assert_eq!(data.score().value(), 90);
        assert!(entry.read(&board).is_none()); // old board no longer matches

        // Newer generation - should replace
        entry.save(
            &board2,
            ScaledScore::from_raw(100),
            Bound::Lower,
            2,
            sq(10),
            Selectivity::Level2,
            5,
            false,
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
            ScaledScore::from_raw(30),
            Bound::Lower,
            12,
            sq(11),
            Selectivity::Level2,
            1,
            false,
        );

        entry.save(
            &board,
            ScaledScore::from_raw(40),
            Bound::Lower,
            11,
            Square::None,
            Selectivity::Level2,
            1,
            false,
        );

        let data = entry.read(&board).unwrap();
        assert_eq!(data.best_move(), sq(11));
        assert_eq!(data.score().value(), 40);
    }

    /// Tests Bound::classify_scaled for different node types.
    #[test]
    fn test_bound_classify_scaled() {
        let s = |v| ScaledScore::from_raw(v);
        assert_eq!(
            Bound::classify_scaled::<PV>(s(100), s(30), s(50)),
            Bound::Lower
        );
        assert_eq!(
            Bound::classify_scaled::<PV>(s(40), s(30), s(50)),
            Bound::Exact
        );
        assert_eq!(
            Bound::classify_scaled::<PV>(s(20), s(30), s(50)),
            Bound::Upper
        );
        assert_eq!(
            Bound::classify_scaled::<NonPV>(s(100), s(30), s(50)),
            Bound::Lower
        );
        assert_eq!(
            Bound::classify_scaled::<NonPV>(s(40), s(30), s(50)),
            Bound::Upper
        );
        assert_eq!(
            Bound::classify_scaled::<NonPV>(s(20), s(30), s(50)),
            Bound::Upper
        );
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
            s(0),
            Bound::Lower,
            0,
            Square::None,
            Selectivity::Level1,
            0,
            false,
        );
        assert!(entry.read(&board).unwrap().is_occupied());

        // can_cut tests
        entry.save(
            &board,
            s(100),
            Bound::Lower,
            0,
            Square::None,
            Selectivity::Level1,
            0,
            false,
        );
        assert!(entry.read(&board).unwrap().can_cut(s(50)));
        assert!(!entry.read(&board).unwrap().can_cut(s(150)));

        entry.save(
            &board,
            s(30),
            Bound::Upper,
            0,
            Square::None,
            Selectivity::Level1,
            0,
            false,
        );
        assert!(entry.read(&board).unwrap().can_cut(s(50)));
        assert!(!entry.read(&board).unwrap().can_cut(s(20)));

        entry.save(
            &board,
            s(30),
            Bound::Exact,
            0,
            Square::None,
            Selectivity::Level1,
            0,
            false,
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
        store_entry(
            &tt,
            &board,
            ScaledScore::from_raw(100),
            Bound::Exact,
            20,
            10,
            Selectivity::Level1,
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
            ScaledScore::from_raw(0),
            Bound::Exact,
            0,
            Square::None,
            Selectivity::Level1,
            126,
            false,
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

    /// Tests that pack_data does not corrupt the is_endgame flag even with large generation values.
    #[test]
    fn test_pack_data_no_endgame_corruption() {
        let entry = TTEntry::default();
        let board = make_board(42, 84);

        // Store with generation=127, is_endgame=false
        entry.save(
            &board,
            ScaledScore::from_raw(50),
            Bound::Exact,
            10,
            sq(5),
            Selectivity::Level1,
            127,
            false,
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.generation(), 127);
        assert!(!data.is_endgame(), "is_endgame should be false");

        // Store with generation=127, is_endgame=true
        entry.save(
            &board,
            ScaledScore::from_raw(50),
            Bound::Exact,
            10,
            sq(5),
            Selectivity::Level1,
            127,
            true,
        );
        let data = entry.read(&board).unwrap();
        assert_eq!(data.generation(), 127);
        assert!(data.is_endgame(), "is_endgame should be true");

        // Defense-in-depth: even if generation=255 is passed (should be masked to 127)
        entry.save(
            &board,
            ScaledScore::from_raw(50),
            Bound::Exact,
            10,
            sq(5),
            Selectivity::Level1,
            255,
            false,
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
            ScaledScore::from_raw(10),
            Bound::Lower,
            5,
            sq(1),
            Selectivity::Level1,
            0,
            false,
        );
        let v1 = entry.read(&board).unwrap().sequence();
        assert!(v1 & 1 == 0, "sequence should be even after save");

        entry.save(
            &board,
            ScaledScore::from_raw(20),
            Bound::Exact,
            6,
            sq(2),
            Selectivity::Level1,
            0,
            false,
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
            ScaledScore::from_raw(10),
            Bound::Exact,
            5,
            sq(1),
            Selectivity::Level1,
            0,
            false,
        );
        let (_, _, snapshot) = entry.try_load_snapshot().unwrap();

        entry.save(
            &board2,
            ScaledScore::from_raw(20),
            Bound::Exact,
            6,
            sq(2),
            Selectivity::Level2,
            1,
            false,
        );

        assert!(!entry.seqlock_write(
            snapshot.seq,
            1,
            2,
            sq(3) as u8,
            ScaledScore::from_raw(30),
            Bound::Exact as u8,
            7,
            Selectivity::Level3.as_u8(),
            2,
            false,
        ));

        assert!(entry.read(&board1).is_none());
        let data = entry.read(&board2).unwrap();
        assert_eq!(data.score().value(), 20);
        assert_eq!(data.depth(), 6);
        assert_eq!(data.best_move(), sq(2));
        assert_eq!(data.generation(), 1);
    }

}
