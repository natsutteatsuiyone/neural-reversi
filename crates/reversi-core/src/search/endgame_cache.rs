//! Lightweight hash table for endgame search bounds,
//! separate from the main transposition table.
//!
//! Stores endgame NWS lower/upper bounds per position.

use crate::{
    board::Board,
    constants::{SCORE_MAX, SCORE_MIN},
    types::Score,
};

/// Raw cache entry.
#[derive(Clone, Copy)]
#[repr(C, packed)]
struct RawEntry {
    player: u64,
    opponent: u64,
    lower: i8,
    upper: i8,
}

impl RawEntry {
    const LOWER_UNSET: i8 = (SCORE_MIN - 1) as i8;
    const UPPER_UNSET: i8 = (SCORE_MAX + 1) as i8;

    const EMPTY: Self = RawEntry {
        player: 0,
        opponent: 0,
        lower: Self::LOWER_UNSET,
        upper: Self::UPPER_UNSET,
    };

    #[inline(always)]
    fn score_for_alpha(self, alpha: Score) -> Option<Score> {
        let lower = self.lower as Score;
        if lower > alpha {
            return Some(lower);
        }

        let upper = self.upper as Score;
        if upper <= alpha {
            return Some(upper);
        }

        None
    }

    #[inline(always)]
    fn store_nws_result(&mut self, alpha: Score, score: Score) {
        if score > alpha {
            self.lower = self.lower.max(score as i8);
        } else {
            self.upper = self.upper.min(score as i8);
        }
    }

    #[inline(always)]
    fn overwrite(&mut self, board: &Board, alpha: Score, score: Score) {
        self.player = board.player.bits();
        self.opponent = board.opponent.bits();
        self.lower = Self::LOWER_UNSET;
        self.upper = Self::UPPER_UNSET;
        self.store_nws_result(alpha, score);
    }
}

/// Hash table for caching endgame search results.
pub struct EndGameCache {
    table: Box<[RawEntry]>,
}

impl EndGameCache {
    /// Creates a new endgame cache with the given memory budget in bytes.
    pub fn new(memory_bytes: usize) -> Self {
        let count = memory_bytes / std::mem::size_of::<RawEntry>();
        assert!(
            count > 0,
            "EndGameCache: memory_bytes ({memory_bytes}) too small for one entry"
        );
        EndGameCache {
            table: vec![RawEntry::EMPTY; count].into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn index(&self, key: u64) -> usize {
        crate::util::mul_hi64(key, self.table.len() as u64) as usize
    }

    /// Probes the cache for a position.
    ///
    /// Returns a cached cutoff score if the stored bounds cut at `alpha`.
    #[inline(always)]
    pub fn probe(&self, key: u64, board: &Board, alpha: Score) -> Option<Score> {
        let idx = self.index(key);
        let entry = self.table[idx];

        if entry.player != board.player.bits() || entry.opponent != board.opponent.bits() {
            return None;
        }

        entry.score_for_alpha(alpha)
    }

    /// Stores or merges an entry.
    #[inline(always)]
    pub fn store(&mut self, key: u64, board: &Board, alpha: Score, score: Score) {
        let idx = self.index(key);
        let entry = &mut self.table[idx];
        if entry.player == board.player.bits() && entry.opponent == board.opponent.bits() {
            entry.store_nws_result(alpha, score);
        } else {
            entry.overwrite(board, alpha, score);
        }
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.table.fill(RawEntry::EMPTY);
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;
    use crate::bitboard::Bitboard;

    fn make_board(player: u64, opponent: u64) -> Board {
        Board {
            player: Bitboard::new(player),
            opponent: Bitboard::new(opponent),
        }
    }

    #[test]
    fn test_store_and_probe_hit() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 12);

        let result = cache.probe(key, &board, 11);
        assert_eq!(result, Some(12));
    }

    #[test]
    fn test_probe_no_cutoff_returns_none() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 12);

        // Lower bound from alpha=11 does not cut at alpha=12.
        assert_eq!(cache.probe(key, &board, 12), None);
    }

    #[test]
    fn test_probe_reuses_lower_bound_for_wider_window() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 12);

        assert_eq!(cache.probe(key, &board, 5), Some(12));
    }

    #[test]
    fn test_probe_reuses_upper_bound_for_higher_alpha() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 8);

        assert_eq!(cache.probe(key, &board, 12), Some(8));
    }

    #[test]
    fn test_store_merges_bounds_for_same_board() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 8, 11);
        cache.store(key, &board, 11, 11);

        assert_eq!(cache.probe(key, &board, 10), Some(11));
        assert_eq!(cache.probe(key, &board, 11), Some(11));
    }

    #[test]
    fn test_store_and_probe_miss_different_board() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board1 = make_board(0x0000000810000000, 0x0000001008000000);
        let board2 = make_board(0x0000001008000000, 0x0000000810000000);
        let key1 = board1.hash();
        let key2 = board2.hash();

        cache.store(key1, &board1, 11, 12);

        assert!(cache.probe(key2, &board2, 11).is_none());
    }

    #[test]
    fn test_forced_index_collision_overwrites_without_false_hit() {
        let mut cache = EndGameCache::new(std::mem::size_of::<RawEntry>());
        let board1 = make_board(0x0000000810000000, 0x0000001008000000);
        let board2 = make_board(0x00000000000000FF, 0x000000000000FF00);
        let key1 = board1.hash();
        let key2 = board2.hash();

        cache.store(key1, &board1, 11, 12);
        assert!(cache.probe(key2, &board2, -8).is_none());

        cache.store(key2, &board2, -8, -8);

        assert!(cache.probe(key1, &board1, 11).is_none());
        assert_eq!(cache.probe(key2, &board2, -8), Some(-8));
    }

    #[test]
    fn test_store_extreme_scores() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x1234, 0x5678);
        let key = board.hash();

        cache.store(key, &board, 63, 64);
        assert_eq!(cache.probe(key, &board, 63), Some(64));

        cache.clear();
        cache.store(key, &board, -65, -64);
        assert_eq!(cache.probe(key, &board, -65), Some(-64));
    }

    #[test]
    fn test_clear() {
        let mut cache = EndGameCache::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 12);
        assert!(cache.probe(key, &board, 11).is_some());

        cache.clear();
        assert!(cache.probe(key, &board, 11).is_none());
    }

    #[test]
    fn test_entry_size() {
        assert_eq!(std::mem::size_of::<RawEntry>(), 18);
    }
}
