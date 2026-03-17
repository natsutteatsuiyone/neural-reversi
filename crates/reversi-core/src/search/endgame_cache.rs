//! Lightweight hash table for endgame search results,
//! separate from the main transposition table.
//!
//! Stores exact NWS results keyed by (board, alpha). Since the endgame
//! NWS is deterministic for a given (position, alpha), entries are valid
//! across selectivity levels without requiring generation tracking.

use crate::board::Board;
use crate::{square::Square, types::Score};

/// Result of probing the endgame cache.
#[derive(Clone, Copy)]
pub struct EndGameCacheProbe {
    /// The exact NWS score, available only when alpha matches.
    pub score: Option<Score>,
    /// The best move from any previous search of this position.
    pub best_move: Square,
}

/// Raw entry stored in the hash table.
///
/// Uses `repr(C, packed)` to achieve 19-byte entries (no padding).
#[derive(Clone, Copy)]
#[repr(C, packed)]
struct RawEntry {
    player: u64,
    opponent: u64,
    alpha: i8,
    score: i8,
    best_move: u8,
}

impl RawEntry {
    const EMPTY: Self = RawEntry {
        player: 0,
        opponent: 0,
        alpha: i8::MIN,
        score: 0,
        best_move: Square::None as u8,
    };
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
        ((key as u128).wrapping_mul(self.table.len() as u128) >> 64) as usize
    }

    /// Probes the cache for a position.
    ///
    /// Returns the exact score when (player, opponent, alpha) all match.
    /// Returns only the best move when the position matches but alpha differs.
    /// Returns `None` when the position doesn't match.
    #[inline(always)]
    pub fn probe(&self, key: u64, board: &Board, alpha: Score) -> Option<EndGameCacheProbe> {
        let idx = self.index(key);
        let entry = self.table[idx];

        if entry.player != board.player.bits() || entry.opponent != board.opponent.bits() {
            return None;
        }

        Some(EndGameCacheProbe {
            score: if entry.alpha == alpha as i8 {
                Some(entry.score as Score)
            } else {
                None
            },
            best_move: Square::from_u8_unchecked(entry.best_move),
        })
    }

    /// Stores an entry, overwriting any existing entry at the same index.
    #[inline(always)]
    pub fn store(
        &mut self,
        key: u64,
        board: &Board,
        alpha: Score,
        score: Score,
        best_move: Square,
    ) {
        let idx = self.index(key);
        self.table[idx] = RawEntry {
            player: board.player.bits(),
            opponent: board.opponent.bits(),
            alpha: alpha as i8,
            score: score as i8,
            best_move: best_move as u8,
        };
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.table.fill(RawEntry::EMPTY);
    }
}

#[cfg(test)]
mod tests {
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
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 12, Square::D3);

        let result = cache.probe(key, &board, 11);
        assert!(result.is_some());
        let probe = result.unwrap();
        assert_eq!(probe.score, Some(12));
        assert_eq!(probe.best_move, Square::D3);
    }

    #[test]
    fn test_probe_different_alpha_returns_best_move_only() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 12, Square::D3);

        // Same position, different alpha → score is None, best_move still available
        let probe = cache.probe(key, &board, 5).unwrap();
        assert_eq!(probe.score, None);
        assert_eq!(probe.best_move, Square::D3);
    }

    #[test]
    fn test_store_and_probe_miss_different_board() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board1 = make_board(0x0000000810000000, 0x0000001008000000);
        let board2 = make_board(0x0000001008000000, 0x0000000810000000);
        let key1 = board1.hash();
        let key2 = board2.hash();

        cache.store(key1, &board1, 11, 12, Square::D3);

        assert!(cache.probe(key2, &board2, 11).is_none());
    }

    #[test]
    fn test_forced_index_collision_overwrites_without_false_hit() {
        let mut cache = EndGameCache::new(std::mem::size_of::<RawEntry>());
        let board1 = make_board(0x0000000810000000, 0x0000001008000000);
        let board2 = make_board(0x00000000000000FF, 0x000000000000FF00);
        let key1 = board1.hash();
        let key2 = board2.hash();

        cache.store(key1, &board1, 11, 12, Square::D3);
        assert!(cache.probe(key2, &board2, -8).is_none());

        cache.store(key2, &board2, -8, -8, Square::A1);

        assert!(cache.probe(key1, &board1, 11).is_none());
        let probe = cache.probe(key2, &board2, -8).unwrap();
        assert_eq!(probe.score, Some(-8));
        assert_eq!(probe.best_move, Square::A1);
    }

    #[test]
    fn test_store_extreme_scores() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0x1234, 0x5678);
        let key = board.hash();

        cache.store(key, &board, 63, 64, Square::H8);
        let probe = cache.probe(key, &board, 63).unwrap();
        assert_eq!(probe.score, Some(64));

        cache.store(key, &board, -65, -64, Square::A1);
        let probe = cache.probe(key, &board, -65).unwrap();
        assert_eq!(probe.score, Some(-64));
    }

    #[test]
    fn test_clear() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 11, 12, Square::D3);
        assert!(cache.probe(key, &board, 11).is_some());

        cache.clear();
        assert!(cache.probe(key, &board, 11).is_none());
    }

    #[test]
    fn test_entry_size() {
        assert_eq!(std::mem::size_of::<RawEntry>(), 19);
    }
}
