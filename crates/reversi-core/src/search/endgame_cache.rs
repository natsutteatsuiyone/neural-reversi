//! Lightweight hash table for endgame search results,
//! separate from the main transposition table.

use crate::board::Board;
use crate::{square::Square, types::Score};

/// Bound type for endgame cache entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EndGameCacheBound {
    /// Score is a lower bound (true value >= score, from fail-high).
    Lower = 0,
    /// Score is an upper bound (true value <= score, from fail-low).
    Upper = 1,
}

impl EndGameCacheBound {
    /// Classifies the score into a bound type.
    #[inline(always)]
    fn classify(score: Score, beta: Score) -> Self {
        if score < beta {
            EndGameCacheBound::Upper
        } else {
            EndGameCacheBound::Lower
        }
    }
}

/// Single entry returned from cache lookups.
#[derive(Clone, Copy)]
pub struct EndGameCacheEntry {
    pub score: Score,
    pub best_move: Square,
    pub bound: EndGameCacheBound,
}

impl EndGameCacheEntry {
    /// Returns the normalized cutoff score if this entry produces a cutoff for the given beta.
    #[inline(always)]
    pub fn try_cut(&self, beta: Score) -> Option<Score> {
        match self.bound {
            EndGameCacheBound::Lower => {
                if self.score >= beta {
                    Some(beta)
                } else {
                    None
                }
            }
            EndGameCacheBound::Upper => {
                if self.score < beta {
                    Some(beta - 1)
                } else {
                    None
                }
            }
        }
    }
}

/// Raw entry stored in the hash table.
///
/// Uses `repr(C, packed)` to achieve 18-byte entries (no padding).
///
/// `move_bound` packing (u8):
/// - bit  [7]:   bound (0=Lower, 1=Upper)
/// - bits [6:0]: best_move (0-64)
#[derive(Clone, Copy)]
#[repr(C, packed)]
struct RawEntry {
    player: u64,
    opponent: u64,
    score: i8,
    move_bound: u8,
}

impl RawEntry {
    const EMPTY: Self = RawEntry {
        player: 0,
        opponent: 0,
        score: 0,
        move_bound: Self::pack_move_bound(EndGameCacheBound::Lower, Square::None as u8),
    };

    #[inline(always)]
    const fn pack_move_bound(bound: EndGameCacheBound, best_move: u8) -> u8 {
        ((bound as u8) << 7) | (best_move & 0x7F)
    }

    #[inline(always)]
    fn bound(self) -> EndGameCacheBound {
        if self.move_bound >> 7 == 0 {
            EndGameCacheBound::Lower
        } else {
            EndGameCacheBound::Upper
        }
    }

    #[inline(always)]
    fn best_move(self) -> Square {
        Square::from_u8_unchecked(self.move_bound & 0x7F)
    }

    #[inline(always)]
    fn unpack(self) -> EndGameCacheEntry {
        EndGameCacheEntry {
            score: self.score as Score,
            best_move: self.best_move(),
            bound: self.bound(),
        }
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
        ((key as u128).wrapping_mul(self.table.len() as u128) >> 64) as usize
    }

    /// Looks up an entry by key, verifying the stored board matches.
    #[inline(always)]
    pub fn probe(&self, key: u64, board: &Board) -> Option<EndGameCacheEntry> {
        let idx = self.index(key);
        let entry = unsafe { self.table.as_ptr().add(idx).read_unaligned() };

        if entry.player != board.player.bits() || entry.opponent != board.opponent.bits() {
            return None;
        }

        Some(entry.unpack())
    }

    /// Stores an entry, overwriting any existing entry at the same index.
    #[inline(always)]
    pub fn store(&mut self, key: u64, board: &Board, score: Score, beta: Score, best_move: Square) {
        debug_assert!(
            (-128..=127).contains(&score),
            "score {score} out of i8 range"
        );
        let bound = EndGameCacheBound::classify(score, beta);
        let idx = self.index(key);
        unsafe {
            self.table.as_mut_ptr().add(idx).write_unaligned(RawEntry {
                player: board.player.bits(),
                opponent: board.opponent.bits(),
                score: score as i8,
                move_bound: RawEntry::pack_move_bound(bound, best_move as u8),
            });
        }
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
    fn test_try_cut_lower_bound() {
        let entry = EndGameCacheEntry {
            score: 10,
            best_move: Square::None,
            bound: EndGameCacheBound::Lower,
        };
        assert_eq!(entry.try_cut(5), Some(5));
        assert_eq!(entry.try_cut(10), Some(10));
        assert_eq!(entry.try_cut(15), None);
    }

    #[test]
    fn test_try_cut_upper_bound() {
        let entry = EndGameCacheEntry {
            score: 10,
            best_move: Square::None,
            bound: EndGameCacheBound::Upper,
        };
        assert_eq!(entry.try_cut(15), Some(14));
        assert_eq!(entry.try_cut(11), Some(10));
        assert_eq!(entry.try_cut(10), None);
        assert_eq!(entry.try_cut(5), None);
    }

    #[test]
    fn test_try_cut_negative_scores() {
        let entry = EndGameCacheEntry {
            score: -20,
            best_move: Square::None,
            bound: EndGameCacheBound::Lower,
        };
        assert_eq!(entry.try_cut(-25), Some(-25));
        assert_eq!(entry.try_cut(-20), Some(-20));
        assert_eq!(entry.try_cut(-10), None);

        let entry = EndGameCacheEntry {
            score: -20,
            best_move: Square::None,
            bound: EndGameCacheBound::Upper,
        };
        assert_eq!(entry.try_cut(-10), Some(-11));
        assert_eq!(entry.try_cut(-19), Some(-20));
        assert_eq!(entry.try_cut(-20), None);
        assert_eq!(entry.try_cut(-30), None);
    }

    #[test]
    fn test_try_cut_extreme_scores() {
        let entry = EndGameCacheEntry {
            score: 64,
            best_move: Square::None,
            bound: EndGameCacheBound::Lower,
        };
        assert_eq!(entry.try_cut(64), Some(64));
        assert_eq!(entry.try_cut(65), None);

        let entry = EndGameCacheEntry {
            score: -64,
            best_move: Square::None,
            bound: EndGameCacheBound::Upper,
        };
        assert_eq!(entry.try_cut(-63), Some(-64));
        assert_eq!(entry.try_cut(-64), None);
    }

    #[test]
    fn test_bound_classify() {
        assert_eq!(EndGameCacheBound::classify(5, 10), EndGameCacheBound::Upper);
        assert_eq!(
            EndGameCacheBound::classify(10, 10),
            EndGameCacheBound::Lower
        );
        assert_eq!(
            EndGameCacheBound::classify(15, 10),
            EndGameCacheBound::Lower
        );
    }

    #[test]
    fn test_store_and_probe_hit() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 12, 12, Square::D3);

        let result = cache.probe(key, &board);
        assert!(result.is_some());
        let entry = result.unwrap();
        assert_eq!(entry.score, 12);
        assert_eq!(entry.bound, EndGameCacheBound::Lower);
        assert_eq!(entry.best_move, Square::D3);
    }

    #[test]
    fn test_store_and_probe_miss() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board1 = make_board(0x0000000810000000, 0x0000001008000000);
        let board2 = make_board(0x0000001008000000, 0x0000000810000000);
        let key1 = board1.hash();
        let key2 = board2.hash();

        cache.store(key1, &board1, 12, 12, Square::D3);

        assert!(cache.probe(key2, &board2).is_none());
    }

    #[test]
    fn test_forced_index_collision_overwrites_without_false_hit() {
        let mut cache = EndGameCache::new(std::mem::size_of::<RawEntry>());
        let board1 = make_board(0x0000000810000000, 0x0000001008000000);
        let board2 = make_board(0x00000000000000FF, 0x000000000000FF00);
        let key1 = board1.hash();
        let key2 = board2.hash();

        cache.store(key1, &board1, 12, 12, Square::D3);
        assert!(cache.probe(key2, &board2).is_none());

        cache.store(key2, &board2, -8, -7, Square::A1);

        assert!(cache.probe(key1, &board1).is_none());
        let entry = cache.probe(key2, &board2).unwrap();
        assert_eq!(entry.score, -8);
        assert_eq!(entry.bound, EndGameCacheBound::Upper);
        assert_eq!(entry.best_move, Square::A1);
    }

    #[test]
    fn test_store_negative_score() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0xFF, 0xAA);
        let key = board.hash();

        cache.store(key, &board, -30, -29, Square::A1);

        let entry = cache.probe(key, &board).unwrap();
        assert_eq!(entry.score, -30);
        assert_eq!(entry.bound, EndGameCacheBound::Upper);
        assert_eq!(entry.best_move, Square::A1);
    }

    #[test]
    fn test_store_extreme_scores() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0x1234, 0x5678);
        let key = board.hash();

        cache.store(key, &board, 64, 64, Square::H8);
        let entry = cache.probe(key, &board).unwrap();
        assert_eq!(entry.score, 64);

        cache.store(key, &board, -64, -63, Square::A1);
        let entry = cache.probe(key, &board).unwrap();
        assert_eq!(entry.score, -64);
    }

    #[test]
    fn test_clear() {
        let mut cache = EndGameCache::new((1 << 14) * 24);
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let key = board.hash();

        cache.store(key, &board, 12, 12, Square::D3);
        assert!(cache.probe(key, &board).is_some());

        cache.clear();
        assert!(cache.probe(key, &board).is_none());
    }

    #[test]
    fn test_entry_size() {
        assert_eq!(std::mem::size_of::<RawEntry>(), 18);
    }
}
