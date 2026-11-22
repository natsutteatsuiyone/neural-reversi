//! Lightweight hash table for endgame search bounds,
//! separate from the main transposition table.
//!
//! Stores endgame NWS lower/upper bounds per position.

use std::marker::PhantomData;

use crate::{
    board::Board,
    score::DiscScore,
    score::{DISC_SCORE_MAX, DISC_SCORE_MIN},
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
    const LOWER_UNSET: i8 = (DISC_SCORE_MIN.value() - 1) as i8;
    const UPPER_UNSET: i8 = (DISC_SCORE_MAX.value() + 1) as i8;

    const EMPTY: Self = RawEntry {
        player: 0,
        opponent: 0,
        lower: Self::LOWER_UNSET,
        upper: Self::UPPER_UNSET,
    };

    #[inline(always)]
    fn score_for_alpha(self, alpha: DiscScore) -> Option<DiscScore> {
        let lower = DiscScore::new(self.lower as i32);
        if lower > alpha {
            return Some(lower);
        }

        let upper = DiscScore::new(self.upper as i32);
        if upper <= alpha {
            return Some(upper);
        }

        None
    }

    #[inline(always)]
    fn store_nws_result(&mut self, alpha: DiscScore, score: DiscScore) {
        if score > alpha {
            self.lower = self.lower.max(score.value() as i8);
        } else {
            self.upper = self.upper.min(score.value() as i8);
        }
    }

    #[inline(always)]
    fn overwrite(&mut self, board: &Board, alpha: DiscScore, score: DiscScore) {
        self.player = board.player().bits();
        self.opponent = board.opponent().bits();
        self.lower = Self::LOWER_UNSET;
        self.upper = Self::UPPER_UNSET;
        self.store_nws_result(alpha, score);
    }
}

/// Marks the [`EndGameCache`] of the null window search above the shallow
/// search.
#[derive(Clone, Copy)]
pub struct Deep;

/// Marks the [`EndGameCache`] of the shallow search.
#[derive(Clone, Copy)]
pub struct Shallow;

/// An entry position in an [`EndGameCache`] of kind `K`.
///
/// Only [`EndGameCache::slot`] creates one, so it is in bounds for the cache
/// that issued it, which unsafe code relies on. A slot of one kind cannot go
/// to a cache of the other kind. Which cache of its kind issued it is not
/// tracked: a slot must go back to that cache or to one of the same size.
#[derive(Clone, Copy)]
pub struct CacheSlot<K>(usize, PhantomData<K>);

/// Hash table for caching endgame search results.
///
/// `K` is [`Deep`] or [`Shallow`] and only keeps the slots of the two caches
/// apart.
pub struct EndGameCache<K> {
    table: Box<[RawEntry]>,
    kind: PhantomData<K>,
}

impl<K> EndGameCache<K> {
    /// Creates a new endgame cache with the given memory budget in bytes.
    pub fn new(memory_bytes: usize) -> Self {
        let count = memory_bytes / std::mem::size_of::<RawEntry>();
        assert!(
            count > 0,
            "EndGameCache: memory_bytes ({memory_bytes}) too small for one entry"
        );
        EndGameCache {
            table: vec![RawEntry::EMPTY; count].into_boxed_slice(),
            kind: PhantomData,
        }
    }

    /// Computes the slot for a given hash key.
    #[inline(always)]
    pub fn slot(&self, key: u64) -> CacheSlot<K> {
        CacheSlot(
            crate::util::mul_hi64(key, self.table.len() as u64) as usize,
            PhantomData,
        )
    }

    /// Probes the cache for a position at a precomputed `slot`.
    ///
    /// Returns a cached cutoff score if the stored bounds cut at `alpha`.
    #[inline(always)]
    pub fn probe(&self, slot: CacheSlot<K>, board: &Board, alpha: DiscScore) -> Option<DiscScore> {
        debug_assert!(slot.0 < self.table.len());
        // SAFETY: only `slot` creates a `CacheSlot`, with an index below the
        // `table.len()` of the cache that issued it.
        let entry = unsafe { self.table.get_unchecked(slot.0) };

        if entry.player != board.player().bits() || entry.opponent != board.opponent().bits() {
            return None;
        }

        entry.score_for_alpha(alpha)
    }

    /// Stores or merges an entry at a precomputed `slot`.
    #[inline(always)]
    pub fn store(&mut self, slot: CacheSlot<K>, board: &Board, alpha: DiscScore, score: DiscScore) {
        debug_assert!(slot.0 < self.table.len());
        // SAFETY: as in `probe`.
        let entry = unsafe { self.table.get_unchecked_mut(slot.0) };
        if entry.player == board.player().bits() && entry.opponent == board.opponent().bits() {
            entry.store_nws_result(alpha, score);
        } else {
            entry.overwrite(board, alpha, score);
        }
    }

    /// Clears all entries.
    #[cfg(test)]
    pub fn clear(&mut self) {
        self.table.fill(RawEntry::EMPTY);
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;
    fn make_board(player: u64, opponent: u64) -> Board {
        Board::from_bitboards(player, opponent)
    }

    fn exhaustive_value(board: Board) -> DiscScore {
        fn alphabeta(board: Board, mut alpha: i32, beta: i32) -> i32 {
            let moves = board.legal_moves();
            if moves.is_empty() {
                let next = board.switch_players();
                return if next.has_legal_moves() {
                    -alphabeta(next, -beta, -alpha)
                } else {
                    board.solve_disc_diff(board.empty_count()).value()
                };
            }
            let mut best = -65;
            for sq in moves.iter() {
                let score = -alphabeta(board.make_move(sq), -beta, -alpha);
                if score > best {
                    best = score;
                    if best >= beta {
                        break;
                    }
                    alpha = alpha.max(best);
                }
            }
            best
        }

        DiscScore::new(alphabeta(board, -65, 65))
    }

    fn assert_stored_bounds<K>(
        cache: &EndGameCache<K>,
        exact_cache: &mut std::collections::HashMap<Board, DiscScore>,
    ) -> bool {
        let mut found = false;
        for entry in &cache.table {
            let (lower, upper) = (entry.lower, entry.upper);
            if (lower, upper) != (RawEntry::LOWER_UNSET, RawEntry::UPPER_UNSET) {
                let board = make_board(entry.player, entry.opponent);
                let exact = exact_cache
                    .entry(board)
                    .or_insert_with(|| exhaustive_value(board))
                    .value();
                assert!(
                    i32::from(lower) <= exact && exact <= i32::from(upper),
                    "stored interval [{lower}, {upper}] excludes disc value {exact}"
                );
                found = true;
            }
        }
        found
    }

    #[test]
    fn recursive_search_preserves_bounds_in_both_caches() {
        use super::super::{EndGameCaches, IgnoreSplitPointCutoff, null_window_search};
        use crate::eval::Eval;
        use crate::probcut::Selectivity;
        use crate::search::context::SearchContext;
        use crate::search::root_move::RootMoves;
        use crate::transposition_table::TranspositionTable;

        let mut caches = EndGameCaches {
            ec: EndGameCache::new(2 * size_of::<RawEntry>()),
            shallow: EndGameCache::new(2 * size_of::<RawEntry>()),
        };
        let eval = Eval::shared_for_tests();
        let tt = TranspositionTable::new(0);
        let mut exact_cache = std::collections::HashMap::new();
        let mut saw_deep = false;
        let mut saw_shallow = false;
        for target in [12, 7] {
            let mut board = Board::new();
            while board.empty_count() > target {
                let moves = board.legal_moves();
                if moves.is_empty() {
                    board = board.switch_players();
                    assert!(board.has_legal_moves());
                } else {
                    let index = (13 * board.empty_count() % moves.count()) as usize;
                    board = board.make_move(moves.iter().nth(index).unwrap());
                }
            }
            let exact = *exact_cache
                .entry(board)
                .or_insert_with(|| exhaustive_value(board));
            let roots = RootMoves::new(&board);
            let mut ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
            for alpha in [
                exact,
                exact - 1,
                DiscScore::new(-128),
                DiscScore::new(127),
                exact,
            ] {
                let result = null_window_search(
                    &mut ctx,
                    &board,
                    alpha,
                    &mut caches,
                    &IgnoreSplitPointCutoff,
                    board.legal_moves(),
                );
                assert!(if result <= alpha {
                    exact <= result
                } else {
                    result <= exact
                });
                assert_eq!(ctx.empty_list.count(), target);
                saw_deep |= assert_stored_bounds(&caches.ec, &mut exact_cache);
                saw_shallow |= assert_stored_bounds(&caches.shallow, &mut exact_cache);
            }
        }
        assert!(saw_deep && saw_shallow);
    }

    #[test]
    fn test_store_and_probe_hit() {
        let mut cache = EndGameCache::<Deep>::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let cache_idx = cache.slot(board.hash());

        cache.store(cache_idx, &board, DiscScore::new(11), DiscScore::new(12));

        let result = cache.probe(cache_idx, &board, DiscScore::new(11));
        assert_eq!(result, Some(DiscScore::new(12)));
    }

    #[test]
    fn test_probe_no_cutoff_returns_none() {
        let mut cache = EndGameCache::<Deep>::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let cache_idx = cache.slot(board.hash());

        cache.store(cache_idx, &board, DiscScore::new(11), DiscScore::new(12));

        // Lower bound from alpha=11 does not cut at alpha=12.
        assert_eq!(cache.probe(cache_idx, &board, DiscScore::new(12)), None);
    }

    #[test]
    fn test_probe_reuses_lower_bound_for_wider_window() {
        let mut cache = EndGameCache::<Deep>::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let cache_idx = cache.slot(board.hash());

        cache.store(cache_idx, &board, DiscScore::new(11), DiscScore::new(12));

        assert_eq!(
            cache.probe(cache_idx, &board, DiscScore::new(5)),
            Some(DiscScore::new(12))
        );
    }

    #[test]
    fn test_probe_reuses_upper_bound_for_higher_alpha() {
        let mut cache = EndGameCache::<Deep>::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let cache_idx = cache.slot(board.hash());

        cache.store(cache_idx, &board, DiscScore::new(11), DiscScore::new(8));

        assert_eq!(
            cache.probe(cache_idx, &board, DiscScore::new(12)),
            Some(DiscScore::new(8))
        );
    }

    #[test]
    fn test_store_merges_bounds_for_same_board() {
        let mut cache = EndGameCache::<Deep>::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x0000000810000000, 0x0000001008000000);
        let cache_idx = cache.slot(board.hash());

        cache.store(cache_idx, &board, DiscScore::new(8), DiscScore::new(11));
        cache.store(cache_idx, &board, DiscScore::new(11), DiscScore::new(11));

        assert_eq!(
            cache.probe(cache_idx, &board, DiscScore::new(10)),
            Some(DiscScore::new(11))
        );
        assert_eq!(
            cache.probe(cache_idx, &board, DiscScore::new(11)),
            Some(DiscScore::new(11))
        );
    }

    #[test]
    fn test_forced_index_collision_overwrites_without_false_hit() {
        let mut cache = EndGameCache::<Deep>::new(std::mem::size_of::<RawEntry>());
        let board1 = make_board(0x0000000810000000, 0x0000001008000000);
        let board2 = make_board(0x00000000000000FF, 0x000000000000FF00);
        let cache_idx1 = cache.slot(board1.hash());
        let cache_idx2 = cache.slot(board2.hash());

        cache.store(cache_idx1, &board1, DiscScore::new(11), DiscScore::new(12));
        assert!(
            cache
                .probe(cache_idx2, &board2, DiscScore::new(-8))
                .is_none()
        );

        cache.store(cache_idx2, &board2, DiscScore::new(-8), DiscScore::new(-8));

        assert!(
            cache
                .probe(cache_idx1, &board1, DiscScore::new(11))
                .is_none()
        );
        assert_eq!(
            cache.probe(cache_idx2, &board2, DiscScore::new(-8)),
            Some(DiscScore::new(-8))
        );
    }

    #[test]
    fn unset_endpoints_are_cutoffs_only_outside_the_disc_range() {
        let mut cache = EndGameCache::<Deep>::new(size_of::<RawEntry>());
        let board = make_board(0x0000_0000_ffff_ffff, 0xffff_ffff_0000_0000);
        let slot = cache.slot(board.hash());
        let lower = DiscScore::new(-65);
        let upper = DiscScore::new(65);

        cache.store(slot, &board, DiscScore::new(1), DiscScore::new(0));
        assert_eq!(cache.probe(slot, &board, DiscScore::new(-66)), Some(lower));
        assert_eq!(cache.probe(slot, &board, lower), None);
        cache.store(slot, &board, DiscScore::new(-66), lower);
        assert_eq!(cache.probe(slot, &board, lower), None);

        let other = board.switch_players();
        cache.store(slot, &other, DiscScore::new(-1), DiscScore::new(0));
        assert_eq!(cache.probe(slot, &board, DiscScore::new(-66)), None);
        assert_eq!(cache.probe(slot, &other, DiscScore::new(64)), None);
        assert_eq!(cache.probe(slot, &other, upper), Some(upper));
        cache.store(slot, &other, upper, upper);
        assert_eq!(cache.probe(slot, &other, DiscScore::new(64)), None);
    }

    #[test]
    fn test_store_extreme_scores() {
        let mut cache = EndGameCache::<Deep>::new((1 << 14) * size_of::<RawEntry>());
        let board = make_board(0x000f, 0x00f0);
        let cache_idx = cache.slot(board.hash());

        cache.store(cache_idx, &board, DiscScore::new(63), DiscScore::new(64));
        assert_eq!(
            cache.probe(cache_idx, &board, DiscScore::new(63)),
            Some(DiscScore::new(64))
        );

        cache.clear();
        cache.store(cache_idx, &board, DiscScore::new(-65), DiscScore::new(-64));
        assert_eq!(
            cache.probe(cache_idx, &board, DiscScore::new(-65)),
            Some(DiscScore::new(-64))
        );
    }
}
