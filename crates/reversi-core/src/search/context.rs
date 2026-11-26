//! Search context for maintaining state during game tree search.

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::constants::{INITIAL_EMPTY_COUNT, MAX_PLY};
use crate::eval::Eval;
use crate::eval::EvalMode;
use crate::eval::pattern_feature::{PatternFeature, PatternFeatures};
use crate::probcut::Selectivity;
use crate::score::Score;
use crate::search::correction_history::CorrectionHistory;
use crate::search::counters::{SearchCounters, SearchStats};
use crate::search::midgame::Helper;
use crate::search::root_move::RootMoves;
use crate::search::side_to_move::SideToMove;
use crate::search::stack::SearchStack;
use crate::search::threading::SplitPointTask;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;

/// The search context that maintains all state during search operations.
// The field order is the layout: `n_nodes`, `tt`, `eval`, the three scalars
// after it and `n_empties` share the cache line after
// `pattern_features`. The order rustc picked on its own moved them apart and
// made ST hard-20 1.5% slower.
#[repr(C)]
pub struct SearchContext<'a> {
    /// Pattern features for efficient neural network input
    pub pattern_features: PatternFeatures,
    /// Container for root moves and Multi-PV state
    pub root_moves: &'a RootMoves,
    /// Number of nodes visited by this context.
    pub n_nodes: u64,
    /// Lazy SMP helper thread searching with this context; `None` for the
    /// main thread's search and for split points.
    pub(in crate::search) helper: Option<Helper>,
    /// Corrections applied to main-network evaluations; set only while a
    /// midgame root search or a Lazy SMP helper search runs.
    pub(in crate::search) correction_history: Option<Box<CorrectionHistory>>,
    /// Shared transposition table for storing search results
    pub tt: &'a TranspositionTable,
    /// Neural network evaluator for position assessment
    pub eval: &'a Eval,
    /// Selectivity level
    pub selectivity: Selectivity,
    /// Current side to move
    pub side_to_move: SideToMove,
    /// Current evaluation mode (midgame vs endgame).
    pub eval_mode: EvalMode,
    /// Number of empty squares. The null-window endgame search does not update
    /// it while it runs.
    pub n_empties: u32,
    /// Search stack for maintaining PV and search state at each ply
    pub stack: SearchStack,
    /// Diagnostic counters other than `n_nodes`; last so that the
    /// `search-stats` feature leaves the offsets above unchanged.
    pub(crate) stats: SearchStats,
}

impl<'a> SearchContext<'a> {
    /// Creates a new search context for the given board position.
    ///
    /// # Panics
    ///
    /// Panics if `board` has fewer than four discs.
    pub fn new(
        board: &Board,
        selectivity: Selectivity,
        root_moves: &'a RootMoves,
        tt: &'a TranspositionTable,
        eval: &'a Eval,
    ) -> SearchContext<'a> {
        let n_empties = board.empty_count();
        let ply = INITIAL_EMPTY_COUNT - n_empties as usize;
        SearchContext {
            n_nodes: 0,
            side_to_move: SideToMove::Player,
            selectivity,
            n_empties,
            tt,
            root_moves,
            eval,
            pattern_features: PatternFeatures::new(board, ply),
            stack: SearchStack::new(),
            eval_mode: EvalMode::Main,
            helper: None,
            correction_history: None,
            stats: SearchStats::default(),
        }
    }

    /// Creates a search context from a published split task.
    ///
    /// # Safety
    ///
    /// The task's table, evaluator, and root moves must remain valid for shared
    /// access throughout the lifetime of the resulting context.
    #[inline]
    pub unsafe fn from_split_task(task: &'a SplitPointTask) -> Self {
        let n_empties = task.board.empty_count();
        let ply = INITIAL_EMPTY_COUNT - n_empties as usize;
        let pattern_features =
            PatternFeatures::from_features(ply, &task.p_feature, &task.o_feature);
        let (tt, eval, root_moves) = task.shared_ptrs();
        // SAFETY: The caller keeps the pointees valid for the resulting borrows.
        let (tt, eval, root_moves) = unsafe { (tt.as_ref(), eval.as_ref(), root_moves.as_ref()) };
        SearchContext {
            n_nodes: 0,
            side_to_move: task.side_to_move,
            n_empties,
            selectivity: task.selectivity,
            tt,
            root_moves,
            eval,
            pattern_features,
            stack: SearchStack::new(),
            eval_mode: task.eval_mode,
            helper: None,
            correction_history: None,
            stats: SearchStats::default(),
        }
    }

    /// Switches the side to move in this context.
    #[inline]
    fn switch_players(&mut self) {
        self.side_to_move = self.side_to_move.switch();
    }

    /// Updates the search context after making a move in midgame search.
    ///
    /// `sq` and `flipped` must describe the legal move from the current position.
    ///
    /// # Panics
    ///
    /// Panics if `flipped` is empty or there is no successor pattern slot.
    #[inline]
    pub fn update(&mut self, sq: Square, flipped: Bitboard) {
        self.increment_nodes();
        self.pattern_features
            .update(sq, flipped, self.ply(), self.side_to_move);
        self.switch_players();
        self.n_empties -= 1;
    }

    /// Undoes the most recent midgame move after its descendant updates are undone.
    #[inline]
    pub fn undo(&mut self) {
        self.n_empties += 1;
        self.switch_players();
    }

    /// Updates the context when a pass move is made.
    #[inline]
    pub fn update_pass(&mut self) {
        self.increment_nodes();
        self.switch_players();
    }

    /// Undoes a pass move by switching back the side to move.
    #[inline]
    pub fn undo_pass(&mut self) {
        self.switch_players();
    }

    /// Returns the current ply in the search tree.
    ///
    /// The ply is calculated from the number of empty squares remaining,
    /// representing how far we are from the start of the game.
    #[inline]
    pub fn ply(&self) -> usize {
        INITIAL_EMPTY_COUNT - self.n_empties as usize
    }

    /// Increments the node counter for search statistics.
    #[inline(always)]
    pub fn increment_nodes(&mut self) {
        self.n_nodes += 1;
    }

    /// Returns a snapshot of the search counters.
    pub fn counters(&self) -> SearchCounters {
        self.stats.with_nodes(self.n_nodes)
    }

    /// Adds `other`, counted by another search, to this context's counters.
    pub fn merge_counters(&mut self, other: &SearchCounters) {
        self.n_nodes += other.n_nodes;
        self.stats.merge(other);
    }

    /// Returns the current pattern feature for neural network evaluation.
    #[inline]
    pub fn pattern_feature(&self) -> &PatternFeature {
        let ply = self.ply();
        if self.side_to_move == SideToMove::Player {
            self.pattern_features.p_feature(ply)
        } else {
            self.pattern_features.o_feature(ply)
        }
    }

    /// Computes the eval-cache key for `board` and prefetches its cache line.
    ///
    /// Call before [`update`](Self::update) so the prefetched load overlaps
    /// the pattern-feature SIMD update, then hand the returned key to
    /// [`evaluate_main_with_key`](Self::evaluate_main_with_key) for the same
    /// board.
    #[inline(always)]
    pub fn prefetch_eval_cache(&self, board: &Board) -> u64 {
        let key = board.hash();
        self.eval.prefetch(key);
        key
    }

    /// Evaluates `board` with the main network, probing the eval cache with
    /// `key`, and applies the search's corrections.
    ///
    /// `key` must come from [`prefetch_eval_cache`](Self::prefetch_eval_cache)
    /// for the same board, and the [`update`](Self::update) for the move
    /// reaching `board` must already have been applied.
    #[inline(always)]
    pub fn evaluate_main_with_key(&self, board: &Board, key: u64) -> Score {
        let eval = self
            .eval
            .evaluate_main(self.pattern_feature(), board, self.ply(), key);
        self.correct_eval(board, eval)
    }

    /// Applies the search's corrections to a main-network evaluation of `board`.
    #[inline(always)]
    pub fn correct_eval(&self, board: &Board, eval: Score) -> Score {
        match &self.correction_history {
            Some(history) => history.correct(board, eval),
            None => eval,
        }
    }

    /// Updates a root move with its search results.
    pub fn update_root_move(&mut self, sq: Square, score: Score, is_pv: bool) {
        if is_pv {
            self.update_pv(sq);
        }

        self.root_moves.update(sq, score, is_pv, self.pv());
    }

    /// Updates the principal variation at the current ply.
    #[inline]
    pub fn update_pv(&mut self, sq: Square) {
        self.stack.update_pv(sq, self.ply());
    }

    /// Clears the principal variation at the current ply.
    #[inline]
    pub fn clear_pv(&mut self) {
        self.stack.clear_pv(self.ply());
    }

    /// Clears the PV buffers at the current ply and its child ply.
    #[inline]
    pub fn prepare_pv(&mut self) {
        self.stack.prepare_pv(self.ply());
    }

    /// Returns the principal variation at the current ply.
    #[inline]
    pub fn pv(&self) -> &[Option<Square>; MAX_PLY] {
        self.stack.pv(self.ply())
    }

    /// Sets the principal variation at the current ply.
    #[inline]
    pub fn set_pv(&mut self, pv: &[Option<Square>; MAX_PLY]) {
        self.stack.set_pv(self.ply(), pv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::pattern_feature::NUM_PATTERN_FEATURES;

    #[test]
    fn pattern_selection_matches_rebuild_through_pass_move_and_undo() {
        let root = Board::from_bitboards(
            Square::D1.bitboard(),
            Square::A1.bitboard() | Square::B1.bitboard() | Square::C1.bitboard(),
        );
        let after_pass = root.switch_players();
        let sq = Square::E1;
        assert!(!root.has_legal_moves());
        assert!(after_pass.is_legal_move(sq));
        let flipped = crate::flip::flip(sq, after_pass.player(), after_pass.opponent());
        let after_move = after_pass.make_move_with_flipped(flipped, sq);
        let roots = RootMoves::new(&root);
        let tt = TranspositionTable::new(1);
        let eval = Eval::shared_for_tests();
        let mut ctx = SearchContext::new(&root, Selectivity::None, &roots, &tt, &eval);

        let check = |ctx: &SearchContext<'_>, board: &Board, ply| {
            assert_eq!(ctx.ply(), ply);
            let rebuilt = PatternFeatures::new(board, ply);
            for lane in 0..NUM_PATTERN_FEATURES {
                assert_eq!(ctx.pattern_feature()[lane], rebuilt.p_feature(ply)[lane]);
            }
        };
        check(&ctx, &root, 0);
        ctx.update_pass();
        check(&ctx, &after_pass, 0);
        ctx.update(sq, flipped);
        check(&ctx, &after_move, 1);
        ctx.undo();
        check(&ctx, &after_pass, 0);
        ctx.undo_pass();
        check(&ctx, &root, 0);
    }
}
