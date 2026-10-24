//! Search result types.

use crate::{
    board::Board,
    eval::{Eval, EvalMode},
    probcut::Selectivity,
    search::context::SearchContext,
    search::counters::SearchCounters,
    search::root_move::RootMove,
    square::Square,
    transposition_table::TranspositionTable,
    types::{Depth, ScaledScore, Scoref},
};

/// A single move with its evaluation score for Multi-PV results.
#[derive(Clone, Debug)]
pub struct PvMove {
    pub sq: Square,
    pub score: Scoref,
    pub pv_line: Vec<Square>,
}

/// Result of a search operation.
pub enum SearchResult {
    /// Search completed with a playable move.
    BestMove {
        sq: Square,
        score: Scoref,
        n_nodes: u64,
        pv_line: Vec<Square>,
        depth: Depth,
        selectivity: Selectivity,
        is_endgame: bool,
        /// All evaluated moves with scores (populated in Multi-PV mode).
        pv_moves: Vec<PvMove>,
        /// Diagnostic counters accumulated during search.
        counters: SearchCounters,
    },
    /// No legal root move is available.
    NoLegalMove,
}

/// Snapshot of a completed root iteration.
pub(super) struct Iteration {
    pub(super) root_moves: Vec<RootMove>,
    pub(super) depth: Depth,
    pub(super) selectivity: Selectivity,
    pub(super) is_endgame: bool,
}

impl Iteration {
    /// Captures a copy of the root moves of `ctx`, sorted by score, as an
    /// iteration of `depth` at the context's selectivity.
    pub(super) fn capture(ctx: &SearchContext, depth: Depth, is_endgame: bool) -> Self {
        let mut root_moves = ctx.root_moves.snapshot();
        root_moves.sort_by_key(|rm| std::cmp::Reverse(rm.score));
        Iteration {
            root_moves,
            depth,
            selectivity: ctx.selectivity,
            is_endgame,
        }
    }

    /// Returns the best root move.
    pub(super) fn best_move(&self) -> &RootMove {
        self.root_moves
            .first()
            .expect("internal error: no completed root moves after search")
    }
}

/// The result a root search reports when it stops: the most recently
/// completed iteration, or the TT move or [`quick_move`] when none completed.
///
/// An iteration completes once every PV line has been searched at the same
/// depth and selectivity; aborted iterations leave root-move scores mixed
/// across levels and must not be reported as reached.
pub(super) struct CompletedState {
    lines: Vec<Option<(Depth, Selectivity)>>,
    pub(super) last: Option<Iteration>,
}

impl CompletedState {
    /// Creates a state with no completed iteration for `pv_count` PV lines.
    pub(super) fn new(pv_count: usize) -> Self {
        CompletedState {
            lines: vec![None; pv_count],
            last: None,
        }
    }

    /// Records that the current PV line of `ctx` finished `depth` at the
    /// context's selectivity. Returns the best move if this completed the
    /// iteration.
    pub(super) fn line_completed(
        &mut self,
        ctx: &SearchContext,
        depth: Depth,
        is_endgame: bool,
    ) -> Option<&RootMove> {
        let level = Some((depth, ctx.selectivity));
        self.lines[ctx.root_moves.pv_idx()] = level;
        if self.lines.iter().all(|&line| line == level) {
            Some(self.iteration_completed(ctx, depth, is_endgame))
        } else {
            None
        }
    }

    /// Reports the root moves of `ctx` as completed, whatever levels its
    /// PV lines reached, and returns the best move.
    pub(super) fn iteration_completed(
        &mut self,
        ctx: &SearchContext,
        depth: Depth,
        is_endgame: bool,
    ) -> &RootMove {
        self.last
            .insert(Iteration::capture(ctx, depth, is_endgame))
            .best_move()
    }

    /// Builds the reported [`SearchResult`] for the root `board` of `ctx`.
    pub(super) fn into_result(self, ctx: &SearchContext, board: &Board) -> SearchResult {
        match self.last {
            Some(it) => SearchResult::from_root_move_snapshot(
                &it.root_moves,
                it.best_move(),
                it.depth,
                it.selectivity,
                it.is_endgame,
                ctx.counters.clone(),
            ),
            None => tt_move(board, &ctx.tt).unwrap_or_else(|| quick_move(board, &ctx.eval)),
        }
    }
}

/// Returns the TT's best move for `board` as a result, if it is legal.
fn tt_move(board: &Board, tt: &TranspositionTable) -> Option<SearchResult> {
    let data = tt.lookup(board, board.hash())?;
    let sq = data.best_move();
    // `Square::None` must be rejected before the legality check.
    (sq != Square::None && board.is_legal_move(sq)).then(|| SearchResult::BestMove {
        sq,
        score: data.score().to_disc_diff_f32(),
        n_nodes: 0,
        pv_line: vec![sq],
        depth: data.depth(),
        selectivity: data.selectivity(),
        is_endgame: data.is_endgame(),
        pv_moves: vec![],
        counters: SearchCounters::default(),
    })
}

/// Selects a move by a shallow 1-ply search, for when there is no time or
/// no completed iteration to report.
pub(super) fn quick_move(board: &Board, eval: &Eval) -> SearchResult {
    let moves = board.get_moves();
    if moves.is_empty() {
        return SearchResult::NoLegalMove;
    }

    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;

    for sq in moves.iter() {
        let next = board.make_move(sq);
        let score = -eval.evaluate_simple(&next, EvalMode::Main);

        if score > best_score {
            best_score = score;
            best_move = sq;
        }
    }

    SearchResult::BestMove {
        sq: best_move,
        score: best_score.to_disc_diff_f32(),
        n_nodes: moves.count() as u64,
        pv_line: vec![best_move],
        depth: 1,
        selectivity: Selectivity::None,
        is_endgame: false,
        pv_moves: vec![],
        counters: SearchCounters::default(),
    }
}

impl SearchResult {
    /// Creates a result for a random move in the opening position.
    pub fn new_random_move(mv: Square) -> Self {
        Self::BestMove {
            sq: mv,
            score: 0.0,
            n_nodes: 0,
            pv_line: vec![],
            depth: 0,
            selectivity: Selectivity::None,
            is_endgame: false,
            pv_moves: vec![],
            counters: SearchCounters::default(),
        }
    }

    pub(crate) fn from_root_move_snapshot(
        root_moves: &[RootMove],
        best_move: &RootMove,
        depth: Depth,
        selectivity: Selectivity,
        is_endgame: bool,
        counters: SearchCounters,
    ) -> Self {
        let pv_moves: Vec<PvMove> = root_moves
            .iter()
            .map(|rm| PvMove {
                sq: rm.sq,
                score: rm.score.to_disc_diff_f32(),
                pv_line: rm.pv.clone(),
            })
            .collect();

        Self::BestMove {
            sq: best_move.sq,
            score: best_move.score.to_disc_diff_f32(),
            n_nodes: counters.n_nodes,
            pv_line: best_move.pv.clone(),
            depth,
            selectivity,
            is_endgame,
            pv_moves,
            counters,
        }
    }

    /// Returns the best move square, if the search produced one.
    #[inline]
    pub fn best_move(&self) -> Option<Square> {
        match self {
            SearchResult::BestMove { sq, .. } => Some(*sq),
            SearchResult::NoLegalMove => None,
        }
    }

    /// Returns the best move score, if the search produced one.
    #[inline]
    pub fn score(&self) -> Option<Scoref> {
        match self {
            SearchResult::BestMove { score, .. } => Some(*score),
            SearchResult::NoLegalMove => None,
        }
    }

    /// Returns the searched node count.
    #[inline]
    pub fn n_nodes(&self) -> u64 {
        match self {
            SearchResult::BestMove { n_nodes, .. } => *n_nodes,
            SearchResult::NoLegalMove => 0,
        }
    }

    /// Returns the principal variation line.
    #[inline]
    pub fn pv_line(&self) -> &[Square] {
        match self {
            SearchResult::BestMove { pv_line, .. } => pv_line,
            SearchResult::NoLegalMove => &[],
        }
    }

    /// Returns the completed search depth.
    #[inline]
    pub fn depth(&self) -> Depth {
        match self {
            SearchResult::BestMove { depth, .. } => *depth,
            SearchResult::NoLegalMove => 0,
        }
    }

    /// Returns the selectivity used by the result.
    #[inline]
    pub fn selectivity(&self) -> Selectivity {
        match self {
            SearchResult::BestMove { selectivity, .. } => *selectivity,
            SearchResult::NoLegalMove => Selectivity::None,
        }
    }

    /// Returns whether this result came from endgame search.
    #[inline]
    pub fn is_endgame(&self) -> bool {
        match self {
            SearchResult::BestMove { is_endgame, .. } => *is_endgame,
            SearchResult::NoLegalMove => false,
        }
    }

    /// Returns all evaluated Multi-PV moves.
    #[inline]
    pub fn pv_moves(&self) -> &[PvMove] {
        match self {
            SearchResult::BestMove { pv_moves, .. } => pv_moves,
            SearchResult::NoLegalMove => &[],
        }
    }

    /// Returns diagnostic counters accumulated during search.
    #[inline]
    pub fn counters(&self) -> SearchCounters {
        match self {
            SearchResult::BestMove { counters, .. } => counters.clone(),
            SearchResult::NoLegalMove => SearchCounters::default(),
        }
    }

    /// Returns the probability percentage based on selectivity.
    #[inline]
    pub fn probability(&self) -> i32 {
        self.selectivity().probability()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::constants::MAX_PLY;
    use crate::transposition_table::Bound;

    fn context(board: &Board) -> SearchContext {
        let eval =
            Eval::with_weight_files(None, None).expect("embedded evaluation weights must load");
        SearchContext::new(
            board,
            Selectivity::None,
            Arc::new(TranspositionTable::new(0)),
            Arc::new(eval),
        )
    }

    #[test]
    fn without_a_completed_iteration_reports_the_tt_move_then_the_quick_move() {
        let board = Board::new();
        let ctx = context(&board);
        let quick = quick_move(&board, &ctx.eval).best_move().unwrap();
        let tt_sq = board.get_moves().iter().find(|&sq| sq != quick).unwrap();
        let probe = ctx.tt.probe(&board, board.hash());
        ctx.tt.store(
            probe.index(),
            &board,
            ScaledScore::from_disc_diff(3),
            Bound::Exact,
            20,
            tt_sq,
            Selectivity::None,
            false,
        );

        let result = CompletedState::new(1).into_result(&ctx, &board);
        assert_eq!(result.best_move(), Some(tt_sq));
        assert_eq!(result.depth(), 20);

        ctx.tt.clear();
        let result = CompletedState::new(1).into_result(&ctx, &board);
        assert_eq!(result.best_move(), Some(quick));
    }

    #[test]
    fn completing_an_iteration_leaves_the_searched_line_order_in_place() {
        let board = Board::new();
        let ctx = context(&board);
        let order: Vec<Square> = ctx.root_moves.snapshot().iter().map(|rm| rm.sq).collect();
        let mut completed = CompletedState::new(2);
        let pv = [Square::None; MAX_PLY];
        for (pv_idx, discs) in [(0, -2), (1, 6)] {
            ctx.root_moves.set_pv_idx(pv_idx);
            ctx.root_moves
                .update(order[pv_idx], ScaledScore::from_disc_diff(discs), true, &pv);
            completed.line_completed(&ctx, 10, true);
        }

        assert_eq!(ctx.root_moves.current_pv().unwrap().sq, order[1]);
        let reported: Vec<Square> = completed
            .last
            .unwrap()
            .root_moves
            .iter()
            .map(|rm| rm.sq)
            .collect();
        assert_eq!(reported[..2], [order[1], order[0]]);
    }

    #[test]
    fn quick_move_returns_legal_one_ply_result_or_no_legal_move() {
        let board = Board::new();
        let ctx = context(&board);

        let best_move = quick_move(&board, &ctx.eval)
            .best_move()
            .expect("initial board has legal moves");
        assert!(board.is_legal_move(best_move));

        let no_move_board = Board::from_bitboards(Square::A1.bitboard(), 0);
        assert!(matches!(
            quick_move(&no_move_board, &ctx.eval),
            SearchResult::NoLegalMove
        ));
    }
}
