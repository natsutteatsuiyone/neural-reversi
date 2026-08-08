//! Game tree search engine facade for the web target.
//!
//! Owns the [`Search`] entry point and the midgame/endgame root drivers; the
//! shared alpha-beta core lives in the `pvs` module.

pub mod context;
mod endgame;
mod pvs;
pub mod result;
pub(crate) mod strategy;
mod task;

pub use pvs::search;

use js_sys::Function;
use std::rc::Rc;

use self::strategy::{EndGameStrategy, MidGameStrategy};
use rand::seq::IteratorRandom;

use reversi_core::{
    board::Board,
    flip,
    probcut::Selectivity,
    search::node_type::{NodeType, PV, Root},
    square::Square,
    types::{Depth, ScaledScore},
};

use crate::transposition_table::{Bound, TranspositionTable};

use crate::{
    eval::Eval,
    level::Level,
    move_list::{MoveList, evaluate_moves_fast},
    probcut,
    search::{context::SearchContext, result::SearchResult, task::SearchTask},
};

/// Single-threaded search implementation intended for the web target.
pub struct Search {
    tt: Rc<TranspositionTable>,
    eval: Rc<Eval>,
}

impl Search {
    /// Constructs a new searcher instance that shares evaluation and table state.
    pub fn new(tt: Rc<TranspositionTable>, eval: Rc<Eval>) -> Self {
        // Ensure probcut tables are initialized.
        probcut::init();

        Search { tt, eval }
    }

    /// Runs a search on the given position and returns the best move and score.
    ///
    /// With `multi_pv` set, every legal root move is scored and the results are
    /// reported in [`SearchResult::multi_pv_scores`].
    pub fn run(
        &mut self,
        board: &Board,
        level: Level,
        selectivity: Selectivity,
        progress_callback: Option<Function>,
        multi_pv: bool,
    ) -> SearchResult {
        self.tt.increment_generation();
        search_root(SearchTask {
            board: *board,
            level,
            selectivity,
            tt: Rc::clone(&self.tt),
            eval: Rc::clone(&self.eval),
            progress_callback,
            multi_pv,
        })
    }

    /// Clears the transposition table to reset search state.
    pub fn init(&mut self) {
        self.tt.clear();
    }
}

/// Performs the root search using iterative deepening with aspiration windows.
pub fn search_root(task: SearchTask) -> SearchResult {
    let board = task.board;
    let level = task.level;
    let multi_pv = task.multi_pv;
    let mut ctx = SearchContext::new(
        &board,
        task.selectivity,
        task.tt.clone(),
        task.eval.clone(),
        task.progress_callback,
    );
    if ctx.root_moves.is_empty() {
        // Handle no legal moves
        return SearchResult {
            score: 0.0,
            best_move: None,
            n_nodes: 0,
            depth: 0,
            multi_pv_scores: Vec::new(),
        };
    }

    let n_empties = ctx.empty_list.count();
    if n_empties == 60 && !multi_pv {
        // Handle opening position with random move
        let mv = random_move(&board);
        return SearchResult {
            score: 0.0,
            best_move: Some(mv),
            n_nodes: 0,
            depth: 0,
            multi_pv_scores: Vec::new(),
        };
    }

    if multi_pv {
        if n_empties <= level.end_depth {
            search_root_endgame_multipv(&board, &mut ctx, level)
        } else {
            search_root_midgame_multipv(board, &mut ctx, level)
        }
    } else if n_empties <= level.end_depth {
        search_root_endgame(&board, &mut ctx, level)
    } else {
        search_root_midgame(board, &mut ctx, level)
    }
}

/// Performs the root search for midgame positions using iterative deepening.
fn search_root_midgame(board: Board, ctx: &mut SearchContext, level: Level) -> SearchResult {
    const INITIAL_DELTA: ScaledScore = ScaledScore::from_raw(3 * ScaledScore::SCALE);
    let mut best_score = ScaledScore::ZERO;
    let mut alpha = -ScaledScore::INF;
    let mut beta = ScaledScore::INF;

    let max_depth = level.mid_depth;
    if max_depth == 0 {
        let score = search::<Root, MidGameStrategy>(ctx, &board, max_depth, alpha, beta);
        return SearchResult {
            score: score.to_disc_diff_f32(),
            best_move: None,
            n_nodes: ctx.n_nodes,
            depth: 0,
            multi_pv_scores: Vec::new(),
        };
    }

    let org_selectivity = ctx.selectivity;
    let start_depth = if max_depth.is_multiple_of(2) { 2 } else { 1 };
    let mut depth = start_depth;
    while depth <= max_depth {
        let depth_diff = (max_depth - depth) as u8;
        ctx.selectivity = Selectivity::from_u8(org_selectivity.as_u8().saturating_sub(depth_diff));

        let mut delta = INITIAL_DELTA;
        if depth <= 8 {
            alpha = -ScaledScore::INF;
            beta = ScaledScore::INF;
        }

        loop {
            best_score = search::<Root, MidGameStrategy>(ctx, &board, depth, alpha, beta);

            if best_score <= alpha {
                beta = alpha;
                alpha = (best_score - delta).max(-ScaledScore::INF);
            } else if best_score >= beta {
                alpha = (beta - delta).max(alpha);
                beta = (best_score + delta).min(ScaledScore::INF);
            } else {
                break;
            }

            delta += delta / 2;
        }

        let best_move = ctx.get_best_root_move().unwrap();
        alpha = (best_move.average_score - INITIAL_DELTA).max(-ScaledScore::INF);
        beta = (best_move.average_score + INITIAL_DELTA).min(ScaledScore::INF);

        if depth <= 10 {
            depth += 2;
        } else {
            depth += 1;
        }
    }

    let rm = ctx.get_best_root_move().unwrap();
    ctx.notify_progress(
        max_depth,
        best_score.to_disc_diff_f32(),
        rm.sq,
        ctx.selectivity,
    );
    SearchResult {
        score: best_score.to_disc_diff_f32(),
        best_move: Some(rm.sq),
        n_nodes: ctx.n_nodes,
        depth: max_depth,
        multi_pv_scores: Vec::new(),
    }
}

/// Performs the root search for endgame positions.
fn search_root_endgame(board: &Board, ctx: &mut SearchContext, level: Level) -> SearchResult {
    let n_empties = ctx.empty_list.count();
    let score = estimate_aspiration_base_score(ctx, board, n_empties);
    let final_selectivity = if n_empties > level.perfect_depth + 2 {
        Selectivity::Level2
    } else if n_empties > level.perfect_depth {
        Selectivity::Level3
    } else {
        Selectivity::None
    };

    let mut best_score = ScaledScore::ZERO;
    let mut alpha = score - ScaledScore::from_disc_diff(5);
    let mut beta = score + ScaledScore::from_disc_diff(5);

    for selectivity in 0..=final_selectivity.as_u8() {
        ctx.selectivity = Selectivity::from_u8(selectivity);
        let mut delta = ScaledScore::from_disc_diff(3);

        loop {
            best_score = search::<Root, EndGameStrategy>(ctx, board, n_empties, alpha, beta);

            if best_score <= alpha {
                beta = alpha;
                alpha = (best_score - delta).max(-ScaledScore::INF);
            } else if best_score >= beta {
                alpha = (beta - delta).max(alpha);
                beta = (best_score + delta).min(ScaledScore::INF);
            } else {
                break;
            }

            delta += delta;
        }

        alpha = (best_score - ScaledScore::from_disc_diff(2)).max(-ScaledScore::INF);
        beta = (best_score + ScaledScore::from_disc_diff(2)).min(ScaledScore::INF);
    }

    let rm = ctx.get_best_root_move().unwrap();
    ctx.notify_progress(
        n_empties,
        best_score.to_disc_diff_f32(),
        rm.sq,
        ctx.selectivity,
    );
    SearchResult {
        score: best_score.to_disc_diff_f32(),
        best_move: Some(rm.sq),
        n_nodes: ctx.n_nodes,
        depth: level.end_depth,
        multi_pv_scores: Vec::new(),
    }
}

/// Builds the final Multi-PV result from the sorted root moves.
fn finish_multipv_result(ctx: &SearchContext, depth: Depth) -> SearchResult {
    let best = &ctx.root_moves[0];
    let multi_pv_scores = ctx
        .root_moves
        .iter()
        .map(|rm| (rm.sq, rm.score.to_disc_diff_f32()))
        .collect();
    SearchResult {
        score: best.score.to_disc_diff_f32(),
        best_move: Some(best.sq),
        n_nodes: ctx.n_nodes,
        depth,
        multi_pv_scores,
    }
}

/// Performs the Multi-PV root search for midgame positions.
fn search_root_midgame_multipv(
    board: Board,
    ctx: &mut SearchContext,
    level: Level,
) -> SearchResult {
    const INITIAL_DELTA: ScaledScore = ScaledScore::from_raw(3 * ScaledScore::SCALE);
    let pv_count = ctx.root_moves.len();
    let max_depth = level.mid_depth.max(1);
    let org_selectivity = ctx.selectivity;

    let start_depth = if max_depth.is_multiple_of(2) { 2 } else { 1 };
    let mut depth = start_depth;
    while depth <= max_depth {
        let depth_diff = (max_depth - depth) as u8;
        ctx.selectivity = Selectivity::from_u8(org_selectivity.as_u8().saturating_sub(depth_diff));
        ctx.save_previous_scores();

        for pv_idx in 0..pv_count {
            ctx.set_pv_idx(pv_idx);

            let (mut alpha, mut beta) = match ctx.current_pv_root_move() {
                Some(rm) if depth > start_depth && rm.previous_score > -ScaledScore::INF => (
                    (rm.previous_score - INITIAL_DELTA).max(-ScaledScore::INF),
                    (rm.previous_score + INITIAL_DELTA).min(ScaledScore::INF),
                ),
                _ => (-ScaledScore::INF, ScaledScore::INF),
            };

            let mut delta = INITIAL_DELTA;
            let mut score;
            loop {
                score = search::<Root, MidGameStrategy>(ctx, &board, depth, alpha, beta);

                if score <= alpha {
                    beta = alpha;
                    alpha = (score - delta).max(-ScaledScore::INF);
                } else if score >= beta {
                    alpha = (beta - delta).max(alpha);
                    beta = (score + delta).min(ScaledScore::INF);
                } else {
                    break;
                }

                delta += delta / 2;
            }

            ctx.sort_root_moves_from_pv_idx();
            if let Some(rm) = ctx.current_pv_root_move() {
                ctx.notify_progress(depth, score.to_disc_diff_f32(), rm.sq, ctx.selectivity);
            }
        }

        if depth <= 10 {
            depth += 2;
        } else {
            depth += 1;
        }
    }
    ctx.set_pv_idx(0);

    finish_multipv_result(ctx, max_depth)
}

/// Performs the Multi-PV root search for endgame positions.
fn search_root_endgame_multipv(
    board: &Board,
    ctx: &mut SearchContext,
    level: Level,
) -> SearchResult {
    let n_empties = ctx.empty_list.count();
    let base_score = estimate_aspiration_base_score(ctx, board, n_empties);
    let final_selectivity = if n_empties > level.perfect_depth + 2 {
        Selectivity::Level2
    } else if n_empties > level.perfect_depth {
        Selectivity::Level3
    } else {
        Selectivity::None
    };

    let pv_count = ctx.root_moves.len();
    for pv_idx in 0..pv_count {
        ctx.set_pv_idx(pv_idx);

        let mut alpha = if pv_idx == 0 {
            base_score - ScaledScore::from_disc_diff(5)
        } else {
            -ScaledScore::INF
        };
        let mut beta = if pv_idx == 0 {
            base_score + ScaledScore::from_disc_diff(5)
        } else if let Some(rm) = ctx.get_best_root_move() {
            rm.score
        } else {
            ScaledScore::INF
        };

        let mut best_score = ScaledScore::ZERO;
        for selectivity in 0..=final_selectivity.as_u8() {
            ctx.selectivity = Selectivity::from_u8(selectivity);
            let mut delta = ScaledScore::from_disc_diff(3);

            loop {
                best_score = search::<Root, EndGameStrategy>(ctx, board, n_empties, alpha, beta);

                if best_score <= alpha {
                    beta = alpha;
                    alpha = (best_score - delta).max(-ScaledScore::INF);
                } else if best_score >= beta {
                    alpha = (beta - delta).max(alpha);
                    beta = (best_score + delta).min(ScaledScore::INF);
                } else {
                    break;
                }

                delta += delta;
            }

            alpha = (best_score - ScaledScore::from_disc_diff(2)).max(-ScaledScore::INF);
            beta = (best_score + ScaledScore::from_disc_diff(2)).min(ScaledScore::INF);
        }

        ctx.sort_root_moves_from_pv_idx();
        if let Some(rm) = ctx.current_pv_root_move() {
            ctx.notify_progress(
                n_empties,
                best_score.to_disc_diff_f32(),
                rm.sq,
                ctx.selectivity,
            );
        }
    }
    ctx.set_pv_idx(0);

    finish_multipv_result(ctx, level.end_depth)
}

/// Estimates a base score to center the aspiration window for endgame search.
fn estimate_aspiration_base_score(
    ctx: &mut SearchContext,
    board: &Board,
    n_empties: u32,
) -> ScaledScore {
    let midgame_depth = n_empties / 2;

    let hash_key = board.hash();
    let tt_probe_result = ctx.tt.probe(hash_key);

    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.bound == Bound::Exact
        && tt_data.depth >= midgame_depth
    {
        return tt_data.score;
    }

    if n_empties >= 16 {
        ctx.selectivity = Selectivity::Level1;
        search::<PV, MidGameStrategy>(
            ctx,
            board,
            midgame_depth,
            -ScaledScore::INF,
            ScaledScore::INF,
        )
    } else if n_empties >= 6 {
        evaluate_depth2(ctx, board, -ScaledScore::INF, ScaledScore::INF)
    } else {
        evaluate(ctx, board)
    }
}

/// Selects a random legal move from the current position.
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    board.get_moves().iter().choose(&mut rng).unwrap()
}

/// Performs alpha-beta search specialized for depth 3.
pub fn evaluate_depth3<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let org_alpha = alpha;

    let moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth3::<NT>(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    let tt_key = board.hash();
    let tt_probe_result = ctx.tt.probe(tt_key);
    let tt_move = tt_probe_result.best_move();

    if !NT::PV_NODE
        && let Some(tt_data) = tt_probe_result.data()
        && tt_data.depth >= 3
        && tt_data.can_cut(beta)
    {
        return tt_data.score;
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move().is_some() {
        return ScaledScore::MAX;
    }

    if move_list.count() >= 2 {
        evaluate_moves_fast(&mut move_list, ctx, board, tt_move);
    }

    let mut best_score = -ScaledScore::INF;
    let mut best_move = Square::None;
    for mv in move_list.best_first_iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);

        ctx.update(mv.sq, mv.flipped);
        let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
        ctx.undo(mv.sq);

        if score > best_score {
            best_score = score;
            if score >= beta {
                best_move = mv.sq;
                break;
            }
            if score > alpha {
                best_move = mv.sq;
                alpha = score;
            }
        }
    }

    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        best_score,
        Bound::classify::<NT>(best_score, org_alpha, beta),
        3,
        best_move,
        Selectivity::None,
        false,
    );

    best_score
}
/// Performs alpha-beta search specialized for depth 2.
pub fn evaluate_depth2(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move().is_some() {
        return ScaledScore::MAX;
    }

    if move_list.count() >= 2 {
        evaluate_moves_fast(&mut move_list, ctx, board, Square::None);
    }

    let mut best_score = -ScaledScore::INF;
    for mv in move_list.best_first_iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);

        ctx.update(mv.sq, mv.flipped);
        let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
        ctx.undo(mv.sq);

        if score > best_score {
            best_score = score;
            if score >= beta {
                break;
            }
            if score > alpha {
                alpha = score;
            }
        }
    }

    best_score
}

/// Performs alpha-beta search specialized for depth 1.
pub fn evaluate_depth1(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count());
        }
    }

    let mut best_score = -ScaledScore::INF;

    for sq in moves.corners().iter() {
        if let Some(score) = search_move_in_evaluate_depth1(ctx, board, sq, beta, &mut best_score) {
            return score;
        }
    }

    for sq in moves.non_corners().iter() {
        if let Some(score) = search_move_in_evaluate_depth1(ctx, board, sq, beta, &mut best_score) {
            return score;
        }
    }

    best_score
}

/// Searches a single move within [`evaluate_depth1`], returning on beta cutoff.
#[inline(always)]
fn search_move_in_evaluate_depth1(
    ctx: &mut SearchContext,
    board: &Board,
    sq: Square,
    beta: ScaledScore,
    best_score: &mut ScaledScore,
) -> Option<ScaledScore> {
    let flipped = flip::flip(sq, board.player(), board.opponent());
    if flipped == board.opponent() {
        return Some(ScaledScore::MAX);
    }
    let next = board.make_move_with_flipped(flipped, sq);

    ctx.update(sq, flipped);
    let score = -evaluate(ctx, &next);
    ctx.undo(sq);

    if score > *best_score {
        *best_score = score;
        if score >= beta {
            return Some(score);
        }
    }
    None
}

/// Evaluates a leaf node position using the neural network.
#[inline(always)]
pub fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore {
    if ctx.ply() == 60 {
        return board.final_score_scaled();
    }

    ctx.eval.evaluate(ctx, board)
}
