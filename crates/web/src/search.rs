mod endgame;
pub mod search_context;
pub mod search_result;
mod search_task;

use js_sys::Function;
use std::rc::Rc;

use rand::seq::IteratorRandom;

use reversi_core::{
    board::Board,
    flip,
    probcut::Selectivity,
    search::node_type::{NodeType, NonPV, PV, Root},
    square::Square,
    stability,
    transposition_table::{Bound, TranspositionTable},
    types::{Depth, ScaledScore},
};

use crate::{
    eval::Eval,
    level::Level,
    move_list::{MoveList, evaluate_moves_fast},
    probcut,
    search::{search_context::SearchContext, search_result::SearchResult, search_task::SearchTask},
};

/// Depth threshold for switching from midgame to endgame search.
pub const DEPTH_MIDGAME_TO_ENDGAME: Depth = 13;

/// Minimum depth for applying Enhanced Transposition Cutoff (ETC).
pub const MIN_ETC_DEPTH: Depth = 6;

/// Single-threaded search implementation intended for the web target.
pub struct Search {
    tt: Rc<TranspositionTable>,
    eval: Rc<Eval>,
}

impl Search {
    /// Construct a new searcher instance that shares evaluation and table state.
    pub fn new(tt: Rc<TranspositionTable>, eval: Rc<Eval>) -> Self {
        // Ensure probcut and stability modules are initialized
        probcut::init();
        stability::init();

        Search { tt, eval }
    }

    pub fn run(
        &mut self,
        board: &Board,
        level: Level,
        selectivity: Selectivity,
        progress_callback: Option<Function>,
    ) -> SearchResult {
        self.tt.increment_generation();
        let task = SearchTask {
            board: *board,
            level,
            selectivity,
            tt: Rc::clone(&self.tt),
            eval: Rc::clone(&self.eval),
            progress_callback,
        };
        search_root(task)
    }

    #[allow(dead_code)]
    pub fn init(&mut self) {
        self.tt.reset_generation();
        self.tt.clear();
    }
}

/// Performs the root search using iterative deepening with aspiration windows.
///
/// # Arguments
///
/// * `task` - Search task containing board position, search parameters, and callbacks
///
/// # Returns
///
/// SearchResult containing the best move, score, and search statistics
pub fn search_root(task: SearchTask) -> SearchResult {
    let board = task.board;
    let level = task.level;
    let mut ctx = SearchContext::new(
        &board,
        task.selectivity,
        task.tt.clone(),
        task.eval.clone(),
        task.progress_callback,
    );

    let n_empties = ctx.empty_list.count;
    if n_empties == 60 {
        let mv = random_move(&board);
        return SearchResult {
            score: 0.0,
            best_move: Some(mv),
            n_nodes: 0,
            depth: 0,
            selectivity: Selectivity::None,
        };
    }

    if n_empties <= level.end_depth {
        search_root_endgame(&board, &mut ctx, level)
    } else {
        search_root_midgame(board, &mut ctx, level)
    }
}

/// Performs the root search for midgame positions using iterative deepening
fn search_root_midgame(board: Board, ctx: &mut SearchContext, level: Level) -> SearchResult {
    const INITIAL_DELTA: ScaledScore = ScaledScore::from_raw(3 * ScaledScore::SCALE);
    let mut best_score = ScaledScore::ZERO;
    let mut alpha = -ScaledScore::INF;
    let mut beta = ScaledScore::INF;

    let max_depth = level.mid_depth;
    if max_depth == 0 {
        let score = search::<Root>(ctx, &board, max_depth, alpha, beta);
        return SearchResult {
            score: score.to_disc_diff_f32(),
            best_move: None,
            n_nodes: ctx.n_nodes,
            depth: 0,
            selectivity: Selectivity::None,
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
            best_score = search::<Root>(ctx, &board, depth, alpha, beta);

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
        selectivity: ctx.selectivity,
    }
}

/// Performs the root search for endgame positions
fn search_root_endgame(board: &Board, ctx: &mut SearchContext, level: Level) -> SearchResult {
    let n_empties = ctx.empty_list.count;
    let score = estimate_aspiration_base_score(ctx, board, n_empties);
    let final_selectivity = if n_empties > level.perfect_depth {
        Selectivity::Level4
    } else {
        Selectivity::None
    };

    let mut best_score = ScaledScore::ZERO;
    let mut alpha = score - ScaledScore::from_disc_diff(5);
    let mut beta = score + ScaledScore::from_disc_diff(5);

    for selectivity in 1..=final_selectivity.as_u8() {
        ctx.selectivity = Selectivity::from_u8(selectivity);
        let mut delta = ScaledScore::from_disc_diff(3);

        loop {
            best_score = search::<Root>(ctx, board, n_empties, alpha, beta);

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
        selectivity: ctx.selectivity,
    }
}

/// Estimates a stable base score to center the aspiration window for the endgame search.
///
/// # Arguments
///
/// * `ctx` - Search context
/// * `board` - Current board position
/// * `n_empties` - Number of empty squares on the board
/// * `thread` - Thread handle for parallel search coordination
///
/// # Returns
///
/// An estimated score used to center the aspiration window at the start of endgame search.
fn estimate_aspiration_base_score(
    ctx: &mut SearchContext,
    board: &Board,
    n_empties: u32,
) -> ScaledScore {
    let midgame_depth = n_empties / 2;

    let hash_key = board.hash();
    let tt_probe_result = ctx.tt.probe(hash_key);

    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.bound() == Bound::Exact
        && tt_data.depth() >= midgame_depth
    {
        return tt_data.score();
    }

    if n_empties >= 16 {
        ctx.selectivity = Selectivity::Level1;
        search::<PV>(
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
///
/// # Arguments
///
/// * `board` - Current board position
///
/// # Returns
///
/// A randomly selected legal move square
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    board.get_moves().iter().choose(&mut rng).unwrap()
}

/// Alpha-beta search function for midgame positions.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state and statistics.
/// * `board` - Current board position to search.
/// * `depth` - Remaining search depth.
/// * `alpha` - Lower bound of the search window
/// * `beta` - Upper bound of the search window
///
/// # Returns
///
/// The best score found for the position.
pub fn search<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let org_alpha = alpha;
    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;
    let n_empties = ctx.empty_list.count;

    if NT::PV_NODE {
        if depth == 0 {
            return evaluate(ctx, board);
        }
    } else {
        if n_empties == depth && depth <= DEPTH_MIDGAME_TO_ENDGAME {
            let score = endgame::null_window_search(ctx, board, alpha.to_disc_diff());
            return ScaledScore::from_disc_diff(score);
        }

        match depth {
            0 => return evaluate(ctx, board),
            1 => return evaluate_depth1(ctx, board, alpha, beta),
            2 => return evaluate_depth2(ctx, board, alpha, beta),
            _ => {}
        }

        if let Some(score) = stability_cutoff(board, n_empties, alpha) {
            return score;
        }
    }

    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -search::<NT>(ctx, &next, depth, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(n_empties);
        }
    } else if let Some(sq) = move_list.wipeout_move() {
        if NT::ROOT_NODE {
            ctx.update_root_move(sq, ScaledScore::MAX, 1, alpha);
        } else if NT::PV_NODE {
            ctx.update_pv(sq);
        }
        return ScaledScore::MAX;
    }

    // Look up position in transposition table
    let tt_key = board.hash();
    let tt_probe_result = ctx.tt.probe(tt_key);
    let tt_move = tt_probe_result.best_move();

    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && tt_data.depth() >= depth
            && tt_data.selectivity() >= ctx.selectivity
            && tt_data.can_cut(beta)
        {
            return tt_data.score();
        }

        if depth >= MIN_ETC_DEPTH
            && let Some(score) = enhanced_transposition_cutoff(
                ctx,
                board,
                &move_list,
                depth,
                alpha,
                tt_key,
                tt_probe_result.index(),
            )
        {
            return score;
        }

        if let Some(score) = probcut::probcut_midgame(ctx, board, depth, beta) {
            return score;
        }
    }

    if move_list.count() > 1 {
        crate::move_list::evaluate_moves(&mut move_list, ctx, board, tt_move);
        move_list.sort();
    }

    let mut move_count = 0;
    for mv in move_list.iter() {
        move_count += 1;

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped.0);

        let mut score = -ScaledScore::INF;
        if depth >= 2 && mv.reduction_depth > 0 {
            let d = depth - 1 - mv.reduction_depth.min(depth - 1);
            score = -search::<NonPV>(ctx, &next, d, -(alpha + 1), -alpha);
            if score > alpha {
                score = -search::<NonPV>(ctx, &next, depth - 1, -(alpha + 1), -alpha);
            }
        } else if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV>(ctx, &next, depth - 1, -(alpha + 1), -alpha);
        }

        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            ctx.clear_pv();
            score = -search::<PV>(ctx, &next, depth - 1, -beta, -alpha);
        }

        ctx.undo(mv.sq);

        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, alpha);
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                best_move = mv.sq;

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                }

                if NT::PV_NODE && score < beta {
                    alpha = score;
                } else {
                    break;
                }
            }
        }
    }

    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        best_score,
        Bound::classify::<NT>(best_score.value(), org_alpha.value(), beta.value()),
        depth,
        best_move,
        ctx.selectivity,
        n_empties == depth,
    );

    best_score
}

/// Specialized evaluation function for positions at depth 2.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
pub fn evaluate_depth2(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(ctx.empty_list.count);
        }
    }

    if move_list.count() >= 3 {
        evaluate_moves_fast(&mut move_list, ctx, board, Square::None);
    }

    let mut best_score = -ScaledScore::INF;
    for mv in move_list.into_best_first_iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);

        ctx.update(mv.sq, mv.flipped.0);
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

/// Specialized evaluation function for positions at depth 1.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
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
            return board.solve_scaled(ctx.empty_list.count);
        }
    }

    let mut best_score = -ScaledScore::INF;
    for sq in moves.iter() {
        let flipped = flip::flip(sq, board.player, board.opponent);
        if flipped == board.opponent {
            return ScaledScore::MAX;
        }
        let next = board.make_move_with_flipped(flipped, sq);

        ctx.update(sq, flipped.0);
        let score = -evaluate(ctx, &next);
        ctx.undo(sq);

        if score > best_score {
            best_score = score;
            if score >= beta {
                break;
            }
        }
    }

    best_score
}

/// Evaluates a leaf node position using the neural network evaluator.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position
///
/// # Returns
///
/// Position score in internal units
#[inline(always)]
pub fn evaluate(ctx: &SearchContext, board: &Board) -> ScaledScore {
    if ctx.ply() == 60 {
        return board.final_score_scaled();
    }

    ctx.eval.evaluate(ctx, board)
}

/// Checks for stability-based cutoffs in the search.
///
/// # Arguments
///
/// * `board` - Current board position
/// * `n_empties` - Number of empty squares
/// * `alpha` - Current alpha bound for pruning decision
///
/// # Returns
///
/// * `Some(score)` - If position can be pruned with this score
/// * `None` - If no stability cutoff is possible
fn stability_cutoff(board: &Board, n_empties: Depth, alpha: ScaledScore) -> Option<ScaledScore> {
    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha.to_disc_diff()) {
        return Some(ScaledScore::from_disc_diff(score));
    }
    None
}

/// Enhanced Transposition Cutoff
fn enhanced_transposition_cutoff(
    ctx: &mut SearchContext,
    board: &Board,
    move_list: &MoveList,
    depth: u32,
    alpha: ScaledScore,
    tt_key: u64,
    tt_entry_index: usize,
) -> Option<ScaledScore> {
    let etc_depth = depth - 1;
    for mv in move_list.iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.increment_nodes();

        let etc_tt_key = next.hash();
        let etc_tt_probe_result = ctx.tt.probe(etc_tt_key);
        if let Some(etc_tt_data) = etc_tt_probe_result.data()
            && etc_tt_data.depth() >= etc_depth
            && etc_tt_data.selectivity() >= ctx.selectivity
        {
            let score = -etc_tt_data.score();
            if (etc_tt_data.bound() == Bound::Exact || etc_tt_data.bound() == Bound::Upper)
                && score > alpha
            {
                ctx.tt.store(
                    tt_entry_index,
                    tt_key,
                    score,
                    Bound::Lower,
                    depth,
                    mv.sq,
                    ctx.selectivity,
                    ctx.empty_list.count == depth,
                );
                return Some(score);
            }
        }
    }
    None
}
