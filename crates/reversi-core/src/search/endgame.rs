//! Endgame search and solving algorithms.
//!
//! # References:
//!
//! - <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/endgame.c

use std::cell::UnsafeCell;
use std::sync::Arc;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::constants::{SCORE_INF, SCORE_MAX};
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
use crate::count_last_flip::count_last_flip;
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
use crate::count_last_flip::count_last_flip_double;
use crate::eval::EvalMode;
use crate::flip;
use crate::move_list::MoveList;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::search::endgame_cache::{EndGameCache, EndGameCacheBound, EndGameCacheEntry};
use crate::search::node_type::Root;
use crate::search::node_type::{NonPV, PV};
use crate::search::search_context::SearchContext;
use crate::search::search_result::SearchResult;
use crate::search::search_strategy::{EndGameStrategy, MidGameStrategy};
use crate::search::threading::Thread;
use crate::search::time_control::should_stop_iteration;
use crate::search::{SearchProgress, SearchTask, midgame, search};
use crate::square::Square;
use crate::stability::stability_cutoff;
use crate::transposition_table::Bound;
use crate::types::{Depth, ScaledScore, Score};

/// Quadrant masks for move ordering in shallow search.
#[rustfmt::skip]
const QUADRANT_MASK: [u64; 16] = [
    0x0000000000000000, 0x000000000F0F0F0F, 0x00000000F0F0F0F0, 0x00000000FFFFFFFF,
    0x0F0F0F0F00000000, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0FF0F0F0F0, 0x0F0F0F0FFFFFFFFF,
    0xF0F0F0F000000000, 0xF0F0F0F00F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0xF0F0F0F0FFFFFFFF,
    0xFFFFFFFF00000000, 0xFFFFFFFF0F0F0F0F, 0xFFFFFFFFF0F0F0F0, 0xFFFFFFFFFFFFFFFF
];

/// Selectivity sequence for endgame search.
const SELECTIVITY_SEQUENCE: [Selectivity; 6] = [
    Selectivity::Level1,
    Selectivity::Level2,
    Selectivity::Level3,
    Selectivity::Level4,
    Selectivity::Level5,
    Selectivity::None,
];

/// Depth threshold for switching to null window search in endgame.
pub const DEPTH_TO_NWS: Depth = 13;

/// Depth threshold for endgame cache null window search.
const DEPTH_TO_NWS_EC: Depth = 11;

/// Depth threshold for switching to specialized shallow search.
const DEPTH_TO_SHALLOW_SEARCH: Depth = 7;

thread_local! {
    static ENDGAME_CACHE: UnsafeCell<EndGameCache> =
        UnsafeCell::new(EndGameCache::new(14));
}

#[inline(always)]
unsafe fn cache() -> &'static mut EndGameCache {
    ENDGAME_CACHE.with(|cell| unsafe { &mut *cell.get() })
}

/// Performs root search for endgame positions using iterative selectivity.
///
/// # Arguments
///
/// * `task` - Search task.
/// * `thread` - Thread handle for parallel search.
///
/// # Returns
///
/// Search result with best move and score.
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let time_manager = task.time_manager.clone();
    let use_time_control = time_manager.is_some();

    let mut ctx = SearchContext::new(&board, task.selectivity, task.tt.clone(), task.eval.clone());

    // Enable endgame mode for time management
    if let Some(ref tm) = time_manager {
        tm.set_endgame_mode(true);
    }

    let n_empties = ctx.empty_list.count;

    // Estimate initial aspiration window center
    let base_score = estimate_aspiration_base_score(&mut ctx, &board, n_empties, thread);

    // Configure for endgame search
    ctx.selectivity = Selectivity::None;
    ctx.eval_mode = EvalMode::Small;

    let pv_count = if task.multi_pv {
        ctx.root_moves_count()
    } else {
        1
    };

    // Multi-PV loop: search each PV line with its own aspiration window
    for pv_idx in 0..pv_count {
        ctx.set_pv_idx(pv_idx);

        // Initialize aspiration window for this PV line
        let mut alpha = if pv_idx == 0 {
            base_score - ScaledScore::from_disc_diff(3)
        } else {
            -ScaledScore::INF
        };
        let mut beta = if pv_idx == 0 {
            base_score + ScaledScore::from_disc_diff(3)
        } else if let Some(rm) = ctx.get_best_root_move() {
            rm.score
        } else {
            ScaledScore::INF
        };

        // Iterative selectivity loop
        for selectivity in SELECTIVITY_SEQUENCE {
            // Check depth limit when not using time control
            if !use_time_control && task.level.get_end_depth(selectivity) < n_empties {
                break;
            }

            ctx.selectivity = selectivity;
            let score = aspiration_search(&mut ctx, &board, &mut alpha, &mut beta, thread);

            // Update aspiration window for next selectivity
            let delta = ScaledScore::from_disc_diff(2);
            alpha = (score - delta).max(-ScaledScore::INF);
            beta = (score + delta).min(ScaledScore::INF);

            if thread.is_search_aborted() {
                break;
            }

            // Stable sort moves from pv_idx to end, bringing best to pv_idx position
            ctx.sort_root_moves_from_pv_idx();

            // Notify progress with the move now at pv_idx (the best for this PV line)
            if let Some(ref callback) = task.callback
                && let Some(rm) = ctx.get_current_pv_root_move()
            {
                callback(SearchProgress {
                    depth: n_empties,
                    target_depth: n_empties,
                    score: score.to_disc_diff_f32(),
                    best_move: rm.sq,
                    probability: ctx.selectivity.probability(),
                    nodes: ctx.n_nodes,
                    pv_line: rm.pv.clone(),
                    eval_mode: ctx.eval_mode,
                });
            }

            // Check time control
            if should_stop_iteration(&time_manager) {
                break;
            }
        }

        // Check abort or time limit
        if thread.is_search_aborted() || time_manager.as_ref().is_some_and(|tm| tm.check_time()) {
            ctx.sort_all_root_moves();
            let best_move = ctx.get_best_root_move().unwrap();
            return SearchResult::from_root_move(
                &ctx.root_moves,
                &best_move,
                ctx.n_nodes,
                n_empties,
                ctx.selectivity,
                EvalMode::Small,
            );
        }
    }

    ctx.sort_all_root_moves();
    let rm = ctx.get_best_root_move().unwrap();
    SearchResult::from_root_move(
        &ctx.root_moves,
        &rm,
        ctx.n_nodes,
        n_empties,
        ctx.selectivity,
        EvalMode::Small,
    )
}

/// Performs aspiration window search for endgame at the current selectivity level.
fn aspiration_search(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: &mut ScaledScore,
    beta: &mut ScaledScore,
    thread: &Arc<Thread>,
) -> ScaledScore {
    let mut delta = ScaledScore::from_disc_diff(2);
    let n_empties = ctx.empty_list.count;

    loop {
        let score = search::<Root, EndGameStrategy>(ctx, board, n_empties, *alpha, *beta, thread);

        if thread.is_search_aborted() {
            return score;
        }

        // Widen window based on fail direction
        if score <= *alpha {
            *beta = *alpha;
            *alpha = (score - delta).max(-ScaledScore::INF);
        } else if score >= *beta {
            *alpha = (*beta - delta).max(*alpha);
            *beta = (score + delta).min(ScaledScore::INF);
        } else {
            return score;
        }

        delta += delta; // Exponential widening
    }
}

/// Estimates a stable base score to center the aspiration window for the endgame search.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `n_empties` - Number of empty squares.
/// * `thread` - Thread handle for parallel search.
///
/// # Returns
///
/// An estimated score used to center the aspiration window at the start of endgame search.
fn estimate_aspiration_base_score(
    ctx: &mut SearchContext,
    board: &Board,
    n_empties: u32,
    thread: &Arc<Thread>,
) -> ScaledScore {
    ctx.eval_mode = EvalMode::Large;
    ctx.selectivity = Selectivity::Level1;
    let midgame_depth = n_empties / 4;

    let hash_key = board.hash();
    let tt_probe_result = ctx.tt.probe(hash_key);
    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.bound() == Bound::Exact
        && tt_data.depth() >= midgame_depth
    {
        return tt_data.score();
    }

    if n_empties >= 24 {
        search::<PV, MidGameStrategy>(
            ctx,
            board,
            midgame_depth,
            -ScaledScore::INF,
            ScaledScore::INF,
            thread,
        )
    } else if n_empties >= 12 {
        midgame::evaluate_depth2(ctx, board, -ScaledScore::INF, ScaledScore::INF)
    } else {
        midgame::evaluate(ctx, board)
    }
}

/// Attempts ProbCut pruning for endgame positions
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `depth` - Depth of the deep search.
/// * `beta` - Beta bound.
/// * `thread` - Thread handle for parallel search.
///
/// # Returns
///
/// * `Some(score)` - If probcut triggers, returns the predicted bound.
/// * `None` - If probcut doesn't trigger, deep search should be performed.
pub fn probcut(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    beta: ScaledScore,
    thread: &Arc<Thread>,
) -> Option<ScaledScore> {
    if !ctx.selectivity.is_enabled() {
        return None;
    }

    let pc_depth = (2.0 * ((depth as f64).sqrt() * 0.30).floor()) as Depth;
    let mean = probcut::get_mean_end(pc_depth, depth);
    let sigma = probcut::get_sigma_end(pc_depth, depth);
    let t = ctx.selectivity.t_value();

    let pc_beta = probcut::compute_probcut_beta(beta, t, mean, sigma);
    if pc_beta >= ScaledScore::MAX {
        return None;
    }

    let eval_score = midgame::evaluate(ctx, board);
    let mean0 = probcut::get_mean_end(0, depth);
    let sigma0 = probcut::get_sigma_end(0, depth);
    let eval_beta = probcut::compute_eval_beta(beta, t, mean, sigma, mean0, sigma0);

    if eval_score >= eval_beta {
        let current_selectivity = ctx.selectivity;
        ctx.selectivity = Selectivity::None;
        let score =
            search::<NonPV, MidGameStrategy>(ctx, board, pc_depth, pc_beta - 1, pc_beta, thread);
        ctx.selectivity = current_selectivity;

        if score >= pc_beta {
            return Some(ScaledScore::new((beta.value() + pc_beta.value()) / 2));
        }
    }

    None
}

/// Null window search for endgame positions.
/// Dispatches to the optimal solver based on empty square count.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Score threshold.
///
/// # Returns
///
/// Best score found.
#[inline(always)]
pub fn null_window_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;

    if n_empties > DEPTH_TO_NWS_EC {
        return null_window_search_with_tt(ctx, board, alpha);
    }

    if n_empties > DEPTH_TO_SHALLOW_SEARCH {
        return null_window_search_with_ec(ctx, board, alpha);
    }

    match n_empties {
        0 => board.final_score(),
        1 => {
            let sq = ctx.empty_list.first();
            solve1(ctx, board.player, alpha, sq)
        }
        2 => {
            let sq1 = ctx.empty_list.first();
            let sq2 = ctx.empty_list.next(sq1);
            solve2(ctx, board, alpha, sq1, sq2)
        }
        3 => {
            let sq1 = ctx.empty_list.first();
            let sq2 = ctx.empty_list.next(sq1);
            let sq3 = ctx.empty_list.next(sq2);
            solve3(ctx, board, alpha, sq1, sq2, sq3)
        }
        4 => {
            let (sq1, sq2, sq3, sq4) = sort_last4(ctx);
            solve4(ctx, board, alpha, sq1, sq2, sq3, sq4)
        }
        _ => shallow_search(ctx, board, alpha),
    }
}

/// Performs a null window search for fast endgame solving.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Score threshold.
///
/// # Returns
///
/// Best score found.
pub fn null_window_search_with_tt(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let beta = alpha + 1;

    if let Some(score) = stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    let moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -null_window_search_with_tt(ctx, &next, -beta);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve(n_empties);
        }
    }

    let tt_probe_result = ctx.tt.probe(tt_key);
    let tt_move = tt_probe_result.best_move();

    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.is_endgame()
        && tt_data.depth() >= n_empties
        && tt_data.selectivity() >= ctx.selectivity
        && tt_data.can_cut(ScaledScore::from_disc_diff(beta))
    {
        return tt_data.score().to_disc_diff();
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move.is_some() {
        return SCORE_MAX;
    }

    let mut best_score = -SCORE_INF;
    let mut best_move = tt_move;
    if move_list.count() >= 4 {
        move_list.evaluate_moves(ctx, board, n_empties, tt_move);
        for mv in move_list.into_best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);

            let score = if (ctx.empty_list.count - 1) <= DEPTH_TO_NWS_EC {
                ctx.update_endgame(mv.sq);
                let score = -null_window_search_with_ec(ctx, &next, -beta);
                ctx.undo_endgame(mv.sq);
                score
            } else {
                ctx.update(mv.sq, mv.flipped);
                let score = -null_window_search_with_tt(ctx, &next, -beta);
                ctx.undo(mv.sq);
                score
            };

            if score > best_score {
                best_score = score;
                if score >= beta {
                    best_move = mv.sq;
                    break;
                }
            }
        }
    } else if move_list.count() >= 2 {
        move_list.evaluate_moves(ctx, board, n_empties, tt_move);
        move_list.sort();
        for mv in move_list.iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);

            let score = if (ctx.empty_list.count - 1) <= DEPTH_TO_NWS_EC {
                ctx.update_endgame(mv.sq);
                let score = -null_window_search_with_ec(ctx, &next, -beta);
                ctx.undo_endgame(mv.sq);
                score
            } else {
                ctx.update(mv.sq, mv.flipped);
                let score = -null_window_search_with_tt(ctx, &next, -beta);
                ctx.undo(mv.sq);
                score
            };

            if score > best_score {
                best_score = score;
                if score >= beta {
                    best_move = mv.sq;
                    break;
                }
            }
        }
    } else if let Some(mv) = move_list.first() {
        // only one move available
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        best_score = if (ctx.empty_list.count - 1) <= DEPTH_TO_NWS_EC {
            ctx.update_endgame(mv.sq);
            let score = -null_window_search_with_ec(ctx, &next, -beta);
            ctx.undo_endgame(mv.sq);
            score
        } else {
            ctx.update(mv.sq, mv.flipped);
            let score = -null_window_search_with_tt(ctx, &next, -beta);
            ctx.undo(mv.sq);
            score
        };
        best_move = mv.sq;
    }

    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        ScaledScore::from_disc_diff(best_score),
        Bound::determine_bound::<NonPV>(best_score, alpha, beta),
        n_empties,
        best_move,
        Selectivity::None,
        true,
    );

    best_score
}

/// Probes the endgame cache for a given position.
///
/// # Arguments
///
/// * `key` - The hash key of the board position.
///
/// # Returns
///
/// The cached entry if found.
#[inline(always)]
fn probe_endgame_cache(key: u64) -> Option<EndGameCacheEntry> {
    unsafe { cache().probe(key) }
}

/// Stores an entry in the endgame cache.
///
/// # Arguments
///
/// * `key` - The hash key of the board position.
/// * `beta` - The beta value for determining the bound.
/// * `score` - The score to store.
/// * `best_move` - The best move found in this position.
#[inline(always)]
fn store_endgame_cache(key: u64, beta: Score, score: Score, best_move: Square) {
    let bound = EndGameCacheBound::determine_bound(score, beta);
    unsafe {
        cache().store(key, score, bound, best_move);
    }
}

/// Null window search with endgame cache probing.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Score threshold.
///
/// # Returns
///
/// Best score found.
fn null_window_search_with_ec(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let beta = alpha + 1;

    if let Some(score) = stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let mut moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -null_window_search_with_ec(ctx, &next, -beta);
        } else {
            return board.solve(n_empties);
        }
    }

    let key = board.hash();
    let entry = probe_endgame_cache(key);
    let mut tt_move = Square::None;
    if let Some(entry_data) = &entry {
        if entry_data.can_cut(beta) {
            return entry_data.score;
        }
        tt_move = entry_data.best_move;
    }

    let mut best_score = -SCORE_INF;
    if tt_move != Square::None && moves.contains(tt_move) {
        let next = board.make_move(tt_move);
        ctx.update_endgame(tt_move);
        let score = if ctx.empty_list.count <= DEPTH_TO_SHALLOW_SEARCH {
            -shallow_search(ctx, &next, -beta)
        } else {
            -null_window_search_with_ec(ctx, &next, -beta)
        };
        ctx.undo_endgame(tt_move);

        moves = moves.remove(tt_move);
        if score >= beta || moves.is_empty() {
            store_endgame_cache(key, beta, score, tt_move);
            return score;
        }

        best_score = score;
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move.is_some() {
        return SCORE_MAX;
    }

    let mut best_move = tt_move;
    if move_list.count() >= 4 {
        move_list.evaluate_moves_fast(ctx, board, Square::None);
        for mv in move_list.into_best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.update_endgame(mv.sq);
            let score = if ctx.empty_list.count <= DEPTH_TO_SHALLOW_SEARCH {
                -shallow_search(ctx, &next, -beta)
            } else {
                -null_window_search_with_ec(ctx, &next, -beta)
            };
            ctx.undo_endgame(mv.sq);

            if score > best_score {
                best_score = score;
                if score >= beta {
                    best_move = mv.sq;
                    break;
                }
            }
        }
    } else if move_list.count() >= 2 {
        move_list.evaluate_moves_fast(ctx, board, Square::None);
        move_list.sort();
        for mv in move_list.iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.update_endgame(mv.sq);
            let score = if ctx.empty_list.count <= DEPTH_TO_SHALLOW_SEARCH {
                -shallow_search(ctx, &next, -beta)
            } else {
                -null_window_search_with_ec(ctx, &next, -beta)
            };
            ctx.undo_endgame(mv.sq);

            if score > best_score {
                best_score = score;
                if score >= beta {
                    best_move = mv.sq;
                    break;
                }
            }
        }
    } else if let Some(mv) = move_list.first() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update_endgame(mv.sq);
        best_score = if ctx.empty_list.count <= DEPTH_TO_SHALLOW_SEARCH {
            -shallow_search(ctx, &next, -beta)
        } else {
            -null_window_search_with_ec(ctx, &next, -beta)
        };
        ctx.undo_endgame(mv.sq);
        best_move = mv.sq;
    }

    store_endgame_cache(key, beta, best_score, best_move);

    best_score
}

/// Optimized search for shallow endgame positions.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Score threshold.
///
/// # Returns
///
/// Best score found.
pub fn shallow_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let beta = alpha + 1;

    if let Some(score) = stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let mut moves = board.get_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -shallow_search(ctx, &next, -beta);
        } else {
            return board.solve(n_empties);
        }
    }

    let key = board.hash();
    let entry = probe_endgame_cache(key);
    let tt_move = if let Some(entry_data) = &entry {
        if entry_data.can_cut(beta) {
            return entry_data.score;
        }
        entry_data.best_move
    } else {
        Square::None
    };

    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;

    // Search tt_move first if valid
    if tt_move != Square::None && moves.contains(tt_move) {
        let score = shallow_search_move(ctx, board, tt_move, beta);
        if score >= beta {
            store_endgame_cache(key, beta, score, tt_move);
            return score;
        }
        best_score = score;
        best_move = tt_move;
        moves = moves.remove(tt_move);
    }

    if moves.is_empty() {
        store_endgame_cache(key, beta, best_score, best_move);
        return best_score;
    }

    // Split moves into priority (matching parity) and remaining
    let quadrant_mask = Bitboard(QUADRANT_MASK[ctx.empty_list.parity as usize]);
    let priority_moves = moves & quadrant_mask;
    let remaining_moves = moves & !quadrant_mask;

    // Process corners first within priority moves
    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        priority_moves.corners(),
        key,
        beta,
        &mut best_score,
        &mut best_move,
    ) {
        return score;
    }

    // Process non-corner priority moves
    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        priority_moves.non_corners(),
        key,
        beta,
        &mut best_score,
        &mut best_move,
    ) {
        return score;
    }

    // Process corners first within remaining moves
    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        remaining_moves.corners(),
        key,
        beta,
        &mut best_score,
        &mut best_move,
    ) {
        return score;
    }

    // Process non-corner remaining moves
    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        remaining_moves.non_corners(),
        key,
        beta,
        &mut best_score,
        &mut best_move,
    ) {
        return score;
    }

    store_endgame_cache(key, beta, best_score, best_move);

    best_score
}

/// Evaluates a single move in shallow search.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `sq` - Square to move to.
/// * `beta` - Beta bound.
///
/// # Returns
///
/// Score after making move.
#[inline(always)]
fn shallow_search_move(ctx: &mut SearchContext, board: &Board, sq: Square, beta: Score) -> Score {
    let next = board.make_move(sq);
    ctx.update_endgame(sq);
    let score = if ctx.empty_list.count == 4 {
        let next_key = next.hash();
        let entry = probe_endgame_cache(next_key);
        let next_beta = -beta + 1;
        if let Some(entry_data) = &entry
            && entry_data.can_cut(next_beta)
        {
            -entry_data.score
        } else if let Some(score) = stability_cutoff(&next, 4, -beta) {
            -score
        } else {
            let (sq1, sq2, sq3, sq4) = sort_last4(ctx);
            let score = solve4(ctx, &next, -beta, sq1, sq2, sq3, sq4);
            store_endgame_cache(next_key, next_beta, score, Square::None);
            -score
        }
    } else {
        -shallow_search(ctx, &next, -beta)
    };
    ctx.undo_endgame(sq);
    score
}

/// Searches all moves in a bitboard for shallow search.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `moves` - Bitboard of moves to search.
/// * `key` - Hash key for endgame cache.
/// * `beta` - Beta bound.
/// * `best_score` - Current best score (updated if better move found).
/// * `best_move` - Current best move (updated if better move found).
///
/// # Returns
///
/// `Some(score)` if beta cutoff occurs, `None` otherwise.
#[inline(always)]
fn shallow_search_moves(
    ctx: &mut SearchContext,
    board: &Board,
    moves: Bitboard,
    key: u64,
    beta: Score,
    best_score: &mut Score,
    best_move: &mut Square,
) -> Option<Score> {
    for sq in moves.iter() {
        let score = shallow_search_move(ctx, board, sq, beta);

        if score > *best_score {
            if score >= beta {
                store_endgame_cache(key, beta, score, sq);
                return Some(score);
            }
            *best_move = sq;
            *best_score = score;
        }
    }

    None
}

/// Sorts the last four empty squares based on quadrant parity.
///
/// # Arguments
///
/// * `ctx` - Search context.
///
/// # Returns
///
/// Tuple of four squares sorted by quadrant parity.
#[inline(always)]
fn sort_last4(ctx: &mut SearchContext) -> (Square, Square, Square, Square) {
    let (sq1, quad_id1) = ctx.empty_list.first_and_quad_id();
    let (sq2, quad_id2) = ctx.empty_list.next_and_quad_id(sq1);
    let (sq3, quad_id3) = ctx.empty_list.next_and_quad_id(sq2);
    let sq4 = ctx.empty_list.next(sq3);
    let parity = ctx.empty_list.parity;

    if parity & quad_id1 == 0 {
        if parity & quad_id2 != 0 {
            if parity & quad_id3 != 0 {
                (sq2, sq3, sq1, sq4)
            } else {
                (sq2, sq4, sq1, sq3)
            }
        } else if parity & quad_id3 != 0 {
            (sq3, sq4, sq1, sq2)
        } else {
            (sq1, sq2, sq3, sq4)
        }
    } else if parity & quad_id2 == 0 {
        if parity & quad_id3 != 0 {
            (sq1, sq3, sq2, sq4)
        } else {
            (sq1, sq4, sq2, sq3)
        }
    } else {
        (sq1, sq2, sq3, sq4)
    }
}

/// Specialized solver for positions with exactly 4 empty squares.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Score threshold.
/// * `sq1..sq4` - Four empty squares in search order.
///
/// # Returns
///
/// Best score with perfect play.
fn solve4(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
    sq4: Square,
) -> Score {
    let beta = alpha + 1;
    let mut best_score = -SCORE_INF;

    if let Some(next) = board.try_make_move(sq1) {
        best_score = -solve3(ctx, &next, -beta, sq2, sq3, sq4);
        if best_score >= beta {
            return best_score;
        }
    }

    if let Some(next) = board.try_make_move(sq2) {
        let score = -solve3(ctx, &next, -beta, sq1, sq3, sq4);
        if score >= beta {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = board.try_make_move(sq3) {
        let score = -solve3(ctx, &next, -beta, sq1, sq2, sq4);
        if score >= beta {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = board.try_make_move(sq4) {
        let score = -solve3(ctx, &next, -beta, sq1, sq2, sq3);
        return score.max(best_score);
    }

    if best_score == -SCORE_INF {
        let pass = board.switch_players();
        if pass.has_legal_moves() {
            best_score = -solve4(ctx, &pass, -beta, sq1, sq2, sq3, sq4);
        } else {
            best_score = board.solve(4);
        }
    }

    best_score
}

/// Specialized solver for positions with exactly 3 empty squares.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Score threshold.
/// * `sq1..sq3` - Three empty squares.
///
/// # Returns
///
/// Best score with perfect play.
fn solve3(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
) -> Score {
    ctx.increment_nodes();
    let beta = alpha + 1;
    let mut best_score = -SCORE_INF;

    // player moves
    if let Some(next) = board.try_make_move(sq1) {
        best_score = -solve2(ctx, &next, -beta, sq2, sq3);
        if best_score >= beta {
            return best_score;
        }
    }

    if let Some(next) = board.try_make_move(sq2) {
        let score = -solve2(ctx, &next, -beta, sq1, sq3);
        if score >= beta {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = board.try_make_move(sq3) {
        let score = -solve2(ctx, &next, -beta, sq1, sq2);
        return score.max(best_score);
    }

    if best_score != -SCORE_INF {
        return best_score;
    }

    // opponent moves
    ctx.increment_nodes();
    best_score = SCORE_INF;
    let pass = board.switch_players();

    if let Some(next) = pass.try_make_move(sq1) {
        best_score = solve2(ctx, &next, alpha, sq2, sq3);
        if best_score <= alpha {
            return best_score;
        }
    }

    if let Some(next) = pass.try_make_move(sq2) {
        let score = solve2(ctx, &next, alpha, sq1, sq3);
        if score <= alpha {
            return score;
        }
        best_score = score.min(best_score);
    }

    if let Some(next) = pass.try_make_move(sq3) {
        let score = solve2(ctx, &next, alpha, sq1, sq2);
        return score.min(best_score);
    }

    if best_score != SCORE_INF {
        return best_score;
    }

    board.solve(3)
}

/// Specialized solver for positions with exactly 2 empty squares.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `board` - Current board position.
/// * `alpha` - Score threshold.
/// * `sq1, sq2` - Two empty squares.
///
/// # Returns
///
/// Exact score with perfect play.
fn solve2(ctx: &mut SearchContext, board: &Board, alpha: Score, sq1: Square, sq2: Square) -> Score {
    ctx.increment_nodes();
    let player = board.player;
    let opponent = board.opponent;
    let beta = alpha + 1;
    let mut flipped: Bitboard;
    let best_score: Score;

    if opponent.has_adjacent_bit(sq1) {
        flipped = flip::flip(sq1, player, opponent);
        if !flipped.is_empty() {
            let next_player = opponent.apply_flip(flipped);
            best_score = -solve1(ctx, next_player, -beta, sq2);
            if best_score >= beta {
                return best_score;
            }

            if opponent.has_adjacent_bit(sq2) {
                flipped = flip::flip(sq2, player, opponent);
                if !flipped.is_empty() {
                    let next_player = opponent.apply_flip(flipped);
                    let score = -solve1(ctx, next_player, -beta, sq1);
                    return score.max(best_score);
                }
            }
            return best_score;
        }
    }

    if opponent.has_adjacent_bit(sq2) {
        flipped = flip::flip(sq2, player, opponent);
        if !flipped.is_empty() {
            let next_player = opponent.apply_flip(flipped);
            return -solve1(ctx, next_player, -beta, sq1);
        }
    }

    ctx.increment_nodes();
    if player.has_adjacent_bit(sq1) {
        flipped = flip::flip(sq1, opponent, player);
        if !flipped.is_empty() {
            let next_player = player.apply_flip(flipped);
            best_score = solve1(ctx, next_player, alpha, sq2);
            if best_score <= alpha {
                return best_score;
            }

            if player.has_adjacent_bit(sq2) {
                flipped = flip::flip(sq2, opponent, player);
                if !flipped.is_empty() {
                    let next_player = player.apply_flip(flipped);
                    let score = solve1(ctx, next_player, alpha, sq1);
                    return score.min(best_score);
                }
            }
            return best_score;
        }
    }

    if player.has_adjacent_bit(sq2) {
        flipped = flip::flip(sq2, opponent, player);
        if !flipped.is_empty() {
            let next_player = player.apply_flip(flipped);
            return solve1(ctx, next_player, alpha, sq1);
        }
    }

    // both players pass
    board.solve(2)
}

/// Specialized solver for positions with exactly 1 empty square.
///
/// # Arguments
///
/// * `ctx` - Search context.
/// * `player` - Current player's bitboard.
/// * `alpha` - Score threshold (unused in branchless version).
/// * `sq` - Last empty square.
///
/// # Returns
///
/// Exact final score after optimal play.
///
/// # References
///
/// Branchless optimization based on:
/// <https://eukaryote.hateblo.jp/entry/2020/05/10/033228>
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn solve1(ctx: &mut SearchContext, player: Bitboard, _alpha: Score, sq: Square) -> Score {
    ctx.increment_nodes();
    let opponent = !player;
    let (p_flip, o_flip) = count_last_flip_double(player, opponent, sq);

    let base = 2 * player.count() as Score - 64;

    let x1 = base + 2 + p_flip;
    let x2 = base - o_flip;
    let x3 = base + ((base >= 0) as Score) * 2;

    let b1 = p_flip != 0;
    let b2 = o_flip != 0;

    let ax = if b2 { x2 } else { x3 };
    if b1 { x1 } else { ax }
}

/// Specialized solver for positions with exactly 1 empty square (fallback version).
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[inline(always)]
fn solve1(ctx: &mut SearchContext, player: Bitboard, alpha: Score, sq: Square) -> Score {
    ctx.increment_nodes();
    let mut n_flipped = count_last_flip(player, sq);
    let mut score = 2 * player.count() as Score - 64 + 2 + n_flipped;

    if n_flipped == 0 {
        if score <= 0 {
            score -= 2;
            if score > alpha {
                n_flipped = count_last_flip(!player, sq);
                score -= n_flipped;
            }
        } else if score > alpha {
            n_flipped = count_last_flip(!player, sq);
            if n_flipped != 0 {
                score -= n_flipped + 2;
            }
        }
    }

    score
}
