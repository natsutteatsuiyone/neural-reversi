//! # References:
//! - https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/endgame.c
//! - https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/search.cpp

use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::sync::Arc;

use crate::board::Board;
use crate::constants::{SCORE_INF, SCORE_MAX, scale_score, unscale_score};
use crate::count_last_flip::count_last_flip;
use crate::move_list::{ConcurrentMoveIterator, MoveList};
use crate::search::endgame_cache::{EndGameCache, EndGameCacheBound, EndGameCacheEntry};
use crate::search::enhanced_transposition_cutoff;
use crate::search::node_type::{NodeType, NonPV, PV, Root};
use crate::search::search_context::SearchContext;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::transposition_table::Bound;
use crate::types::{Depth, Score, Scoref, Selectivity};
use crate::{bitboard, flip, probcut, stability};

use super::search_context::GamePhase;
use super::search_result::SearchResult;
use super::threading::Thread;
use super::{SearchTask, midgame};

/// Quadrant masks for move ordering in shallow search.
#[rustfmt::skip]
const QUADRANT_MASK: [u64; 16] = [
    0x0000000000000000, 0x000000000F0F0F0F, 0x00000000F0F0F0F0, 0x00000000FFFFFFFF,
    0x0F0F0F0F00000000, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0FF0F0F0F0, 0x0F0F0F0FFFFFFFFF,
    0xF0F0F0F000000000, 0xF0F0F0F00F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0xF0F0F0F0FFFFFFFF,
    0xFFFFFFFF00000000, 0xFFFFFFFF0F0F0F0F, 0xFFFFFFFFF0F0F0F0, 0xFFFFFFFFFFFFFFFF
];

/// Selectivity sequence for endgame search
const SELECTIVITY_SEQUENCE: [Selectivity; 6] = [
    Selectivity::Level1,
    Selectivity::Level2,
    Selectivity::Level3,
    Selectivity::Level4,
    Selectivity::Level5,
    Selectivity::None,
];

/// Minimum depth required for parallel search splitting.
const MIN_SPLIT_DEPTH: Depth = 7;

/// Minimum depth for enhanced transposition table cutoff.
const MIN_ETC_DEPTH: Depth = 6;

/// Depth threshold for switching from midgame to endgame search.
pub const DEPTH_TO_NWS: Depth = 14;

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
/// * `task` - Search task containing board position, search parameters, and callbacks
/// * `thread` - Thread handle for parallel search coordination
///
/// # Returns
///
/// SearchResult containing the best move, score, and search statistics
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let time_manager = task.time_manager.clone();
    let use_time_control = time_manager.is_some();

    let mut ctx = SearchContext::new(
        &board,
        task.generation,
        task.selectivity,
        task.tt.clone(),
        task.eval.clone(),
    );

    // Enable endgame mode for time management
    if let Some(ref tm) = time_manager {
        tm.set_endgame_mode(true);
    }

    if let Some(ref callback) = task.callback {
        ctx.set_callback(callback.clone());
    }

    let n_empties = ctx.empty_list.count;

    // Estimate initial aspiration window center
    let base_score = estimate_aspiration_base_score(&mut ctx, &board, n_empties, thread);

    // Configure for endgame search
    ctx.selectivity = Selectivity::None;
    ctx.game_phase = GamePhase::EndGame;

    let pv_count = if task.multi_pv {
        ctx.root_moves_count()
    } else {
        1
    };
    let mut best_score = 0;
    let mut alpha = base_score - 3;
    let mut beta = base_score + 3;

    // Multi-PV loop
    for pv_idx in 0..pv_count {
        if pv_idx >= 1 {
            alpha = -SCORE_INF;
            beta = best_score;
        }

        let mut best_move_sq = Square::None;

        // Iterative selectivity loop
        for selectivity in SELECTIVITY_SEQUENCE {
            // Check depth limit when not using time control
            if !use_time_control && task.level.get_end_depth(selectivity) < n_empties {
                break;
            }

            ctx.selectivity = selectivity;
            best_score = aspiration_search(&mut ctx, &board, &mut alpha, &mut beta, thread);

            let best_move = ctx.get_best_root_move(true).unwrap();

            // Update aspiration window for next selectivity
            alpha = (best_score - 2).max(-SCORE_INF);
            beta = (best_score + 2).min(SCORE_INF);
            best_move_sq = best_move.sq;

            if thread.is_search_aborted() {
                break;
            }

            ctx.notify_progress(
                n_empties,
                best_score as Scoref,
                best_move.sq,
                ctx.selectivity,
            );

            // Check time control
            if should_stop_iteration(&time_manager) {
                break;
            }
        }

        ctx.mark_root_move_searched(best_move_sq);

        let best_move = ctx.get_best_root_move(false).unwrap();
        best_score = best_move.score;

        // Check abort or time limit
        if thread.is_search_aborted() || time_manager.as_ref().is_some_and(|tm| tm.check_time()) {
            return build_endgame_result(&ctx, &best_move, n_empties);
        }
    }

    let rm = ctx.get_best_root_move(false).unwrap();
    build_endgame_result(&ctx, &rm, n_empties)
}

/// Performs aspiration window search for endgame at the current selectivity level.
fn aspiration_search(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: &mut Score,
    beta: &mut Score,
    thread: &Arc<Thread>,
) -> Score {
    let mut delta = 2;

    loop {
        let score = search::<Root>(ctx, board, *alpha, *beta, thread);

        if thread.is_search_aborted() {
            return score;
        }

        // Widen window based on fail direction
        if score <= *alpha {
            *beta = *alpha;
            *alpha = (score - delta).max(-SCORE_INF);
        } else if score >= *beta {
            *alpha = (*beta - delta).max(*alpha);
            *beta = (score + delta).min(SCORE_INF);
        } else {
            return score;
        }

        delta += delta; // Exponential widening
    }
}

/// Determines whether to stop the current selectivity iteration based on time control.
fn should_stop_iteration(time_manager: &Option<Arc<super::time_control::TimeManager>>) -> bool {
    match time_manager {
        Some(tm) => tm.check_time() || !tm.should_continue_iteration(),
        None => false,
    }
}

/// Builds the search result for endgame.
fn build_endgame_result(
    ctx: &SearchContext,
    best_move: &super::root_move::RootMove,
    n_empties: Depth,
) -> SearchResult {
    SearchResult {
        score: best_move.score as Scoref,
        best_move: Some(best_move.sq),
        n_nodes: ctx.n_nodes,
        pv_line: best_move.pv.clone(),
        depth: n_empties,
        selectivity: ctx.selectivity,
        game_phase: GamePhase::EndGame,
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
    thread: &Arc<Thread>,
) -> Score {
    ctx.game_phase = GamePhase::MidGame;
    ctx.selectivity = Selectivity::Level1;
    let midgame_depth = n_empties / 4;

    let hash_key = board.hash();
    let tt_probe_result = ctx.tt.probe(hash_key, ctx.generation);
    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.bound() == Bound::Exact
        && tt_data.depth() >= midgame_depth
    {
        return unscale_score(tt_data.score());
    }

    let score = if n_empties >= 24 {
        midgame::search::<PV>(ctx, board, midgame_depth, -SCORE_INF, SCORE_INF, thread)
    } else if n_empties >= 12 {
        midgame::evaluate_depth2(ctx, board, -SCORE_INF, SCORE_INF)
    } else {
        midgame::evaluate(ctx, board)
    };

    unscale_score(score)
}

/// Alpha-beta search function for endgame positions.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state and statistics.
/// * `board` - Current board position to solve.
/// * `alpha` - Lower bound of the search window.
/// * `beta` - Upper bound of the search window.
/// * `thread` - Thread handle for parallel search coordination.
///
/// # Returns
///
/// The exact score in disc difference.
pub fn search<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: Score,
    beta: Score,
    thread: &Arc<Thread>,
) -> Score {
    let org_alpha = alpha;
    let n_empties = ctx.empty_list.count;

    if NT::PV_NODE {
        if n_empties == 0 {
            return calculate_final_score(board);
        }
    } else {
        if n_empties <= DEPTH_TO_NWS {
            return null_window_search(ctx, board, alpha);
        }

        if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
            return score;
        }
    }

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -search::<NT>(ctx, &next, -beta, -alpha, thread);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, n_empties);
        }
    } else if let Some(sq) = move_list.wipeout_move {
        if NT::ROOT_NODE {
            ctx.update_root_move(sq, SCORE_MAX, 1, alpha);
        } else if NT::PV_NODE {
            ctx.update_pv(sq);
        }
        return SCORE_MAX;
    }

    // Look up position in transposition table
    let tt_probe_result = ctx.tt.probe(tt_key, ctx.generation);
    let tt_move = tt_probe_result.best_move();

    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && tt_data.is_endgame()
            && tt_data.depth() >= n_empties
            && tt_data.selectivity() >= ctx.selectivity
            && tt_data.can_cut(scale_score(beta))
        {
            return unscale_score(tt_data.score());
        }

        if n_empties >= MIN_ETC_DEPTH
            && let Some(score) = enhanced_transposition_cutoff(
                ctx,
                board,
                &move_list,
                n_empties,
                scale_score(alpha),
                tt_key,
                tt_probe_result.index(),
            )
        {
            return unscale_score(score);
        }

        if let Some(score) = probcut::probcut_endgame(ctx, board, n_empties, beta, thread) {
            return score;
        }
    }

    if move_list.count() > 1 {
        move_list.evaluate_moves::<NT>(ctx, board, n_empties, tt_move);
        move_list.sort();
    }

    let move_iter = Arc::new(ConcurrentMoveIterator::new(move_list));
    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;

    while let Some((mv, move_count)) = move_iter.next() {
        if NT::ROOT_NODE && ctx.is_move_searched(mv.sq) {
            continue;
        }

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -SCORE_INF;
        if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV>(ctx, &next, -(alpha + 1), -alpha, thread);
        }

        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            ctx.clear_pv();
            score = -search::<PV>(ctx, &next, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return 0;
        }

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

        if n_empties >= MIN_SPLIT_DEPTH && move_iter.count() > 1 && thread.can_split() {
            let (s, m, n) = thread.split(
                ctx,
                board,
                alpha,
                beta,
                best_score,
                best_move,
                n_empties,
                &move_iter,
                NT::TYPE_ID,
            );
            best_score = s;
            best_move = m;
            ctx.n_nodes += n;

            if thread.is_search_aborted() || thread.cutoff_occurred() {
                return 0;
            }

            if best_score >= beta {
                break;
            }
        }
    }

    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        scale_score(best_score),
        Bound::determine_bound::<NT>(best_score, org_alpha, beta),
        n_empties,
        best_move,
        ctx.selectivity,
        ctx.generation,
        true,
    );

    best_score
}

/// Alpha-beta search function for splitpoint nodes in parallel endgame search.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state and statistics.
/// * `board` - Current board position to solve.
/// * `thread` - Thread handle for parallel search coordination.
/// * `split_point` - Split point for parallel search coordination.
///
/// # Returns
///
/// The exact score in disc difference.
pub fn search_sp<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    thread: &Arc<Thread>,
    split_point: &Arc<SplitPoint>,
) -> Score {
    let beta = split_point.state().beta;
    let move_iter = split_point.state().move_iter.clone().unwrap();

    while let Some((mv, move_count)) = move_iter.next() {
        if NT::ROOT_NODE && ctx.is_move_searched(mv.sq) {
            continue;
        }

        split_point.unlock();

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let alpha = split_point.state().alpha();
        let mut score = -search::<NonPV>(ctx, &next, -(alpha + 1), -alpha, thread);

        if NT::PV_NODE && score > alpha {
            ctx.clear_pv();
            let alpha = split_point.state().alpha();
            score = -search::<PV>(ctx, &next, -beta, -alpha, thread);
        }

        ctx.undo(mv.sq);

        split_point.lock();

        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return 0;
        }

        let sp = split_point.state();

        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, sp.alpha());
        }

        if score > sp.best_score() {
            sp.set_best_score(score);

            if score > sp.alpha() {
                sp.set_best_move(mv.sq);

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                    split_point.state_mut().copy_pv(ctx.get_pv());
                }

                if NT::PV_NODE && score < beta {
                    sp.set_alpha(score);
                } else {
                    sp.set_cutoff(true);
                    break;
                }
            }
        }
    }

    split_point.state().best_score()
}

/// Null window search for endgame positions.
/// Dispatches to the optimal solver based on empty square count.
///
/// # Arguments
///
/// * `ctx` - Search context
/// * `board` - Current board position
/// * `alpha` - Score threshold for null window search
///
/// # Returns
///
/// Best score found
#[inline(always)]
fn null_window_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;

    if n_empties > DEPTH_TO_NWS_EC {
        return null_window_search_with_tt(ctx, board, alpha);
    }

    if n_empties > DEPTH_TO_SHALLOW_SEARCH {
        return null_window_search_with_ec(ctx, board, alpha);
    }

    match n_empties {
        0 => calculate_final_score(board),
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
            let (sq1, sq2, sq3, sq4) = sort_empties_at_4(ctx);
            solve4(ctx, board, alpha, sq1, sq2, sq3, sq4)
        }
        _ => shallow_search(ctx, board, alpha),
    }
}

/// Performs a null window search for fast endgame solving.
///
/// # Arguments
///
/// * `ctx` - Search context for tracking statistics and TT access
/// * `board` - Current board position to evaluate
/// * `alpha` - Score threshold to beat
///
/// # Returns
///
/// Best score found
pub fn null_window_search_with_tt(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let beta = alpha + 1;

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    let mut move_list = MoveList::new(board);
    if move_list.wipeout_move.is_some() {
        return SCORE_MAX;
    } else if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -null_window_search_with_tt(ctx, &next, -beta);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, n_empties);
        }
    }

    let tt_probe_result = ctx.tt.probe(tt_key, ctx.generation);
    let tt_move = tt_probe_result.best_move();

    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.is_endgame()
        && tt_data.depth() >= n_empties
        && tt_data.selectivity() >= ctx.selectivity
        && tt_data.can_cut(scale_score(beta))
    {
        return unscale_score(tt_data.score());
    }

    if n_empties == DEPTH_TO_NWS
        && let Some(score) = enhanced_transposition_cutoff(
            ctx,
            board,
            &move_list,
            n_empties,
            scale_score(alpha),
            tt_key,
            tt_probe_result.index(),
        )
    {
        return unscale_score(score);
    }

    let mut best_score = -SCORE_INF;
    let mut best_move = tt_move;
    if move_list.count() >= 4 {
        move_list.evaluate_moves::<NonPV>(ctx, board, n_empties, tt_move);
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
        move_list.evaluate_moves::<NonPV>(ctx, board, n_empties, tt_move);
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
    } else {
        // only one move available
        let mv = move_list.first().unwrap();
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
        scale_score(best_score),
        Bound::determine_bound::<NonPV>(best_score, alpha, beta),
        n_empties,
        best_move,
        Selectivity::None,
        ctx.generation,
        true,
    );

    best_score
}

/// Probe the endgame cache for a given position
///
/// # Arguments
///
/// * `key` - The hash key of the board position
///
/// * `n_empties` - The number of empty squares on the board
///
/// # Returns
///
/// * `Option<EndGameCacheEntry>` - The cached entry if found
#[inline(always)]
fn probe_endgame_cache(key: u64) -> Option<EndGameCacheEntry> {
    unsafe { cache().probe(key) }
}

/// Store an entry in the endgame cache
///
/// # Arguments
///
/// * `key` - The hash key of the board position
/// * `n_empties` - The number of empty squares on the board
/// * `beta` - The beta value for determining the bound
/// * `score` - The score to store
/// * `best_move` - The best move found in this position
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
/// * `ctx` - Search context for node counting and empty square tracking
/// * `board` - Current board position
/// * `alpha` - Score threshold for null window search
///
/// # Returns
///
/// Best score found
fn null_window_search_with_ec(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let beta = alpha + 1;

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let mut move_list = MoveList::new(board);
    if move_list.wipeout_move.is_some() {
        return SCORE_MAX;
    } else if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -null_window_search_with_ec(ctx, &next, -beta);
        } else {
            return solve(board, n_empties);
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
    let mut best_move = tt_move;
    if move_list.count() >= 4 {
        move_list.evaluate_moves_fast(ctx, board, tt_move);
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
        move_list.evaluate_moves_fast(ctx, board, tt_move);
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
    } else {
        // only one move available
        let mv = move_list.first().unwrap();
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
/// * `ctx` - Search context for node counting and empty square tracking
/// * `board` - Current board position
/// * `alpha` - Score threshold for null window search
///
/// # Returns
///
/// Best score found
pub fn shallow_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let beta = alpha + 1;

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    #[inline(always)]
    fn search_child(ctx: &mut SearchContext, next: &Board, beta: Score) -> Score {
        if ctx.empty_list.count == 4 {
            let key = next.hash();
            let entry = probe_endgame_cache(key);
            let next_beta = -beta + 1;
            if let Some(entry_data) = &entry
                && entry_data.can_cut(next_beta)
            {
                return -entry_data.score;
            }

            if let Some(score) = stability::stability_cutoff(next, 4, -beta) {
                -score
            } else {
                let (sq1, sq2, sq3, sq4) = sort_empties_at_4(ctx);
                let score = solve4(ctx, next, -beta, sq1, sq2, sq3, sq4);
                store_endgame_cache(key, next_beta, score, Square::None);
                -score
            }
        } else {
            -shallow_search(ctx, next, -beta)
        }
    }

    #[inline(always)]
    fn search_move(
        ctx: &mut SearchContext,
        board: &Board,
        sq: Square,
        beta: Score,
        best_score: &mut Score,
        best_move: &mut Square,
    ) -> Option<Score> {
        let next = board.make_move(sq);
        ctx.update_endgame(sq);
        let score = search_child(ctx, &next, beta);
        ctx.undo_endgame(sq);

        if score > *best_score {
            if score >= beta {
                return Some(score);
            }
            *best_move = sq;
            *best_score = score;
        }
        None
    }

    let mut moves = board.get_moves();
    if moves == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -shallow_search(ctx, &next, -beta);
        } else {
            return solve(board, n_empties);
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

    // Search tt_move first if valid (now validated against moves bitboard)
    if tt_move != Square::None && bitboard::is_set(moves, tt_move) {
        moves &= !tt_move.bitboard();
        if let Some(score) = search_move(ctx, board, tt_move, beta, &mut best_score, &mut best_move)
        {
            store_endgame_cache(key, beta, score, tt_move);
            return score;
        }
        best_move = tt_move;
    }

    if moves == 0 {
        store_endgame_cache(key, beta, best_score, best_move);
        return best_score;
    }

    // Split moves into priority (matching parity) and remaining
    let quadrant_mask = QUADRANT_MASK[ctx.empty_list.parity as usize];
    let priority_moves = moves & quadrant_mask;
    let remaining_moves = moves & !quadrant_mask;

    // Process corners first within priority moves
    let mut current = priority_moves & bitboard::CORNER_MASK;
    while current != 0 {
        let sq = Square::from_u32_unchecked(current.trailing_zeros());
        current &= current - 1;

        if let Some(score) = search_move(ctx, board, sq, beta, &mut best_score, &mut best_move) {
            store_endgame_cache(key, beta, score, sq);
            return score;
        }
    }

    // Process non-corner priority moves
    current = priority_moves & !bitboard::CORNER_MASK;
    while current != 0 {
        let sq = Square::from_u32_unchecked(current.trailing_zeros());
        current &= current - 1;

        if let Some(score) = search_move(ctx, board, sq, beta, &mut best_score, &mut best_move) {
            store_endgame_cache(key, beta, score, sq);
            return score;
        }
    }

    // Process corners first within remaining moves
    current = remaining_moves & bitboard::CORNER_MASK;
    while current != 0 {
        let sq = Square::from_u32_unchecked(current.trailing_zeros());
        current &= current - 1;

        if let Some(score) = search_move(ctx, board, sq, beta, &mut best_score, &mut best_move) {
            store_endgame_cache(key, beta, score, sq);
            return score;
        }
    }

    // Process non-corner remaining moves
    current = remaining_moves & !bitboard::CORNER_MASK;
    while current != 0 {
        let sq = Square::from_u32_unchecked(current.trailing_zeros());
        current &= current - 1;

        if let Some(score) = search_move(ctx, board, sq, beta, &mut best_score, &mut best_move) {
            store_endgame_cache(key, beta, score, sq);
            return score;
        }
    }

    store_endgame_cache(key, beta, best_score, best_move);

    best_score
}

/// Sorts the four remaining empty squares based on quadrant parity.
///
/// # Arguments
///
/// * `ctx` - Search context containing empty square list and parity
///
/// # Returns
///
/// Tuple of four squares in optimized search order
#[inline(always)]
fn sort_empties_at_4(ctx: &mut SearchContext) -> (Square, Square, Square, Square) {
    let (sq1, quad_id1) = ctx.empty_list.first_with_quad_id();
    let (sq2, quad_id2) = ctx.empty_list.next_with_quad_id(sq1);
    let (sq3, quad_id3) = ctx.empty_list.next_with_quad_id(sq2);
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
/// * `ctx` - Search context for node counting
/// * `board` - Current board position
/// * `alpha` - Score threshold for pruning
/// * `sq1..sq4` - The four empty squares in search order
///
/// # Returns
///
/// Best score achievable with perfect play
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
            best_score = solve(board, 4);
        }
    }

    best_score
}

/// Specialized solver for positions with exactly 3 empty squares.
///
/// # Arguments
///
/// * `ctx` - Search context for node counting
/// * `board` - Current board position
/// * `alpha` - Score threshold for pruning
/// * `sq1..sq3` - The three empty squares
///
/// # Returns
///
/// Best score achievable with perfect play
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

    solve(board, 3)
}

/// Specialized solver for positions with exactly 2 empty squares.
///
/// # Arguments
///
/// * `ctx` - Search context for node counting
/// * `board` - Current board position
/// * `alpha` - Score threshold for pruning
/// * `sq1, sq2` - The two remaining empty squares
///
/// # Returns
///
/// Exact score with perfect play
fn solve2(ctx: &mut SearchContext, board: &Board, alpha: Score, sq1: Square, sq2: Square) -> Score {
    ctx.increment_nodes();
    let player = board.player;
    let opponent = board.opponent;
    let beta = alpha + 1;
    let mut flipped: u64;
    let best_score: Score;

    if bitboard::has_adjacent_bit(opponent, sq1) {
        flipped = flip::flip(sq1, player, opponent);
        if flipped != 0 {
            let next_player = bitboard::opponent_flip(opponent, flipped);
            best_score = -solve1(ctx, next_player, -beta, sq2);
            if best_score >= beta {
                return best_score;
            }

            if bitboard::has_adjacent_bit(opponent, sq2) {
                flipped = flip::flip(sq2, player, opponent);
                if flipped != 0 {
                    let next_player = bitboard::opponent_flip(opponent, flipped);
                    let score = -solve1(ctx, next_player, -beta, sq1);
                    return score.max(best_score);
                }
            }
            return best_score;
        }
    }

    if bitboard::has_adjacent_bit(opponent, sq2) {
        flipped = flip::flip(sq2, player, opponent);
        if flipped != 0 {
            let next_player = bitboard::opponent_flip(opponent, flipped);
            return -solve1(ctx, next_player, -beta, sq1);
        }
    }

    ctx.increment_nodes();
    if bitboard::has_adjacent_bit(player, sq1) {
        flipped = flip::flip(sq1, opponent, player);
        if flipped != 0 {
            let next_player = bitboard::opponent_flip(player, flipped);
            best_score = solve1(ctx, next_player, alpha, sq2);
            if best_score <= alpha {
                return best_score;
            }

            if bitboard::has_adjacent_bit(player, sq2) {
                flipped = flip::flip(sq2, opponent, player);
                if flipped != 0 {
                    let next_player = bitboard::opponent_flip(player, flipped);
                    let score = solve1(ctx, next_player, alpha, sq1);
                    return score.min(best_score);
                }
            }
            return best_score;
        }
    }

    if bitboard::has_adjacent_bit(player, sq2) {
        flipped = flip::flip(sq2, opponent, player);
        if flipped != 0 {
            let next_player = bitboard::opponent_flip(player, flipped);
            return solve1(ctx, next_player, alpha, sq1);
        }
    }

    // both players pass
    solve(board, 2)
}

/// Specialized solver for positions with exactly 1 empty square.
///
/// # Arguments
///
/// * `ctx` - Search context for node counting
/// * `player` - Current player's bitboard
/// * `alpha` - Score threshold
/// * `sq` - The single remaining empty square
///
/// # Returns
///
/// Exact final score after optimal play
#[inline(always)]
fn solve1(ctx: &mut SearchContext, player: u64, alpha: Score, sq: Square) -> Score {
    ctx.increment_nodes();
    let mut n_flipped = count_last_flip(player, sq);
    let mut score = 2 * player.count_ones() as Score - 64 + 2 + n_flipped;

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

/// Calculates the final score when no moves remain for either player.
///
/// # Scoring Rules
///
/// - If scores are tied: Empty squares are split (returns 0)
/// - If current player ahead: Gets all empty squares
/// - If current player behind: Gets no empty squares
///
/// # Arguments
///
/// * `board` - Final board position
/// * `n_empties` - Number of remaining empty squares
///
/// # Returns
///
/// Final score in disc difference
#[inline(always)]
pub fn solve(board: &Board, n_empties: u32) -> Score {
    let score = board.get_player_count() as Score * 2 - 64;
    let diff = score + n_empties as Score;

    match diff.cmp(&0) {
        Ordering::Equal => diff,
        Ordering::Greater => diff + n_empties as Score,
        Ordering::Less => score,
    }
}

/// Calculates the final score of a completed game (0 empty squares).
///
/// # Arguments
///
/// * `board` - Board position with all 64 squares filled
///
/// # Returns
///
/// Score as disc difference (positive = current player won)
#[inline(always)]
pub fn calculate_final_score(board: &Board) -> Score {
    board.get_player_count() as Score * 2 - 64
}
