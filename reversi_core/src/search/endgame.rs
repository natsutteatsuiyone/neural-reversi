//! # References:
//! - https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/endgame.c
//! - https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/search.cpp

use std::cell::RefCell;
use std::cmp::Ordering;
use std::sync::Arc;

use crate::board::Board;
use crate::constants::{SCORE_INF, SCORE_MAX, to_endgame_score, to_midgame_score};
use crate::count_last_flip::count_last_flip;
use crate::move_list::{ConcurrentMoveIterator, MoveList};
use crate::probcut::NO_SELECTIVITY;
use crate::search::endgame_cache::{ENDGAME_CACHE_MAX_EMPTIES, EndGameCache, EndGameCacheEntry};
use crate::search::enhanced_transposition_cutoff;
use crate::search::node_type::{NodeType, NonPV, PV, Root};
use crate::search::search_context::SearchContext;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::transposition_table::Bound;
use crate::types::{Depth, Score, Scoref};
use crate::{bitboard, probcut, stability};

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

/// Depth threshold for switching to specialized shallow search.
const DEPTH_TO_SHALLOW_SEARCH: Depth = 7;

/// Minimum depth required for parallel search splitting.
const MIN_SPLIT_DEPTH: Depth = 7;

/// Minimum depth for enhanced transposition table cutoff.
const MIN_ETC_DEPTH: Depth = 6;

/// Depth threshold for switching from midgame to endgame search.
pub const DEPTH_MIDGAME_TO_ENDGAME: Depth = 13;

/// Depth threshold for endgame cache null window search.
const EC_NWS_DEPTH: Depth = ENDGAME_CACHE_MAX_EMPTIES;

thread_local! {
    static ENDGAME_CACHE: RefCell<EndGameCache> =
        RefCell::new(EndGameCache::new(16));
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
    let level = task.level;
    let multi_pv = task.multi_pv;
    let mut ctx = SearchContext::new(
        &board,
        task.generation,
        task.selectivity,
        task.tt.clone(),
        task.eval.clone(),
    );

    if let Some(ref callback) = task.callback {
        ctx.set_callback(callback.clone());
    }

    let n_empties = ctx.empty_list.count;
    let score = estimate_aspiration_base_score(&mut ctx, &board, n_empties, thread);

    ctx.selectivity = NO_SELECTIVITY;
    ctx.game_phase = GamePhase::EndGame;

    let mut best_score = 0;
    let mut alpha = score - 3;
    let mut beta = score + 3;

    let num_root_moves = ctx.root_moves_count();
    let pv_count = if multi_pv { num_root_moves } else { 1 };

    for pv_idx in 0..pv_count {
        if pv_idx >= 1 {
            alpha = -SCORE_INF;
            beta = best_score;
        }

        let mut best_move_sq = Square::None;
        for selectivity in 1..=NO_SELECTIVITY {
            if level.get_end_depth(selectivity) < n_empties {
                break;
            }

            ctx.selectivity = selectivity;
            let mut delta = 2;

            loop {
                best_score = search::<Root, false>(&mut ctx, &board, alpha, beta, thread, None);

                if thread.is_search_aborted() {
                    break;
                }

                if best_score <= alpha {
                    beta = alpha;
                    alpha = (best_score - delta).max(-SCORE_INF);
                } else if best_score >= beta {
                    alpha = (beta - delta).max(alpha);
                    beta = (best_score + delta).min(SCORE_INF);
                } else {
                    break;
                }

                delta += delta;
            }

            let best_move = ctx.get_best_root_move(true).unwrap();
            alpha = (best_score - 2).max(-SCORE_INF);
            beta = (best_score + 2).min(SCORE_INF);
            best_move_sq = best_move.sq;

            ctx.notify_progress(
                n_empties,
                best_score as Scoref,
                best_move.sq,
                ctx.selectivity,
            );

            if thread.is_search_aborted() {
                break;
            }
        }

        ctx.mark_root_move_searched(best_move_sq);

        let best_move = ctx.get_best_root_move(false).unwrap();
        best_score = best_move.score;

        if thread.is_search_aborted() {
            return SearchResult {
                score: best_score as Scoref,
                best_move: Some(best_move.sq),
                n_nodes: ctx.n_nodes,
                pv_line: best_move.pv,
                depth: n_empties,
                selectivity: ctx.selectivity,
            };
        }
    }

    let rm = ctx.get_best_root_move(false).unwrap();
    SearchResult {
        score: best_score as Scoref,
        best_move: Some(rm.sq),
        n_nodes: ctx.n_nodes,
        pv_line: rm.pv,
        depth: n_empties,
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
    thread: &Arc<Thread>,
) -> i32 {
    ctx.game_phase = GamePhase::MidGame;
    let midgame_depth = n_empties / 2;

    let hash_key = board.hash();
    let (tt_hit, tt_data, _) = ctx.tt.probe(hash_key, ctx.generation);
    let score = if tt_hit && tt_data.bound == Bound::Exact && tt_data.depth >= midgame_depth {
        tt_data.score
    } else if n_empties >= 22 {
        ctx.selectivity = 0;
        midgame::search::<PV, false>(
            ctx,
            board,
            midgame_depth,
            -SCORE_INF,
            SCORE_INF,
            thread,
            None,
        )
    } else if n_empties >= 12 {
        midgame::evaluate_depth2(ctx, board, -SCORE_INF, SCORE_INF)
    } else {
        midgame::evaluate(ctx, board)
    };

    to_endgame_score(score)
}

/// Alpha-beta search function for endgame positions.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
/// * `SP_NODE` - Whether this is a split point node in parallel search.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state and statistics.
/// * `board` - Current board position to solve.
/// * `alpha` - Lower bound of the search window.
/// * `beta` - Upper bound of the search window.
/// * `thread` - Thread handle for parallel search coordination.
/// * `split_point` - Optional split point for parallel search nodes.
///
/// # Returns
///
/// The exact score in disc difference.
pub fn search<NT: NodeType, const SP_NODE: bool>(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: Score,
    beta: Score,
    thread: &Arc<Thread>,
    split_point: Option<&Arc<SplitPoint>>,
) -> Score {
    let org_alpha = alpha;
    let n_empties = ctx.empty_list.count;
    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;
    let move_iter: Arc<ConcurrentMoveIterator>;
    let tt_key;
    let tt_entry_index;

    if SP_NODE {
        let sp_state = split_point.as_ref().unwrap().state();
        best_move = sp_state.best_move;
        best_score = sp_state.best_score;
        move_iter = sp_state.move_iter.clone().unwrap();
        tt_key = 0;
        tt_entry_index = 0;
    } else {
        let n_empties = ctx.empty_list.count;

        if NT::PV_NODE {
            if n_empties == 0 {
                return calculate_final_score(board);
            }
        } else {
            if n_empties <= DEPTH_MIDGAME_TO_ENDGAME {
                return null_window_search(ctx, board, alpha);
            }

            if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
                return score;
            }
        }

        tt_key = board.hash();
        ctx.tt.prefetch(tt_key);

        let mut move_list = MoveList::new(board);
        if move_list.count() == 0 {
            let next = board.switch_players();
            if next.has_legal_moves() {
                ctx.update_pass();
                let score = -search::<NT, false>(ctx, &next, -beta, -alpha, thread, None);
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
        let (tt_hit, tt_data, _tt_entry_index) = ctx.tt.probe(tt_key, ctx.generation);
        tt_entry_index = _tt_entry_index;
        let tt_move = if tt_hit {
            tt_data.best_move
        } else {
            Square::None
        };

        if !NT::PV_NODE
            && tt_hit
            && tt_data.depth >= n_empties
            && tt_data.selectivity >= ctx.selectivity
            && tt_data.should_cut(to_midgame_score(beta))
        {
            return to_endgame_score(tt_data.score);
        }

        if !NT::PV_NODE && n_empties >= MIN_ETC_DEPTH {
            if let Some(score) = enhanced_transposition_cutoff(
                ctx,
                board,
                &move_list,
                n_empties,
                to_midgame_score(alpha),
                tt_key,
                tt_entry_index,
            ) {
                return to_endgame_score(score);
            }
        }

        if !NT::PV_NODE
            && let Some(score) =
                probcut::probcut_endgame(ctx, board, n_empties, alpha, beta, thread)
        {
            return score;
        }

        if move_list.count() > 1 {
            move_list.evaluate_moves::<NT>(ctx, board, n_empties, tt_move);
            move_list.sort();
        }

        move_iter = Arc::new(ConcurrentMoveIterator::new(move_list));
    }

    while let Some((mv, move_count)) = move_iter.next() {
        if NT::ROOT_NODE && ctx.is_move_searched(mv.sq) {
            continue;
        }

        if SP_NODE {
            split_point.as_ref().unwrap().unlock();
        }

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv);

        let mut score = -SCORE_INF;
        if !NT::PV_NODE || move_count > 1 {
            if SP_NODE {
                let sp_state = split_point.as_ref().unwrap().state();
                alpha = sp_state.alpha;
            }
            score = -search::<NonPV, false>(ctx, &next, -(alpha + 1), -alpha, thread, None);
        }

        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            ctx.clear_pv();
            score = -search::<PV, false>(ctx, &next, -beta, -alpha, thread, None);
        }

        ctx.undo(mv);

        if SP_NODE {
            let sp = split_point.as_ref().unwrap();
            sp.lock();
            let sp_state = sp.state();
            best_score = sp_state.best_score;
            alpha = sp_state.alpha;
        }

        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return 0;
        }

        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, alpha);
        }

        if score > best_score {
            if SP_NODE {
                let sp_state = split_point.as_ref().unwrap().state_mut();
                sp_state.best_score = score;
            }
            best_score = score;

            if score > alpha {
                if SP_NODE {
                    let sp_state = split_point.as_ref().unwrap().state_mut();
                    sp_state.best_move = mv.sq;
                }
                best_move = mv.sq;

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                }

                if NT::PV_NODE && score < beta {
                    if SP_NODE {
                        let sp_state = split_point.as_ref().unwrap().state_mut();
                        sp_state.alpha = score;
                    }
                    alpha = score;
                } else {
                    if SP_NODE {
                        let sp = split_point.as_ref().unwrap();
                        sp.state_mut().cutoff = true;
                    }

                    break;
                }
            }
        }

        if !SP_NODE && n_empties >= MIN_SPLIT_DEPTH && move_iter.count() > 1 && thread.can_split() {
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

    if SP_NODE {
        return best_score;
    }

    ctx.tt.store(
        tt_entry_index,
        tt_key,
        to_midgame_score(best_score),
        Bound::determine_bound::<NT>(best_score, org_alpha, beta),
        n_empties,
        best_move,
        ctx.selectivity,
        ctx.generation,
    );

    best_score
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
pub fn null_window_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let beta = alpha + 1;

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let mut move_list = MoveList::new(board);
    if move_list.wipeout_move.is_some() {
        return SCORE_MAX;
    } else if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -null_window_search(ctx, &next, -beta);
        } else {
            return solve(board, n_empties);
        }
    }

    // transposition table lookup
    let (tt_hit, tt_data, tt_entry_index) = ctx.tt.probe(tt_key, ctx.generation);
    let tt_move = if tt_hit {
        tt_data.best_move
    } else {
        Square::None
    };

    if tt_hit
        && tt_data.depth >= n_empties
        && tt_data.selectivity >= ctx.selectivity
        && tt_data.should_cut(to_midgame_score(beta))
    {
        return to_endgame_score(tt_data.score);
    }

    let mut best_score = -SCORE_INF;
    let mut best_move = Square::None;
    if move_list.count() >= 2 {
        move_list.evaluate_moves_fast(board, tt_move);
        for mv in move_list.best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);

            ctx.update_endgame(mv.sq);
            let score = if ctx.empty_list.count <= EC_NWS_DEPTH {
                -null_window_search_with_ec(ctx, &next, -beta)
            } else {
                -null_window_search(ctx, &next, -beta)
            };
            ctx.undo_endgame(mv.sq);

            if score > best_score {
                best_move = mv.sq;
                best_score = score;
                if score >= beta {
                    break;
                }
            }
        }
    } else {
        // only one move available
        let mv = move_list.first().unwrap();
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update_endgame(mv.sq);
        best_score = if ctx.empty_list.count <= EC_NWS_DEPTH {
            -null_window_search_with_ec(ctx, &next, -beta)
        } else {
            -null_window_search(ctx, &next, -beta)
        };
        ctx.undo_endgame(mv.sq);
        best_move = mv.sq;
    }

    ctx.tt.store(
        tt_entry_index,
        tt_key,
        to_midgame_score(best_score),
        Bound::determine_bound::<NonPV>(best_score, alpha, beta),
        n_empties,
        best_move,
        NO_SELECTIVITY,
        ctx.generation,
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
fn probe_endgame_cache(key: u64, n_empties: Depth) -> Option<EndGameCacheEntry> {
    ENDGAME_CACHE.with(|cell| {
        let cache = cell.borrow();
        cache.probe(key, n_empties)
    })
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
fn store_endgame_cache(
    key: u64,
    n_empties: Depth,
    alpha: Score,
    beta: Score,
    score: Score,
    best_move: Square,
) {
    let bound = Bound::determine_bound::<NonPV>(score, alpha, beta);
    ENDGAME_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        cache.store(key, n_empties, score, bound, best_move);
    });
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

    let key = board.hash();
    let entry = probe_endgame_cache(key, n_empties);
    let mut tt_move = Square::None;
    if let Some(entry_data) = &entry {
        if entry_data.should_cut(beta) {
            return entry_data.score;
        }
        tt_move = entry_data.best_move;
    }

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

    let mut best_score = -SCORE_INF;
    let mut best_move = Square::None;
    if move_list.count() >= 2 {
        move_list.evaluate_moves_fast(board, tt_move);
        for mv in move_list.best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.update_endgame(mv.sq);
            let score = if ctx.empty_list.count <= DEPTH_TO_SHALLOW_SEARCH {
                -shallow_search(ctx, &next, -beta)
            } else {
                -null_window_search_with_ec(ctx, &next, -beta)
            };
            ctx.undo_endgame(mv.sq);

            if score > best_score {
                best_move = mv.sq;
                best_score = score;
                if score >= beta {
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

    store_endgame_cache(key, n_empties, alpha, beta, best_score, best_move);

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

    fn search_child(ctx: &mut SearchContext, next: &Board, beta: Score) -> Score {
        if ctx.empty_list.count == 4 {
            if let Some(score) = stability::stability_cutoff(next, 4, -beta) {
                -score
            } else {
                let (sq1, sq2, sq3, sq4) = sort_empties_at_4(ctx);
                -solve4(ctx, next, -beta, sq1, sq2, sq3, sq4)
            }
        } else {
            -shallow_search(ctx, next, -beta)
        }
    }

    let key = board.hash();
    let entry = probe_endgame_cache(key, n_empties);
    let mut tt_move = Square::None;
    if let Some(entry_data) = &entry {
        if entry_data.should_cut(beta) {
            return entry_data.score;
        }
        tt_move = entry_data.best_move;
    }

    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;
    if tt_move != Square::None {
        if let Some(next) = board.try_make_move(tt_move) {
            ctx.update_endgame(tt_move);
            let score = search_child(ctx, &next, beta);
            ctx.undo_endgame(tt_move);

            if score > best_score {
                if score >= beta {
                    store_endgame_cache(key, n_empties, alpha, beta, score, tt_move);
                    return score;
                }
                best_move = tt_move;
                best_score = score;
            }
        }
    }

    let mut moves = board.get_moves();
    if moves == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -shallow_search(ctx, &next, -beta);
        } else {
            return solve(board, n_empties);
        }
    } else if best_move != Square::None {
        moves &= !best_move.bitboard();
        if moves == 0 {
            store_endgame_cache(key, n_empties, alpha, beta, best_score, best_move);
            return best_score;
        }
    }

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let mut priority_moves = moves & QUADRANT_MASK[ctx.empty_list.parity as usize];
    if priority_moves == 0 {
        priority_moves = moves;
    }

    loop {
        moves ^= priority_moves;
        let mut sq = ctx.empty_list.first();
        loop {
            while !bitboard::is_set(priority_moves, sq) {
                sq = ctx.empty_list.next(sq);
            }

            priority_moves &= !sq.bitboard();
            let next = board.make_move(sq);

            ctx.update_endgame(sq);
            let score = search_child(ctx, &next, beta);
            ctx.undo_endgame(sq);

            if score > best_score {
                if score >= beta {
                    store_endgame_cache(key, n_empties, alpha, beta, score, sq);
                    return score;
                }
                best_move = sq;
                best_score = score;
            }

            if priority_moves == 0 {
                break;
            }
        }

        priority_moves = moves;
        if priority_moves == 0 {
            break;
        }
    }

    store_endgame_cache(key, n_empties, alpha, beta, best_score, best_move);

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
        if best_score > alpha {
            return best_score;
        }
    }

    if let Some(next) = board.try_make_move(sq2) {
        let score = -solve3(ctx, &next, -beta, sq1, sq3, sq4);
        if score > alpha {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = board.try_make_move(sq3) {
        let score = -solve3(ctx, &next, -beta, sq1, sq2, sq4);
        if score > alpha {
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
        if best_score > alpha {
            return best_score;
        }
    }

    if let Some(next) = board.try_make_move(sq2) {
        let score = -solve2(ctx, &next, -beta, sq1, sq3);
        if score > alpha {
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
#[inline(always)]
fn solve2(ctx: &mut SearchContext, board: &Board, alpha: Score, sq1: Square, sq2: Square) -> Score {
    ctx.increment_nodes();
    let beta = alpha + 1;

    // player moves
    if let Some(next) = board.try_make_move(sq1) {
        let best_score = -solve1(ctx, &next, -beta, sq2);
        if best_score > alpha {
            return best_score;
        }
        if let Some(next) = board.try_make_move(sq2) {
            let score = -solve1(ctx, &next, -beta, sq1);
            return score.max(best_score);
        } else {
            return best_score;
        }
    } else if let Some(next) = board.try_make_move(sq2) {
        return -solve1(ctx, &next, -beta, sq1);
    }

    // opponent moves
    ctx.increment_nodes();
    let pass = board.switch_players();
    if let Some(next) = pass.try_make_move(sq1) {
        let best_score = solve1(ctx, &next, alpha, sq2);
        if best_score <= alpha {
            return best_score;
        }
        if let Some(next) = pass.try_make_move(sq2) {
            let score = solve1(ctx, &next, alpha, sq1);
            return score.min(best_score);
        } else {
            return best_score;
        }
    } else if let Some(next) = pass.try_make_move(sq2) {
        return solve1(ctx, &next, alpha, sq1);
    }

    // both players pass
    solve(board, 2)
}

/// Specialized solver for positions with exactly 1 empty square.
///
/// # Arguments
///
/// * `ctx` - Search context for node counting
/// * `board` - Current board position
/// * `alpha` - Score threshold (for pruning opponent check)
/// * `sq` - The single remaining empty square
///
/// # Returns
///
/// Exact final score after optimal play
#[inline(always)]
fn solve1(ctx: &mut SearchContext, board: &Board, alpha: Score, sq: Square) -> Score {
    ctx.increment_nodes();
    let mut score = board.get_player_count() as Score * 2 - 64 + 2;
    let mut n_flipped = count_last_flip(board.player, sq);
    score += n_flipped;

    if n_flipped == 0 {
        // pass
        let score2 = score - 2;
        if score <= 0 {
            score = score2;
        }

        if score > alpha {
            n_flipped = count_last_flip(board.opponent, sq);
            if n_flipped != 0 {
                score = score2 - n_flipped;
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
