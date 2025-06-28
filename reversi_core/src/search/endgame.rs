use std::cmp::Ordering;
use std::sync::Arc;

use crate::board::Board;
use crate::constants::{EVAL_SCORE_SCALE_BITS, SCORE_INF};
use crate::count_last_flip::count_last_flip;
use crate::move_list::{ConcurrentMoveIterator, MoveList};
use crate::probcut::NO_SELECTIVITY;
use crate::search::search_context::SearchContext;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::transposition_table::Bound;
use crate::types::{Depth, NodeType, NonPV, Root, Score, Scoref, PV};
use crate::{bitboard, probcut, stability};

use super::search_context::GamePhase;
use super::search_result::SearchResult;
use super::threading::Thread;
use super::{midgame, SearchTask};

/// Quadrant masks used for prioritizing moves in shallow search.
#[rustfmt::skip]
const QUADRANT_MASK: [u64; 16] = [
    0x0000000000000000, 0x000000000F0F0F0F, 0x00000000F0F0F0F0, 0x00000000FFFFFFFF,
    0x0F0F0F0F00000000, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0FF0F0F0F0, 0x0F0F0F0FFFFFFFFF,
    0xF0F0F0F000000000, 0xF0F0F0F00F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0xF0F0F0F0FFFFFFFF,
    0xFFFFFFFF00000000, 0xFFFFFFFF0F0F0F0F, 0xFFFFFFFFF0F0F0F0, 0xFFFFFFFFFFFFFFFF
];

/// Depth at which to switch to shallow search.
const DEPTH_TO_SHALLOW_SEARCH: Depth = 7;
const MIN_SPLIT_DEPTH: Depth = 7;

pub const DEPTH_MIDGAME_TO_ENDGAME: Depth = 12;

/// Search the root position
///
/// # Arguments
///
/// * `ctx` - Search context containing game state and statistics
/// * `board` - Current board position
/// * `level` - Search level
/// * `multi_pv` - Flag indicating if multiple principal variations should be searched
///
/// # Returns
///
/// The score of the current position, the search depth, and the selectivity
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
    let mut best_score = 0;
    ctx.game_phase = GamePhase::EndGame;

    ctx.selectivity = NO_SELECTIVITY;
    let score = midgame::evaluate(&ctx, &board) >> EVAL_SCORE_SCALE_BITS;

    let mut alpha = score - 6;
    let mut beta = score + 6;

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
                    beta = (alpha + beta) / 2;
                    alpha = (best_score - delta).max(-SCORE_INF);
                } else if best_score >= beta {
                    alpha = (alpha + beta) / 2;
                    beta = (best_score + delta).min(SCORE_INF);
                } else {
                    break;
                }

                delta += delta / 2;
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

/// Performs an alpha-beta search with principal variation (PV) or non-PV nodes
///
/// # Arguments
/// * `ctx` - Search context containing game state and statistics
/// * `board` - Current board position
/// * `depth` - Remaining search depth
/// * `alpha` - Lower bound of the search window
/// * `beta` - Upper bound of the search window
///
/// # Returns
///
/// The score of the current position
pub fn search<NT: NodeType, const SP_NODE: bool>(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: Score,
    beta: Score,
    thread: &Arc<Thread>,
    split_point: Option<&Arc<SplitPoint>>,
) -> Score {
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
            && tt_data.should_cutoff(beta)
        {
            return tt_data.score;
        }

        if !NT::PV_NODE {
            if let Some(score) = probcut::probcut_endgame(ctx, board, n_empties, alpha, beta, thread) {
                return score;
            }
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

        if !SP_NODE
            && n_empties >= MIN_SPLIT_DEPTH
            && move_iter.count() > 1
            && thread.can_split()
        {
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
        best_score,
        Bound::determine_bound::<NT>(best_score, beta),
        n_empties,
        best_move,
        ctx.selectivity,
        ctx.generation,
    );

    best_score
}

pub fn null_window_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let mut best_score = -SCORE_INF;
    let n_empties = ctx.empty_list.count;

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let mut move_list = MoveList::new(board);
    if move_list.count() >= 2 {
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
            && tt_data.should_cutoff(alpha + 1)
        {
            return tt_data.score;
        }

        move_list.evaluate_moves_fast(board, tt_move);
        let mut best_move = Square::None;
        for mv in move_list.best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);

            ctx.update_endgame(mv.sq);
            let score = if ctx.empty_list.count <= DEPTH_TO_SHALLOW_SEARCH {
                -shallow_search(ctx, &next, -(alpha + 1))
            } else {
                -null_window_search(ctx, &next, -(alpha + 1))
            };
            ctx.undo_endgame(mv.sq);

            if score > best_score {
                best_move = mv.sq;
                best_score = score;
                if score > alpha {
                    break;
                }
            }
        }

        ctx.tt.store(
            tt_entry_index,
            tt_key,
            best_score,
            Bound::determine_bound::<NonPV>(best_score, alpha + 1),
            n_empties,
            best_move,
            NO_SELECTIVITY,
            ctx.generation,
        );
    } else if move_list.count() == 1 {
        let mv = move_list.first().unwrap();
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update_endgame(mv.sq);
        best_score = if ctx.empty_list.count <= DEPTH_TO_SHALLOW_SEARCH {
            -shallow_search(ctx, &next, -(alpha + 1))
        } else {
            -null_window_search(ctx, &next, -(alpha + 1))
        };
        ctx.undo_endgame(mv.sq);
    } else {
        let next = board.switch_players();
        if next.has_legal_moves() {
            best_score = -null_window_search(ctx, &next, -(alpha + 1));
        } else {
            best_score = solve(board, n_empties);
        }
    }

    best_score
}

pub fn shallow_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count;
    let mut moves = board.get_moves();
    if moves == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -shallow_search(ctx, &next, -(alpha + 1));
        } else {
            return solve(board, n_empties);
        }
    }

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let mut priority_moves = moves & QUADRANT_MASK[ctx.empty_list.parity as usize];
    if priority_moves == 0 {
        priority_moves = moves;
    }

    let mut best_score = -SCORE_INF;
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
            let score = if ctx.empty_list.count == 4 {
                if let Some(score) = stability::stability_cutoff(&next, 4, -(alpha + 1)) {
                    -score
                } else {
                    let (sq1, sq2, sq3, sq4) = sort_empties_at_4(ctx);
                    -solve4(ctx, &next, -(alpha + 1), sq1, sq2, sq3, sq4)
                }
            } else {
                -shallow_search(ctx, &next, -(alpha + 1))
            };
            ctx.undo_endgame(sq);

            if score > best_score {
                if score > alpha {
                    return score;
                }
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

    debug_assert!(best_score != -SCORE_INF);
    debug_assert!((-64..=64).contains(&alpha));
    best_score
}

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

fn solve4(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
    sq4: Square,
) -> Score {
    let mut best_score = -SCORE_INF;

    if let Some(next) = board.try_make_move(sq1) {
        best_score = -solve3(ctx, &next, -(alpha + 1), sq2, sq3, sq4);
        if best_score > alpha {
            return best_score;
        }
    }

    if let Some(next) = board.try_make_move(sq2) {
        let score = -solve3(ctx, &next, -(alpha + 1), sq1, sq3, sq4);
        if score > alpha {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = board.try_make_move(sq3) {
        let score = -solve3(ctx, &next, -(alpha + 1), sq1, sq2, sq4);
        if score > alpha {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = board.try_make_move(sq4) {
        let score = -solve3(ctx, &next, -(alpha + 1), sq1, sq2, sq3);
        return score.max(best_score);
    }

    if best_score == -SCORE_INF {
        let pass = board.switch_players();
        if pass.has_legal_moves() {
            best_score = -solve4(ctx, &pass, -(alpha + 1), sq1, sq2, sq3, sq4);
        } else {
            best_score = solve(board, 4);
        }
    }

    best_score
}

fn solve3(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
) -> Score {
    ctx.increment_nodes();
    let mut best_score = -SCORE_INF;

    // player moves
    if let Some(next) = board.try_make_move(sq1) {
        best_score = -solve2(ctx, &next, -(alpha + 1), sq2, sq3);
        if best_score > alpha {
            return best_score;
        }
    }

    if let Some(next) = board.try_make_move(sq2) {
        let score = -solve2(ctx, &next, -(alpha + 1), sq1, sq3);
        if score > alpha {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = board.try_make_move(sq3) {
        let score = -solve2(ctx, &next, -(alpha + 1), sq1, sq2);
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

#[inline(always)]
fn solve2(ctx: &mut SearchContext, board: &Board, alpha: Score, sq1: Square, sq2: Square) -> Score {
    ctx.increment_nodes();

    // player moves
    if let Some(next) = board.try_make_move(sq1) {
        let best_score = -solve1(ctx, &next, -(alpha + 1), sq2);
        if best_score > alpha {
            return best_score;
        }
        if let Some(next) = board.try_make_move(sq2) {
            let score = -solve1(ctx, &next, -(alpha + 1), sq1);
            return score.max(best_score);
        } else {
            return best_score;
        }
    } else if let Some(next) = board.try_make_move(sq2) {
        return -solve1(ctx, &next, -(alpha + 1), sq1);
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

#[inline(always)]
pub fn calculate_final_score(board: &Board) -> Score {
    board.get_player_count() as Score * 2 - 64
}
