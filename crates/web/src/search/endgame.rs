use std::cell::RefCell;

use reversi_core::board::Board;
use reversi_core::constants::{SCORE_INF, SCORE_MAX};
use reversi_core::count_last_flip::count_last_flip;
use reversi_core::search::endgame_cache::EndGameCache;
use reversi_core::square::Square;
use reversi_core::types::{Depth, Score};
use reversi_core::{bitboard, stability};

use crate::move_list::{MoveList, evaluate_moves_fast};
use crate::search::search_context::SearchContext;

/// Quadrant masks for move ordering in shallow search.
#[rustfmt::skip]
const QUADRANT_MASK: [u64; 16] = [
    0x0000000000000000, 0x000000000F0F0F0F, 0x00000000F0F0F0F0, 0x00000000FFFFFFFF,
    0x0F0F0F0F00000000, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0FF0F0F0F0, 0x0F0F0F0FFFFFFFFF,
    0xF0F0F0F000000000, 0xF0F0F0F00F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0xF0F0F0F0FFFFFFFF,
    0xFFFFFFFF00000000, 0xFFFFFFFF0F0F0F0F, 0xFFFFFFFFF0F0F0F0, 0xFFFFFFFFFFFFFFFF
];

/// Depth threshold for switching from PVS to null-window endgame search.
pub const DEPTH_TO_NWS: Depth = 11;

/// Depth threshold for switching to specialized shallow search.
const DEPTH_TO_SHALLOW_SEARCH: Depth = 6;

struct EndGameCaches {
    ec: EndGameCache,
    shallow: EndGameCache,
}

thread_local! {
    static ENDGAME_CACHES: RefCell<EndGameCaches> = RefCell::new(EndGameCaches {
        ec: EndGameCache::new(128 * 1024),
        shallow: EndGameCache::new(128 * 1024),
    });
}

/// Dispatches to the optimal solver based on empty square count.
pub fn null_window_search(ctx: &mut SearchContext, board: &Board, alpha: Score) -> Score {
    let n_empties = ctx.empty_list.count();

    if n_empties > DEPTH_TO_SHALLOW_SEARCH {
        return ENDGAME_CACHES.with_borrow_mut(|caches| {
            null_window_search_with_ec(ctx, board, alpha, &mut caches.ec, &mut caches.shallow)
        });
    }

    match n_empties {
        0 => board.final_score(),
        1 => {
            let sq = ctx.empty_list.first();
            solve1(ctx, board, alpha, sq)
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
        _ => ENDGAME_CACHES
            .with_borrow_mut(|caches| shallow_search(ctx, board, alpha, &mut caches.shallow)),
    }
}

/// Performs a null window search with endgame cache probing.
fn null_window_search_with_ec(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    ec: &mut EndGameCache,
    sc: &mut EndGameCache,
) -> Score {
    let n_empties = ctx.empty_list.count();
    let beta = alpha + 1;

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let key = board.hash();
    if let Some(score) = ec.probe(key, board, alpha) {
        return score;
    }

    let mut move_list = MoveList::new(board);
    if move_list.wipeout_move().is_some() {
        return SCORE_MAX;
    } else if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -null_window_search_with_ec(ctx, &next, -beta, ec, sc);
        } else {
            return board.solve(n_empties);
        }
    }

    let mut best_score = -SCORE_INF;
    if move_list.count() >= 2 {
        evaluate_moves_fast(&mut move_list, ctx, board, Square::None);
        for mv in move_list.best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.update_endgame(mv.sq);
            let score = if ctx.empty_list.count() <= DEPTH_TO_SHALLOW_SEARCH {
                -shallow_search(ctx, &next, -beta, sc)
            } else {
                -null_window_search_with_ec(ctx, &next, -beta, ec, sc)
            };
            ctx.undo_endgame(mv.sq);

            if score > best_score {
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
        best_score = if ctx.empty_list.count() <= DEPTH_TO_SHALLOW_SEARCH {
            -shallow_search(ctx, &next, -beta, sc)
        } else {
            -null_window_search_with_ec(ctx, &next, -beta, ec, sc)
        };
        ctx.undo_endgame(mv.sq);
    }

    ec.store(key, board, alpha, best_score);

    best_score
}

/// Performs a null window search optimized for shallow endgame positions.
fn shallow_search(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sc: &mut EndGameCache,
) -> Score {
    let n_empties = ctx.empty_list.count();
    let beta = alpha + 1;

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let moves = board.get_moves();
    if moves.is_empty() {
        ctx.update_pass();
        let next = board.switch_players();
        ctx.undo_pass();
        if next.has_legal_moves() {
            return -shallow_search(ctx, &next, -beta, sc);
        } else {
            return board.solve(n_empties);
        }
    }

    let key = board.hash();
    if let Some(score) = sc.probe(key, board, alpha) {
        return score;
    }

    let mut best_score = -SCORE_INF;

    // Split moves by quadrant parity: odd-empty quadrants first
    let quadrant_mask = bitboard::Bitboard::new(QUADRANT_MASK[ctx.empty_list.parity() as usize]);
    let odd_moves = moves & quadrant_mask;
    let even_moves = moves & !quadrant_mask;

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        odd_moves.corners(),
        key,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        odd_moves.non_corners(),
        key,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        even_moves.corners(),
        key,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        even_moves.non_corners(),
        key,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    sc.store(key, board, alpha, best_score);

    best_score
}

/// Evaluates a single move in shallow search.
#[inline(always)]
fn shallow_search_move(
    ctx: &mut SearchContext,
    board: &Board,
    sq: Square,
    beta: Score,
    sc: &mut EndGameCache,
) -> Score {
    let next = board.make_move(sq);
    ctx.update_endgame(sq);
    let next_alpha = -beta;
    let score = if ctx.empty_list.count() == 4 {
        let next_key = next.hash();
        if let Some(score) = sc.probe(next_key, &next, next_alpha) {
            -score
        } else if let Some(score) = stability::stability_cutoff(&next, 4, next_alpha) {
            -score
        } else {
            let (sq1, sq2, sq3, sq4) = sort_empties_at_4(ctx);
            let score = solve4(ctx, &next, next_alpha, sq1, sq2, sq3, sq4);
            sc.store(next_key, &next, next_alpha, score);
            -score
        }
    } else {
        -shallow_search(ctx, &next, -beta, sc)
    };
    ctx.undo_endgame(sq);
    score
}

/// Searches all moves in a bitboard, returning [`Some`] with the score on a beta cutoff.
#[inline(always)]
fn shallow_search_moves(
    ctx: &mut SearchContext,
    board: &Board,
    moves: bitboard::Bitboard,
    key: u64,
    beta: Score,
    best_score: &mut Score,
    sc: &mut EndGameCache,
) -> Option<Score> {
    for sq in moves.iter() {
        let score = shallow_search_move(ctx, board, sq, beta, sc);

        if score > *best_score {
            *best_score = score;
            if score >= beta {
                sc.store(key, board, beta - 1, score);
                return Some(score);
            }
        }
    }

    None
}

/// Sorts the four remaining empty squares by quadrant parity for optimal search order.
#[inline(always)]
fn sort_empties_at_4(ctx: &mut SearchContext) -> (Square, Square, Square, Square) {
    let (sq1, quad_id1) = ctx.empty_list.first_and_quad_id();
    let (sq2, quad_id2) = ctx.empty_list.next_and_quad_id(sq1);
    let (sq3, quad_id3) = ctx.empty_list.next_and_quad_id(sq2);
    let sq4 = ctx.empty_list.next(sq3);
    let parity = ctx.empty_list.parity();

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

/// Solves positions with exactly 4 empty squares.
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
            best_score = board.solve(4);
        }
    }

    best_score
}

/// Solves positions with exactly 3 empty squares.
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

    board.solve(3)
}

/// Solves positions with exactly 2 empty squares.
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
    board.solve(2)
}

/// Solves positions with exactly 1 empty square.
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
