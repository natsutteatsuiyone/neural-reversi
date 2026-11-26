mod cache;

use std::cell::RefCell;

use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::legal_moves::LegalMoves;
use reversi_core::score::DiscScore;
use reversi_core::score::{DISC_SCORE_INF, DISC_SCORE_MAX};
use reversi_core::search::{Empties, odd_quadrants, quadrant, solve_last1};
use reversi_core::square::Square;
use reversi_core::types::Depth;
use reversi_core::{flip, stability};

use crate::move_list::{MoveList, evaluate_moves_fast};
use crate::search::context::SearchContext;

use self::cache::{CacheIndex, EndGameCache};

/// Depth threshold for switching from PVS to null-window endgame search.
pub const DEPTH_TO_NWS: Depth = 11;

/// Depth threshold for switching to specialized shallow search.
const DEPTH_TO_SHALLOW_SEARCH: Depth = 6;

const EC_CACHE_BYTES: usize = 2 * 1024 * 1024;
const SHALLOW_CACHE_BYTES: usize = 256 * 1024;

struct EndGameCaches {
    ec: EndGameCache,
    shallow: EndGameCache,
}

#[derive(Clone, Copy)]
struct CacheSlot {
    index: CacheIndex,
    key: u64,
}

thread_local! {
    static ENDGAME_CACHES: RefCell<EndGameCaches> = RefCell::new(EndGameCaches {
        ec: EndGameCache::new(EC_CACHE_BYTES),
        shallow: EndGameCache::new(SHALLOW_CACHE_BYTES),
    });
}

/// Dispatches to the optimal solver based on empty square count.
pub fn null_window_search(ctx: &mut SearchContext, board: &Board, alpha: DiscScore) -> DiscScore {
    let empties = Empties::new(board);
    let n_empties = empties.count();

    if n_empties > DEPTH_TO_SHALLOW_SEARCH {
        return ENDGAME_CACHES.with_borrow_mut(|caches| {
            let (ec, sc) = (&mut caches.ec, &mut caches.shallow);
            null_window_search_with_ec(ctx, board, alpha, ec, sc, empties)
        });
    }

    match n_empties {
        0 => board.final_disc_diff(),
        1 => {
            let [sq] = empties.first();
            solve1(ctx, board, alpha, sq)
        }
        2 => {
            let [sq1, sq2] = empties.first();
            solve2(ctx, board, alpha, sq1, sq2)
        }
        3 => {
            let [sq1, sq2, sq3] = empties.first();
            solve3(ctx, board, alpha, sq1, sq2, sq3)
        }
        4 => {
            let (sq1, sq2, sq3, sq4) = sort_last4(empties, odd_quadrants(board.empty_squares()));
            solve4(ctx, board, alpha, sq1, sq2, sq3, sq4)
        }
        _ => ENDGAME_CACHES.with_borrow_mut(|caches| {
            shallow_search(ctx, board, alpha, &mut caches.shallow, empties)
        }),
    }
}

/// Performs a null window search with endgame cache probing.
///
/// `empties` must be the empty squares of `board`.
fn null_window_search_with_ec(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: DiscScore,
    ec: &mut EndGameCache,
    sc: &mut EndGameCache,
    empties: Empties,
) -> DiscScore {
    let n_empties = empties.count();
    let beta = alpha + 1;

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let hash_key = board.hash();
    let cache_idx = ec.index(hash_key);
    if let Some(entry) = ec.probe(cache_idx, hash_key)
        && entry.can_cut(beta)
    {
        return entry.score;
    }

    let moves = board.legal_moves();
    if moves.is_empty() {
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -null_window_search_with_ec(ctx, &next, -beta, ec, sc, empties);
        } else {
            return board.solve_disc_diff(n_empties);
        }
    }

    if let Some(sq) = moves.only_move() {
        let flipped = flip::flip(sq, board.player(), board.opponent());
        if flipped == board.opponent() {
            return DISC_SCORE_MAX;
        }

        let next = board.make_move_with_flipped(flipped, sq);
        let score = search_move_nws_ec(ctx, &next, empties.without(sq), beta, ec, sc);
        ec.store(cache_idx, hash_key, score, beta);
        return score;
    }

    let mut move_list = MoveList::with_moves(board, moves);
    if move_list.wipeout_move().is_some() {
        return DISC_SCORE_MAX;
    }

    let mut best_score = -DISC_SCORE_INF;

    evaluate_moves_fast(&mut move_list, ctx, board, None);
    for mv in move_list.best_first_iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        let score = search_move_nws_ec(ctx, &next, empties.without(mv.sq), beta, ec, sc);

        if score > best_score {
            best_score = score;
            if score >= beta {
                break;
            }
        }
    }

    ec.store(cache_idx, hash_key, best_score, beta);

    best_score
}

/// Searches a move with null window, dispatching to shallow or EC search based on depth.
///
/// `next_empties` must be the empty squares of `next`.
#[inline(always)]
fn search_move_nws_ec(
    ctx: &mut SearchContext,
    next: &Board,
    next_empties: Empties,
    beta: DiscScore,
    ec: &mut EndGameCache,
    sc: &mut EndGameCache,
) -> DiscScore {
    ctx.increment_nodes();
    if next_empties.count() <= DEPTH_TO_SHALLOW_SEARCH {
        -shallow_search(ctx, next, -beta, sc, next_empties)
    } else {
        -null_window_search_with_ec(ctx, next, -beta, ec, sc, next_empties)
    }
}

/// Performs a null window search optimized for shallow endgame positions.
///
/// `empties` must be the empty squares of `board`.
fn shallow_search(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: DiscScore,
    sc: &mut EndGameCache,
    empties: Empties,
) -> DiscScore {
    let n_empties = empties.count();
    let beta = alpha + 1;

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
    }

    let moves = board.legal_moves();
    if moves.is_empty() {
        ctx.increment_nodes();
        let next = board.switch_players();
        if next.has_legal_moves() {
            return -shallow_search(ctx, &next, -beta, sc, empties);
        } else {
            return board.solve_disc_diff(n_empties);
        }
    }

    let hash_key = board.hash();
    let cache_slot = CacheSlot {
        index: sc.index(hash_key),
        key: hash_key,
    };
    if let Some(entry) = sc.probe(cache_slot.index, cache_slot.key)
        && entry.can_cut(beta)
    {
        return entry.score;
    }

    let mut best_score = -DISC_SCORE_INF;

    // Split moves by quadrant parity: odd-empty quadrants first
    let node = ShallowNode {
        empties,
        odd_quadrants: odd_quadrants(board.empty_squares()),
        beta,
    };
    let odd_moves = moves & node.odd_quadrants;
    let even_moves = moves & !node.odd_quadrants;

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        odd_moves.corners(),
        cache_slot,
        node,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        odd_moves.non_corners(),
        cache_slot,
        node,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        even_moves.corners(),
        cache_slot,
        node,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        even_moves.non_corners(),
        cache_slot,
        node,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    sc.store(cache_slot.index, cache_slot.key, best_score, beta);

    best_score
}

/// A [`shallow_search`] node as its moves see it.
#[derive(Clone, Copy)]
struct ShallowNode {
    /// Empty squares of the node.
    empties: Empties,
    /// [`odd_quadrants`] of the node.
    odd_quadrants: Bitboard,
    /// Upper bound of the node's null window.
    beta: DiscScore,
}

/// Evaluates a single move in shallow search.
#[inline(always)]
fn shallow_search_move(
    ctx: &mut SearchContext,
    board: &Board,
    sq: Square,
    node: ShallowNode,
    sc: &mut EndGameCache,
) -> DiscScore {
    let next = board.make_move(sq);
    ctx.increment_nodes();
    let next_alpha = -node.beta;
    let next_empties = node.empties.without(sq);
    if next_empties.count() == 4 {
        let next_hash_key = next.hash();
        let next_cache_idx = sc.index(next_hash_key);
        if let Some(entry) = sc.probe(next_cache_idx, next_hash_key)
            && entry.can_cut(next_alpha + 1)
        {
            -entry.score
        } else if let Some(score) = stability::stability_cutoff(&next, 4, next_alpha) {
            -score
        } else {
            let next_odd_quadrants = node.odd_quadrants ^ quadrant(sq);
            let (sq1, sq2, sq3, sq4) = sort_last4(next_empties, next_odd_quadrants);
            let score = solve4(ctx, &next, next_alpha, sq1, sq2, sq3, sq4);
            sc.store(next_cache_idx, next_hash_key, score, next_alpha + 1);
            -score
        }
    } else {
        -shallow_search(ctx, &next, next_alpha, sc, next_empties)
    }
}

/// Searches all moves in a bitboard, returning [`Some`] with the score on a beta cutoff.
#[inline(always)]
fn shallow_search_moves(
    ctx: &mut SearchContext,
    board: &Board,
    moves: LegalMoves,
    cache_slot: CacheSlot,
    node: ShallowNode,
    best_score: &mut DiscScore,
    sc: &mut EndGameCache,
) -> Option<DiscScore> {
    let beta = node.beta;
    for sq in moves.iter() {
        let score = shallow_search_move(ctx, board, sq, node, sc);

        if score > *best_score {
            *best_score = score;
            if score >= beta {
                sc.store(cache_slot.index, cache_slot.key, score, beta);
                return Some(score);
            }
        }
    }

    None
}

/// Sorts the last four empty squares so that odd-parity quadrants are searched first.
///
/// `odd_quadrants` must be [`odd_quadrants`] of the four squares in `empties`.
#[inline(always)]
fn sort_last4(empties: Empties, odd_quadrants: Bitboard) -> (Square, Square, Square, Square) {
    let [sq1, sq2, sq3, sq4] = empties.first();

    if odd_quadrants.is_empty() {
        return (sq1, sq2, sq3, sq4);
    }

    if !odd_quadrants.contains(sq1) {
        if odd_quadrants.contains(sq2) {
            if odd_quadrants.contains(sq3) {
                (sq2, sq3, sq1, sq4)
            } else {
                (sq2, sq4, sq1, sq3)
            }
        } else if odd_quadrants.contains(sq3) {
            (sq3, sq4, sq1, sq2)
        } else {
            (sq1, sq2, sq3, sq4)
        }
    } else if !odd_quadrants.contains(sq2) {
        if odd_quadrants.contains(sq3) {
            (sq1, sq3, sq2, sq4)
        } else {
            (sq1, sq4, sq2, sq3)
        }
    } else {
        (sq1, sq2, sq3, sq4)
    }
}

/// Plays `sq`, an empty square, returning [`None`] if it flips no disc.
#[inline(always)]
fn try_make_move(board: &Board, sq: Square) -> Option<Board> {
    if !board.opponent().has_adjacent_bit(sq) {
        return None;
    }
    let flipped = flip::flip(sq, board.player(), board.opponent());
    if flipped.is_empty() {
        return None;
    }
    Some(board.make_move_with_flipped(flipped, sq))
}

/// Solves positions with exactly 4 empty squares.
fn solve4(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: DiscScore,
    sq1: Square,
    sq2: Square,
    sq3: Square,
    sq4: Square,
) -> DiscScore {
    let beta = alpha + 1;
    let mut best_score = -DISC_SCORE_INF;

    if let Some(next) = try_make_move(board, sq1) {
        best_score = -solve3(ctx, &next, -beta, sq2, sq3, sq4);
        if best_score > alpha {
            return best_score;
        }
    }

    if let Some(next) = try_make_move(board, sq2) {
        let score = -solve3(ctx, &next, -beta, sq1, sq3, sq4);
        if score > alpha {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = try_make_move(board, sq3) {
        let score = -solve3(ctx, &next, -beta, sq1, sq2, sq4);
        if score > alpha {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = try_make_move(board, sq4) {
        let score = -solve3(ctx, &next, -beta, sq1, sq2, sq3);
        return score.max(best_score);
    }

    if best_score == -DISC_SCORE_INF {
        let pass = board.switch_players();
        if pass.has_legal_moves() {
            best_score = -solve4(ctx, &pass, -beta, sq1, sq2, sq3, sq4);
        } else {
            best_score = board.solve_disc_diff(4);
        }
    }

    best_score
}

/// Solves positions with exactly 3 empty squares.
fn solve3(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: DiscScore,
    sq1: Square,
    sq2: Square,
    sq3: Square,
) -> DiscScore {
    ctx.increment_nodes();
    let beta = alpha + 1;
    let mut best_score = -DISC_SCORE_INF;

    // player moves
    if let Some(next) = try_make_move(board, sq1) {
        best_score = -solve2(ctx, &next, -beta, sq2, sq3);
        if best_score > alpha {
            return best_score;
        }
    }

    if let Some(next) = try_make_move(board, sq2) {
        let score = -solve2(ctx, &next, -beta, sq1, sq3);
        if score > alpha {
            return score;
        }
        best_score = score.max(best_score);
    }

    if let Some(next) = try_make_move(board, sq3) {
        let score = -solve2(ctx, &next, -beta, sq1, sq2);
        return score.max(best_score);
    }

    if best_score != -DISC_SCORE_INF {
        return best_score;
    }

    // opponent moves
    ctx.increment_nodes();
    best_score = DISC_SCORE_INF;
    let pass = board.switch_players();

    if let Some(next) = try_make_move(&pass, sq1) {
        best_score = solve2(ctx, &next, alpha, sq2, sq3);
        if best_score <= alpha {
            return best_score;
        }
    }

    if let Some(next) = try_make_move(&pass, sq2) {
        let score = solve2(ctx, &next, alpha, sq1, sq3);
        if score <= alpha {
            return score;
        }
        best_score = score.min(best_score);
    }

    if let Some(next) = try_make_move(&pass, sq3) {
        let score = solve2(ctx, &next, alpha, sq1, sq2);
        return score.min(best_score);
    }

    if best_score != DISC_SCORE_INF {
        return best_score;
    }

    board.solve_disc_diff(3)
}

/// Solves positions with exactly 2 empty squares.
#[inline(always)]
fn solve2(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: DiscScore,
    sq1: Square,
    sq2: Square,
) -> DiscScore {
    ctx.increment_nodes();
    let beta = alpha + 1;

    // player moves
    if let Some(next) = try_make_move(board, sq1) {
        let best_score = -solve1(ctx, &next, -beta, sq2);
        if best_score > alpha {
            return best_score;
        }
        if let Some(next) = try_make_move(board, sq2) {
            let score = -solve1(ctx, &next, -beta, sq1);
            return score.max(best_score);
        } else {
            return best_score;
        }
    } else if let Some(next) = try_make_move(board, sq2) {
        return -solve1(ctx, &next, -beta, sq1);
    }

    // opponent moves
    ctx.increment_nodes();
    let pass = board.switch_players();
    if let Some(next) = try_make_move(&pass, sq1) {
        let best_score = solve1(ctx, &next, alpha, sq2);
        if best_score <= alpha {
            return best_score;
        }
        if let Some(next) = try_make_move(&pass, sq2) {
            let score = solve1(ctx, &next, alpha, sq1);
            return score.min(best_score);
        } else {
            return best_score;
        }
    } else if let Some(next) = try_make_move(&pass, sq2) {
        return solve1(ctx, &next, alpha, sq1);
    }

    // both players pass
    board.solve_disc_diff(2)
}

/// Solves positions with exactly 1 empty square.
#[inline(always)]
fn solve1(ctx: &mut SearchContext, board: &Board, alpha: DiscScore, sq: Square) -> DiscScore {
    ctx.increment_nodes();
    solve_last1(board.player(), alpha, sq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_last4_puts_odd_quadrants_first() {
        use Square::{A1, A3, A6, A8, C1, C8, F1, H1, H6, H8};

        // Empties in list order, then the expected search order.
        let cases = [
            ([A1, H1, C1, F1], (A1, H1, C1, F1)), // every quadrant even
            ([A1, H1, A8, H8], (A1, H1, A8, H8)), // every quadrant odd
            ([A1, H1, C1, A3], (A1, H1, C1, A3)), // three empties in one quadrant
            ([A1, C1, A6, H6], (A6, H6, A1, C1)), // even pair 1st and 2nd
            ([A1, H1, C1, A6], (H1, A6, A1, C1)), // even pair 1st and 3rd
            ([A1, H1, A8, C1], (H1, A8, A1, C1)), // even pair 1st and 4th
            ([A1, H1, F1, A6], (A1, A6, H1, F1)), // even pair 2nd and 3rd
            ([A1, H1, A8, F1], (A1, A8, H1, F1)), // even pair 2nd and 4th
            ([A1, H1, A8, C8], (A1, H1, A8, C8)), // even pair 3rd and 4th
        ];
        for (empties, expected) in cases {
            let empty_bits = empties.iter().fold(0u64, |bits, sq| bits | 1 << sq.index());
            let board = Board::from_bitboards(!empty_bits, 0u64);
            let odd_quadrants = odd_quadrants(board.empty_squares());
            assert_eq!(
                sort_last4(Empties::new(&board), odd_quadrants),
                expected,
                "{empties:?}"
            );
        }
    }
}
