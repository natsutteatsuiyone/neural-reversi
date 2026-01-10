use std::cell::RefCell;
use std::cmp::Ordering;

use reversi_core::board::Board;
use reversi_core::constants::{SCORE_INF, SCORE_MAX};
use reversi_core::count_last_flip::count_last_flip;
use reversi_core::search::endgame_cache::{EndGameCache, EndGameCacheBound, EndGameCacheEntry};
use reversi_core::search::node_type::NonPV;
use reversi_core::square::Square;
use reversi_core::transposition_table::Bound;
use reversi_core::types::ScaledScore;
use reversi_core::types::{Depth, Score, Selectivity};
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

/// Depth threshold for switching to specialized shallow search.
const DEPTH_TO_SHALLOW_SEARCH: Depth = 7;

/// Depth threshold for endgame cache null window search.
const EC_NWS_DEPTH: Depth = 12;

thread_local! {
    static ENDGAME_CACHE: RefCell<EndGameCache> =
        RefCell::new(EndGameCache::new(16));
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
    let tt_key = board.hash();
    let tt_probe_result = ctx.tt.probe(tt_key);
    let tt_move = tt_probe_result.best_move();

    if let Some(tt_data) = tt_probe_result.data()
        && tt_data.depth() >= n_empties
        && tt_data.selectivity() >= ctx.selectivity
        && tt_data.can_cut(ScaledScore::from_disc_diff(beta))
    {
        return tt_data.score().to_disc_diff();
    }

    let mut best_score = -SCORE_INF;
    let mut best_move = Square::None;
    if move_list.count() >= 2 {
        evaluate_moves_fast(&mut move_list, ctx, board, tt_move);
        for mv in move_list.into_best_first_iter() {
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);

            ctx.update_endgame(mv.sq);
            let score = if ctx.empty_list.count <= EC_NWS_DEPTH {
                -null_window_search_with_ec(ctx, &next, -beta)
            } else {
                -null_window_search(ctx, &next, -beta)
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
        best_score = if ctx.empty_list.count <= EC_NWS_DEPTH {
            -null_window_search_with_ec(ctx, &next, -beta)
        } else {
            -null_window_search(ctx, &next, -beta)
        };
        ctx.undo_endgame(mv.sq);
        best_move = mv.sq;
    }

    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        ScaledScore::new(best_score),
        Bound::determine_bound::<NonPV>(best_score, alpha, beta),
        n_empties,
        best_move,
        Selectivity::None,
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
/// # Returns
///
/// * `Option<EndGameCacheEntry>` - The cached entry if found
#[inline(always)]
fn probe_endgame_cache(key: u64) -> Option<EndGameCacheEntry> {
    ENDGAME_CACHE.with(|cell| {
        let cache = cell.borrow();
        cache.probe(key)
    })
}

/// Store an entry in the endgame cache
///
/// # Arguments
///
/// * `key` - The hash key of the board position
/// * `beta` - The beta value for determining the bound
/// * `score` - The score to store
/// * `best_move` - The best move found in this position
#[inline(always)]
fn store_endgame_cache(key: u64, beta: Score, score: Score, best_move: Square) {
    let bound = EndGameCacheBound::determine_bound(score, beta);
    ENDGAME_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        cache.store(key, score, bound, best_move);
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

    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha) {
        return score;
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
        evaluate_moves_fast(&mut move_list, ctx, board, tt_move);
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
    let entry = probe_endgame_cache(key);
    let mut tt_move = Square::None;
    if let Some(entry_data) = &entry {
        if entry_data.can_cut(beta) {
            return entry_data.score;
        }
        tt_move = entry_data.best_move;
    }

    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;
    if tt_move != Square::None
        && let Some(next) = board.try_make_move(tt_move)
    {
        ctx.update_endgame(tt_move);
        let score = search_child(ctx, &next, beta);
        ctx.undo_endgame(tt_move);

        if score > best_score {
            if score >= beta {
                store_endgame_cache(key, beta, score, tt_move);
                return score;
            }
            best_move = tt_move;
            best_score = score;
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
            store_endgame_cache(key, beta, best_score, best_move);
            return best_score;
        }
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
                    store_endgame_cache(key, beta, score, sq);
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
