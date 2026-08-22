//! Specialized exact solvers for the final endgame plies.
//!
//! Contains solvers for positions with one to four empty squares and the
//! parity ordering used before the four-empty solver.

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
mod solve1_bmi2;
#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
use solve1_bmi2 as solve1_imp;

#[cfg(target_arch = "aarch64")]
mod solve1_neon;
#[cfg(target_arch = "aarch64")]
use solve1_neon as solve1_imp;

#[cfg(all(
    not(target_arch = "aarch64"),
    not(all(target_arch = "x86_64", target_feature = "bmi2"))
))]
mod solve1_scalar;
#[cfg(all(
    not(target_arch = "aarch64"),
    not(all(target_arch = "x86_64", target_feature = "bmi2"))
))]
use solve1_scalar as solve1_imp;

mod solve1_tables;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::constants::SCORE_INF;
use crate::empty_list::EmptyList;
use crate::flip;
use crate::search::context::SearchContext;
use crate::square::Square;
use crate::types::Score;

/// Sorts the last four empty squares so that odd-parity quadrants are searched first.
///
/// Nearly half of all four-empty positions (~48% measured on hard-20) have
/// all-even quadrant counts and keep the presorted order, so that case is
/// checked from `parity` alone before any quadrant lookup.
#[inline(always)]
pub(super) fn sort_last4(empty_list: &EmptyList) -> (Square, Square, Square, Square) {
    let parity = empty_list.parity();

    if parity == 0 {
        let sq1 = empty_list.first();
        let sq2 = empty_list.next(sq1);
        let sq3 = empty_list.next(sq2);
        let sq4 = empty_list.next(sq3);
        return (sq1, sq2, sq3, sq4);
    }

    let (sq1, quad_id1) = empty_list.first_and_quad_id();
    let (sq2, quad_id2) = empty_list.next_and_quad_id(sq1);
    let (sq3, quad_id3) = empty_list.next_and_quad_id(sq2);
    let sq4 = empty_list.next(sq3);

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
/// Uses the eager SIMD flip path when available, otherwise the guarded fallback.
///
/// Both paths produce identical scores and node counts.
#[inline(always)]
pub(super) fn solve4(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
    sq4: Square,
) -> Score {
    cfg_select! {
        all(
            target_arch = "x86_64",
            target_feature = "avx512cd",
            target_feature = "avx512vl"
        ) => solve4_eager(ctx, board, alpha, sq1, sq2, sq3, sq4),
        all(target_arch = "x86_64", target_feature = "avx2") => {
            solve4_eager(ctx, board, alpha, sq1, sq2, sq3, sq4)
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            solve4_eager(ctx, board, alpha, sq1, sq2, sq3, sq4)
        }
        all(target_arch = "wasm32", target_feature = "simd128") => {
            solve4_eager(ctx, board, alpha, sq1, sq2, sq3, sq4)
        }
        _ => solve4_fallback(ctx, board, alpha, sq1, sq2, sq3, sq4),
    }
}

/// Eager-SIMD `solve4` dispatch target. A non-empty flip already
/// implies an adjacent opponent disc, so dropping the `has_adjacent_bit`
/// guard is safe; the `is_empty` checks subsume it. Like
/// `solve4_fallback`, this function does not count itself as a node.
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512cd",
        target_feature = "avx512vl"
    ),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
fn solve4_eager(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
    sq4: Square,
) -> Score {
    let player = board.player();
    let opponent = board.opponent();
    let beta = alpha + 1;
    let mut best_score = -SCORE_INF;

    // Player to move.
    let (fp1, fp2, fp3, fp4) = flip::flip4(sq1, sq2, sq3, sq4, player, opponent);

    if !fp1.is_empty() {
        best_score = -solve3(
            ctx,
            &board.make_move_with_flipped(fp1, sq1),
            -beta,
            sq2,
            sq3,
            sq4,
        );
        if best_score >= beta {
            return best_score;
        }
    }

    if !fp2.is_empty() {
        let score = -solve3(
            ctx,
            &board.make_move_with_flipped(fp2, sq2),
            -beta,
            sq1,
            sq3,
            sq4,
        );
        if score >= beta {
            return score;
        }
        best_score = score.max(best_score);
    }

    if !fp3.is_empty() {
        let score = -solve3(
            ctx,
            &board.make_move_with_flipped(fp3, sq3),
            -beta,
            sq1,
            sq2,
            sq4,
        );
        if score >= beta {
            return score;
        }
        best_score = score.max(best_score);
    }

    if !fp4.is_empty() {
        let score = -solve3(
            ctx,
            &board.make_move_with_flipped(fp4, sq4),
            -beta,
            sq1,
            sq2,
            sq3,
        );
        return score.max(best_score);
    }

    if best_score != -SCORE_INF {
        return best_score;
    }

    solve4_eager_pass(ctx, board, alpha, sq1, sq2, sq3, sq4)
}

/// Cold continuation of [`solve4_eager`] for positions where the side to move
/// has no legal move on any of the four empties.
///
/// Kept out of line so the hot path does not keep the shared flip masks and
/// board broadcasts alive (spilled to the stack) across its `solve3` calls
/// just to rematerialize them for this rare opponent-side retry.
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512cd",
        target_feature = "avx512vl"
    ),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[cold]
#[inline(never)]
fn solve4_eager_pass(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
    sq4: Square,
) -> Score {
    let player = board.player();
    let opponent = board.opponent();
    let pass = board.switch_players();
    let mut best_score = SCORE_INF;
    let (fo1, fo2, fo3, fo4) = flip::flip4(sq1, sq2, sq3, sq4, opponent, player);

    if !fo1.is_empty() {
        best_score = solve3(
            ctx,
            &pass.make_move_with_flipped(fo1, sq1),
            alpha,
            sq2,
            sq3,
            sq4,
        );
        if best_score <= alpha {
            return best_score;
        }
    }

    if !fo2.is_empty() {
        let score = solve3(
            ctx,
            &pass.make_move_with_flipped(fo2, sq2),
            alpha,
            sq1,
            sq3,
            sq4,
        );
        if score <= alpha {
            return score;
        }
        best_score = score.min(best_score);
    }

    if !fo3.is_empty() {
        let score = solve3(
            ctx,
            &pass.make_move_with_flipped(fo3, sq3),
            alpha,
            sq1,
            sq2,
            sq4,
        );
        if score <= alpha {
            return score;
        }
        best_score = score.min(best_score);
    }

    if !fo4.is_empty() {
        let score = solve3(
            ctx,
            &pass.make_move_with_flipped(fo4, sq4),
            alpha,
            sq1,
            sq2,
            sq3,
        );
        return score.min(best_score);
    }

    if best_score == SCORE_INF {
        return board.solve(4);
    }

    best_score
}

/// Fallback `solve4` dispatch target: the original
/// `try_make_move`-guarded, short-circuiting path, kept so the scalar
/// fallback is not slowed by eager unguarded flips.
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512cd",
        target_feature = "avx512vl"
    ),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
fn solve4_fallback(
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
        best_score = SCORE_INF;

        if let Some(next) = pass.try_make_move(sq1) {
            best_score = solve3(ctx, &next, alpha, sq2, sq3, sq4);
            if best_score <= alpha {
                return best_score;
            }
        }

        if let Some(next) = pass.try_make_move(sq2) {
            let score = solve3(ctx, &next, alpha, sq1, sq3, sq4);
            if score <= alpha {
                return score;
            }
            best_score = score.min(best_score);
        }

        if let Some(next) = pass.try_make_move(sq3) {
            let score = solve3(ctx, &next, alpha, sq1, sq2, sq4);
            if score <= alpha {
                return score;
            }
            best_score = score.min(best_score);
        }

        if let Some(next) = pass.try_make_move(sq4) {
            let score = solve3(ctx, &next, alpha, sq1, sq2, sq3);
            return score.min(best_score);
        }

        if best_score == SCORE_INF {
            return board.solve(4);
        }
    }

    best_score
}

/// Specialized solver for positions with exactly 3 empty squares.
///
/// Uses the eager SIMD flip path when available, otherwise the guarded fallback.
///
/// Both paths produce identical scores and node counts.
#[inline(always)]
pub(super) fn solve3(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
) -> Score {
    cfg_select! {
        all(
            target_arch = "x86_64",
            target_feature = "avx512cd",
            target_feature = "avx512vl"
        ) => solve3_eager(ctx, board, alpha, sq1, sq2, sq3),
        all(target_arch = "x86_64", target_feature = "avx2") => {
            solve3_eager(ctx, board, alpha, sq1, sq2, sq3)
        }
        all(target_arch = "aarch64", target_feature = "neon") => {
            solve3_eager(ctx, board, alpha, sq1, sq2, sq3)
        }
        all(target_arch = "wasm32", target_feature = "simd128") => {
            solve3_eager(ctx, board, alpha, sq1, sq2, sq3)
        }
        _ => solve3_fallback(ctx, board, alpha, sq1, sq2, sq3),
    }
}

/// Eager-SIMD `solve3` dispatch target. A non-empty flip already
/// implies an adjacent opponent disc, so dropping the `has_adjacent_bit`
/// guard is safe; the `is_empty` checks subsume it.
#[cfg(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512cd",
        target_feature = "avx512vl"
    ),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
))]
fn solve3_eager(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
    sq3: Square,
) -> Score {
    ctx.increment_nodes();
    let player = board.player();
    let opponent = board.opponent();
    let beta = alpha + 1;
    let mut best_score = -SCORE_INF;

    // Player to move.
    let (fp1, fp2, fp3) = flip::flip3(sq1, sq2, sq3, player, opponent);

    if !fp1.is_empty() {
        best_score = -solve2(
            ctx,
            &board.make_move_with_flipped(fp1, sq1),
            -beta,
            sq2,
            sq3,
        );
        if best_score >= beta {
            return best_score;
        }
    }

    if !fp2.is_empty() {
        let score = -solve2(
            ctx,
            &board.make_move_with_flipped(fp2, sq2),
            -beta,
            sq1,
            sq3,
        );
        if score >= beta {
            return score;
        }
        best_score = score.max(best_score);
    }

    if !fp3.is_empty() {
        let score = -solve2(
            ctx,
            &board.make_move_with_flipped(fp3, sq3),
            -beta,
            sq1,
            sq2,
        );
        return score.max(best_score);
    }

    if best_score != -SCORE_INF {
        return best_score;
    }

    ctx.increment_nodes();
    let pass = board.switch_players();
    best_score = SCORE_INF;
    let (fo1, fo2, fo3) = flip::flip3(sq1, sq2, sq3, opponent, player);

    if !fo1.is_empty() {
        best_score = solve2(ctx, &pass.make_move_with_flipped(fo1, sq1), alpha, sq2, sq3);
        if best_score <= alpha {
            return best_score;
        }
    }

    if !fo2.is_empty() {
        let score = solve2(ctx, &pass.make_move_with_flipped(fo2, sq2), alpha, sq1, sq3);
        if score <= alpha {
            return score;
        }
        best_score = score.min(best_score);
    }

    if !fo3.is_empty() {
        let score = solve2(ctx, &pass.make_move_with_flipped(fo3, sq3), alpha, sq1, sq2);
        return score.min(best_score);
    }

    if best_score != SCORE_INF {
        return best_score;
    }

    board.solve(3)
}

/// Fallback `solve3` dispatch target: the original
/// `try_make_move`-guarded, short-circuiting path, kept so the scalar
/// fallback is not slowed by eager unguarded flips.
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512cd",
        target_feature = "avx512vl"
    ),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
fn solve3_fallback(
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
#[inline(always)]
pub(super) fn solve2(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
) -> Score {
    cfg_select! {
        all(
            target_arch = "x86_64",
            target_feature = "avx512cd",
            target_feature = "avx512vl"
        ) => solve2_eager(ctx, board, alpha, sq1, sq2),
        _ => solve2_fallback(ctx, board, alpha, sq1, sq2),
    }
}

/// AVX-512 `solve2` dispatch target. A non-empty flip already
/// implies an adjacent opponent disc, so dropping the `has_adjacent_bit`
/// guards is safe; the `is_empty` checks subsume them.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512cd",
    target_feature = "avx512vl"
))]
#[inline(always)]
fn solve2_eager(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
) -> Score {
    ctx.increment_nodes();
    let player = board.player();
    let opponent = board.opponent();
    let beta = alpha + 1;

    // Player to move.
    let (fp1, fp2) = flip::flip2(sq1, sq2, player, opponent);
    if !fp1.is_empty() {
        let best_score = -solve1(ctx, opponent.apply_flip(fp1), -beta, sq2);
        if best_score >= beta {
            return best_score;
        }
        if !fp2.is_empty() {
            let score = -solve1(ctx, opponent.apply_flip(fp2), -beta, sq1);
            return score.max(best_score);
        }
        return best_score;
    } else if !fp2.is_empty() {
        return -solve1(ctx, opponent.apply_flip(fp2), -beta, sq1);
    }

    // Player passed at both empties: opponent to move.
    ctx.increment_nodes();
    let (fo1, fo2) = flip::flip2(sq1, sq2, opponent, player);
    if !fo1.is_empty() {
        let best_score = solve1(ctx, player.apply_flip(fo1), alpha, sq2);
        if best_score <= alpha {
            return best_score;
        }
        if !fo2.is_empty() {
            let score = solve1(ctx, player.apply_flip(fo2), alpha, sq1);
            return score.min(best_score);
        }
        return best_score;
    } else if !fo2.is_empty() {
        return solve1(ctx, player.apply_flip(fo2), alpha, sq1);
    }

    // both players pass
    board.solve(2)
}

/// Fallback `solve2` dispatch target (non-AVX-512): the original
/// `has_adjacent_bit`-guarded, short-circuiting path, kept so the scalar/AVX2
/// fallback is not slowed by eager unguarded flips.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512cd",
    target_feature = "avx512vl"
)))]
#[inline(always)]
fn solve2_fallback(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: Score,
    sq1: Square,
    sq2: Square,
) -> Score {
    ctx.increment_nodes();
    let player = board.player();
    let opponent = board.opponent();
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
#[inline(always)]
pub(super) fn solve1(ctx: &mut SearchContext, player: Bitboard, alpha: Score, sq: Square) -> Score {
    ctx.increment_nodes();
    solve_last1(player, alpha, sq)
}

/// Scores a position with exactly one empty square.
#[inline(always)]
pub(super) fn solve_last1(player: Bitboard, alpha: Score, sq: Square) -> Score {
    solve1_imp::solve1(player.bits(), alpha, sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{SCORE_MAX, SCORE_MIN};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn last_flip_count(player: Bitboard, sq: Square) -> Score {
        let sq_bit = 1u64 << sq.index();
        let player = Bitboard::new(player.bits() & !sq_bit);
        let opponent = Bitboard::new(!player.bits() & !sq_bit);
        2 * flip::flip(sq, player, opponent).bits().count_ones() as Score
    }

    #[test]
    fn solve_last1_matches_direct_counts() {
        let mut rng = StdRng::seed_from_u64(0x51d3_f00d);
        for _ in 0..2048 {
            let p: u64 = rng.random();
            for sq_idx in 0..64u8 {
                let sq_bit = 1u64 << sq_idx;
                if p & sq_bit != 0 {
                    continue;
                }

                let sq = Square::from_u8(sq_idx).unwrap();
                let player = Bitboard::new(p);
                let alpha = rng.random_range(SCORE_MIN..=SCORE_MAX);
                let mut n_flipped = last_flip_count(player, sq);
                let mut expected = 2 * player.count() as Score - SCORE_MAX + 2 + n_flipped;

                if n_flipped == 0 {
                    let score_if_opp_passes = if expected > 0 { expected } else { expected - 2 };
                    if score_if_opp_passes > alpha {
                        n_flipped = last_flip_count(!player, sq);
                        expected = if n_flipped > 0 {
                            expected - 2 - n_flipped
                        } else {
                            score_if_opp_passes
                        };
                    } else {
                        expected = score_if_opp_passes;
                    }
                }

                assert_eq!(solve_last1(player, alpha, sq), expected);
            }
        }
    }

    /// The original branchy `sort_last4` logic, kept as the behavioral reference.
    fn sort_last4_reference(empty_list: &EmptyList) -> (Square, Square, Square, Square) {
        let (sq1, quad_id1) = empty_list.first_and_quad_id();
        let (sq2, quad_id2) = empty_list.next_and_quad_id(sq1);
        let (sq3, quad_id3) = empty_list.next_and_quad_id(sq2);
        let sq4 = empty_list.next(sq3);
        let parity = empty_list.parity();

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

    #[test]
    fn sort_last4_matches_branchy_reference() {
        // One pool of four distinct squares per quadrant (masks 1, 2, 4, 8).
        const POOLS: [[Square; 4]; 4] = [
            [Square::B2, Square::C2, Square::B3, Square::D3],
            [Square::G2, Square::F2, Square::G3, Square::E3],
            [Square::B7, Square::C7, Square::B6, Square::D6],
            [Square::G7, Square::F7, Square::G6, Square::E6],
        ];

        // Every assignment of the four empties to quadrants (4^4 patterns)
        // covers all reachable parity/membership combinations.
        for pattern in 0..256usize {
            let mut used = [0usize; 4];
            let mut empty_bits = 0u64;
            for i in 0..4 {
                let quad = (pattern >> (2 * i)) & 3;
                let sq = POOLS[quad][used[quad]];
                used[quad] += 1;
                empty_bits |= 1 << sq.index();
            }

            let occupied = !empty_bits;
            let player = occupied & 0xAAAA_AAAA_AAAA_AAAA;
            let opponent = occupied & !0xAAAA_AAAA_AAAA_AAAAu64;
            let board = Board::from_bitboards(player, opponent);
            let empty_list = EmptyList::new(&board);

            assert_eq!(
                sort_last4(&empty_list),
                sort_last4_reference(&empty_list),
                "pattern {pattern:#010b}"
            );
        }
    }
}
