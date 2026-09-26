//! Move list extensions for reversi_web.

pub use reversi_core::move_list::MoveList;

use reversi_core::{
    bitboard::Bitboard,
    board::Board,
    move_list::Move,
    square::Square,
    types::{Depth, ScaledScore},
};

use crate::search::{self, context::SearchContext, strategy::SearchStrategy};

/// Ordering value assigned to wipeout moves.
const WIPEOUT_VALUE: i32 = 1 << 30;

/// Ordering value assigned to moves suggested by the transposition table.
const TT_MOVE_VALUE: i32 = 1 << 20;

/// Assigns ordering values to each move in the list.
pub fn evaluate_moves<SS: SearchStrategy>(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    tt_move: Square,
) {
    // Minimum depth required for shallow search evaluation based on empty squares
    // When depth is below this threshold, use fast heuristic evaluation instead
    #[rustfmt::skip]
    const MIN_DEPTH: [u32; 64] = [
        19, 18, 18, 18, 17, 17, 17, 16,  // 0-7 empty squares
        16, 16, 15, 15, 15, 14, 14, 14,  // 8-15 empty squares
        13, 13, 13, 12, 12, 12, 11, 11,  // 16-23 empty squares
        11, 10, 10, 10, 9,  9,  9,  9,   // 24-31 empty squares
        9,  9,  9,  9,  9,  9,  9,  9,   // 32-39 empty squares
        9,  9,  9,  9,  9,  9,  9,  9,   // 40-47 empty squares
        9,  9,  9,  9,  9,  9,  9,  9,   // 48-55 empty squares
        9,  9,  9,  9,  9,  9,  9,  9    // 56-63 empty squares
    ];

    if depth < MIN_DEPTH[ctx.empty_list.count() as usize] {
        evaluate_moves_fast(move_list, ctx, board, tt_move);
        return;
    }

    evaluate_moves_shallow::<SS>(move_list, ctx, board, depth, tt_move);
}

/// Assigns ordering values from a shallow search, with endgame mobility adjustments.
///
/// `SS::IS_ENDGAME` is an associated constant, so the phase-specific branches
/// fold away when the function is monomorphized.
fn evaluate_moves_shallow<SS: SearchStrategy>(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    tt_move: Square,
) {
    const MOBILITY_SCALE: i32 = ScaledScore::SCALE * 2;
    const POTENTIAL_MOBILITY_SCALE: i32 = ScaledScore::SCALE;

    let sort_depth = if SS::IS_ENDGAME {
        match depth {
            0..=18 => 0,
            19..=26 => 1,
            _ => 2,
        }
    } else {
        match depth {
            0..=15 => 0,
            16..=25 => 1,
            _ => 2,
        }
    };

    for mv in move_list.iter_mut() {
        if mv.flipped == board.opponent() {
            // Wipeout move
            mv.value = WIPEOUT_VALUE;
        } else if mv.sq == tt_move {
            // Transposition table move
            mv.value = TT_MOVE_VALUE;
        } else {
            // Evaluate using shallow search
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.update(mv.sq, mv.flipped);

            let score = match sort_depth {
                0 => -search::evaluate(ctx, &next),
                1 => -search::evaluate_depth1(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                2 => -search::evaluate_depth2(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                _ => unreachable!(),
            };
            mv.value = score.value();

            if SS::IS_ENDGAME {
                let (moves, potential) = next.get_moves_and_potential();
                let mobility = moves.corner_weighted_count() as i32;
                let potential_mobility = potential.corner_weighted_count() as i32;
                mv.value -= mobility * MOBILITY_SCALE;
                mv.value -= potential_mobility * POTENTIAL_MOBILITY_SCALE;
            }

            ctx.undo(mv.sq);
        }
    }
}

/// Assigns ordering values using fast heuristics without shallow search.
pub(crate) fn evaluate_moves_fast(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    tt_move: Square,
) {
    // Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/move.c#L30
    #[rustfmt::skip]
    const SQUARE_VALUE: [i32; 64] = [
        18,  4, 16, 12, 12, 16,  4, 18,
         4,  2,  6,  8,  8,  6,  2,  4,
        16,  6, 14, 10, 10, 14,  6, 16,
        12,  8, 10,  0,  0, 10,  8, 12,
        12,  8, 10,  0,  0, 10,  8, 12,
        16,  6, 14, 10, 10, 14,  6, 16,
         4,  2,  6,  8,  8,  6,  2,  4,
        18,  4, 16, 12, 12, 16,  4, 18,
    ];

    const SQUARE_VALUE_WEIGHT: i32 = 128;
    const CORNER_STABILITY_WEIGHT: i32 = 2048;
    const MOBILITY_WEIGHT: i32 = 16384;

    let score = |sq: Square, next: &Board, moves: Bitboard| {
        let corner_stability = next.opponent().corner_stability() as i32;
        let weighted_mobility = moves.corner_weighted_count() as i32;
        let mut value = SQUARE_VALUE[sq.index()] * SQUARE_VALUE_WEIGHT;
        value += corner_stability * CORNER_STABILITY_WEIGHT;
        value += (36 - weighted_mobility) * MOBILITY_WEIGHT;
        value
    };

    // Children are generated in pairs so their mobility can share one
    // two-board move generation.
    let mut pending: Option<(&mut Move, Board)> = None;
    for mv in move_list.iter_mut() {
        if mv.flipped == board.opponent() {
            // Wipeout move (capture all opponent pieces)
            mv.value = WIPEOUT_VALUE;
        } else if mv.sq == tt_move {
            // Transposition table move
            mv.value = TT_MOVE_VALUE;
        } else {
            ctx.increment_nodes();
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            match pending.take() {
                None => pending = Some((mv, next)),
                Some((prev, prev_next)) => {
                    let (prev_moves, moves) = get_moves_pair(&prev_next, &next);
                    prev.value = score(prev.sq, &prev_next, prev_moves);
                    mv.value = score(mv.sq, &next, moves);
                }
            }
        }
    }
    if let Some((mv, next)) = pending {
        mv.value = score(mv.sq, &next, next.get_moves());
    }
}

/// Returns the legal moves of two boards.
#[inline(always)]
fn get_moves_pair(a: &Board, b: &Board) -> (Bitboard, Bitboard) {
    cfg_select! {
        all(target_arch = "wasm32", target_feature = "simd128") => {
            use core::arch::wasm32::*;

            let player = u64x2(a.player().bits(), b.player().bits());
            let opponent = u64x2(a.opponent().bits(), b.opponent().bits());
            let moves = get_moves_x2(player, opponent);
            (
                Bitboard::new(u64x2_extract_lane::<0>(moves)),
                Bitboard::new(u64x2_extract_lane::<1>(moves)),
            )
        }
        _ => (a.get_moves(), b.get_moves()),
    }
}

/// Lane-wise `get_moves` for two boards packed into `u64x2` lanes.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn get_moves_x2(
    player: core::arch::wasm32::v128,
    opponent: core::arch::wasm32::v128,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    let h_opp = v128_and(opponent, u64x2_splat(0x7E7E_7E7E_7E7E_7E7E));
    let shl = |x, n| u64x2_shl(x, n);
    let shr = |x, n| u64x2_shr(x, n);

    let mut flip7 = v128_and(h_opp, shl(player, 7));
    let mut flip9 = v128_and(h_opp, shl(player, 9));
    let mut flip8 = v128_and(opponent, shl(player, 8));
    let mut flip1 = v128_and(h_opp, shl(player, 1));

    flip7 = v128_or(flip7, v128_and(h_opp, shl(flip7, 7)));
    flip9 = v128_or(flip9, v128_and(h_opp, shl(flip9, 9)));
    flip8 = v128_or(flip8, v128_and(opponent, shl(flip8, 8)));
    let mut moves = i64x2_add(h_opp, flip1);

    let mut pre7 = v128_and(h_opp, shl(h_opp, 7));
    let mut pre9 = v128_and(h_opp, shl(h_opp, 9));
    let mut pre8 = v128_and(opponent, shl(opponent, 8));

    flip7 = v128_or(flip7, v128_and(pre7, shl(flip7, 14)));
    flip9 = v128_or(flip9, v128_and(pre9, shl(flip9, 18)));
    flip8 = v128_or(flip8, v128_and(pre8, shl(flip8, 16)));
    flip7 = v128_or(flip7, v128_and(pre7, shl(flip7, 14)));
    flip9 = v128_or(flip9, v128_and(pre9, shl(flip9, 18)));
    flip8 = v128_or(flip8, v128_and(pre8, shl(flip8, 16)));

    moves = v128_or(
        moves,
        v128_or(v128_or(shl(flip7, 7), shl(flip9, 9)), shl(flip8, 8)),
    );

    flip7 = v128_and(h_opp, shr(player, 7));
    flip9 = v128_and(h_opp, shr(player, 9));
    flip8 = v128_and(opponent, shr(player, 8));
    flip1 = v128_and(h_opp, shr(player, 1));

    flip7 = v128_or(flip7, v128_and(h_opp, shr(flip7, 7)));
    flip9 = v128_or(flip9, v128_and(h_opp, shr(flip9, 9)));
    flip8 = v128_or(flip8, v128_and(opponent, shr(flip8, 8)));
    flip1 = v128_or(flip1, v128_and(h_opp, shr(flip1, 1)));

    pre7 = shr(pre7, 7);
    pre9 = shr(pre9, 9);
    pre8 = shr(pre8, 8);
    let pre1 = v128_and(h_opp, shr(h_opp, 1));

    flip7 = v128_or(flip7, v128_and(pre7, shr(flip7, 14)));
    flip9 = v128_or(flip9, v128_and(pre9, shr(flip9, 18)));
    flip8 = v128_or(flip8, v128_and(pre8, shr(flip8, 16)));
    flip1 = v128_or(flip1, v128_and(pre1, shr(flip1, 2)));
    flip7 = v128_or(flip7, v128_and(pre7, shr(flip7, 14)));
    flip9 = v128_or(flip9, v128_and(pre9, shr(flip9, 18)));
    flip8 = v128_or(flip8, v128_and(pre8, shr(flip8, 16)));
    flip1 = v128_or(flip1, v128_and(pre1, shr(flip1, 2)));

    moves = v128_or(
        moves,
        v128_or(
            v128_or(shr(flip7, 7), shr(flip9, 9)),
            v128_or(shr(flip8, 8), shr(flip1, 1)),
        ),
    );
    v128_andnot(moves, v128_or(player, opponent))
}
