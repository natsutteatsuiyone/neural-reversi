//! Move list extensions for reversi_web
//!
//! This module re-exports move list types from reversi_core and provides
//! web-specific implementation of evaluate_moves that uses the web search module.

// Re-export core move list types
pub use reversi_core::move_list::MoveList;

use reversi_core::{board::Board, square::Square, types::Depth};

use crate::{
    probcut,
    search::{self, search_context::SearchContext},
};

/// Value assigned to wipeout moves (capturing all opponent pieces).
const WIPEOUT_VALUE: i32 = 1 << 30;

/// Value assigned to moves suggested by the transposition table.
const TT_MOVE_VALUE: i32 = 1 << 20;

/// Web-specific implementation of move evaluation.
///
/// This function evaluates all moves to assign ordering values and reduction depths,
/// using the web search module's evaluation functions.
///
/// # Arguments
///
/// * `move_list` - The move list to evaluate
/// * `ctx` - Search context with transposition table and statistics
/// * `board` - Current position before making any move
/// * `depth` - Remaining search depth at this node
/// * `tt_move` - Best move from transposition table (if any)
/// * `is_endgame` - Whether we are in endgame search mode
pub fn evaluate_moves(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    tt_move: Square,
    is_endgame: bool,
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

    if depth <= MIN_DEPTH[ctx.empty_list.count() as usize] {
        evaluate_moves_fast(move_list, ctx, board, tt_move);
        return;
    }

    if is_endgame {
        evaluate_moves_endgame(move_list, ctx, board, depth, tt_move);
    } else {
        evaluate_moves_midgame(move_list, ctx, board, depth, tt_move);
    }
}

/// Evaluates moves specifically for midgame positions.
fn evaluate_moves_midgame(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    tt_move: Square,
) {
    use reversi_core::types::ScaledScore;

    const MAX_SORT_DEPTH: i32 = 2;
    let mut sort_depth = (depth as i32 - 15) / 3;
    sort_depth = sort_depth.clamp(0, MAX_SORT_DEPTH);

    let mut best_sort_value = i32::MIN;

    for mv in move_list.iter_mut() {
        if mv.flipped == board.opponent {
            // Wipeout move
            mv.value = WIPEOUT_VALUE;
        } else if mv.sq == tt_move {
            // Transposition table move
            mv.value = TT_MOVE_VALUE;
        } else {
            // Evaluate using shallow search
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.update(mv.sq, mv.flipped.0);

            let score = match sort_depth {
                0 => -search::evaluate(ctx, &next),
                1 => -search::evaluate_depth1(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                2 => -search::evaluate_depth2(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                _ => unreachable!(),
            };
            mv.value = score.value();

            ctx.undo(mv.sq);
            best_sort_value = best_sort_value.max(mv.value);
        }
    }

    if best_sort_value == i32::MIN || !ctx.selectivity.is_enabled() {
        return;
    }

    // Score-Based Reduction: reduce depth for poor moves
    // This implements a form of late move reduction based on evaluation scores,
    // using the same statistical error model as ProbCut.
    let sigma = probcut::get_sigma(ctx.ply(), sort_depth as Depth, depth);
    let sbr_margin = (ctx.selectivity.t_value() * sigma).ceil() as i32;
    if sbr_margin == 0 {
        return;
    }

    // best_lower_bound = best_sort_value - sbr_margin
    // other_upper_bound = mv.value + sbr_margin
    // Condition: best_lower_bound > other_upper_bound
    //         => best_sort_value - sbr_margin > mv.value + sbr_margin
    //         => mv.value < best_sort_value - 2 * sbr_margin
    let reduction_threshold = best_sort_value - 2 * sbr_margin;
    for mv in move_list.iter_mut() {
        if mv.value < reduction_threshold {
            mv.reduction_depth = 1;
        }
    }
}

/// Evaluates moves specifically for endgame positions.
fn evaluate_moves_endgame(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    tt_move: Square,
) {
    use reversi_core::types::ScaledScore;

    let sort_depth = match depth {
        0..=18 => 0,
        19..=26 => 1,
        _ => 2,
    };

    for mv in move_list.iter_mut() {
        if mv.flipped == board.opponent {
            // Wipeout move
            mv.value = WIPEOUT_VALUE;
        } else if mv.sq == tt_move {
            // Transposition table move
            mv.value = TT_MOVE_VALUE;
        } else {
            // Evaluate using shallow search
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            ctx.update(mv.sq, mv.flipped.0);

            let score = match sort_depth {
                0 => -search::evaluate(ctx, &next),
                1 => -search::evaluate_depth1(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                2 => -search::evaluate_depth2(ctx, &next, -ScaledScore::INF, ScaledScore::INF),
                _ => unreachable!(),
            };
            mv.value = score.value();

            const MOBILITY_SCALE: i32 = ScaledScore::SCALE * 2;
            const POTENTIAL_MOBILITY_SCALE: i32 = ScaledScore::SCALE;

            let (moves, potential) = next.get_moves_and_potential();
            let mobility = moves.corner_weighted_count() as i32;
            let potential_mobility = potential.corner_weighted_count() as i32;
            mv.value -= mobility * MOBILITY_SCALE;
            mv.value -= potential_mobility * POTENTIAL_MOBILITY_SCALE;

            ctx.undo(mv.sq);
        }
    }
}

pub fn evaluate_moves_fast(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    tt_move: Square,
) {
    /// Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/move.c#L30
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

    const SQUARE_VALUE_WEIGHT: i32 = 1 << 8;
    const CORNER_STABILITY_WEIGHT: i32 = 1 << 12;
    const POTENTIAL_MOBILITY_WEIGHT: i32 = 1 << 10;
    const MOBILITY_WEIGHT: i32 = 1 << 14;

    for mv in move_list.iter_mut() {
        mv.value = if mv.flipped == board.opponent {
            // Wipeout move (capture all opponent pieces)
            WIPEOUT_VALUE
        } else if mv.sq == tt_move {
            // Transposition table move
            TT_MOVE_VALUE
        } else {
            ctx.increment_nodes();
            let next = board.make_move_with_flipped(mv.flipped, mv.sq);
            let (moves, potential) = next.get_moves_and_potential();
            let potential_mobility = potential.corner_weighted_count() as i32;
            let corner_stability = next.opponent.corner_stability() as i32;
            let weighted_mobility = moves.corner_weighted_count() as i32;
            let mut value = SQUARE_VALUE[mv.sq.index()] * SQUARE_VALUE_WEIGHT;
            value += corner_stability * CORNER_STABILITY_WEIGHT;
            value += (36 - potential_mobility) * POTENTIAL_MOBILITY_WEIGHT;
            value += (36 - weighted_mobility) * MOBILITY_WEIGHT;
            value
        }
    }
}
