//! Move list extensions for reversi_web
//!
//! This module re-exports move list types from reversi_core and provides
//! web-specific implementation of evaluate_moves that uses the web search module.

// Re-export core move list types
pub use reversi_core::move_list::MoveList;

use reversi_core::{board::Board, square::Square};

use crate::search::{self, search_context::SearchContext};

/// Value assigned to wipeout moves (capturing all opponent pieces).
const WIPEOUT_VALUE: i32 = 1 << 30;

/// Value assigned to moves suggested by the transposition table.
const TT_MOVE_VALUE: i32 = 1 << 20;

/// Web-specific implementation of move evaluation.
///
/// This function evaluates all moves to assign ordering values and reduction depths,
/// using the web search module's evaluation functions.
///
/// # Type Parameters
///
/// * `NT` - Node type affecting evaluation depth and strategy
/// * `USE_REDUCTION` - Whether to apply depth reduction to poor moves
///
/// # Arguments
///
/// * `move_list` - The move list to evaluate
/// * `ctx` - Search context with transposition table and statistics
/// * `board` - Current position before making any move
/// * `tt_move` - Best move from transposition table (if any)
pub fn evaluate_moves(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    tt_move: Square,
) {
    if ctx.empty_list.count() <= 18 {
        evaluate_moves_fast(move_list, ctx, board, tt_move);
        return;
    }

    for i in 0..move_list.count() {
        let mv = &mut move_list.iter_mut().nth(i).unwrap();

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
            mv.value = (-search::evaluate(ctx, &next)).value();
            ctx.undo(mv.sq);
        };
    }
}

pub fn evaluate_moves_fast(
    move_list: &mut MoveList,
    ctx: &mut SearchContext,
    board: &Board,
    tt_move: Square,
) {
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
            let mut value = corner_stability * CORNER_STABILITY_WEIGHT;
            value += (36 - potential_mobility) * POTENTIAL_MOBILITY_WEIGHT;
            value += (36 - weighted_mobility) * MOBILITY_WEIGHT;
            value
        }
    }
}
