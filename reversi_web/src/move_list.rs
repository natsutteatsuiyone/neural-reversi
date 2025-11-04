//! Move list extensions for reversi_web
//!
//! This module re-exports move list types from reversi_core and provides
//! web-specific implementation of evaluate_moves that uses the web search module.

// Re-export core move list types
pub use reversi_core::move_list::{Move, MoveList};

use reversi_core::{
    board::Board,
    constants::{EVAL_SCORE_SCALE_BITS, SCORE_INF},
    search::node_type::NodeType,
    square::Square,
    types::Depth,
};

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
/// * `depth` - Remaining search depth at this node
/// * `tt_move` - Best move from transposition table (if any)
pub fn evaluate_moves<NT: NodeType, const USE_REDUCTION: bool>(
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

    if depth <= MIN_DEPTH[ctx.empty_list.count as usize] {
        move_list.evaluate_moves_fast(board, tt_move);
        return;
    }

    const MAX_SORT_DEPTH: i32 = 2;
    let mut sort_depth = (depth as i32 - 15) / 3;
    sort_depth = sort_depth.clamp(1, MAX_SORT_DEPTH);

    let mut max_evaluated_value = -SCORE_INF;

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
            ctx.update(mv);

            mv.value = match sort_depth {
                0 => -search::evaluate(ctx, &next),
                1 => -search::evaluate_depth1(ctx, &next, -SCORE_INF, SCORE_INF),
                2 => -search::evaluate_depth2(ctx, &next, -SCORE_INF, SCORE_INF),
                _ => unreachable!(),
            };

            ctx.undo(mv);
            max_evaluated_value = max_evaluated_value.max(mv.value);
        };
    }

    if USE_REDUCTION {
        // Score-Based Reduction: reduce depth for poor moves
        // This implements a form of late move reduction based on evaluation scores
        let sbr_margin: i32 = (12 + (MAX_SORT_DEPTH - sort_depth) * 2) << EVAL_SCORE_SCALE_BITS;
        let reduction_threshold = max_evaluated_value - sbr_margin;

        for i in 0..move_list.count() {
            let mv = &mut move_list.iter_mut().nth(i).unwrap();
            if mv.value < reduction_threshold {
                // Calculate reduction based on how much worse this move is
                let diff = max_evaluated_value - mv.value;
                let step = sbr_margin * 2;
                mv.reduction_depth = ((diff + sbr_margin) / step) as Depth;
            }
        }
    }
}
