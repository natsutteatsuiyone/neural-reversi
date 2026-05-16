//! Move evaluation and ordering.

use crate::board::Board;
use crate::search::midgame;
use crate::search::node_type::NodeType;
use crate::search::search_context::SearchContext;
use crate::search::search_strategy::SearchStrategy;
use crate::square::Square;
use crate::types::{Depth, ScaledScore};

use super::{Move, MoveList};

/// Value assigned to moves suggested by the transposition table.
///
/// Must be the highest ordering value so that callers relying on `tt_move`
/// landing at index 0 after [`MoveList::sort`] remain correct.
const TT_MOVE_VALUE: i32 = 1 << 30;

/// Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/move.c#L30>
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

impl MoveList {
    /// Evaluates all moves and assigns ordering values.
    pub fn evaluate_moves<NT: NodeType, SS: SearchStrategy>(
        &mut self,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        tt_move: Square,
        alpha: ScaledScore,
        cut_node: bool,
    ) {
        #[rustfmt::skip]
        const ENDGAME_MIN_SORT_DEPTH: [u32; 64] = [
            19, 18, 18, 18, 17, 17, 17, 16,  // 0-7 empty squares
            16, 16, 15, 15, 15, 14, 14, 14,  // 8-15 empty squares
            13, 13, 13, 12, 12, 12, 11, 11,  // 16-23 empty squares
            11, 10, 10, 10, 9,  9,  9,  9,   // 24-31 empty squares
            9,  9,  9,  9,  9,  9,  9,  9,   // 32-39 empty squares
            9,  9,  9,  9,  9,  9,  9,  9,   // 40-47 empty squares
            9,  9,  9,  9,  9,  9,  9,  9,   // 48-55 empty squares
            9,  9,  9,  9,  9,  9,  9,  9    // 56-63 empty squares
        ];

        let use_fast = if SS::IS_ENDGAME {
            depth < ENDGAME_MIN_SORT_DEPTH[ctx.empty_list.count() as usize]
        } else {
            depth < midgame::LMR_MIN_DEPTH
        };

        if use_fast {
            self.evaluate_moves_fast(ctx, board, tt_move);
            return;
        }

        self.evaluate_moves_by_search::<NT, SS>(ctx, board, depth, tt_move, alpha, cut_node);
    }

    /// Evaluates moves via shallow search, with an endgame-only mobility penalty.
    fn evaluate_moves_by_search<NT: NodeType, SS: SearchStrategy>(
        &mut self,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        tt_move: Square,
        alpha: ScaledScore,
        cut_node: bool,
    ) {
        let sort_depth: i32 = if SS::IS_ENDGAME {
            match depth {
                0..=19 => 0,
                20..=28 => 1,
                _ => 2,
            }
        } else {
            match depth {
                0..=15 => 0,
                16..=25 => 1,
                _ => 2,
            }
        };

        // All-nodes search every move regardless of order, so shallow-search ordering
        // doesn't pay off - cheap mobility ordering suffices. In a non-cut non-PV
        // context, the caller has already tried the TT move without producing a
        // beta-cut, which is a strong All-node signal; otherwise fall back to a
        // static-eval margin check.
        let skip_search_ordering = if SS::IS_ENDGAME && !NT::PV_NODE && !cut_node {
            tt_move != Square::None || {
                const SKIP_MARGIN: ScaledScore = ScaledScore::from_disc_diff(8);
                let static_eval = midgame::evaluate(ctx, board);
                static_eval + SKIP_MARGIN < alpha
            }
        } else {
            false
        };

        const MOBILITY_SCALE_SEARCH: i32 = ScaledScore::SCALE * 15 / 8;
        const POTENTIAL_MOBILITY_SCALE_SEARCH: i32 = ScaledScore::SCALE;
        const MOBILITY_SCALE_SKIP: i32 = ScaledScore::SCALE * 8;
        const POTENTIAL_MOBILITY_SCALE_SKIP: i32 = ScaledScore::SCALE;

        let (mobility_scale, potential_mobility_scale) = if skip_search_ordering {
            (MOBILITY_SCALE_SKIP, POTENTIAL_MOBILITY_SCALE_SKIP)
        } else {
            (MOBILITY_SCALE_SEARCH, POTENTIAL_MOBILITY_SCALE_SEARCH)
        };

        // Wipeout moves are filtered out by callers via `wipeout_move()` before
        // reaching this loop, so `mv.flipped == board.opponent` cannot occur here.
        for mv in self.iter_mut() {
            if mv.sq == tt_move {
                mv.value = TT_MOVE_VALUE;
            } else {
                let next = board.make_move_with_flipped(mv.flipped, mv.sq);
                if skip_search_ordering {
                    ctx.increment_nodes();
                    mv.value = 0;
                } else {
                    ctx.update(mv.sq, mv.flipped);
                    mv.value = shallow_search_score(ctx, &next, sort_depth).value();
                    ctx.undo(mv.sq);
                }

                if SS::IS_ENDGAME {
                    let (moves, potential) = next.get_moves_and_potential();
                    let mobility = moves.corner_weighted_count() as i32;
                    let potential_mobility = potential.corner_weighted_count() as i32;
                    mv.value -= mobility * mobility_scale;
                    mv.value -= potential_mobility * potential_mobility_scale;
                }
            }
        }
    }

    /// Evaluates moves using fast heuristics for move ordering.
    pub fn evaluate_moves_fast(&mut self, ctx: &mut SearchContext, board: &Board, tt_move: Square) {
        // Wipeout moves are filtered out by callers via `wipeout_move()` before
        // reaching this loop, so `mv.flipped == board.opponent` cannot occur here.
        if tt_move == Square::None {
            for mv in self.iter_mut() {
                mv.value = evaluate_fast_value(ctx, board, *mv);
            }
            return;
        }

        for mv in self.iter_mut() {
            mv.value = if mv.sq == tt_move {
                TT_MOVE_VALUE
            } else {
                evaluate_fast_value(ctx, board, *mv)
            };
        }
    }

    /// Sorts all moves in descending order of their evaluation values.
    #[inline]
    pub fn sort(&mut self) {
        self.moves.sort_by_value_desc();
    }
}

/// Evaluates a position using a shallow search at the specified depth.
///
/// Used for move ordering in both midgame and endgame evaluation.
#[inline(always)]
fn shallow_search_score(ctx: &mut SearchContext, next: &Board, sort_depth: i32) -> ScaledScore {
    match sort_depth {
        0 => -midgame::evaluate(ctx, next),
        1 => -midgame::evaluate_depth1(ctx, next, -ScaledScore::INF, ScaledScore::INF),
        2 => -midgame::evaluate_depth2(ctx, next, -ScaledScore::INF, ScaledScore::INF),
        _ => unreachable!(),
    }
}

#[inline(always)]
fn evaluate_fast_value(ctx: &mut SearchContext, board: &Board, mv: Move) -> i32 {
    ctx.increment_nodes();
    let next = board.make_move_with_flipped(mv.flipped, mv.sq);
    let corner_stability = next.opponent.corner_stability() as i32;
    let weighted_mobility = next.get_moves().corner_weighted_count() as i32;
    // SAFETY: `mv.sq.index() < 64` since `Move::new` rejects `Square::None`.
    let square_value = unsafe { SQUARE_VALUE.get_unchecked(mv.sq.index()) };

    square_value * SQUARE_VALUE_WEIGHT
        + corner_stability * CORNER_STABILITY_WEIGHT
        + (36 - weighted_mobility) * MOBILITY_WEIGHT
}
