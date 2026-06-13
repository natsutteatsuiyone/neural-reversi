//! Shared principal variation search core.
//!
//! Node-level alpha-beta search used by both the midgame and endgame phases,
//! specialized at compile time by [`NodeType`] and [`SearchStrategy`].

use reversi_core::{
    board::Board,
    search::node_type::{NodeType, NonPV, PV},
    square::Square,
    stability,
    types::{Depth, ScaledScore},
};

use crate::move_list::MoveList;
use crate::transposition_table::Bound;

use super::{context::SearchContext, endgame, strategy::SearchStrategy};

/// Depth threshold for switching from midgame to endgame search.
pub const DEPTH_MIDGAME_TO_ENDGAME: Depth = 12;

/// Performs alpha-beta search parameterized by node type and search strategy.
pub fn search<NT: NodeType, SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: ScaledScore,
    beta: ScaledScore,
) -> ScaledScore {
    let org_alpha = alpha;
    let mut best_move = Square::None;
    let mut best_score = -ScaledScore::INF;
    let n_empties = ctx.empty_list.count();

    if NT::PV_NODE {
        if depth == 0 {
            return SS::evaluate(ctx, board);
        }
    } else {
        if n_empties == depth && depth <= DEPTH_MIDGAME_TO_ENDGAME {
            let score = endgame::null_window_search(ctx, board, alpha.to_disc_diff());
            return ScaledScore::from_disc_diff(score);
        }

        if depth <= SS::DEPTH_TO_SHALLOW {
            return SS::shallow_search(ctx, board, depth, alpha, beta);
        }

        if let Some(score) = stability_cutoff(board, n_empties, alpha) {
            return score;
        }
    }

    let mut move_list = MoveList::new(board);
    if NT::ROOT_NODE && ctx.pv_idx() > 0 {
        move_list.retain(board, |mv| ctx.root_move_in_pv_window(mv.sq));
    }
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -search::<NT, SS>(ctx, &next, depth, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return board.solve_scaled(n_empties);
        }
    } else if let Some(sq) = move_list.wipeout_move() {
        if NT::ROOT_NODE {
            ctx.update_root_move(sq, ScaledScore::MAX, 1, alpha);
        }
        return ScaledScore::MAX;
    }

    // Look up position in transposition table
    let tt_key = board.hash();
    let tt_probe_result = ctx.tt.probe(tt_key);
    let tt_move = tt_probe_result.best_move();

    if !NT::PV_NODE {
        if let Some(tt_data) = tt_probe_result.data()
            && (!SS::IS_ENDGAME || tt_data.is_endgame)
            && tt_data.depth >= depth
            && tt_data.selectivity >= ctx.selectivity
            && tt_data.can_cut(beta)
        {
            return tt_data.score;
        }

        if depth >= SS::MIN_ETC_DEPTH
            && let Some(score) = enhanced_transposition_cutoff::<SS>(
                ctx,
                board,
                &move_list,
                depth,
                alpha,
                tt_key,
                tt_probe_result.index(),
            )
        {
            return score;
        }

        if depth >= SS::MIN_PROBCUT_DEPTH
            && let Some(score) = SS::try_probcut(ctx, board, depth, beta)
        {
            return score;
        }
    }

    if move_list.count() > 1 {
        crate::move_list::evaluate_moves::<SS>(&mut move_list, ctx, board, depth, tt_move);
        move_list.sort();
    }

    let mut move_count = 0;
    for mv in move_list.iter() {
        move_count += 1;

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv.sq, mv.flipped);

        let mut score = -ScaledScore::INF;
        if !NT::PV_NODE || move_count > 1 {
            score = -search::<NonPV, SS>(ctx, &next, depth - 1, -(alpha + 1), -alpha);
        }

        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            score = -search::<PV, SS>(ctx, &next, depth - 1, -beta, -alpha);
        }

        ctx.undo(mv.sq);

        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, alpha);
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                best_move = mv.sq;


                if NT::PV_NODE && score < beta {
                    alpha = score;
                } else {
                    break;
                }
            }
        }
    }

    ctx.tt.store(
        tt_probe_result.index(),
        tt_key,
        best_score,
        Bound::classify::<NT>(best_score, org_alpha, beta),
        depth,
        best_move,
        ctx.selectivity,
        SS::IS_ENDGAME,
    );

    best_score
}

/// Returns a stability-based cutoff score, or [`None`] if no cutoff applies.
fn stability_cutoff(board: &Board, n_empties: Depth, alpha: ScaledScore) -> Option<ScaledScore> {
    if let Some(score) = stability::stability_cutoff(board, n_empties, alpha.to_disc_diff()) {
        return Some(ScaledScore::from_disc_diff(score));
    }
    None
}

/// Attempts an Enhanced Transposition Cutoff (ETC).
#[allow(clippy::too_many_arguments)]
fn enhanced_transposition_cutoff<SS: SearchStrategy>(
    ctx: &mut SearchContext,
    board: &Board,
    move_list: &MoveList,
    depth: u32,
    alpha: ScaledScore,
    tt_key: u64,
    tt_entry_index: usize,
) -> Option<ScaledScore> {
    let etc_depth = depth - 1;
    for mv in move_list.iter() {
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.increment_nodes();

        let etc_tt_key = next.hash();
        if let Some(etc_tt_data) = ctx.tt.lookup(etc_tt_key)
            && (!SS::IS_ENDGAME || etc_tt_data.is_endgame)
            && etc_tt_data.depth >= etc_depth
            && etc_tt_data.selectivity >= ctx.selectivity
        {
            let score = -etc_tt_data.score;
            if (etc_tt_data.bound == Bound::Exact || etc_tt_data.bound == Bound::Upper)
                && score > alpha
            {
                ctx.tt.store(
                    tt_entry_index,
                    tt_key,
                    score,
                    Bound::Lower,
                    depth,
                    mv.sq,
                    ctx.selectivity,
                    SS::IS_ENDGAME,
                );
                return Some(score);
            }
        }
    }
    None
}
