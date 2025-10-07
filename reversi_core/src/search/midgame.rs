//! # References:
//! - https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/midgame.c
//! - https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/search.cpp

use std::sync::Arc;

use rand::seq::IteratorRandom;

use super::SearchTask;
use super::search_result::SearchResult;
use super::threading::Thread;
use crate::bitboard::BitboardIterator;
use crate::board::Board;
use crate::constants::EVAL_SCORE_SCALE;
use crate::constants::EVAL_SCORE_SCALE_BITS;
use crate::constants::MID_SCORE_MAX;
use crate::constants::SCORE_INF;
use crate::flip;
use crate::move_list::ConcurrentMoveIterator;
use crate::move_list::Move;
use crate::move_list::MoveList;
use crate::probcut;
use crate::probcut::NO_SELECTIVITY;
use crate::search::endgame;
use crate::search::node_type::{NodeType, NonPV, PV, Root};
use crate::search::search_context::GamePhase;
use crate::search::search_context::SearchContext;
use crate::search::threading::SplitPoint;
use crate::square::Square;
use crate::stability;
use crate::transposition_table::Bound;
use crate::types::Depth;
use crate::types::Score;
use crate::types::Scoref;

/// Minimum depth required before considering parallel split.
const MIN_SPLIT_DEPTH: Depth = 4;

/// Performs the root search using iterative deepening with aspiration windows.
///
/// # Arguments
///
/// * `task` - Search task containing board position, search parameters, and callbacks
/// * `thread` - Thread handle for parallel search coordination
///
/// # Returns
///
/// SearchResult containing the best move, score, and search statistics
///
/// # Special Cases
///
/// - Returns a random move for the opening position (60 empty squares)
/// - Returns evaluation only for depth 0 searches
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let level = task.level;
    let multi_pv = task.multi_pv;

    let mut ctx = SearchContext::new(
        &board,
        task.generation,
        task.selectivity,
        task.tt.clone(),
        task.eval.clone(),
    );

    let n_empties = ctx.empty_list.count;
    if n_empties == 60 && !task.multi_pv {
        let mv = random_move(&board);
        return SearchResult {
            score: 0.0,
            best_move: Some(mv),
            n_nodes: 0,
            pv_line: vec![],
            depth: 0,
            selectivity: NO_SELECTIVITY,
        };
    }

    if let Some(ref callback) = task.callback {
        ctx.set_callback(callback.clone());
    }

    const INITIAL_DELTA: Score = 3 << EVAL_SCORE_SCALE_BITS;
    let mut best_score = 0;
    let mut alpha = -SCORE_INF;
    let mut beta = SCORE_INF;
    ctx.game_phase = GamePhase::MidGame;

    let max_depth = level.mid_depth;
    if max_depth == 0 {
        let score = search::<Root, false>(&mut ctx, &board, max_depth, alpha, beta, thread, None);
        return SearchResult {
            score: to_scoref(score),
            best_move: None,
            n_nodes: ctx.n_nodes,
            pv_line: Vec::new(),
            depth: 0,
            selectivity: NO_SELECTIVITY,
        };
    }

    let num_root_moves = ctx.root_moves_count();
    let pv_count = if multi_pv { num_root_moves } else { 1 };

    let org_selectivty = ctx.selectivity;
    let start_depth = if max_depth.is_multiple_of(2) { 2 } else { 1 };
    let mut depth = start_depth;
    while depth <= max_depth {
        ctx.selectivity = org_selectivty - ((max_depth - depth) as u8).min(org_selectivty);
        ctx.reset_root_move_searched();

        let mut delta = INITIAL_DELTA;
        if depth <= 10 {
            alpha = -SCORE_INF;
            beta = SCORE_INF;
        }

        for pv_idx in 0..pv_count {
            if pv_idx >= 1 {
                alpha = -SCORE_INF;
                beta = best_score;
            }

            loop {
                best_score =
                    search::<Root, false>(&mut ctx, &board, depth, alpha, beta, thread, None);

                if thread.is_search_aborted() {
                    break;
                }

                if best_score <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (best_score - delta).max(-SCORE_INF);
                } else if best_score >= beta {
                    alpha = (alpha + beta) / 2;
                    beta = (best_score + delta).min(SCORE_INF);
                } else {
                    break;
                }

                delta += delta / 2;
            }

            let best_move = ctx.get_best_root_move(true).unwrap();
            ctx.mark_root_move_searched(best_move.sq);
            ctx.notify_progress(depth, to_scoref(best_score), best_move.sq, ctx.selectivity);

            if thread.is_search_aborted() {
                break;
            }
        }

        let best_move = ctx.get_best_root_move(false).unwrap();
        alpha = (best_move.average_score - INITIAL_DELTA).max(-SCORE_INF);
        beta = (best_move.average_score + INITIAL_DELTA).min(SCORE_INF);

        if thread.is_search_aborted() {
            return SearchResult {
                score: to_scoref(best_move.score),
                best_move: Some(best_move.sq),
                n_nodes: ctx.n_nodes,
                pv_line: best_move.pv,
                depth,
                selectivity: ctx.selectivity,
            };
        }

        if depth <= 10 {
            depth += 2;
        } else {
            depth += 1;
        }
    }

    let rm = ctx.get_best_root_move(false).unwrap();
    SearchResult {
        score: to_scoref(best_score),
        best_move: Some(rm.sq),
        n_nodes: ctx.n_nodes,
        pv_line: rm.pv,
        depth: max_depth,
        selectivity: ctx.selectivity,
    }
}

/// Converts an internal score to floating-point disc units.
///
/// Internal scores are scaled by `EVAL_SCORE_SCALE` for precision.
/// This function converts them back to the standard disc count scale.
///
/// # Arguments
///
/// * `score` - Internal score representation
///
/// # Returns
///
/// Score in disc units as a floating-point number
fn to_scoref(score: Score) -> Scoref {
    score as Scoref / EVAL_SCORE_SCALE as Scoref
}

/// Selects a random legal move from the current position.
///
/// # Arguments
///
/// * `board` - Current board position
///
/// # Returns
///
/// A randomly selected legal move square
fn random_move(board: &Board) -> Square {
    let mut rng = rand::rng();
    BitboardIterator::new(board.get_moves())
        .choose(&mut rng)
        .unwrap()
}

/// Alpha-beta search function for midgame positions.
///
/// # Type Parameters
///
/// * `NT` - Node type (Root, PV, or NonPV) determining search behavior.
/// * `SP_NODE` - Whether this is a split point node in parallel search.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state and statistics.
/// * `board` - Current board position to search.
/// * `depth` - Remaining search depth.
/// * `alpha` - Lower bound of the search window
/// * `beta` - Upper bound of the search window
/// * `thread` - Thread handle for parallel search coordination.
/// * `split_point` - Optional split point for parallel search nodes.
///
/// # Returns
///
/// The best score found for the position.
pub fn search<NT: NodeType, const SP_NODE: bool>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: Score,
    beta: Score,
    thread: &Arc<Thread>,
    split_point: Option<&Arc<SplitPoint>>,
) -> Score {
    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;
    let move_iter: Arc<ConcurrentMoveIterator>;
    let tt_key;
    let tt_entry_index;

    if SP_NODE {
        let sp_state = split_point.as_ref().unwrap().state();
        best_move = sp_state.best_move;
        best_score = sp_state.best_score;
        move_iter = sp_state.move_iter.clone().unwrap();
        tt_key = 0;
        tt_entry_index = 0;
    } else {
        let n_empties = ctx.empty_list.count;

        if NT::PV_NODE {
            if depth == 0 {
                return evaluate(ctx, board);
            }
        } else {
            if depth == 2 {
                return evaluate_depth2(ctx, board, alpha, beta);
            } else if depth == 1 {
                return evaluate_depth1(ctx, board, alpha, beta);
            } else if depth == 0 {
                return evaluate(ctx, board);
            }

            if let Some(score) = stability_cutoff(board, n_empties, alpha) {
                return score;
            }
        }

        tt_key = board.hash();
        ctx.tt.prefetch(tt_key);

        let mut move_list = MoveList::new(board);
        if move_list.count() == 0 {
            let next = board.switch_players();
            if next.has_legal_moves() {
                ctx.update_pass();
                let score = -search::<NT, false>(ctx, &next, depth, -beta, -alpha, thread, None);
                ctx.undo_pass();
                return score;
            } else {
                return solve(board, n_empties);
            }
        }

        // Look up position in transposition table
        let (tt_hit, tt_data, _tt_entry_index) = ctx.tt.probe(tt_key, ctx.generation);
        tt_entry_index = _tt_entry_index;
        let tt_move = if tt_hit {
            tt_data.best_move
        } else {
            Square::None
        };

        if !NT::PV_NODE
            && tt_hit
            && tt_data.depth >= depth
            && tt_data.selectivity >= ctx.selectivity
            && tt_data.should_cutoff(beta)
        {
            return tt_data.score;
        }

        if !NT::PV_NODE
            && let Some(score) = probcut::probcut_midgame(ctx, board, depth, alpha, beta, thread)
        {
            return score;
        }

        if move_list.count() > 1 {
            move_list.evaluate_moves::<NT>(ctx, board, depth, tt_move);
            move_list.sort();
        }

        move_iter = Arc::new(ConcurrentMoveIterator::new(move_list));
    }

    while let Some((mv, move_count)) = move_iter.next() {
        if NT::ROOT_NODE && ctx.is_move_searched(mv.sq) {
            continue;
        }

        if SP_NODE {
            split_point.as_ref().unwrap().unlock();
        }

        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv);

        let mut score = -SCORE_INF;
        if depth >= 2 && mv.reduction_depth > 0 {
            if SP_NODE {
                let sp_state = split_point.as_ref().unwrap().state();
                alpha = sp_state.alpha;
            }

            let d = depth - 1 - mv.reduction_depth.min(depth - 1);
            score = -search::<NonPV, false>(ctx, &next, d, -(alpha + 1), -alpha, thread, None);
            if score > alpha {
                score = -search::<NonPV, false>(
                    ctx,
                    &next,
                    depth - 1,
                    -(alpha + 1),
                    -alpha,
                    thread,
                    None,
                );
            }
        } else if !NT::PV_NODE || move_count > 1 {
            if SP_NODE {
                let sp_state = split_point.as_ref().unwrap().state();
                alpha = sp_state.alpha;
            }
            score =
                -search::<NonPV, false>(ctx, &next, depth - 1, -(alpha + 1), -alpha, thread, None);
        }

        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            ctx.clear_pv();
            score = -search::<PV, false>(ctx, &next, depth - 1, -beta, -alpha, thread, None);
        }

        ctx.undo(mv);

        if SP_NODE {
            let sp = split_point.as_ref().unwrap();
            sp.lock();
            let sp_state = sp.state();
            best_score = sp_state.best_score;
            alpha = sp_state.alpha;
        }

        if thread.is_search_aborted() || thread.cutoff_occurred() {
            return 0;
        }

        if NT::ROOT_NODE {
            ctx.update_root_move(mv.sq, score, move_count, alpha);
        }

        if score > best_score {
            if SP_NODE {
                let sp_state = split_point.as_ref().unwrap().state_mut();
                sp_state.best_score = score;
            }
            best_score = score;

            if score > alpha {
                if SP_NODE {
                    let sp_state = split_point.as_ref().unwrap().state_mut();
                    sp_state.best_move = mv.sq;
                }
                best_move = mv.sq;

                if NT::PV_NODE && !NT::ROOT_NODE {
                    ctx.update_pv(mv.sq);
                }

                if NT::PV_NODE && score < beta {
                    if SP_NODE {
                        let sp_state = split_point.as_ref().unwrap().state_mut();
                        sp_state.alpha = score;
                    }
                    alpha = score;
                } else {
                    if SP_NODE {
                        let sp = split_point.as_ref().unwrap();
                        sp.state_mut().cutoff = true;
                    }

                    break;
                }
            }
        }

        if !SP_NODE && depth >= MIN_SPLIT_DEPTH && move_iter.count() > 1 && thread.can_split() {
            let (s, m, n) = thread.split(
                ctx,
                board,
                alpha,
                beta,
                best_score,
                best_move,
                depth,
                &move_iter,
                NT::TYPE_ID,
            );
            best_score = s;
            best_move = m;
            ctx.n_nodes += n;

            if thread.is_search_aborted() || thread.cutoff_occurred() {
                return 0;
            }

            if best_score >= beta {
                break;
            }
        }
    }

    if SP_NODE {
        return best_score;
    }

    ctx.tt.store(
        tt_entry_index,
        tt_key,
        best_score,
        Bound::determine_bound::<NT>(best_score, beta),
        depth,
        best_move,
        ctx.selectivity,
        ctx.generation,
    );

    best_score
}

/// Specialized evaluation function for positions at depth 2.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
pub fn evaluate_depth2(
    ctx: &mut SearchContext,
    board: &Board,
    mut alpha: Score,
    beta: Score,
) -> Score {
    let moves = board.get_moves();
    if moves == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth2(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, ctx.empty_list.count);
        }
    }

    let mut best_score = -SCORE_INF;
    for sq in BitboardIterator::new(moves) {
        let flipped = flip::flip(sq, board.player, board.opponent);
        let mv = Move::new(sq, flipped);
        let next = board.make_move_with_flipped(flipped, sq);

        ctx.update(&mv);
        let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
        ctx.undo(&mv);

        if score > best_score {
            best_score = score;
            if score >= beta {
                break;
            }
            if score > alpha {
                alpha = score;
            }
        }
    }

    best_score
}

/// Specialized evaluation function for positions at depth 1.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
pub fn evaluate_depth1(ctx: &mut SearchContext, board: &Board, alpha: Score, beta: Score) -> Score {
    let moves = board.get_moves();
    if moves == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -evaluate_depth1(ctx, &next, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, ctx.empty_list.count);
        }
    }

    let mut best_score = -SCORE_INF;
    for sq in BitboardIterator::new(moves) {
        let flipped = flip::flip(sq, board.player, board.opponent);
        if flipped == board.opponent {
            return MID_SCORE_MAX;
        }
        let next = board.make_move_with_flipped(flipped, sq);

        let mv = Move::new(sq, flipped);
        ctx.update(&mv);
        let score = -evaluate(ctx, &next);
        ctx.undo(&mv);

        if score > best_score {
            best_score = score;
            if score >= beta {
                break;
            }
        }
    }

    best_score
}

/// Evaluates a leaf node position using the neural network evaluator.
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position
///
/// # Returns
///
/// Position score in internal units
#[inline(always)]
pub fn evaluate(ctx: &SearchContext, board: &Board) -> Score {
    if ctx.ply() == 60 {
        return endgame::calculate_final_score(board) << EVAL_SCORE_SCALE_BITS;
    }

    ctx.eval.evaluate(ctx, board)
}

/// Performs a simplified search for shallow depths.
///
/// # Type Parameters
///
/// * `NT` - Node type (PV or NonPV) for search behavior
///
/// # Arguments
///
/// * `ctx` - Search context tracking game state
/// * `board` - Current board position to search
/// * `depth` - Remaining search depth
/// * `alpha` - Lower bound of search window
/// * `beta` - Upper bound of search window
///
/// # Returns
///
/// Best score found for the position
pub fn shallow_search<NT: NodeType>(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    mut alpha: Score,
    beta: Score,
) -> Score {
    if depth == 2 {
        return evaluate_depth2(ctx, board, alpha, beta);
    }

    let tt_key = board.hash();
    ctx.tt.prefetch(tt_key);

    let mut move_list = MoveList::new(board);
    if move_list.count() == 0 {
        let next = board.switch_players();
        if next.has_legal_moves() {
            ctx.update_pass();
            let score = -shallow_search::<NT>(ctx, &next, depth, -beta, -alpha);
            ctx.undo_pass();
            return score;
        } else {
            return solve(board, ctx.empty_list.count);
        }
    }

    let (tt_hit, tt_data, _tt_entry_index) = ctx.tt.probe(tt_key, ctx.generation);
    let tt_entry_index = _tt_entry_index;
    let tt_move = if tt_hit {
        tt_data.best_move
    } else {
        Square::None
    };

    if !NT::PV_NODE
        && tt_hit
        && tt_data.depth >= depth
        && tt_data.selectivity >= ctx.selectivity
        && tt_data.should_cutoff(beta)
    {
        return tt_data.score;
    }

    if move_list.count() > 1 {
        move_list.evaluate_moves::<NT>(ctx, board, depth, tt_move);
        move_list.sort();
    }

    let mut best_move = Square::None;
    let mut best_score = -SCORE_INF;
    let mut move_count = 0;
    for mv in move_list.iter() {
        move_count += 1;
        let next = board.make_move_with_flipped(mv.flipped, mv.sq);
        ctx.update(mv);

        let mut score = -SCORE_INF;
        if !NT::PV_NODE || move_count > 1 {
            score = -shallow_search::<NonPV>(ctx, &next, depth - 1, -(alpha + 1), -alpha);
        }

        if NT::PV_NODE && (move_count == 1 || score > alpha) {
            score = -shallow_search::<PV>(ctx, &next, depth - 1, -beta, -alpha);
        }

        ctx.undo(mv);

        if score > best_score {
            best_move = mv.sq;
            best_score = score;
            if score >= beta {
                break;
            }
            if score > alpha {
                alpha = score;
            }
        }
    }

    ctx.tt.store(
        tt_entry_index,
        tt_key,
        best_score,
        Bound::determine_bound::<NT>(best_score, beta),
        depth,
        best_move,
        ctx.selectivity,
        ctx.generation,
    );

    best_score
}

/// Calls the endgame solver for terminal nodes where both players must pass.
///
/// # Arguments
///
/// * `board` - The terminal board position to be evaluated.
/// * `n_empties` - The number of empty squares remaining on the board.
///
/// # Returns
///
/// The exact final score of the position, scaled to internal units.
fn solve(board: &Board, n_empties: Depth) -> Score {
    endgame::solve(board, n_empties) << EVAL_SCORE_SCALE_BITS
}

/// Checks for stability-based cutoffs in the search.
///
/// # Arguments
///
/// * `board` - Current board position
/// * `n_empties` - Number of empty squares
/// * `alpha` - Current alpha bound for pruning decision
///
/// # Returns
///
/// * `Some(score)` - If position can be pruned with this score
/// * `None` - If no stability cutoff is possible
fn stability_cutoff(board: &Board, n_empties: Depth, alpha: Score) -> Option<Score> {
    if let Some(score) =
        stability::stability_cutoff(board, n_empties, alpha >> EVAL_SCORE_SCALE_BITS)
    {
        return Some(score << EVAL_SCORE_SCALE_BITS);
    }
    None
}
