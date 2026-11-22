//! Endgame search and solving algorithms.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/endgame.c>

mod cache;
mod small_move_list;
mod solve;

use std::sync::Arc;

use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::eval::EvalMode;
use crate::flip;
use crate::legal_moves::LegalMoves;
use crate::level::Level;
use crate::probcut;
use crate::probcut::Selectivity;
use crate::score::{DISC_SCORE_INF, DISC_SCORE_MAX};
use crate::score::{DiscScore, Score};
use crate::search::context::SearchContext;
use crate::search::node_type::Root;
use crate::search::result::{CompletedState, SearchResult};
use crate::search::root_move::RootMoves;
use crate::search::strategy::EndGameStrategy;
use crate::search::threading::Thread;
use crate::search::time_control::{TimeManager, record_abort};
use crate::search::{SearchProgress, SearchTask, midgame, search, widen_aspiration_window};
use crate::square::Square;
use crate::stability::stability_cutoff;
use crate::transposition_table::Bound;
use crate::types::Depth;

use self::cache::{CacheSlot, Deep, EndGameCache, Shallow};
use self::small_move_list::SmallMoveList;
pub(super) use self::solve::{solve1, solve2, solve3, solve4, sort_last4};

/// Depth threshold for switching to null window search in endgame.
pub const DEPTH_TO_NWS: Depth = 12;

/// Minimum depth at which an endgame node splits its moves among threads.
pub(super) const MIN_SPLIT_DEPTH: Depth = DEPTH_TO_NWS;

/// Maximum depth at which an All-node may split before its first child is
/// searched.
pub(super) const SPECULATIVE_SPLIT_MAX_DEPTH: Depth = 24;

/// Depth threshold for switching to specialized shallow search.
pub(super) const DEPTH_TO_SHALLOW_SEARCH: Depth = 6;

/// Minimum empties for enhanced transposition cutoff via child EC entries.
const MIN_EC_ETC_EMPTIES: u32 = 12;

/// Total EC cache budget distributed across endgame search threads, in bytes.
const TOTAL_EC_CACHE_BYTES: usize = 2 * 1024 * 1024;

/// Total shallow cache budget distributed across endgame search threads, in bytes.
const TOTAL_SHALLOW_CACHE_BYTES: usize = 256 * 1024;

/// Minimum per-thread EC cache budget, in bytes.
const MIN_EC_CACHE_BYTES: usize = 128 * 1024;

/// Minimum per-thread shallow cache budget, in bytes.
const MIN_SHALLOW_CACHE_BYTES: usize = 32 * 1024;

/// Maximum per-thread shallow cache budget, in bytes.
const MAX_SHALLOW_CACHE_BYTES: usize = 128 * 1024;

/// Initial aspiration window half-width.
const INITIAL_ASPIRATION_WINDOW: Score = Score::from_disc_diff(DiscScore::new(1));

/// Inter-selectivity window narrowing delta.
const INTER_SELECTIVITY_DELTA: Score = Score::from_disc_diff(DiscScore::new(1));

/// Initial aspiration window widening delta.
const ASPIRATION_DELTA: Score = Score::from_disc_diff(DiscScore::new(1));

/// Dummy score returned while unwinding a branch after a split-point beta cutoff.
const SPLIT_CUTOFF_SCORE: DiscScore = DiscScore::new(0);

pub(in crate::search) struct EndGameCaches {
    ec: EndGameCache<Deep>,
    pub(super) shallow: EndGameCache<Shallow>,
}

impl EndGameCaches {
    pub(in crate::search) fn for_thread_count(n_threads: usize) -> Self {
        let n_threads = n_threads.max(1);
        Self {
            ec: EndGameCache::new((TOTAL_EC_CACHE_BYTES / n_threads).max(MIN_EC_CACHE_BYTES)),
            shallow: EndGameCache::new(
                (TOTAL_SHALLOW_CACHE_BYTES / n_threads)
                    .clamp(MIN_SHALLOW_CACHE_BYTES, MAX_SHALLOW_CACHE_BYTES),
            ),
        }
    }
}

/// Performs root search for endgame positions using iterative selectivity.
pub fn search_root(task: SearchTask, thread: &Arc<Thread>) -> SearchResult {
    let board = task.board;
    let root_moves = RootMoves::new(&board);

    let mut ctx = SearchContext::new(
        &board,
        task.mid_selectivity,
        &root_moves,
        &task.tt,
        &task.eval,
    );
    if ctx.root_moves.count() == 0 {
        // Handle no legal moves
        return SearchResult::NoLegalMove;
    }

    // Estimate initial aspiration window center
    let base_score = estimate_aspiration_base_score(&mut ctx, &board);

    let completed = CompletedState::new(task.pv_count(ctx.root_moves.count()));
    solve_root(
        &mut ctx,
        &board,
        &task,
        thread,
        base_score,
        completed,
        Selectivity::Level1,
    )
}

/// Runs the multi-PV, iterative-selectivity solve loop on an existing context.
pub(super) fn solve_root(
    ctx: &mut SearchContext,
    board: &Board,
    task: &SearchTask,
    thread: &Arc<Thread>,
    base_score: Score,
    mut completed: CompletedState,
    min_reported: Selectivity,
) -> SearchResult {
    let time_manager = task.time_manager.as_deref();
    let use_time_control = time_manager.is_some();
    #[cfg(feature = "search-stats")]
    let mut endgame_start = (completed.last.is_some() && task.callback.is_some())
        .then(|| (std::time::Instant::now(), ctx.counters()));

    if let Some(tm) = time_manager {
        // Enable endgame time management; the midgame handoff arrives with
        // the midgame allocation still active.
        tm.set_endgame_mode(true);
    }
    let endgame_clock = time_manager.map(TimeManager::endgame_clock);

    let n_empties = ctx.empty_list.count();
    completed.begin_phase();

    // Configure for endgame search
    ctx.selectivity = Selectivity::None;
    ctx.eval_mode = EvalMode::Small;

    let pv_count = task.pv_count(ctx.root_moves.count());

    // Multi-PV loop: search each PV line with its own aspiration window
    for pv_idx in 0..pv_count {
        ctx.root_moves.set_pv_idx(pv_idx);

        let previous_pv_score = pv_idx
            .checked_sub(1)
            .and_then(|idx| ctx.root_moves.get(idx))
            .map(|rm| rm.score);
        let (mut alpha, mut beta) =
            initial_aspiration_window(pv_idx, base_score, previous_pv_score);

        // Iterative selectivity loop
        for (step, selectivity) in Level::ENDGAME_SELECTIVITY.into_iter().enumerate() {
            // Check depth limit when not using time control
            if !use_time_control && task.level.end_depth_for(selectivity) < n_empties {
                break;
            }

            ctx.selectivity = selectivity;
            let score = aspiration_search(ctx, board, &mut alpha, &mut beta, thread);

            // Update aspiration window for next selectivity
            let delta = INTER_SELECTIVITY_DELTA;
            alpha = (score - delta).max(-Score::INF);
            beta = (score + delta).min(Score::INF);

            if thread.is_search_aborted() {
                record_abort(time_manager);
                break;
            }

            // Stable sort moves from pv_idx to end, bringing best to pv_idx position
            ctx.root_moves.sort_from_pv_idx();

            if selectivity >= min_reported {
                completed.line_completed(ctx, n_empties, true);
            }

            // Notify progress with the move now at pv_idx (the best for this PV line)
            if let Some(ref callback) = task.callback
                && let Some(rm) = ctx.root_moves.current_pv()
            {
                let progress = SearchProgress::from_iteration(
                    ctx,
                    &rm,
                    n_empties,
                    n_empties,
                    score,
                    ctx.n_nodes,
                    true,
                );
                #[cfg(feature = "search-stats")]
                let progress = SearchProgress {
                    endgame_start: endgame_start.take(),
                    ..progress
                };
                callback(progress);
            }

            // Check time control
            if endgame_clock
                .as_ref()
                .is_some_and(|clock| clock.should_stop(step))
            {
                break;
            }
        }

        // Check abort or time limit
        if thread.is_search_aborted() || time_manager.is_some_and(|tm| tm.check_time()) {
            return completed.into_result(ctx, board);
        }
    }

    // The fixed budget and increasing elapsed time make each later PV stop at
    // the same or a lower selectivity, so `ctx.selectivity` is that level.
    if ctx.selectivity >= min_reported {
        completed.iteration_completed(ctx, n_empties, true);
    }
    completed.into_result(ctx, board)
}

/// Estimates a base score to center the aspiration window for endgame search.
fn estimate_aspiration_base_score(ctx: &mut SearchContext, board: &Board) -> Score {
    if let Some(tt_data) = ctx.tt.probe(board, board.hash()).data()
        && tt_data.bound() == Bound::Exact
    {
        return tt_data.score();
    }

    ctx.eval_mode = EvalMode::Main;
    if ctx.empty_list.count() >= 20 {
        return midgame::evaluate_depth2(ctx, board, -Score::INF, Score::INF);
    }

    midgame::evaluate(ctx, board)
}

fn initial_aspiration_window(
    pv_idx: usize,
    base_score: Score,
    previous_pv_score: Option<Score>,
) -> (Score, Score) {
    if pv_idx == 0 {
        return (
            base_score - INITIAL_ASPIRATION_WINDOW,
            base_score + INITIAL_ASPIRATION_WINDOW,
        );
    }

    (
        -Score::INF,
        previous_pv_score
            .map(|score| Score::from_raw((score.value() + 1).min(Score::INF.value())))
            .unwrap_or(Score::INF),
    )
}

/// Performs aspiration window search for endgame at the current selectivity level.
fn aspiration_search(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: &mut Score,
    beta: &mut Score,
    thread: &Arc<Thread>,
) -> Score {
    let mut delta = ASPIRATION_DELTA;
    let n_empties = ctx.empty_list.count();

    loop {
        let score =
            search::<Root, EndGameStrategy>(ctx, board, n_empties, *alpha, *beta, thread, false);

        if thread.is_search_aborted() {
            return score;
        }

        if !widen_aspiration_window(score, alpha, beta, delta) {
            return score;
        }

        delta += delta; // Exponential widening
    }
}

/// Attempts ProbCut pruning for endgame positions.
pub fn try_probcut(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    beta: Score,
    cut_node: bool,
) -> Option<Score> {
    if !ctx.selectivity.is_enabled() {
        return None;
    }
    // The screen and the verification use the main network; the rest of the
    // endgame search stays on the small one.
    ctx.eval_mode = EvalMode::Main;
    let result = probcut_decision(ctx, board, depth, beta, cut_node);
    ctx.eval_mode = EvalMode::Small;
    #[cfg(feature = "probcut-trace")]
    probcut::trace::record(board, beta, ctx.selectivity, cut_node, result);
    result
}

fn probcut_decision(
    ctx: &mut SearchContext,
    board: &Board,
    depth: Depth,
    beta: Score,
    cut_node: bool,
) -> Option<Score> {
    let eval = midgame::evaluate(ctx, board);
    let window = probcut::endgame_probcut_window(depth, ctx.selectivity, cut_node, beta, eval);
    let pc_beta = window.pc_beta;
    if pc_beta >= Score::MAX || eval < window.screen_beta {
        return None;
    }

    const _: () = assert!(probcut::VERIFY_DEPTH == 2);
    let score = midgame::evaluate_depth2(ctx, board, pc_beta - 1, pc_beta);

    (score >= pc_beta).then_some(beta)
}

pub(super) trait SplitPointCutoff {
    fn occurred(&self) -> bool;
}

pub(super) struct IgnoreSplitPointCutoff;

impl SplitPointCutoff for IgnoreSplitPointCutoff {
    #[inline(always)]
    fn occurred(&self) -> bool {
        false
    }
}

impl SplitPointCutoff for Thread {
    #[inline(always)]
    fn occurred(&self) -> bool {
        self.split_point_cutoff_occurred()
    }
}

/// Null window search using the endgame cache
/// (`DEPTH_TO_SHALLOW_SEARCH + 1..=DEPTH_TO_NWS` empties).
///
/// `moves` must be the legal moves of `board`.
pub(super) fn null_window_search<C: SplitPointCutoff>(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: DiscScore,
    caches: &mut EndGameCaches,
    cutoff: &C,
    moves: LegalMoves,
) -> DiscScore {
    debug_assert_eq!(moves, board.legal_moves());
    let n_empties = ctx.empty_list.count();
    let beta = alpha + 1;

    if moves.is_empty() {
        let next = board.switch_players();
        let next_moves = next.legal_moves();
        if next_moves.is_empty() {
            return board.solve_disc_diff(n_empties);
        }
        ctx.increment_nodes();
        return -null_window_search(ctx, &next, -beta, caches, cutoff, next_moves);
    }

    let cache_idx = caches.ec.slot(board.hash());
    if let Some(score) = caches.ec.probe(cache_idx, board, alpha) {
        return score;
    }

    if let Some(score) = stability_cutoff(board, n_empties, alpha) {
        ctx.stats.increment_stability_cut();
        return score;
    }

    if let Some(sq) = moves.only_move() {
        let flipped = flip::flip(sq, board.player(), board.opponent());
        if flipped == board.opponent() {
            return DISC_SCORE_MAX;
        }
        let next = board.make_move_with_flipped(flipped, sq);
        let score = search_move_nws_ec(ctx, &next, next.legal_moves(), sq, beta, caches, cutoff);
        if cutoff.occurred() {
            return SPLIT_CUTOFF_SCORE;
        }
        caches.ec.store(cache_idx, board, alpha, score);
        return score;
    }

    let mut move_list = SmallMoveList::new();
    if !move_list.generate(board, moves) {
        return DISC_SCORE_MAX;
    }

    // Enhanced transposition cutoff: a child upper bound at or below `-beta`
    // proves this node fails high.
    if n_empties >= MIN_EC_ETC_EMPTIES {
        let ec = &mut caches.ec;
        for mv in move_list.iter() {
            if let Some(score) = ec.probe(ec.slot(mv.next.hash()), &mv.next, -beta)
                && score <= -beta
            {
                return -score;
            }
        }
    }

    let mut best_score = -DISC_SCORE_INF;
    move_list.evaluate_moves(ctx);
    for mv in move_list.best_first_iter() {
        let score = search_move_nws_ec(ctx, &mv.next, mv.next_moves, mv.sq, beta, caches, cutoff);

        if score >= beta {
            best_score = score;
            break;
        }
        best_score = best_score.max(score);
    }

    if cutoff.occurred() {
        return SPLIT_CUTOFF_SCORE;
    }

    caches.ec.store(cache_idx, board, alpha, best_score);

    best_score
}

/// Searches a move with null window, dispatching to shallow or EC search based on depth.
///
/// `next_moves` must be the legal moves of `next`.
#[inline(always)]
fn search_move_nws_ec<C: SplitPointCutoff>(
    ctx: &mut SearchContext,
    next: &Board,
    next_moves: LegalMoves,
    sq: Square,
    beta: DiscScore,
    caches: &mut EndGameCaches,
    cutoff: &C,
) -> DiscScore {
    if cutoff.occurred() {
        return SPLIT_CUTOFF_SCORE;
    }

    ctx.update_endgame(sq);
    let score = if ctx.empty_list.count() <= DEPTH_TO_SHALLOW_SEARCH {
        -shallow_search(ctx, next, -beta, &mut caches.shallow, next_moves)
    } else {
        -null_window_search(ctx, next, -beta, caches, cutoff, next_moves)
    };
    ctx.undo_endgame(sq);
    score
}

/// Null window search for positions with at most [`DEPTH_TO_SHALLOW_SEARCH`] empties.
///
/// Orders moves by isolated empties, quadrant parity, and corner priority.
///
/// `moves` must be the legal moves of `board`.
pub(super) fn shallow_search(
    ctx: &mut SearchContext,
    board: &Board,
    alpha: DiscScore,
    sc: &mut EndGameCache<Shallow>,
    moves: LegalMoves,
) -> DiscScore {
    debug_assert_eq!(moves, board.legal_moves());
    let n_empties = ctx.empty_list.count();
    let beta = alpha + 1;

    if moves.is_empty() {
        let next = board.switch_players();
        let next_moves = next.legal_moves();
        if next_moves.is_empty() {
            return board.solve_disc_diff(n_empties);
        }
        ctx.increment_nodes();
        return -shallow_search(ctx, &next, -beta, sc, next_moves);
    }

    let cache_idx = sc.slot(board.hash());
    if let Some(score) = sc.probe(cache_idx, board, alpha) {
        return score;
    }

    if let Some(score) = stability_cutoff(board, n_empties, alpha) {
        ctx.stats.increment_stability_cut();
        return score;
    }

    if let Some(sq) = moves.only_move() {
        let score = shallow_search_move(ctx, board, sq, beta, sc);
        sc.store(cache_idx, board, alpha, score);
        return score;
    }

    let mut best_score = -DISC_SCORE_INF;

    // Split moves by quadrant parity: odd-empty quadrants first
    #[rustfmt::skip]
    const QUADRANT_MASK: [u64; 16] = [
        0x0000000000000000, 0x000000000F0F0F0F, 0x00000000F0F0F0F0, 0x00000000FFFFFFFF,
        0x0F0F0F0F00000000, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0FF0F0F0F0, 0x0F0F0F0FFFFFFFFF,
        0xF0F0F0F000000000, 0xF0F0F0F00F0F0F0F, 0xF0F0F0F0F0F0F0F0, 0xF0F0F0F0FFFFFFFF,
        0xFFFFFFFF00000000, 0xFFFFFFFF0F0F0F0F, 0xFFFFFFFFF0F0F0F0, 0xFFFFFFFFFFFFFFFF
    ];

    // An empty square with no empty neighbor is a one-square region: search it
    // before the quadrant-parity groups.
    let empties = board.empty_squares().bits();
    let horizontal =
        ((empties << 1) & 0xFEFE_FEFE_FEFE_FEFE) | ((empties >> 1) & 0x7F7F_7F7F_7F7F_7F7F);
    let row = empties | horizontal;
    let isolated_moves = moves & !Bitboard::new(horizontal | (row << 8) | (row >> 8));
    let moves = moves.and_not(isolated_moves.bitboard());

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        isolated_moves,
        cache_idx,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    let quadrant_mask = Bitboard::new(QUADRANT_MASK[ctx.empty_list.parity() as usize]);
    let odd_moves = moves & quadrant_mask;
    let even_moves = moves & !quadrant_mask;

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        odd_moves.corners(),
        cache_idx,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        odd_moves.non_corners(),
        cache_idx,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        even_moves.corners(),
        cache_idx,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    if let Some(score) = shallow_search_moves(
        ctx,
        board,
        even_moves.non_corners(),
        cache_idx,
        beta,
        &mut best_score,
        sc,
    ) {
        return score;
    }

    sc.store(cache_idx, board, alpha, best_score);

    best_score
}

/// Searches all moves in a bitboard subset for shallow search, returning on beta cutoff.
#[inline(always)]
fn shallow_search_moves(
    ctx: &mut SearchContext,
    board: &Board,
    moves: LegalMoves,
    cache_idx: CacheSlot<Shallow>,
    beta: DiscScore,
    best_score: &mut DiscScore,
    sc: &mut EndGameCache<Shallow>,
) -> Option<DiscScore> {
    for sq in moves.iter() {
        let score = shallow_search_move(ctx, board, sq, beta, sc);

        if score >= beta {
            sc.store(cache_idx, board, beta - 1, score);
            return Some(score);
        }
        *best_score = (*best_score).max(score);
    }

    None
}

/// Evaluates a single move in shallow search.
#[inline(always)]
fn shallow_search_move(
    ctx: &mut SearchContext,
    board: &Board,
    sq: Square,
    beta: DiscScore,
    sc: &mut EndGameCache<Shallow>,
) -> DiscScore {
    let next = board.make_move(sq);
    ctx.update_endgame(sq);
    let next_alpha = -beta;
    let score = if ctx.empty_list.count() == 4 {
        let next_cache_idx = sc.slot(next.hash());
        if let Some(score) = sc.probe(next_cache_idx, &next, next_alpha) {
            -score
        } else if let Some(score) = stability_cutoff(&next, 4, next_alpha) {
            ctx.stats.increment_stability_cut();
            -score
        } else {
            let (sq1, sq2, sq3, sq4) = sort_last4(&ctx.empty_list);
            let score = solve4(ctx, &next, next_alpha, sq1, sq2, sq3, sq4);
            sc.store(next_cache_idx, &next, next_alpha, score);
            -score
        }
    } else {
        -shallow_search(ctx, &next, -beta, sc, next.legal_moves())
    };
    ctx.undo_endgame(sq);
    score
}

/// Scores a position with exactly one empty square.
#[doc(hidden)]
#[inline(always)]
pub fn solve_last1(player: Bitboard, alpha: DiscScore, sq: Square) -> DiscScore {
    solve::solve_last1(player, alpha, sq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Eval;
    use crate::player_color::PlayerColor;
    use crate::search::result::Iteration;
    use crate::search::strategy::SearchStrategy;
    use crate::search::threading::ThreadPool;
    use crate::search::time_control::TimeControlMode;
    use crate::transposition_table::TranspositionTable;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CutoffOccurred;

    impl SplitPointCutoff for CutoffOccurred {
        fn occurred(&self) -> bool {
            true
        }
    }

    #[test]
    fn aspiration_retries_both_window_boundaries_and_keeps_the_selected_root_moves() {
        let board = Board::from_bitboards(0xde52_7100_7814_a8a2_u64, 0x21ad_0eff_87eb_175d_u64);
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let root_moves = RootMoves::new(&board);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &root_moves, &tt, &eval);
        let pool = ThreadPool::new(1);
        let score_of = |discs| Score::from_disc_diff(DiscScore::new(discs));

        let windows = [(0, 1, Square::H6, -4), (-30, -29, Square::G2, -16)];
        for (idx, (low, high, square, exact)) in windows.into_iter().enumerate() {
            root_moves.set_pv_idx(idx);
            let original = (score_of(low), score_of(high));
            let (mut alpha, mut beta) = original;
            let score = aspiration_search(&mut ctx, &board, &mut alpha, &mut beta, pool.main());
            assert_ne!((alpha, beta), original);
            assert!(alpha < score && score < beta);
            assert_eq!(score, score_of(exact));
            root_moves.sort_from_pv_idx();
            let chosen = root_moves
                .current_pv()
                .expect("expected a selected root move");
            assert_eq!((chosen.sq, chosen.score), (square, score));
        }

        let first = root_moves.get(0).expect("expected the first PV line");
        assert_eq!((first.sq, first.score), (Square::H6, score_of(-4)));
    }

    #[test]
    fn shallow_search_preserves_a_negative_fractional_window_on_pass() {
        let player = 0xffff_fffe_u64;
        let board = Board::from_bitboards(player, !(player | 1));
        let after_pass = board.switch_players().make_move(Square::A1);
        let exact = after_pass.final_score();
        assert_eq!(exact.to_disc_diff().value(), -14);
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let root_moves = RootMoves::new(&board);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &root_moves, &tt, &eval);
        let pool = ThreadPool::new(1);

        let pass_score = Score::from_disc_diff(DiscScore::new(-2));
        for alpha in [exact - 1, exact, pass_score - 1, pass_score] {
            let beta = alpha + 1;
            let result =
                EndGameStrategy::shallow_search(&mut ctx, &board, 1, alpha, beta, pool.main());
            if result >= beta {
                assert!(result <= exact);
            } else {
                assert!(exact <= result && result <= alpha);
            }
        }
    }

    #[test]
    fn split_point_cutoff_unwinds_ec_search_without_caching_dummy_score() {
        let board = Board::from_string(
            "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX",
            PlayerColor::Black,
        )
        .unwrap();
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let root_moves = RootMoves::new(&board);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &root_moves, &tt, &eval);
        let mut caches = EndGameCaches::for_thread_count(1);
        let original_empty_count = ctx.empty_list.count();

        let score = null_window_search(
            &mut ctx,
            &board,
            DiscScore::new(49),
            &mut caches,
            &CutoffOccurred,
            board.legal_moves(),
        );

        assert_eq!(score, SPLIT_CUTOFF_SCORE);
        assert_eq!(ctx.empty_list.count(), original_empty_count);
        let cache_idx = caches.ec.slot(board.hash());
        assert!(
            caches
                .ec
                .probe(cache_idx, &board, DiscScore::new(49))
                .is_none()
        );
    }

    #[test]
    fn split_point_cutoff_restores_played_squares_and_preserves_completed_child_bound() {
        use std::cell::Cell;

        struct CutoffAfter(Cell<usize>);

        impl SplitPointCutoff for CutoffAfter {
            fn occurred(&self) -> bool {
                let remaining = self.0.get();
                self.0.set(remaining.saturating_sub(1));
                remaining == 0
            }
        }

        let mut board = Board::new();
        while board.empty_count() > 8 {
            let moves = board.legal_moves();
            if moves.is_empty() {
                board = board.switch_players();
            } else {
                let index = (13 * board.empty_count() % moves.count()) as usize;
                board = board.make_move(moves.iter().nth(index).unwrap());
            }
        }

        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let root_moves = RootMoves::new(&board);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &root_moves, &tt, &eval);
        let mut caches = EndGameCaches::for_thread_count(1);
        let original = ctx.empty_list.clone();
        let moves = board.legal_moves();
        assert!(moves.count() > 1);
        let alpha = DiscScore::new(0);
        let score = null_window_search(
            &mut ctx,
            &board,
            alpha,
            &mut caches,
            &CutoffAfter(Cell::new(2)),
            moves,
        );
        assert_eq!(ctx.empty_list.count(), original.count());
        assert_eq!(ctx.empty_list.parity(), original.parity());
        let mut square = original.first();
        assert_eq!(ctx.empty_list.first(), square);
        for _ in 1..original.count() {
            let next = original.next(square);
            assert_eq!(ctx.empty_list.next(square), next);
            square = next;
        }

        let slot = caches.ec.slot(board.hash());
        assert_eq!(caches.ec.probe(slot, &board, alpha), None);
        assert_eq!(score, SPLIT_CUTOFF_SCORE);
        let mut completed_shallow_children = 0;
        for sq in moves.iter() {
            let child = board.make_move(sq);
            let slot = caches.ec.slot(child.hash());
            assert_eq!(caches.ec.probe(slot, &child, -(alpha + 1)), None);

            for reply in child.legal_moves().iter() {
                let grandchild = child.make_move(reply);
                let slot = caches.shallow.slot(grandchild.hash());
                if caches.shallow.probe(slot, &grandchild, alpha).is_some() {
                    completed_shallow_children += 1;
                }
            }
        }
        assert_eq!(completed_shallow_children, 1);
    }

    /// Builds a 9-empty solve task whose `CompletedState` fallback carries a
    /// recognizable midgame result (depth 7, Level2, first move +4).
    fn solve_fallback_fixture(
        pool: &Arc<ThreadPool>,
        multi_pv: bool,
        callback: Option<Arc<crate::search::SearchProgressCallback>>,
    ) -> (RootMoves, SearchTask, CompletedState) {
        let board = Board::from_string(
            "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX",
            PlayerColor::Black,
        )
        .unwrap();
        let tt = Arc::new(TranspositionTable::new(0));
        let eval = Eval::shared_for_tests();
        let root_moves = RootMoves::new(&board);
        let mut fallback_moves = root_moves.snapshot();
        fallback_moves[0].score = Score::from_disc_diff(DiscScore::new(4));
        let task = SearchTask {
            board,
            mid_selectivity: Selectivity::None,
            tt,
            pool: pool.clone(),
            eval,
            level: Level::perfect(),
            multi_pv,
            callback,
            time_manager: None,
            eval_mode: None,
            corrections: Default::default(),
        };
        let mut completed = CompletedState::new(task.pv_count(root_moves.count()));
        completed.last = Some(Iteration {
            root_moves: fallback_moves,
            depth: 7,
            selectivity: Selectivity::Level2,
            is_endgame: false,
        });
        (root_moves, task, completed)
    }

    fn fallback_context<'a>(task: &'a SearchTask, root_moves: &'a RootMoves) -> SearchContext<'a> {
        SearchContext::new(
            &task.board,
            Selectivity::None,
            root_moves,
            &task.tt,
            &task.eval,
        )
    }

    fn assert_is_fallback_result(result: &SearchResult) {
        assert!(!result.is_endgame());
        assert_eq!(result.depth(), 7);
        assert_eq!(result.selectivity(), Selectivity::Level2);
        assert_eq!(
            result.score(),
            Some(Score::from_disc_diff(DiscScore::new(4)))
        );
    }

    #[test]
    fn solve_root_returns_fallback_when_aborted_before_completing_a_level() {
        let pool = ThreadPool::new(1);
        pool.abort_search();
        let (root_moves, task, completed) = solve_fallback_fixture(&pool, false, None);
        let mut ctx = fallback_context(&task, &root_moves);

        let result = solve_root(
            &mut ctx,
            &task.board,
            &task,
            pool.main(),
            Score::ZERO,
            completed,
            Selectivity::Level2,
        );

        assert_is_fallback_result(&result);
    }

    #[test]
    fn solve_root_records_timeout_only_when_discarding_a_level() {
        for expire_before_level in [false, true] {
            let pool = ThreadPool::new(1);
            let abort_state = pool.abort_state();
            let generation = abort_state.begin_search();
            let (root_moves, mut task, completed) = solve_fallback_fixture(&pool, false, None);
            let timer = Arc::new(TimeManager::new(
                TimeControlMode::Fixed {
                    time_per_move_ms: 0,
                },
                abort_state,
                task.board.empty_count(),
            ));
            task.time_manager = Some(timer.clone());
            if expire_before_level {
                assert!(timer.check_time_for(generation));
            }
            let mut ctx = fallback_context(&task, &root_moves);

            let result = solve_root(
                &mut ctx,
                &task.board,
                &task,
                pool.main(),
                Score::ZERO,
                completed,
                Selectivity::Level2,
            );

            assert_is_fallback_result(&result);
            assert!(pool.main().is_search_aborted());
            assert_eq!(timer.timed_out(), expire_before_level);
        }
    }

    #[test]
    fn multi_pv_solve_keeps_entry_fallback_until_all_pv_lines_complete() {
        let pool = ThreadPool::new(1);
        // Abort once the first PV line reports a completed selectivity level.
        let abort_pool = pool.clone();
        let (root_moves, task, completed) = solve_fallback_fixture(
            &pool,
            true,
            Some(Arc::new(move |_progress: SearchProgress| {
                abort_pool.abort_search();
            })),
        );
        let mut ctx = fallback_context(&task, &root_moves);

        let result = solve_root(
            &mut ctx,
            &task.board,
            &task,
            pool.main(),
            Score::ZERO,
            completed,
            Selectivity::Level2,
        );

        assert_is_fallback_result(&result);
    }

    /// Returns a callback that aborts `pool` on the progress report numbered
    /// by `abort_at` (1-based), which the test may set after building the task.
    fn abort_on_report(
        pool: &Arc<ThreadPool>,
        abort_at: &Arc<AtomicUsize>,
    ) -> Option<Arc<crate::search::SearchProgressCallback>> {
        let (pool, abort_at, reports) = (pool.clone(), abort_at.clone(), AtomicUsize::new(0));
        Some(Arc::new(move |_progress: SearchProgress| {
            if reports.fetch_add(1, Ordering::Relaxed) + 1 == abort_at.load(Ordering::Relaxed) {
                pool.abort_search();
            }
        }))
    }

    #[test]
    fn single_pv_solve_keeps_fallback_when_aborted_below_level2() {
        let pool = ThreadPool::new(1);
        let abort_at = Arc::new(AtomicUsize::new(1));
        let (root_moves, task, completed) =
            solve_fallback_fixture(&pool, false, abort_on_report(&pool, &abort_at));
        let mut ctx = fallback_context(&task, &root_moves);

        let result = solve_root(
            &mut ctx,
            &task.board,
            &task,
            pool.main(),
            Score::ZERO,
            completed,
            Selectivity::Level2,
        );

        assert_is_fallback_result(&result);
    }

    #[test]
    fn single_pv_solve_reports_the_last_completed_level_on_abort() {
        let pool = ThreadPool::new(1);
        let abort_at = Arc::new(AtomicUsize::new(2));
        let (root_moves, task, completed) =
            solve_fallback_fixture(&pool, false, abort_on_report(&pool, &abort_at));
        let mut ctx = fallback_context(&task, &root_moves);

        let result = solve_root(
            &mut ctx,
            &task.board,
            &task,
            pool.main(),
            Score::ZERO,
            completed,
            Selectivity::Level2,
        );

        assert!(result.is_endgame());
        assert_eq!(result.depth(), 9);
        assert_eq!(result.selectivity(), Level::ENDGAME_SELECTIVITY[1]);
    }

    #[test]
    fn multi_pv_solve_keeps_fallback_while_pv_lines_differ_in_selectivity() {
        let pool = ThreadPool::new(1);
        let abort_at = Arc::new(AtomicUsize::new(0));
        let (root_moves, task, completed) =
            solve_fallback_fixture(&pool, true, abort_on_report(&pool, &abort_at));
        let mut ctx = fallback_context(&task, &root_moves);
        // Abort once the last PV line finishes its first level, while the
        // other lines have finished every level.
        let n_levels = Level::ENDGAME_SELECTIVITY.len();
        abort_at.store(
            (ctx.root_moves.count() - 1) * n_levels + 1,
            Ordering::Relaxed,
        );

        let result = solve_root(
            &mut ctx,
            &task.board,
            &task,
            pool.main(),
            Score::ZERO,
            completed,
            Selectivity::Level2,
        );

        assert_is_fallback_result(&result);
    }

    #[test]
    fn endgame_handoff_does_not_count_midgame_lines_as_completed_endgame_lines() {
        let pool = ThreadPool::new(1);
        let abort_at = Arc::new(AtomicUsize::new(Level::ENDGAME_SELECTIVITY.len()));
        let (root_moves, task, mut completed) =
            solve_fallback_fixture(&pool, true, abort_on_report(&pool, &abort_at));
        let mut ctx = fallback_context(&task, &root_moves);
        let depth = ctx.empty_list.count();
        let midgame_score = Score::from_disc_diff(DiscScore::new(4));
        let pv = [None; crate::constants::MAX_PLY];
        assert_eq!(ctx.selectivity, Selectivity::None);
        assert!(root_moves.count() > 1);
        for (idx, row) in root_moves.snapshot().iter().enumerate() {
            root_moves.set_pv_idx(idx);
            root_moves.update(row.sq, midgame_score, true, &pv);
            completed.line_completed(&ctx, depth, false);
        }

        let result = solve_root(
            &mut ctx,
            &task.board,
            &task,
            pool.main(),
            midgame_score,
            completed,
            Selectivity::Level2,
        );

        assert!(!result.is_endgame());
        assert_eq!(result.depth(), depth);
        assert_eq!(result.selectivity(), Selectivity::None);
        assert_eq!(result.score(), Some(midgame_score));
    }

    #[test]
    fn multi_pv_solve_reports_the_iteration_once_every_pv_line_finished() {
        let pool = ThreadPool::new(1);
        let abort_at = Arc::new(AtomicUsize::new(0));
        let (root_moves, task, completed) =
            solve_fallback_fixture(&pool, true, abort_on_report(&pool, &abort_at));
        let mut ctx = fallback_context(&task, &root_moves);
        let n_levels = Level::ENDGAME_SELECTIVITY.len();
        abort_at.store(ctx.root_moves.count() * n_levels, Ordering::Relaxed);

        let result = solve_root(
            &mut ctx,
            &task.board,
            &task,
            pool.main(),
            Score::ZERO,
            completed,
            Selectivity::Level2,
        );

        assert!(result.is_endgame());
        assert_eq!(result.pv_moves().len(), ctx.root_moves.count());
    }

    #[test]
    fn initial_aspiration_window_centers_first_pv_on_base_score() {
        let base_score = Score::from_disc_diff(DiscScore::new(10));

        let (alpha, beta) = initial_aspiration_window(0, base_score, None);

        assert_eq!(
            (alpha, beta),
            (
                Score::from_disc_diff(DiscScore::new(9)),
                Score::from_disc_diff(DiscScore::new(11))
            )
        );
    }

    #[test]
    fn initial_aspiration_window_uses_previous_pv_as_later_pv_upper_bound() {
        let previous_score = Score::from_disc_diff(DiscScore::new(6));

        let (alpha, beta) = initial_aspiration_window(2, Score::ZERO, Some(previous_score));

        assert_eq!(alpha, -Score::INF);
        assert_eq!(beta, Score::from_raw(previous_score.value() + 1));
    }

    #[test]
    fn initial_aspiration_window_falls_back_to_unbounded_beta_without_previous_pv() {
        let (_alpha, beta) = initial_aspiration_window(1, Score::ZERO, None);

        assert_eq!(beta, Score::INF);
    }
}
