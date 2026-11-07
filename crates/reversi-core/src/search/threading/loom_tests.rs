//! Loom model of the YBWC split-point protocol on the real thread pool.
//!
//! Built only with `--cfg loom`, where `pvs::search` returns a fixed score
//! per position: the model explores the split-point protocol (booking, late
//! join, cutoff marking, teardown and slot reuse) with its real atomics,
//! locks and memory orderings, and loom reports any access to split-point
//! or thread state that is not ordered after a conflicting one.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::{SplitRequest, Thread, ThreadPool};
use crate::board::Board;
use crate::eval::Eval;
use crate::move_list::MoveList;
use crate::probcut::{self, Selectivity};
use crate::search::context::SearchContext;
use crate::search::node_type::NodeTypeId;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::ScaledScore;
use crate::util::sync::thread;

/// Moves the owner searches before splitting; the rest go to the split point.
const MOVES_BEFORE_SPLIT: usize = 2;

fn split(
    main: &Arc<Thread>,
    ctx: &mut SearchContext,
    board: &Board,
    node_type: NodeTypeId,
    alpha: i32,
    beta: i32,
) -> ScaledScore {
    let alpha = ScaledScore::from_disc_diff(alpha);
    main.split(
        ctx,
        SplitRequest {
            board,
            alpha,
            beta: ScaledScore::from_disc_diff(beta),
            best_score: alpha,
            best_move: Square::None,
            depth: 10,
            move_list: MoveList::new(board),
            move_count: MOVES_BEFORE_SPLIT,
            node_type,
            cut_node: false,
        },
    )
    .0
}

/// One root split: node type, alpha and beta (in discs).
type Split = (NodeTypeId, i32, i32);

/// A PV split whose window no move leaves (helpers publish PVs), then the
/// same slot again with a null window that some moves cut.
const PV_THEN_CUT: &[Split] = &[(NodeTypeId::Pv, -20, 20), (NodeTypeId::NonPv, -5, -4)];

/// Runs thread 0's part: the splits at the root, then drops the pool the way
/// `main_thread_loop` ends a search. A worker that has just left a split
/// point may still hold the pool it upgraded in `try_late_join`, so the pool
/// can be dropped on that worker.
fn run_main(
    pool_size: usize,
    splits: &[Split],
    board: Board,
    tt: Arc<TranspositionTable>,
    eval: Arc<Eval>,
) {
    let pool = ThreadPool::new(pool_size);
    pool.thinking.store(true, Ordering::Release);
    pool.notify_all();

    let main = pool.main().clone();
    main.searching.store(true, Ordering::Release);
    let mut ctx = SearchContext::new(&board, Selectivity::None, tt, eval);

    for &(node_type, alpha, beta) in splits {
        let score = split(&main, &mut ctx, &board, node_type, alpha, beta);
        if node_type == NodeTypeId::NonPv {
            assert!(score >= ScaledScore::from_disc_diff(beta));
        }
    }

    main.searching.store(false, Ordering::Release);
    pool.thinking.store(false, Ordering::Release);
}

/// Explores every interleaving of `run_main` on a pool of `pool_size`.
fn check(pool_size: usize, splits: &'static [Split]) {
    probcut::init();
    let eval = Arc::new(Eval::with_weight_files(None, None).unwrap());
    let tt = Arc::new(TranspositionTable::new(1));
    let board = Board::new();
    let iterations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = iterations.clone();

    loom::model(move || {
        counter.fetch_add(1, Ordering::Relaxed);
        let (tt, eval) = (tt.clone(), eval.clone());
        // On a spawned thread for its larger stack.
        thread::spawn(move || run_main(pool_size, splits, board, tt, eval))
            .join()
            .unwrap();
    });
    eprintln!(
        "loom explored {} executions",
        iterations.load(Ordering::Relaxed)
    );
}

#[test]
fn loom_split_point_protocol() {
    check(3, PV_THEN_CUT);
}

#[test]
fn loom_pool_dropped_by_worker() {
    check(2, &PV_THEN_CUT[1..]);
}
