use super::context::SearchContext;
use super::*;
use crate::eval::{EVAL_FILE_NAME, EVAL_SM_FILE_NAME};
use crate::level::Level;
use crate::score::{DiscScore, Score};
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static SECOND: AtomicBool = AtomicBool::new(false);
static WITNESSES: AtomicUsize = AtomicUsize::new(0);
static STORES: Mutex<Vec<(usize, i32, i32)>> = Mutex::new(Vec::new());
thread_local! {
    static SHADOW: Cell<bool> = const { Cell::new(false) };
    static CONTINUOUS: RefCell<endgame::EndGameCaches> = RefCell::new(endgame::EndGameCaches::for_thread_count(1));
    static RESTARTED: RefCell<endgame::EndGameCaches> = RefCell::new(endgame::EndGameCaches::for_thread_count(1));
    static VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn stored(slot: usize, board: &Board, alpha: DiscScore, score: DiscScore) {
    if !SECOND.load(Ordering::Relaxed)
        && !SHADOW.with(Cell::get)
        && board.player().bits() == 0x004602466a1283ff
        && board.opponent().bits() == 0x3f317db995ed7c00
    {
        STORES
            .lock()
            .unwrap()
            .push((slot, alpha.value(), score.value()));
    }
}

fn replay(
    ctx: &SearchContext,
    board: &Board,
    alpha: DiscScore,
    cache: &mut endgame::EndGameCaches,
) -> DiscScore {
    let mut independent =
        SearchContext::new(board, ctx.selectivity, ctx.root_moves, ctx.tt, ctx.eval);
    independent.n_empties = ctx.n_empties;
    let empties = endgame::Empties::new(board);
    match empties.count() {
        5 => endgame::shallow_search::<5>(
            &mut independent,
            board,
            alpha,
            &mut cache.shallow,
            board.legal_moves(),
            empties,
        ),
        6 => endgame::shallow_search::<6>(
            &mut independent,
            board,
            alpha,
            &mut cache.shallow,
            board.legal_moves(),
            empties,
        ),
        _ => endgame::null_window_search(
            &mut independent,
            board,
            alpha,
            cache,
            &endgame::IgnoreSplitPointCutoff,
            board.legal_moves(),
            empties,
        ),
    }
}

pub(super) fn observe(
    ctx: &SearchContext,
    board: &Board,
    alpha: DiscScore,
    raw_alpha: Score,
    raw_beta: Score,
    hit: Option<DiscScore>,
    native: DiscScore,
) {
    if ctx.n_empties <= 4 {
        return;
    }
    assert!(ctx.n_empties < 12);
    println!(
        "{} {} {} {} {} {} {} {}",
        SECOND.load(Ordering::Relaxed) as u8,
        board.player().bits(),
        board.opponent().bits(),
        ctx.n_empties,
        raw_alpha.value(),
        raw_beta.value(),
        ctx.selectivity as u8,
        Score::from_disc_diff(native).value()
    );
    SHADOW.with(|flag| flag.set(true));
    CONTINUOUS.with_borrow_mut(|cache| assert_eq!(replay(ctx, board, alpha, cache), native));
    if SECOND.load(Ordering::Relaxed) {
        let visit = VISITS.with(|count| {
            count.set(count.get() + 1);
            count.get()
        });
        let cold = RESTARTED.with_borrow_mut(|cache| replay(ctx, board, alpha, cache));
        if visit == 1 {
            assert_eq!(board.player().bits(), 0x004602466a1283ff);
            assert_eq!(board.opponent().bits(), 0x3f317db995ed7c00);
            assert_eq!(ctx.n_empties, 5);
            assert_eq!(alpha.value(), -1);
            assert_eq!(hit, Some(native));
            assert_eq!((native.value(), cold.value()), (2, 0));
            assert_eq!(*STORES.lock().unwrap(), [(6905, -1, 0), (6905, 0, 2)]);
            WITNESSES.fetch_add(1, Ordering::Relaxed);
        }
    }
    SHADOW.with(|flag| flag.set(false));
}

pub fn run(root: &Path) {
    let eval = Eval::with_weight_files(
        Some(&root.join(EVAL_FILE_NAME)),
        Some(&root.join(EVAL_SM_FILE_NAME)),
    )
    .unwrap();
    let resources = SearchSharedResources::from_eval(Arc::new(eval), 1, Some(1));
    let mut engine = Search::from_shared_resources(&resources);
    let board = Board::from_bitboards(0x2101072397ef7e7c, 0x0e76685c28108181);
    let options = SearchRunOptions::with_level(Level::preset(30));
    for second in [false, true] {
        SECOND.store(second, Ordering::Relaxed);
        match engine.run(&board, &options) {
            SearchResult::BestMove {
                score,
                is_endgame: true,
                selectivity: Selectivity::None,
                ..
            } => assert_eq!(score.value(), 0),
            _ => panic!("expected a normally completed exact endgame"),
        }
    }
    assert_eq!(WITNESSES.load(Ordering::Relaxed), 1);
    eprintln!(
        "Both exact searches returned 0. At the second search's first cached boundary, the native and continuous caches returned 2 and the restarted cache returned 0."
    );
}
