use std::cell::SyncUnsafeCell;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicUsize, Ordering};

use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::empty_list::EmptyList;
use crate::eval::Eval;
use crate::eval::EvalMode;
use crate::eval::pattern_feature::PatternFeature;
use crate::move_list::ConcurrentMoveIterator;
use crate::probcut::Selectivity;
use crate::score::Score;
use crate::search::context::SearchContext;
use crate::search::counters::SearchCounters;
use crate::search::node_type::NodeTypeId;
use crate::search::root_move::RootMoves;
use crate::search::side_to_move::SideToMove;
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::Depth;
use crate::util::align::Align64;
use crate::util::bitset::AtomicBitSet;
use crate::util::spinlock;

use super::thread::SplitRequest;

/// Maximum number of threads recorded in a split point mask.
pub(super) const MAX_THREADS_PER_SPLITPOINT: u32 = 4;

/// State information for a split point in the parallel search.
pub struct SplitPointState {
    /// Whether every thread assigned to this split point is still searching.
    all_helpers_searching: AtomicBool,

    /// Alpha bound for the alpha-beta search at this node.
    alpha: AtomicI32,

    /// Best score found so far at this split point.
    best_score: AtomicI32,

    /// Best move found so far at this split point.
    best_move: AtomicU8,

    /// Flag indicating if a beta cutoff has occurred.
    cutoff: Align64<AtomicBool>,

    /// Index of the owner thread that created this split point.
    ///
    /// Relaxed atomic because late-join pre-checks read it lock-free as a
    /// heuristic hint; booking is re-validated under the split-point lock.
    owner_thread_idx: AtomicUsize,

    /// Bitmask tracking which threads are working on this split point.
    pub(super) helpers_mask: Align64<AtomicBitSet>,

    /// Depth in the split-point tree.
    ///
    /// Immutable after initialization; cached so idle helpers do not walk ancestors when choosing which split point to join.
    ///
    /// Relaxed atomic because late-join pre-checks read it lock-free as a
    /// heuristic hint; booking is re-validated under the split-point lock.
    level: AtomicUsize,

    /// Whether the parent expected this node to produce a beta cutoff.
    cut_node: AtomicBool,
}

impl SplitPointState {
    /// Returns the owner thread index.
    #[inline]
    pub(super) fn owner_thread_idx(&self) -> usize {
        self.owner_thread_idx.load(Ordering::Relaxed)
    }

    /// Sets the owner thread index.
    #[inline]
    pub(super) fn set_owner_thread_idx(&self, idx: usize) {
        self.owner_thread_idx.store(idx, Ordering::Relaxed);
    }

    /// Returns the split-point tree depth.
    #[inline]
    pub(super) fn level(&self) -> usize {
        self.level.load(Ordering::Relaxed)
    }

    /// Sets the split-point tree depth.
    #[inline]
    pub(super) fn set_level(&self, level: usize) {
        self.level.store(level, Ordering::Relaxed);
    }

    /// Returns whether the parent expected this node to produce a beta cutoff.
    #[inline]
    pub(in crate::search) fn cut_node(&self) -> bool {
        self.cut_node.load(Ordering::Relaxed)
    }

    /// Sets whether the parent expected this node to produce a beta cutoff.
    #[inline]
    pub(super) fn set_cut_node(&self, value: bool) {
        self.cut_node.store(value, Ordering::Relaxed);
    }

    /// Returns the current alpha value atomically.
    #[inline]
    pub fn alpha(&self) -> Score {
        Score::from_raw(self.alpha.load(Ordering::Relaxed))
    }

    /// Sets the alpha value atomically.
    #[inline]
    pub fn set_alpha(&self, value: Score) {
        self.alpha.store(value.value(), Ordering::Relaxed);
    }

    /// Returns the `all_helpers_searching` flag.
    #[inline]
    pub fn all_helpers_searching(&self) -> bool {
        self.all_helpers_searching.load(Ordering::Relaxed)
    }

    /// Sets the `all_helpers_searching` flag.
    #[inline]
    pub fn set_all_helpers_searching_locked(&self, value: bool) {
        self.all_helpers_searching.store(value, Ordering::Relaxed);
    }

    /// Returns the cutoff flag.
    #[inline]
    pub fn cutoff(&self) -> bool {
        self.cutoff.load(Ordering::Relaxed)
    }

    /// Clears the cutoff flag.
    #[inline]
    pub(super) fn clear_cutoff(&self) {
        self.cutoff.store(false, Ordering::Release);
    }

    /// Marks this split point as cut off, returning whether this is the first marker.
    #[inline]
    pub(super) fn mark_cutoff(&self) -> bool {
        !self.cutoff.swap(true, Ordering::AcqRel)
    }

    /// Returns the best score.
    #[inline]
    pub fn best_score(&self) -> Score {
        Score::from_raw(self.best_score.load(Ordering::Relaxed))
    }

    /// Sets the best score.
    #[inline]
    pub fn set_best_score(&self, value: Score) {
        self.best_score.store(value.value(), Ordering::Relaxed);
    }

    /// Returns the best move, if one has been found.
    #[inline]
    pub fn best_move(&self) -> Option<Square> {
        Square::from_u8(self.best_move.load(Ordering::Relaxed))
    }

    /// Sets the best move.
    #[inline]
    pub fn set_best_move(&self, value: Option<Square>) {
        self.best_move
            .store(value.map_or(u8::MAX, |sq| sq as u8), Ordering::Relaxed);
    }
}

struct SharedSearchPointers {
    tt: NonNull<TranspositionTable>,

    root_moves: NonNull<RootMoves>,

    eval: NonNull<Eval>,
}

const _: () = {
    const fn require_sync<T: Sync>() {}
    require_sync::<TranspositionTable>();
    require_sync::<RootMoves>();
    require_sync::<Eval>();
};

// SAFETY: The pointers come from shared references to Sync values that outlive every dispatch.
unsafe impl Send for SharedSearchPointers {}

// SAFETY: Dispatches create only shared references while participating in the split.
unsafe impl Sync for SharedSearchPointers {}

/// Task data for a split point containing all information needed for search.
pub struct SplitPointTask {
    /// Current board position to search from.
    pub board: Board,

    /// Which player is to move in this position.
    pub side_to_move: SideToMove,

    /// Search selectivity level (affects pruning aggressiveness).
    pub selectivity: Selectivity,

    /// Current evaluation mode (midgame or endgame).
    pub eval_mode: EvalMode,

    shared: SharedSearchPointers,

    /// List of empty squares for move generation optimization.
    pub empty_list: EmptyList,

    /// Pre-computed player pattern feature at the split point ply.
    pub p_feature: PatternFeature,

    /// Pre-computed opponent pattern feature at the split point ply.
    pub o_feature: PatternFeature,

    /// Search depth remaining from this position.
    pub depth: Depth,

    /// Beta bound for the alpha-beta search at this node.
    pub beta: Score,

    /// Type of node (PV, NonPV, or Root) for search specialization.
    pub node_type: NodeTypeId,

    /// Parent split point in the tree hierarchy.
    pub parent_split_point: Option<Arc<SplitPoint>>,
}

impl SplitPointTask {
    /// Creates task data for a split point.
    #[inline]
    pub(super) fn new(
        req: &SplitRequest,
        parent_split_point: Option<Arc<SplitPoint>>,
        ctx: &SearchContext,
    ) -> Self {
        let ply = ctx.ply();
        Self {
            board: *req.board,
            side_to_move: ctx.side_to_move,
            selectivity: ctx.selectivity,
            eval_mode: ctx.eval_mode,
            shared: SharedSearchPointers {
                tt: NonNull::from(ctx.tt),
                root_moves: NonNull::from(ctx.root_moves),
                eval: NonNull::from(ctx.eval),
            },
            empty_list: ctx.empty_list.clone(),
            p_feature: *ctx.pattern_features.p_feature(ply),
            o_feature: *ctx.pattern_features.o_feature(ply),
            depth: req.depth,
            beta: req.beta,
            node_type: req.node_type,
            parent_split_point,
        }
    }

    /// Returns raw pointers to the shared `tt`, `eval`, and `root_moves`.
    #[inline]
    pub(in crate::search) fn shared_ptrs(
        &self,
    ) -> (
        NonNull<TranspositionTable>,
        NonNull<Eval>,
        NonNull<RootMoves>,
    ) {
        (self.shared.tt, self.shared.eval, self.shared.root_moves)
    }
}

/// A split point in the parallel search tree.
///
/// Teardown ordering: fields stored outside `state` (`move_iter`, `task`, `pv`,
/// `counters`) are mutated only while the split-point lock is held, or after all
/// searchers finish (each searcher resets `helpers_mask` with Release before the
/// owner proceeds).
pub struct SplitPoint {
    /// Spinlock for fast synchronization between threads.
    mutex: spinlock::SpinLock,

    /// Atomic state, written under the mutex and read lock-free in join
    /// pre-checks. `level` and `owner_thread_idx` are Relaxed
    /// atomics used as lock-free heuristic hints; split-point initialization is
    /// published by the owner's Release update to `split_points_size` and
    /// consumed by Acquire observers.
    state: SplitPointState,

    /// Shared move iterator for the active split point.
    ///
    /// Stored separately from `state` so searchers can keep a reference to it
    /// after releasing the split-point lock.
    /// Mutated only while the split-point lock is held, or after all searchers
    /// finish (see struct-level teardown ordering).
    move_iter: SyncUnsafeCell<Option<ConcurrentMoveIterator>>,

    /// Task data containing the position and search context.
    ///
    /// Mutated only while the split-point lock is held, or after all searchers
    /// finish.
    task: SyncUnsafeCell<Option<SplitPointTask>>,

    /// Principal variation line from the best move found at this split point.
    ///
    /// Mutated under the split-point lock, or after all searchers finish (see
    /// struct-level teardown ordering).
    pv: SyncUnsafeCell<[Option<Square>; MAX_PLY]>,

    /// Accumulated search counters from all threads that searched this split point.
    ///
    /// Mutated under the split-point lock, or after all searchers finish (see
    /// struct-level teardown ordering).
    counters: SyncUnsafeCell<SearchCounters>,
}

impl Default for SplitPoint {
    fn default() -> Self {
        SplitPoint {
            mutex: spinlock::SpinLock::new(),
            state: SplitPointState {
                all_helpers_searching: AtomicBool::new(false),
                alpha: AtomicI32::new(0),
                best_score: AtomicI32::new(0),
                best_move: AtomicU8::new(u8::MAX),
                cutoff: Align64(AtomicBool::new(false)),
                owner_thread_idx: AtomicUsize::new(0),
                helpers_mask: Align64(AtomicBitSet::new()),
                level: AtomicUsize::new(0),
                cut_node: AtomicBool::new(false),
            },
            move_iter: SyncUnsafeCell::new(None),
            task: SyncUnsafeCell::new(None),
            pv: SyncUnsafeCell::new([None; MAX_PLY]),
            counters: SyncUnsafeCell::new(SearchCounters::default()),
        }
    }
}

impl SplitPoint {
    /// Returns the split point state.
    #[inline]
    pub fn state(&self) -> &SplitPointState {
        &self.state
    }

    /// Returns the shared move iterator for the active split point.
    ///
    /// # Safety
    ///
    /// The iterator must be initialized and retained for the returned borrow.
    #[inline]
    pub(super) unsafe fn move_iter(&self) -> &ConcurrentMoveIterator {
        // SAFETY: The caller keeps the initialized iterator in place.
        unsafe { (*self.move_iter.get()).as_ref().unwrap() }
    }

    /// Installs the move iterator for a new split point.
    #[inline]
    pub(super) fn set_move_iter(&self, move_iter: ConcurrentMoveIterator) {
        // SAFETY: Caller has exclusive access while initializing the split point.
        unsafe { *self.move_iter.get() = Some(move_iter) };
    }

    /// Clears the move iterator after the split point has finished.
    #[inline]
    pub(super) fn clear_move_iter(&self) {
        // SAFETY: All `move_iter()` references are scoped to `search_split_point`,
        // which completes before the searcher resets its `helpers_mask` bit.
        unsafe { *self.move_iter.get() = None };
    }

    /// Sets the task for this split point. Caller must hold the split-point lock.
    #[inline]
    pub(super) fn set_task(&self, task: SplitPointTask) {
        // SAFETY: Caller holds the split-point lock; no helper reads the task
        // before the split point is published.
        unsafe { *self.task.get() = Some(task) };
    }

    /// Returns the task for the active split point.
    ///
    /// # Safety
    ///
    /// The task must be initialized and retained for the returned borrow.
    #[inline]
    pub(super) unsafe fn task(&self) -> &SplitPointTask {
        // SAFETY: The caller keeps the initialized task in place.
        unsafe {
            (*self.task.get())
                .as_ref()
                .expect("active split point must have a task")
        }
    }

    /// Clears the task after all searchers have finished.
    #[inline]
    pub(super) fn clear_task(&self) {
        // SAFETY: All searchers have finished (`helpers_mask` is empty), so no
        // reference to the task is live; the raw write does not form a
        // `&mut SplitPointState`.
        unsafe { *self.task.get() = None };
    }

    /// Copies PV from source to the split point's internal PV storage.
    #[inline(always)]
    pub fn copy_pv(&self, src: &[Option<Square>; MAX_PLY]) {
        // SAFETY: Caller holds the split-point lock, or no helpers are active.
        unsafe { (*self.pv.get()).copy_from_slice(src) };
    }

    /// Returns a reference to the internal PV.
    ///
    /// Borrow scope must not outlive the next reuse of this split point; callers
    /// consume it immediately after `finalize_split_point`.
    #[inline]
    pub(super) fn pv(&self) -> &[Option<Square>; MAX_PLY] {
        // SAFETY: Called after all searchers have finished this split point,
        // so no concurrent writer exists.
        unsafe { &*self.pv.get() }
    }

    /// Resets counters while initializing or reusing a split point.
    #[inline]
    pub(super) fn reset_counters_locked(&self) {
        // SAFETY: Caller holds the split-point lock, or no helpers are active.
        unsafe { *self.counters.get() = SearchCounters::default() };
    }

    /// Merges helper counters into this split point's aggregate counters.
    #[inline]
    pub(super) fn merge_counters_locked(&self, counters: &SearchCounters) {
        // SAFETY: Caller holds the split-point lock, so no other writer can
        // access the aggregate counters concurrently.
        unsafe { (*self.counters.get()).merge(counters) };
    }

    /// Takes counters after all searchers have finished this split point.
    #[inline]
    pub(super) fn take_counters_after_finished(&self) -> SearchCounters {
        // SAFETY: The owner calls this only after observing all `helpers_mask`
        // bits cleared with Acquire, so no searcher can still merge counters.
        unsafe { std::mem::take(&mut *self.counters.get()) }
    }

    /// Acquires the split point's lock; the returned guard releases it on drop.
    #[inline]
    pub(in crate::search) fn lock(&self) -> spinlock::SpinLockGuard<'_> {
        self.mutex.lock()
    }
}
