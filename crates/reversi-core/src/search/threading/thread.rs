use std::cell::SyncUnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Weak};

use crate::board::Board;
use crate::move_list::{ConcurrentMoveIterator, MoveList};
use crate::search;
use crate::search::context::SearchContext;
use crate::search::counters::SearchCounters;
use crate::search::endgame::EndGameCaches;
use crate::search::node_type::{NodeTypeId, NonPV, PV, Root};
use crate::search::pvs::search_split_point;
use crate::search::strategy::{EndGameStrategy, MidGameStrategy};
use crate::square::Square;
use crate::types::{Depth, ScaledScore};
use crate::util::align::Align64;
use crate::util::spinlock;

use super::AbortState;
use super::pool::{Message, ThreadPool};
use super::split_point::{SplitPoint, SplitPointState, SplitPointTask};

/// Maximum number of split points that a single thread can have active at once.
const MAX_SPLITPOINTS_PER_THREAD: usize = 8;

/// A worker thread in the thread pool.
pub struct Thread {
    /// Mutex used with the condition variable for thread sleeping.
    pub(super) mutex_for_sleep_condition: std::sync::Mutex<()>,

    /// Spinlock protecting access to `active_split_point`.
    mutex_for_state: spinlock::SpinLock,

    /// Condition variable for waking up idle threads.
    sleep_condition: std::sync::Condvar,

    /// Unique index of this thread in the pool.
    pub(super) idx: usize,

    /// Weak reference to the thread pool this thread belongs to.
    pool: Weak<ThreadPool>,

    /// Shared abort state, cached to avoid `Weak::upgrade()` on every node.
    abort_state: Arc<AbortState>,

    /// Global cutoff generation shared by all threads in this pool.
    cutoff_epoch: Arc<Align64<AtomicU64>>,

    /// Last cutoff generation checked by this thread.
    local_seen_cutoff_epoch: AtomicU64,

    /// Whether the current active split-point chain was cut off at the seen epoch.
    local_chain_cutoff: AtomicBool,

    /// Cached pool size, avoids `Weak::upgrade()` in `can_split()`.
    pool_size: usize,

    /// Endgame cache set owned by this search thread.
    ///
    /// Access is owner-only through `endgame_caches`, on the OS thread that is
    /// currently executing search work for this `Thread`.
    endgame_caches: SyncUnsafeCell<EndGameCaches>,

    /// Shared flag indicating if the engine is thinking.
    thinking: Arc<AtomicBool>,

    /// Number of split points currently active for this thread.
    /// Atomic because it is read lock-free by other threads in `can_join` / `try_late_join`.
    ///
    /// Cache-padded: scanned cross-thread; isolate from owner-only neighbour writes.
    split_points_size: Align64<AtomicUsize>,

    /// Stack of split points created by this thread.
    /// Immutable after construction; each element is a pre-allocated `Arc<SplitPoint>`.
    split_points: [Arc<SplitPoint>; MAX_SPLITPOINTS_PER_THREAD],

    /// The split point this thread is currently working on.
    /// Protected by `mutex_for_state`.
    active_split_point: SyncUnsafeCell<Option<Arc<SplitPoint>>>,

    /// Flag indicating if the thread has completed initialization.
    pub(super) ready: AtomicBool,

    /// Flag indicating if this thread is currently searching.
    ///
    /// Cache-padded: hot cross-thread atomic; isolate from neighbour writes.
    pub(super) searching: Align64<AtomicBool>,

    /// Flag signaling the thread to exit.
    pub(super) exit: AtomicBool,
}

/// Node snapshot passed from the search to `Thread::split`.
pub struct SplitRequest<'a> {
    pub board: &'a Board,
    pub alpha: ScaledScore,
    pub beta: ScaledScore,
    pub best_score: ScaledScore,
    pub best_move: Square,
    pub depth: Depth,
    pub move_list: MoveList,
    pub move_count: usize,
    pub node_type: NodeTypeId,
    pub is_endgame: bool,
    pub cut_node: bool,
}

impl Thread {
    /// Creates a new thread with the given index.
    pub(super) fn new(
        idx: usize,
        thinking: Arc<AtomicBool>,
        abort_state: Arc<AbortState>,
        cutoff_epoch: Arc<Align64<AtomicU64>>,
        pool_size: usize,
        pool: Weak<ThreadPool>,
    ) -> Thread {
        let split_points = std::array::from_fn(|_| Arc::new(SplitPoint::default()));

        Thread {
            mutex_for_sleep_condition: std::sync::Mutex::new(()),
            mutex_for_state: spinlock::SpinLock::new(()),
            sleep_condition: std::sync::Condvar::new(),
            idx,
            pool,
            abort_state,
            cutoff_epoch,
            local_seen_cutoff_epoch: AtomicU64::new(0),
            local_chain_cutoff: AtomicBool::new(false),
            pool_size,
            endgame_caches: SyncUnsafeCell::new(EndGameCaches::for_thread_count(pool_size)),
            thinking,
            split_points_size: Align64(AtomicUsize::new(0)),
            split_points,
            active_split_point: SyncUnsafeCell::new(None),
            ready: AtomicBool::new(false),
            searching: Align64(AtomicBool::new(false)),
            exit: AtomicBool::new(false),
        }
    }

    /// Borrows this thread's endgame cache set.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(in crate::search) fn endgame_caches(&self) -> &mut EndGameCaches {
        // SAFETY: endgame caches are owner-only search scratch state. Search
        // recursion receives the `Thread` for the currently executing OS thread,
        // and no cross-thread code calls this method.
        unsafe { &mut *self.endgame_caches.get() }
    }

    /// Acquires the thread's state lock; the returned guard releases it on drop.
    pub(super) fn lock(&self) -> spinlock::SpinLockGuard<'_> {
        self.mutex_for_state.lock()
    }

    /// Checks whether this thread can create a new split point.
    ///
    /// A thread can split if:
    /// 1. Multiple threads are available (pool size > 1)
    /// 2. The thread hasn't reached its split point limit
    /// 3. Either:
    ///    - The thread has no active split point, OR
    ///    - Some threads assigned to the current split point have finished, OR
    ///    - The current split point is at the per-split-point cap and the
    ///      pool has more threads than fit in one split point
    pub fn can_split(&self) -> bool {
        let thread_pool_size = self.pool_size as u32;
        if thread_pool_size <= 1 {
            return false;
        }

        let cond = if let Some(sp) = self.active_split_point() {
            let sp_state = sp.state();
            let max_threads = sp_state.max_threads();
            !sp_state.all_helpers_searching()
                || (thread_pool_size > max_threads && sp_state.helpers_mask.count() == max_threads)
        } else {
            true
        };

        cond && (self.split_points_size.load(Ordering::Relaxed) < MAX_SPLITPOINTS_PER_THREAD)
    }

    /// Returns an immutable reference to the active split point.
    ///
    /// Safe when called by the owning thread while no concurrent writer can
    /// run (e.g., while `searching` is true, since
    /// `ThreadPool::assign_helpers_to_split_point` skips searching threads
    /// via `can_join`). Otherwise, `mutex_for_state` must be held.
    #[inline]
    fn active_split_point(&self) -> &Option<Arc<SplitPoint>> {
        // SAFETY: Either called by the owning thread, or caller holds mutex_for_state.
        unsafe { &*self.active_split_point.get() }
    }

    /// Returns a mutable reference to the active split point.
    ///
    /// Writes occur from the owning thread (in `initialize_split_point`,
    /// `try_late_join`, `finalize_split_point`) and from another thread via
    /// `ThreadPool::assign_helpers_to_split_point`. Other-thread writers
    /// always hold `mutex_for_state`; the owning thread writes lock-free
    /// only while `searching` is true (which excludes those external writers).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(super) fn active_split_point_mut(&self) -> &mut Option<Arc<SplitPoint>> {
        // SAFETY: Either called by the owning thread, or caller holds mutex_for_state.
        unsafe { &mut *self.active_split_point.get() }
    }

    /// Returns whether this thread is currently searching under a split point.
    #[inline]
    pub(in crate::search) fn has_active_split_point(&self) -> bool {
        self.active_split_point().is_some()
    }

    /// Returns `true` when the pool has more than one search thread.
    #[inline]
    pub(in crate::search) fn is_parallel_pool(&self) -> bool {
        self.pool_size > 1
    }

    /// Wakes up this thread when there is work to do.
    pub(super) fn notify_one(&self) {
        let _lock = self.mutex_for_sleep_condition.lock().unwrap();
        self.sleep_condition.notify_one();
    }

    /// Returns the deepest active split point for an observed active count.
    #[inline]
    fn deepest_split_point(&self, active_count: usize) -> Option<&Arc<SplitPoint>> {
        debug_assert!(active_count <= MAX_SPLITPOINTS_PER_THREAD);
        active_count
            .checked_sub(1)
            .and_then(|idx| self.split_points.get(idx))
    }

    /// Checks whether a beta cutoff has occurred in the current or ancestor split points.
    #[inline]
    fn cutoff_occurred(&self) -> bool {
        let Some(sp) = self.active_split_point().as_ref() else {
            return false;
        };
        self.cutoff_occurred_slow(sp)
    }

    /// Walks the split point ancestor chain checking for cutoffs.
    #[cold]
    fn cutoff_occurred_slow(&self, sp: &Arc<SplitPoint>) -> bool {
        let mut current = Some(sp);
        while let Some(sp) = current {
            let sp_state = sp.state();
            if sp_state.cutoff() {
                return true;
            }
            current = sp_state.parent_split_point.as_ref();
        }
        false
    }

    /// Forces the next `should_stop()` call to re-check the active split-point chain.
    #[inline]
    pub(super) fn reset_cutoff_cache(&self) {
        self.local_chain_cutoff.store(false, Ordering::Relaxed);
        self.local_seen_cutoff_epoch.store(
            self.cutoff_epoch.load(Ordering::Relaxed).wrapping_sub(1),
            Ordering::Relaxed,
        );
    }

    /// Marks a split point as cut off and publishes a new cutoff epoch.
    #[inline]
    pub fn mark_split_point_cutoff(&self, sp_state: &SplitPointState) {
        if sp_state.mark_cutoff() {
            self.cutoff_epoch.fetch_add(1, Ordering::Release);
        }
    }

    /// Checks whether this thread can join the given split point.
    ///
    /// A thread can join a split point if:
    /// 1. The thread is not currently searching (is idle)
    /// 2. For the "helpful owner" concept: if the thread is an owner of other
    ///    split points, it can only join split points created by its helpers
    pub(super) fn can_join(&self, sp: &Arc<SplitPoint>) -> bool {
        // Acquire pairs with Release store after updating thread state
        if self.searching.load(Ordering::Acquire) {
            return false;
        }

        // Make a local copy to be sure it doesn't become zero under our feet while
        // testing next condition and so leading to an out of bounds access.
        let size = self.split_points_size.load(Ordering::Acquire);

        // No split points means available as helper for any thread
        if size == 0 {
            return true;
        }

        // Apply "helpful owner" concept
        let Some(active_sp) = self.deepest_split_point(size) else {
            return false;
        };
        let sp_state = active_sp.state();
        sp_state.helpers_mask.test(sp.state().owner_thread_idx())
    }

    /// Books this thread as a helper on `sp`.
    ///
    /// Caller must hold this thread's state lock and `sp`'s split-point lock,
    /// and must have re-checked `can_join` under the thread lock. `sp` must
    /// still be accepting helpers under that same split-point lock hold (no
    /// cutoff, `all_helpers_searching`, below `max_threads()`).
    #[inline]
    pub(super) fn book_into(&self, sp: &Arc<SplitPoint>, sp_state: &SplitPointState) {
        sp_state.helpers_mask.set(self.idx);
        *self.active_split_point_mut() = Some(sp.clone());
        self.reset_cutoff_cache();
        self.searching.store(true, Ordering::Release);
    }

    /// Creates a split point and distributes work among available threads.
    ///
    /// This is the main entry point for parallel search. When a thread has multiple
    /// moves to search at a node, it can call this method to get help from other
    /// idle threads. The method:
    ///
    /// 1. Creates a new split point with the current search parameters
    /// 2. Finds idle threads and assigns them to help search
    /// 3. The calling thread also participates in the search (helpful owner)
    /// 4. Waits for all assigned threads to complete their work
    /// 5. Returns the best score, best move, and accumulated counters
    pub fn split(
        self: &Arc<Self>,
        ctx: &mut SearchContext,
        req: SplitRequest,
    ) -> (ScaledScore, Square, SearchCounters) {
        // Pick the next available split point
        let sp = &self.split_points[self.split_points_size.load(Ordering::Relaxed)];

        // Initialize the split point with search parameters
        self.initialize_split_point(sp, ctx, req);

        // Enter idle loop as owner thread - will return when all searchers finish
        self.idle_loop();

        // Clean up the split point
        self.finalize_split_point(sp);

        // All searchers have finished this split point, but other threads may
        // still hold brief lock-free `&SplitPointState` borrows via
        // `try_late_join`'s pre-check. Stay on `sp.state()` (`&`): `task`, `pv`,
        // and `counters` live outside `SplitPointState`, so none requires
        // `&mut SplitPointState`; this follows the same aliasing discipline as
        // the atomic fields.
        let sp_state = sp.state();
        ctx.set_pv(sp.pv());
        let counters = sp.take_counters_after_finished();

        (sp_state.best_score(), sp_state.best_move(), counters)
    }

    /// Initializes a split point with search parameters and finds workers.
    #[inline]
    fn initialize_split_point(&self, sp: &Arc<SplitPoint>, ctx: &SearchContext, req: SplitRequest) {
        let move_iter = ConcurrentMoveIterator::from_offset(req.move_list, req.move_count);

        debug_assert!(self.searching.load(Ordering::Acquire));
        debug_assert!(self.split_points_size.load(Ordering::Relaxed) < MAX_SPLITPOINTS_PER_THREAD);

        let _guard = sp.lock();
        sp.set_move_iter(move_iter);
        // No contention here until split_points_size is incremented
        let sp_state = sp.state_mut();
        sp_state.set_owner_thread_idx(self.idx);
        sp_state.parent_split_point = self.active_split_point().clone();
        sp_state.set_level(
            sp_state
                .parent_split_point
                .as_ref()
                .map_or(0, |parent| parent.state().level() + 1),
        );
        sp_state.helpers_mask.clear();
        sp_state.helpers_mask.set(self.idx);
        sp_state.depth = req.depth;
        sp_state.set_best_score(req.best_score);
        sp_state.set_best_move(req.best_move);
        sp_state.set_alpha(req.alpha);
        sp_state.beta = req.beta;
        sp_state.node_type = req.node_type;
        sp.set_task(SplitPointTask::new(req.board, ctx));
        sp.reset_counters_locked();
        sp_state.clear_cutoff();
        sp_state.set_all_helpers_searching(true); // Must be set under lock protection
        sp.copy_pv(ctx.get_pv());
        sp_state.is_endgame = req.is_endgame;
        sp_state.set_cut_node(req.cut_node);

        self.split_points_size.fetch_add(1, Ordering::Release);
        *self.active_split_point_mut() = Some(sp.clone());
        self.reset_cutoff_cache();

        // Try to allocate available threads
        if let Some(pool) = self.pool.upgrade() {
            pool.assign_helpers_to_split_point(sp);
        }

        // Everything is set up. The owner thread enters the idle loop, from which
        // it will instantly launch a search, because its 'searching' flag is set.
        // The thread will return from the idle loop when all searchers have finished
        // their work at this split point.
    }

    /// Cleans up after all threads have finished working on a split point.
    #[inline]
    fn finalize_split_point(&self, sp: &Arc<SplitPoint>) {
        debug_assert!(!self.searching.load(Ordering::Acquire));

        // In the helpful owner concept, an owner can help only a sub-tree of its
        // split point and because everything is finished here, it's not possible
        // for the owner to be booked.
        {
            let _guard = self.lock();

            // We have returned from the idle loop, which means that all threads are
            // finished. Note that decreasing split_points_size must be done under lock
            // protection to avoid a race with Thread::can_join().
            self.searching.store(true, Ordering::Release);
            self.split_points_size.fetch_sub(1, Ordering::Release);
            *self.active_split_point_mut() = sp.state().parent_split_point.clone();
            self.reset_cutoff_cache();
        }

        // Clear task data after releasing thread lock to minimize lock duration
        sp.clear_task();
        sp.clear_move_iter();
    }

    /// Runs the main loop for worker threads and split-point owners.
    ///
    /// This method implements the core logic for thread synchronization:
    ///
    /// 1. **Owner Mode**: If called from split(), acts as the owner thread
    ///    and waits for all assigned searchers to finish before returning
    ///
    /// 2. **Helper Mode**: If called at thread creation, waits for work assignments
    ///    and executes search tasks when assigned to split points
    pub(super) fn idle_loop(self: &Arc<Self>) {
        // 'this_sp' is Some only when called from split() (not at thread creation).
        // This means we are the split point's owner.
        let this_sp = self.active_split_point().clone();

        // Main loop - continues until thread exit is signaled
        while !self.exit.load(Ordering::Acquire) {
            // Check if we're the owner of a split point and all searchers have finished
            if let Some(ref sp) = this_sp
                && sp.state().helpers_mask.none()
            {
                break;
            }

            // If this thread has been assigned work, launch a search
            while self.searching.load(Ordering::Acquire) {
                let sp = {
                    let _guard = self.lock();
                    self.active_split_point()
                        .clone()
                        .expect("searching thread must have an active split point")
                };

                // The lock acquisition synchronizes with initialize_split_point,
                // after which the task and split parameters stay stable until
                // every helper has finished.
                let guard = sp.lock();
                let board = sp.task().board;
                let depth = sp.state().depth;
                let node_type = sp.state().node_type;
                let mut ctx = SearchContext::from_split_point(&sp);
                drop(guard);

                self.dispatch_search(&mut ctx, &board, depth, node_type, &sp);

                {
                    let _guard = self.lock();
                    self.searching.store(false, Ordering::Release);
                }

                // Publish counters before clearing helpers_mask: the Release on
                // reset() pairs with the owner's Acquire in helpers_mask.none().
                // Stay on `sp.state()` (`&`): counters live outside `state`,
                // and the remaining writes are atomic, so this path stays
                // aliasing-compatible with concurrent lock-free readers in
                // `try_late_join`.
                let guard = sp.lock();
                let sp_state = sp.state();
                sp.merge_counters_locked(&ctx.counters);
                sp_state.set_all_helpers_searching(false);
                sp_state.helpers_mask.reset(self.idx);

                // After clearing our helpers_mask bit, the owner may observe none() and
                // tear down the split point, so we must not access sp data after this.
                drop(guard);

                self.try_late_join();
            }

            // If search is finished then sleep, otherwise just yield
            if !self.thinking.load(Ordering::Acquire) {
                debug_assert!(this_sp.is_none());

                let lock = self.mutex_for_sleep_condition.lock().unwrap();
                self.ready.store(true, Ordering::Release);
                let _guard = self
                    .sleep_condition
                    .wait_while(lock, |_| {
                        !self.exit.load(Ordering::Acquire) && !self.thinking.load(Ordering::Acquire)
                    })
                    .unwrap();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Runs the main thread message processing loop.
    ///
    /// This is the entry point for the main thread (thread 0), which handles
    /// control messages from the thread pool and runs the root search. Worker
    /// threads wait in `idle_loop` for split-point assignments.
    ///
    /// The main thread:
    ///
    /// 1. Receives search tasks via the message channel
    /// 2. Coordinates the search by waking worker threads
    /// 3. Executes the root search
    /// 4. Sends results back via the result channel
    /// 5. Handles shutdown requests
    pub(super) fn main_thread_loop(self: Arc<Self>, receiver: Receiver<Message>) {
        loop {
            // Check exit flag before blocking on receive
            if self.exit.load(Ordering::Acquire) {
                break;
            }

            // Block waiting for next message
            let message = receiver.recv();

            match message {
                Ok(Message::StartThinking(task, result_sender)) => {
                    // Mark this thread as actively searching
                    self.searching.store(true, Ordering::Release);

                    // Keep a reference to the pool before task is consumed
                    let pool = task.pool.clone();

                    // Wake up worker threads to participate in parallel search
                    pool.notify_all();

                    // Execute root search - this is where the main work happens
                    let result = search::search_root(task, &self);

                    // Search complete - update state and send result
                    self.searching.store(false, Ordering::Release);
                    pool.thinking.store(false, Ordering::Release);

                    // Send result back to caller
                    // Ignore error if receiver was dropped (caller gave up waiting)
                    let _ = result_sender.send(result);
                }
                Ok(Message::Exit) => {
                    self.exit.store(true, Ordering::Release);
                    break;
                }
                Err(_) => {
                    // Channel disconnected - sender was dropped
                    break;
                }
            }
        }
    }

    /// Dispatches to the appropriate search function based on search strategy and node type.
    fn dispatch_search(
        self: &Arc<Self>,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        node_type: NodeTypeId,
        sp: &Arc<SplitPoint>,
    ) {
        match (sp.state().is_endgame, node_type) {
            // Endgame searches
            (true, NodeTypeId::NonPv) => {
                search_split_point::<NonPV, EndGameStrategy>(ctx, board, depth, self, sp);
            }
            (true, NodeTypeId::Pv) => {
                search_split_point::<PV, EndGameStrategy>(ctx, board, depth, self, sp);
            }
            (true, NodeTypeId::Root) => {
                search_split_point::<Root, EndGameStrategy>(ctx, board, depth, self, sp);
            }
            // Midgame searches
            (false, NodeTypeId::NonPv) => {
                search_split_point::<NonPV, MidGameStrategy>(ctx, board, depth, self, sp);
            }
            (false, NodeTypeId::Pv) => {
                search_split_point::<PV, MidGameStrategy>(ctx, board, depth, self, sp);
            }
            (false, NodeTypeId::Root) => {
                search_split_point::<Root, MidGameStrategy>(ctx, board, depth, self, sp);
            }
        }
    }

    /// Tries to join an existing split point after finishing current work.
    ///
    /// When a thread finishes its work, it can try to help other threads
    /// by joining their split points. This method finds the best available
    /// split point to join based on:
    ///
    /// 1. The split point must have room for more helpers
    /// 2. All currently assigned threads must still be searching
    /// 3. The thread must be able to join (helpful owner rules)
    /// 4. Prefer split points higher in the tree (lower level)
    fn try_late_join(&self) {
        let Some(pool) = self.pool.upgrade() else {
            return;
        };

        let mut best_sp = None;
        let mut min_level = usize::MAX;

        for th in &pool.threads {
            // split_points_size is atomic; Acquire pairs with Release in split()/finalize_split_point().
            let size = th.split_points_size.load(Ordering::Acquire);
            if size == 0 {
                continue;
            }

            // split_points[] elements are immutable Arc pointers (created once in Thread::new),
            // so a valid observed size can be used without the thread lock.
            let Some(sp) = th.deepest_split_point(size) else {
                continue;
            };
            let sp_state = sp.state();

            if sp_state.cutoff()
                || sp_state.helpers_mask.count() >= sp_state.max_threads()
                || !sp_state.all_helpers_searching()
                || !self.can_join(sp)
            {
                continue;
            }

            let level = sp_state.level();
            if level < min_level {
                min_level = level;
                best_sp = Some(sp);
                if level == 0 {
                    break;
                }
            }
        }

        let Some(sp) = best_sp else {
            return;
        };

        let _guard = sp.lock();

        let sp_state = sp.state();
        if !sp_state.cutoff()
            && sp_state.all_helpers_searching()
            && sp_state.helpers_mask.count() < sp_state.max_threads()
        {
            let _thread_guard = self.lock();

            if self.can_join(sp) {
                self.book_into(sp, sp_state);
            }
        }
    }

    /// Returns `true` if the search has been aborted (e.g., by deadline or external request).
    #[inline]
    pub fn is_search_aborted(&self) -> bool {
        self.abort_state.is_aborted()
    }

    /// Returns `true` when a beta cutoff has occurred in the current or
    /// ancestor split points.
    #[inline]
    pub(in crate::search) fn split_point_cutoff_occurred(&self) -> bool {
        if self.local_chain_cutoff.load(Ordering::Relaxed) {
            return true;
        }

        let cutoff_epoch = self.cutoff_epoch.load(Ordering::Acquire);
        if cutoff_epoch != self.local_seen_cutoff_epoch.load(Ordering::Relaxed) {
            self.local_seen_cutoff_epoch
                .store(cutoff_epoch, Ordering::Relaxed);
            let cutoff_occurred = self.cutoff_occurred();
            self.local_chain_cutoff
                .store(cutoff_occurred, Ordering::Relaxed);
            if cutoff_occurred {
                return true;
            }
        }

        false
    }

    /// Returns `true` when the current branch should abandon its result:
    /// either a beta cutoff has occurred on an ancestor split point, or the
    /// whole search has been aborted.
    #[inline]
    pub fn should_stop(&self) -> bool {
        if self.split_point_cutoff_occurred() {
            return true;
        }

        self.is_search_aborted()
    }
}
