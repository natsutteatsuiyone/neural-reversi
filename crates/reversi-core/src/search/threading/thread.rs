use std::cell::SyncUnsafeCell;
use std::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread;

use crate::board::Board;
use crate::move_list::{ConcurrentMoveIterator, MoveList};
use crate::score::Score;
use crate::search;
use crate::search::context::SearchContext;
use crate::search::counters::SearchCounters;
use crate::search::endgame::EndGameCaches;
use crate::search::node_type::{NodeTypeId, NonPV, PV, Root};
use crate::search::pvs::search_split_point;
use crate::square::Square;
use crate::types::Depth;
use crate::util::align::Align64;
use crate::util::spinlock;

use super::AbortState;
use super::in_flight::{Claim, InFlightTable};
use super::pool::{Message, ThreadPool};
use super::split_point::{MAX_THREADS_PER_SPLITPOINT, SplitPoint, SplitPointState, SplitPointTask};

/// Maximum number of split points that a single thread can have active at once.
const MAX_SPLITPOINTS_PER_THREAD: usize = 8;

/// Search the main thread hands to the other threads of the pool, such as a
/// Lazy SMP helper search; each runs it until it returns or is aborted.
pub(in crate::search) type HelperJob = Arc<dyn Fn(&Arc<Thread>) + Send + Sync>;

/// A worker thread in the thread pool.
pub struct Thread {
    /// Mutex used with the condition variable for thread sleeping.
    pub(super) mutex_for_sleep_condition: Mutex<()>,

    /// Spinlock protecting access to `active_split_point`.
    mutex_for_state: spinlock::SpinLock,

    /// Condition variable for waking up idle threads.
    sleep_condition: Condvar,

    /// Unique index of this thread in the pool.
    pub(super) idx: usize,

    /// Weak reference to the thread pool this thread belongs to.
    pool: Weak<ThreadPool>,

    /// Shared abort state, cached to avoid `Weak::upgrade()` on every node.
    abort_state: Arc<AbortState>,

    /// Helper job this thread is booked for, taken when the job starts.
    helper_job: Mutex<Option<HelperJob>>,

    /// Set by the main thread to end this thread's helper job.
    helper_job_aborted: AtomicBool,

    /// Global cutoff generation shared by all threads in this pool.
    cutoff_epoch: Arc<Align64<AtomicU64>>,

    /// Last cutoff generation checked by this thread.
    local_seen_cutoff_epoch: AtomicU64,

    /// Whether the current active split-point chain was cut off at the seen epoch.
    local_chain_cutoff: AtomicBool,

    /// Cached pool size, avoids `Weak::upgrade()` in `can_split()`.
    pool_size: usize,

    in_flight: Option<Arc<InFlightTable>>,

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

    failed: AtomicBool,

    /// Flag signaling the thread to exit.
    pub(super) exit: AtomicBool,
}

/// Node snapshot passed from the search to `Thread::split`.
pub struct SplitRequest<'a> {
    pub board: &'a Board,
    pub alpha: Score,
    pub beta: Score,
    pub best_score: Score,
    pub best_move: Option<Square>,
    pub depth: Depth,
    pub move_list: MoveList,
    pub move_count: usize,
    pub node_type: NodeTypeId,
    pub cut_node: bool,
}

struct SplitUnwindGuard<'a> {
    thread: &'a Thread,
    sp: &'a Arc<SplitPoint>,
}

impl Drop for SplitUnwindGuard<'_> {
    fn drop(&mut self) {
        self.thread.abandon_active_split();
        while !self.sp.state().helpers_mask.none() {
            thread::yield_now();
        }
        self.thread.finalize_split_point(self.sp);
    }
}

impl Thread {
    /// Creates a new thread with the given index.
    pub(super) fn new(
        idx: usize,
        thinking: Arc<AtomicBool>,
        abort_state: Arc<AbortState>,
        cutoff_epoch: Arc<Align64<AtomicU64>>,
        pool_size: usize,
        in_flight: Option<Arc<InFlightTable>>,
        pool: Weak<ThreadPool>,
    ) -> Thread {
        let split_points = std::array::from_fn(|_| Arc::new(SplitPoint::default()));

        Thread {
            mutex_for_sleep_condition: Mutex::new(()),
            mutex_for_state: spinlock::SpinLock::new(),
            sleep_condition: Condvar::new(),
            idx,
            pool,
            abort_state,
            helper_job: Mutex::new(None),
            helper_job_aborted: AtomicBool::new(false),
            cutoff_epoch,
            local_seen_cutoff_epoch: AtomicU64::new(0),
            local_chain_cutoff: AtomicBool::new(false),
            pool_size,
            in_flight,
            endgame_caches: SyncUnsafeCell::new(EndGameCaches::for_thread_count(pool_size)),
            thinking,
            split_points_size: Align64(AtomicUsize::new(0)),
            split_points,
            active_split_point: SyncUnsafeCell::new(None),
            ready: AtomicBool::new(false),
            searching: Align64(AtomicBool::new(false)),
            failed: AtomicBool::new(false),
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
            !sp_state.all_helpers_searching()
                || (thread_pool_size > MAX_THREADS_PER_SPLITPOINT
                    && sp_state.helpers_mask.count() == MAX_THREADS_PER_SPLITPOINT)
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
    /// `try_late_join`, `finalize_split_point`, and when it leaves a split
    /// point) and from another thread via
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

    /// Returns this thread's index in the pool; the main thread is 0.
    #[inline]
    pub(in crate::search) fn idx(&self) -> usize {
        self.idx
    }

    /// Returns `true` when the pool has more than one search thread.
    #[inline]
    pub(in crate::search) fn is_parallel_pool(&self) -> bool {
        self.pool_size > 1
    }

    #[inline]
    pub(in crate::search) fn claim(&self, child: &Board) -> Claim<'_> {
        self.in_flight
            .as_ref()
            .map_or(Claim::Untracked, |table| table.claim(child.hash()))
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
            // SAFETY: The worker's participation retains its active task and each ancestor.
            current = unsafe { sp.task() }.parent_split_point.as_ref();
        }
        false
    }

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
        if self.searching.load(Ordering::Acquire) || self.failed.load(Ordering::Relaxed) {
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
    /// cutoff, `all_helpers_searching`, below [`MAX_THREADS_PER_SPLITPOINT`]).
    #[inline]
    pub(super) fn book_into(&self, sp: &Arc<SplitPoint>, sp_state: &SplitPointState) {
        let assignment = sp.clone();
        sp_state.helpers_mask.set(self.idx);
        *self.active_split_point_mut() = Some(assignment);
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
    ) -> (Score, Option<Square>, SearchCounters) {
        // Pick the next available split point
        let sp = &self.split_points[self.split_points_size.load(Ordering::Relaxed)];

        let unwind_guard = self.initialize_split_point(sp, ctx, req);

        // Enter idle loop as owner thread - will return when all searchers finish
        self.idle_loop();

        std::mem::forget(unwind_guard);

        // Clean up the split point
        self.finalize_split_point(sp);

        let sp_state = sp.state();
        ctx.set_pv(sp.pv());
        let counters = sp.take_counters_after_finished();

        (sp_state.best_score(), sp_state.best_move(), counters)
    }

    /// Initializes a split point with search parameters and finds workers.
    #[inline]
    fn initialize_split_point<'a>(
        &'a self,
        sp: &'a Arc<SplitPoint>,
        ctx: &'a SearchContext,
        req: SplitRequest,
    ) -> SplitUnwindGuard<'a> {
        let task = SplitPointTask::new(&req, self.active_split_point().clone(), ctx);
        let move_iter = ConcurrentMoveIterator::from_offset(req.move_list, req.move_count);
        let assignment = sp.clone();

        debug_assert!(self.searching.load(Ordering::Acquire));
        debug_assert!(self.split_points_size.load(Ordering::Relaxed) < MAX_SPLITPOINTS_PER_THREAD);

        let unwind_guard;
        {
            let _guard = sp.lock();
            sp.set_move_iter(move_iter);
            let sp_state = sp.state();
            sp_state.set_owner_thread_idx(self.idx);
            sp_state.set_level(
                task.parent_split_point
                    .as_ref()
                    .map_or(0, |parent| parent.state().level() + 1),
            );
            sp_state.helpers_mask.clear();
            sp_state.helpers_mask.set(self.idx);
            sp_state.set_best_score(req.best_score);
            sp_state.set_best_move(req.best_move);
            sp_state.set_alpha(req.alpha);
            sp.set_task(task);
            sp.reset_counters_locked();
            sp_state.clear_cutoff();
            sp_state.set_all_helpers_searching_locked(true);
            sp.copy_pv(ctx.pv());
            sp_state.set_cut_node(req.cut_node);

            *self.active_split_point_mut() = Some(assignment);
            self.reset_cutoff_cache();
            unwind_guard = SplitUnwindGuard { thread: self, sp };
            self.split_points_size.fetch_add(1, Ordering::Release);

            if let Some(pool) = self.pool.upgrade() {
                pool.assign_helpers_to_split_point(sp);
            }
        }

        unwind_guard
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
            // SAFETY: This owner has not cleared or reused the completed slot.
            *self.active_split_point_mut() = unsafe { sp.task() }.parent_split_point.clone();
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
    /// 2. **Worker Mode**: If called at thread creation, waits to be booked and
    ///    then runs a helper job or searches its share of a split point
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

            // If this thread has been booked, run what it was booked for
            while self.searching.load(Ordering::Acquire) {
                let helper_job = self.helper_job.lock().unwrap().take();
                match helper_job {
                    Some(job) => self.run_helper_job(&job),
                    None => self.search_active_split_point(),
                }
            }

            // If search is finished then sleep, otherwise just yield
            if !self.thinking.load(Ordering::Acquire) {
                debug_assert!(this_sp.is_none());

                let mut guard = self.mutex_for_sleep_condition.lock().unwrap();
                self.ready.store(true, Ordering::Release);
                while !self.exit.load(Ordering::Acquire) && !self.thinking.load(Ordering::Acquire) {
                    guard = self.sleep_condition.wait(guard).unwrap();
                }
            } else {
                thread::yield_now();
            }
        }
    }

    /// Runs the helper job this thread is booked for, then becomes idle.
    fn run_helper_job(self: &Arc<Self>, job: &HelperJob) {
        struct FinishJob<'a>(&'a Thread);

        impl Drop for FinishJob<'_> {
            fn drop(&mut self) {
                let _guard = self.0.lock();
                if thread::panicking() {
                    self.0.failed.store(true, Ordering::Relaxed);
                }
                self.0.searching.store(false, Ordering::Release);
            }
        }

        let _finish_job = FinishJob(self);
        job(self);
    }

    fn detach_split(&self, _guard: &spinlock::SpinLockGuard<'_>) -> Option<Arc<SplitPoint>> {
        let sp = self.active_split_point_mut().take();
        self.reset_cutoff_cache();
        self.searching.store(false, Ordering::Release);
        sp
    }

    fn abandon_active_split(&self) {
        let sp = {
            let guard = self.lock();
            self.failed.store(true, Ordering::Relaxed);
            self.detach_split(&guard)
        };
        if let Some(sp) = sp {
            let _guard = sp.lock();
            let state = sp.state();
            if state.owner_thread_idx() != self.idx {
                self.abort_state.request_abort();
            }
            state.set_all_helpers_searching_locked(false);
            state.helpers_mask.reset(self.idx);
        }
    }

    /// Searches this thread's share of its active split point, then tries to
    /// join another one.
    fn search_active_split_point(self: &Arc<Self>) {
        struct UnwindCleanup<'a>(&'a Thread);

        impl Drop for UnwindCleanup<'_> {
            fn drop(&mut self) {
                self.0.abandon_active_split();
            }
        }

        let cleanup = UnwindCleanup(self);
        let sp = {
            let _guard = self.lock();
            self.active_split_point()
                .clone()
                .expect("searching thread must have an active split point")
        };

        let counters = {
            // SAFETY: Our participation bit retains the published slot until this scope ends.
            let (task, moves) = unsafe { (sp.task(), sp.move_iter()) };
            let mut ctx = unsafe { SearchContext::from_split_task(task) };
            self.dispatch_search(&mut ctx, task, moves, &sp);
            ctx.counters()
        };

        // Publish counters before clearing helpers_mask: the Release on
        // reset() pairs with the owner's Acquire in helpers_mask.none().
        // Stay on `sp.state()` (`&`): counters live outside `state`,
        // and the remaining writes are atomic, so this path stays
        // aliasing-compatible with concurrent lock-free readers in
        // `try_late_join`.
        let guard = sp.lock();
        let sp_state = sp.state();
        sp.merge_counters_locked(&counters);
        {
            let thread_guard = self.lock();
            self.detach_split(&thread_guard);
            sp_state.set_all_helpers_searching_locked(false);
            sp_state.helpers_mask.reset(self.idx);
        }

        // After clearing our helpers_mask bit, the owner may observe none() and
        // tear down the split point, so we must not access sp data after this.
        drop(guard);
        std::mem::forget(cleanup);

        self.try_late_join();
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
            if self.exit.load(Ordering::Acquire) {
                break;
            }

            let message = receiver.recv();

            match message {
                Ok(Message::StartThinking(task, stop, result_sender)) => {
                    self.searching.store(true, Ordering::Release);

                    // Keep a reference to the pool before task is consumed
                    let pool = task.pool.clone();

                    // Wake up worker threads to participate in parallel search
                    pool.notify_all();

                    // Execute root search - this is where the main work happens
                    let result = search::search_root(task, &self);

                    // Untie the caller's stop token: this search is over.
                    drop(stop);

                    // Search complete - update state and send result
                    self.searching.store(false, Ordering::Release);
                    pool.thinking.store(false, Ordering::Release);
                    // Leave ThreadPool::drop to the caller, which owns the pool.
                    drop(pool);

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

    /// Dispatches to the endgame split-point search for the node type.
    fn dispatch_search(
        self: &Arc<Self>,
        ctx: &mut SearchContext,
        task: &SplitPointTask,
        moves: &ConcurrentMoveIterator,
        sp: &Arc<SplitPoint>,
    ) {
        match task.node_type {
            NodeTypeId::NonPv => {
                search_split_point::<NonPV>(ctx, task, moves, self, sp);
            }
            NodeTypeId::Pv => {
                search_split_point::<PV>(ctx, task, moves, self, sp);
            }
            NodeTypeId::Root => {
                search_split_point::<Root>(ctx, task, moves, self, sp);
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
                || sp_state.helpers_mask.count() >= MAX_THREADS_PER_SPLITPOINT
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
            && sp_state.helpers_mask.count() < MAX_THREADS_PER_SPLITPOINT
        {
            let _thread_guard = self.lock();

            if self.can_join(sp) {
                self.book_into(sp, sp_state);
            }
        }
    }

    /// Books this idle thread for the helper job `job`.
    pub(super) fn book_helper_job(&self, job: HelperJob) {
        let _guard = self.lock();
        if self.failed.load(Ordering::Relaxed) {
            return;
        }
        debug_assert!(!self.searching.load(Ordering::Relaxed));
        *self.helper_job.lock().unwrap() = Some(job);
        self.searching.store(true, Ordering::Release);
    }

    /// Asks this thread to end its helper job.
    pub(super) fn abort_helper_job(&self) {
        self.helper_job_aborted.store(true, Ordering::Relaxed);
    }

    /// Waits until this thread has ended its helper job, then withdraws the
    /// abort request so later searches run normally.
    pub(super) fn wait_for_helper_job(&self) {
        while self.searching.load(Ordering::Acquire) {
            spin_loop();
        }
        self.helper_job_aborted.store(false, Ordering::Relaxed);
    }

    /// Returns `true` if this thread's search has been aborted: the whole
    /// search by a deadline or an external request, or this thread's helper
    /// job by the main thread.
    #[inline]
    pub fn is_search_aborted(&self) -> bool {
        self.abort_state.is_aborted() || self.helper_job_aborted.load(Ordering::Relaxed)
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

    #[inline]
    pub(in crate::search) fn should_stop_after_split(&self) -> bool {
        if self.cutoff_occurred() {
            self.local_chain_cutoff.store(true, Ordering::Relaxed);
            return true;
        }

        self.is_search_aborted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Eval;
    use crate::probcut::Selectivity;
    use crate::search::root_move::RootMoves;
    use crate::transposition_table::TranspositionTable;

    #[test]
    fn ancestor_cutoff_is_detected_after_epoch_wrap_and_cache_reset() {
        let board = Board::new();
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let root_moves = RootMoves::new(&board);
        let ctx = SearchContext::new(&board, Selectivity::None, &root_moves, &tt, &eval);
        let epoch = Arc::new(Align64(AtomicU64::new(u64::MAX)));
        let worker = Thread::new(
            0,
            Arc::new(AtomicBool::new(true)),
            Arc::new(AbortState::new()),
            epoch.clone(),
            1,
            None,
            Weak::new(),
        );
        worker.searching.store(true, Ordering::Relaxed);
        let request = || SplitRequest {
            board: &board,
            alpha: Score::ZERO,
            beta: Score::ZERO + 1,
            best_score: -Score::INF,
            best_move: None,
            depth: board.empty_count(),
            move_list: MoveList::new(&board),
            move_count: 0,
            node_type: NodeTypeId::NonPv,
            cut_node: true,
        };
        let [parent, child, ..] = &worker.split_points;
        std::mem::forget(worker.initialize_split_point(parent, &ctx, request()));
        std::mem::forget(worker.initialize_split_point(child, &ctx, request()));
        assert!(!worker.split_point_cutoff_occurred());

        let unrelated = SplitPoint::default();
        let publish = |point: &SplitPoint| {
            let _guard = point.lock();
            worker.mark_split_point_cutoff(point.state());
        };
        publish(&unrelated);
        assert_eq!(epoch.load(Ordering::Relaxed), 0);
        assert!(!worker.should_stop());

        publish(parent);
        publish(&SplitPoint::default());
        assert_eq!(epoch.load(Ordering::Relaxed), 2);
        assert!(!child.state().cutoff());
        assert!(worker.split_point_cutoff_occurred());
        publish(parent);
        assert_eq!(epoch.load(Ordering::Relaxed), 2);
        assert!(worker.should_stop());

        worker.reset_cutoff_cache();
        assert!(worker.should_stop());
    }

    #[test]
    fn completed_split_rejects_parent_cutoff_with_epoch_alias() {
        let board = Board::new();
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let roots = RootMoves::new(&board);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
        let epoch = Arc::new(Align64(AtomicU64::new(0)));
        let worker = Arc::new(Thread::new(
            0,
            Arc::new(AtomicBool::new(true)),
            Arc::new(AbortState::new()),
            epoch.clone(),
            1,
            None,
            Weak::new(),
        ));
        worker.searching.store(true, Ordering::Relaxed);
        let request = || {
            let move_list = MoveList::new(&board);
            SplitRequest {
                board: &board,
                alpha: Score::ZERO,
                beta: Score::ZERO + 1,
                best_score: Score::ZERO,
                best_move: None,
                depth: board.empty_count(),
                move_count: move_list.count(),
                move_list,
                node_type: NodeTypeId::NonPv,
                cut_node: true,
            }
        };
        let parent = &worker.split_points[0];
        std::mem::forget(worker.initialize_split_point(parent, &ctx, request()));
        worker.split(&mut ctx, request());
        assert!(Arc::ptr_eq(
            worker.active_split_point().as_ref().unwrap(),
            parent
        ));

        {
            let _guard = parent.lock();
            worker.mark_split_point_cutoff(parent.state());
        }
        epoch.store(u64::MAX, Ordering::Release);
        assert_eq!(
            worker.local_seen_cutoff_epoch.load(Ordering::Relaxed),
            u64::MAX
        );
        assert!(
            worker.should_stop_after_split(),
            "the completed split missed an ancestor cutoff at the cached epoch"
        );
        assert!(worker.should_stop());

        worker.search_active_split_point();
        worker.finalize_split_point(parent);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn nested_dispatch_unwind_restores_parent_and_allows_slot_reuse() {
        use crate::player_color::PlayerColor;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let board = Board::from_string(
            "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX",
            PlayerColor::Black,
        )
        .unwrap();
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let roots = RootMoves::new(&board);
        let mut ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
        let worker = Arc::new(Thread::new(
            0,
            Arc::new(AtomicBool::new(true)),
            Arc::new(AbortState::new()),
            Arc::new(Align64(AtomicU64::new(0))),
            1,
            None,
            Weak::new(),
        ));
        worker.searching.store(true, Ordering::Relaxed);
        fn request(board: &Board, node_type: NodeTypeId) -> SplitRequest<'_> {
            SplitRequest {
                board,
                alpha: Score::ZERO,
                beta: Score::ZERO + 1,
                best_score: -Score::INF,
                best_move: None,
                depth: board.empty_count(),
                move_list: MoveList::new(board),
                move_count: 0,
                node_type,
                cut_node: true,
            }
        }
        let [parent, child, ..] = &worker.split_points;
        std::mem::forget(worker.initialize_split_point(
            parent,
            &ctx,
            request(&board, NodeTypeId::NonPv),
        ));
        let played = MoveList::new(&board)[0];
        let next = board.make_move_with_flipped(played.flipped, played.sq);
        ctx.update(played.sq, played.flipped);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            worker.split(&mut ctx, request(&next, NodeTypeId::Pv))
        }))
        .expect_err("expected the PV split assertion");
        assert!(
            panic
                .downcast_ref::<&str>()
                .is_some_and(|message| message.contains("move_count > 1"))
        );
        assert_eq!(worker.split_points_size.load(Ordering::Relaxed), 1);
        assert!(Arc::ptr_eq(
            worker.active_split_point().as_ref().unwrap(),
            parent
        ));
        assert!(parent.state().helpers_mask.test(worker.idx));
        assert!(child.state().helpers_mask.none());

        let (score, _, counters) = worker.split(&mut ctx, request(&next, NodeTypeId::NonPv));
        assert!(score > -Score::INF);
        assert!(counters.n_nodes > 0);
        assert_eq!(worker.split_points_size.load(Ordering::Relaxed), 1);
        assert!(Arc::ptr_eq(
            worker.active_split_point().as_ref().unwrap(),
            parent
        ));
        assert!(parent.state().helpers_mask.test(worker.idx));
        assert!(child.state().helpers_mask.none());

        worker.search_active_split_point();
        assert!(parent.state().helpers_mask.none());
        worker.finalize_split_point(parent);
        assert!(!worker.has_active_split_point());
    }

    #[cfg(all(debug_assertions, panic = "unwind"))]
    #[test]
    fn panicking_split_helper_cancels_search_and_cannot_be_rebooked() {
        use crate::player_color::PlayerColor;
        use std::time::{Duration, Instant};

        let board = Board::from_string(
            "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX",
            PlayerColor::Black,
        )
        .unwrap();
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let root_moves = RootMoves::new(&board);
        let ctx = SearchContext::new(&board, Selectivity::None, &root_moves, &tt, &eval);
        let pool = ThreadPool::new(2);
        let owner = pool.main();
        let helper = &pool.threads[1];
        owner.searching.store(true, Ordering::Relaxed);
        pool.thinking.store(true, Ordering::Release);

        let request = |node_type| SplitRequest {
            board: &board,
            alpha: Score::ZERO,
            beta: Score::ZERO + 1,
            best_score: -Score::INF,
            best_move: None,
            depth: board.empty_count(),
            move_list: MoveList::new(&board),
            move_count: 0,
            node_type,
            cut_node: true,
        };
        let first = &owner.split_points[0];
        std::mem::forget(owner.initialize_split_point(first, &ctx, request(NodeTypeId::Pv)));
        pool.notify_all();

        let deadline = Instant::now() + Duration::from_secs(10);
        while first.state().all_helpers_searching() || first.state().helpers_mask.test(helper.idx) {
            assert!(
                Instant::now() < deadline,
                "helper did not release its first split"
            );
            thread::yield_now();
        }

        owner.search_active_split_point();
        assert!(first.state().helpers_mask.none());
        owner.finalize_split_point(first);
        assert!(
            owner.should_stop(),
            "split owner could accept the result after the helper lost a move"
        );

        owner.abort_state.begin_search();
        assert!(!owner.should_stop());
        std::mem::forget(owner.initialize_split_point(first, &ctx, request(NodeTypeId::NonPv)));
        assert!(
            !first.state().helpers_mask.test(helper.idx),
            "a new split booked the panicking helper"
        );

        let job: HelperJob = Arc::new(|_| unreachable!());
        pool.start_helpers(&job);
        assert!(!helper.searching.load(Ordering::Acquire));
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn unwind_after_initial_booking_retires_the_split() {
        use crate::player_color::PlayerColor;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let board = Board::from_string(
            "XXXXXXXXXXXXXXXXOOOXXXOXXOXXXXOX-OOXXOOX--OOOXXX--OOXXXX----XXXX",
            PlayerColor::Black,
        )
        .unwrap();
        let tt = TranspositionTable::new(0);
        let eval = Eval::shared_for_tests();
        let roots = RootMoves::new(&board);
        let ctx = SearchContext::new(&board, Selectivity::None, &roots, &tt, &eval);
        let pool = ThreadPool::new(2);
        let owner = pool.main();
        let sp = &owner.split_points[0];
        owner.searching.store(true, Ordering::Relaxed);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _activation = owner.initialize_split_point(
                sp,
                &ctx,
                SplitRequest {
                    board: &board,
                    alpha: Score::ZERO,
                    beta: Score::ZERO + 1,
                    best_score: -Score::INF,
                    best_move: None,
                    depth: board.empty_count(),
                    move_list: MoveList::new(&board),
                    move_count: 0,
                    node_type: NodeTypeId::NonPv,
                    cut_node: true,
                },
            );
            let booked = sp.state().helpers_mask.test(pool.threads[1].idx);
            pool.thinking.store(true, Ordering::Release);
            pool.notify_all();
            assert!(booked, "fixture did not book the sleeping helper");
            panic!("after initial booking");
        }))
        .unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"after initial booking"));
        assert_eq!(owner.split_points_size.load(Ordering::Relaxed), 0);
        assert!(sp.state().helpers_mask.none());
        assert!(!owner.has_active_split_point());
    }

    #[cfg(panic = "unwind")]
    #[test]
    fn panicking_helper_job_clears_assignment_and_stops_accepting_work() {
        let worker = Arc::new(Thread::new(
            1,
            Arc::new(AtomicBool::new(true)),
            Arc::new(AbortState::new()),
            Arc::new(Align64(AtomicU64::new(0))),
            2,
            None,
            Weak::new(),
        ));
        worker.book_helper_job(Arc::new(|_| panic!("helper job fixture")));
        let handle = {
            let worker = worker.clone();
            thread::spawn(move || worker.idle_loop())
        };
        let panic = handle.join().unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"helper job fixture"));
        assert!(
            !worker.searching.load(Ordering::Acquire),
            "exited helper job remained assigned"
        );
        worker.book_helper_job(Arc::new(|_| unreachable!()));
        assert!(!worker.searching.load(Ordering::Acquire));
    }
}
