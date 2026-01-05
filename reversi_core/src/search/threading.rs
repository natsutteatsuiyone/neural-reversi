//! Reference: https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/thread.cpp

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{JoinHandle, sleep};
use std::time::{Duration, Instant};

use lock_api::RawMutex;

use crate::board::Board;
use crate::constants::MAX_PLY;
use crate::empty_list::EmptyList;
use crate::eval::Eval;
use crate::move_list::ConcurrentMoveIterator;
use crate::search::node_type::{NodeType, NonPV, PV, Root};
use crate::search::root_move::RootMove;
use crate::search::search_context::{GamePhase, SearchContext};
use crate::search::search_result::SearchResult;
use crate::search::side_to_move::SideToMove;
use crate::search::{self, SearchTask, endgame, midgame, time_control::TimeManager};
use crate::square::Square;
use crate::transposition_table::TranspositionTable;
use crate::types::{Depth, ScaledScore, Selectivity};
use crate::util::bitset::AtomicBitSet;
use crate::util::spinlock;

/// Maximum number of split points that a single thread can have active at once.
const MAX_SPLITPOINTS_PER_THREAD: usize = 8;

/// Maximum number of helper threads that can join a single split point.
const MAX_HELPERS_PER_SPLITPOINT: u32 = 5;

/// Interval between checks for abort flag in milliseconds.
const CHECK_INTERVAL_MS: u64 = 1;

/// State information for a split point in the parallel search.
pub struct SplitPointState {
    /// Flag indicating if all helper threads are actively searching.
    all_helpers_searching: AtomicBool,

    /// Alpha bound for the alpha-beta search at this node.
    alpha: AtomicI32,

    /// Beta bound for the alpha-beta search at this node.
    pub beta: ScaledScore,

    /// Best score found so far at this split point.
    best_score: AtomicI32,

    /// Best move found so far at this split point.
    best_move: AtomicU8,

    /// Iterator over moves to be searched, shared among threads.
    pub move_iter: Option<Arc<ConcurrentMoveIterator>>,

    /// Type of node (PV, NonPV, or Root) for search specialization.
    node_type: u32,

    /// Flag indicating if a beta cutoff has occurred.
    cutoff: AtomicBool,

    /// Index of the owner thread that created this split point.
    owner_thread_idx: usize,

    /// Bitmask tracking which threads are working on this split point.
    helpers_mask: AtomicBitSet,

    /// Search depth remaining from this position.
    depth: Depth,

    /// Total number of nodes searched by all threads at this split point.
    n_nodes: AtomicU64,

    /// Task data containing the position and search context.
    pub task: Option<SplitPointTask>,

    /// Parent split point in the tree hierarchy.
    parent_split_point: Option<Arc<SplitPoint>>,

    /// Principal variation line from the best move found at this split point.
    pv: [Square; MAX_PLY],
}

impl SplitPointState {
    /// Gets the current alpha value atomically.
    #[inline]
    pub fn alpha(&self) -> ScaledScore {
        ScaledScore::new(self.alpha.load(Ordering::Relaxed))
    }

    /// Sets the alpha value atomically.
    #[inline]
    pub fn set_alpha(&self, value: ScaledScore) {
        self.alpha.store(value.value(), Ordering::Relaxed);
    }

    /// Gets the all_helpers_searching flag.
    #[inline]
    pub fn all_helpers_searching(&self) -> bool {
        self.all_helpers_searching.load(Ordering::Relaxed)
    }

    /// Sets the all_helpers_searching flag.
    #[inline]
    pub fn set_all_helpers_searching(&self, value: bool) {
        self.all_helpers_searching.store(value, Ordering::Relaxed);
    }

    /// Gets the cutoff flag.
    #[inline]
    pub fn cutoff(&self) -> bool {
        self.cutoff.load(Ordering::Relaxed)
    }

    /// Sets the cutoff flag.
    #[inline]
    pub fn set_cutoff(&self, value: bool) {
        self.cutoff.store(value, Ordering::Relaxed);
    }

    /// Gets the best score.
    #[inline]
    pub fn best_score(&self) -> ScaledScore {
        ScaledScore::new(self.best_score.load(Ordering::Relaxed))
    }

    /// Sets the best score.
    #[inline]
    pub fn set_best_score(&self, value: ScaledScore) {
        self.best_score.store(value.value(), Ordering::Relaxed);
    }

    /// Gets the best move.
    #[inline]
    pub fn best_move(&self) -> Square {
        Square::from_u8_unchecked(self.best_move.load(Ordering::Relaxed))
    }

    /// Sets the best move.
    #[inline]
    pub fn set_best_move(&self, value: Square) {
        self.best_move.store(value as u8, Ordering::Relaxed);
    }

    /// Gets the node count.
    #[inline]
    pub fn n_nodes(&self) -> u64 {
        self.n_nodes.load(Ordering::Relaxed)
    }

    /// Adds to the node count.
    #[inline]
    pub fn add_nodes(&self, count: u64) {
        self.n_nodes.fetch_add(count, Ordering::Relaxed);
    }

    /// Copies PV from source to the split point's internal PV storage.
    #[inline]
    pub fn copy_pv(&mut self, src: &[Square; MAX_PLY]) {
        self.pv.copy_from_slice(src);
    }

    /// Gets a reference to the internal PV.
    #[inline]
    pub fn pv(&self) -> &[Square; MAX_PLY] {
        &self.pv
    }
}

/// Task data for a split point containing all information needed for search.
pub struct SplitPointTask {
    /// Current board position to search from.
    pub board: Board,

    /// Which player is to move in this position.
    pub side_to_move: SideToMove,

    /// Search selectivity level (affects pruning aggressiveness).
    pub selectivity: Selectivity,

    /// Current game phase (opening, midgame, endgame).
    pub game_phase: GamePhase,

    /// Shared transposition table for storing search results.
    pub tt: Arc<TranspositionTable>,

    /// List of moves being searched at the root (for root node only).
    pub root_moves: Arc<Mutex<Vec<RootMove>>>,

    /// Current PV index for Multi-PV search.
    pub pv_idx: Arc<AtomicUsize>,

    /// Neural network evaluator for position evaluation.
    pub eval: Arc<Eval>,

    /// List of empty squares for move generation optimization.
    pub empty_list: EmptyList,
}

/// A split point in the parallel search tree.
pub struct SplitPoint {
    /// Spinlock for fast synchronization between threads.
    mutex: spinlock::RawSpinLock,

    /// Mutable state protected by the mutex.
    state: UnsafeCell<SplitPointState>,
}

unsafe impl Sync for SplitPoint {}

impl Default for SplitPoint {
    /// Create a new split point with default values.
    fn default() -> Self {
        SplitPoint {
            mutex: spinlock::RawSpinLock::INIT,
            state: UnsafeCell::new(SplitPointState {
                all_helpers_searching: AtomicBool::new(false),
                alpha: AtomicI32::new(0),
                beta: ScaledScore::new(0),
                best_score: AtomicI32::new(0),
                best_move: std::sync::atomic::AtomicU8::new(Square::None as u8),
                move_iter: None,
                node_type: 0,
                cutoff: AtomicBool::new(false),
                owner_thread_idx: 0,
                helpers_mask: AtomicBitSet::new(),
                depth: 0,
                n_nodes: AtomicU64::new(0),
                task: None,
                parent_split_point: None,
                pv: [Square::None; MAX_PLY],
            }),
        }
    }
}

impl SplitPoint {
    /// Get an immutable reference to the split point state.
    #[inline]
    pub fn state(&self) -> &SplitPointState {
        unsafe { &*self.state.get() }
    }

    /// Get a mutable reference to the split point state.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn state_mut(&self) -> &mut SplitPointState {
        unsafe { &mut *self.state.get() }
    }

    /// Acquire the split point's lock.
    #[inline]
    pub fn lock(&self) {
        self.mutex.lock();
    }

    /// Release the split point's lock.
    #[inline]
    pub fn unlock(&self) {
        unsafe { self.mutex.unlock() };
    }
}

/// State information for a worker thread.
pub struct ThreadState {
    /// The split point this thread is currently working on.
    pub active_split_point: Option<Arc<SplitPoint>>,

    /// Number of split points in the split_points array.
    pub split_points_size: usize,

    /// Stack of split points created by this thread.
    split_points: [Arc<SplitPoint>; MAX_SPLITPOINTS_PER_THREAD],
}

/// A worker thread in the thread pool.
pub struct Thread {
    /// Mutex used with the condition variable for thread sleeping.
    mutex_for_sleep_condition: std::sync::Mutex<()>,

    /// Spinlock protecting the thread state.
    mutex_for_state: spinlock::RawSpinLock,

    /// Condition variable for waking up idle threads.
    sleep_condition: std::sync::Condvar,

    /// Unique index of this thread in the pool.
    idx: usize,

    /// Weak reference to the thread pool this thread belongs to.
    pool: Weak<ThreadPool>,

    /// Shared flag indicating if the engine is thinking.
    thinking: Arc<AtomicBool>,

    /// Mutable thread state protected by mutex_for_state.
    state: UnsafeCell<ThreadState>,

    /// Flag indicating if the thread has completed initialization.
    ready: AtomicBool,

    /// Flag indicating if this thread is currently searching.
    searching: AtomicBool,

    /// Flag signaling the thread to exit.
    exit: AtomicBool,
}

unsafe impl Sync for Thread {}

impl Thread {
    /// Create a new thread with the given index.
    ///
    /// # Arguments
    ///
    /// * `idx` - Unique index for this thread
    /// * `thinking` - Shared flag indicating if the engine is thinking
    /// * `pool` - Weak reference to the thread pool this thread belongs to
    ///
    /// # Returns
    ///
    /// A new thread in the initial idle state
    fn new(idx: usize, thinking: Arc<AtomicBool>, pool: Weak<ThreadPool>) -> Thread {
        let split_points = std::array::from_fn(|_| Arc::new(SplitPoint::default()));

        Thread {
            mutex_for_sleep_condition: std::sync::Mutex::new(()),
            mutex_for_state: spinlock::RawSpinLock::INIT,
            sleep_condition: std::sync::Condvar::new(),
            idx,
            pool,
            thinking,
            state: UnsafeCell::new(ThreadState {
                active_split_point: None,
                split_points_size: 0,
                split_points,
            }),
            ready: AtomicBool::new(false),
            searching: AtomicBool::new(false),
            exit: AtomicBool::new(false),
        }
    }

    /// Acquire the thread's state lock.
    pub fn lock(&self) {
        self.mutex_for_state.lock();
    }

    /// Release the thread's state lock.
    pub fn unlock(&self) {
        unsafe { self.mutex_for_state.unlock() };
    }

    /// Check if this thread can create a new split point.
    ///
    /// A thread can split if:
    /// 1. Multiple threads are available (pool size > 1)
    /// 2. The thread hasn't reached its split point limit
    /// 3. Either:
    ///    - The thread has no active split point, OR
    ///    - Not all helpers are searching (room for more), OR
    ///    - We can steal helpers from the current split point
    ///
    /// # Returns
    ///
    /// `true` if the thread can create a new split point
    pub fn can_split(&self) -> bool {
        let thread_pool_size = self.pool.upgrade().map_or(1, |p| p.size) as u32;
        if thread_pool_size <= 1 {
            return false;
        }

        let th_state = self.state();

        let cond = if let Some(sp) = &th_state.active_split_point {
            let sp_state = sp.state();
            !sp_state.all_helpers_searching()
                || thread_pool_size > MAX_HELPERS_PER_SPLITPOINT
                    && sp_state.helpers_mask.count() == MAX_HELPERS_PER_SPLITPOINT
        } else {
            true
        };

        cond && (th_state.split_points_size < MAX_SPLITPOINTS_PER_THREAD)
    }

    /// Get an immutable reference to the thread state.
    #[inline]
    pub fn state(&self) -> &ThreadState {
        unsafe { &*self.state.get() }
    }

    /// Get a mutable reference to the thread state.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn state_mut(&self) -> &mut ThreadState {
        unsafe { &mut *self.state.get() }
    }

    /// Wake up this thread when there is work to do.
    fn notify_one(&self) {
        let _lock = self.mutex_for_sleep_condition.lock();
        self.sleep_condition.notify_one();
    }

    /// Check if a beta cutoff has occurred in the current or ancestor split points.
    ///
    /// # Returns
    ///
    /// `true` if a beta cutoff has occurred in this thread's split point hierarchy
    pub fn cutoff_occurred(&self) -> bool {
        let mut current_sp = self.state().active_split_point.as_ref();
        while let Some(sp) = current_sp {
            let sp_state = sp.state();
            if sp_state.cutoff() {
                return true;
            }
            current_sp = sp_state.parent_split_point.as_ref();
        }
        false
    }

    /// Check if this thread can join the given split point.
    ///
    /// A thread can join a split point if:
    /// 1. The thread is not currently searching (is idle)
    /// 2. For the "helpful owner" concept: if the thread is an owner of other
    ///    split points, it can only join split points created by its helpers
    ///
    /// # Arguments
    ///
    /// * `sp` - The split point to potentially join
    ///
    /// # Returns
    ///
    /// `true` if the thread can safely join the split point
    fn can_join(&self, sp: &Arc<SplitPoint>) -> bool {
        if self.searching.load(Ordering::Acquire) {
            return false;
        }

        // Make a local copy to be sure it doesn't become zero under our feet while
        // testing next condition and so leading to an out of bounds access.
        let th_state = self.state();
        let size = th_state.split_points_size;

        // No split points means that the thread is available as a helper for any
        // other thread otherwise apply the "helpful owner" concept if possible.
        if size == 0 {
            return true;
        }

        let sp_state = th_state.split_points[size - 1].state();
        let owner_idx = sp.state().owner_thread_idx;
        sp_state.helpers_mask.test(owner_idx)
    }

    /// Create a split point and distribute work among available threads.
    ///
    /// This is the main entry point for parallel search. When a thread has multiple
    /// moves to search at a node, it can call this method to get help from other
    /// idle threads. The method:
    ///
    /// 1. Creates a new split point with the current search parameters
    /// 2. Finds idle threads and assigns them to help search
    /// 3. The calling thread also participates in the search (helpful owner)
    /// 4. Waits for all threads to complete their work
    /// 5. Returns the best move and score found
    ///
    /// # Arguments
    ///
    /// * `ctx` - Current search context
    /// * `board` - Current board position
    /// * `alpha` - Alpha bound for alpha-beta search
    /// * `beta` - Beta bound for alpha-beta search
    /// * `best_score` - Best score found so far
    /// * `best_move` - Best move found so far
    /// * `depth` - Remaining search depth
    /// * `move_iter` - Iterator over moves to search
    /// * `node_type` - Type of search node (PV, NonPV, Root)
    ///
    /// # Returns
    ///
    /// Tuple of (best_score, best_move, nodes_searched)
    #[allow(clippy::too_many_arguments)]
    pub fn split(
        self: &Arc<Self>,
        ctx: &mut SearchContext,
        board: &Board,
        alpha: ScaledScore,
        beta: ScaledScore,
        best_score: ScaledScore,
        best_move: Square,
        depth: Depth,
        move_iter: &Arc<ConcurrentMoveIterator>,
        node_type: u32,
    ) -> (ScaledScore, Square, u64) {
        let th_state = self.state();
        // Pick the next available split point
        let sp = &th_state.split_points[th_state.split_points_size];

        // Initialize the split point with search parameters
        self.initialize_split_point(
            sp, ctx, depth, best_score, best_move, alpha, beta, node_type, move_iter, board,
        );

        // Enter idle loop as owner thread - will return when all helpers finish
        self.idle_loop();

        // Clean up the split point
        self.finalize_split_point(sp);

        // Extract results - split point data is now immutable
        let sp_state = sp.state();

        // Copy PV from split point back to coordinator's stack
        ctx.set_pv(sp_state.pv());

        (
            sp_state.best_score(),
            sp_state.best_move(),
            sp_state.n_nodes(),
        )
    }

    /// Initialize a split point with search parameters and find workers.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn initialize_split_point(
        &self,
        sp: &Arc<SplitPoint>,
        ctx: &SearchContext,
        depth: Depth,
        best_score: ScaledScore,
        best_move: Square,
        alpha: ScaledScore,
        beta: ScaledScore,
        node_type: u32,
        move_iter: &Arc<ConcurrentMoveIterator>,
        board: &Board,
    ) {
        let th_state = self.state_mut();
        debug_assert!(self.searching.load(Ordering::Acquire));
        debug_assert!(th_state.split_points_size < MAX_SPLITPOINTS_PER_THREAD);

        sp.lock();
        // No contention here until we don't increment splitPointsSize
        let sp_state = sp.state_mut();
        sp_state.owner_thread_idx = self.idx;
        sp_state.parent_split_point = th_state.active_split_point.clone();
        // Initialize split point state
        sp_state.helpers_mask.clear();
        sp_state.helpers_mask.set(self.idx);
        sp_state.depth = depth;
        sp_state.set_best_score(best_score);
        sp_state.set_best_move(best_move);
        sp_state.set_alpha(alpha);
        sp_state.beta = beta;
        sp_state.node_type = node_type;
        sp_state.move_iter = Some(move_iter.clone());
        sp_state.task = Some(SplitPointTask {
            board: *board,
            side_to_move: ctx.side_to_move,
            selectivity: ctx.selectivity,
            tt: ctx.tt.clone(),
            root_moves: ctx.root_moves.clone(),
            pv_idx: ctx.pv_idx.clone(),
            eval: ctx.eval.clone(),
            game_phase: ctx.game_phase,
            empty_list: ctx.empty_list.clone(),
        });
        sp_state.n_nodes.store(0, Ordering::Relaxed);
        sp_state.set_cutoff(false);
        sp_state.set_all_helpers_searching(true); // Must be set under lock protection
        sp_state.copy_pv(ctx.get_pv()); // Initialize PV from coordinator's current PV

        th_state.split_points_size += 1;
        th_state.active_split_point = Some(sp.clone());

        // Try to allocate available threads
        self.pool
            .upgrade()
            .unwrap()
            .assign_helpers_to_split_point(sp);

        // Everything is set up. The owner thread enters the idle loop, from which
        // it will instantly launch a search, because its 'searching' flag is set.
        // The thread will return from the idle loop when all helpers have finished
        // their work at this split point.
        sp.unlock();
    }

    /// Clean up after all threads have finished working on a split point.
    ///
    /// # Arguments
    ///
    /// * `sp` - The split point that has completed
    #[inline]
    fn finalize_split_point(&self, sp: &Arc<SplitPoint>) {
        debug_assert!(!self.searching.load(Ordering::Acquire));

        // In the helpful owner concept, an owner can help only a sub-tree of its
        // split point and because everything is finished here, it's not possible
        // for the owner to be booked.
        self.lock();

        // We have returned from the idle loop, which means that all threads are
        // finished. Note that decreasing splitPointsSize must be done under lock
        // protection to avoid a race with Thread::can_join().
        self.searching.store(true, Ordering::Release);
        let th_state = self.state_mut();
        th_state.split_points_size -= 1;
        th_state.active_split_point = sp.state().parent_split_point.clone();

        self.unlock();

        // Clear task data after releasing thread lock to minimize lock duration
        sp.state_mut().task = None;
    }

    /// Main loop for worker threads.
    ///
    /// This method implements the core logic for thread synchronization:
    ///
    /// 1. **Owner Mode**: If called from split(), acts as the owner thread
    ///    and waits for all helpers to finish before returning
    ///
    /// 2. **Helper Mode**: If called at thread creation, waits for work assignments
    ///    and executes search tasks when assigned to split points
    fn idle_loop(self: &Arc<Self>) {
        // Pointer 'this_sp' is not null only if we are called from split(), and not
        // at the thread creation. This means we are the split point's owner.
        let this_sp = self.state().active_split_point.clone();

        // Main loop - continues until thread exit is signaled
        while !self.exit.load(Ordering::Acquire) {
            // Check if we're the owner of a split point and all helpers have finished
            if let Some(ref sp) = this_sp
                && sp.state().helpers_mask.none()
            {
                break;
            }

            // If this thread has been assigned work, launch a search
            // This inner loop handles the actual search work
            while self.searching.load(Ordering::Acquire) {
                self.lock();
                let sp = self.state().active_split_point.clone().unwrap();
                self.unlock();

                // Extract search parameters from split point
                let (board, depth, node_type) = {
                    sp.lock();
                    let task = sp.state().task.as_ref().unwrap();
                    let sp_state = sp.state();
                    (task.board, sp_state.depth, sp_state.node_type)
                };

                let mut ctx = SearchContext::from_split_point(&sp);
                self.dispatch_search(&mut ctx, &board, depth, node_type, &sp);

                self.lock();
                self.searching.store(false, Ordering::Release);
                self.unlock();

                // Update split point state
                let sp_state = sp.state_mut();
                sp_state.helpers_mask.reset(self.idx);
                sp_state.set_all_helpers_searching(false);
                sp_state.add_nodes(ctx.n_nodes);

                // After releasing the lock we can't access any SplitPoint related data
                // in a safe way because it could have been released under our feet by
                // the sp owner.
                sp.unlock();

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
                // Wait for a new job or for our helpers to finish
                std::thread::yield_now();
            }
        }
    }

    /// Main thread message processing loop.
    ///
    /// This is the entry point for the main thread (thread 0), which handles
    /// control messages from the thread pool. Unlike worker threads which
    /// participate in split-point searches, the main thread:
    ///
    /// 1. Receives search tasks via the message channel
    /// 2. Coordinates the search by waking worker threads
    /// 3. Executes the root search
    /// 4. Sends results back via the result channel
    /// 5. Handles shutdown requests
    fn main_thread_loop(self: Arc<Self>, receiver: Arc<std::sync::Mutex<Receiver<Message>>>) {
        loop {
            // Check exit flag before blocking on receive
            if self.exit.load(Ordering::Acquire) {
                break;
            }

            // Block waiting for next message
            let message = {
                let receiver_guard = receiver.lock().unwrap();
                receiver_guard.recv()
            };

            match message {
                Ok(Message::StartThinking(task, thread, result_sender)) => {
                    // Mark thread as actively searching
                    thread.searching.store(true, Ordering::Release);

                    // Wake up worker threads to participate in parallel search
                    task.pool.notify_all();

                    // Execute root search - this is where the main work happens
                    let result = search::search_root(task, &thread);

                    // Search complete - update state and send result
                    thread.searching.store(false, Ordering::Release);

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

    /// Dispatch to the appropriate search function based on game phase and node type.
    ///
    /// # Arguments
    ///
    /// * `self` - A reference to the thread.
    /// * `ctx` - A mutable reference to the search context.
    /// * `board` - A reference to the board.
    /// * `depth` - The search depth.
    /// * `node_type` - The node type.
    /// * `sp` - A reference to the split point.
    fn dispatch_search(
        self: &Arc<Self>,
        ctx: &mut SearchContext,
        board: &Board,
        depth: Depth,
        node_type: u32,
        sp: &Arc<SplitPoint>,
    ) {
        let is_endgame = ctx.game_phase == GamePhase::EndGame && ctx.empty_list.count == depth;

        match (is_endgame, node_type) {
            // Endgame searches
            (true, NonPV::TYPE_ID) => {
                endgame::search_sp::<NonPV>(ctx, board, self, sp);
            }
            (true, PV::TYPE_ID) => {
                endgame::search_sp::<PV>(ctx, board, self, sp);
            }
            (true, Root::TYPE_ID) => {
                endgame::search_sp::<Root>(ctx, board, self, sp);
            }
            // Midgame searches
            (false, NonPV::TYPE_ID) => {
                midgame::search_sp::<NonPV>(ctx, board, depth, self, sp);
            }
            (false, PV::TYPE_ID) => {
                midgame::search_sp::<PV>(ctx, board, depth, self, sp);
            }
            (false, Root::TYPE_ID) => {
                midgame::search_sp::<Root>(ctx, board, depth, self, sp);
            }
            _ => unreachable!("Invalid node type: {}", node_type),
        }
    }

    /// Try to join an existing split point after finishing current work.
    ///
    /// When a thread finishes its work, it can try to help other threads
    /// by joining their split points. This method finds the best available
    /// split point to join based on:
    ///
    /// 1. The split point must have room for more helpers
    /// 2. All current helpers must still be searching
    /// 3. The thread must be able to join (helpful owner rules)
    /// 4. Prefer split points higher in the tree (lower level)
    fn try_late_join(&self) {
        let mut best_sp = None;
        let mut min_level = i32::MAX;
        let pool = self.pool.upgrade().unwrap();
        for th in &pool.threads {
            let size = th.state().split_points_size;
            if size == 0 {
                continue;
            }

            let sp = &th.state().split_points[size - 1];
            let sp_state = sp.state();
            if sp_state.all_helpers_searching()
                && sp_state.helpers_mask.count() < MAX_HELPERS_PER_SPLITPOINT
                && self.can_join(sp)
            {
                let mut level = 0;
                let mut active_sp = &th.state().active_split_point;
                while let Some(p) = active_sp {
                    level += 1;
                    active_sp = &p.state().parent_split_point;
                }

                if level < min_level {
                    min_level = level;
                    best_sp = Some(sp);
                }
            }
        }

        if let Some(sp) = best_sp {
            // Recheck the conditions under lock protection
            sp.lock();

            let sp_state = sp.state_mut();
            if sp_state.all_helpers_searching()
                && sp_state.helpers_mask.count() < MAX_HELPERS_PER_SPLITPOINT
            {
                self.lock();

                if self.can_join(sp) {
                    sp_state.helpers_mask.set(self.idx);
                    let th_state = self.state_mut();
                    th_state.active_split_point = Some(sp.clone());
                    self.searching.store(true, Ordering::Release);
                }

                self.unlock();
            }

            sp.unlock();
        }
    }

    pub fn is_search_aborted(&self) -> bool {
        self.pool.upgrade().is_some_and(|pool| pool.is_aborted())
    }
}

/// Messages that can be sent to the main thread.
enum Message {
    /// Start a new search with the given task and return results via the sender.
    StartThinking(SearchTask, Arc<Thread>, Sender<SearchResult>),

    /// Signal the thread to exit.
    Exit,
}

/// Thread pool for parallel game tree search.
pub struct ThreadPool {
    /// Collection of all threads in the pool.
    threads: Vec<Arc<Thread>>,

    /// Join handles for thread cleanup on shutdown.
    thread_handles: Vec<JoinHandle<()>>,

    /// Number of threads in the pool.
    pub size: usize,

    /// Global flag indicating if the engine is thinking.
    thinking: Arc<AtomicBool>,

    /// Channel sender for sending messages to the main thread.
    sender: Arc<Sender<Message>>,

    /// Channel receiver for the main thread (protected by mutex).
    receiver: Arc<std::sync::Mutex<Receiver<Message>>>,

    /// Flag for aborting the current search.
    abort_flag: Arc<AtomicBool>,

    /// Handle for the timer thread (protected by mutex for interior mutability).
    timer_handle: Mutex<Option<JoinHandle<()>>>,

    /// Flag to signal the timer thread to stop.
    timer_stop: Arc<AtomicBool>,
}

impl ThreadPool {
    /// Create a new thread pool with the specified number of threads.
    ///
    /// This only creates the ThreadPool structure. Call init() to actually
    /// create and start the threads.
    ///
    /// # Arguments
    ///
    /// * `n_threads` - Number of threads to create (must be at least 1)
    ///
    /// # Returns
    ///
    /// A new uninitialized thread pool
    pub fn new(n_threads: usize) -> Arc<ThreadPool> {
        Arc::new_cyclic(|weak| {
            let (sender, receiver) = std::sync::mpsc::channel();

            let mut pool = ThreadPool {
                threads: Vec::new(),
                thread_handles: Vec::new(),
                size: n_threads,
                thinking: Arc::new(AtomicBool::new(false)),
                sender: Arc::new(sender),
                receiver: Arc::new(Mutex::new(receiver)),
                abort_flag: Arc::new(AtomicBool::new(false)),
                timer_handle: Mutex::new(None),
                timer_stop: Arc::new(AtomicBool::new(false)),
            };

            pool.init(weak);
            pool
        })
    }

    /// Initialize the thread pool by creating and starting all threads.
    pub fn init(&mut self, pool: &std::sync::Weak<ThreadPool>) {
        self.create_main_thread(pool);
        self.create_worker_threads(pool);
        self.wait_for_threads_ready();
    }

    /// Create and start the main thread that handles control messages.
    fn create_main_thread(&mut self, pool: &std::sync::Weak<ThreadPool>) {
        let main_thread = Arc::new(Thread::new(0, self.thinking.clone(), pool.clone()));
        let main_thread_clone = main_thread.clone();
        let receiver_clone = self.receiver.clone();

        let handle = std::thread::spawn(move || main_thread_clone.main_thread_loop(receiver_clone));

        self.threads.push(main_thread);
        self.thread_handles.push(handle);
    }

    /// Create and start worker threads that wait in idle loops.
    fn create_worker_threads(&mut self, pool: &std::sync::Weak<ThreadPool>) {
        for i in 1..self.size {
            let thread = Arc::new(Thread::new(i, self.thinking.clone(), pool.clone()));
            let thread_clone = thread.clone();

            let handle = std::thread::spawn(move || thread_clone.idle_loop());

            self.threads.push(thread);
            self.thread_handles.push(handle);
        }
    }

    /// Wait for all threads to signal they are ready.
    fn wait_for_threads_ready(&self) {
        self.main().ready.store(true, Ordering::Release);

        while !self.all_threads_ready() {
            sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Check if all threads have signaled they are ready.
    fn all_threads_ready(&self) -> bool {
        self.threads
            .iter()
            .all(|thread| thread.ready.load(Ordering::Relaxed))
    }

    /// Shut down the thread pool and wait for all threads to exit.
    fn exit(&mut self) {
        // Already shut down - nothing to do
        if self.threads.is_empty() {
            return;
        }

        // Stop timer thread first to prevent it from setting abort flags
        self.stop_timer();

        // Signal all worker threads to exit and wake them up
        for thread in &self.threads {
            let _lock = thread.mutex_for_sleep_condition.lock();
            thread.exit.store(true, Ordering::Release);
        }

        // Wake up all sleeping threads so they can observe the exit flag
        for thread in &self.threads {
            thread.notify_one();
        }

        // Send exit message to main thread's message loop
        // Ignore send error if receiver is already dropped
        let _ = self.sender.send(Message::Exit);

        // Join all threads, collecting any panic information
        for (_idx, thread_handle) in self.thread_handles.drain(..).enumerate() {
            if let Err(_panic_info) = thread_handle.join() {
                #[cfg(debug_assertions)]
                eprintln!("Warning: Thread {} panicked during shutdown", _idx);
            }
        }

        // Clear thread references
        self.threads.clear();
    }

    /// Assign idle threads to work on a split point.
    ///
    /// # Arguments
    ///
    /// * `sp` - The split point that needs workers
    fn assign_helpers_to_split_point(&self, sp: &Arc<SplitPoint>) {
        let sp_state = sp.state_mut();
        while sp_state.helpers_mask.count() < MAX_HELPERS_PER_SPLITPOINT {
            if let Some(helper) = self.find_available_thread(sp) {
                helper.lock();

                if helper.can_join(sp) {
                    sp_state.helpers_mask.set(helper.idx);
                    let helper_state = helper.state_mut();
                    helper_state.active_split_point = Some(sp.clone());
                    helper.searching.store(true, Ordering::Release);
                }
                helper.unlock();
            } else {
                break;
            }
        }
    }

    /// Find an available thread that can join the given split point.
    ///
    /// # Arguments
    ///
    /// * `sp` - The split point to find a worker for
    ///
    /// # Returns
    ///
    /// The first available thread that can join, or None if none available
    fn find_available_thread(&self, sp: &Arc<SplitPoint>) -> Option<Arc<Thread>> {
        self.threads
            .iter()
            .find(|thread| thread.can_join(sp))
            .cloned()
    }

    /// Start a new search task on the thread pool.
    ///
    /// # Arguments
    ///
    /// * `task` - The search task containing position, depth, time control, etc.
    ///
    /// # Returns
    ///
    /// A receiver channel that will receive the search result
    pub fn start_thinking(&self, task: SearchTask) -> std::sync::mpsc::Receiver<SearchResult> {
        debug_assert!(
            !self.threads.is_empty(),
            "Cannot start thinking: thread pool has been shut down"
        );

        let (result_sender, result_receiver) = std::sync::mpsc::channel();

        // Ensure clean state before starting new search
        self.reset_abort_flag();

        // Mark pool as actively thinking before sending message
        self.thinking.store(true, Ordering::Release);

        // Dispatch task to main thread
        self.sender
            .send(Message::StartThinking(
                task,
                self.main().clone(),
                result_sender,
            ))
            .expect("Failed to send search task: main thread receiver disconnected");

        result_receiver
    }

    /// Get a reference to the main thread (thread 0).
    ///
    /// # Returns
    ///
    /// Reference to the main thread
    pub fn main(&self) -> &Arc<Thread> {
        &self.threads[0]
    }

    /// Wake up all threads in the pool.
    fn notify_all(&self) {
        for thread in &self.threads {
            thread.notify_one();
        }
    }

    /// Wait for the current search to complete.
    pub fn wait_for_think_finished(&self) {
        const POLL_INTERVAL: Duration = Duration::from_millis(5);

        while self.thinking.load(Ordering::Acquire) {
            sleep(POLL_INTERVAL);
        }
    }

    /// Signal all threads to abort the current search.
    pub fn abort_search(&self) {
        self.abort_flag.store(true, Ordering::Release);
    }

    /// Reset the abort flag for a new search.
    fn reset_abort_flag(&self) {
        self.abort_flag.store(false, Ordering::Release);
    }

    /// Check if the current search has been aborted.
    ///
    /// # Returns
    ///
    /// `true` if the search should be aborted
    #[inline]
    pub fn is_aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Acquire)
    }

    /// Get a clone of the abort flag for external use (e.g., time management).
    ///
    /// # Returns
    ///
    /// Arc reference to the abort flag
    pub fn get_abort_flag(&self) -> Arc<AtomicBool> {
        self.abort_flag.clone()
    }

    /// Starts a timer thread that will set abort_flag when deadline is reached.
    ///
    /// - Checks every `CHECK_INTERVAL_MS` milliseconds against the current deadline
    /// - Responds to dynamic time extensions from the `TimeManager`
    /// - Exits cleanly when:
    ///   - Deadline is reached (sets abort_flag)
    ///   - `stop_timer()` is called (search completed early)
    ///   - No deadline is set (infinite time mode)
    ///
    /// # Arguments
    ///
    /// * `time_manager` - Shared time manager for computing the deadline
    pub fn start_timer(&self, time_manager: Arc<TimeManager>) {
        // Reset stop flag before spawning new timer
        self.timer_stop.store(false, Ordering::Release);

        let abort_flag = self.abort_flag.clone();
        let stop_flag = self.timer_stop.clone();

        let handle = std::thread::Builder::new()
            .name("search-timer".to_string())
            .spawn(move || {
                Self::timer_loop(&time_manager, &abort_flag, &stop_flag);
            })
            .expect("Failed to spawn timer thread");

        *self.timer_handle.lock().unwrap() = Some(handle);
    }

    /// Timer thread loop.
    fn timer_loop(time_manager: &TimeManager, abort_flag: &AtomicBool, stop_flag: &AtomicBool) {
        const CHECK_INTERVAL: Duration = Duration::from_millis(CHECK_INTERVAL_MS);

        loop {
            // Check if search completed early
            if stop_flag.load(Ordering::Acquire) {
                return;
            }

            // Recompute deadline to honor potential time extensions
            match time_manager.deadline() {
                Some(deadline) if Instant::now() >= deadline => {
                    // Time's up - signal abort and exit
                    abort_flag.store(true, Ordering::Release);
                    return;
                }
                Some(_) => {
                    // Still have time - continue monitoring
                }
                None => {
                    // No deadline (infinite mode) - timer not needed
                    return;
                }
            }

            std::thread::sleep(CHECK_INTERVAL);
        }
    }

    /// Stops the timer thread if running.
    ///
    /// This should be called when search completes (before waiting for result)
    /// to ensure clean shutdown.
    pub fn stop_timer(&self) {
        // Signal timer to stop
        self.timer_stop.store(true, Ordering::Release);

        // Join the timer thread if it exists
        if let Some(handle) = self.timer_handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.exit();
    }
}
