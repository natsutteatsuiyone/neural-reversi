//! https://github.com/official-stockfish/Stockfish/blob/5b555525d2f9cbff446b7461d1317948e8e21cd1/src/thread.cpp

use std::{
    cell::UnsafeCell,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
        Arc,
    },
    thread::{sleep, JoinHandle},
};

use lock_api::RawMutex;

use crate::{
    board::Board,
    empty_list::EmptyList,
    eval,
    misc::BitSet,
    move_list::ConcurrentMoveIterator,
    search::{self, SearchResult, SearchTask},
    square::Square,
    transposition_table::TranspositionTable,
    types::{Depth, NodeType, NonPV, Root, Score, PV},
};

use crate::search::root_move::RootMove;
use crate::search::search_context::SearchContext;
use crate::search::spinlock;

use super::search_context::{GamePhase, SideToMove};

const MAX_SPLITPOINTS_PER_THREAD: usize = 8;
const MAX_SLAVES_PER_SPLITPOINT: usize = 3;

pub struct SplitPointState {
    all_slaves_searching: bool,
    pub alpha: Score,
    beta: Score,
    pub best_score: Score,
    pub best_move: Square,
    pub move_iter: Option<Arc<ConcurrentMoveIterator>>,
    node_type: u32,
    pub cutoff: bool,
    master_thread_idx: usize,
    slaves_mask: BitSet,
    depth: Depth,
    n_nodes: u64,
    pub task: Option<SplitPointSharedTask>,
    parent_split_point: Option<Arc<SplitPoint>>,
}

pub struct SplitPointSharedTask {
    pub board: Board,
    pub side_to_move: SideToMove,
    pub generation: u8,
    pub selectivity: u8,
    pub game_phase: GamePhase,
    pub tt: Arc<TranspositionTable>,
    pub root_moves: Arc<std::sync::Mutex<Vec<RootMove>>>,
    pub pool: Arc<ThreadPool>,
    pub eval: Arc<eval::Eval>,
    pub empty_list: EmptyList,
}

pub struct SplitPoint {
    mutex: spinlock::RawSpinLock,
    state: UnsafeCell<SplitPointState>,
}

unsafe impl Sync for SplitPoint {}

impl Default for SplitPoint {
    fn default() -> Self {
        SplitPoint {
            mutex: spinlock::RawSpinLock::INIT,
            state: UnsafeCell::new(SplitPointState {
                all_slaves_searching: false,
                alpha: 0,
                beta: 0,
                best_score: 0,
                best_move: Square::None,
                move_iter: None,
                node_type: 0,
                cutoff: false,
                master_thread_idx: 0,
                slaves_mask: BitSet::new(),
                depth: 0,
                n_nodes: 0,
                task: None,
                parent_split_point: None,
            }),
        }
    }
}

impl SplitPoint {
    #[inline]
    pub fn state(&self) -> &SplitPointState {
        unsafe { &*self.state.get() }
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub fn state_mut(&self) -> &mut SplitPointState {
        unsafe { &mut *self.state.get() }
    }

    #[inline]
    pub fn lock(&self) {
        self.mutex.lock();
    }

    #[inline]
    pub fn unlock(&self) {
        unsafe { self.mutex.unlock() };
    }
}

pub struct ThreadState {
    pub active_split_point: Option<Arc<SplitPoint>>,
    pub split_points_size: usize,
    split_points: [Arc<SplitPoint>; MAX_SPLITPOINTS_PER_THREAD],
    searching: bool,
    exit: bool,
}

pub struct Thread {
    mutex_for_sleep_condition: std::sync::Mutex<()>,
    mutex_for_state: spinlock::RawSpinLock,
    sleep_condition: std::sync::Condvar,
    idx: usize,
    thinking: Arc<AtomicBool>,
    state: UnsafeCell<ThreadState>,
    ready: AtomicBool,
}

unsafe impl Sync for Thread {}

impl Thread {
    fn new(idx: usize, thinking: Arc<AtomicBool>) -> Thread {
        let split_points = std::array::from_fn(|_| Arc::new(SplitPoint::default()));

        Thread {
            mutex_for_sleep_condition: std::sync::Mutex::new(()),
            mutex_for_state: spinlock::RawSpinLock::INIT,
            sleep_condition: std::sync::Condvar::new(),
            idx,
            thinking,
            state: UnsafeCell::new(ThreadState {
                active_split_point: None,
                split_points_size: 0,
                split_points,
                searching: false,
                exit: false,
            }),
            ready: AtomicBool::new(false),
        }
    }

    pub fn lock(&self) {
        self.mutex_for_state.lock();
    }

    pub fn unlock(&self) {
        unsafe { self.mutex_for_state.unlock() };
    }

    pub fn can_split(&self, thread_pool_size: usize) -> bool {
        if thread_pool_size <= 1 {
            return false;
        }

        let th_state = self.state();

        let cond = if let Some(sp) = &th_state.active_split_point {
            let sp_state = sp.state();
            !sp_state.all_slaves_searching
                || thread_pool_size > MAX_SLAVES_PER_SPLITPOINT
                    && sp_state.slaves_mask.count == MAX_SLAVES_PER_SPLITPOINT
        } else {
            true
        };

        cond && (th_state.split_points_size < MAX_SPLITPOINTS_PER_THREAD)
    }

    #[inline]
    pub fn state(&self) -> &ThreadState {
        unsafe { &*self.state.get() }
    }

    #[inline]
    #[allow(clippy::mut_from_ref)]
    fn state_mut(&self) -> &mut ThreadState {
        unsafe { &mut *self.state.get() }
    }

    /// ThreadBase::notify_one() wakes up the thread when there is some work to do
    fn notify_one(&self) {
        let _lock = self.mutex_for_sleep_condition.lock();
        self.sleep_condition.notify_one();
    }

    /// Thread::cutoff_occurred() checks whether a beta cutoff has occurred in the
    /// current active split point, or in some ancestor of the split point.
    ///
    /// # Returns
    /// * true if a beta cutoff has occurred
    pub fn cutoff_occurred(&self) -> bool {
        let mut current_sp = self.state().active_split_point.as_ref();
        while let Some(sp) = current_sp {
            let sp_state = sp.state();
            if sp_state.cutoff {
                return true;
            }
            current_sp = sp_state.parent_split_point.as_ref();
        }
        false
    }

    /// Thread::can_join() checks whether the thread is available to join the split
    /// point 'sp'. An obvious requirement is that thread must be idle. With more than
    /// two threads, this is not sufficient: If the thread is the master of some split
    /// point, it is only available as a slave for the split points below his active
    /// one (the "helpful master" concept in YBWC terminology).
    ///
    /// # Arguments
    /// * 'sp' the split point to check
    ///
    /// # Returns
    /// * true if the thread can join the split point
    fn can_join(&self, sp: &Arc<SplitPoint>) -> bool {
        let th_state = self.state();
        if th_state.searching {
            return false;
        }

        // Make a local copy to be sure it doesn't become zero under our feet while
        // testing next condition and so leading to an out of bounds access.
        let size = th_state.split_points_size;

        // No split points means that the thread is available as a slave for any
        // other thread otherwise apply the "helpful master" concept if possible.
        if size == 0 {
            return true;
        }

        let sp_state = th_state.split_points[size - 1].state();
        let master_idx = sp.state().master_thread_idx;
        sp_state.slaves_mask.test(master_idx)
    }

    /// Thread::split() does the actual work of distributing the work at a node between
    /// several available threads. If it does not succeed in splitting the node
    /// (because no idle threads are available), the function immediately returns.
    /// If splitting is possible, a SplitPoint object is initialized with all the
    /// data that must be copied to the helper threads and then helper threads are
    /// informed that they have been assigned work. This will cause them to instantly
    /// leave their idle loops and call search(). When all threads have returned from
    /// search() then split() returns.
    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &self,
        ctx: &SearchContext,
        board: &Board,
        alpha: Score,
        beta: Score,
        best_score: Score,
        best_move: Square,
        depth: Depth,
        move_iter: &Arc<ConcurrentMoveIterator>,
        node_type: u32,
    ) -> (Score, Square, u64) {
        let th_state = self.state();
        // Pick and init the next available split point
        let sp = &th_state.split_points[th_state.split_points_size];
        self.initialize_split_point(
            sp, ctx, depth, best_score, best_move, alpha, beta, node_type, move_iter, board,
        );

        idle_loop(&ctx.this_thread);

        self.finalize_split_point(sp);

        // Split point data cannot be changed now, so no need to lock protect
        let sp_state = sp.state();
        (sp_state.best_score, sp_state.best_move, sp_state.n_nodes)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn initialize_split_point(
        &self,
        sp: &Arc<SplitPoint>,
        ctx: &SearchContext,
        depth: Depth,
        best_score: Score,
        best_move: Square,
        alpha: Score,
        beta: Score,
        node_type: u32,
        move_iter: &Arc<ConcurrentMoveIterator>,
        board: &Board,
    ) {
        let th_state = self.state_mut();
        debug_assert!(th_state.searching);
        debug_assert!(th_state.split_points_size < MAX_SPLITPOINTS_PER_THREAD);

        sp.lock();
        // No contention here until we don't increment splitPointsSize
        let sp_state = sp.state_mut();
        sp_state.master_thread_idx = self.idx;
        sp_state.parent_split_point = th_state.active_split_point.clone();
        sp_state.slaves_mask.clear();
        sp_state.slaves_mask.set(self.idx);
        sp_state.depth = depth;
        sp_state.best_score = best_score;
        sp_state.best_move = best_move;
        sp_state.alpha = alpha;
        sp_state.beta = beta;
        sp_state.node_type = node_type;
        sp_state.move_iter = Some(move_iter.clone());
        sp_state.task = Some(SplitPointSharedTask {
            board: *board,
            side_to_move: ctx.side_to_move,
            generation: ctx.generation,
            selectivity: ctx.selectivity,
            tt: ctx.tt.clone(),
            root_moves: ctx.root_moves.clone(),
            pool: ctx.pool.clone(),
            eval: ctx.eval.clone(),
            game_phase: ctx.game_phase,
            empty_list: ctx.empty_list.clone(),
        });
        sp_state.n_nodes = 0;
        sp_state.cutoff = false;
        sp_state.all_slaves_searching = true; // Must be set under lock protection

        th_state.split_points_size += 1;
        th_state.active_split_point = Some(sp.clone());

        // Try to allocate available threads
        ctx.pool.assign_task_to_slaves(sp);

        // Everything is set up. The master thread enters the idle loop, from which
        // it will instantly launch a search, because its 'searching' flag is set.
        // The thread will return from the idle loop when all slaves have finished
        // their work at this split point.
        sp.unlock();
    }

    #[inline]
    fn finalize_split_point(&self, sp: &Arc<SplitPoint>) {
        let th_state = self.state_mut();
        debug_assert!(!th_state.searching);

        // In the helpful master concept, a master can help only a sub-tree of its
        // split point and because everything is finished here, it's not possible
        // for the master to be booked.
        self.lock();

        // We have returned from the idle loop, which means that all threads are
        // finished. Note that decreasing splitPointsSize must be done under lock
        // protection to avoid a race with Thread::can_join().
        th_state.searching = true;
        th_state.split_points_size -= 1;
        sp.state_mut().task = None;
        th_state.active_split_point = sp.state().parent_split_point.clone();

        self.unlock();
    }
}

enum Message {
    StartThinking(SearchTask, Arc<Thread>, Sender<SearchResult>),
    Exit,
}

pub struct ThreadPool {
    threads: Vec<Arc<Thread>>,
    thread_handles: Vec<JoinHandle<()>>,
    pub size: usize,
    thinking: Arc<AtomicBool>,
    sender: Arc<Sender<Message>>,
    receiver: Arc<std::sync::Mutex<Receiver<Message>>>,
    abort_flag: Arc<AtomicBool>,
}

impl ThreadPool {
    pub fn new(n_threads: usize) -> ThreadPool {
        let (sender, receiver) = std::sync::mpsc::channel();

        ThreadPool {
            threads: Vec::new(),
            thread_handles: Vec::new(),
            size: n_threads,
            thinking: Arc::new(AtomicBool::new(false)),
            sender: Arc::new(sender),
            receiver: Arc::new(std::sync::Mutex::new(receiver)),
            abort_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn init(&mut self) {
        let main_thread = Arc::new(Thread::new(0, self.thinking.clone()));
        let main_thread_clone = main_thread.clone();
        let receiver_clone = self.receiver.clone();
        let main_thread_handle =
            std::thread::spawn(move || main_thread_loop(main_thread_clone, receiver_clone));

        self.threads.push(main_thread);
        self.thread_handles.push(main_thread_handle);

        for i in 1..self.size {
            let thread = Arc::new(Thread::new(i, self.thinking.clone()));
            let thread_clone = thread.clone();
            let handle = std::thread::spawn(move || idle_loop(&thread_clone));
            self.threads.push(thread);
            self.thread_handles.push(handle);
        }

        // Wait for all threads to be ready
        self.main().ready.store(true, Ordering::SeqCst);
        loop {
            sleep(std::time::Duration::from_millis(10));
            let ready = self
                .threads
                .iter()
                .all(|thread| thread.ready.load(Ordering::Relaxed));
            if ready {
                break;
            }
        }
    }

    fn exit(&mut self) {
        for thread in &self.threads {
            let lock = thread.mutex_for_sleep_condition.lock();
            thread.state_mut().exit = true;
            drop(lock);

            thread.notify_one();
        }

        self.sender.send(Message::Exit).unwrap();

        for thread_handle in self.thread_handles.drain(..) {
            thread_handle.join().expect("Thread panicked");
        }

        self.threads.clear();
        drop(self.sender.clone());
    }

    fn assign_task_to_slaves(&self, sp: &Arc<SplitPoint>) {
        let sp_state = sp.state_mut();
        while sp_state.slaves_mask.count < MAX_SLAVES_PER_SPLITPOINT {
            if let Some(slave) = self.find_available_thread(sp) {
                slave.lock();

                if slave.can_join(sp) {
                    sp_state.slaves_mask.set(slave.idx);
                    let slave_state = slave.state_mut();
                    slave_state.active_split_point = Some(sp.clone());
                    slave_state.searching = true;
                }
                slave.unlock();
            } else {
                break;
            }
        }
    }

    fn find_available_thread(&self, sp: &Arc<SplitPoint>) -> Option<Arc<Thread>> {
        self.threads
            .iter()
            .find(|thread| thread.can_join(sp))
            .cloned()
    }

    /// Try to late join to another split point if none of its slaves has already finished.
    fn try_late_join(&self, thread: &Arc<Thread>) {
        let mut best_sp = None;
        let mut min_level = i32::MAX;
        for th in &self.threads {
            let size = th.state().split_points_size;
            if size == 0 {
                continue;
            }

            let sp = &th.state().split_points[size - 1];
            let sp_state = sp.state();
            if sp_state.all_slaves_searching
                && sp_state.slaves_mask.count < MAX_SLAVES_PER_SPLITPOINT
                && thread.can_join(sp)
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
            if sp_state.all_slaves_searching
                && sp_state.slaves_mask.count < MAX_SLAVES_PER_SPLITPOINT
            {
                thread.lock();

                if thread.can_join(sp) {
                    sp_state.slaves_mask.set(thread.idx);
                    let th_state = thread.state_mut();
                    th_state.active_split_point = Some(sp.clone());
                    th_state.searching = true;
                }

                thread.unlock();
            }

            sp.unlock();
        }
    }

    pub fn start_thinking(&self, task: SearchTask) -> std::sync::mpsc::Receiver<SearchResult> {
        let (result_sender, result_receiver) = std::sync::mpsc::channel();

        self.reset_abort_flag();

        self.thinking.store(true, Ordering::SeqCst);
        self.sender
            .send(Message::StartThinking(
                task,
                self.main().clone(),
                result_sender,
            ))
            .unwrap();

        result_receiver
    }

    pub fn main(&self) -> &Arc<Thread> {
        &self.threads[0]
    }

    pub fn notify_all(&self) {
        for thread in &self.threads {
            thread.notify_one();
        }
    }

    pub fn wait_for_search_finished(&self) {
        loop {
            let searching = self.threads.iter().any(|thread| thread.state().searching);
            if !searching {
                break;
            }
            sleep(std::time::Duration::from_millis(1));
        }
    }

    pub fn abort_search(&self) {
        self.abort_flag.store(true, Ordering::SeqCst);
    }

    pub fn reset_abort_flag(&self) {
        self.abort_flag.store(false, Ordering::SeqCst);
    }

    pub fn is_aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.exit();
    }
}

fn main_thread_loop(thread: Arc<Thread>, receiver: Arc<std::sync::Mutex<Receiver<Message>>>) {
    while !thread.state().exit {
        let receiver = receiver.lock().unwrap();
        if let Ok(message) = receiver.recv() {
            match message {
                Message::StartThinking(task, thread, result_sender) => {
                    thread.state_mut().searching = true;

                    task.pool.notify_all();

                    let result = search::search_root(task, &thread);
                    thread.state_mut().searching = false;

                    thread.thinking.store(false, Ordering::SeqCst);
                    result_sender.send(result).unwrap();
                }
                Message::Exit => {
                    thread.state_mut().exit = true;
                    break;
                }
            }
        }
    }
}

fn idle_loop(thread: &Arc<Thread>) {
    // Pointer 'this_sp' is not null only if we are called from split(), and not
    // at the thread creation. This means we are the split point's master.
    let this_sp = thread.state().active_split_point.clone();

    while !thread.state().exit {
        if let Some(ref sp) = this_sp {
            if sp.state().slaves_mask.none() {
                break;
            }
        }

        // If this thread has been assigned work, launch a search
        while thread.state().searching {
            thread.lock();
            debug_assert!(thread.state().active_split_point.is_some());
            let sp = thread.state().active_split_point.as_ref().unwrap().clone();
            thread.unlock();

            sp.lock();
            let task = sp.state().task.as_ref().unwrap();
            let task_board = task.board;
            let sp_state = sp.state();
            let depth = sp_state.depth;
            let alpha = sp_state.alpha;
            let beta = sp_state.beta;
            let node_type = sp_state.node_type;

            let mut ctx = SearchContext::from_split_point(&sp, thread);

            if ctx.empty_list.count == depth {
                if node_type == NonPV::TYPE_ID {
                    search::endgame::search::<NonPV, true>(
                        &mut ctx,
                        &task_board,
                        alpha,
                        beta,
                        Some(&sp),
                    );
                } else if node_type == PV::TYPE_ID {
                    search::endgame::search::<PV, true>(
                        &mut ctx,
                        &task_board,
                        alpha,
                        beta,
                        Some(&sp),
                    );
                } else if node_type == Root::TYPE_ID {
                    search::endgame::search::<Root, true>(
                        &mut ctx,
                        &task_board,
                        alpha,
                        beta,
                        Some(&sp),
                    );
                } else {
                    unreachable!();
                }
            } else if node_type == NonPV::TYPE_ID {
                search::midgame::search::<NonPV, true>(
                    &mut ctx,
                    &task_board,
                    depth,
                    alpha,
                    beta,
                    Some(&sp),
                );
            } else if node_type == PV::TYPE_ID {
                search::midgame::search::<PV, true>(
                    &mut ctx,
                    &task_board,
                    depth,
                    alpha,
                    beta,
                    Some(&sp),
                );
            } else if node_type == Root::TYPE_ID {
                search::midgame::search::<Root, true>(
                    &mut ctx,
                    &task_board,
                    depth,
                    alpha,
                    beta,
                    Some(&sp),
                );
            } else {
                unreachable!();
            }

            let th_state = thread.state_mut();

            thread.lock();
            th_state.searching = false;
            thread.unlock();

            let sp_state = sp.state_mut();
            sp_state.slaves_mask.reset(thread.idx);
            sp_state.all_slaves_searching = false;
            sp_state.n_nodes += ctx.n_nodes;

            // After releasing the lock we can't access any SplitPoint related data
            // in a safe way because it could have been released under our feet by
            // the sp master.
            sp.unlock();
            drop(sp);

            ctx.pool.try_late_join(thread);
        }

        // If search is finished then sleep, otherwise just yield
        if !thread.thinking.load(Ordering::SeqCst) {
            debug_assert!(this_sp.is_none());

            let mut lock = thread.mutex_for_sleep_condition.lock().unwrap();
            thread.ready.store(true, Ordering::SeqCst);
            while !thread.state().exit && !thread.thinking.load(Ordering::Relaxed) {
                lock = thread.sleep_condition.wait(lock).unwrap();
            }
        } else {
            // Wait for a new job or for our slaves to finish
            std::thread::yield_now();
        }
    }
}
