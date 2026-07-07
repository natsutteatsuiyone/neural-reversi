use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, Weak};
use std::thread::{JoinHandle, sleep};
use std::time::{Duration, Instant};

use crate::search::SearchTask;
use crate::search::result::SearchResult;
use crate::search::time_control::TimeManager;
use crate::util::align::Align64;

use super::AbortState;
use super::message::Message;
use super::split_point::SplitPoint;
use super::thread::Thread;

/// Interval between checks for abort flag in milliseconds.
const CHECK_INTERVAL_MS: u64 = 1;

/// Interval between polls while waiting for threads to signal ready.
const READY_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Thread pool for parallel game tree search.
pub struct ThreadPool {
    /// Collection of all threads in the pool.
    pub(super) threads: Vec<Arc<Thread>>,

    /// Join handles for thread cleanup on shutdown.
    thread_handles: Vec<JoinHandle<()>>,

    /// Number of threads in the pool.
    pub size: usize,

    /// Global flag indicating if the engine is thinking.
    pub(super) thinking: Arc<AtomicBool>,

    /// Channel sender for sending messages to the main thread.
    sender: Sender<Message>,

    /// Generation-tagged state for aborting the current search.
    abort_state: Arc<AbortState>,

    /// Incremented whenever a split point first records a cutoff.
    cutoff_epoch: Arc<Align64<AtomicU64>>,

    /// Handle for the timer thread (protected by mutex for interior mutability).
    timer_handle: Mutex<Option<JoinHandle<()>>>,

    /// Flag to signal the timer thread to stop.
    timer_stop: Arc<AtomicBool>,
}

impl ThreadPool {
    /// Creates a new thread pool with the specified number of threads and starts them
    /// immediately.
    pub fn new(n_threads: usize) -> Arc<ThreadPool> {
        Arc::new_cyclic(|weak| {
            let (sender, receiver) = mpsc::channel();

            let mut pool = ThreadPool {
                threads: Vec::new(),
                thread_handles: Vec::new(),
                size: n_threads,
                thinking: Arc::new(AtomicBool::new(false)),
                sender,
                abort_state: Arc::new(AbortState::new()),
                cutoff_epoch: Arc::new(Align64(AtomicU64::new(0))),
                timer_handle: Mutex::new(None),
                timer_stop: Arc::new(AtomicBool::new(false)),
            };

            pool.init(weak, receiver);
            pool
        })
    }

    /// Initializes the thread pool by creating and starting all threads.
    fn init(&mut self, pool: &Weak<ThreadPool>, receiver: Receiver<Message>) {
        self.create_main_thread(pool, receiver);
        self.create_worker_threads(pool);
        self.wait_for_threads_ready();
    }

    /// Creates a `Thread` sharing this pool's flags and registers it in `threads`.
    fn add_thread(&mut self, idx: usize, pool: &Weak<ThreadPool>) -> Arc<Thread> {
        let thread = Arc::new(Thread::new(
            idx,
            self.thinking.clone(),
            self.abort_state.clone(),
            self.cutoff_epoch.clone(),
            self.size,
            pool.clone(),
        ));
        self.threads.push(thread.clone());
        thread
    }

    /// Creates and starts the main thread that handles control messages.
    fn create_main_thread(&mut self, pool: &Weak<ThreadPool>, receiver: Receiver<Message>) {
        let thread = self.add_thread(0, pool);
        let handle = std::thread::spawn(move || thread.main_thread_loop(receiver));

        self.thread_handles.push(handle);
    }

    /// Creates and starts worker threads that wait in idle loops.
    fn create_worker_threads(&mut self, pool: &Weak<ThreadPool>) {
        for i in 1..self.size {
            let thread = self.add_thread(i, pool);
            let handle = std::thread::spawn(move || thread.idle_loop());

            self.thread_handles.push(handle);
        }
    }

    /// Waits for all threads to signal they are ready.
    fn wait_for_threads_ready(&self) {
        self.main().ready.store(true, Ordering::Release);

        while !self.all_threads_ready() {
            sleep(READY_POLL_INTERVAL);
        }
    }

    /// Checks whether all threads have signaled they are ready.
    fn all_threads_ready(&self) -> bool {
        self.threads
            .iter()
            .all(|thread| thread.ready.load(Ordering::Acquire))
    }

    /// Shuts down the thread pool and waits for all threads to exit.
    fn exit(&mut self) {
        // Already shut down - nothing to do
        if self.threads.is_empty() {
            return;
        }

        // Stop timer thread first to prevent it from setting abort flags
        self.stop_timer();

        // Signal all worker threads to exit and wake them up
        for thread in &self.threads {
            let _lock = thread.mutex_for_sleep_condition.lock().unwrap();
            thread.exit.store(true, Ordering::Release);
        }

        // Wake up all sleeping threads so they can observe the exit flag
        self.notify_all();

        // Send exit message to main thread's message loop
        // Ignore send error if receiver is already dropped
        let _ = self.sender.send(Message::Exit);

        // Join all threads; panic info is absorbed (the panicking thread has
        // already logged at the panic site).
        for handle in self.thread_handles.drain(..) {
            let _ = handle.join();
        }

        // Clear thread references
        self.threads.clear();
    }

    /// Assigns idle threads to work on a split point.
    pub(super) fn assign_helpers_to_split_point(&self, sp: &Arc<SplitPoint>) {
        let sp_state = sp.state_mut();
        let max_threads = sp_state.max_threads();

        for thread in &self.threads {
            if sp_state.helpers_mask.count() >= max_threads {
                break;
            }

            // Quick check before acquiring thread lock
            if !thread.can_join(sp) {
                continue;
            }

            thread.lock();
            if thread.can_join(sp) {
                thread.book_into(sp, sp_state);
            }
            thread.unlock();
        }
    }

    /// Starts a new search task on the thread pool and returns a receiver for the result.
    pub fn start_thinking(&self, task: SearchTask) -> Receiver<SearchResult> {
        debug_assert!(
            !self.threads.is_empty(),
            "Cannot start thinking: thread pool has been shut down"
        );

        let (result_sender, result_receiver) = mpsc::channel();

        // Start a fresh generation before publishing the active search.
        self.abort_state.begin_search();

        // Mark pool as actively thinking before sending message
        self.thinking.store(true, Ordering::Release);

        // Dispatch task to main thread
        self.sender
            .send(Message::StartThinking(
                task,
                self.main().clone(),
                result_sender,
            ))
            .expect("main thread receiver must be alive while the pool exists");

        result_receiver
    }

    /// Returns a reference to the main thread (thread 0).
    pub fn main(&self) -> &Arc<Thread> {
        &self.threads[0]
    }

    /// Wakes up all threads in the pool.
    pub(super) fn notify_all(&self) {
        for thread in &self.threads {
            thread.notify_one();
        }
    }

    /// Waits for the current search to complete.
    pub fn wait_for_think_finished(&self) {
        const POLL_INTERVAL: Duration = Duration::from_millis(5);

        while self.thinking.load(Ordering::Acquire) {
            sleep(POLL_INTERVAL);
        }
    }

    /// Signals all threads to abort the current search.
    pub fn abort_search(&self) {
        self.abort_state.request_abort();
    }

    /// Checks whether the current search has been aborted.
    #[inline]
    pub fn is_aborted(&self) -> bool {
        self.abort_state.is_aborted()
    }

    /// Returns a clone of the abort state for external use (e.g., time management).
    pub(crate) fn abort_state(&self) -> Arc<AbortState> {
        self.abort_state.clone()
    }

    /// Starts a timer thread that will request abort when deadline is reached.
    ///
    /// - Checks every `CHECK_INTERVAL_MS` milliseconds against the current deadline
    /// - Responds to dynamic time extensions from the [`TimeManager`]
    /// - Exits cleanly when:
    ///   - Deadline is reached (requests abort)
    ///   - [`stop_timer`](Self::stop_timer) is called (search completed early)
    ///   - No deadline is set (infinite time mode)
    pub fn start_timer(&self, time_manager: Arc<TimeManager>) {
        // Reset stop flag before spawning new timer
        self.timer_stop.store(false, Ordering::Release);

        let abort_state = self.abort_state.clone();
        let stop_flag = self.timer_stop.clone();

        let handle = std::thread::Builder::new()
            .name("search-timer".to_string())
            .spawn(move || {
                Self::timer_loop(&time_manager, &abort_state, &stop_flag);
            })
            .expect("failed to spawn timer thread");

        *self.timer_handle.lock().unwrap() = Some(handle);
    }

    /// Runs the timer thread loop.
    fn timer_loop(time_manager: &TimeManager, abort_state: &AbortState, stop_flag: &AtomicBool) {
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
                    abort_state.request_abort();
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
    /// This should be called after the search result is received, or before
    /// explicitly aborting a search, to ensure clean shutdown.
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
