//! Serialized, off-thread access to the native engine.
//!
//! `new` initializes asynchronously and reports `Ready` (or `Error`) with id 0.
//! Every submission and cancellation supersedes earlier generations. Cancelled
//! jobs have no terminal event; `try_recv` also discards already-queued stale
//! events. Callers should retain the returned id when associating events with UI
//! state. Initialize completes with `Ready`; all search requests complete with
//! `Finished`. Dropping the handle cancels without joining engine threads.

use std::any::Any;
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use reversi_core::analysis::{self, GameAnalysisMove, GameAnalysisProgress};
use reversi_core::board::Board;
use reversi_core::level::{ENDGAME_STEPS, Level, get_level};
use reversi_core::search::options::SearchOptions;
use reversi_core::search::result::SearchResult;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{Search, SearchProgress, SearchRunOptions, StopToken};
use reversi_core::square::Square;

#[derive(Debug)]
pub enum Request {
    Initialize {
        hash_size: usize,
    },
    Move {
        board: Board,
        level: usize,
        /// Remaining Fischer main time; increment is applied by the game clock.
        /// When present, the fixed difficulty level is intentionally ignored.
        remaining_ms: Option<u64>,
    },
    Hint {
        board: Board,
        level: usize,
    },
    Solver {
        board: Board,
        /// Index into `Level::ENDGAME_SELECTIVITY` (the last step is exact).
        selectivity: usize,
        multi_pv: bool,
    },
    AnalyzeGame {
        /// Initial position relative to its side to move, before `moves`.
        board: Board,
        /// Includes explicit passes, which retain their indices in analysis.
        moves: Vec<GameAnalysisMove>,
        level: usize,
    },
}

pub struct Event {
    pub id: u64,
    pub kind: EventKind,
}

pub enum EventKind {
    Ready,
    Progress(Progress),
    Analysis(GameAnalysisProgress),
    Finished(Outcome),
    Error(String),
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Progress {
    pub square: Option<Square>,
    pub score: f32,
    pub depth: u32,
    pub target_depth: u32,
    pub probability: i32,
    pub nodes: u64,
    pub nps: f64,
    pub pv: String,
    pub is_endgame: bool,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub best_move: Option<Square>,
    pub score: Option<f32>,
    pub elapsed_ms: u64,
}

pub struct Engine {
    requests: Sender<Job>,
    events: Receiver<Event>,
    // Retained so a failed worker submission can use the normal error surface.
    errors: Sender<Event>,
    generation: Arc<AtomicU64>,
    /// Stops the latest submitted job; `None` once it is cancelled.
    current_stop: Mutex<Option<StopToken>>,
}

impl Engine {
    #[cfg(test)]
    pub(crate) fn emit_for_test(&self, id: u64, kind: EventKind) {
        self.errors.send(Event { id, kind }).unwrap();
    }

    pub fn new(hash_size: usize) -> Self {
        let (requests, jobs) = mpsc::channel();
        let (errors, events) = mpsc::channel();
        let generation = Arc::new(AtomicU64::new(0));
        let worker_generation = generation.clone();
        let worker_events = errors.clone();
        if let Err(error) = thread::Builder::new()
            .name("reversi-engine".into())
            .spawn(move || worker(jobs, worker_events, worker_generation, hash_size))
        {
            let _ = errors.send(Event {
                id: 0,
                kind: EventKind::Error(format!("Cannot start the engine worker: {error}")),
            });
        }
        Self {
            requests,
            events,
            errors,
            generation,
            current_stop: Mutex::new(None),
        }
    }

    /// Supersedes any previous job and returns this request's generation.
    pub fn submit(&self, request: Request) -> u64 {
        let id = self
            .generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let stop = StopToken::new();
        if let Some(previous) = self.current_stop.lock().unwrap().replace(stop.clone()) {
            previous.stop();
        }
        if self.requests.send(Job { id, request, stop }).is_err() {
            let _ = self.errors.send(Event {
                id,
                kind: EventKind::Error("The engine worker is no longer available".into()),
            });
        }
        id
    }

    /// Invalidates queued events immediately and stops the current job.
    pub fn cancel(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        if let Some(stop) = self.current_stop.lock().unwrap().take() {
            stop.stop();
        }
    }

    /// Stops thinking but preserves the current generation and its best-so-far
    /// `Finished` result. Unlike `cancel`, this is suitable for an AI Move's
    /// Stop button. The core supplies a legal shallow move if no iteration ended.
    pub fn stop(&self) {
        if let Some(stop) = &*self.current_stop.lock().unwrap() {
            stop.stop();
        }
    }

    pub fn try_recv(&self) -> Option<Event> {
        while let Ok(event) = self.events.try_recv() {
            if is_current(&self.generation, event.id) {
                return Some(event);
            }
        }
        None
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.cancel();
        // Disconnecting `requests` wakes the worker. Search and pool destruction
        // (including their joins) remain on background threads.
    }
}

struct Job {
    id: u64,
    request: Request,
    /// Carried by every search run of the job, so stopping it before the
    /// worker reaches the job still takes effect.
    stop: StopToken,
}

fn is_current(generation: &AtomicU64, id: u64) -> bool {
    generation.load(Ordering::Acquire) == id
}

fn emit(events: &Sender<Event>, generation: &AtomicU64, id: u64, kind: EventKind) {
    if is_current(generation, id) {
        let _ = events.send(Event { id, kind });
    }
}

fn worker(
    jobs: Receiver<Job>,
    events: Sender<Event>,
    generation: Arc<AtomicU64>,
    hash_size: usize,
) {
    let mut search = None;
    let mut initialization_error = None;
    let mut job = Job {
        id: 0,
        request: Request::Initialize { hash_size },
        stop: StopToken::new(),
    };
    loop {
        // Initial construction is required even if a request arrived before the
        // worker started. Later obsolete jobs can be skipped without searching.
        if job.id == 0 || is_current(&generation, job.id) {
            let active_id = job.id;
            // Core initialization currently exposes only a panicking constructor.
            // This recovers unwind panics, not panic=abort builds or allocator OOM.
            let result = catch_unwind(AssertUnwindSafe(|| {
                execute(
                    job,
                    &mut search,
                    initialization_error.as_deref(),
                    &events,
                    &generation,
                )
            }));
            let result = match result {
                Ok(result) => result,
                Err(payload) => {
                    let error = panic_message(payload);
                    search = None;
                    initialization_error = Some(error.clone());
                    Err(error)
                }
            };
            match result {
                Ok((id, kind)) => {
                    initialization_error = None;
                    emit(&events, &generation, id, kind);
                }
                Err(error) => {
                    // `execute` returns only errors for the job being processed;
                    // do not retag a stale failure as a successor's result.
                    emit(&events, &generation, active_id, EventKind::Error(error));
                }
            }
        }
        match jobs.recv() {
            Ok(next) => job = next,
            Err(_) => return,
        }
    }
}

fn execute(
    job: Job,
    search: &mut Option<Search>,
    initialization_error: Option<&str>,
    events: &Sender<Event>,
    generation: &Arc<AtomicU64>,
) -> Result<(u64, EventKind), String> {
    let Job { id, request, stop } = job;
    if let Request::Initialize { hash_size } = request {
        if let Some(search) = search {
            search.resize_tt(hash_size);
            search.init();
        } else {
            *search = Some(Search::new(&SearchOptions::new(hash_size)));
        }
        return Ok((id, EventKind::Ready));
    }
    let search = search.as_mut().ok_or_else(|| {
        format!(
            "AI backend unavailable: {}",
            initialization_error.unwrap_or("initialization did not complete")
        )
    })?;
    let start = Instant::now();
    let (board, options) = match request {
        Request::Move {
            board,
            level,
            remaining_ms,
        } => {
            let options = if let Some(main_time_ms) = remaining_ms {
                SearchRunOptions::with_time(TimeControlMode::Fischer {
                    main_time_ms,
                    increment_ms: 0,
                })
            } else {
                SearchRunOptions::with_level(get_level(level))
            };
            (board, options)
        }
        Request::Hint { board, level } => (
            board,
            SearchRunOptions::with_level(get_level(level)).multi_pv(true),
        ),
        Request::Solver {
            board,
            selectivity,
            multi_pv,
        } => (
            board,
            SearchRunOptions::with_level(solver_level(selectivity)).multi_pv(multi_pv),
        ),
        Request::AnalyzeGame {
            board,
            moves,
            level,
        } => {
            let options = SearchRunOptions::with_level(get_level(level)).stop_token(stop);
            let search = RefCell::new(search);
            analysis::analyze_game(
                board,
                &moves,
                |board| {
                    if !is_current(generation, id) {
                        return Err("Game analysis cancelled".into());
                    }
                    let result = search.borrow_mut().run(board, &options);
                    match (result.best_move(), result.score()) {
                        (Some(best_move), Some(score)) => Ok(analysis::Analysis {
                            pv: result.pv_line().to_vec(),
                            best_move,
                            score: round_score(score),
                            depth: result.depth(),
                        }),
                        _ => Err("Search returned no legal move for game analysis position".into()),
                    }
                },
                || !is_current(generation, id),
                |progress| emit(events, generation, id, EventKind::Analysis(progress)),
            )?;
            return Ok((
                id,
                EventKind::Finished(Outcome {
                    best_move: None,
                    score: None,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                }),
            ));
        }
        Request::Initialize { .. } => unreachable!(),
    };
    let callback_events = events.clone();
    let callback_generation = generation.clone();
    let options = options.stop_token(stop).callback(move |progress| {
        emit(
            &callback_events,
            &callback_generation,
            id,
            EventKind::Progress(progress_payload(progress, start.elapsed())),
        );
    });
    let result = search.run(&board, &options);
    Ok((id, EventKind::Finished(outcome(&result, start))))
}

fn solver_level(selectivity: usize) -> Level {
    let mut end_depth = [0; ENDGAME_STEPS];
    end_depth[..=selectivity].fill(60);
    Level::with_depths(60, end_depth)
}

fn round_score(score: f32) -> f32 {
    (score * 10.0).round() / 10.0
}

pub fn format_pv(squares: &[Square]) -> String {
    squares
        .iter()
        .map(Square::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

fn progress_payload(progress: SearchProgress, elapsed: Duration) -> Progress {
    let elapsed_ms = elapsed.as_millis();
    Progress {
        square: (progress.best_move != Square::None).then_some(progress.best_move),
        score: round_score(progress.score),
        depth: progress.depth,
        target_depth: progress.target_depth,
        probability: progress.probability,
        nodes: progress.nodes,
        nps: if elapsed_ms > 0 {
            progress.nodes as f64 * 1000. / elapsed_ms as f64
        } else {
            0.
        },
        pv: format_pv(&progress.pv_line),
        is_endgame: progress.is_endgame,
    }
}

fn outcome(result: &SearchResult, start: Instant) -> Outcome {
    Outcome {
        best_move: result.best_move(),
        score: result.score().map(round_score),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    let detail = if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else {
        "unknown engine panic"
    };
    format!("AI backend failed: {detail}")
}

#[cfg(test)]
mod tests {
    use reversi_core::level::MAX_LEVEL;

    use super::*;

    #[test]
    fn stop_before_the_worker_reaches_the_job_still_ends_its_search() {
        let engine = Engine::new(1);
        let board = Board::new().try_make_move(Square::D3).unwrap();
        // The worker is still initializing, so this stop lands before the
        // search begins; a lost stop would leave a level-max search running.
        let id = engine.submit(Request::Move {
            board,
            level: MAX_LEVEL,
            remaining_ms: None,
        });
        engine.stop();

        // Generous: the worker first loads the weights, which is slow in
        // debug builds; a lost stop would run the level-max search far longer.
        let deadline = Instant::now() + Duration::from_secs(60);
        let outcome = loop {
            match engine.try_recv() {
                Some(Event {
                    id: event_id,
                    kind: EventKind::Finished(outcome),
                }) if event_id == id => break outcome,
                Some(Event {
                    kind: EventKind::Error(error),
                    ..
                }) => panic!("{error}"),
                _ => {}
            }
            assert!(Instant::now() < deadline, "the stop was lost");
            thread::sleep(Duration::from_millis(5));
        };
        assert!(board.is_legal_move(outcome.best_move.unwrap()));
    }
}
