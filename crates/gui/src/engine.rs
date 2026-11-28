//! Serialized, off-thread access to the native engine.
//!
//! `new` initializes asynchronously and reports `Ready` with id 0.
//! Every submission and cancellation supersedes earlier generations. Cancelled
//! jobs have no terminal event, though a cancelled Initialize is still applied
//! before later jobs; `try_recv` also discards already-queued stale
//! events. Callers should retain the returned id when associating events with UI
//! state. Initialize completes with `Ready`; all search requests complete with
//! `Finished`. Dropping the handle cancels without joining engine threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use reversi_core::analysis::{self, GameAnalysisMove, GameAnalysisProgress};
use reversi_core::board::Board;
use reversi_core::engine::{Engine as NativeEngine, GameId};
use reversi_core::level::Level;
use reversi_core::probcut::Selectivity;
use reversi_core::score::Score;
#[cfg(not(test))]
use reversi_core::search::options::SearchOptions;
use reversi_core::search::result::SearchResult;
use reversi_core::search::time_control::TimeControlMode;
use reversi_core::search::{SearchProgress, SearchRunOptions, StopToken};
use reversi_core::square::Square;

#[derive(Debug)]
pub enum Request {
    Initialize {
        hash_size: usize,
    },
    Move {
        board: Board,
        level: usize,
        /// Remaining time on the AI's per-game clock.
        /// When present, the fixed difficulty level is intentionally ignored.
        remaining_ms: Option<u64>,
    },
    Hint {
        board: Board,
        level: usize,
    },
    Solver {
        board: Board,
        selectivity: Selectivity,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    WorkerStart(String),
    WorkerUnavailable,
    AnalysisNoMove,
    Other(String),
}

pub enum EventKind {
    Ready,
    Progress(Progress),
    Analysis(GameAnalysisProgress),
    Finished(Outcome),
    Error(EngineError),
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Progress {
    pub square: Option<Square>,
    pub score: Score,
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
    pub score: Option<Score>,
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
                kind: EventKind::Error(EngineError::WorkerStart(error.to_string())),
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
                kind: EventKind::Error(EngineError::WorkerUnavailable),
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

struct PanicGuard {
    events: Sender<Event>,
    generation: Arc<AtomicU64>,
}

impl Drop for PanicGuard {
    fn drop(&mut self) {
        if thread::panicking() {
            let id = self.generation.load(Ordering::Acquire);
            let _ = self.events.send(Event {
                id,
                kind: EventKind::Error(EngineError::WorkerUnavailable),
            });
        }
    }
}

fn worker(
    jobs: Receiver<Job>,
    events: Sender<Event>,
    generation: Arc<AtomicU64>,
    hash_size: usize,
) {
    let _panic_guard = PanicGuard {
        events: events.clone(),
        generation: generation.clone(),
    };
    let mut engine = None;
    let mut job = Job {
        id: 0,
        request: Request::Initialize { hash_size },
        stop: StopToken::new(),
    };
    loop {
        if should_execute(&job, &generation) {
            let active_id = job.id;
            match execute(job, &mut engine, &events, &generation) {
                Ok((id, kind)) => emit(&events, &generation, id, kind),
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

/// Obsolete searches are skipped, but Initialize always runs: it builds the
/// engine or applies a new hash size, which later jobs and the saved settings
/// rely on even after a newer request superseded it.
fn should_execute(job: &Job, generation: &AtomicU64) -> bool {
    matches!(job.request, Request::Initialize { .. }) || is_current(generation, job.id)
}

#[cfg(test)]
fn engine_for_tests(hash_size: usize) -> NativeEngine {
    use std::sync::OnceLock;

    use reversi_core::eval::Eval;

    static EVAL: OnceLock<Arc<Eval>> = OnceLock::new();
    let eval = EVAL
        .get_or_init(|| {
            Arc::new(
                Eval::with_weight_files(None, None).expect("embedded evaluation weights must load"),
            )
        })
        .clone();
    NativeEngine::from_eval(eval, hash_size, Some(1))
}

#[cfg(not(test))]
fn create_engine(hash_size: usize) -> NativeEngine {
    NativeEngine::new(&SearchOptions::new(hash_size))
}

#[cfg(test)]
fn create_engine(hash_size: usize) -> NativeEngine {
    engine_for_tests(hash_size)
}

fn execute(
    job: Job,
    engine: &mut Option<(NativeEngine, GameId)>,
    events: &Sender<Event>,
    generation: &Arc<AtomicU64>,
) -> Result<(u64, EventKind), EngineError> {
    let Job { id, request, stop } = job;
    if let Request::Initialize { hash_size } = request {
        if let Some((engine, game)) = engine {
            engine.resize_tt(*game, hash_size);
            engine.reset(*game);
        } else {
            let native = create_engine(hash_size);
            let game = native.open();
            *engine = Some((native, game));
        }
        return Ok((id, EventKind::Ready));
    }
    let (engine, game) = engine
        .as_ref()
        .expect("the worker runs Initialize before any other job");
    let game = *game;
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
                SearchRunOptions::with_level(Level::preset(level))
            };
            (board, options)
        }
        Request::Hint { board, level } => (
            board,
            SearchRunOptions::with_level(Level::preset(level)).multi_pv(true),
        ),
        Request::Solver {
            board,
            selectivity,
            multi_pv,
        } => (
            board,
            SearchRunOptions::with_level(Level::perfect().stopping_at(selectivity))
                .multi_pv(multi_pv),
        ),
        Request::AnalyzeGame {
            board,
            moves,
            level,
        } => {
            let options = SearchRunOptions::with_level(Level::preset(level)).stop_token(stop);
            let mut analysis_error = None;
            if let Err(error) = analysis::analyze_game(
                board,
                &moves,
                |board| {
                    if !is_current(generation, id) {
                        return Err(String::new());
                    }
                    let result = engine.search(game, board, &options);
                    match (result.best_move(), result.score()) {
                        (Some(best_move), Some(score)) => Ok(analysis::Analysis {
                            pv: result.pv_line().to_vec(),
                            best_move,
                            score,
                        }),
                        _ => {
                            analysis_error = Some(EngineError::AnalysisNoMove);
                            Err(String::new())
                        }
                    }
                },
                || !is_current(generation, id),
                |progress| emit(events, generation, id, EventKind::Analysis(progress)),
            ) {
                return Err(analysis_error.unwrap_or(EngineError::Other(error.to_string())));
            }
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
    let result = engine.search(game, &board, &options);
    Ok((id, EventKind::Finished(outcome(&result, start))))
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
        square: progress.best_move,
        score: progress.score,
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
        score: result.score(),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use reversi_core::level::MAX_LEVEL;

    use super::*;

    #[test]
    fn superseded_initialize_still_runs_but_superseded_search_does_not() {
        let generation = AtomicU64::new(2);
        let job = |request| Job {
            id: 1,
            request,
            stop: StopToken::new(),
        };
        assert!(should_execute(
            &job(Request::Initialize { hash_size: 2 }),
            &generation
        ));
        assert!(!should_execute(
            &job(Request::Move {
                board: Board::new(),
                level: 1,
                remaining_ms: None,
            }),
            &generation
        ));
    }

    #[test]
    fn stop_before_the_worker_reaches_the_job_still_ends_its_search() {
        let engine = Engine::new(1);
        let board = Board::new().make_move(Square::D3);
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
                }) => panic!("{error:?}"),
                _ => {}
            }
            assert!(Instant::now() < deadline, "the stop was lost");
            thread::sleep(Duration::from_millis(5));
        };
        assert!(board.is_legal_move(outcome.best_move.unwrap()));
    }

    #[test]
    fn worker_panic_emits_worker_unavailable_error() {
        let (tx, rx) = mpsc::channel();
        let generation = Arc::new(AtomicU64::new(7));
        let guard_generation = generation.clone();
        let handle = thread::spawn(move || {
            let _guard = PanicGuard {
                events: tx,
                generation: guard_generation,
            };
            panic!("simulated worker panic");
        });
        assert!(handle.join().is_err());
        let event = rx.try_recv().expect("panic guard emits error");
        assert_eq!(event.id, 7);
        assert!(matches!(
            event.kind,
            EventKind::Error(EngineError::WorkerUnavailable)
        ));
    }
}
