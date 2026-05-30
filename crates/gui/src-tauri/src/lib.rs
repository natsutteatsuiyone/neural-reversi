use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};

use reversi_core::disc::Disc;
use reversi_core::level::get_level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::search_result::SearchResult;
use reversi_core::search::{SearchRunOptions, time_control::TimeControlMode};
use reversi_core::square::Square;
use reversi_core::types::Scoref;
use reversi_core::{board, search};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

mod game_analysis;

const SELECTIVITY: Selectivity = Selectivity::Level1;

/// The current game-analysis generation (CONTEXT.md → Engine Search).
///
/// A monotonically increasing counter behind one interface: a run
/// `claim()`s a generation, checks `is_current()` at each await-point to
/// bail when superseded, and an abort `supersede()`s it. The atomic
/// orderings and the wrap are owned here so they cannot drift between the
/// six call sites that previously open-coded them.
struct GameAnalysisGeneration(AtomicU64);

impl GameAnalysisGeneration {
    fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Claim a fresh generation. Any later claim or supersede makes this
    /// one observe a mismatch in `is_current`.
    fn claim(&self) -> u64 {
        self.0.fetch_add(1, Ordering::AcqRel).wrapping_add(1)
    }

    /// Whether `generation` is still the latest claimed generation.
    fn is_current(&self, generation: u64) -> bool {
        self.0.load(Ordering::Acquire) == generation
    }

    /// Supersede any in-flight run without claiming a new generation.
    fn supersede(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

struct AppState {
    search: Arc<Mutex<search::Search>>,
    thread_pool: Arc<search::threading::ThreadPool>,
    game_analysis_run_id: Arc<GameAnalysisGeneration>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AIMoveResult {
    pub best_move: Option<usize>,
    pub row: i32,
    pub col: i32,
    pub score: Scoref,
    pub depth: u32,
    pub acc: i32,
    pub time_taken: u64,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SearchProgressPayload {
    pub best_move: String,
    pub row: i32,
    pub col: i32,
    pub score: Scoref,
    pub depth: u32,
    pub target_depth: u32,
    pub acc: i32,
    pub nodes: u64,
    pub pv_line: String,
    pub is_endgame: bool,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct SolverProgressPayload {
    run_id: u64,
    #[serde(flatten)]
    progress: SearchProgressPayload,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GameAnalysisProgressPayload {
    pub move_index: usize,
    pub best_move: String,
    pub best_score: Scoref,
    pub played_score: Scoref,
    pub score_loss: Scoref,
    pub depth: u32,
}

fn round_score(score: Scoref) -> Scoref {
    (score * 10.0).round() / 10.0
}

fn build_progress_payload(progress: &search::SearchProgress) -> SearchProgressPayload {
    SearchProgressPayload {
        depth: progress.depth,
        target_depth: progress.target_depth,
        score: round_score(progress.score),
        best_move: format!("{}", progress.best_move),
        row: progress.best_move as i32 / 8,
        col: progress.best_move as i32 % 8,
        acc: progress.probability,
        nodes: progress.nodes,
        pv_line: progress
            .pv_line
            .iter()
            .map(|sq| format!("{}", sq))
            .collect::<Vec<_>>()
            .join(" "),
        is_endgame: progress.is_endgame,
    }
}

fn build_game_analysis(result: SearchResult) -> Result<game_analysis::Analysis, String> {
    match (result.best_move(), result.score()) {
        (Some(best_move), Some(score)) => Ok(game_analysis::Analysis {
            best_move,
            score: round_score(score),
            depth: result.depth(),
        }),
        _ => Err("search returned no legal move for game analysis position".to_string()),
    }
}

fn decode_game_analysis_moves(
    moves: Vec<String>,
) -> Result<Vec<game_analysis::GameAnalysisMove>, String> {
    moves
        .into_iter()
        .map(|move_notation| {
            if move_notation == "--" {
                Ok(game_analysis::GameAnalysisMove::Pass)
            } else {
                let square: Square = move_notation
                    .parse()
                    .map_err(|e| format!("Invalid move notation '{move_notation}': {e}"))?;
                Ok(game_analysis::GameAnalysisMove::Play(square))
            }
        })
        .collect()
}

fn lock_search(
    search: &Arc<Mutex<search::Search>>,
) -> Result<std::sync::MutexGuard<'_, search::Search>, String> {
    search
        .lock()
        .map_err(|e| format!("AI backend unavailable: {e}"))
}

async fn spawn_blocking_result<T, F>(f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| e.to_string())?
}

/// Runs one search on the single shared engine (CONTEXT.md → Engine Search):
/// parse the board, take the engine lock, run with the caller's options, and
/// hand the result + elapsed time to `map_result` — all inside one
/// `spawn_blocking`. Concentrates the clone / spawn_blocking / parse / lock
/// scaffold every engine-search command otherwise repeats; callers only
/// describe the search (options) and translate the result.
///
/// `build_options` and `map_result` run on the blocking thread, so the
/// `SearchResult` never crosses the task boundary (matching the prior
/// per-command code).
async fn run_engine_search<R, B, M>(
    search: Arc<Mutex<search::Search>>,
    board_string: String,
    build_options: B,
    map_result: M,
) -> Result<R, String>
where
    R: Send + 'static,
    B: FnOnce() -> SearchRunOptions + Send + 'static,
    M: FnOnce(&SearchResult, u64) -> R + Send + 'static,
{
    spawn_blocking_result(move || {
        let board = board::Board::from_string(&board_string, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;
        let start_time = std::time::Instant::now();
        let mut search_guard = lock_search(&search)?;
        let options = build_options();
        let result = search_guard.run(&board, &options);
        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        Ok(map_result(&result, elapsed_ms))
    })
    .await
}

/// Takes the engine lock on a blocking thread and applies `f`. The scaffold
/// shared by the non-search engine commands (`init`, `resize_tt`).
async fn with_search_lock<F>(search: Arc<Mutex<search::Search>>, f: F) -> Result<(), String>
where
    F: FnOnce(&mut search::Search) + Send + 'static,
{
    spawn_blocking_result(move || {
        let mut guard = lock_search(&search)?;
        f(&mut guard);
        Ok(())
    })
    .await
}

/// Builds a [`reversi_core::level::Level`] whose endgame iterative-deepening
/// loop stops at `target`.
fn solver_level(target: Selectivity) -> reversi_core::level::Level {
    let mut end_depth = [0u32; 4];
    for (i, &sel) in reversi_core::level::Level::ENDGAME_SELECTIVITY
        .iter()
        .enumerate()
    {
        if sel <= target {
            end_depth[i] = 60;
        }
    }
    reversi_core::level::Level {
        mid_depth: 60,
        end_depth,
    }
}

#[tauri::command]
async fn init_ai_command(state: State<'_, AppState>) -> Result<(), String> {
    with_search_lock(state.search.clone(), |s| s.init()).await
}

#[tauri::command]
async fn check_ai_ready_command(state: State<'_, AppState>) -> Result<(), String> {
    match state.search.try_lock() {
        Ok(_search) => Ok(()),
        Err(TryLockError::WouldBlock) => Ok(()),
        Err(TryLockError::Poisoned(e)) => Err(format!("AI backend is unavailable: {e}")),
    }
}

#[tauri::command]
async fn resize_tt_command(state: State<'_, AppState>, hash_size: usize) -> Result<(), String> {
    let hash_size = hash_size.clamp(1, 16384);
    with_search_lock(state.search.clone(), move |s| s.resize_tt(hash_size)).await
}

async fn abort_and_wait(thread_pool: Arc<search::threading::ThreadPool>) -> Result<(), String> {
    thread_pool.abort_search();
    tauri::async_runtime::spawn_blocking(move || {
        thread_pool.wait_for_think_finished();
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn abort_ai_search_command(state: State<'_, AppState>) -> Result<(), String> {
    abort_and_wait(state.thread_pool.clone()).await
}

#[tauri::command]
async fn ai_move_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    level: usize,
    time_limit: Option<u64>,
    remaining_time: Option<u64>,
) -> Result<AIMoveResult, String> {
    run_engine_search(
        state.search.clone(),
        board_string,
        move || {
            let callback = move |progress: search::SearchProgress| {
                let _ = app.emit("ai-move-progress", build_progress_payload(&progress));
            };
            if let Some(remaining_ms) = remaining_time {
                SearchRunOptions::with_time(
                    TimeControlMode::Fischer {
                        main_time_ms: remaining_ms,
                        increment_ms: 0,
                    },
                    SELECTIVITY,
                )
                .callback(callback)
            } else if let Some(limit_ms) = time_limit {
                SearchRunOptions::with_time(
                    TimeControlMode::Byoyomi {
                        time_per_move_ms: limit_ms,
                    },
                    SELECTIVITY,
                )
                .callback(callback)
            } else {
                SearchRunOptions::with_level(get_level(level), SELECTIVITY).callback(callback)
            }
        },
        |result, elapsed_ms| {
            let best_move = result.best_move();
            AIMoveResult {
                best_move: best_move.map(|square| square.index()),
                row: best_move.map(|square| square as i32 / 8).unwrap_or(-1),
                col: best_move.map(|square| square as i32 % 8).unwrap_or(-1),
                score: round_score(result.score().unwrap_or(0.0)),
                depth: result.depth(),
                acc: result.get_probability(),
                time_taken: elapsed_ms,
            }
        },
    )
    .await
}

#[tauri::command]
async fn analyze_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    level: usize,
) -> Result<(), String> {
    run_engine_search(
        state.search.clone(),
        board_string,
        move || {
            let callback = move |progress: search::SearchProgress| {
                let _ = app.emit("ai-move-progress", build_progress_payload(&progress));
            };
            SearchRunOptions::with_level(get_level(level), SELECTIVITY)
                .multi_pv(true)
                .callback(callback)
        },
        |_result, _elapsed_ms| (),
    )
    .await
}

#[tauri::command]
async fn solver_search_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    target_selectivity: u8,
    multi_pv: bool,
    run_id: u64,
) -> Result<(), String> {
    if target_selectivity > 3 {
        return Err(format!(
            "Invalid target_selectivity: {target_selectivity} (expected 0..=3)"
        ));
    }

    run_engine_search(
        state.search.clone(),
        board_string,
        move || {
            let selectivity = Selectivity::from_u8(target_selectivity);
            let level = solver_level(selectivity);
            let callback = move |progress: search::SearchProgress| {
                let _ = app.emit(
                    "solver-progress",
                    SolverProgressPayload {
                        run_id,
                        progress: build_progress_payload(&progress),
                    },
                );
            };
            SearchRunOptions::with_level(level, selectivity)
                .multi_pv(multi_pv)
                .callback(callback)
        },
        |_result, _elapsed_ms| (),
    )
    .await
}

#[tauri::command]
async fn analyze_game_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    moves: Vec<String>,
    level: usize,
) -> Result<(), String> {
    // Claim a unique run id. Any later claim/supersede makes the injected
    // `is_cancelled` predicate observe a mismatch and this run bail.
    let run_id = state.game_analysis_run_id.claim();
    let search_arc = state.search.clone();
    let current_run_id = state.game_analysis_run_id.clone();

    spawn_blocking_result(move || {
        let initial = board::Board::from_string(&board_string, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;
        let moves = decode_game_analysis_moves(moves)?;
        let options = SearchRunOptions::with_level(get_level(level), SELECTIVITY);

        game_analysis::analyze_game(
            initial,
            &moves,
            // Engine seam: lock per Position (never held across the loop), run,
            // and narrow the SearchResult to the data the analysis needs.
            |board| {
                let mut guard = lock_search(&search_arc)?;
                let result = guard.run(board, &options);
                drop(guard);
                build_game_analysis(result)
            },
            || !current_run_id.is_current(run_id),
            |progress| {
                let _ = app.emit(
                    "game-analysis-progress",
                    GameAnalysisProgressPayload {
                        move_index: progress.move_index,
                        best_move: progress.best_move.to_string(),
                        best_score: progress.best_score,
                        played_score: progress.played_score,
                        score_loss: progress.score_loss,
                        depth: progress.depth,
                    },
                );
            },
        )
    })
    .await
}

#[tauri::command]
async fn abort_game_analysis_command(state: State<'_, AppState>) -> Result<(), String> {
    // Superseding makes any in-flight run observe a mismatch and exit.
    state.game_analysis_run_id.supersede();
    abort_and_wait(state.thread_pool.clone()).await
}

#[tauri::command]
fn get_app_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[tauri::command]
fn get_license_text() -> String {
    const NOTICE: &str = include_str!("../../../../NOTICE");
    const LICENSE: &str = include_str!("../../../../LICENSE");
    format!("{NOTICE}\n{LICENSE}")
}

#[tauri::command]
fn get_third_party_licenses_text() -> &'static str {
    include_str!("../THIRD_PARTY_LICENSES.txt")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let search_options = SearchOptions::default();
    let search = Arc::new(Mutex::new(search::Search::new(&search_options)));

    let thread_pool = {
        let search_guard = search.lock().unwrap();
        search_guard.thread_pool()
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .setup(|app| {
            app.manage(AppState {
                search,
                thread_pool,
                game_analysis_run_id: Arc::new(GameAnalysisGeneration::new()),
            });
            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            ai_move_command,
            check_ai_ready_command,
            init_ai_command,
            resize_tt_command,
            abort_ai_search_command,
            analyze_command,
            analyze_game_command,
            abort_game_analysis_command,
            solver_search_command,
            get_app_version,
            get_license_text,
            get_third_party_licenses_text,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_level_none_equals_perfect() {
        let lvl = solver_level(Selectivity::None);
        assert_eq!(lvl.mid_depth, 60);
        assert_eq!(lvl.end_depth, [60, 60, 60, 60]);
    }

    #[test]
    fn solver_level_level3_caps_last_entry() {
        let lvl = solver_level(Selectivity::Level3);
        assert_eq!(lvl.end_depth, [60, 60, 60, 0]);
    }

    #[test]
    fn solver_level_level2_caps_after_level2() {
        let lvl = solver_level(Selectivity::Level2);
        assert_eq!(lvl.end_depth, [60, 60, 0, 0]);
    }

    #[test]
    fn solver_level_level1_only_first() {
        let lvl = solver_level(Selectivity::Level1);
        assert_eq!(lvl.end_depth, [60, 0, 0, 0]);
    }

    #[test]
    fn decode_game_analysis_moves_converts_play_and_pass() {
        let moves = decode_game_analysis_moves(vec!["d3".to_string(), "--".to_string()]).unwrap();
        assert_eq!(
            moves,
            vec![
                game_analysis::GameAnalysisMove::Play(Square::D3),
                game_analysis::GameAnalysisMove::Pass
            ]
        );
    }

    #[test]
    fn decode_game_analysis_moves_rejects_invalid_notation() {
        let err = decode_game_analysis_moves(vec!["zz".to_string()]).unwrap_err();
        assert!(err.contains("Invalid move notation"), "got: {err}");
    }
}
