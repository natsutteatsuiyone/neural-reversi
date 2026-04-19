use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};

use reversi_core::disc::Disc;
use reversi_core::level::get_level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::{SearchRunOptions, time_control::TimeControlMode};
use reversi_core::square::Square;
use reversi_core::types::Scoref;
use reversi_core::{board, search};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

const SELECTIVITY: Selectivity = Selectivity::Level1;

struct AppState {
    search: Arc<Mutex<search::Search>>,
    thread_pool: Arc<search::threading::ThreadPool>,
    game_analysis_run_id: Arc<AtomicU64>,
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

fn is_current_game_analysis_run(current_run_id: &AtomicU64, run_id: u64) -> bool {
    current_run_id.load(Ordering::Acquire) == run_id
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

/// Builds a [`reversi_core::level::Level`] whose endgame iterative-deepening
/// loop stops at `target`. Selectivities missing from
/// [`reversi_core::level::Level::ENDGAME_SELECTIVITY`] round **down** to the
/// next more-aggressive entry.
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
    let search_arc = state.search.clone();

    spawn_blocking_result(move || {
        lock_search(&search_arc)?.init();
        Ok(())
    })
    .await
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
    let search_arc = state.search.clone();

    spawn_blocking_result(move || {
        lock_search(&search_arc)?.resize_tt(hash_size);
        Ok(())
    })
    .await
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
    let search_arc = state.search.clone();

    spawn_blocking_result(move || {
        let board = board::Board::from_string(&board_string, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;
        let lv = get_level(level);
        let start_time = std::time::Instant::now();

        let mut search_guard = lock_search(&search_arc)?;
        let callback = move |progress: search::SearchProgress| {
            let _ = app.emit("ai-move-progress", build_progress_payload(&progress));
        };

        let result = if let Some(remaining_ms) = remaining_time {
            let options = SearchRunOptions::with_time(
                TimeControlMode::Fischer {
                    main_time_ms: remaining_ms,
                    increment_ms: 0,
                },
                SELECTIVITY,
            )
            .callback(callback);
            search_guard.run(&board, &options)
        } else if let Some(limit_ms) = time_limit {
            let options = SearchRunOptions::with_time(
                TimeControlMode::Byoyomi {
                    time_per_move_ms: limit_ms,
                },
                SELECTIVITY,
            )
            .callback(callback);
            search_guard.run(&board, &options)
        } else {
            let options = SearchRunOptions::with_level(lv, SELECTIVITY).callback(callback);
            search_guard.run(&board, &options)
        };

        let time_taken_ms = start_time.elapsed().as_millis() as u64;

        Ok(AIMoveResult {
            best_move: result.best_move.map(|square| square.index()),
            row: result
                .best_move
                .map(|square| square as i32 / 8)
                .unwrap_or(-1),
            col: result
                .best_move
                .map(|square| square as i32 % 8)
                .unwrap_or(-1),
            score: round_score(result.score),
            depth: result.depth,
            acc: result.get_probability(),
            time_taken: time_taken_ms,
        })
    })
    .await
}

#[tauri::command]
async fn analyze_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    level: usize,
) -> Result<(), String> {
    let search_arc = state.search.clone();

    spawn_blocking_result(move || {
        let board = board::Board::from_string(&board_string, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;
        let lv = get_level(level);

        let mut search_guard = lock_search(&search_arc)?;
        let callback = move |progress: search::SearchProgress| {
            let _ = app.emit("ai-move-progress", build_progress_payload(&progress));
        };
        let options = SearchRunOptions::with_level(lv, SELECTIVITY)
            .multi_pv(true)
            .callback(callback);
        search_guard.run(&board, &options);
        Ok(())
    })
    .await
}

#[tauri::command]
async fn solver_search_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    target_selectivity: u8,
    run_id: u64,
) -> Result<(), String> {
    let search_arc = state.search.clone();

    spawn_blocking_result(move || {
        if target_selectivity > 5 {
            return Err(format!(
                "Invalid target_selectivity: {target_selectivity} (expected 0..=5)"
            ));
        }

        let board = board::Board::from_string(&board_string, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;

        let selectivity = Selectivity::from_u8(target_selectivity);
        let level = solver_level(selectivity);

        let mut search_guard = lock_search(&search_arc)?;
        let callback = move |progress: search::SearchProgress| {
            let _ = app.emit(
                "solver-progress",
                SolverProgressPayload {
                    run_id,
                    progress: build_progress_payload(&progress),
                },
            );
        };

        let options = SearchRunOptions::with_level(level, selectivity)
            .multi_pv(true)
            .callback(callback);
        search_guard.run(&board, &options);
        Ok(())
    })
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
    // Claim a unique run id. Any later call bumps the counter, making our
    // guard checks observe a mismatch and this run bail.
    let run_id = state
        .game_analysis_run_id
        .fetch_add(1, Ordering::AcqRel)
        .wrapping_add(1);
    let search_arc = state.search.clone();
    let current_run_id = state.game_analysis_run_id.clone();

    spawn_blocking_result(move || {
        let initial_board = board::Board::from_string(&board_string, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;
        let lv = get_level(level);

        let mut boards: Vec<board::Board> = Vec::with_capacity(moves.len() + 1);
        let mut is_pass: Vec<bool> = Vec::with_capacity(moves.len());
        let mut current = initial_board;
        boards.push(current);

        for move_notation in &moves {
            if move_notation == "--" {
                is_pass.push(true);
                current = current.switch_players();
            } else {
                is_pass.push(false);
                let sq: Square = move_notation
                    .parse()
                    .map_err(|e| format!("Invalid move notation '{move_notation}': {e}"))?;
                current = current.make_move(sq);
            }
            boards.push(current);
        }

        if !is_current_game_analysis_run(&current_run_id, run_id) {
            return Ok(());
        }

        let options = SearchRunOptions::with_level(lv, SELECTIVITY);
        let final_board = boards.last().unwrap();
        let mut score: Scoref = if !final_board.get_moves().is_empty() {
            let mut guard = lock_search(&search_arc)?;
            let result = guard.run(final_board, &options);
            drop(guard);

            if !is_current_game_analysis_run(&current_run_id, run_id) {
                return Ok(());
            }

            round_score(result.score)
        } else {
            round_score(final_board.solve(final_board.get_empty_count()) as Scoref)
        };

        for i in (0..moves.len()).rev() {
            if !is_current_game_analysis_run(&current_run_id, run_id) {
                return Ok(());
            }

            if is_pass[i] {
                score = -score;
                continue;
            }

            let board = &boards[i];
            let mut search_guard = lock_search(&search_arc)?;
            let result = search_guard.run(board, &options);
            drop(search_guard);

            if !is_current_game_analysis_run(&current_run_id, run_id) {
                return Ok(());
            }

            let best_move_str = result.best_move.map(|s| s.to_string()).unwrap_or_default();
            let best_score = round_score(result.score);

            let played_score = -score;
            let score_loss = (best_score - played_score).max(0.0);

            let payload = GameAnalysisProgressPayload {
                move_index: i,
                best_move: best_move_str,
                best_score,
                played_score,
                score_loss,
                depth: result.depth,
            };
            let _ = app.emit("game-analysis-progress", payload);

            score = best_score.max(played_score);
        }

        Ok(())
    })
    .await
}

#[tauri::command]
async fn abort_game_analysis_command(state: State<'_, AppState>) -> Result<(), String> {
    // Bumping the counter makes any in-flight run observe a mismatch and exit.
    state.game_analysis_run_id.fetch_add(1, Ordering::AcqRel);
    abort_and_wait(state.thread_pool.clone()).await
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
                game_analysis_run_id: Arc::new(AtomicU64::new(0)),
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
    fn solver_level_level5_caps_last_entry() {
        let lvl = solver_level(Selectivity::Level5);
        assert_eq!(lvl.end_depth, [60, 60, 60, 0]);
    }

    #[test]
    fn solver_level_level3_caps_after_level3() {
        let lvl = solver_level(Selectivity::Level3);
        assert_eq!(lvl.end_depth, [60, 60, 0, 0]);
    }

    #[test]
    fn solver_level_level1_only_first() {
        let lvl = solver_level(Selectivity::Level1);
        assert_eq!(lvl.end_depth, [60, 0, 0, 0]);
    }

    #[test]
    fn solver_level_level2_rounds_down_to_level1() {
        let lvl = solver_level(Selectivity::Level2);
        assert_eq!(lvl.end_depth, [60, 0, 0, 0]);
    }

    #[test]
    fn solver_level_level4_rounds_down_to_level3() {
        let lvl = solver_level(Selectivity::Level4);
        assert_eq!(lvl.end_depth, [60, 60, 0, 0]);
    }
}
