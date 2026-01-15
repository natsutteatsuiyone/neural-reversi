use std::sync::{Arc, Mutex};

use reversi_core::disc::Disc;
use reversi_core::level::get_level;
use reversi_core::probcut::Selectivity;
use reversi_core::search::options::SearchOptions;
use reversi_core::search::search_context::GamePhase;
use reversi_core::search::{SearchRunOptions, time_control::TimeControlMode};
use reversi_core::types::Scoref;
use reversi_core::{board, search};
use tauri::{AppHandle, Emitter, Manager, State};

use serde::Serialize;

const SELECTIVITY: Selectivity = Selectivity::Level1;

struct AppState {
    search: Arc<Mutex<search::Search>>,
    thread_pool: Arc<search::threading::ThreadPool>,
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

#[tauri::command]
fn init_ai_command(state: State<'_, AppState>) -> Result<(), String> {
    let search_arc = state.search.clone();

    match search_arc.lock() {
        Ok(mut search) => {
            search.init();
            Ok(())
        }
        Err(e) => Err(format!("Failed to initialize search: {e}")),
    }
}

#[tauri::command]
async fn abort_ai_search_command(state: State<'_, AppState>) -> Result<(), String> {
    state.thread_pool.abort_search();

    let thread_pool = state.thread_pool.clone();
    tauri::async_runtime::spawn_blocking(move || {
        thread_pool.wait_for_think_finished();
    })
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
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
    let board_string_clone = board_string.clone();
    let level_clone = level;

    let result = tauri::async_runtime::spawn_blocking(move || {
        let board = board::Board::from_string(&board_string_clone, Disc::Black);
        let lv = get_level(level_clone);
        let start_time = std::time::Instant::now();

        let mut search_guard = search_arc.lock().unwrap();
        let callback = move |progress: search::SearchProgress| {
            let payload = SearchProgressPayload {
                depth: progress.depth,
                target_depth: progress.target_depth,
                score: (progress.score * 10.0).round() / 10.0,
                best_move: format!("{}", progress.best_move),
                row: (progress.best_move as i32 / 8),
                col: (progress.best_move as i32 % 8),
                acc: progress.probability,
                nodes: progress.nodes,
                pv_line: progress
                    .pv_line
                    .iter()
                    .map(|sq| format!("{}", sq))
                    .collect::<Vec<_>>()
                    .join(" "),
                is_endgame: progress.game_phase == GamePhase::EndGame,
            };
            app.emit("ai-move-progress", payload).unwrap();
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

        AIMoveResult {
            best_move: result.best_move.map(|square| square.index()),
            row: result
                .best_move
                .map(|square| square as i32 / 8)
                .unwrap_or(-1),
            col: result
                .best_move
                .map(|square| square as i32 % 8)
                .unwrap_or(-1),
            score: (result.score * 10.0).round() / 10.0,
            depth: result.depth,
            acc: result.get_probability(),
            time_taken: time_taken_ms,
        }
    })
    .await
    .map_err(|e| e.to_string())?;

    Ok(result)
}

#[tauri::command]
async fn analyze_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    level: usize,
) -> Result<(), String> {
    let search_arc = state.search.clone();
    let board_string_clone = board_string.clone();
    let level_clone = level;

    tauri::async_runtime::spawn_blocking(move || {
        let board = board::Board::from_string(&board_string_clone, Disc::Black);
        let lv = get_level(level_clone);

        let mut search_guard = search_arc.lock().unwrap();
        let callback = move |progress: search::SearchProgress| {
            let payload = SearchProgressPayload {
                depth: progress.depth,
                target_depth: progress.target_depth,
                score: (progress.score * 10.0).round() / 10.0,
                best_move: format!("{}", progress.best_move),
                row: (progress.best_move as i32 / 8),
                col: (progress.best_move as i32 % 8),
                acc: progress.probability,
                nodes: progress.nodes,
                pv_line: progress
                    .pv_line
                    .iter()
                    .map(|sq| format!("{}", sq))
                    .collect::<Vec<_>>()
                    .join(" "),
                is_endgame: progress.game_phase == GamePhase::EndGame,
            };
            app.emit("ai-move-progress", payload).unwrap();
        };
        let options = SearchRunOptions::with_level(lv, SELECTIVITY)
            .multi_pv(true)
            .callback(callback);
        search_guard.run(&board, &options);
    })
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let search_options = SearchOptions::default();
    let search = Arc::new(Mutex::new(search::Search::new(&search_options)));

    let thread_pool = {
        let search_guard = search.lock().unwrap();
        search_guard.get_thread_pool()
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .setup(|app| {
            app.manage(AppState {
                search,
                thread_pool,
            });
            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            ai_move_command,
            init_ai_command,
            abort_ai_search_command,
            analyze_command
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
