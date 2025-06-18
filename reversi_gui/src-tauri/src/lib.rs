use std::sync::{Arc, Mutex};

use reversi_core::level::get_level;
use reversi_core::piece::Piece;
use reversi_core::search::SearchOptions;
use reversi_core::types::{Scoref, Selectivity};
use reversi_core::{board, search};
use tauri::{AppHandle, Emitter, Manager, State};

use serde::Serialize;

struct AppState {
    search: Arc<Mutex<search::Search>>,
    thread_pool: Arc<search::threading::ThreadPool>,
}

#[derive(Serialize)]
pub struct AIMoveResult {
    pub best_move: Option<usize>,
    pub row: i32,
    pub col: i32,
    pub score: Scoref,
    pub depth: u32,
    pub acc: i32,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SearchProgressPayload {
    pub best_move: String,
    pub row: i32,
    pub col: i32,
    pub score: Scoref,
    pub depth: u32,
    pub acc: i32,
}

#[tauri::command]
fn init_ai_command(state: State<'_, AppState>) -> Result<(), String> {
    let search_arc = state.search.clone();

    match search_arc.lock() {
        Ok(mut search) => {
            search.init();
            Ok(())
        },
        Err(e) => Err(format!("Failed to initialize search: {}", e))
    }
}

#[tauri::command]
async fn abort_ai_search_command(state: State<'_, AppState>) -> Result<(), String> {
    state.thread_pool.abort_search();

    let thread_pool = state.thread_pool.clone();
    tauri::async_runtime::spawn_blocking(move || {
        thread_pool.wait_for_search_finished();
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
    selectivity: Selectivity,
) -> Result<AIMoveResult, String> {
    let search_arc = state.search.clone();
    let board_string_clone = board_string.clone();
    let level_clone = level;

    let result = tauri::async_runtime::spawn_blocking(move || {
        let board = board::Board::from_string(&board_string_clone, Piece::Black);
        let lv = get_level(level_clone);

        let mut search_guard = search_arc.lock().unwrap();
        let callback = move |progress: search::SearchProgress| {
            let payload = SearchProgressPayload {
                depth: progress.depth,
                score: (progress.score * 10.0).round() / 10.0,
                best_move: format!("{}", progress.best_move),
                row: (progress.best_move as i32 / 8),
                col: (progress.best_move as i32 % 8),
                acc: progress.probability,
            };
            app.emit("ai-move-progress", payload).unwrap();
        };
        let result = search_guard.run(&board, lv, selectivity, false, Some(callback));

        AIMoveResult {
            best_move: result.best_move.map(|square| square.index()),
            row: result.best_move.map(|square| square as i32 / 8).unwrap_or(-1),
            col: result.best_move.map(|square| square as i32 % 8).unwrap_or(-1),
            score: (result.score * 10.0).round() / 10.0,
            depth: result.depth,
            acc: result.get_probability(),
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
    selectivity: Selectivity,
) -> Result<(), String> {
    let search_arc = state.search.clone();
    let board_string_clone = board_string.clone();
    let level_clone = level;

    tauri::async_runtime::spawn_blocking(move || {
        let board = board::Board::from_string(&board_string_clone, Piece::Black);
        let lv = get_level(level_clone);

        let mut search_guard = search_arc.lock().unwrap();
        let callback = move |progress: search::SearchProgress| {
            let payload = SearchProgressPayload {
                depth: progress.depth,
                score: (progress.score * 10.0).round() / 10.0,
                best_move: format!("{}", progress.best_move),
                row: (progress.best_move as i32 / 8),
                col: (progress.best_move as i32 % 8),
                acc: progress.probability,
            };
            app.emit("ai-move-progress", payload).unwrap();
        };
        search_guard.run(&board, lv, selectivity, true, Some(callback));
    })
    .await
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    reversi_core::init();

    let search_options = SearchOptions::default();
    let search = Arc::new(Mutex::new(search::Search::new(&search_options)));

    let thread_pool = {
        let search_guard = search.lock().unwrap();
        search_guard.get_thread_pool()
    };

    tauri::Builder::default()
        .setup(|app| {
            app.manage(AppState {
                search,
                thread_pool,
            });
            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![ai_move_command, init_ai_command, abort_ai_search_command, analyze_command])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
