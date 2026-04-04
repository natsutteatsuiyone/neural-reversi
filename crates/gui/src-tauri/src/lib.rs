use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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
    game_analysis_abort: Arc<AtomicBool>,
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

#[tauri::command]
async fn init_ai_command(state: State<'_, AppState>) -> Result<(), String> {
    let search_arc = state.search.clone();

    tauri::async_runtime::spawn_blocking(move || match search_arc.lock() {
        Ok(mut search) => {
            search.init();
            Ok(())
        }
        Err(e) => Err(format!("Failed to initialize search: {e}")),
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn resize_tt_command(state: State<'_, AppState>, hash_size: usize) -> Result<(), String> {
    let hash_size = hash_size.clamp(1, 16384);
    let search_arc = state.search.clone();

    tauri::async_runtime::spawn_blocking(move || match search_arc.lock() {
        Ok(mut search) => {
            search.resize_tt(hash_size);
            Ok(())
        }
        Err(e) => Err(format!("Failed to resize TT: {e}")),
    })
    .await
    .map_err(|e| e.to_string())?
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
    let board_string_clone = board_string.clone();
    let level_clone = level;

    tauri::async_runtime::spawn_blocking(move || -> Result<AIMoveResult, String> {
        let board = board::Board::from_string(&board_string_clone, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;
        let lv = get_level(level_clone);
        let start_time = std::time::Instant::now();

        let mut search_guard = search_arc.lock().unwrap();
        let callback = move |progress: search::SearchProgress| {
            let payload = SearchProgressPayload {
                depth: progress.depth,
                target_depth: progress.target_depth,
                score: round_score(progress.score),
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
                is_endgame: progress.is_endgame,
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
    .map_err(|e| e.to_string())?
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

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let board = board::Board::from_string(&board_string_clone, Disc::Black)
            .map_err(|e| format!("Invalid board string: {e}"))?;
        let lv = get_level(level_clone);

        let mut search_guard = search_arc.lock().unwrap();
        let callback = move |progress: search::SearchProgress| {
            let payload = SearchProgressPayload {
                depth: progress.depth,
                target_depth: progress.target_depth,
                score: round_score(progress.score),
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
                is_endgame: progress.is_endgame,
            };
            app.emit("ai-move-progress", payload).unwrap();
        };
        let options = SearchRunOptions::with_level(lv, SELECTIVITY)
            .multi_pv(true)
            .callback(callback);
        search_guard.run(&board, &options);
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn analyze_game_command(
    state: State<'_, AppState>,
    app: AppHandle,
    board_string: String,
    moves: Vec<String>,
    level: usize,
) -> Result<(), String> {
    let search_arc = state.search.clone();
    let abort_flag = state.game_analysis_abort.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        abort_flag.store(false, Ordering::Release);
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

        if abort_flag.load(Ordering::Acquire) {
            return Ok(());
        }

        let options = SearchRunOptions::with_level(lv, SELECTIVITY);
        let final_board = boards.last().unwrap();
        let mut score: Scoref = if !final_board.get_moves().is_empty() {
            let mut guard = search_arc.lock().unwrap();
            let result = guard.run(final_board, &options);
            round_score(result.score)
        } else {
            final_board.solve(final_board.get_empty_count()) as Scoref
        };

        for i in (0..moves.len()).rev() {
            if abort_flag.load(Ordering::Acquire) {
                break;
            }

            if is_pass[i] {
                score = -score;
                continue;
            }

            let board = &boards[i];
            let mut search_guard = search_arc.lock().unwrap();
            let result = search_guard.run(board, &options);
            drop(search_guard);

            let best_move_str = result
                .best_move
                .map(|s| s.to_string())
                .unwrap_or_default();
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
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn abort_game_analysis_command(state: State<'_, AppState>) -> Result<(), String> {
    state.game_analysis_abort.store(true, Ordering::Release);
    abort_and_wait(state.thread_pool.clone()).await
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
                game_analysis_abort: Arc::new(AtomicBool::new(false)),
            });
            Ok(())
        })
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            ai_move_command,
            init_ai_command,
            resize_tt_command,
            abort_ai_search_command,
            analyze_command,
            analyze_game_command,
            abort_game_analysis_command,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
