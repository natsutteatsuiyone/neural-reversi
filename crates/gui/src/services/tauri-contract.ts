/**
 * Single source of truth for the Tauri IPC wire contract.
 *
 * Every command name passed to `invoke()` and every event name passed to
 * `listen()` lives here and nowhere else. The TS↔Rust contract has no
 * compile-time link, so concentrating the names in one module is what
 * keeps it auditable: this file is the one place to diff against
 * `src-tauri/src/lib.rs` (the `#[tauri::command]` fns and `app.emit(...)`
 * calls), and a backend rename is a one-line change here rather than a
 * silent runtime break scattered across the service classes.
 */

export const TAURI_COMMAND = {
  checkAiReady: "check_ai_ready_command",
  aiMove: "ai_move_command",
  initAi: "init_ai_command",
  resizeTt: "resize_tt_command",
  abortAiSearch: "abort_ai_search_command",
  analyze: "analyze_command",
  analyzeGame: "analyze_game_command",
  abortGameAnalysis: "abort_game_analysis_command",
  solverSearch: "solver_search_command",
  getAppVersion: "get_app_version",
  getLicenseText: "get_license_text",
  getThirdPartyLicensesText: "get_third_party_licenses_text",
} as const;

export const TAURI_EVENT = {
  aiMoveProgress: "ai-move-progress",
  solverProgress: "solver-progress",
  gameAnalysisProgress: "game-analysis-progress",
} as const;
