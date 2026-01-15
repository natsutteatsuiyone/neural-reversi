//! Interactive TUI (Terminal User Interface) for the reversi engine.
//!
//! This module provides a full-featured terminal interface using ratatui,
//! supporting keyboard navigation, mouse input, and real-time game updates.

use std::path::Path;

use reversi_core::probcut::Selectivity;

use crate::tui;

/// Main TUI loop.
///
/// Runs the terminal user interface, handling user input and game state.
/// Automatically triggers AI moves based on the current game mode.
///
/// # Arguments
/// * `hash_size` - Size of transposition table in MB
/// * `initial_level` - Initial AI search level
/// * `selectivity` - Search selectivity setting
/// * `threads` - Number of threads to use for search
/// * `eval_path` - Path to the main evaluation weights file
/// * `eval_sm_path` - Path to the small evaluation weights file
pub fn ui_loop(
    hash_size: usize,
    initial_level: usize,
    selectivity: Selectivity,
    threads: Option<usize>,
    eval_path: Option<&Path>,
    eval_sm_path: Option<&Path>,
) -> Result<(), String> {
    let app = tui::App::new(
        hash_size,
        initial_level,
        selectivity,
        threads,
        eval_path,
        eval_sm_path,
    )?;

    let terminal = ratatui::init();
    let result = app.run(terminal);
    ratatui::restore();

    result.map_err(|e| e.to_string())
}
