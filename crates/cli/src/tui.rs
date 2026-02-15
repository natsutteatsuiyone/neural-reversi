//! TUI (Terminal User Interface) module for Neural Reversi CLI.
//!
//! This module provides a full-featured terminal interface using ratatui,
//! supporting keyboard navigation, mouse input, and real-time game updates.

use std::path::Path;

use reversi_core::probcut::Selectivity;

mod app;
mod event;
mod parse;
mod render;
mod widgets;

use app::App;

/// Runs the TUI, handling user input and game state.
pub fn run(
    hash_size: usize,
    initial_level: usize,
    selectivity: Selectivity,
    threads: Option<usize>,
    eval_path: Option<&Path>,
    eval_sm_path: Option<&Path>,
) -> Result<(), String> {
    let app = App::new(
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
