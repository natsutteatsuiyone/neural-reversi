//! TUI (Terminal User Interface) module for Neural Reversi CLI.
//!
//! This module provides a full-featured terminal interface using ratatui,
//! supporting keyboard navigation, mouse input, and real-time game updates.

mod app;
mod event;
mod parse;
mod render;
mod widgets;

use crate::config::EngineConfig;

use app::App;

/// Runs the TUI, handling user input and game state.
pub fn run(config: &EngineConfig) -> Result<(), String> {
    let app = App::new(
        config.hash_size,
        config.level,
        config.selectivity,
        config.threads,
        config.eval_file.as_deref(),
        config.eval_sm_file.as_deref(),
    )?;

    let terminal = ratatui::init();
    let result = app.run(terminal);
    ratatui::restore();

    result.map_err(|e| e.to_string())
}
