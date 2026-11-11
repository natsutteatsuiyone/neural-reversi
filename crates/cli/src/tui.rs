//! TUI (Terminal User Interface) module for Neural Reversi CLI.
//!
//! This module provides a full-featured terminal interface using ratatui,
//! supporting keyboard navigation, mouse input, and real-time game updates.

mod app;
mod event;
mod parse;
mod render;
mod search_worker;
mod widgets;

use crate::config::EngineConfig;

use app::App;

/// Runs the TUI, handling user input and game state.
pub fn run(config: &EngineConfig) -> Result<(), String> {
    let app = App::new(config);

    let terminal = match ratatui::try_init() {
        Ok(terminal) => terminal,
        Err(err) => {
            ratatui::restore();
            return Err(err.to_string());
        }
    };
    let result = crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)
        .and_then(|()| app.run(terminal));
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    ratatui::restore();

    result.map_err(|e| e.to_string())
}
