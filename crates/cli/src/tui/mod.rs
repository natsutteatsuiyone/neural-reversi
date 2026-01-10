//! TUI (Terminal User Interface) module for Neural Reversi CLI.
//!
//! This module provides a full-featured terminal interface using ratatui,
//! supporting keyboard navigation, mouse input, and real-time game updates.

mod app;
mod event;
mod render;
mod widgets;

pub use app::App;
