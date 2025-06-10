//! Automatch - Automated tournament system for Reversi engines.
//!
//! This crate provides a complete framework for running automated matches between
//! GTP-compatible Reversi engines. It handles engine communication, game execution,
//! statistical analysis, and result visualization.

pub mod config;
pub mod display;
pub mod engine;
pub mod error;
pub mod game;
pub mod match_runner;
pub mod statistics;
