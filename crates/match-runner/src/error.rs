//! Error types for the match runner crate.
//!
//! This module defines the error hierarchy used throughout the match runner
//! application, providing structured error handling for different failure modes.

use std::error::Error;
use std::fmt;
use std::io;

/// Comprehensive error type for match runner operations.
///
/// This enum covers all possible error conditions that can occur during
/// automated match execution, from I/O failures to engine communication errors.
#[derive(Debug)]
pub enum MatchRunnerError {
    /// I/O operation failed
    Io(io::Error),
    /// Engine communication or protocol error  
    Engine(String),
    /// Game logic or move validation error
    Game(String),
    /// Configuration validation error
    Config(String),
}

impl fmt::Display for MatchRunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchRunnerError::Io(err) => write!(f, "IO error: {err}"),
            MatchRunnerError::Engine(msg) => write!(f, "Engine error: {msg}"),
            MatchRunnerError::Game(msg) => write!(f, "Game error: {msg}"),
            MatchRunnerError::Config(msg) => write!(f, "Configuration error: {msg}"),
        }
    }
}

impl Error for MatchRunnerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            MatchRunnerError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for MatchRunnerError {
    fn from(err: io::Error) -> Self {
        MatchRunnerError::Io(err)
    }
}

impl From<String> for MatchRunnerError {
    fn from(msg: String) -> Self {
        MatchRunnerError::Game(msg)
    }
}

/// Convenience type alias for Results with MatchRunnerError.
///
/// This type alias simplifies function signatures throughout the crate
/// by providing a default Result type with MatchRunnerError as the error type.
///
/// # Examples
///
/// ```
/// # use match_runner::error::Result;
/// fn might_fail() -> Result<String> {
///     Ok("success".to_string())
/// }
/// ```
pub type Result<T> = std::result::Result<T, MatchRunnerError>;
