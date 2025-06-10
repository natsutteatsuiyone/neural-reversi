//! Error types for the automatch crate.
//!
//! This module defines the error hierarchy used throughout the automatch
//! application, providing structured error handling for different failure modes.

use std::fmt;
use std::error::Error;
use std::io;

/// Comprehensive error type for automatch operations.
/// 
/// This enum covers all possible error conditions that can occur during
/// automated match execution, from I/O failures to engine communication errors.
#[derive(Debug)]
pub enum AutomatchError {
    /// I/O operation failed
    Io(io::Error),
    /// Engine communication or protocol error  
    Engine(String),
    /// Game logic or move validation error
    Game(String),
    /// Configuration validation error
    Config(String),
}

impl fmt::Display for AutomatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutomatchError::Io(err) => write!(f, "IO error: {}", err),
            AutomatchError::Engine(msg) => write!(f, "Engine error: {}", msg),
            AutomatchError::Game(msg) => write!(f, "Game error: {}", msg),
            AutomatchError::Config(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl Error for AutomatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            AutomatchError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for AutomatchError {
    fn from(err: io::Error) -> Self {
        AutomatchError::Io(err)
    }
}

impl From<String> for AutomatchError {
    fn from(msg: String) -> Self {
        AutomatchError::Game(msg)
    }
}

/// Convenience type alias for Results with AutomatchError.
/// 
/// This type alias simplifies function signatures throughout the crate
/// by providing a default Result type with AutomatchError as the error type.
/// 
/// # Examples
/// 
/// ```
/// # use automatch::error::Result;
/// fn might_fail() -> Result<String> {
///     Ok("success".to_string())
/// }
/// ```
pub type Result<T> = std::result::Result<T, AutomatchError>;