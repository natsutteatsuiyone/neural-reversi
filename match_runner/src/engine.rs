//! GTP (Go Text Protocol) engine communication.
//!
//! This module provides functionality for communicating with external Reversi engines
//! that implement the GTP protocol. It handles process management, command sending,
//! and response parsing.

use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

use crate::error::{MatchRunnerError, Result};

// GTP protocol constants
const GTP_SUCCESS_PREFIX: &str = "= ";
const GTP_FAILURE_PREFIX: &str = "? ";

// GTP commands
const GTP_CMD_NAME: &str = "name";
const GTP_CMD_VERSION: &str = "version";
const GTP_CMD_CLEAR_BOARD: &str = "clear_board";
const GTP_CMD_PLAY: &str = "play";
const GTP_CMD_GENMOVE: &str = "genmove";
const GTP_CMD_QUIT: &str = "quit";

// Error messages
const ERR_STDIN_FAILED: &str = "Failed to open stdin";
const ERR_STDOUT_FAILED: &str = "Failed to open stdout";
const ERR_PROCESS_CLOSED: &str = "Process closed stdout";

/// A GTP-compatible Reversi engine process.
///
/// This struct manages communication with an external Reversi engine process
/// that implements the GTP (Go Text Protocol). It handles starting the process,
/// sending commands, and receiving responses.
pub struct GtpEngine {
    process: Child,
    /// Engine name
    name: String,
    /// Engine version
    version: String,
}

impl GtpEngine {
    // =============================================================================
    // Initialization
    // =============================================================================

    /// Create a new GTP engine instance.
    ///
    /// Starts the engine process with the specified executable, arguments, and working directory.
    /// The process is configured with piped stdin, stdout, and stderr for communication.
    /// Immediately queries the engine for its name and version to cache these values.
    ///
    /// # Arguments
    ///
    /// * `executable` - Path to the engine executable
    /// * `args` - Command-line arguments for the engine
    /// * `working_dir` - Optional working directory for the engine process
    ///
    /// # Returns
    ///
    /// A new `GtpEngine` instance ready for communication.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine process cannot be started or if initial
    /// communication with the engine fails.
    pub fn new(executable: &str, args: &[String], working_dir: Option<PathBuf>) -> Result<Self> {
        let exec_path = Path::new(executable);
        let default_working_dir = if let Some(parent) = exec_path.parent() {
            parent.to_path_buf()
        } else {
            PathBuf::from(".")
        };

        let working_dir = working_dir.unwrap_or(default_working_dir);

        let mut process = Command::new(executable)
            .args(args)
            .current_dir(&working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        // Get name and version
        let (name, version) = Self::get_name_and_version(&mut process)?;

        Ok(GtpEngine {
            process,
            name,
            version,
        })
    }

    /// Get the engine's name and version from GTP commands.
    ///
    /// Queries the engine for its name and version and returns them.
    /// This is called once during engine initialization.
    fn get_name_and_version(process: &mut Child) -> Result<(String, String)> {
        // Get name
        let name_response = Self::send_command_to_process(process, GTP_CMD_NAME)?;
        let name = Self::parse_success_response(&name_response)?;

        // Get version
        let version_response = Self::send_command_to_process(process, GTP_CMD_VERSION)?;
        let version_raw = Self::parse_optional_response(&version_response)?;
        let version = Self::format_version(&version_raw);

        Ok((name, version))
    }

    // =============================================================================
    // Core Communication
    // =============================================================================

    /// Core GTP communication logic.
    fn communicate_with_process(
        stdin: &mut dyn Write,
        stdout: &mut dyn BufRead,
        command: &str,
    ) -> Result<String> {
        writeln!(stdin, "{command}")?;
        stdin.flush()?;

        let mut response = String::new();
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = stdout.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(MatchRunnerError::Engine(ERR_PROCESS_CLOSED.to_string()));
            }

            if line.trim().is_empty() {
                break;
            }

            response.push_str(&line);
        }

        Ok(response)
    }

    /// Send a GTP command to a process and wait for response.
    fn send_command_to_process(process: &mut Child, command: &str) -> Result<String> {
        let stdin = process
            .stdin
            .as_mut()
            .ok_or_else(|| MatchRunnerError::Engine(ERR_STDIN_FAILED.to_string()))?;

        let stdout = process
            .stdout
            .as_mut()
            .ok_or_else(|| MatchRunnerError::Engine(ERR_STDOUT_FAILED.to_string()))?;

        let mut reader = BufReader::new(stdout);
        Self::communicate_with_process(stdin, &mut reader, command)
    }

    /// Send a GTP command to the engine and wait for response.
    ///
    /// Sends the command string to the engine's stdin and reads the response
    /// from stdout until an empty line is encountered (GTP protocol standard).
    ///
    /// # Arguments
    ///
    /// * `command` - The GTP command to send (e.g., "genmove black")
    ///
    /// # Returns
    ///
    /// The engine's response as a string.
    ///
    /// # Errors
    ///
    /// Returns an error if communication with the engine fails or if the
    /// engine process terminates unexpectedly.
    pub fn send_command(&mut self, command: &str) -> Result<String> {
        Self::send_command_to_process(&mut self.process, command)
    }

    // =============================================================================
    // Engine Information
    // =============================================================================

    /// Get the engine's name and version.
    ///
    /// Returns the cached name and version that were retrieved during
    /// engine initialization.
    ///
    /// # Returns
    ///
    /// A formatted string containing the engine name and version.
    pub fn name(&self) -> String {
        if !self.version.is_empty() {
            format!("{} {}", self.name, self.version)
        } else {
            self.name.clone()
        }
    }

    // =============================================================================
    // Game Control
    // =============================================================================

    /// Clear the engine's internal board state.
    ///
    /// Sends the "clear_board" GTP command to reset the engine to an
    /// initial empty board state.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine doesn't accept the clear_board command.
    pub fn clear_board(&mut self) -> Result<()> {
        let response = self.send_command(GTP_CMD_CLEAR_BOARD)?;
        Self::parse_success_response(&response)?;
        Ok(())
    }

    /// Make a move on the engine's internal board.
    ///
    /// Sends a "play" GTP command to inform the engine of a move made
    /// by the opponent or during opening setup.
    ///
    /// # Arguments
    ///
    /// * `color` - The color making the move ("black" or "white")
    /// * `mv` - The move in algebraic notation (e.g., "f5" or "pass")
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine rejects the move or if communication fails.
    pub fn play(&mut self, color: &str, mv: &str) -> Result<()> {
        let response = self.send_command(&format!("{GTP_CMD_PLAY} {color} {mv}"))?;
        Self::parse_success_response(&response)?;
        Ok(())
    }

    /// Request the engine to generate a move.
    ///
    /// Sends a "genmove" GTP command asking the engine to choose and
    /// return its best move for the specified color.
    ///
    /// # Arguments
    ///
    /// * `color` - The color for which to generate a move ("black" or "white")
    ///
    /// # Returns
    ///
    /// The engine's chosen move in algebraic notation (e.g., "f5" or "pass").
    ///
    /// # Errors
    ///
    /// Returns an error if the engine fails to generate a move or if
    /// communication fails.
    pub fn genmove(&mut self, color: &str) -> Result<String> {
        let response = self.send_command(&format!("{GTP_CMD_GENMOVE} {color}"))?;
        Self::parse_success_response(&response)
    }

    // =============================================================================
    // Engine Management
    // =============================================================================

    /// Send a quit command to the engine.
    ///
    /// Attempts to gracefully shut down the engine by sending the "quit"
    /// GTP command. Note that this method doesn't wait for the process to
    /// actually terminate.
    ///
    /// # Returns
    ///
    /// `Ok(())` regardless of whether the engine responds to the quit command.
    pub fn quit(&mut self) -> Result<()> {
        let _ = self.send_command(GTP_CMD_QUIT);
        Ok(())
    }

    // =============================================================================
    // Helper Methods
    // =============================================================================

    /// Parse a GTP success response and extract the content.
    fn parse_success_response(response: &str) -> Result<String> {
        if response.starts_with(GTP_SUCCESS_PREFIX) {
            Ok(response
                .strip_prefix(GTP_SUCCESS_PREFIX)
                .unwrap()
                .trim()
                .to_string())
        } else if response.trim() == "=" {
            // Handle empty success response (just "=")
            Ok(String::new())
        } else {
            Err(MatchRunnerError::Engine(format!(
                "Expected success response, got: {response}"
            )))
        }
    }

    /// Parse a GTP response that might be unsupported (returns empty string for "?" responses).
    fn parse_optional_response(response: &str) -> Result<String> {
        if response.starts_with(GTP_SUCCESS_PREFIX) {
            Ok(response
                .strip_prefix(GTP_SUCCESS_PREFIX)
                .unwrap()
                .trim()
                .to_string())
        } else if response.trim() == "=" {
            // Handle empty success response (just "=")
            Ok(String::new())
        } else if response.starts_with(GTP_FAILURE_PREFIX) {
            Ok(String::new())
        } else {
            Err(MatchRunnerError::Engine(format!(
                "Invalid response: {response}"
            )))
        }
    }

    /// Format version string with 'v' prefix if needed.
    fn format_version(version: &str) -> String {
        if version.is_empty() {
            String::new()
        } else if version.starts_with('v') || version.starts_with('V') {
            version.to_string()
        } else {
            format!("v{version}")
        }
    }
}

impl Drop for GtpEngine {
    fn drop(&mut self) {
        let _ = self.quit();
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
