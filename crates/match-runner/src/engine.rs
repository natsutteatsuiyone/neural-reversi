//! GTP (Go Text Protocol) engine communication.
//!
//! This module provides functionality for communicating with external Reversi engines
//! that implement the GTP protocol. It handles process management, command sending,
//! and response parsing.

use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
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
// Error messages
const ERR_STDIN_FAILED: &str = "Failed to open stdin";
const ERR_STDOUT_FAILED: &str = "Failed to open stdout";
const ERR_PROCESS_CLOSED: &str = "Process closed stdout";

/// A GTP-compatible Reversi engine process.
///
/// This struct manages communication with an external Reversi engine process
/// that implements the GTP (Go Text Protocol). It handles starting the process,
/// sending commands, and receiving responses.
///
/// The engine's stdin and stdout are stored as fields to maintain a persistent
/// `BufReader`, preventing potential data loss from repeated buffer recreation.
pub struct GtpEngine {
    process: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
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
    /// The process is configured with piped stdin and stdout for communication.
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
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = process
            .stdin
            .take()
            .ok_or_else(|| MatchRunnerError::Engine(ERR_STDIN_FAILED.to_string()))?;
        let stdout = process
            .stdout
            .take()
            .ok_or_else(|| MatchRunnerError::Engine(ERR_STDOUT_FAILED.to_string()))?;
        let reader = BufReader::new(stdout);

        let mut engine = GtpEngine {
            process,
            stdin,
            reader,
            name: String::new(),
            version: String::new(),
        };

        // Get name and version
        let name_response = engine.send_command(GTP_CMD_NAME)?;
        engine.name = Self::parse_success_response(&name_response)?;

        let version_response = engine.send_command(GTP_CMD_VERSION)?;
        let version_raw = Self::parse_optional_response(&version_response)?;
        engine.version = Self::format_version(&version_raw);

        Ok(engine)
    }

    // =============================================================================
    // Core Communication
    // =============================================================================

    /// Core GTP communication logic.
    fn communicate(
        stdin: &mut dyn Write,
        reader: &mut dyn BufRead,
        command: &str,
    ) -> Result<String> {
        writeln!(stdin, "{command}")?;
        stdin.flush()?;

        let mut response = String::new();
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
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
        Self::communicate(&mut self.stdin, &mut self.reader, command)
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
    pub fn genmove(&mut self, color: &str) -> Result<String> {
        let response = self.send_command(&format!("{GTP_CMD_GENMOVE} {color}"))?;
        Self::parse_success_response(&response)
    }

    // =============================================================================
    // Time Control
    // =============================================================================

    /// Configure time control settings for the engine.
    ///
    /// Sends a "time_settings" GTP command to configure the engine's time control.
    /// This command is optional and some engines may not support it.
    pub fn time_settings(
        &mut self,
        main_time: u64,
        byoyomi_time: u64,
        byoyomi_stones: u32,
    ) -> Result<()> {
        let response = self.send_command(&format!(
            "time_settings {main_time} {byoyomi_time} {byoyomi_stones}"
        ))?;
        // Ignore errors - not all engines support time_settings
        let _ = Self::parse_optional_response(&response);
        Ok(())
    }

    /// Update the remaining time for a player.
    ///
    /// Sends a "time_left" GTP command to inform the engine of the current
    /// remaining time for a player.
    pub fn time_left(&mut self, color: &str, time: u64, stones: u32) -> Result<()> {
        let response = self.send_command(&format!("time_left {color} {time} {stones}"))?;
        // Ignore errors - not all engines support time_left
        let _ = Self::parse_optional_response(&response);
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
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_success_response_with_content() {
        let result = GtpEngine::parse_success_response("= hello world\n").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_parse_success_response_empty() {
        let result = GtpEngine::parse_success_response("=\n").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_parse_success_response_just_equals_space() {
        let result = GtpEngine::parse_success_response("= \n").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_parse_success_response_rejects_error() {
        let result = GtpEngine::parse_success_response("? unknown command\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_optional_response_success() {
        let result = GtpEngine::parse_optional_response("= 1.0\n").unwrap();
        assert_eq!(result, "1.0");
    }

    #[test]
    fn test_parse_optional_response_failure_returns_empty() {
        let result = GtpEngine::parse_optional_response("? unsupported\n").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_parse_optional_response_invalid() {
        let result = GtpEngine::parse_optional_response("garbage\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_format_version_empty() {
        assert_eq!(GtpEngine::format_version(""), "");
    }

    #[test]
    fn test_format_version_with_v_prefix() {
        assert_eq!(GtpEngine::format_version("v1.0"), "v1.0");
    }

    #[test]
    fn test_format_version_with_capital_v_prefix() {
        assert_eq!(GtpEngine::format_version("V2.0"), "V2.0");
    }

    #[test]
    fn test_format_version_without_prefix() {
        assert_eq!(GtpEngine::format_version("1.0"), "v1.0");
    }

    #[test]
    fn test_communicate_success() {
        let response_data = b"= hello\n\n";
        let mut reader = std::io::Cursor::new(response_data.to_vec());
        let mut writer = Vec::new();

        let result = GtpEngine::communicate(&mut writer, &mut reader, "test_cmd").unwrap();

        assert_eq!(result, "= hello\n");
        assert_eq!(String::from_utf8(writer).unwrap(), "test_cmd\n");
    }

    #[test]
    fn test_communicate_multiline_response() {
        let response_data = b"= line1\nline2\n\n";
        let mut reader = std::io::Cursor::new(response_data.to_vec());
        let mut writer = Vec::new();

        let result = GtpEngine::communicate(&mut writer, &mut reader, "cmd").unwrap();

        assert_eq!(result, "= line1\nline2\n");
    }

    #[test]
    fn test_communicate_process_closed() {
        let response_data = b"";
        let mut reader = std::io::Cursor::new(response_data.to_vec());
        let mut writer = Vec::new();

        let result = GtpEngine::communicate(&mut writer, &mut reader, "cmd");

        assert!(result.is_err());
    }
}
