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

/// A GTP-compatible Reversi engine process.
///
/// This struct manages communication with an external Reversi engine process
/// that implements the GTP (Go Text Protocol). It handles starting the process,
/// sending commands, and receiving responses.
pub struct GtpEngine {
    process: Child,
}

impl GtpEngine {
    /// Create a new GTP engine instance.
    ///
    /// Starts the engine process with the specified executable, arguments, and working directory.
    /// The process is configured with piped stdin, stdout, and stderr for communication.
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
    /// Returns an error if the engine process cannot be started.
    pub fn new(executable: &str, args: &[String], working_dir: Option<PathBuf>) -> Result<Self> {
        let exec_path = Path::new(executable);
        let default_working_dir = if let Some(parent) = exec_path.parent() {
            parent.to_path_buf()
        } else {
            PathBuf::from(".")
        };

        let working_dir = working_dir.unwrap_or(default_working_dir);

        let process = Command::new(executable)
            .args(args)
            .current_dir(&working_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        Ok(GtpEngine {
            process,
        })
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
        let stdin = self
            .process
            .stdin
            .as_mut()
            .ok_or_else(|| MatchRunnerError::Engine("Failed to open stdin".to_string()))?;

        writeln!(stdin, "{command}")?;
        stdin.flush()?;

        let stdout = self.process.stdout.as_mut()
            .ok_or_else(|| MatchRunnerError::Engine("Failed to open stdout".to_string()))?;

        let mut reader = BufReader::new(stdout);
        let mut response = String::new();
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(MatchRunnerError::Engine("Process closed stdout".to_string()));
            }

            if line.trim().is_empty() {
                break;
            }

            response.push_str(&line);
        }

        Ok(response)
    }

    /// Get the engine's name and version.
    ///
    /// Sends the "name" and "version" GTP commands to retrieve the engine's
    /// identification information and combines them into a single string.
    ///
    /// # Returns
    ///
    /// A formatted string containing the engine name and version.
    ///
    /// # Errors
    ///
    /// Returns an error if the engine doesn't respond properly to the
    /// name or version commands.
    pub fn name(&mut self) -> Result<String> {
        let name_response = self.send_command("name")?;
        let version_response = self.send_command("version")?;

        let name = if name_response.starts_with("= ") {
            name_response.strip_prefix("= ").unwrap().trim()
        } else {
            return Err(MatchRunnerError::Engine(
                format!("Invalid name response: {name_response}")
            ));
        };

        let version = if version_response.starts_with("= ") {
            let v = version_response.strip_prefix("= ").unwrap().trim();
            if !v.is_empty() && !v.starts_with('v') && !v.starts_with('V') {
                format!("v{v}")
            } else {
                v.to_string()
            }
        } else if version_response.starts_with("? ") {
            "".to_string()
        } else {
            return Err(MatchRunnerError::Engine(
                format!("Invalid version response: {version_response}")
            ));
        };

        if !version.is_empty() {
            Ok(format!("{name} {version}"))
        } else {
            Ok(name.to_string())
        }
    }

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
        let response = self.send_command("clear_board")?;
        if response.starts_with("=") {
            Ok(())
        } else {
            Err(MatchRunnerError::Engine(
                format!("Failed to clear board: {response}")
            ))
        }
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
        let response = self.send_command(&format!("play {color} {mv}"))?;
        if response.starts_with("=") {
            Ok(())
        } else {
            Err(MatchRunnerError::Engine(
                format!("Failed to play move: {response}")
            ))
        }
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
        let response = self.send_command(&format!("genmove {color}"))?;
        if response.starts_with("=") {
            Ok(response.strip_prefix("=").unwrap().trim().to_string())
        } else {
            Err(MatchRunnerError::Engine(
                format!("Failed to generate move: {response}")
            ))
        }
    }

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
        let _ = self.send_command("quit");
        Ok(())
    }
}

impl Drop for GtpEngine {
    fn drop(&mut self) {
        let _ = self.quit();
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
