//! Configuration management for automatch.
//!
//! This module handles command-line argument parsing and opening file loading
//! for the automatch engine testing tool.

use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::error::{AutomatchError, Result};

/// Configuration for running automated matches between two GTP engines.
///
/// This struct defines all the necessary parameters for setting up and running
/// a match between two Reversi engines, including engine commands, working directories,
/// and opening positions.
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Tool for running matches between GTP-compatible Reversi engines"
)]
pub struct Config {
    /// Command for the first engine (program path and arguments)
    #[arg(short = '1', long)]
    pub engine1: String,

    /// Working directory for the first program
    #[arg(long)]
    pub engine1_working_dir: Option<PathBuf>,

    /// Command for the second engine (program path and arguments)
    #[arg(short = '2', long)]
    pub engine2: String,

    /// Working directory for the second program
    #[arg(long)]
    pub engine2_working_dir: Option<PathBuf>,

    /// Opening file (required)
    #[arg(short, long)]
    pub opening_file: Option<PathBuf>,
}

impl Config {
    /// Parse command-line arguments into a Config instance.
    ///
    /// # Returns
    ///
    /// A new `Config` instance with parsed command-line arguments.
    pub fn parse_args() -> Self {
        Self::parse()
    }

    /// Validate the configuration parameters.
    ///
    /// Ensures that all required parameters are present and valid.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the configuration is valid, otherwise returns an error
    /// describing what is missing or invalid.
    ///
    /// # Errors
    ///
    /// Returns `AutomatchError::Config` if the opening file is not specified.
    pub fn validate(&self) -> Result<()> {
        if self.opening_file.is_none() {
            return Err(AutomatchError::Config(
                "Opening file not specified. Please specify a file with the -o or --opening-file option.".to_string()
            ));
        }

        Ok(())
    }

    /// Load opening positions from the configured opening file.
    ///
    /// Reads the opening file line by line, filtering out comments (lines starting with '#')
    /// and empty lines.
    ///
    /// # Returns
    ///
    /// A vector of opening position strings. Each string represents a sequence of moves
    /// in algebraic notation (e.g., "f5d6c3d3c4f4").
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if there are I/O issues.
    pub fn load_openings(&self) -> Result<Vec<String>> {
        match &self.opening_file {
            Some(path) => read_opening_file(path),
            None => Ok(Vec::new()),
        }
    }

    /// Parse an engine command string into program and arguments.
    ///
    /// Splits a command string by whitespace, treating the first part as the program
    /// path and the rest as arguments.
    ///
    /// # Arguments
    ///
    /// * `engine_cmd` - The full command string (e.g., "./engine --level 10")
    ///
    /// # Returns
    ///
    /// A tuple containing the program path and a vector of arguments.
    pub fn parse_engine_command(&self, engine_cmd: &str) -> (String, Vec<String>) {
        let parts: Vec<&str> = engine_cmd.split_whitespace().collect();
        let program = parts[0].to_string();
        let args = parts[1..].iter().map(|s| s.to_string()).collect();
        (program, args)
    }

    /// Get the parsed command for engine 1.
    ///
    /// # Returns
    ///
    /// A tuple containing the program path and arguments for engine 1.
    pub fn get_engine1_command(&self) -> (String, Vec<String>) {
        self.parse_engine_command(&self.engine1)
    }

    /// Get the parsed command for engine 2.
    ///
    /// # Returns
    ///
    /// A tuple containing the program path and arguments for engine 2.
    pub fn get_engine2_command(&self) -> (String, Vec<String>) {
        self.parse_engine_command(&self.engine2)
    }
}

/// Read opening positions from a file.
///
/// Each line in the file represents an opening position. Lines starting with '#'
/// are treated as comments and ignored. Empty lines are also ignored.
///
/// # Arguments
///
/// * `path` - Path to the opening file
///
/// # Returns
///
/// A vector of opening position strings.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or read.
fn read_opening_file(path: &Path) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut openings = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            openings.push(line.to_string());
        }
    }

    Ok(openings)
}
