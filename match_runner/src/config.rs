//! Configuration management for match runner.
//!
//! This module handles command-line argument parsing and opening file loading
//! for the match runner engine testing tool.

use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::error::Result;

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
    #[arg(short, long, required = true)]
    pub opening_file: PathBuf,
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
        read_opening_file(&self.opening_file)
    }

    /// Parse an engine command string into program and arguments.
    ///
    /// Uses platform-specific command parsing:
    /// - On Windows: Uses Windows command-line parsing rules
    /// - On Unix: Uses shell-like parsing with shlex
    ///
    /// # Arguments
    ///
    /// * `engine_cmd` - The full command string (e.g., "./engine --level 10")
    ///
    /// # Returns
    ///
    /// A tuple containing the program path and a vector of arguments.
    pub fn parse_engine_command(&self, engine_cmd: &str) -> (String, Vec<String>) {
        #[cfg(target_os = "windows")]
        {
            parse_windows_command(engine_cmd)
        }

        #[cfg(not(target_os = "windows"))]
        {
            parse_unix_command(engine_cmd)
        }
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

/// Parse a command string using Unix shell-like rules.
#[cfg(not(target_os = "windows"))]
fn parse_unix_command(cmd: &str) -> (String, Vec<String>) {
    match shlex::split(cmd) {
        Some(parts) if !parts.is_empty() => {
            let program = parts[0].clone();
            let args = parts[1..].to_vec();
            (program, args)
        }
        _ => {
            // Fallback to simple whitespace splitting if shlex fails
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if parts.is_empty() {
                (String::new(), Vec::new())
            } else {
                let program = parts[0].to_string();
                let args = parts[1..].iter().map(|s| s.to_string()).collect();
                (program, args)
            }
        }
    }
}

/// Parse a command string using Windows command-line rules.
#[cfg(target_os = "windows")]
fn parse_windows_command(cmd: &str) -> (String, Vec<String>) {
    // Handle empty command string
    if cmd.trim().is_empty() {
        return (String::new(), Vec::new());
    }

    use std::ffi::{OsStr, OsString};
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use windows_sys::Win32::UI::Shell::CommandLineToArgvW;
    use windows_sys::Win32::Foundation::LocalFree;

    unsafe {
        let cmd_wide: Vec<u16> = OsStr::new(cmd)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut argc = 0;
        let argv_ptr = CommandLineToArgvW(cmd_wide.as_ptr(), &mut argc);

        if argv_ptr.is_null() || argc == 0 {
            // Fallback to simple parsing
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if parts.is_empty() {
                return (String::new(), Vec::new());
            }
            let program = parts[0].to_string();
            let args = parts[1..].iter().map(|s| s.to_string()).collect();
            return (program, args);
        }

        let mut args = Vec::new();
        for i in 0..argc {
            let arg_ptr = *argv_ptr.add(i as usize);
            let len = (0..).take_while(|&j| *arg_ptr.add(j) != 0).count();
            let arg_slice = std::slice::from_raw_parts(arg_ptr, len);
            let arg = OsString::from_wide(arg_slice).to_string_lossy().into_owned();
            args.push(arg);
        }

        LocalFree(argv_ptr as _);

        if args.is_empty() {
            (String::new(), Vec::new())
        } else {
            let program = args[0].clone();
            let args = args[1..].to_vec();
            (program, args)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_command() {
        let config = Config {
            engine1: "./engine --level 10".to_string(),
            engine2: "engine2".to_string(),
            engine1_working_dir: None,
            engine2_working_dir: None,
            opening_file: PathBuf::from("test_openings.txt"),
        };

        let (program, args) = config.parse_engine_command("./reversi_cli --level 10");
        assert_eq!(program, "./reversi_cli");
        assert_eq!(args, vec!["--level", "10"]);
    }

    #[test]
    #[allow(unused_variables)]
    fn test_parse_quoted_command() {
        let config = Config {
            engine1: "".to_string(),
            engine2: "".to_string(),
            engine1_working_dir: None,
            engine2_working_dir: None,
            opening_file: PathBuf::from("test_openings.txt"),
        };

        // Test with quotes (behavior varies by platform)
        let (program, args) = config.parse_engine_command(r#""./my engine" --arg "value with spaces""#);

        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(program, "./my engine");
            assert_eq!(args, vec!["--arg", "value with spaces"]);
        }
    }

    #[test]
    fn test_parse_empty_command() {
        let config = Config {
            engine1: "".to_string(),
            engine2: "".to_string(),
            engine1_working_dir: None,
            engine2_working_dir: None,
            opening_file: PathBuf::from("test_openings.txt"),
        };

        let (program, args) = config.parse_engine_command("");
        assert_eq!(program, "");
        assert_eq!(args.len(), 0);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_parse_windows_path_with_spaces() {
        let config = Config {
            engine1: "".to_string(),
            engine2: "".to_string(),
            engine1_working_dir: None,
            engine2_working_dir: None,
            opening_file: PathBuf::from("test_openings.txt"),
        };

        // Test Windows path with spaces
        let (program, args) = config.parse_engine_command(r#""C:\Program Files\My Engine\engine.exe" --level 10"#);
        assert_eq!(program, r"C:\Program Files\My Engine\engine.exe");
        assert_eq!(args, vec!["--level", "10"]);

        // Test with multiple quoted arguments
        let (program, args) = config.parse_engine_command(r#""C:\Program Files\engine.exe" --config "C:\My Documents\config.txt""#);
        assert_eq!(program, r"C:\Program Files\engine.exe");
        assert_eq!(args, vec!["--config", r"C:\My Documents\config.txt"]);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_parse_windows_backslash_paths() {
        let config = Config {
            engine1: "".to_string(),
            engine2: "".to_string(),
            engine1_working_dir: None,
            engine2_working_dir: None,
            opening_file: PathBuf::from("test_openings.txt"),
        };

        // Test simple backslash path
        let (program, args) = config.parse_engine_command(r"C:\engines\reversi.exe --level 5");
        assert_eq!(program, r"C:\engines\reversi.exe");
        assert_eq!(args, vec!["--level", "5"]);

        // Test UNC path
        let (program, args) = config.parse_engine_command(r"\\server\share\engine.exe --mode fast");
        assert_eq!(program, r"\\server\share\engine.exe");
        assert_eq!(args, vec!["--mode", "fast"]);
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn test_parse_unix_escaped_spaces() {
        let config = Config {
            engine1: "".to_string(),
            engine2: "".to_string(),
            engine1_working_dir: None,
            engine2_working_dir: None,
            opening_file: PathBuf::from("test_openings.txt"),
        };

        // Test escaped spaces (shell-style) - shlex interprets the escape
        let (program, args) = config.parse_engine_command(r"./my\ engine --level 10");
        assert_eq!(program, "./my engine");  // shlex interprets the escape
        assert_eq!(args, vec!["--level", "10"]);

        // Test single quotes
        let (program, args) = config.parse_engine_command("'./my engine' --level 10");
        assert_eq!(program, "./my engine");
        assert_eq!(args, vec!["--level", "10"]);
    }
}
