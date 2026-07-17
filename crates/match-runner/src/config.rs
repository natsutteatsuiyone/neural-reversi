//! Configuration management for match runner.
//!
//! This module handles command-line argument parsing and opening file loading
//! for the match runner engine testing tool.

use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use crate::error::{MatchRunnerError, Result};
use crate::sprt::SprtConfig;

/// Configuration for running automated matches between two GTP engines.
///
/// This struct defines all the necessary parameters for setting up and running
/// a match between two Reversi engines, including engine commands, working directories,
/// and opening positions.
///
/// # Time Control
///
/// Time control follows the GTP `time_settings` command format:
/// `time_settings main_time byo_yomi_time byo_yomi_stones`
///
/// The time control mode is automatically determined by the combination of parameters:
///
/// - No time control: `--main-time 0 --byoyomi-time 0`
/// - Pure byoyomi (fixed time per move): `--main-time 0 --byoyomi-time N --byoyomi-stones 0`
/// - Fischer (main time + increment): `--main-time M --byoyomi-time N --byoyomi-stones 0`
/// - Japanese byo-yomi: `--main-time M --byoyomi-time N --byoyomi-stones 1`
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

    /// Main time in seconds (0 for no main time, starts in byoyomi)
    #[arg(long, default_value_t = 0)]
    pub main_time: u64,

    /// Byoyomi time in seconds (time per move or increment depending on byoyomi-stones)
    #[arg(long, default_value_t = 0)]
    pub byoyomi_time: u64,

    /// Byoyomi stones (0: time is increment/per-move, 1+: stones per byoyomi period)
    #[arg(long, default_value_t = 0)]
    pub byoyomi_stones: u32,

    /// Hard per-command engine timeout in seconds (default: none)
    #[arg(long)]
    pub move_timeout: Option<u64>,

    /// Stop the match early once SPRT accepts either configured hypothesis
    #[arg(long)]
    pub sprt: bool,

    /// SPRT: Elo difference under the null hypothesis H0
    #[arg(long, default_value_t = -10.0, allow_negative_numbers = true, requires = "sprt")]
    pub sprt_elo0: f64,

    /// SPRT: Elo difference under the alternative hypothesis H1
    #[arg(
        long,
        default_value_t = 10.0,
        allow_negative_numbers = true,
        requires = "sprt"
    )]
    pub sprt_elo1: f64,

    /// SPRT: type I error rate α
    #[arg(long, default_value_t = 0.05, requires = "sprt")]
    pub sprt_alpha: f64,

    /// SPRT: type II error rate β
    #[arg(long, default_value_t = 0.05, requires = "sprt")]
    pub sprt_beta: f64,
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

    /// Build the SPRT configuration, if early termination is enabled.
    ///
    /// # Returns
    ///
    /// `Ok(None)` when `--sprt` was not given, otherwise the validated
    /// [`SprtConfig`].
    ///
    /// # Errors
    ///
    /// Returns a configuration error if either Elo hypothesis is non-finite,
    /// if `sprt_elo0 >= sprt_elo1`, if either error rate is outside `(0, 1)`,
    /// or if `alpha + beta >= 1` (which would invert the decision bounds).
    pub fn sprt_config(&self) -> Result<Option<SprtConfig>> {
        if !self.sprt {
            return Ok(None);
        }

        for (name, value) in [
            ("--sprt-elo0", self.sprt_elo0),
            ("--sprt-elo1", self.sprt_elo1),
        ] {
            if !value.is_finite() {
                return Err(MatchRunnerError::Config(format!(
                    "{name} ({value}) must be finite"
                )));
            }
        }
        if self.sprt_elo0 >= self.sprt_elo1 {
            return Err(MatchRunnerError::Config(format!(
                "--sprt-elo0 ({}) must be less than --sprt-elo1 ({})",
                self.sprt_elo0, self.sprt_elo1
            )));
        }
        for (name, value) in [
            ("--sprt-alpha", self.sprt_alpha),
            ("--sprt-beta", self.sprt_beta),
        ] {
            if !(value > 0.0 && value < 1.0) {
                return Err(MatchRunnerError::Config(format!(
                    "{name} ({value}) must be strictly between 0 and 1"
                )));
            }
        }
        if self.sprt_alpha + self.sprt_beta >= 1.0 {
            return Err(MatchRunnerError::Config(format!(
                "--sprt-alpha + --sprt-beta ({}) must be less than 1",
                self.sprt_alpha + self.sprt_beta
            )));
        }

        Ok(Some(SprtConfig {
            elo0: self.sprt_elo0,
            elo1: self.sprt_elo1,
            alpha: self.sprt_alpha,
            beta: self.sprt_beta,
        }))
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
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::UI::Shell::CommandLineToArgvW;

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
            let arg = OsString::from_wide(arg_slice)
                .to_string_lossy()
                .into_owned();
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
    fn parses_optional_move_timeout() {
        let config = Config::try_parse_from([
            "match-runner",
            "--engine1",
            "engine-one",
            "--engine2",
            "engine-two",
            "--opening-file",
            "openings.txt",
            "--move-timeout",
            "2",
        ])
        .unwrap();

        assert_eq!(config.move_timeout, Some(2));
    }

    #[test]
    fn sprt_is_disabled_by_default() {
        let config = Config::try_parse_from([
            "match-runner",
            "--engine1",
            "engine-one",
            "--engine2",
            "engine-two",
            "--opening-file",
            "openings.txt",
        ])
        .unwrap();

        assert!(config.sprt_config().unwrap().is_none());
    }

    #[test]
    fn parses_sprt_options() {
        let config = Config::try_parse_from([
            "match-runner",
            "--engine1",
            "engine-one",
            "--engine2",
            "engine-two",
            "--opening-file",
            "openings.txt",
            "--sprt",
            "--sprt-elo0",
            "-5",
            "--sprt-elo1",
            "5",
            "--sprt-alpha",
            "0.01",
            "--sprt-beta",
            "0.1",
        ])
        .unwrap();

        let sprt = config.sprt_config().unwrap().unwrap();
        assert_eq!(sprt.elo0, -5.0);
        assert_eq!(sprt.elo1, 5.0);
        assert_eq!(sprt.alpha, 0.01);
        assert_eq!(sprt.beta, 0.1);
    }

    #[test]
    fn sprt_options_require_the_sprt_flag() {
        let result = Config::try_parse_from([
            "match-runner",
            "--engine1",
            "engine-one",
            "--engine2",
            "engine-two",
            "--opening-file",
            "openings.txt",
            "--sprt-elo1",
            "20",
        ]);

        assert!(result.is_err());
    }

    #[test]
    fn rejects_inverted_sprt_elo_bounds() {
        let config = Config::try_parse_from([
            "match-runner",
            "--engine1",
            "engine-one",
            "--engine2",
            "engine-two",
            "--opening-file",
            "openings.txt",
            "--sprt",
            "--sprt-elo0",
            "10",
            "--sprt-elo1",
            "-10",
        ])
        .unwrap();

        assert!(config.sprt_config().is_err());
    }

    #[test]
    fn rejects_non_finite_sprt_elo_hypotheses() {
        let mut config = Config::try_parse_from([
            "match-runner",
            "--engine1",
            "engine-one",
            "--engine2",
            "engine-two",
            "--opening-file",
            "openings.txt",
            "--sprt",
            "--sprt-elo0",
            "NaN",
        ])
        .unwrap();

        assert!(config.sprt_config().is_err());

        for (elo0, elo1) in [
            (f64::NEG_INFINITY, 10.0),
            (-10.0, f64::INFINITY),
            (-10.0, f64::NAN),
        ] {
            config.sprt_elo0 = elo0;
            config.sprt_elo1 = elo1;

            assert!(
                config.sprt_config().is_err(),
                "accepted non-finite hypotheses [{elo0}, {elo1}]"
            );
        }
    }

    #[test]
    fn rejects_out_of_range_sprt_error_rates() {
        let config = Config::try_parse_from([
            "match-runner",
            "--engine1",
            "engine-one",
            "--engine2",
            "engine-two",
            "--opening-file",
            "openings.txt",
            "--sprt",
            "--sprt-alpha",
            "1.5",
        ])
        .unwrap();

        assert!(config.sprt_config().is_err());
    }

    #[test]
    fn test_parse_simple_command() {
        let config = Config {
            engine1: "./engine --level 10".to_string(),
            engine2: "engine2".to_string(),
            engine1_working_dir: None,
            engine2_working_dir: None,
            opening_file: PathBuf::from("test_openings.txt"),
            main_time: 0,
            byoyomi_time: 0,
            byoyomi_stones: 0,
            move_timeout: None,
            sprt: false,
            sprt_elo0: -10.0,
            sprt_elo1: 10.0,
            sprt_alpha: 0.05,
            sprt_beta: 0.05,
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
            main_time: 0,
            byoyomi_time: 0,
            byoyomi_stones: 0,
            move_timeout: None,
            sprt: false,
            sprt_elo0: -10.0,
            sprt_elo1: 10.0,
            sprt_alpha: 0.05,
            sprt_beta: 0.05,
        };

        // Test with quotes (behavior varies by platform)
        let (program, args) =
            config.parse_engine_command(r#""./my engine" --arg "value with spaces""#);

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
            main_time: 0,
            byoyomi_time: 0,
            byoyomi_stones: 0,
            move_timeout: None,
            sprt: false,
            sprt_elo0: -10.0,
            sprt_elo1: 10.0,
            sprt_alpha: 0.05,
            sprt_beta: 0.05,
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
            main_time: 0,
            byoyomi_time: 0,
            byoyomi_stones: 0,
            move_timeout: None,
            sprt: false,
            sprt_elo0: -10.0,
            sprt_elo1: 10.0,
            sprt_alpha: 0.05,
            sprt_beta: 0.05,
        };

        // Test Windows path with spaces
        let (program, args) =
            config.parse_engine_command(r#""C:\Program Files\My Engine\engine.exe" --level 10"#);
        assert_eq!(program, r"C:\Program Files\My Engine\engine.exe");
        assert_eq!(args, vec!["--level", "10"]);

        // Test with multiple quoted arguments
        let (program, args) = config.parse_engine_command(
            r#""C:\Program Files\engine.exe" --config "C:\My Documents\config.txt""#,
        );
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
            main_time: 0,
            byoyomi_time: 0,
            byoyomi_stones: 0,
            move_timeout: None,
            sprt: false,
            sprt_elo0: -10.0,
            sprt_elo1: 10.0,
            sprt_alpha: 0.05,
            sprt_beta: 0.05,
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
            main_time: 0,
            byoyomi_time: 0,
            byoyomi_stones: 0,
            move_timeout: None,
            sprt: false,
            sprt_elo0: -10.0,
            sprt_elo1: 10.0,
            sprt_alpha: 0.05,
            sprt_beta: 0.05,
        };

        // Test escaped spaces (shell-style) - shlex interprets the escape
        let (program, args) = config.parse_engine_command(r"./my\ engine --level 10");
        assert_eq!(program, "./my engine"); // shlex interprets the escape
        assert_eq!(args, vec!["--level", "10"]);

        // Test single quotes
        let (program, args) = config.parse_engine_command("'./my engine' --level 10");
        assert_eq!(program, "./my engine");
        assert_eq!(args, vec!["--level", "10"]);
    }
}
