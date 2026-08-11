use std::io;

use clap::Parser;

mod colors;
mod config;
mod display;
mod engine;
mod error;
mod runner;
mod sprt;
mod statistics;
mod time_tracker;

use config::Config;
use error::MatchRunnerError;

fn main() -> io::Result<()> {
    let config = Config::parse();

    if let Err(e) = runner::run_match(&config) {
        match e {
            MatchRunnerError::Io(io_err) => return Err(io_err),
            error @ MatchRunnerError::Interrupted => {
                eprintln!("Error: {error}");
                std::process::exit(130);
            }
            error => {
                eprintln!("Error: {error}");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
