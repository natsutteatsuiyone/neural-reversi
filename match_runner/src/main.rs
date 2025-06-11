use std::io;

mod config;
mod display;
mod engine;
mod error;
mod game;
mod match_runner;
mod statistics;

use config::Config;
use match_runner::MatchRunner;
use error::MatchRunnerError;

fn main() -> io::Result<()> {
    let config = Config::parse_args();

    if let Err(e) = config.validate() {
        eprintln!("Error: {}", e);
        return Ok(());
    }

    let mut match_runner = MatchRunner::new();
    
    if let Err(e) = match_runner.run_match(&config) {
        match e {
            MatchRunnerError::Io(io_err) => return Err(io_err),
            _ => {
                eprintln!("Error: {}", e);
                return Ok(());
            }
        }
    }

    Ok(())
}

