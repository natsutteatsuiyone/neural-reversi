use clap::Parser;

mod colors;
mod config;
mod display;
mod engine;
mod error;
mod opening;
mod runner;
mod sprt;
mod statistics;
mod time_tracker;

use config::Config;
use error::MatchRunnerError;

fn main() {
    let config = Config::parse();

    if let Err(e) = runner::run_match(&config) {
        match e {
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
}
