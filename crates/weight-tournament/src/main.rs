//! One-ply tournament runner for `reversi-core` main-network weights.

mod game;
mod tournament;

use std::num::NonZeroUsize;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

/// Compare `reversi-core` main-network weights with one-ply games.
#[derive(Debug, Parser)]
#[command(version, about)]
pub(crate) struct Args {
    /// Directory containing *.zst main-network weights
    weights_dir: PathBuf,

    /// Opening file in match-runner format
    #[arg(short, long)]
    opening_file: Option<PathBuf>,

    /// Parallel comparisons
    #[arg(short, long, default_value_t = NonZeroUsize::MIN)]
    jobs: NonZeroUsize,
}

fn main() -> Result<()> {
    tournament::run(&Args::parse())
}
