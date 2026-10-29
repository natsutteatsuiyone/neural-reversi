mod endgame_probcut;
mod fit_endgame_probcut;
mod fit_probcut;
mod opening;
mod overwrite_scores;
mod probcut;
mod record;
mod score_openings;
mod selfplay;
mod shuffle;

use clap::{Parser, Subcommand};
use reversi_core::level::{ENDGAME_STEPS, Level};
use reversi_core::types::Depth;

use crate::shuffle::FilterConfig;

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: SubCommands,
}

#[derive(Debug, Subcommand)]
enum SubCommands {
    Selfplay {
        #[arg(long, default_value = "100000000")]
        games: u32,

        #[arg(long, default_value = "10000", value_parser = clap::value_parser!(u32).range(1..=65535))]
        games_per_file: u32,

        #[arg(long, default_value = "512")]
        hash_size: usize,

        #[arg(long, default_value = "12", value_parser = clap::value_parser!(u32).range(1..=60),
            help = "Midgame search depth")]
        mid_depth: u32,

        #[arg(long, default_value = "21", value_parser = parse_end_depth,
            help = "Endgame search depth. Single value for all selectivities, or 5 comma-separated values (Level1,Level2,Level3,Level4,None)")]
        end_depth: [Depth; ENDGAME_STEPS],

        #[arg(long, help = "Output file prefix [default: hostname]")]
        prefix: Option<String>,

        #[arg(short, long)]
        output_dir: String,

        #[arg(long)]
        openings: Option<String>,

        #[arg(long, default_value = "false")]
        resume: bool,
    },
    Opening {
        #[arg(short, long, value_parser = clap::value_parser!(u32).range(1..=20))]
        depth: Depth,
    },
    Probcut {
        #[arg(short, long)]
        input: String,

        #[arg(short, long)]
        output: String,
    },
    /// Writes one endgame position per held-out game as OBF, for evaltest.
    EndgameHoldout {
        #[arg(short, long)]
        input: String,

        #[arg(short, long)]
        output: String,

        #[arg(long, default_value_t = 400)]
        count: usize,

        #[arg(long, default_value_t = 20)]
        min_empties: u32,

        #[arg(long, default_value_t = 26)]
        max_empties: u32,

        /// Draw positions from the fitting folds instead of the holdout fold.
        #[arg(long)]
        fit_folds: bool,
    },
    /// Traces endgame ProbCut attempts from game positions and labels them
    /// with exact scores (requires the `trace` feature).
    EndgameTrace {
        #[arg(short, long)]
        input: String,

        #[arg(short, long)]
        output: String,

        #[arg(long, default_value_t = 400)]
        roots: usize,

        #[arg(long, default_value_t = 20)]
        root_min_empties: u32,

        #[arg(long, default_value_t = 24)]
        root_max_empties: u32,

        /// Draw roots from the fitting folds instead of the holdout fold.
        #[arg(long)]
        fit_folds: bool,

        /// Keep attempts whose position hash is divisible by this.
        #[arg(long, default_value_t = 64)]
        one_in: u64,

        /// Keep at most this many distinct positions per empty count.
        #[arg(long, default_value_t = usize::MAX)]
        max_per_empties: usize,

        #[arg(long, default_value_t = std::thread::available_parallelism().map_or(1, |n| n.get()))]
        threads: usize,
    },
    FitProbcut {
        #[arg(short, long)]
        input: String,
    },
    /// Fits the endgame ProbCut models from `endgame-trace` samples and prints
    /// the Rust block to stdout; diagnostics go to stderr.
    FitEndgameProbcut {
        #[arg(short, long)]
        input: String,

        /// `endgame-trace` output to check the fitted models against.
        #[arg(long)]
        validate: Option<String>,
    },
    Shuffle {
        #[arg(short, long)]
        input_dir: String,

        #[arg(short, long)]
        output_dir: String,

        #[arg(short = 'p', long, default_value = "*.bin")]
        pattern: String,

        #[arg(short = 'c', long, default_value_t = 10,
            value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
        files_per_chunk: usize,

        #[arg(short = 'n', long)]
        num_output_files: Option<usize>,

        #[arg(
            long,
            default_value_t = false,
            help = "Validate existing shuffled_*.bin shards and append new records instead of refusing to reuse the output directory."
        )]
        append: bool,

        #[arg(short = 'm', long, default_value_t = 0)]
        min_ply: u8,

        #[arg(
            long,
            value_parser = parse_score_diff_threshold,
            help = "Drop records where |score - game_score| exceeds this threshold (in discs). Records with unavailable game_score are kept."
        )]
        max_score_diff: Option<f32>,

        #[arg(
            long,
            default_value_t = false,
            help = "Drop records whose move was chosen randomly (is_random = true)."
        )]
        drop_random: bool,

        #[arg(
            long,
            value_parser = clap::value_parser!(u8).range(0..=60),
            help = "Keep all records with ply >= this value, bypassing --drop-random and --max-score-diff filters."
        )]
        keep_above_ply: Option<u8>,
    },
    ScoreOpenings {
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=20),
            help = "Number of plies to enumerate from the initial position")]
        depth: u8,

        #[arg(long, default_value = "512")]
        hash_size: usize,

        #[arg(long, default_value = "16", value_parser = clap::value_parser!(u32).range(1..=60),
            help = "Midgame search depth")]
        mid_depth: u32,

        #[arg(long, default_value = "24", value_parser = parse_end_depth,
            help = "Endgame search depth. Single value for all selectivities, or 5 comma-separated values")]
        end_depth: [Depth; ENDGAME_STEPS],

        #[arg(short, long)]
        output: String,
    },
    OverwriteScores {
        #[arg(
            short,
            long,
            help = "Source binary file whose scores overwrite the targets"
        )]
        source: String,

        #[arg(short = 't', long, help = "Directory containing files to be updated")]
        target_dir: String,

        #[arg(short = 'p', long, default_value = "*.bin")]
        pattern: String,
    },
}

fn parse_score_diff_threshold(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("invalid f32 '{s}': {e}"))?;
    if !v.is_finite() || v < 0.0 {
        return Err(format!("expected a non-negative finite number, got {v}"));
    }
    Ok(v)
}

fn parse_end_depth(s: &str) -> Result<[Depth; ENDGAME_STEPS], String> {
    let values: Vec<Depth> = s
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<Depth>()
                .map_err(|e| format!("invalid value '{v}': {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    for &v in &values {
        if !(1..=60).contains(&v) {
            return Err(format!("value {v} out of range, expected 1..=60"));
        }
    }

    match values.len() {
        1 => Ok([values[0]; ENDGAME_STEPS]),
        ENDGAME_STEPS => Ok(values.try_into().unwrap()),
        n => Err(format!("expected 1 or {ENDGAME_STEPS} values, got {n}")),
    }
}

fn main() {
    let args = Cli::parse();
    match args.command {
        SubCommands::Selfplay {
            games,
            games_per_file,
            hash_size,
            mid_depth,
            end_depth,
            prefix,
            output_dir,
            openings,
            resume,
        } => {
            let prefix =
                prefix.unwrap_or_else(|| gethostname::gethostname().to_string_lossy().into_owned());
            let level = Level::with_depths(mid_depth, end_depth);
            if let Some(openings_path) = openings {
                selfplay::execute_with_openings(
                    &openings_path,
                    resume,
                    games_per_file,
                    hash_size,
                    level,
                    &prefix,
                    &output_dir,
                )
                .expect("Failed to execute selfplay with openings");
            } else {
                selfplay::execute(
                    games,
                    games_per_file,
                    hash_size,
                    level,
                    &prefix,
                    &output_dir,
                )
                .expect("Failed to execute selfplay");
            }
        }
        SubCommands::Opening { depth } => {
            opening::generate(depth);
        }
        SubCommands::Probcut { input, output } => {
            probcut::execute(&input, &output).expect("Failed to execute probcut");
        }
        SubCommands::EndgameHoldout {
            input,
            output,
            count,
            min_empties,
            max_empties,
            fit_folds,
        } => {
            endgame_probcut::export_holdout(
                &input,
                &output,
                count,
                min_empties..=max_empties,
                fit_folds,
            )
            .expect("Failed to execute endgame-holdout");
        }
        #[cfg(feature = "trace")]
        SubCommands::EndgameTrace {
            input,
            output,
            roots,
            root_min_empties,
            root_max_empties,
            fit_folds,
            one_in,
            max_per_empties,
            threads,
        } => {
            endgame_probcut::trace(
                &input,
                &output,
                &endgame_probcut::TraceOptions {
                    roots,
                    root_empties: root_min_empties..=root_max_empties,
                    fit_folds,
                    one_in,
                    max_per_empties,
                    threads,
                },
            )
            .expect("Failed to execute endgame-trace");
        }
        #[cfg(not(feature = "trace"))]
        SubCommands::EndgameTrace { .. } => {
            eprintln!("endgame-trace needs datagen built with `--features trace`");
            std::process::exit(1);
        }
        SubCommands::FitProbcut { input } => {
            fit_probcut::execute(&input).expect("Failed to execute fit-probcut");
        }
        SubCommands::FitEndgameProbcut { input, validate } => {
            fit_endgame_probcut::execute(&input, validate.as_deref())
                .expect("Failed to execute fit-endgame-probcut");
        }
        SubCommands::Shuffle {
            input_dir,
            output_dir,
            pattern,
            files_per_chunk,
            num_output_files,
            append,
            min_ply,
            max_score_diff,
            drop_random,
            keep_above_ply,
        } => {
            let filter = FilterConfig {
                min_ply,
                max_score_diff,
                drop_random,
                keep_above_ply,
            };
            shuffle::execute(
                &input_dir,
                &output_dir,
                &pattern,
                files_per_chunk,
                num_output_files,
                append,
                filter,
            )
            .unwrap();
        }
        SubCommands::ScoreOpenings {
            depth,
            hash_size,
            mid_depth,
            end_depth,
            output,
        } => {
            let level = Level::with_depths(mid_depth, end_depth);
            score_openings::execute(depth, hash_size, level, &output)
                .expect("Failed to execute score-openings");
        }
        SubCommands::OverwriteScores {
            source,
            target_dir,
            pattern,
        } => {
            overwrite_scores::execute(&source, &target_dir, &pattern)
                .expect("Failed to execute overwrite-scores");
        }
    }
}
