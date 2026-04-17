mod opening;
mod probcut;
mod record;
mod score_openings;
mod selfplay;
mod shuffle;

use clap::{Parser, Subcommand};
use reversi_core::level::Level;
use reversi_core::probcut::Selectivity;
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

        #[arg(long, default_value = "10000")]
        games_per_file: u32,

        #[arg(long, default_value = "512")]
        hash_size: usize,

        #[arg(long, default_value = "12", value_parser = clap::value_parser!(u32).range(1..=60),
            help = "Midgame search depth")]
        mid_depth: u32,

        #[arg(long, default_value = "21", value_parser = parse_end_depth,
            help = "Endgame search depth. Single value for all selectivities, or 4 comma-separated values (Level1,Level3,Level5,None)")]
        end_depth: [Depth; 4],

        #[arg(long, default_value = "0", value_parser = clap::value_parser!(u8).range(0..=5))]
        selectivity: u8,

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
        #[arg(short, long)]
        depth: Depth,
    },
    Probcut {
        #[arg(short, long)]
        input: String,

        #[arg(short, long)]
        output: String,

        #[arg(long, default_value = "false")]
        endgame: bool,
    },
    Shuffle {
        #[arg(short, long)]
        input_dir: String,

        #[arg(short, long)]
        output_dir: String,

        #[arg(short = 'p', long, default_value = "*.bin")]
        pattern: String,

        #[arg(short = 'c', long, default_value_t = 10)]
        files_per_chunk: usize,

        #[arg(short = 'n', long)]
        num_output_files: Option<usize>,

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
            help = "Endgame search depth. Single value for all selectivities, or 4 comma-separated values")]
        end_depth: [Depth; 4],

        #[arg(long, default_value = "0", value_parser = clap::value_parser!(u8).range(0..=5))]
        selectivity: u8,

        #[arg(short, long)]
        output: String,
    },
}

fn parse_score_diff_threshold(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("invalid f32 '{s}': {e}"))?;
    if !v.is_finite() || v < 0.0 {
        return Err(format!("expected a non-negative finite number, got {v}"));
    }
    Ok(v)
}

fn parse_end_depth(s: &str) -> Result<[Depth; 4], String> {
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
        1 => Ok([values[0]; 4]),
        4 => Ok(values.try_into().unwrap()),
        n => Err(format!("expected 1 or 4 values, got {n}")),
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
            selectivity,
            prefix,
            output_dir,
            openings,
            resume,
        } => {
            let prefix =
                prefix.unwrap_or_else(|| gethostname::gethostname().to_string_lossy().into_owned());
            let level = Level {
                mid_depth,
                end_depth,
            };
            if let Some(openings_path) = openings {
                selfplay::execute_with_openings(
                    &openings_path,
                    resume,
                    games_per_file,
                    hash_size,
                    level,
                    Selectivity::from_u8(selectivity),
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
                    Selectivity::from_u8(selectivity),
                    &prefix,
                    &output_dir,
                )
                .expect("Failed to execute selfplay");
            }
        }
        SubCommands::Opening { depth } => {
            opening::generate(depth);
        }
        SubCommands::Probcut {
            input,
            output,
            endgame,
        } => {
            if endgame {
                probcut::execute_endgame(&input, &output)
                    .expect("Failed to execute probcut endgame");
            } else {
                probcut::execute(&input, &output).expect("Failed to execute probcut");
            }
        }
        SubCommands::Shuffle {
            input_dir,
            output_dir,
            pattern,
            files_per_chunk,
            num_output_files,
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
                filter,
            )
            .unwrap();
        }
        SubCommands::ScoreOpenings {
            depth,
            hash_size,
            mid_depth,
            end_depth,
            selectivity,
            output,
        } => {
            let level = Level {
                mid_depth,
                end_depth,
            };
            score_openings::execute(
                depth,
                hash_size,
                level,
                Selectivity::from_u8(selectivity),
                &output,
            )
            .expect("Failed to execute score-openings");
        }
    }
}
