//! Shared ProbCut game-fold assignment, move replay, and midgame ProbCut CSV I/O.
//!
//! Generates midgame shallow/deep search correlation CSVs (`datagen probcut`)
//! and streams validated [`MidgameSample`] rows back for parameter fitting
//! (`datagen fit-probcut`).

use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use reversi_core::{
    board::Board,
    constants::INITIAL_EMPTY_COUNT,
    eval::{Eval, EvalMode},
    game_state::GameState,
    level::Level,
    player_color::PlayerColor,
    score::Score,
    search::{Search, SearchRunOptions, options::SearchOptions},
    square::Square,
    types::Depth,
};

pub(crate) const NUM_FOLDS: usize = 5;

/// Game fold kept out of every fit.
pub(crate) const HOLDOUT_FOLD: usize = NUM_FOLDS - 1;

/// Transposition table size in MB for search
const TT_SIZE_MB: usize = 256;

/// Maximum deep search depth for midgame ProbCut analysis
const MAX_SEARCH_DEPTH: Depth = 14;

/// Maximum shallow depth for midgame ProbCut analysis
const MAX_SHALLOW_DEPTH: Depth = 7;

/// The deep search depth must exceed the shallow depth by more than this
const MIN_DEPTH_DIFFERENCE: Depth = 2;

const MIDGAME_CSV_HEADER: &str = "game,ply,shallow_depth,shallow_score,deep_depth,deep_score,diff";

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MidgameSample {
    pub game: u64,
    ply: usize,
    pub shallow_depth: Depth,
    pub deep_depth: Depth,
    pub diff: f64,
}

impl MidgameSample {
    #[must_use]
    pub(crate) fn new(
        game: u64,
        ply: usize,
        shallow_depth: Depth,
        deep_depth: Depth,
        diff: f64,
    ) -> Option<Self> {
        (ply < INITIAL_EMPTY_COUNT).then_some(Self {
            game,
            ply,
            shallow_depth,
            deep_depth,
            diff,
        })
    }

    #[inline]
    #[must_use]
    pub(crate) fn ply(&self) -> usize {
        self.ply
    }
}

#[derive(Debug)]
struct ProbCutSample {
    ply: u32,
    shallow_depth: Depth,
    shallow_score: Score,
    deep_depth: Depth,
}

struct MidgameCsvWriter<W> {
    writer: W,
}

impl MidgameCsvWriter<BufWriter<File>> {
    fn create(path: &Path) -> anyhow::Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("failed to create output file '{}'", path.display()))?;
        Self::new(BufWriter::new(file))
    }
}

impl<W: Write> MidgameCsvWriter<W> {
    fn new(mut writer: W) -> anyhow::Result<Self> {
        writeln!(writer, "{MIDGAME_CSV_HEADER}")?;
        Ok(Self { writer })
    }

    fn write_sample(
        &mut self,
        game: usize,
        sample: &ProbCutSample,
        deep_score: Score,
    ) -> io::Result<()> {
        writeln!(
            self.writer,
            "{game},{},{},{},{},{},{}",
            sample.ply,
            sample.shallow_depth,
            sample.shallow_score.to_disc_diff_f32(),
            sample.deep_depth,
            deep_score.to_disc_diff_f32(),
            (deep_score - sample.shallow_score).to_disc_diff_f32()
        )
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

#[must_use]
pub(crate) fn game_fold(game: u64) -> usize {
    // SplitMix64 gives stable, well-mixed folds even when game ids are
    // consecutive input line numbers.
    let mut value = game.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((value ^ (value >> 31)) % NUM_FOLDS as u64) as usize
}

fn open_game_input(path: &Path) -> anyhow::Result<BufReader<File>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open input file '{}'", path.display()))?;
    Ok(BufReader::new(file))
}

fn read_game_lines(
    reader: impl BufRead,
    path: &Path,
    mut visit: impl FnMut(usize, &[Square]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    for (line_index, line_result) in reader.lines().enumerate() {
        let game = line_index + 1;
        let line = line_result
            .with_context(|| format!("failed to read line {game} of '{}'", path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let moves =
            Square::parse_sequence(line).map_err(|e| anyhow::anyhow!("Line {game}: {e}"))?;
        visit(game, &moves).with_context(|| format!("Line {game}"))?;
    }
    Ok(())
}

pub(crate) fn for_each_game_line(
    path: &Path,
    visit: impl FnMut(usize, &[Square]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    read_game_lines(open_game_input(path)?, path, visit)
}

/// Replays `moves` from the initial position, passing every reached position
/// and move to `visit` before playing it, and returns the final position and
/// side to move.
///
/// Moves left over after the game has ended are ignored.
///
/// # Errors
///
/// Returns an error if a move is illegal when reached.
pub(crate) fn replay(
    moves: &[Square],
    mut visit: impl FnMut(&Board, PlayerColor, Square),
) -> anyhow::Result<(Board, PlayerColor)> {
    let mut game = GameState::new();

    for &sq in moves {
        if game.is_game_over() {
            break;
        }

        visit(game.board(), game.side_to_move(), sq);
        game.make_move(sq).map_err(anyhow::Error::msg)?;
    }

    Ok((*game.board(), game.side_to_move()))
}

/// Records an exact midgame position and returns whether it has not been sampled yet.
///
/// Symmetric positions remain distinct because the neural evaluation is not
/// guaranteed to produce identical depth-limited scores for every orientation.
fn insert_midgame_position(visited: &mut HashSet<Board>, board: &Board) -> bool {
    visited.insert(*board)
}

/// Scores `board` at every depth in `0..=max_depth` strictly below the empty
/// count so no sample comes from an exact endgame solve.
///
/// Depth `0` uses static evaluation rather than a one-ply search, matching how
/// the engine resolves a shallow depth of `0`.
fn depth_scores(
    search: &mut Search,
    eval: &Eval,
    board: &Board,
    max_depth: Depth,
) -> Vec<(Depth, Score)> {
    let n_empties = board.empty_count();

    (0..=max_depth)
        .filter(|&depth| depth < n_empties)
        .map(|depth| {
            let score = if depth == 0 {
                eval.evaluate_simple(board, EvalMode::Main)
            } else {
                let level = Level::uniform(depth, depth);
                let run_options = SearchRunOptions::with_level(level)
                    .disable_probcut()
                    .with_eval_mode(EvalMode::Main);
                search
                    .run(board, &run_options)
                    .score()
                    .expect("search returned no legal move")
            };
            (depth, score)
        })
        .collect()
}

pub(crate) fn execute(input: &Path, output: &Path) -> anyhow::Result<()> {
    let reader = open_game_input(input)?;
    let mut writer = MidgameCsvWriter::create(output)?;

    let options = SearchOptions::new(TT_SIZE_MB);
    let mut search = Search::new(&options);
    let eval = search.eval().clone();
    let mut visited: HashSet<Board> = HashSet::new();

    read_game_lines(reader, input, |game, moves| {
        let mut samples: Vec<(ProbCutSample, Score)> = Vec::new();
        replay(moves, |board, _, _| {
            if !insert_midgame_position(&mut visited, board) {
                return;
            }

            // A fresh TT per position keeps deep results of previously analyzed
            // positions from leaking into the shallow searches.
            search.init();
            let depth_scores = depth_scores(&mut search, &eval, board, MAX_SEARCH_DEPTH);

            let ply = INITIAL_EMPTY_COUNT as u32 - board.empty_count();
            for (shallow_depth, shallow_score) in depth_scores
                .iter()
                .filter(|(depth, _)| *depth <= MAX_SHALLOW_DEPTH)
            {
                samples.extend(
                    depth_scores
                        .iter()
                        .filter(|(deep_depth, _)| {
                            *deep_depth > *shallow_depth + MIN_DEPTH_DIFFERENCE
                        })
                        .map(|(deep_depth, deep_score)| {
                            (
                                ProbCutSample {
                                    ply,
                                    shallow_depth: *shallow_depth,
                                    shallow_score: *shallow_score,
                                    deep_depth: *deep_depth,
                                },
                                *deep_score,
                            )
                        }),
                );
            }
        })?;

        for (sample, deep_score) in &samples {
            writer.write_sample(game, sample, *deep_score)?;
        }
        writer.flush()?;

        println!("Processed {game} lines");
        Ok(())
    })?;

    println!("ProbCut training data generation completed successfully");
    Ok(())
}

pub(crate) fn read_midgame_samples(
    path: &Path,
    mut visit: impl FnMut(MidgameSample),
) -> anyhow::Result<()> {
    let file = File::open(path)
        .with_context(|| format!("failed to open input file '{}'", path.display()))?;
    let reader = BufReader::new(file);

    for (line_no, line_result) in reader.lines().enumerate() {
        let line_number = line_no + 1;
        let line = line_result.with_context(|| format!("failed to read line {line_number}"))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with("game,") {
            continue;
        }
        if line.starts_with("ply,") {
            anyhow::bail!(
                "CSV has no game column; regenerate it with the current `probcut` command"
            );
        }

        let sample = parse_midgame_row(line)
            .ok_or_else(|| anyhow::anyhow!("invalid CSV row at line {line_number}: '{line}'"))?;
        visit(sample);
    }

    Ok(())
}

fn parse_midgame_row(line: &str) -> Option<MidgameSample> {
    let mut fields = line.split(',');
    let game = fields.next()?.trim().parse().ok()?;
    let ply = fields.next()?.trim().parse().ok()?;
    let shallow_depth = fields.next()?.trim().parse().ok()?;
    let shallow_score: f64 = fields.next()?.trim().parse().ok()?;
    let deep_depth = fields.next()?.trim().parse().ok()?;
    let deep_score: f64 = fields.next()?.trim().parse().ok()?;
    let _f32_diff = fields.next()?;
    if fields.next().is_some() {
        return None;
    }

    MidgameSample::new(
        game,
        ply,
        shallow_depth,
        deep_depth,
        deep_score - shallow_score,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempFile {
        path: PathBuf,
    }

    impl TempFile {
        fn new(name: &str) -> Self {
            let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "neural-reversi-probcut-{name}-{}-{id}.txt",
                std::process::id()
            ));
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    #[test]
    fn csv_rows_include_the_input_game_line_number() -> anyhow::Result<()> {
        let sample = ProbCutSample {
            ply: 12,
            shallow_depth: 2,
            shallow_score: Score::from_raw(320),
            deep_depth: 7,
        };
        let mut writer = MidgameCsvWriter::new(Vec::new())?;

        writer.write_sample(17, &sample, Score::from_raw(-128))?;
        writer.flush()?;

        assert_eq!(
            String::from_utf8(writer.writer)?,
            format!("{MIDGAME_CSV_HEADER}\n17,12,2,1.25,7,-0.5,-1.75\n")
        );
        Ok(())
    }

    #[test]
    fn midgame_csv_writer_round_trips_through_reader() -> anyhow::Result<()> {
        let csv_file = TempFile::new("midgame-round-trip");
        {
            let mut writer = MidgameCsvWriter::create(csv_file.path())?;
            writer.write_sample(
                17,
                &ProbCutSample {
                    ply: 12,
                    shallow_depth: 2,
                    shallow_score: Score::from_raw(320),
                    deep_depth: 7,
                },
                Score::from_raw(-128),
            )?;
            writer.flush()?;
        }

        let mut samples = Vec::new();
        read_midgame_samples(csv_file.path(), |sample| samples.push(sample))?;
        assert_eq!(samples, [MidgameSample::new(17, 12, 2, 7, -1.75).unwrap()]);

        let legacy = TempFile::new("midgame-legacy");
        fs::write(
            legacy.path(),
            "ply,shallow_depth,shallow_score,deep_depth,deep_score,diff\n",
        )?;
        let err = read_midgame_samples(legacy.path(), |_| {}).unwrap_err();
        assert!(err.to_string().contains("CSV has no game column"));
        Ok(())
    }

    #[test]
    fn parses_explicit_game_ids_and_recomputes_diff() {
        let sample = parse_midgame_row("17,12,2,1.25,7,-0.5,999").unwrap();

        assert_eq!(sample.game, 17);
        assert_eq!(sample.ply(), 12);
        assert_eq!(sample.shallow_depth, 2);
        assert_eq!(sample.deep_depth, 7);
        assert_eq!(sample.diff, -1.75);
        assert!(parse_midgame_row("17,60,2,1.25,7,-0.5,-1.75").is_none());
        assert!(parse_midgame_row("17,12,2,1.25,7,-0.5").is_none());
        assert!(parse_midgame_row("17,12,2,1.25,7,-0.5,-1.75,0").is_none());
    }

    #[test]
    fn for_each_game_line_preserves_one_based_ids_across_blank_lines() -> anyhow::Result<()> {
        let games_file = TempFile::new("game-lines");
        fs::write(games_file.path(), "\nf5d6\n   \nf5f6\n")?;

        let mut seen = Vec::new();
        for_each_game_line(games_file.path(), |game, moves| {
            seen.push((game, moves.to_vec()));
            Ok(())
        })?;

        assert_eq!(
            seen,
            vec![
                (2, vec![Square::F5, Square::D6]),
                (4, vec![Square::F5, Square::F6]),
            ]
        );
        Ok(())
    }

    #[test]
    fn game_folds_are_stable_and_non_degenerate() {
        // Pinned values guard the cross-run stability that game-blocked splits
        // rely on.
        assert_eq!(
            (1..=8).map(game_fold).collect::<Vec<_>>(),
            [0, 0, 3, 3, 3, 2, 2, 2]
        );
        let assignments: Vec<_> = (1..=100).map(game_fold).collect();
        for fold in 0..NUM_FOLDS {
            assert!(assignments.contains(&fold));
        }
    }

    #[test]
    fn replay_rejects_illegal_move() {
        let moves = Square::parse_sequence("f5f5").unwrap();

        assert!(replay(&moves, |_, _, _| {}).is_err());
    }

    #[test]
    fn midgame_dedup_keeps_symmetric_positions_distinct() {
        let moves = Square::parse_sequence("f5d6c4d3").unwrap();
        let rotated_moves: Vec<_> = moves
            .iter()
            .map(|sq| Square::from_u8((63 - sq.index()) as u8).unwrap())
            .collect();
        let board = replay(&moves, |_, _, _| {}).unwrap().0;
        let rotated_board = replay(&rotated_moves, |_, _, _| {}).unwrap().0;

        assert_eq!(rotated_board, board.rotate_180_clockwise());
        assert_eq!(board.unique(), rotated_board.unique());

        let mut visited = HashSet::new();
        assert!(insert_midgame_position(&mut visited, &board));
        assert!(insert_midgame_position(&mut visited, &rotated_board));
        assert!(!insert_midgame_position(&mut visited, &board));
    }

    #[test]
    fn execute_preserves_existing_output_when_input_open_fails() -> anyhow::Result<()> {
        let missing = TempFile::new("missing-input");
        let output = TempFile::new("existing-output");
        fs::write(output.path(), "prior,csv,content\n")?;

        assert!(execute(missing.path(), output.path()).is_err());
        assert_eq!(fs::read_to_string(output.path())?, "prior,csv,content\n");
        Ok(())
    }
}
