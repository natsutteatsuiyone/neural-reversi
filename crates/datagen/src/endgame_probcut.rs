//! Endgame ProbCut tracing, and held-out positions for evaluation.
//!
//! Games are split by [`game_fold`]. Fold [`HOLDOUT_FOLD`] is kept for
//! validation and evaluation; the other folds are used for fitting. Traced
//! positions are labeled by exact solves, so any game records can supply
//! roots.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::ops::RangeInclusive;
use std::path::Path;
#[cfg(feature = "trace")]
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use anyhow::Context;
use reversi_core::{
    board::Board, eval::EVAL_FILE_NAME, player_color::PlayerColor, probcut::VERIFY_DEPTH,
    square::Square,
};
#[cfg(feature = "trace")]
use reversi_core::{
    eval::EvalMode,
    level::Level,
    search::{Search, SearchRunOptions, SearchSharedResources, options::SearchOptions},
};
#[cfg(any(feature = "trace", test))]
use reversi_core::{probcut::Selectivity, score::Score};

use crate::probcut::{HOLDOUT_FOLD, for_each_game_line, game_fold, replay};

const WEIGHTS_HEADER_PREFIX: &str = "# main_weights=";

/// Per-worker transposition table.
#[cfg(feature = "trace")]
const WORKER_TT_MB: usize = 64;

/// Traced positions with at least this many empties are labeled one at a
/// time using every thread. Solving them single-threaded takes too long.
#[cfg(feature = "trace")]
const SEQUENTIAL_LABEL_EMPTIES: u32 = 24;

/// Transposition table of the multi-threaded tracing and labeling search.
#[cfg(feature = "trace")]
const SEARCH_TT_MB: usize = 1024;

/// One traced ProbCut attempt. Scores are in discs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TraceRow {
    pub game: u64,
    pub empties: u32,
    /// Static evaluation.
    pub eval: f64,
    /// Verification score.
    pub score: f64,
    pub exact: f64,
    pub beta: f64,
    pub cut_node: bool,
}

#[cfg(any(feature = "trace", test))]
struct TraceWriteRow {
    game: usize,
    empties: u32,
    eval: Score,
    score: Score,
    exact: i32,
    beta: Score,
    selectivity: Selectivity,
    cut_node: bool,
    engine_cut: bool,
    replayed_cut: bool,
}

#[cfg(any(feature = "trace", test))]
struct EndgameTraceCsvWriter<W> {
    writer: W,
}

#[cfg(any(feature = "trace", test))]
impl EndgameTraceCsvWriter<BufWriter<File>> {
    fn create(path: &Path) -> anyhow::Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("failed to create output file '{}'", path.display()))?;
        Self::new(BufWriter::new(file))
    }
}

#[cfg(any(feature = "trace", test))]
impl<W: Write> EndgameTraceCsvWriter<W> {
    fn new(mut writer: W) -> anyhow::Result<Self> {
        writeln!(writer, "{WEIGHTS_HEADER_PREFIX}{EVAL_FILE_NAME}")?;
        writeln!(writer, "{}", trace_header())?;
        Ok(Self { writer })
    }

    fn write_row(&mut self, row: &TraceWriteRow) -> std::io::Result<()> {
        writeln!(
            self.writer,
            "{},{},{},{},{},{},{},{},{},{}",
            row.game,
            row.empties,
            row.eval.to_disc_diff_f32(),
            row.score.to_disc_diff_f32(),
            row.exact,
            row.beta.to_disc_diff_f32(),
            row.selectivity.as_u8(),
            u8::from(row.cut_node),
            u8::from(row.engine_cut),
            u8::from(row.replayed_cut),
        )
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

/// The positions of one game whose empty count is in a given range.
///
/// Each entry holds the board, the side to move, and the move played.
struct GamePositions {
    game: usize,
    positions: Vec<(Board, PlayerColor, Square)>,
    /// The final disc difference from Black's side, or `None` if the record
    /// stops before the game is over (a resignation, a time loss, a cut line).
    result: Option<i32>,
}

impl GamePositions {
    /// Returns the game's result from `side`, if the game was played out.
    ///
    /// It is the exact score of a position only if the game is played
    /// perfectly from that position on.
    fn result_for(&self, side: PlayerColor) -> Option<i32> {
        self.result.map(|black| match side {
            PlayerColor::Black => black,
            PlayerColor::White => -black,
        })
    }
}

fn trace_header() -> String {
    format!(
        "game,empties,s0,s{VERIFY_DEPTH},exact,beta,selectivity,cut_node,engine_cut,replayed_cut"
    )
}

fn read_games(input: &Path, empties: RangeInclusive<u32>) -> anyhow::Result<Vec<GamePositions>> {
    let mut games = Vec::new();
    for_each_game_line(input, |game, moves| {
        let mut positions = Vec::new();
        let (final_board, final_side) = replay(moves, |board, side, sq| {
            if empties.contains(&board.empty_count()) {
                positions.push((*board, side, sq));
            }
        })?;
        let result = final_board.is_game_over().then(|| {
            let score = final_board
                .solve_disc_diff(final_board.empty_count())
                .value();
            match final_side {
                PlayerColor::Black => score,
                PlayerColor::White => -score,
            }
        });
        games.push(GamePositions {
            game,
            positions,
            result,
        });
        Ok(())
    })?;
    Ok(games)
}

/// Picks one position per game from the holdout fold (with `fit_folds`, from
/// the fitting folds).
///
/// The empty count cycles through `empties` by game id, so the picked set
/// covers the range evenly.
fn pick_roots(
    games: &[GamePositions],
    empties: RangeInclusive<u32>,
    fit_folds: bool,
) -> impl Iterator<Item = (&GamePositions, &(Board, PlayerColor, Square))> {
    let span = empties.end() - empties.start() + 1;
    games
        .iter()
        .filter(move |g| (game_fold(g.game as u64) == HOLDOUT_FOLD) != fit_folds)
        .filter_map(move |g| {
            let target = empties.start() + g.game as u32 % span;
            let position = g
                .positions
                .iter()
                .find(|(board, ..)| board.empty_count() == target)?;
            Some((g, position))
        })
}

/// Writes one position per game (see `pick_roots`) as an OBF line, scored
/// with the game's move and result, for search-test's per-level report.
///
/// Games whose record stops before the end have no result and are left out.
/// The score is exact only if the game is played perfectly from the position
/// on. Otherwise, run search-test with `--solve-reference`.
pub(crate) fn export_holdout(
    input: &Path,
    output: &Path,
    count: usize,
    empties: RangeInclusive<u32>,
    fit_folds: bool,
) -> anyhow::Result<()> {
    let games = read_games(input, empties.clone())?;
    let file = File::create(output)
        .with_context(|| format!("failed to create output file '{}'", output.display()))?;
    let mut writer = BufWriter::new(file);
    let mut written = 0;
    let mut unfinished = 0;
    let roots = pick_roots(&games, empties, fit_folds).filter_map(|(game, position)| {
        let result = game.result_for(position.1);
        unfinished += usize::from(result.is_none());
        Some((position, result?))
    });
    for ((board, side, sq), result) in roots.take(count) {
        let cells: String = Square::iter()
            .map(|cell| board.disc_at(cell, *side).to_char())
            .collect();
        writeln!(writer, "{cells} {}; {sq:?}:{result:+};", side.to_char())?;
        written += 1;
    }
    writer.flush()?;
    eprintln!("Wrote {written} held-out positions to {}", output.display());
    if unfinished > 0 {
        eprintln!("Skipped {unfinished} games that stop before the end and so have no result");
    }
    Ok(())
}

pub(crate) fn read_trace_rows(path: &Path) -> anyhow::Result<Vec<TraceRow>> {
    let file = File::open(path).with_context(|| format!("failed to open '{}'", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let weights_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty input"))??;
    let weights = weights_line
        .strip_prefix(WEIGHTS_HEADER_PREFIX)
        .ok_or_else(|| anyhow::anyhow!("missing weights header: '{weights_line}'"))?;
    anyhow::ensure!(
        weights == EVAL_FILE_NAME,
        "samples were generated with {weights}, but the engine uses {EVAL_FILE_NAME}"
    );

    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing CSV header"))??;
    let expected_header = trace_header();
    anyhow::ensure!(
        header == expected_header,
        "'{header}' is not the endgame-trace header '{expected_header}'"
    );

    let mut rows = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let line_number = line_no + 3;
        let line = line.with_context(|| format!("failed to read line {line_number}"))?;
        rows.push(parse_trace_row(&line, line_number)?);
    }
    Ok(rows)
}

fn parse_flag(value: &str, name: &str, line_number: usize) -> anyhow::Result<bool> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => anyhow::bail!("line {line_number}: invalid {name} flag '{value}' (expected 0 or 1)"),
    }
}

fn parse_trace_row(line: &str, line_number: usize) -> anyhow::Result<TraceRow> {
    let fields: Vec<&str> = line.split(',').collect();
    anyhow::ensure!(
        fields.len() == 10,
        "line {line_number}: expected 10 columns, got {}",
        fields.len()
    );
    let parse_f64 = |index: usize, name: &str| -> anyhow::Result<f64> {
        let value: f64 = fields[index]
            .parse()
            .with_context(|| format!("line {line_number}: bad field {index} ({name})"))?;
        anyhow::ensure!(
            value.is_finite(),
            "line {line_number}: non-finite field {index} ({name})"
        );
        Ok(value)
    };

    let game: u64 = fields[0]
        .parse()
        .with_context(|| format!("line {line_number}: bad field 0 (game)"))?;
    let empties: u32 = fields[1]
        .parse()
        .with_context(|| format!("line {line_number}: bad field 1 (empties)"))?;
    let eval = parse_f64(2, "s0")?;
    let score = parse_f64(3, "verify_score")?;
    let exact = parse_f64(4, "exact")?;
    let beta = parse_f64(5, "beta")?;
    let _selectivity: u8 = fields[6]
        .parse()
        .with_context(|| format!("line {line_number}: bad field 6 (selectivity)"))?;
    let cut_node = parse_flag(fields[7], "cut_node", line_number)?;
    let _engine_cut = parse_flag(fields[8], "engine_cut", line_number)?;
    let _replayed_cut = parse_flag(fields[9], "replayed_cut", line_number)?;

    Ok(TraceRow {
        game,
        empties,
        eval,
        score,
        exact,
        beta,
        cut_node,
    })
}

/// Returns `([static evaluation, verification score], exact score)` of `board`.
///
/// The first two are computed as endgame ProbCut sees them: main network, no
/// selective pruning.
#[cfg(feature = "trace")]
fn label(search: &mut Search, board: &Board) -> ([Score; 2], i32) {
    search.init();
    let eval = search.eval().evaluate_simple(board, EvalMode::Main);
    let options = SearchRunOptions::with_level(Level::uniform(VERIFY_DEPTH, VERIFY_DEPTH))
        .disable_probcut()
        .with_eval_mode(EvalMode::Main);
    let score = search
        .run(board, &options)
        .score()
        .expect("sampled positions have a legal move");
    search.init();
    let exact = search
        .run(board, &SearchRunOptions::with_level(Level::perfect()))
        .score()
        .expect("sampled positions have a legal move")
        .to_disc_diff()
        .value();
    ([eval, score], exact)
}

/// Which roots `endgame-trace` searches and which attempts it keeps.
#[cfg(feature = "trace")]
pub(crate) struct TraceOptions {
    /// Number of roots to search.
    pub roots: usize,
    /// Empty counts the roots are drawn from, cycled by game id.
    pub root_empties: RangeInclusive<u32>,
    /// Draw roots from the fitting folds instead of the holdout fold.
    pub fit_folds: bool,
    /// Keep attempts whose position hash is divisible by this.
    pub one_in: u64,
    /// Keep at most this many distinct positions per empty count.
    pub max_per_empties: usize,
    /// Worker threads.
    pub threads: usize,
}

/// Traces the endgame ProbCut attempts of Level1 to Level4 solves, then
/// labels each traced position with its shallow scores and exact score.
///
/// Roots are game positions chosen by [`TraceOptions`].
///
/// Each row also replays the cut decision from the labeled scores with the
/// engine's thresholds. `engine_cut` and `replayed_cut` must agree; if not,
/// the labels differ from what the search saw.
#[cfg(feature = "trace")]
pub(crate) fn trace(input: &Path, output: &Path, options: &TraceOptions) -> anyhow::Result<()> {
    use std::collections::{HashMap, HashSet};

    use reversi_core::probcut::{self, trace::TraceRecord};

    let games = read_games(input, options.root_empties.clone())?;
    let root_boards: Vec<(usize, Board)> =
        pick_roots(&games, options.root_empties.clone(), options.fit_folds)
            .map(|(game, (board, ..))| (game.game, *board))
            .take(options.roots)
            .collect();

    // Roots run one at a time with every thread, so each drained batch of
    // records belongs to the root that produced it.
    probcut::trace::set_sampling(options.one_in, 0..=60);
    let mut search =
        Search::new(&SearchOptions::new(SEARCH_TT_MB).with_threads(Some(options.threads.max(1))));
    let mut traced: Vec<(usize, TraceRecord)> = Vec::new();
    for (index, &(game, board)) in root_boards.iter().enumerate() {
        let n_empties = board.empty_count();
        let level = Level::uniform(n_empties, n_empties).stopping_at(Selectivity::Level4);
        search.init();
        search.run(&board, &SearchRunOptions::with_level(level));
        traced.extend(probcut::trace::take().into_iter().map(|r| (game, r)));
        if index.is_multiple_of(50) {
            eprintln!("Traced {index}/{} roots", root_boards.len());
        }
    }
    probcut::trace::set_sampling(0, 0..=0);
    eprintln!(
        "Traced {} ProbCut attempts from {} roots",
        traced.len(),
        root_boards.len()
    );

    // Most traced nodes have few empties. Keep at most `max_per_empties`
    // distinct positions per empty count, in hash order so that the kept set
    // is an unbiased subsample.
    let mut unique: Vec<Board> = traced.iter().map(|(_, r)| r.board).collect();
    unique.sort_unstable_by_key(|b| {
        (
            b.empty_count(),
            b.hash(),
            b.player().bits(),
            b.opponent().bits(),
        )
    });
    unique.dedup();
    let unique: Vec<Board> = unique
        .chunk_by(|a, b| a.empty_count() == b.empty_count())
        .flat_map(|boards| boards.iter().take(options.max_per_empties).copied())
        .collect();
    let kept: HashSet<Board> = unique.iter().copied().collect();
    traced.retain(|(_, r)| kept.contains(&r.board));
    let (large, unique): (Vec<Board>, Vec<Board>) = unique
        .into_iter()
        .partition(|b| b.empty_count() >= SEQUENTIAL_LABEL_EMPTIES);
    let labels: Mutex<HashMap<Board, ([Score; 2], i32)>> = Mutex::new(HashMap::new());
    let started = std::time::Instant::now();
    for (index, board) in large.iter().enumerate() {
        let label = label(&mut search, board);
        labels.lock().unwrap().insert(*board, label);
        if index.is_multiple_of(10) {
            eprintln!(
                "Labeled {index}/{} large positions ({} empties, {:.0}s)",
                large.len(),
                board.empty_count(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    drop(search);
    let next = AtomicUsize::new(0);
    // One copy of the network for every worker.
    let shared =
        SearchSharedResources::new(&SearchOptions::new(WORKER_TT_MB).with_threads(Some(1)));
    std::thread::scope(|scope| {
        for _ in 0..options.threads.max(1) {
            scope.spawn(|| {
                let mut search = Search::from_shared_resources(&shared);
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(board) = unique.get(index) else {
                        break;
                    };
                    let label = label(&mut search, board);
                    labels.lock().unwrap().insert(*board, label);
                    if index.is_multiple_of(1000) {
                        eprintln!("Labeled {index}/{} positions", unique.len());
                    }
                }
            });
        }
    });
    let labels = labels.into_inner().unwrap();

    let mut writer = EndgameTraceCsvWriter::create(output)?;
    let mut mismatches = 0;
    for (game, record) in &traced {
        let (scores, exact) = labels[&record.board];
        let n_empties = record.board.empty_count();
        let replayed_cut = {
            let eval = scores[0];
            let window = probcut::endgame_probcut_window(
                n_empties,
                record.selectivity,
                record.cut_node,
                record.beta,
                eval,
            );
            window.pc_beta < Score::MAX && eval >= window.screen_beta && scores[1] >= window.pc_beta
        };
        let engine_cut = record.result.is_some();
        mismatches += usize::from(engine_cut != replayed_cut);
        writer.write_row(&TraceWriteRow {
            game: *game,
            empties: n_empties,
            eval: scores[0],
            score: scores[1],
            exact,
            beta: record.beta,
            selectivity: record.selectivity,
            cut_node: record.cut_node,
            engine_cut,
            replayed_cut,
        })?;
    }
    writer.flush()?;
    eprintln!(
        "Engine and replayed decisions disagree on {mismatches} of {} attempts",
        traced.len()
    );
    Ok(())
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
                "neural-reversi-endgame-trace-{name}-{}-{id}.csv",
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
    fn endgame_trace_csv_writer_round_trips_through_reader() -> anyhow::Result<()> {
        let file = TempFile::new("round-trip");
        {
            let mut writer = EndgameTraceCsvWriter::create(file.path())?;
            writer.write_row(&TraceWriteRow {
                game: 42,
                empties: 20,
                eval: Score::from_raw(320),
                score: Score::from_raw(-128),
                exact: -3,
                beta: Score::from_raw(256),
                selectivity: Selectivity::Level2,
                cut_node: true,
                engine_cut: false,
                replayed_cut: false,
            })?;
            writer.flush()?;
        }

        let rows = read_trace_rows(file.path())?;
        assert_eq!(
            rows,
            [TraceRow {
                game: 42,
                empties: 20,
                eval: 1.25,
                score: -0.5,
                exact: -3.0,
                beta: 1.0,
                cut_node: true,
            }]
        );

        let wrong_weights = TempFile::new("wrong-weights");
        fs::write(
            wrong_weights.path(),
            format!("# main_weights=stale.zst\n{}\n", trace_header()),
        )?;
        assert!(
            read_trace_rows(wrong_weights.path())
                .unwrap_err()
                .to_string()
                .contains("samples were generated with stale.zst")
        );

        let bad_flag = TempFile::new("bad-flag");
        fs::write(
            bad_flag.path(),
            format!(
                "{WEIGHTS_HEADER_PREFIX}{EVAL_FILE_NAME}\n{}\n1,20,0,0,-1,0,1,2,0,0\n",
                trace_header()
            ),
        )?;
        assert!(
            read_trace_rows(bad_flag.path())
                .unwrap_err()
                .to_string()
                .contains("invalid cut_node flag")
        );

        let short_row = TempFile::new("short-row");
        fs::write(
            short_row.path(),
            format!(
                "{WEIGHTS_HEADER_PREFIX}{EVAL_FILE_NAME}\n{}\n1,20,0,0,-1,0,1,0,0\n",
                trace_header()
            ),
        )?;
        assert!(
            read_trace_rows(short_row.path())
                .unwrap_err()
                .to_string()
                .contains("expected 10 columns")
        );
        Ok(())
    }

    #[test]
    fn holdout_leaves_out_games_that_stop_before_the_end() -> anyhow::Result<()> {
        let mut game = reversi_core::game_state::GameState::new();
        let mut moves = Vec::new();
        while !game.is_game_over() {
            match game.board().legal_moves().iter().next() {
                Some(sq) => {
                    moves.push(sq);
                    game.make_move(sq).unwrap();
                }
                None => game.make_pass().unwrap(),
            }
        }
        let (black, white) = game.score();
        let result = black as i32 - white as i32;
        assert_eq!(
            game.board().empty_count(),
            0,
            "the fixture game fills the board"
        );
        let line = |moves: &[Square]| moves.iter().map(|sq| format!("{sq}")).collect::<String>();

        // Games 1 to 8 are all in the fitting folds.
        let export = |moves: &[Square]| -> anyhow::Result<String> {
            let games = TempFile::new("holdout-games");
            let obf = TempFile::new("holdout-obf");
            fs::write(
                games.path(),
                format!(
                    "{}
",
                    line(moves)
                )
                .repeat(8),
            )?;
            export_holdout(games.path(), obf.path(), 400, 20..=26, true)?;
            Ok(fs::read_to_string(obf.path())?)
        };

        let finished = export(&moves)?;
        assert_eq!(finished.lines().count(), 8);
        for position in finished.lines() {
            let black_to_move = position.contains(" X; ");
            let expected = if black_to_move { result } else { -result };
            assert!(position.ends_with(&format!(":{expected:+};")), "{position}");
        }

        // The record stops with 10 empties, after the exported positions.
        assert_eq!(export(&moves[..50])?, "");
        Ok(())
    }
}
