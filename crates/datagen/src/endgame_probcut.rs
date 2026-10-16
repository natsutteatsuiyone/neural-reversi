//! Endgame ProbCut tracing, and held-out positions for evaluation.
//!
//! Games are split by [`game_fold`]. Fold [`HOLDOUT_FOLD`] is kept for
//! validation and evaluation; the other folds are used for fitting. Traced
//! positions are labeled by exact solves, so any game records can supply
//! roots.

#[cfg(feature = "trace")]
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
};

use reversi_core::{board::Board, disc::Disc, probcut::VERIFY_DEPTH, square::Square};
#[cfg(feature = "trace")]
use reversi_core::{
    eval::{EVAL_FILE_NAME, EvalMode},
    level::{ENDGAME_STEPS, Level},
    search::{Search, SearchRunOptions, SearchSharedResources, options::SearchOptions},
    types::Scoref,
};

use crate::fit_probcut::{NUM_FOLDS, game_fold};
use crate::probcut::replay;

/// Game fold kept out of every fit.
pub const HOLDOUT_FOLD: usize = NUM_FOLDS - 1;

/// CSV header of `endgame-trace` output, whose columns `fit-endgame-probcut`
/// reads by position.
pub fn trace_header() -> String {
    format!(
        "game,empties,s0,s{VERIFY_DEPTH},exact,beta,selectivity,cut_node,engine_cut,replayed_cut"
    )
}

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

/// The positions of one game whose empty count is in a given range.
///
/// Each entry holds the board, the side to move, the move played, and the
/// game's final result from the side to move. The result is the exact score
/// only if the game is played perfectly from that position on.
struct GamePositions {
    game: usize,
    positions: Vec<(Board, Disc, Square, i32)>,
}

/// Parses `input` into games (1-based line numbers as game ids), keeping the
/// positions with `empties` empty squares.
fn read_games(
    input: &str,
    empties: std::ops::RangeInclusive<u32>,
) -> io::Result<Vec<GamePositions>> {
    let reader = BufReader::new(File::open(input).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open input file '{input}': {e}"),
        )
    })?);
    let mut games = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let game = line_no + 1;
        let moves = Square::parse_sequence(line)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Line {game}: {e}")))?;
        let mut positions = Vec::new();
        let mut ply_moves = moves.iter();
        let (final_board, final_side) = replay(&moves, |board, side| {
            let sq = *ply_moves.next().unwrap();
            if empties.contains(&board.get_empty_count()) {
                positions.push((*board, side, sq, 0));
            }
        })
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Line {game}: {e}")))?;
        let final_score = final_board.solve(final_board.get_empty_count());
        for (_, side, _, result) in &mut positions {
            *result = if *side == final_side {
                final_score
            } else {
                -final_score
            };
        }
        games.push(GamePositions { game, positions });
    }
    Ok(games)
}

/// Picks one position per game from the holdout fold (with `fit_folds`, from
/// the fitting folds).
///
/// The empty count cycles through `empties` by game id, so the picked set
/// covers the range evenly.
fn pick_roots(
    games: &[GamePositions],
    empties: std::ops::RangeInclusive<u32>,
    fit_folds: bool,
) -> impl Iterator<Item = (usize, &(Board, Disc, Square, i32))> {
    let span = empties.end() - empties.start() + 1;
    games
        .iter()
        .filter(move |g| (game_fold(g.game as u64) == HOLDOUT_FOLD) != fit_folds)
        .filter_map(move |g| {
            let target = empties.start() + g.game as u32 % span;
            let position = g
                .positions
                .iter()
                .find(|(board, ..)| board.get_empty_count() == target)?;
            Some((g.game, position))
        })
}

/// Writes one position per game (see `pick_roots`) as an OBF line, scored
/// with the game's move and result, for evaltest's per-level report.
///
/// The score is exact only if the game is played perfectly from the position
/// on. Otherwise, run evaltest with `--solve-reference`.
pub fn export_holdout(
    input: &str,
    output: &str,
    count: usize,
    empties: std::ops::RangeInclusive<u32>,
    fit_folds: bool,
) -> io::Result<()> {
    let games = read_games(input, empties.clone())?;
    let mut writer = BufWriter::new(File::create(output)?);
    let mut written = 0;
    for (_, (board, side, sq, result)) in pick_roots(&games, empties, fit_folds).take(count) {
        let cells: String = Square::iter()
            .map(|cell| board.get_disc_at(cell, *side).to_char())
            .collect();
        writeln!(writer, "{cells} {}; {sq:?}:{result:+};", side.to_char())?;
        written += 1;
    }
    writer.flush()?;
    eprintln!("Wrote {written} held-out positions to {output}");
    Ok(())
}

/// Returns `([static evaluation, verification score], exact score)` of
/// `board`.
///
/// The first two are computed as endgame ProbCut sees them: main network, no
/// selective pruning.
#[cfg(feature = "trace")]
fn label(search: &mut Search, board: &Board) -> ([Scoref; 2], i32) {
    use reversi_core::probcut::VERIFY_DEPTH as D;
    search.init();
    let eval = search
        .eval()
        .evaluate_simple(board, EvalMode::Main)
        .to_disc_diff_f32();
    let options = SearchRunOptions::with_level(Level::uniform(D, D))
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
        .expect("sampled positions have a legal move") as i32;
    ([eval, score], exact)
}

/// Which roots `endgame-trace` searches and which attempts it keeps.
#[cfg(feature = "trace")]
pub struct TraceOptions {
    /// Number of roots to search.
    pub roots: usize,
    /// Empty counts the roots are drawn from, cycled by game id.
    pub root_empties: std::ops::RangeInclusive<u32>,
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
pub fn trace(input: &str, output: &str, options: &TraceOptions) -> io::Result<()> {
    use std::collections::{HashMap, HashSet};

    use reversi_core::probcut::{self, trace::TraceRecord};
    use reversi_core::types::ScaledScore;

    let games = read_games(input, options.root_empties.clone())?;
    let root_boards: Vec<(usize, Board)> =
        pick_roots(&games, options.root_empties.clone(), options.fit_folds)
            .map(|(game, (board, ..))| (game, *board))
            .take(options.roots)
            .collect();

    // Roots run one at a time with every thread, so each drained batch of
    // records belongs to the root that produced it.
    probcut::trace::set_sampling(options.one_in, 0..=60);
    let mut search =
        Search::new(&SearchOptions::new(SEARCH_TT_MB).with_threads(Some(options.threads.max(1))));
    let mut traced: Vec<(usize, TraceRecord)> = Vec::new();
    for (index, &(game, board)) in root_boards.iter().enumerate() {
        let n_empties = board.get_empty_count();
        // An end depth of 0 for the last level stops the ladder before the exact solve.
        let mut end_depth = [n_empties; ENDGAME_STEPS];
        end_depth[ENDGAME_STEPS - 1] = 0;
        let level = Level::with_depths(n_empties, end_depth);
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
            b.get_empty_count(),
            b.hash(),
            b.player().bits(),
            b.opponent().bits(),
        )
    });
    unique.dedup();
    let unique: Vec<Board> = unique
        .chunk_by(|a, b| a.get_empty_count() == b.get_empty_count())
        .flat_map(|boards| boards.iter().take(options.max_per_empties).copied())
        .collect();
    let kept: HashSet<Board> = unique.iter().copied().collect();
    traced.retain(|(_, r)| kept.contains(&r.board));
    let (large, unique): (Vec<Board>, Vec<Board>) = unique
        .into_iter()
        .partition(|b| b.get_empty_count() >= SEQUENTIAL_LABEL_EMPTIES);
    let labels: Mutex<HashMap<Board, ([Scoref; 2], i32)>> = Mutex::new(HashMap::new());
    let started = std::time::Instant::now();
    for (index, board) in large.iter().enumerate() {
        let label = label(&mut search, board);
        labels.lock().unwrap().insert(*board, label);
        if index.is_multiple_of(10) {
            eprintln!(
                "Labeled {index}/{} large positions ({} empties, {:.0}s)",
                large.len(),
                board.get_empty_count(),
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

    let to_raw = |score: Scoref| (score * ScaledScore::SCALE as Scoref).round() as i32;
    let mut writer = BufWriter::new(File::create(output)?);
    writeln!(writer, "# main_weights={EVAL_FILE_NAME}")?;
    writeln!(writer, "{}", trace_header())?;
    let mut mismatches = 0;
    for (game, record) in &traced {
        let (scores, exact) = labels[&record.board];
        let n_empties = record.board.get_empty_count();
        let replayed_cut = {
            let eval = ScaledScore::from_raw(to_raw(scores[0]));
            let window = probcut::endgame_probcut_window(
                n_empties,
                record.selectivity,
                record.cut_node,
                record.beta,
                eval,
            );
            window.pc_beta < ScaledScore::MAX
                && eval >= window.screen_beta
                && to_raw(scores[1]) >= window.pc_beta.value()
        };
        let engine_cut = record.result.is_some();
        mismatches += usize::from(engine_cut != replayed_cut);
        writeln!(
            writer,
            "{game},{n_empties},{},{},{exact},{},{},{},{},{}",
            scores[0],
            scores[1],
            record.beta.to_disc_diff_f32(),
            record.selectivity.as_u8(),
            u8::from(record.cut_node),
            u8::from(engine_cut),
            u8::from(replayed_cut),
        )?;
    }
    writer.flush()?;
    eprintln!(
        "Engine and replayed decisions disagree on {mismatches} of {} attempts",
        traced.len()
    );
    Ok(())
}
