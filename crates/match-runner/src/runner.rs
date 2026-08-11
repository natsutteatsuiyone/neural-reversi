//! Match execution and game management.

use std::{
    cmp::Ordering as CmpOrdering,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use indicatif::ProgressBar;
use reversi_core::{disc::Disc, game_state::GameState, square::Square};

use crate::config::{Config, parse_engine_command, read_opening_file};
use crate::display;
use crate::engine::GtpEngine;
use crate::error::{MatchRunnerError, Result};
use crate::sprt::{SprtConfig, SprtResult, SprtStatus};
use crate::statistics::{MatchStatistics, MatchWinner, PentanomialFrequencies};
use crate::time_tracker::TimeTracker;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static INTERRUPT_HANDLER: OnceLock<std::result::Result<(), String>> = OnceLock::new();

fn install_interrupt_handler() -> Result<()> {
    INTERRUPTED.store(false, Ordering::SeqCst);
    match INTERRUPT_HANDLER.get_or_init(|| {
        ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
            .map_err(|error| error.to_string())
    }) {
        Ok(()) => Ok(()),
        Err(message) => Err(MatchRunnerError::Config(format!(
            "Failed to install Ctrl-C handler: {message}"
        ))),
    }
}

fn check_interrupted() -> Result<()> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        Err(MatchRunnerError::Interrupted)
    } else {
        Ok(())
    }
}

fn sprt_snapshot(
    sprt: Option<&SprtConfig>,
    frequencies: &PentanomialFrequencies,
) -> Option<SprtResult> {
    sprt.map(|config| config.evaluate(frequencies))
}

fn contextualize_game_error(error: MatchRunnerError, game_number: usize) -> MatchRunnerError {
    match error {
        MatchRunnerError::Interrupted => MatchRunnerError::Interrupted,
        MatchRunnerError::Timeout(message) => MatchRunnerError::Timeout(message),
        other => MatchRunnerError::Game(format!("Fatal error in game {game_number}: {other}")),
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
}

struct MatchResult {
    result: GameResult,
    score: i32,
}

pub(crate) fn run_match(config: &Config) -> Result<()> {
    install_interrupt_handler()?;
    let openings = read_opening_file(&config.opening_file)?;
    if openings.is_empty() {
        return Err(MatchRunnerError::Config(
            "The opening file doesn't contain any valid positions.".to_string(),
        ));
    }

    let mut engines = initialize_engines(config)?;
    let engine_names = (engines.0.name(), engines.1.name());
    let mut time_tracker =
        TimeTracker::new(config.main_time, config.byoyomi_time, config.byoyomi_stones);
    let sprt = config.sprt_config()?;
    let total_games = openings.len() * 2;
    let mut statistics = MatchStatistics::default();
    let mut sprt_frequencies = PentanomialFrequencies::default();

    display::show_match_header()?;
    display::update_live_visualization(
        &statistics,
        &engine_names.0,
        &engine_names.1,
        sprt_snapshot(sprt.as_ref(), &sprt_frequencies).as_ref(),
    )?;
    let progress_bar = display::create_progress_bar(total_games as u64);

    for (opening_idx, opening) in openings.iter().enumerate() {
        let result = check_interrupted().and_then(|()| {
            play_opening_pair(
                &mut engines,
                &mut statistics,
                &engine_names,
                opening,
                opening_idx,
                &progress_bar,
                &mut time_tracker,
                &mut sprt_frequencies,
                sprt.as_ref(),
            )
        });
        if let Err(error) = result {
            progress_bar.finish_and_clear();
            if statistics.total_games() > 0 {
                let _ = display::clear_screen();
                let final_sprt = sprt
                    .as_ref()
                    .map(|config| (config, config.evaluate(&sprt_frequencies)));
                let _ =
                    statistics.print_final_results(&engine_names.0, &engine_names.1, final_sprt);
            }
            return Err(error);
        }

        if let Some(snapshot) = sprt_snapshot(sprt.as_ref(), &sprt_frequencies)
            && snapshot.status != SprtStatus::Continue
        {
            break;
        }
    }

    progress_bar.finish_and_clear();
    display::clear_screen()?;
    let final_sprt = sprt
        .as_ref()
        .map(|config| (config, config.evaluate(&sprt_frequencies)));
    statistics.print_final_results(&engine_names.0, &engine_names.1, final_sprt)?;
    Ok(())
}

fn play_game(
    black_engine: &mut GtpEngine,
    white_engine: &mut GtpEngine,
    opening: &str,
    time_tracker: &mut TimeTracker,
) -> Result<MatchResult> {
    black_engine.clear_board()?;
    white_engine.clear_board()?;
    time_tracker.reset();

    if time_tracker.is_enabled() {
        black_engine.time_settings(
            time_tracker.main_time_secs(),
            time_tracker.byoyomi_time_secs(),
            time_tracker.byoyomi_stones(),
        )?;
        white_engine.time_settings(
            time_tracker.main_time_secs(),
            time_tracker.byoyomi_time_secs(),
            time_tracker.byoyomi_stones(),
        )?;
    }

    let mut game_state = GameState::new();
    apply_opening_moves(&mut game_state, black_engine, white_engine, opening)?;

    while !game_state.is_game_over() {
        check_interrupted()?;
        let is_black = game_state.side_to_move() == Disc::Black;
        let current_color = if is_black { "black" } else { "white" };

        if time_tracker.is_enabled() {
            let (black_time, black_stones) = time_tracker.black_time_left();
            let (white_time, white_stones) = time_tracker.white_time_left();
            black_engine.time_left("black", black_time, black_stones)?;
            white_engine.time_left("black", black_time, black_stones)?;
            black_engine.time_left("white", white_time, white_stones)?;
            white_engine.time_left("white", white_time, white_stones)?;
        }

        time_tracker.start_move();
        let mv = if is_black {
            black_engine.genmove("black")?
        } else {
            white_engine.genmove("white")?
        };

        if !time_tracker.end_move(is_black) && time_tracker.is_enabled() {
            return Ok(if is_black {
                MatchResult {
                    result: GameResult::WhiteWin,
                    score: -64,
                }
            } else {
                MatchResult {
                    result: GameResult::BlackWin,
                    score: 64,
                }
            });
        }

        execute_move(
            &mut game_state,
            black_engine,
            white_engine,
            &mv,
            current_color,
        )?;
    }

    let (black_count, white_count) = game_state.get_score();
    let (result, score) = match black_count.cmp(&white_count) {
        CmpOrdering::Greater => (GameResult::BlackWin, 64 - white_count as i32 * 2),
        CmpOrdering::Less => (GameResult::WhiteWin, black_count as i32 * 2 - 64),
        CmpOrdering::Equal => (GameResult::Draw, 0),
    };
    Ok(MatchResult { result, score })
}

fn apply_opening_moves(
    game_state: &mut GameState,
    black_engine: &mut GtpEngine,
    white_engine: &mut GtpEngine,
    opening: &str,
) -> Result<()> {
    let moves = Square::parse_sequence(opening)
        .map_err(|error| MatchRunnerError::Game(format!("Invalid opening sequence: {error}")))?;

    for square in moves {
        let color = if game_state.side_to_move() == Disc::Black {
            "black"
        } else {
            "white"
        };
        game_state
            .make_move(square)
            .map_err(MatchRunnerError::Game)?;

        let mv = square.to_string();
        black_engine.play(color, &mv)?;
        white_engine.play(color, &mv)?;
    }
    Ok(())
}

fn execute_move(
    game_state: &mut GameState,
    black_engine: &mut GtpEngine,
    white_engine: &mut GtpEngine,
    mv: &str,
    current_color: &str,
) -> Result<()> {
    let move_to_send = if mv.eq_ignore_ascii_case("pass") {
        game_state.make_pass().map_err(MatchRunnerError::Game)?;
        "pass"
    } else {
        let square = mv
            .parse::<Square>()
            .map_err(|_| MatchRunnerError::Game(format!("Invalid move: {mv}")))?;
        game_state
            .make_move(square)
            .map_err(MatchRunnerError::Game)?;
        mv
    };

    let opponent_engine = if current_color == "black" {
        white_engine
    } else {
        black_engine
    };
    opponent_engine.play(current_color, move_to_send)
}

fn initialize_engines(config: &Config) -> Result<(GtpEngine, GtpEngine)> {
    let (engine1_program, engine1_args) = parse_engine_command(&config.engine1);
    let (engine2_program, engine2_args) = parse_engine_command(&config.engine2);
    let move_timeout = config.move_timeout.map(Duration::from_secs);

    let engine1 = GtpEngine::new(
        &engine1_program,
        &engine1_args,
        config.engine1_working_dir.clone(),
        move_timeout,
    )?;
    let engine2 = GtpEngine::new(
        &engine2_program,
        &engine2_args,
        config.engine2_working_dir.clone(),
        move_timeout,
    )?;
    Ok((engine1, engine2))
}

#[allow(clippy::too_many_arguments)]
fn play_opening_pair(
    engines: &mut (GtpEngine, GtpEngine),
    statistics: &mut MatchStatistics,
    engine_names: &(String, String),
    opening: &str,
    opening_idx: usize,
    progress_bar: &ProgressBar,
    time_tracker: &mut TimeTracker,
    sprt_frequencies: &mut PentanomialFrequencies,
    sprt: Option<&SprtConfig>,
) -> Result<()> {
    let mut first_result = None;

    for game_round in 0..2 {
        let is_swapped = game_round == 1;
        let game_number = opening_idx * 2 + game_round + 1;
        let (black_engine, white_engine) = if is_swapped {
            (&mut engines.1, &mut engines.0)
        } else {
            (&mut engines.0, &mut engines.1)
        };

        let match_result = play_game(black_engine, white_engine, opening, time_tracker)
            .map_err(|error| contextualize_game_error(error, game_number))?;
        let winner = match match_result.result {
            GameResult::BlackWin if is_swapped => MatchWinner::Engine2,
            GameResult::BlackWin => MatchWinner::Engine1,
            GameResult::WhiteWin if is_swapped => MatchWinner::Engine1,
            GameResult::WhiteWin => MatchWinner::Engine2,
            GameResult::Draw => MatchWinner::Draw,
        };
        let score = if is_swapped {
            -match_result.score
        } else {
            match_result.score
        };

        statistics.add_result(winner, score, opening.to_string(), !is_swapped);
        let game_result = (winner, score);
        if let Some(first_result) = first_result {
            statistics.add_paired_result(first_result, game_result);
            sprt_frequencies.add_pair(first_result.0, game_result.0);
        } else {
            first_result = Some(game_result);
        }

        display::update_live_visualization(
            statistics,
            &engine_names.0,
            &engine_names.1,
            sprt_snapshot(sprt, sprt_frequencies).as_ref(),
        )?;
        progress_bar.inc(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_context_preserves_interrupted_error() {
        assert!(matches!(
            contextualize_game_error(MatchRunnerError::Interrupted, 3),
            MatchRunnerError::Interrupted
        ));
    }

    #[test]
    fn game_context_preserves_timeout_error_and_message() {
        let error =
            contextualize_game_error(MatchRunnerError::Timeout("engine stalled".to_string()), 3);
        match error {
            MatchRunnerError::Timeout(message) => assert_eq!(message, "engine stalled"),
            other => panic!("expected timeout error, got {other}"),
        }
    }

    #[test]
    fn game_context_wraps_errors_with_game_number() {
        let error =
            contextualize_game_error(MatchRunnerError::Engine("bad response".to_string()), 3);
        match error {
            MatchRunnerError::Game(message) => {
                assert_eq!(message, "Fatal error in game 3: Engine error: bad response");
            }
            other => panic!("expected game error, got {other}"),
        }
    }
}
