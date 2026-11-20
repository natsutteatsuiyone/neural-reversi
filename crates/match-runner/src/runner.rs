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
use rand::{SeedableRng, rngs::StdRng};
use reversi_core::{game_state::GameState, player_color::PlayerColor, square::Square};

use crate::config::{Config, parse_engine_command, read_opening_file};
use crate::display;
use crate::engine::GtpEngine;
use crate::error::{MatchRunnerError, Result};
use crate::opening::{Opening, board_string, random_position};
use crate::sprt::{Sprt, SprtResult, SprtStatus};
use crate::statistics::{MatchStatistics, MatchWinner};
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

pub(crate) fn check_interrupted() -> Result<()> {
    if INTERRUPTED.load(Ordering::SeqCst) {
        Err(MatchRunnerError::Interrupted)
    } else {
        Ok(())
    }
}

fn sprt_snapshot(sprt: Option<&mut Sprt>, statistics: &MatchStatistics) -> Option<SprtResult> {
    sprt.map(|sprt| statistics.update_sprt(sprt))
}

fn contextualize_game_error(error: MatchRunnerError, game_number: usize) -> MatchRunnerError {
    match error {
        MatchRunnerError::Interrupted => MatchRunnerError::Interrupted,
        MatchRunnerError::Timeout(message) => MatchRunnerError::Timeout(message),
        other => MatchRunnerError::Game(format!("Fatal error in game {game_number}: {other}")),
    }
}

/// Returns Black's score for a finished game, with the empty squares given
/// to the winner.
fn black_score(black_count: u32, white_count: u32) -> i32 {
    match black_count.cmp(&white_count) {
        CmpOrdering::Greater => 64 - white_count as i32 * 2,
        CmpOrdering::Less => black_count as i32 * 2 - 64,
        CmpOrdering::Equal => 0,
    }
}

/// Returns Black's score when `loser` runs out of time or resigns.
fn forfeit_black_score(loser: PlayerColor) -> i32 {
    match loser {
        PlayerColor::Black => -64,
        PlayerColor::White => 64,
    }
}

/// Converts Black's score into the winner and the score for engine 1.
fn engine1_result(black_score: i32, engine1_is_black: bool) -> (MatchWinner, i32) {
    let score = if engine1_is_black {
        black_score
    } else {
        -black_score
    };
    let winner = match score.cmp(&0) {
        CmpOrdering::Greater => MatchWinner::Engine1,
        CmpOrdering::Less => MatchWinner::Engine2,
        CmpOrdering::Equal => MatchWinner::Draw,
    };
    (winner, score)
}

pub(crate) fn run_match(config: &Config) -> Result<()> {
    install_interrupt_handler()?;
    let random_seed = config
        .ggs_random_discs
        .map(|_| config.seed.unwrap_or_else(rand::random));
    let openings = if let Some(discs) = config.ggs_random_discs {
        let seed = random_seed.expect("random mode has a seed");
        println!("GGS random openings seed {seed}");
        let mut rng = StdRng::seed_from_u64(seed);
        (0..config.pairs.expect("clap requires pairs"))
            .map(|_| Opening::Position(random_position(discs, &mut rng)))
            .collect::<Vec<_>>()
    } else {
        let sequences = read_opening_file(
            config
                .opening_file
                .as_ref()
                .expect("clap requires opening file"),
        )?;
        if sequences.is_empty() {
            return Err(MatchRunnerError::Config(
                "The opening file doesn't contain any valid positions.".to_string(),
            ));
        }
        validate_openings(&sequences)?;
        sequences.into_iter().map(Opening::Sequence).collect()
    };

    let mut engines = initialize_engines(config)?;
    if config.ggs_random_discs.is_some() {
        engines.0.require_setboard()?;
        engines.1.require_setboard()?;
    }
    let engine_names = (engines.0.name(), engines.1.name());
    let mut time_tracker =
        TimeTracker::new(config.main_time, config.byoyomi_time, config.byoyomi_stones);
    let mut sprt = config.sprt_config()?.map(Sprt::new);
    let total_games = openings.len() * 2;
    let mut statistics = MatchStatistics::default();
    statistics.random_seed = random_seed;
    statistics.synchro = config.ggs_random_discs.is_some();

    display::show_match_header()?;
    display::update_live_visualization(
        &statistics,
        &engine_names.0,
        &engine_names.1,
        sprt_snapshot(sprt.as_mut(), &statistics).as_ref(),
    )?;
    let progress_bar = display::create_progress_bar(total_games as u64);

    for opening in &openings {
        let result = check_interrupted().and_then(|()| {
            play_opening_pair(
                &mut engines,
                &mut statistics,
                &engine_names,
                opening,
                &progress_bar,
                &mut time_tracker,
                sprt.as_mut(),
            )
        });
        if let Err(error) = result {
            progress_bar.finish_and_clear();
            if statistics.total_games() > 0 || statistics.random_seed.is_some() {
                let _ = display::clear_screen();
                let final_sprt = sprt.as_mut().map(|sprt| {
                    let result = statistics.update_sprt(sprt);
                    (sprt.config, result)
                });
                statistics.print_final_results(&engine_names.0, &engine_names.1, final_sprt);
            }
            return Err(error);
        }

        if let Some(snapshot) = sprt_snapshot(sprt.as_mut(), &statistics)
            && snapshot.status != SprtStatus::Continue
        {
            break;
        }
    }

    progress_bar.finish_and_clear();
    display::clear_screen()?;
    let final_sprt = sprt.as_mut().map(|sprt| {
        let result = statistics.update_sprt(sprt);
        (sprt.config, result)
    });
    statistics.print_final_results(&engine_names.0, &engine_names.1, final_sprt);
    Ok(())
}

fn play_game(
    black_engine: &mut GtpEngine,
    white_engine: &mut GtpEngine,
    opening: &Opening,
    time_tracker: &mut TimeTracker,
) -> Result<i32> {
    black_engine.clear_board()?;
    white_engine.clear_board()?;
    time_tracker.reset();

    if time_tracker.is_enabled() {
        let settings = time_tracker.settings();
        black_engine.time_settings(settings)?;
        white_engine.time_settings(settings)?;
    }

    let mut game_state = match opening {
        Opening::Sequence(sequence) => {
            let mut state = GameState::new();
            apply_opening_moves(&mut state, black_engine, white_engine, sequence)?;
            state
        }
        Opening::Position(state) => {
            let board = board_string(state);
            let side = state.side_to_move();
            black_engine.setboard(&board, side)?;
            white_engine.setboard(&board, side)?;
            state.clone()
        }
    };

    while !game_state.is_game_over() {
        check_interrupted()?;
        let mover = game_state.side_to_move();

        if time_tracker.is_enabled() {
            for color in [PlayerColor::Black, PlayerColor::White] {
                let (secs, stones) = time_tracker.time_left(color);
                black_engine.time_left(color, secs, stones)?;
                white_engine.time_left(color, secs, stones)?;
            }
        }

        time_tracker.start_move();
        let mv = match mover {
            PlayerColor::Black => black_engine.genmove(mover)?,
            PlayerColor::White => white_engine.genmove(mover)?,
        };

        let flagged = !time_tracker.end_move(mover) && time_tracker.is_enabled();
        if flagged || mv.eq_ignore_ascii_case("resign") {
            return Ok(forfeit_black_score(mover));
        }

        execute_move(&mut game_state, black_engine, white_engine, &mv, mover)?;
    }

    let (black_count, white_count) = game_state.score();
    Ok(black_score(black_count, white_count))
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
        let color = game_state.side_to_move();
        game_state.make_move(square)?;

        let mv = square.to_string();
        black_engine.play(color, &mv)?;
        white_engine.play(color, &mv)?;
    }
    Ok(())
}

fn validate_openings(openings: &[String]) -> Result<()> {
    for (index, opening) in openings.iter().enumerate() {
        let moves = Square::parse_sequence(opening).map_err(|error| {
            MatchRunnerError::Config(format!("Invalid opening {}: {error}", index + 1))
        })?;
        let mut game_state = GameState::new();
        for square in moves {
            game_state.make_move(square).map_err(|error| {
                MatchRunnerError::Config(format!("Invalid opening {}: {error}", index + 1))
            })?;
        }
    }
    Ok(())
}

fn execute_move(
    game_state: &mut GameState,
    black_engine: &mut GtpEngine,
    white_engine: &mut GtpEngine,
    mv: &str,
    mover: PlayerColor,
) -> Result<()> {
    let move_to_send = if mv.eq_ignore_ascii_case("pass") {
        game_state.make_pass()?;
        "pass"
    } else {
        let square = mv
            .parse::<Square>()
            .map_err(|_| MatchRunnerError::Game(format!("Invalid move: {mv}")))?;
        game_state.make_move(square)?;
        mv
    };

    let opponent_engine = match mover {
        PlayerColor::Black => white_engine,
        PlayerColor::White => black_engine,
    };
    opponent_engine.play(mover, move_to_send)
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

fn play_opening_pair(
    engines: &mut (GtpEngine, GtpEngine),
    statistics: &mut MatchStatistics,
    engine_names: &(String, String),
    opening: &Opening,
    progress_bar: &ProgressBar,
    time_tracker: &mut TimeTracker,
    mut sprt: Option<&mut Sprt>,
) -> Result<()> {
    let mut first_result = None;

    for game_round in 0..2 {
        let is_swapped = game_round == 1;
        let (black_engine, white_engine) = if is_swapped {
            (&mut engines.1, &mut engines.0)
        } else {
            (&mut engines.0, &mut engines.1)
        };

        let game_number = statistics.total_games() as usize + 1;
        let black_score = play_game(black_engine, white_engine, opening, time_tracker)
            .map_err(|error| contextualize_game_error(error, game_number))?;
        let (winner, score) = engine1_result(black_score, !is_swapped);

        statistics.add_game(winner, score, opening.description(), !is_swapped);
        if let Some(first) = first_result {
            statistics.add_pair(first, (winner, score), opening.description());
        } else {
            first_result = Some((winner, score));
        }

        display::update_live_visualization(
            statistics,
            &engine_names.0,
            &engine_names.1,
            sprt_snapshot(sprt.as_deref_mut(), statistics).as_ref(),
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

    #[test]
    fn black_score_gives_empty_squares_to_the_winner() {
        assert_eq!(black_score(40, 20), 24);
        assert_eq!(black_score(20, 40), -24);
        assert_eq!(black_score(30, 30), 0);
        assert_eq!(black_score(13, 0), 64);
        assert_eq!(black_score(0, 13), -64);
        assert_eq!(black_score(33, 31), 2);
    }

    #[test]
    fn engine1_result_follows_the_color_engine1_plays() {
        assert_eq!(engine1_result(24, true), (MatchWinner::Engine1, 24));
        assert_eq!(engine1_result(-24, true), (MatchWinner::Engine2, -24));
        assert_eq!(engine1_result(24, false), (MatchWinner::Engine2, -24));
        assert_eq!(engine1_result(-24, false), (MatchWinner::Engine1, 24));
        assert_eq!(engine1_result(0, true), (MatchWinner::Draw, 0));
        assert_eq!(engine1_result(0, false), (MatchWinner::Draw, 0));
    }

    #[test]
    fn forfeit_counts_as_a_64_disc_loss_for_the_engine_that_forfeits() {
        use PlayerColor::{Black, White};

        for (loser, engine1_is_black, expected) in [
            (Black, true, (MatchWinner::Engine2, -64)),
            (White, true, (MatchWinner::Engine1, 64)),
            (Black, false, (MatchWinner::Engine1, 64)),
            (White, false, (MatchWinner::Engine2, -64)),
        ] {
            assert_eq!(
                engine1_result(forfeit_black_score(loser), engine1_is_black),
                expected,
                "loser {loser:?}, engine1_is_black {engine1_is_black}"
            );
        }
    }

    #[test]
    fn validates_all_openings_before_play() {
        let openings = vec!["f5".to_string(), "z9".to_string()];

        assert!(matches!(
            validate_openings(&openings),
            Err(MatchRunnerError::Config(message)) if message.contains("opening 2")
        ));
    }
}
