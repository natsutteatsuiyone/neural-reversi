//! Match execution and game management.
//!
//! This module contains the core logic for running automated matches between
//! two GTP engines, including game execution, progress tracking, and result
//! aggregation.

use indicatif::ProgressBar;

use crate::config::Config;
use crate::display::DisplayManager;
use crate::engine::GtpEngine;
use crate::error::{MatchRunnerError, Result};
use crate::game::GameState;
use crate::statistics::{MatchStatistics, MatchWinner};
use crate::time_tracker::TimeTracker;
use reversi_core::disc::Disc;
use reversi_core::square::Square;

/// Possible outcomes of a single game.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum GameResult {
    BlackWin,
    WhiteWin,
    Draw,
}

/// Result of a completed game, including outcome and score.
///
/// The score represents the disc difference from the perspective of the black player
/// (positive means black won by that margin, negative means white won).
pub struct MatchResult {
    /// The game outcome (win/loss/draw)
    pub result: GameResult,
    /// Score difference (black perspective)
    pub score: i32,
}

/// Orchestrates and executes automated matches between two engines.
///
/// The MatchRunner handles the complete lifecycle of a match, from engine
/// initialization through game execution to final result reporting. It manages
/// both individual game execution and overall match coordination.
pub struct MatchRunner {
    display: DisplayManager,
}

impl Default for MatchRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchRunner {
    /// Create a new MatchRunner instance.
    ///
    /// # Returns
    ///
    /// A new MatchRunner with an initialized display manager.
    pub fn new() -> Self {
        Self {
            display: DisplayManager::new(),
        }
    }

    /// Execute a complete match using the provided configuration.
    ///
    /// This is the main entry point for running automated matches. It handles:
    /// - Loading opening positions
    /// - Initializing both engines
    /// - Running all games with progress tracking
    /// - Displaying final results
    ///
    /// # Arguments
    ///
    /// * `config` - Match configuration including engine commands and opening file
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful match completion.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The opening file is empty or invalid
    /// - Either engine fails to start
    /// - Any game encounters a fatal error
    pub fn run_match(&mut self, config: &Config) -> Result<()> {
        let openings = config.load_openings()?;

        if openings.is_empty() {
            return Err(MatchRunnerError::Config(
                "The opening file doesn't contain any valid positions.".to_string(),
            ));
        }

        let mut engines = self.initialize_engines(config)?;
        let engine_names = self.get_engine_names(&mut engines)?;

        // Create time tracker (mode is auto-detected from GTP time_settings parameters)
        let mut time_tracker =
            TimeTracker::new(config.main_time, config.byoyomi_time, config.byoyomi_stones);

        let total_games = openings.len() * 2;
        let mut statistics = MatchStatistics::new();

        self.display.show_match_header()?;

        // Show initial empty statistics
        self.display
            .update_live_visualization(&statistics, &engine_names.0, &engine_names.1)?;

        let progress_bar = self.display.create_progress_bar(total_games as u64);

        for (opening_idx, opening_str) in openings.iter().enumerate() {
            if let Err(e) = self.play_opening_pair(
                &mut engines,
                &mut statistics,
                &engine_names,
                opening_str,
                opening_idx,
                &progress_bar,
                &mut time_tracker,
            ) {
                progress_bar.finish_and_clear();
                return Err(e);
            }
        }

        progress_bar.finish_and_clear();
        self.display.clear_screen()?;
        statistics.print_final_results(&engine_names.0, &engine_names.1)?;

        Ok(())
    }

    /// Execute a single game between two engines.
    ///
    /// Runs one complete game from the initial position (with optional opening moves)
    /// to completion, handling all move generation and validation.
    ///
    /// # Arguments
    ///
    /// * `black_engine` - Engine playing as black
    /// * `white_engine` - Engine playing as white
    /// * `opening_moves` - Optional opening sequence in algebraic notation
    /// * `time_tracker` - Time tracker for managing time control
    ///
    /// # Returns
    ///
    /// A `MatchResult` containing the game outcome and final score.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either engine fails to respond to commands
    /// - An invalid move is generated or played
    /// - Engine communication is lost
    pub fn play_game(
        &self,
        black_engine: &mut GtpEngine,
        white_engine: &mut GtpEngine,
        opening_moves: Option<&str>,
        time_tracker: &mut TimeTracker,
    ) -> Result<MatchResult> {
        black_engine.clear_board()?;
        white_engine.clear_board()?;

        // Reset time tracker for new game
        time_tracker.reset();

        // Send time settings to both engines
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

        if let Some(opening) = opening_moves {
            self.apply_opening_moves(&mut game_state, black_engine, white_engine, opening)?;
        }

        while !game_state.is_game_over() {
            let is_black = game_state.side_to_move() == Disc::Black;
            let current_color = if is_black { "black" } else { "white" };

            // Send time_left to both engines before move generation
            if time_tracker.is_enabled() {
                // GTP time_left uses stones=0 for both byoyomi and Fischer modes.
                // The engine uses time_settings to determine the time control mode,
                // and time_left simply updates the remaining time.
                black_engine.time_left("black", time_tracker.black_time_secs(), 0)?;
                white_engine.time_left("black", time_tracker.black_time_secs(), 0)?;
                black_engine.time_left("white", time_tracker.white_time_secs(), 0)?;
                white_engine.time_left("white", time_tracker.white_time_secs(), 0)?;
            }

            // Start timing this move
            time_tracker.start_move();

            let mv = if is_black {
                black_engine.genmove("black")?
            } else {
                white_engine.genmove("white")?
            };

            // End timing and update remaining time
            let has_time = time_tracker.end_move(is_black);
            if !has_time && time_tracker.is_enabled() {
                // Player ran out of time - they lose
                let result = if is_black {
                    GameResult::WhiteWin
                } else {
                    GameResult::BlackWin
                };
                return Ok(MatchResult { result, score: 64 });
            }

            self.execute_move(
                &mut game_state,
                black_engine,
                white_engine,
                &mv,
                current_color,
            )?;
        }

        let (black_count, white_count) = game_state.get_score();
        let result = self.determine_game_result(black_count, white_count);
        let score = self.calculate_score(black_count, white_count);

        Ok(MatchResult { result, score })
    }

    fn apply_opening_moves(
        &self,
        game_state: &mut GameState,
        black_engine: &mut GtpEngine,
        white_engine: &mut GtpEngine,
        opening: &str,
    ) -> Result<()> {
        let mut i = 0;
        while i + 1 < opening.len() {
            let file = opening.chars().nth(i).unwrap();
            let rank = opening.chars().nth(i + 1).unwrap();

            if !('a'..='h').contains(&file) || !('1'..='8').contains(&rank) {
                return Err(MatchRunnerError::Game(format!(
                    "Invalid move in opening sequence: {file}{rank}"
                )));
            }

            let mv = format!("{file}{rank}");
            let square = self.parse_move(&mv)?;

            let color = if game_state.side_to_move() == Disc::Black {
                "black"
            } else {
                "white"
            };

            game_state
                .make_move(Some(square))
                .map_err(MatchRunnerError::Game)?;

            black_engine.play(color, &mv)?;
            white_engine.play(color, &mv)?;

            i += 2;
        }

        Ok(())
    }

    fn execute_move(
        &self,
        game_state: &mut GameState,
        black_engine: &mut GtpEngine,
        white_engine: &mut GtpEngine,
        mv: &str,
        current_color: &str,
    ) -> Result<()> {
        if mv.to_lowercase() == "pass" {
            game_state.make_move(None).map_err(MatchRunnerError::Game)?;

            let opponent_engine = if current_color == "black" {
                white_engine
            } else {
                black_engine
            };
            opponent_engine.play(current_color, "pass")?;
        } else {
            let square = self.parse_move(mv)?;

            game_state
                .make_move(Some(square))
                .map_err(MatchRunnerError::Game)?;

            let opponent_engine = if current_color == "black" {
                white_engine
            } else {
                black_engine
            };
            opponent_engine.play(current_color, mv)?;
        }

        Ok(())
    }

    fn parse_move(&self, move_str: &str) -> Result<Square> {
        move_str
            .parse::<Square>()
            .map_err(|_| MatchRunnerError::Game(format!("Invalid move: {move_str}")))
    }

    fn determine_game_result(&self, black_count: u32, white_count: u32) -> GameResult {
        match black_count.cmp(&white_count) {
            std::cmp::Ordering::Greater => GameResult::BlackWin,
            std::cmp::Ordering::Less => GameResult::WhiteWin,
            std::cmp::Ordering::Equal => GameResult::Draw,
        }
    }

    fn calculate_score(&self, black_count: u32, white_count: u32) -> i32 {
        match black_count.cmp(&white_count) {
            std::cmp::Ordering::Greater => 64 - (white_count as i32) * 2,
            std::cmp::Ordering::Less => (black_count as i32) * 2 - 64,
            std::cmp::Ordering::Equal => 0,
        }
    }

    fn initialize_engines(&self, config: &Config) -> Result<(GtpEngine, GtpEngine)> {
        let (engine1_program, engine1_args) = config.get_engine1_command();
        let (engine2_program, engine2_args) = config.get_engine2_command();

        let engine1 = GtpEngine::new(
            &engine1_program,
            &engine1_args,
            config.engine1_working_dir.clone(),
        )?;
        let engine2 = GtpEngine::new(
            &engine2_program,
            &engine2_args,
            config.engine2_working_dir.clone(),
        )?;

        Ok((engine1, engine2))
    }

    fn get_engine_names(&self, engines: &mut (GtpEngine, GtpEngine)) -> Result<(String, String)> {
        let engine1_name = engines.0.name();
        let engine2_name = engines.1.name();
        Ok((engine1_name, engine2_name))
    }

    #[allow(clippy::too_many_arguments)]
    fn play_opening_pair(
        &mut self,
        engines: &mut (GtpEngine, GtpEngine),
        statistics: &mut MatchStatistics,
        engine_names: &(String, String),
        opening_str: &str,
        opening_idx: usize,
        progress_bar: &ProgressBar,
        time_tracker: &mut TimeTracker,
    ) -> Result<()> {
        let mut paired_results = Vec::new();

        for game_round in 0..2 {
            let is_swapped = game_round == 1;
            let game_number = opening_idx * 2 + game_round + 1;

            let (black_engine, white_engine) = if is_swapped {
                (&mut engines.1, &mut engines.0)
            } else {
                (&mut engines.0, &mut engines.1)
            };

            match self.play_game(black_engine, white_engine, Some(opening_str), time_tracker) {
                Ok(match_result) => {
                    let winner = self.determine_match_winner(match_result.result, is_swapped);
                    let score = if is_swapped {
                        -match_result.score
                    } else {
                        match_result.score
                    };

                    statistics.add_result(winner, score, opening_str.to_string(), !is_swapped);
                    paired_results.push((winner, score));

                    self.display.update_live_visualization(
                        statistics,
                        &engine_names.0,
                        &engine_names.1,
                    )?;
                    progress_bar.inc(1);
                }
                Err(e) => {
                    return Err(MatchRunnerError::Game(format!(
                        "Fatal error in game {game_number}: {e}"
                    )));
                }
            }
        }

        // Add paired result after both games are complete
        if paired_results.len() == 2 {
            statistics.add_paired_result(paired_results[0], paired_results[1]);
        }

        Ok(())
    }

    fn determine_match_winner(&self, result: GameResult, is_swapped: bool) -> MatchWinner {
        match result {
            GameResult::BlackWin => {
                if is_swapped {
                    MatchWinner::Engine2
                } else {
                    MatchWinner::Engine1
                }
            }
            GameResult::WhiteWin => {
                if is_swapped {
                    MatchWinner::Engine1
                } else {
                    MatchWinner::Engine2
                }
            }
            GameResult::Draw => MatchWinner::Draw,
        }
    }
}
