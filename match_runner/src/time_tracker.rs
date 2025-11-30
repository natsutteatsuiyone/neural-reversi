//! Time tracking for timed matches.
//!
//! This module provides functionality for tracking time usage during matches,
//! supporting various time control modes.

use std::time::Instant;

use crate::config::TimeControlMode;

/// Tracks time for both players during a game.
#[derive(Debug)]
pub struct TimeTracker {
    /// Time control mode
    mode: TimeControlMode,
    /// Main time in milliseconds
    main_time_ms: u64,
    /// Byoyomi/increment time in milliseconds
    byoyomi_time_ms: u64,
    /// Black's remaining main time in milliseconds
    black_time_ms: u64,
    /// White's remaining main time in milliseconds
    white_time_ms: u64,
    /// Whether black is in byoyomi phase (only for Byoyomi mode)
    black_in_byoyomi: bool,
    /// Whether white is in byoyomi phase (only for Byoyomi mode)
    white_in_byoyomi: bool,
    /// Start time of current move
    move_start: Option<Instant>,
}

impl TimeTracker {
    /// Creates a new TimeTracker with the specified time control settings.
    ///
    /// # Arguments
    ///
    /// * `mode` - Time control mode
    /// * `main_time_secs` - Main time in seconds
    /// * `byoyomi_time_secs` - Byoyomi/increment time in seconds
    pub fn new(mode: TimeControlMode, main_time_secs: u64, byoyomi_time_secs: u64) -> Self {
        let main_time_ms = main_time_secs * 1000;
        let byoyomi_time_ms = byoyomi_time_secs * 1000;

        // Initialize remaining time based on mode
        let (black_time_ms, white_time_ms, black_in_byoyomi, white_in_byoyomi) = match mode {
            TimeControlMode::None => (u64::MAX, u64::MAX, false, false),
            TimeControlMode::Byoyomi => {
                // For Canadian byo yomi: start with main time, then enter byoyomi when exhausted
                let in_byoyomi = main_time_ms == 0;
                (main_time_ms, main_time_ms, in_byoyomi, in_byoyomi)
            }
            TimeControlMode::Fischer => (main_time_ms, main_time_ms, false, false),
        };

        Self {
            mode,
            main_time_ms,
            byoyomi_time_ms,
            black_time_ms,
            white_time_ms,
            black_in_byoyomi,
            white_in_byoyomi,
            move_start: None,
        }
    }

    /// Returns true if time control is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self.mode, TimeControlMode::None)
    }

    /// Returns the time control mode.
    pub fn mode(&self) -> TimeControlMode {
        self.mode
    }

    /// Returns the main time setting in seconds (for time_settings command).
    pub fn main_time_secs(&self) -> u64 {
        match self.mode {
            TimeControlMode::None => 0,
            TimeControlMode::Byoyomi | TimeControlMode::Fischer => self.main_time_ms / 1000,
        }
    }

    /// Returns the byoyomi/increment time in seconds (for time_settings command).
    pub fn byoyomi_time_secs(&self) -> u64 {
        self.byoyomi_time_ms / 1000
    }

    /// Returns the byoyomi stones setting (for time_settings command).
    pub fn byoyomi_stones(&self) -> u32 {
        match self.mode {
            TimeControlMode::None => 0,
            TimeControlMode::Byoyomi => 1,
            // Fischer uses increment per move; GTP `stones` should be 0 so the
            // engine interprets the second value as increment rather than a
            // stones-per-period budget.
            TimeControlMode::Fischer => 0,
        }
    }

    /// Returns black's remaining time in seconds.
    pub fn black_time_secs(&self) -> u64 {
        self.black_time_ms / 1000
    }

    /// Returns white's remaining time in seconds.
    pub fn white_time_secs(&self) -> u64 {
        self.white_time_ms / 1000
    }

    /// Starts the clock for a move.
    pub fn start_move(&mut self) {
        self.move_start = Some(Instant::now());
    }

    /// Ends the clock for a move and updates the player's remaining time.
    ///
    /// # Arguments
    ///
    /// * `is_black` - True if black made the move
    ///
    /// # Returns
    ///
    /// True if the player has time remaining, false if they flagged.
    pub fn end_move(&mut self, is_black: bool) -> bool {
        let elapsed_ms = self
            .move_start
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);
        self.move_start = None;

        match self.mode {
            TimeControlMode::None => true,
            TimeControlMode::Byoyomi => {
                // Canadian byo yomi: use main time first, then enter byoyomi
                let (time_ms, in_byoyomi) = if is_black {
                    (&mut self.black_time_ms, &mut self.black_in_byoyomi)
                } else {
                    (&mut self.white_time_ms, &mut self.white_in_byoyomi)
                };

                if *in_byoyomi {
                    // In byoyomi phase: succeed if within byoyomi time
                    elapsed_ms <= self.byoyomi_time_ms
                } else if elapsed_ms <= *time_ms {
                    // In main time phase and within budget
                    *time_ms -= elapsed_ms;
                    true
                } else {
                    // Main time exhausted, transition to byoyomi
                    let overtime = elapsed_ms - *time_ms;
                    *time_ms = 0;
                    *in_byoyomi = true;
                    // Check if this move exceeded byoyomi
                    overtime <= self.byoyomi_time_ms
                }
            }
            TimeControlMode::Fischer => {
                // Deduct time and add increment
                let time_ms = if is_black {
                    &mut self.black_time_ms
                } else {
                    &mut self.white_time_ms
                };

                if elapsed_ms >= *time_ms {
                    *time_ms = 0;
                    false
                } else {
                    *time_ms = time_ms.saturating_sub(elapsed_ms);
                    *time_ms = time_ms.saturating_add(self.byoyomi_time_ms);
                    true
                }
            }
        }
    }

    /// Resets the time tracker for a new game.
    pub fn reset(&mut self) {
        let (black_time_ms, white_time_ms, black_in_byoyomi, white_in_byoyomi) = match self.mode {
            TimeControlMode::None => (u64::MAX, u64::MAX, false, false),
            TimeControlMode::Byoyomi => {
                let in_byoyomi = self.main_time_ms == 0;
                (self.main_time_ms, self.main_time_ms, in_byoyomi, in_byoyomi)
            }
            TimeControlMode::Fischer => (self.main_time_ms, self.main_time_ms, false, false),
        };

        self.black_time_ms = black_time_ms;
        self.white_time_ms = white_time_ms;
        self.black_in_byoyomi = black_in_byoyomi;
        self.white_in_byoyomi = white_in_byoyomi;
        self.move_start = None;
    }
}
