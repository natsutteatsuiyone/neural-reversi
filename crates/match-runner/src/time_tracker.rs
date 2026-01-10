//! Time tracking for timed matches.
//!
//! This module provides functionality for tracking time usage during matches,
//! supporting various time control modes following the GTP `time_settings` format.

use std::time::Instant;

/// Time control mode, automatically determined from GTP time_settings parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimeControlMode {
    /// No time control
    None,
    /// Fixed time per move (pure byoyomi)
    /// GTP: `time_settings 0 N 0` means N seconds per move
    Byoyomi,
    /// Main time + increment per move (Fischer)
    /// GTP: `time_settings M N 0` means M seconds main time + N seconds increment
    Fischer,
    /// Main time + fixed time per move after main time expires (Japanese byo-yomi)
    /// GTP: `time_settings M N 1` means M seconds main time, then N seconds per move
    JapaneseByo,
}

/// Tracks time for both players during a game.
#[derive(Debug)]
pub struct TimeTracker {
    /// Time control mode (auto-detected from parameters)
    mode: TimeControlMode,
    /// Main time in milliseconds
    main_time_ms: u64,
    /// Byoyomi/increment time in milliseconds
    byoyomi_time_ms: u64,
    /// Byoyomi stones setting (from GTP time_settings)
    byoyomi_stones: u32,
    /// Black's remaining main time in milliseconds
    black_time_ms: u64,
    /// White's remaining main time in milliseconds
    white_time_ms: u64,
    /// Whether black is in byoyomi phase (only for JapaneseByo mode)
    black_in_byoyomi: bool,
    /// Whether white is in byoyomi phase (only for JapaneseByo mode)
    white_in_byoyomi: bool,
    /// Start time of current move
    move_start: Option<Instant>,
}

impl TimeTracker {
    /// Creates a new TimeTracker with the specified time control settings.
    ///
    /// Time control mode is automatically determined from the parameters,
    /// following GTP `time_settings` semantics:
    ///
    /// - `main_time=0, byoyomi_time=0`: No time control
    /// - `main_time=0, byoyomi_time>0, stones=0`: Pure byoyomi (N seconds per move)
    /// - `main_time>0, byoyomi_time>0, stones=0`: Fischer (main + increment)
    /// - `main_time>0, byoyomi_time>0, stones>0`: Japanese byo-yomi (main + fixed per move)
    /// - `main_time>0, byoyomi_time=0`: Sudden death
    ///
    /// # Arguments
    ///
    /// * `main_time_secs` - Main time in seconds (0 for no main time)
    /// * `byoyomi_time_secs` - Byoyomi/increment time in seconds
    /// * `byoyomi_stones` - Byoyomi stones (0 for increment/per-move, >0 for period)
    pub fn new(main_time_secs: u64, byoyomi_time_secs: u64, byoyomi_stones: u32) -> Self {
        let main_time_ms = main_time_secs * 1000;
        let byoyomi_time_ms = byoyomi_time_secs * 1000;

        // Determine time control mode from parameters
        let mode = Self::determine_mode(main_time_secs, byoyomi_time_secs, byoyomi_stones);

        // Initialize remaining time based on mode
        let (black_time_ms, white_time_ms, black_in_byoyomi, white_in_byoyomi) = match mode {
            TimeControlMode::None => (u64::MAX, u64::MAX, false, false),
            TimeControlMode::Byoyomi => {
                // Pure byoyomi: no main time, fixed time per move
                (0, 0, true, true)
            }
            TimeControlMode::Fischer => (main_time_ms, main_time_ms, false, false),
            TimeControlMode::JapaneseByo => {
                // Start with main time, enter byoyomi when exhausted
                let in_byoyomi = main_time_ms == 0;
                (main_time_ms, main_time_ms, in_byoyomi, in_byoyomi)
            }
        };

        Self {
            mode,
            main_time_ms,
            byoyomi_time_ms,
            byoyomi_stones,
            black_time_ms,
            white_time_ms,
            black_in_byoyomi,
            white_in_byoyomi,
            move_start: None,
        }
    }

    /// Determines the time control mode from GTP time_settings parameters.
    fn determine_mode(main_time: u64, byoyomi_time: u64, byoyomi_stones: u32) -> TimeControlMode {
        if main_time == 0 && byoyomi_time == 0 {
            // No time control
            TimeControlMode::None
        } else if main_time == 0 && byoyomi_time > 0 {
            // Pure byoyomi: fixed time per move from the start
            TimeControlMode::Byoyomi
        } else if main_time > 0 && byoyomi_time > 0 && byoyomi_stones == 0 {
            // Fischer: main time + increment per move
            TimeControlMode::Fischer
        } else if main_time > 0 && byoyomi_time > 0 && byoyomi_stones > 0 {
            // Japanese byo-yomi: main time + fixed time per move after main time expires
            TimeControlMode::JapaneseByo
        } else if main_time > 0 && byoyomi_time == 0 {
            // Sudden death: only main time, no increment
            TimeControlMode::Fischer
        } else {
            TimeControlMode::None
        }
    }

    /// Returns true if time control is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self.mode, TimeControlMode::None)
    }

    /// Returns the main time setting in seconds (for time_settings command).
    pub fn main_time_secs(&self) -> u64 {
        self.main_time_ms / 1000
    }

    /// Returns the byoyomi/increment time in seconds (for time_settings command).
    pub fn byoyomi_time_secs(&self) -> u64 {
        self.byoyomi_time_ms / 1000
    }

    /// Returns the byoyomi stones setting (for time_settings command).
    pub fn byoyomi_stones(&self) -> u32 {
        self.byoyomi_stones
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
                // Pure byoyomi: each move must be within the fixed time
                elapsed_ms <= self.byoyomi_time_ms
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
            TimeControlMode::JapaneseByo => {
                // Japanese byo-yomi: use main time first, then enter byoyomi
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
        }
    }

    /// Resets the time tracker for a new game.
    pub fn reset(&mut self) {
        let (black_time_ms, white_time_ms, black_in_byoyomi, white_in_byoyomi) = match self.mode {
            TimeControlMode::None => (u64::MAX, u64::MAX, false, false),
            TimeControlMode::Byoyomi => (0, 0, true, true),
            TimeControlMode::Fischer => (self.main_time_ms, self.main_time_ms, false, false),
            TimeControlMode::JapaneseByo => {
                let in_byoyomi = self.main_time_ms == 0;
                (self.main_time_ms, self.main_time_ms, in_byoyomi, in_byoyomi)
            }
        };

        self.black_time_ms = black_time_ms;
        self.white_time_ms = white_time_ms;
        self.black_in_byoyomi = black_in_byoyomi;
        self.white_in_byoyomi = white_in_byoyomi;
        self.move_start = None;
    }
}
