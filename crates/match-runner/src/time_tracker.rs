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
    /// Black's remaining stones in current byoyomi period
    black_byo_stones_left: u32,
    /// White's remaining stones in current byoyomi period
    white_byo_stones_left: u32,
    /// Black's elapsed time in current byoyomi period (ms)
    black_byo_time_used_ms: u64,
    /// White's elapsed time in current byoyomi period (ms)
    white_byo_time_used_ms: u64,
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

        let (initial_time_ms, initial_in_byoyomi) = Self::initial_player_state(mode, main_time_ms);

        Self {
            mode,
            main_time_ms,
            byoyomi_time_ms,
            byoyomi_stones,
            black_time_ms: initial_time_ms,
            white_time_ms: initial_time_ms,
            black_in_byoyomi: initial_in_byoyomi,
            white_in_byoyomi: initial_in_byoyomi,
            black_byo_stones_left: byoyomi_stones,
            white_byo_stones_left: byoyomi_stones,
            black_byo_time_used_ms: 0,
            white_byo_time_used_ms: 0,
            move_start: None,
        }
    }

    /// Returns the initial (time_ms, in_byoyomi) state for a player.
    fn initial_player_state(mode: TimeControlMode, main_time_ms: u64) -> (u64, bool) {
        match mode {
            TimeControlMode::None => (u64::MAX, false),
            TimeControlMode::Byoyomi => (0, true),
            TimeControlMode::Fischer => (main_time_ms, false),
            TimeControlMode::JapaneseByo => (main_time_ms, main_time_ms == 0),
        }
    }

    /// Determines the time control mode from GTP time_settings parameters.
    fn determine_mode(main_time: u64, byoyomi_time: u64, byoyomi_stones: u32) -> TimeControlMode {
        if main_time == 0 && byoyomi_time == 0 {
            // No time control
            TimeControlMode::None
        } else if main_time == 0 && byoyomi_time > 0 && byoyomi_stones == 0 {
            // Pure byoyomi: fixed time per move from the start
            TimeControlMode::Byoyomi
        } else if main_time == 0 && byoyomi_time > 0 && byoyomi_stones > 0 {
            // Japanese byo-yomi with no main time: start directly in byoyomi phase
            TimeControlMode::JapaneseByo
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

    /// Returns `(time_secs, stones)` for the GTP `time_left` command for black.
    ///
    /// During main time: returns (main_time_remaining, 0).
    /// During byoyomi: returns (period_time_remaining, stones_left_in_period).
    pub fn black_time_left(&self) -> (u64, u32) {
        self.player_time_left(true)
    }

    /// Returns `(time_secs, stones)` for the GTP `time_left` command for white.
    pub fn white_time_left(&self) -> (u64, u32) {
        self.player_time_left(false)
    }

    fn player_time_left(&self, is_black: bool) -> (u64, u32) {
        let (time_ms, in_byoyomi, byo_stones_left, byo_time_used_ms) = if is_black {
            (
                self.black_time_ms,
                self.black_in_byoyomi,
                self.black_byo_stones_left,
                self.black_byo_time_used_ms,
            )
        } else {
            (
                self.white_time_ms,
                self.white_in_byoyomi,
                self.white_byo_stones_left,
                self.white_byo_time_used_ms,
            )
        };

        match self.mode {
            TimeControlMode::Byoyomi => {
                // Pure byoyomi: each move gets the full byoyomi time budget
                (self.byoyomi_time_ms / 1000, 0)
            }
            TimeControlMode::JapaneseByo if in_byoyomi => {
                // In byoyomi phase: report remaining period time and stones
                let period_remaining_ms = self.byoyomi_time_ms.saturating_sub(byo_time_used_ms);
                (period_remaining_ms / 1000, byo_stones_left)
            }
            _ => {
                // Main time phase (Fischer, JapaneseByo pre-byoyomi, None)
                (time_ms / 1000, 0)
            }
        }
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
        self.apply_elapsed(is_black, elapsed_ms)
    }

    /// Apply elapsed time and update the player's remaining time.
    ///
    /// This is the core time control logic, separated from clock measurement
    /// for testability.
    fn apply_elapsed(&mut self, is_black: bool, elapsed_ms: u64) -> bool {
        match self.mode {
            TimeControlMode::None => true,
            TimeControlMode::Byoyomi => {
                // Pure byoyomi: each move must be within the fixed time
                elapsed_ms <= self.byoyomi_time_ms
            }
            TimeControlMode::Fischer => {
                // Deduct time and add increment
                let increment = self.byoyomi_time_ms;
                let time_ms = self.player_time_mut(is_black);

                if elapsed_ms >= *time_ms {
                    *time_ms = 0;
                    false
                } else {
                    *time_ms = (*time_ms - elapsed_ms).saturating_add(increment);
                    true
                }
            }
            TimeControlMode::JapaneseByo => self.apply_japanese_byo(is_black, elapsed_ms),
        }
    }

    /// Apply Japanese byo-yomi time control for a move.
    ///
    /// In Japanese byo-yomi, after main time expires the player enters byoyomi:
    /// they get `byoyomi_stones` moves within `byoyomi_time` seconds.
    /// Completing all stones resets the period.
    fn apply_japanese_byo(&mut self, is_black: bool, elapsed_ms: u64) -> bool {
        let byoyomi_time_ms = self.byoyomi_time_ms;
        let byoyomi_stones = self.byoyomi_stones;
        let (time_ms, in_byoyomi) = self.player_state_mut(is_black);

        if !*in_byoyomi {
            // Still in main time
            if elapsed_ms < *time_ms {
                *time_ms -= elapsed_ms;
                return true;
            }
            // Main time exhausted, transition to byoyomi
            let overtime = elapsed_ms - *time_ms;
            *time_ms = 0;
            *in_byoyomi = true;

            // Start first byoyomi period with the overtime already consumed
            let (stones_left, time_used) = self.player_byo_period_mut(is_black);
            *stones_left = byoyomi_stones;
            *time_used = overtime;

            if overtime > byoyomi_time_ms {
                return false; // Exceeded first period immediately
            }

            if overtime == 0 {
                // Exhausting main time exactly starts the first byoyomi period,
                // but does not consume a stone until the next move actually uses it.
                return true;
            }

            *stones_left -= 1;
            if *stones_left == 0 {
                // Completed the period, reset for next
                *stones_left = byoyomi_stones;
                *time_used = 0;
            }
            return true;
        }

        // Already in byoyomi phase
        let (stones_left, time_used) = self.player_byo_period_mut(is_black);
        *time_used += elapsed_ms;

        if *time_used > byoyomi_time_ms {
            return false; // Period time exceeded
        }

        *stones_left -= 1;
        if *stones_left == 0 {
            // Completed all stones in this period, reset for next
            *stones_left = byoyomi_stones;
            *time_used = 0;
        }
        true
    }

    /// Returns a mutable reference to the specified player's remaining time.
    fn player_time_mut(&mut self, is_black: bool) -> &mut u64 {
        if is_black {
            &mut self.black_time_ms
        } else {
            &mut self.white_time_ms
        }
    }

    /// Returns mutable references to the specified player's time and byoyomi state.
    fn player_state_mut(&mut self, is_black: bool) -> (&mut u64, &mut bool) {
        if is_black {
            (&mut self.black_time_ms, &mut self.black_in_byoyomi)
        } else {
            (&mut self.white_time_ms, &mut self.white_in_byoyomi)
        }
    }

    /// Returns mutable references to the specified player's byoyomi period state.
    fn player_byo_period_mut(&mut self, is_black: bool) -> (&mut u32, &mut u64) {
        if is_black {
            (
                &mut self.black_byo_stones_left,
                &mut self.black_byo_time_used_ms,
            )
        } else {
            (
                &mut self.white_byo_stones_left,
                &mut self.white_byo_time_used_ms,
            )
        }
    }

    /// Resets the time tracker for a new game.
    pub fn reset(&mut self) {
        let (initial_time_ms, initial_in_byoyomi) =
            Self::initial_player_state(self.mode, self.main_time_ms);

        self.black_time_ms = initial_time_ms;
        self.white_time_ms = initial_time_ms;
        self.black_in_byoyomi = initial_in_byoyomi;
        self.white_in_byoyomi = initial_in_byoyomi;
        self.black_byo_stones_left = self.byoyomi_stones;
        self.white_byo_stones_left = self.byoyomi_stones;
        self.black_byo_time_used_ms = 0;
        self.white_byo_time_used_ms = 0;
        self.move_start = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_time_control() {
        let tracker = TimeTracker::new(0, 0, 0);
        assert!(!tracker.is_enabled());
    }

    #[test]
    fn test_no_time_control_always_has_time() {
        let mut tracker = TimeTracker::new(0, 0, 0);
        assert!(tracker.apply_elapsed(true, 999_999));
        assert!(tracker.apply_elapsed(false, 999_999));
    }

    #[test]
    fn test_mode_detection_byoyomi() {
        let tracker = TimeTracker::new(0, 10, 0);
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_mode_detection_fischer() {
        let tracker = TimeTracker::new(300, 5, 0);
        assert!(tracker.is_enabled());
        assert_eq!(tracker.main_time_secs(), 300);
        assert_eq!(tracker.byoyomi_time_secs(), 5);
    }

    #[test]
    fn test_mode_detection_japanese_byo() {
        let tracker = TimeTracker::new(300, 30, 1);
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_mode_detection_sudden_death() {
        let tracker = TimeTracker::new(60, 0, 0);
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_byoyomi_within_time() {
        let mut tracker = TimeTracker::new(0, 10, 0);
        assert!(tracker.apply_elapsed(true, 5_000)); // 5s < 10s
    }

    #[test]
    fn test_byoyomi_exact_time() {
        let mut tracker = TimeTracker::new(0, 10, 0);
        assert!(tracker.apply_elapsed(true, 10_000)); // 10s == 10s
    }

    #[test]
    fn test_byoyomi_over_time() {
        let mut tracker = TimeTracker::new(0, 10, 0);
        assert!(!tracker.apply_elapsed(true, 15_000)); // 15s > 10s
    }

    #[test]
    fn test_fischer_normal_move() {
        let mut tracker = TimeTracker::new(300, 5, 0); // 300s + 5s increment
        assert!(tracker.apply_elapsed(true, 10_000)); // use 10s
        // Remaining: 300000 - 10000 + 5000 = 295000
        assert_eq!(tracker.black_time_left().0, 295);
    }

    #[test]
    fn test_fischer_increment_accumulates() {
        let mut tracker = TimeTracker::new(10, 5, 0); // 10s + 5s increment
        assert!(tracker.apply_elapsed(true, 3_000)); // use 3s
        // Remaining: 10000 - 3000 + 5000 = 12000
        assert_eq!(tracker.black_time_left().0, 12);
    }

    #[test]
    fn test_fischer_flag_exact_time() {
        let mut tracker = TimeTracker::new(10, 5, 0);
        assert!(!tracker.apply_elapsed(true, 10_000)); // use exactly 10s -> flag
        assert_eq!(tracker.black_time_left().0, 0);
    }

    #[test]
    fn test_fischer_flag_over_time() {
        let mut tracker = TimeTracker::new(10, 5, 0);
        assert!(!tracker.apply_elapsed(true, 15_000)); // use 15s > 10s
        assert_eq!(tracker.black_time_left().0, 0);
    }

    #[test]
    fn test_sudden_death() {
        let mut tracker = TimeTracker::new(60, 0, 0); // 60s, no increment
        assert!(tracker.apply_elapsed(true, 30_000)); // use 30s
        // Remaining: 60000 - 30000 + 0 = 30000
        assert_eq!(tracker.black_time_left().0, 30);
        assert!(!tracker.apply_elapsed(true, 30_000)); // exactly 30s -> flag
    }

    #[test]
    fn test_japanese_byo_main_time() {
        let mut tracker = TimeTracker::new(300, 30, 1);
        assert!(tracker.apply_elapsed(true, 10_000)); // use 10s of main time
        assert_eq!(tracker.black_time_left().0, 290);
    }

    #[test]
    fn test_japanese_byo_transition_success() {
        let mut tracker = TimeTracker::new(10, 30, 1); // 10s main, 30s byo
        // Use 15s -> exhausts 10s main time, 5s overtime (5s <= 30s -> OK)
        // With stones=1, the stone is consumed and period resets
        assert!(tracker.apply_elapsed(true, 15_000));
        // Now in byoyomi with a fresh period (stone consumed, period reset)
        assert_eq!(tracker.black_time_left(), (30, 1));
    }

    #[test]
    fn test_japanese_byo_transition_fail() {
        let mut tracker = TimeTracker::new(10, 5, 1); // 10s main, 5s byo
        // Use 20s -> exhausts 10s main, 10s overtime (10s > 5s -> flag)
        assert!(!tracker.apply_elapsed(true, 20_000));
    }

    #[test]
    fn test_japanese_byo_in_byoyomi_phase() {
        let mut tracker = TimeTracker::new(10, 30, 1);
        // Use exactly 10s → transitions to byoyomi immediately (overtime=0)
        assert!(tracker.apply_elapsed(true, 10_000));
        // Now in byoyomi phase: fresh period
        assert_eq!(tracker.black_time_left(), (30, 1));
        // Move in byoyomi: 25s <= 30s -> OK, period resets (stones=1)
        assert!(tracker.apply_elapsed(true, 25_000));
        // Exceed byoyomi: 35s > 30s -> flag
        assert!(!tracker.apply_elapsed(true, 35_000));
    }

    #[test]
    fn test_white_time_independent() {
        let mut tracker = TimeTracker::new(300, 5, 0);
        tracker.apply_elapsed(true, 100_000); // black uses 100s
        assert_eq!(tracker.white_time_left().0, 300); // white unchanged
    }

    #[test]
    fn test_reset() {
        let mut tracker = TimeTracker::new(300, 5, 0);
        tracker.apply_elapsed(true, 100_000); // black uses 100s
        tracker.reset();
        assert_eq!(tracker.black_time_left().0, 300);
        assert_eq!(tracker.white_time_left().0, 300);
    }

    #[test]
    fn test_reset_byoyomi() {
        let mut tracker = TimeTracker::new(0, 10, 0);
        tracker.apply_elapsed(true, 5_000);
        tracker.reset();
        // Pure byoyomi: time_left reports per-move budget
        assert_eq!(tracker.black_time_left(), (10, 0));
    }

    // =========================================================================
    // time_left GTP reporting
    // =========================================================================

    #[test]
    fn test_time_left_pure_byoyomi_reports_budget() {
        let tracker = TimeTracker::new(0, 10, 0);
        // Pure byoyomi: each move gets 10s, reported every time
        assert_eq!(tracker.black_time_left(), (10, 0));
        assert_eq!(tracker.white_time_left(), (10, 0));
    }

    #[test]
    fn test_time_left_after_exact_main_time_exhaustion() {
        let mut tracker = TimeTracker::new(10, 30, 1);
        // Use exactly all main time → transitions to byoyomi
        assert!(tracker.apply_elapsed(true, 10_000));
        // Should report byoyomi period, not (0, 0)
        assert_eq!(tracker.black_time_left(), (30, 1));
    }

    #[test]
    fn test_time_left_fischer_reports_main_time() {
        let mut tracker = TimeTracker::new(300, 5, 0);
        assert_eq!(tracker.black_time_left(), (300, 0));
        tracker.apply_elapsed(true, 10_000);
        assert_eq!(tracker.black_time_left(), (295, 0));
        assert_eq!(tracker.white_time_left(), (300, 0));
    }

    #[test]
    fn test_time_left_japanese_byo_main_time_phase() {
        let tracker = TimeTracker::new(300, 30, 3);
        // In main time: reports main time, stones=0
        assert_eq!(tracker.black_time_left(), (300, 0));
    }

    #[test]
    fn test_time_left_japanese_byo_byoyomi_phase() {
        let mut tracker = TimeTracker::new(10, 30, 3);
        // Use exactly 10s → transitions to byoyomi (overtime=0, stone 1 consumed)
        assert!(tracker.apply_elapsed(true, 10_000));
        assert_eq!(tracker.black_time_left(), (30, 3)); // fresh period, no stone consumed yet
        // Move 1 in byoyomi: 8s used, 2 stones left
        assert!(tracker.apply_elapsed(true, 8_000));
        assert_eq!(tracker.black_time_left(), (22, 2));
    }

    #[test]
    fn test_time_left_japanese_byo_period_reset() {
        let mut tracker = TimeTracker::new(0, 10, 2);
        // Start directly in byoyomi
        // Stone 1: 4s
        assert!(tracker.apply_elapsed(true, 4_000));
        // time_used=4s, 1 stone left, remaining=6s
        assert_eq!(tracker.black_time_left(), (6, 1));
        // Stone 2: 3s → period resets
        assert!(tracker.apply_elapsed(true, 3_000));
        // After reset: time_used=0, stones=2, remaining=10s
        assert_eq!(tracker.black_time_left(), (10, 2));
    }

    // =========================================================================
    // Japanese byo-yomi with multiple stones per period
    // =========================================================================

    #[test]
    fn test_japanese_byo_multi_stones_within_period() {
        // 10s main, 30s byoyomi, 3 stones per period
        let mut tracker = TimeTracker::new(10, 30, 3);
        // Exhaust main time
        assert!(tracker.apply_elapsed(true, 10_000));
        // Enter byoyomi: 3 stones in 30s
        // Stone 1: 8s → cumulative 8s ≤ 30s, 2 stones left
        assert!(tracker.apply_elapsed(true, 8_000));
        // Stone 2: 10s → cumulative 18s ≤ 30s, 1 stone left
        assert!(tracker.apply_elapsed(true, 10_000));
        // Stone 3: 10s → cumulative 28s ≤ 30s, 0 stones left → period resets
        assert!(tracker.apply_elapsed(true, 10_000));
        // New period starts: Stone 1: 5s → cumulative 5s ≤ 30s
        assert!(tracker.apply_elapsed(true, 5_000));
    }

    #[test]
    fn test_japanese_byo_exact_main_time_does_not_consume_stone() {
        let mut tracker = TimeTracker::new(10, 30, 3);

        assert!(tracker.apply_elapsed(true, 10_000));
        assert_eq!(tracker.black_time_left(), (30, 3));

        assert!(tracker.apply_elapsed(true, 7_000));
        assert_eq!(tracker.black_time_left(), (23, 2));
    }

    #[test]
    fn test_japanese_byo_multi_stones_exceed_period() {
        // 10s main, 30s byoyomi, 3 stones per period
        let mut tracker = TimeTracker::new(10, 30, 3);
        // Exhaust main time
        assert!(tracker.apply_elapsed(true, 10_000));
        // Enter byoyomi: 3 stones in 30s
        // Stone 1: 20s → cumulative 20s ≤ 30s
        assert!(tracker.apply_elapsed(true, 20_000));
        // Stone 2: 15s → cumulative 35s > 30s → flag
        assert!(!tracker.apply_elapsed(true, 15_000));
    }

    #[test]
    fn test_japanese_byo_multi_stones_transition_with_overtime() {
        // 5s main, 20s byoyomi, 2 stones per period
        let mut tracker = TimeTracker::new(5, 20, 2);
        // Use 12s → exhausts 5s main, 7s overtime into byoyomi
        // 7s ≤ 20s → OK, stone 1 consumed, 1 stone left
        assert!(tracker.apply_elapsed(true, 12_000));
        // Stone 2: 10s → cumulative 7+10=17s ≤ 20s → OK, period resets
        assert!(tracker.apply_elapsed(true, 10_000));
        // New period, stone 1: 15s → cumulative 15s ≤ 20s
        assert!(tracker.apply_elapsed(true, 15_000));
    }

    #[test]
    fn test_japanese_byo_multi_stones_period_resets_correctly() {
        // 0s main (start in byoyomi immediately), 10s byoyomi, 2 stones per period
        let mut tracker = TimeTracker::new(0, 10, 2);
        // Immediately in byoyomi (main_time=0, byoyomi_stones>0 → JapaneseByo mode
        // with initial_in_byoyomi=true)
        // Stone 1: 4s ≤ 10s
        assert!(tracker.apply_elapsed(true, 4_000));
        // Stone 2: 5s → cumulative 9s ≤ 10s → period resets
        assert!(tracker.apply_elapsed(true, 5_000));
        // New period: Stone 1: 6s → 6s ≤ 10s
        assert!(tracker.apply_elapsed(true, 6_000));
        // Stone 2: 5s → cumulative 11s > 10s → flag
        assert!(!tracker.apply_elapsed(true, 5_000));
    }
}
