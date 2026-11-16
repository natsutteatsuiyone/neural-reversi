//! Time tracking for timed matches.
//!
//! This module provides functionality for tracking time usage during matches,
//! supporting various time control modes following the GTP `time_settings` format.

use std::time::Instant;

use reversi_core::player_color::PlayerColor;
use reversi_core::search::time_control::TimeSettings;

#[derive(Debug, Clone, Copy)]
struct PlayerClock {
    time_ms: u64,
    byo_time_used_ms: u64,
    byo_stones_left: u32,
    in_byoyomi: bool,
}

impl PlayerClock {
    fn new(settings: TimeSettings) -> Self {
        match settings {
            TimeSettings::Infinite => Self {
                time_ms: u64::MAX,
                byo_time_used_ms: 0,
                byo_stones_left: 0,
                in_byoyomi: false,
            },
            TimeSettings::Byoyomi { .. } => Self {
                time_ms: 0,
                byo_time_used_ms: 0,
                byo_stones_left: 0,
                in_byoyomi: true,
            },
            TimeSettings::Fischer { main_time_ms, .. } => Self {
                time_ms: main_time_ms,
                byo_time_used_ms: 0,
                byo_stones_left: 0,
                in_byoyomi: false,
            },
            TimeSettings::JapaneseByo {
                main_time_ms,
                byoyomi_stones,
                ..
            } => Self {
                time_ms: main_time_ms,
                byo_time_used_ms: 0,
                byo_stones_left: byoyomi_stones.get(),
                in_byoyomi: main_time_ms == 0,
            },
        }
    }
}

/// Tracks time for both players during a game.
#[derive(Debug)]
pub struct TimeTracker {
    settings: TimeSettings,
    players: [PlayerClock; 2],
    move_start: Option<Instant>,
}

impl TimeTracker {
    /// Creates a new [`TimeTracker`] from GTP `time_settings` parameters (in seconds and stones).
    pub fn new(main_time_secs: u64, byoyomi_time_secs: u64, byoyomi_stones: u32) -> Self {
        let settings =
            TimeSettings::from_gtp_secs(main_time_secs, byoyomi_time_secs, byoyomi_stones);
        let player = PlayerClock::new(settings);
        Self {
            settings,
            players: [player; 2],
            move_start: None,
        }
    }

    /// Returns the match [`TimeSettings`].
    #[must_use]
    pub fn settings(&self) -> TimeSettings {
        self.settings
    }

    /// Returns `true` if time control is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.settings.is_enabled()
    }

    /// Returns `(time_secs, stones)` for the GTP `time_left` command for `color`.
    ///
    /// During main time, returns `(main_time_remaining, 0)`. During Japanese
    /// overtime, returns `(period_time_remaining, stones_left_in_period)`.
    ///
    /// GTP carries whole seconds, so the time is rounded down: an engine that
    /// spends all the time it is told about must not flag. A clock under one
    /// second is therefore reported as `0`.
    #[must_use]
    pub fn time_left(&self, color: PlayerColor) -> (u64, u32) {
        let player = &self.players[color.index()];
        match self.settings {
            TimeSettings::Byoyomi { time_per_move_ms } => (time_per_move_ms / 1000, 0),
            TimeSettings::JapaneseByo {
                byoyomi_time_ms, ..
            } if player.in_byoyomi => {
                let period_remaining_ms = byoyomi_time_ms.saturating_sub(player.byo_time_used_ms);
                (period_remaining_ms / 1000, player.byo_stones_left)
            }
            TimeSettings::Infinite
            | TimeSettings::Fischer { .. }
            | TimeSettings::JapaneseByo { .. } => (player.time_ms / 1000, 0),
        }
    }

    /// Starts the clock for a move.
    pub fn start_move(&mut self) {
        self.move_start = Some(Instant::now());
    }

    /// Ends the clock for a move, returning `false` if `color` flagged.
    pub fn end_move(&mut self, color: PlayerColor) -> bool {
        let elapsed_ms = self
            .move_start
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0);
        self.move_start = None;
        self.apply_elapsed(color, elapsed_ms)
    }

    fn apply_elapsed(&mut self, color: PlayerColor, elapsed_ms: u64) -> bool {
        match self.settings {
            TimeSettings::Infinite => true,
            TimeSettings::Byoyomi { time_per_move_ms } => elapsed_ms <= time_per_move_ms,
            TimeSettings::Fischer { increment_ms, .. } => {
                let player = &mut self.players[color.index()];
                if elapsed_ms >= player.time_ms {
                    player.time_ms = 0;
                    false
                } else {
                    player.time_ms = (player.time_ms - elapsed_ms).saturating_add(increment_ms);
                    true
                }
            }
            TimeSettings::JapaneseByo {
                byoyomi_time_ms,
                byoyomi_stones,
                ..
            } => Self::apply_japanese_byo(
                &mut self.players[color.index()],
                byoyomi_time_ms,
                byoyomi_stones.get(),
                elapsed_ms,
            ),
        }
    }

    fn apply_japanese_byo(
        player: &mut PlayerClock,
        byoyomi_time_ms: u64,
        byoyomi_stones: u32,
        elapsed_ms: u64,
    ) -> bool {
        if !player.in_byoyomi {
            if elapsed_ms < player.time_ms {
                player.time_ms -= elapsed_ms;
                return true;
            }
            let overtime = elapsed_ms - player.time_ms;
            player.time_ms = 0;
            player.in_byoyomi = true;
            player.byo_stones_left = byoyomi_stones;
            player.byo_time_used_ms = overtime;

            if overtime > byoyomi_time_ms {
                return false;
            }

            if overtime == 0 {
                // Exhausting main time exactly starts the first byoyomi period,
                // but does not consume a stone until the next move actually uses it.
                return true;
            }

            player.byo_stones_left -= 1;
            if player.byo_stones_left == 0 {
                player.byo_stones_left = byoyomi_stones;
                player.byo_time_used_ms = 0;
            }
            return true;
        }

        player.byo_time_used_ms += elapsed_ms;

        if player.byo_time_used_ms > byoyomi_time_ms {
            return false;
        }

        player.byo_stones_left -= 1;
        if player.byo_stones_left == 0 {
            player.byo_stones_left = byoyomi_stones;
            player.byo_time_used_ms = 0;
        }
        true
    }

    /// Resets the time tracker for a new game.
    pub fn reset(&mut self) {
        let player = PlayerClock::new(self.settings);
        self.players = [player; 2];
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
        assert!(tracker.apply_elapsed(PlayerColor::Black, 999_999));
        assert!(tracker.apply_elapsed(PlayerColor::White, 999_999));
    }

    #[test]
    fn test_stones_zeroed_outside_japanese_byo() {
        let tracker = TimeTracker::new(60, 0, 3);
        assert_eq!(tracker.settings().byoyomi_stones(), 0);
        assert!(tracker.is_enabled());
    }

    #[test]
    fn test_byoyomi_within_time() {
        let mut tracker = TimeTracker::new(0, 10, 0);
        assert!(tracker.is_enabled());
        assert!(tracker.apply_elapsed(PlayerColor::Black, 5_000)); // 5s < 10s
        assert_eq!(tracker.time_left(PlayerColor::Black), (10, 0));
    }

    #[test]
    fn test_byoyomi_exact_time() {
        let mut tracker = TimeTracker::new(0, 10, 0);
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000)); // 10s == 10s
    }

    #[test]
    fn test_byoyomi_over_time() {
        let mut tracker = TimeTracker::new(0, 10, 0);
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 15_000)); // 15s > 10s
    }

    #[test]
    fn test_fischer_increment_accumulates() {
        let mut tracker = TimeTracker::new(10, 5, 0); // 10s + 5s increment
        assert!(tracker.apply_elapsed(PlayerColor::Black, 3_000)); // use 3s
        // Remaining: 10000 - 3000 + 5000 = 12000
        assert_eq!(tracker.time_left(PlayerColor::Black).0, 12);
    }

    #[test]
    fn test_fischer_flag_exact_time() {
        let mut tracker = TimeTracker::new(10, 5, 0);
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 10_000)); // use exactly 10s -> flag
        assert_eq!(tracker.time_left(PlayerColor::Black).0, 0);
    }

    #[test]
    fn test_fischer_flag_over_time() {
        let mut tracker = TimeTracker::new(10, 5, 0);
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 15_000)); // use 15s > 10s
        assert_eq!(tracker.time_left(PlayerColor::Black).0, 0);
    }

    #[test]
    fn test_sudden_death() {
        let mut tracker = TimeTracker::new(60, 0, 0); // 60s, no increment
        assert!(tracker.is_enabled());
        assert!(tracker.apply_elapsed(PlayerColor::Black, 30_000)); // use 30s
        // Remaining: 60000 - 30000 + 0 = 30000
        assert_eq!(tracker.time_left(PlayerColor::Black).0, 30);
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 30_000)); // exactly 30s -> flag
    }

    #[test]
    fn test_japanese_byo_main_time() {
        let mut tracker = TimeTracker::new(300, 30, 1);
        assert!(tracker.is_enabled());
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000)); // use 10s of main time
        assert_eq!(tracker.time_left(PlayerColor::Black).0, 290);
    }

    #[test]
    fn test_japanese_byo_transition_success() {
        let mut tracker = TimeTracker::new(10, 30, 1); // 10s main, 30s byo
        // Use 15s -> exhausts 10s main time, 5s overtime (5s <= 30s -> OK)
        // With stones=1, the stone is consumed and period resets
        assert!(tracker.apply_elapsed(PlayerColor::Black, 15_000));
        // Now in byoyomi with a fresh period (stone consumed, period reset)
        assert_eq!(tracker.time_left(PlayerColor::Black), (30, 1));
    }

    #[test]
    fn test_japanese_byo_transition_fail() {
        let mut tracker = TimeTracker::new(10, 5, 1); // 10s main, 5s byo
        // Use 20s -> exhausts 10s main, 10s overtime (10s > 5s -> flag)
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 20_000));
    }

    #[test]
    fn test_japanese_byo_in_byoyomi_phase() {
        let mut tracker = TimeTracker::new(10, 30, 1);
        // Use exactly 10s → transitions to byoyomi immediately (overtime=0)
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000));
        // Now in byoyomi phase: fresh period
        assert_eq!(tracker.time_left(PlayerColor::Black), (30, 1));
        // Move in byoyomi: 25s <= 30s -> OK, period resets (stones=1)
        assert!(tracker.apply_elapsed(PlayerColor::Black, 25_000));
        // Exceed byoyomi: 35s > 30s -> flag
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 35_000));
    }

    #[test]
    fn test_reset() {
        let mut tracker = TimeTracker::new(300, 5, 0);
        tracker.apply_elapsed(PlayerColor::Black, 100_000); // black uses 100s
        tracker.reset();
        assert_eq!(tracker.time_left(PlayerColor::Black).0, 300);
        assert_eq!(tracker.time_left(PlayerColor::White).0, 300);
    }

    #[test]
    fn test_reset_restores_byoyomi_period_and_clears_running_clock() {
        let mut tracker = TimeTracker::new(0, 10, 2);
        assert!(tracker.apply_elapsed(PlayerColor::Black, 4_000));
        assert!(tracker.apply_elapsed(PlayerColor::White, 6_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (6, 1));
        assert_eq!(tracker.time_left(PlayerColor::White), (4, 1));
        tracker.start_move();
        assert!(tracker.move_start.is_some());
        tracker.reset();
        assert_eq!(tracker.time_left(PlayerColor::Black), (10, 2));
        assert_eq!(tracker.time_left(PlayerColor::White), (10, 2));
        assert!(tracker.move_start.is_none());
    }

    // =========================================================================
    // time_left GTP reporting
    // =========================================================================

    #[test]
    fn test_time_left_fischer_reports_main_time() {
        let mut tracker = TimeTracker::new(300, 5, 0);
        assert!(tracker.is_enabled());
        assert_eq!(tracker.settings().main_time_secs(), 300);
        assert_eq!(tracker.settings().byoyomi_time_secs(), 5);
        assert_eq!(tracker.time_left(PlayerColor::Black), (300, 0));
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (295, 0));
        assert_eq!(tracker.time_left(PlayerColor::White), (300, 0));
    }

    #[test]
    fn time_left_never_reports_more_than_the_clock_holds() {
        // Sudden death: 1,001 ms and then 1 ms on the clock.
        let mut tracker = TimeTracker::new(2, 0, 0);
        assert!(tracker.apply_elapsed(PlayerColor::Black, 999));
        assert_eq!(tracker.time_left(PlayerColor::Black), (1, 0));
        assert!(tracker.apply_elapsed(PlayerColor::Black, 1_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (0, 0));

        // Fischer: 1,001 ms after the increment.
        let mut tracker = TimeTracker::new(1, 1, 0);
        assert!(tracker.apply_elapsed(PlayerColor::Black, 999));
        assert_eq!(tracker.time_left(PlayerColor::Black), (1, 0));

        // Japanese byo-yomi main time: 1 ms left.
        let mut tracker = TimeTracker::new(1, 10, 2);
        assert!(tracker.apply_elapsed(PlayerColor::Black, 999));
        assert_eq!(tracker.time_left(PlayerColor::Black), (0, 0));

        // Japanese byo-yomi period: 999 ms left for the second stone.
        let mut tracker = TimeTracker::new(0, 10, 2);
        assert!(tracker.apply_elapsed(PlayerColor::Black, 9_001));
        assert_eq!(tracker.time_left(PlayerColor::Black), (0, 1));
    }

    #[test]
    fn test_time_left_japanese_byo_main_time_phase() {
        let tracker = TimeTracker::new(300, 30, 3);
        // In main time: reports main time, stones=0
        assert_eq!(tracker.time_left(PlayerColor::Black), (300, 0));
    }

    #[test]
    fn test_time_left_japanese_byo_period_reset() {
        let mut tracker = TimeTracker::new(0, 10, 2);
        // Start directly in byoyomi
        // Stone 1: 4s
        assert!(tracker.apply_elapsed(PlayerColor::Black, 4_000));
        // time_used=4s, 1 stone left, remaining=6s
        assert_eq!(tracker.time_left(PlayerColor::Black), (6, 1));
        // Stone 2: 3s → period resets
        assert!(tracker.apply_elapsed(PlayerColor::Black, 3_000));
        // After reset: time_used=0, stones=2, remaining=10s
        assert_eq!(tracker.time_left(PlayerColor::Black), (10, 2));
    }

    // =========================================================================
    // Japanese byo-yomi with multiple stones per period
    // =========================================================================

    #[test]
    fn test_japanese_byo_multi_stones_within_period() {
        // 10s main, 30s byoyomi, 3 stones per period
        let mut tracker = TimeTracker::new(10, 30, 3);
        // Exhaust main time
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (30, 3));
        // Enter byoyomi: 3 stones in 30s
        // Stone 1: 8s → cumulative 8s ≤ 30s, 2 stones left
        assert!(tracker.apply_elapsed(PlayerColor::Black, 8_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (22, 2));
        // Stone 2: 10s → cumulative 18s ≤ 30s, 1 stone left
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (12, 1));
        // Stone 3: 10s → cumulative 28s ≤ 30s, 0 stones left → period resets
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (30, 3));
        // New period starts: Stone 1: 5s → cumulative 5s ≤ 30s
        assert!(tracker.apply_elapsed(PlayerColor::Black, 5_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (25, 2));
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 25_001));
    }

    #[test]
    fn test_japanese_byo_multi_stones_transition_with_overtime() {
        // 5s main, 20s byoyomi, 2 stones per period
        let mut tracker = TimeTracker::new(5, 20, 2);
        // Use 12s → exhausts 5s main, 7s overtime into byoyomi
        // 7s ≤ 20s → OK, stone 1 consumed, 1 stone left
        assert!(tracker.apply_elapsed(PlayerColor::Black, 12_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (13, 1));
        // Stone 2: 10s → cumulative 7+10=17s ≤ 20s → OK, period resets
        assert!(tracker.apply_elapsed(PlayerColor::Black, 10_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (20, 2));
        // New period, stone 1: 15s → cumulative 15s ≤ 20s
        assert!(tracker.apply_elapsed(PlayerColor::Black, 15_000));
        assert_eq!(tracker.time_left(PlayerColor::Black), (5, 1));
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 5_001));
    }

    #[test]
    fn test_japanese_byo_multi_stones_period_resets_correctly() {
        // 0s main (start in byoyomi immediately), 10s byoyomi, 2 stones per period
        let mut tracker = TimeTracker::new(0, 10, 2);
        // Immediately in byoyomi (main_time=0, byoyomi_stones>0 → JapaneseByo mode
        // with initial_in_byoyomi=true)
        // Stone 1: 4s ≤ 10s
        assert!(tracker.apply_elapsed(PlayerColor::Black, 4_000));
        // Stone 2: 5s → cumulative 9s ≤ 10s → period resets
        assert!(tracker.apply_elapsed(PlayerColor::Black, 5_000));
        // New period: Stone 1: 6s → 6s ≤ 10s
        assert!(tracker.apply_elapsed(PlayerColor::Black, 6_000));
        // Stone 2: 5s → cumulative 11s > 10s → flag
        assert!(!tracker.apply_elapsed(PlayerColor::Black, 5_000));
    }

    #[test]
    fn test_all_eight_time_settings_combinations() {
        let untimed_zero = TimeTracker::new(0, 0, 0);
        assert!(!untimed_zero.is_enabled());
        assert_eq!(untimed_zero.settings().byoyomi_stones(), 0);

        let untimed_stones = TimeTracker::new(0, 0, 3);
        assert!(!untimed_stones.is_enabled());
        assert_eq!(untimed_stones.settings().byoyomi_stones(), 0);

        let pure_byo = TimeTracker::new(0, 5, 0);
        assert!(pure_byo.is_enabled());
        assert_eq!(pure_byo.settings().byoyomi_stones(), 0);
        assert_eq!(pure_byo.time_left(PlayerColor::Black), (5, 0));

        let mainless_jp_byo = TimeTracker::new(0, 10, 2);
        assert!(mainless_jp_byo.is_enabled());
        assert_eq!(mainless_jp_byo.settings().byoyomi_stones(), 2);
        assert_eq!(mainless_jp_byo.time_left(PlayerColor::Black), (10, 2));

        let sudden_death = TimeTracker::new(60, 0, 0);
        assert!(sudden_death.is_enabled());
        assert_eq!(sudden_death.settings().byoyomi_stones(), 0);
        assert_eq!(sudden_death.time_left(PlayerColor::Black), (60, 0));

        let sudden_death_stones = TimeTracker::new(60, 0, 3);
        assert!(sudden_death_stones.is_enabled());
        assert_eq!(sudden_death_stones.settings().byoyomi_stones(), 0);
        assert_eq!(sudden_death_stones.time_left(PlayerColor::Black), (60, 0));

        let fischer = TimeTracker::new(60, 5, 0);
        assert!(fischer.is_enabled());
        assert_eq!(fischer.settings().byoyomi_stones(), 0);
        assert_eq!(fischer.time_left(PlayerColor::Black), (60, 0));

        let jp_byo = TimeTracker::new(60, 10, 2);
        assert!(jp_byo.is_enabled());
        assert_eq!(jp_byo.settings().byoyomi_stones(), 2);
        assert_eq!(jp_byo.time_left(PlayerColor::Black), (60, 0));
    }
}
