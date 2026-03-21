//! Time control management for timed games.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::square::Square;
use crate::types::Depth;

/// Safety buffer in milliseconds to avoid time forfeit.
const TIME_BUFFER_MS: u64 = 50;

/// Depth threshold after which PV/score instability becomes meaningful.
const MIN_STABILITY_CHECK_DEPTH: Depth = 10;

/// Score drop (in discs) that triggers an emergency extension.
const SCORE_DROP_THRESHOLD: f32 = 3.0;

/// Fraction of the reserve (hard_limit - base_maxi) used as the extension budget.
const EXTENSION_RESERVE_RATIO: f64 = 0.5;

/// Maximum number of incremental time extensions allowed per move.
const MAX_EXTENSION_STEPS: u8 = 5;

/// Consecutive stable best-move iterations before allowing early stop.
const STABILITY_THRESHOLD: u32 = 3;

/// Sentinel value indicating no previous best move has been recorded.
const NO_PREV_MOVE: u8 = Square::None as u8;

// Time allocation percentages (0-100)
const MIN_PERCENT_NORMAL: u64 = 45;
const MIN_PERCENT_ENDGAME: u64 = 80;
const BYOYOMI_MAX_PERCENT: u64 = 90;
const FISCHER_MAX_PERCENT: u64 = 90;
const MOVESTOGO_MAX_PERCENT: u64 = 95;
const JP_BYO_MAIN_MIN_PERCENT_NORMAL: u64 = 60;
const JP_BYO_MAIN_MIN_PERCENT_ENDGAME: u64 = 85;

/// Calculates a time allocation factor based on game phase using a smooth bell curve.
///
/// Uses an asymmetric Gaussian that peaks during midgame and tapers toward
/// opening (wider spread) and endgame (narrower spread).
fn get_time_allocation_factor(n_empties: u32) -> f64 {
    const AMPLITUDE: f64 = 2.9;
    const BASE: f64 = 0.1;
    const CENTER: f64 = 38.0;
    const SIGMA_OPENING: f64 = 12.0;
    const SIGMA_ENDGAME: f64 = 8.0;

    let x = n_empties as f64;
    let sigma = if x >= CENTER {
        SIGMA_OPENING
    } else {
        SIGMA_ENDGAME
    };
    let d = x - CENTER;
    let exponent = -d * d / (2.0 * sigma * sigma);
    exponent.exp().mul_add(AMPLITUDE, BASE)
}

/// Calculates the sum of time allocation factors for remaining moves.
fn calculate_remaining_factor_sum(n_empties: u32) -> f64 {
    let mut sum = 0.0;
    let mut e = n_empties as i32;
    while e > 0 {
        sum += get_time_allocation_factor(e as u32);
        e -= 2;
    }
    sum
}

/// Returns the default minimum time percentage based on game phase.
fn default_min_percent(is_endgame: bool) -> u64 {
    if is_endgame {
        MIN_PERCENT_ENDGAME
    } else {
        MIN_PERCENT_NORMAL
    }
}

/// Time control mode for a game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeControlMode {
    /// No time limit.
    #[default]
    Infinite,

    /// Fixed time per move.
    /// `time_per_move_ms` is the time allowed for each move in milliseconds.
    Byoyomi { time_per_move_ms: u64 },

    /// Fischer time control (increment per move).
    /// `main_time_ms` is the initial time bank.
    /// `increment_ms` is the time added after each move.
    Fischer {
        main_time_ms: u64,
        increment_ms: u64,
    },

    /// Fixed time for a number of moves.
    /// `time_ms` is the total time for `moves` moves.
    MovesToGo { time_ms: u64, moves: u32 },

    /// Japanese byoyomi.
    /// `main_time_ms` is the initial time bank (can be 0 to start in byoyomi).
    /// `time_per_move_ms` is the fixed time per move after main time expires.
    JapaneseByo {
        main_time_ms: u64,
        time_per_move_ms: u64,
    },
}

/// Manages time allocation and tracking during search.
#[derive(Debug)]
pub struct TimeManager {
    /// Time control mode for the current game.
    mode: TimeControlMode,

    /// Start time of the current search.
    start_time: Instant,

    /// Minimum time to use before considering stopping (milliseconds).
    /// Search should use at least this much time unless forced move.
    min_time_ms: AtomicU64,

    /// Maximum time allowed for this move (milliseconds).
    /// Search must stop before this time is reached.
    max_time_ms: AtomicU64,

    /// Baseline maximum time for the current move before any extensions.
    base_max_time_ms: AtomicU64,

    /// Absolute hard limit for this move (remaining time - buffer).
    /// Neither initial allocation nor extensions can exceed this.
    hard_time_limit_ms: AtomicU64,

    /// Number of extension steps already applied this move.
    extension_steps: AtomicU8,

    /// Reference to the abort flag for signaling search termination.
    abort_flag: Arc<AtomicBool>,

    /// Previous iteration's score (for detecting score drops).
    prev_score: Mutex<Option<f32>>,

    /// Number of empty squares at search start (for estimating remaining moves).
    n_empties: u32,

    /// Flag indicating if we are in endgame search mode.
    is_endgame_mode: AtomicBool,

    /// Consecutive iterations where the best move has not changed.
    best_move_stability: AtomicU32,

    /// Best move from the previous iteration (raw u8, NO_PREV_MOVE if unset).
    prev_best_move: AtomicU8,

    /// Suppresses one early-stop check after a score-drop extension is granted.
    skip_early_stop_once: AtomicBool,
}

/// Reason for requesting a time extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtensionReason {
    ScoreDrop,
    PvChange,
}

impl TimeManager {
    /// Creates a new time manager with the specified mode and abort flag.
    pub fn new(mode: TimeControlMode, abort_flag: Arc<AtomicBool>, n_empties: u32) -> Self {
        let (mini_time_ms, maxi_time_ms, hard_limit_ms) =
            Self::calculate_time_limits(mode, n_empties, false);

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] New: mode={:?}, empties={}, mini={}ms, maxi={}ms, hard_limit={}ms",
                mode, n_empties, mini_time_ms, maxi_time_ms, hard_limit_ms
            );
        }

        TimeManager {
            mode,
            start_time: Instant::now(),
            min_time_ms: AtomicU64::new(mini_time_ms),
            max_time_ms: AtomicU64::new(maxi_time_ms),
            base_max_time_ms: AtomicU64::new(maxi_time_ms),
            hard_time_limit_ms: AtomicU64::new(hard_limit_ms),
            extension_steps: AtomicU8::new(0),
            abort_flag,
            prev_score: Mutex::new(None),
            n_empties,
            is_endgame_mode: AtomicBool::new(false),
            best_move_stability: AtomicU32::new(0),
            prev_best_move: AtomicU8::new(NO_PREV_MOVE),
            skip_early_stop_once: AtomicBool::new(false),
        }
    }

    /// Calculates safe time limit based on time control mode.
    fn calculate_safe_time(main_time_ms: u64, n_empties: u32) -> u64 {
        let my_future_moves = n_empties.saturating_sub(1).div_ceil(2);
        let total_buffer = TIME_BUFFER_MS + ((my_future_moves as u64 * TIME_BUFFER_MS) / 2);

        main_time_ms.saturating_sub(total_buffer)
    }

    /// Calculates mini and maxi time limits based on time control mode.
    fn calculate_time_limits(
        mode: TimeControlMode,
        n_empties: u32,
        is_endgame: bool,
    ) -> (u64, u64, u64) {
        match mode {
            TimeControlMode::Infinite => (u64::MAX, u64::MAX, u64::MAX),

            TimeControlMode::Byoyomi { time_per_move_ms } => {
                Self::byoyomi_limits(time_per_move_ms, is_endgame)
            }

            TimeControlMode::Fischer {
                main_time_ms,
                increment_ms,
            } => {
                let hard_limit = Self::calculate_safe_time(main_time_ms, n_empties);
                let budget = Self::allocate_budget(main_time_ms, increment_ms, n_empties);
                Self::compute_limits(
                    budget,
                    budget,
                    default_min_percent(is_endgame),
                    FISCHER_MAX_PERCENT,
                    hard_limit,
                )
            }

            TimeControlMode::MovesToGo { time_ms, moves } => {
                let hard_limit = time_ms.saturating_sub(TIME_BUFFER_MS);
                let moves = moves.max(1) as u64;
                let time_per_move = time_ms / moves;
                Self::compute_limits(
                    time_per_move,
                    time_per_move,
                    default_min_percent(is_endgame),
                    MOVESTOGO_MAX_PERCENT,
                    hard_limit,
                )
            }

            TimeControlMode::JapaneseByo {
                main_time_ms,
                time_per_move_ms,
            } => {
                if main_time_ms == 0 {
                    Self::byoyomi_limits(time_per_move_ms, is_endgame)
                } else {
                    let hard_limit = Self::calculate_safe_time(main_time_ms, n_empties);
                    let allocated_time = Self::allocate_budget(main_time_ms, 0, n_empties);
                    let mini_pct = if is_endgame {
                        JP_BYO_MAIN_MIN_PERCENT_ENDGAME
                    } else {
                        JP_BYO_MAIN_MIN_PERCENT_NORMAL
                    };
                    Self::compute_limits(allocated_time, allocated_time, mini_pct, 100, hard_limit)
                }
            }
        }
    }

    /// Calculates budget based on time factor sum.
    fn allocate_budget(main_time_ms: u64, increment_ms: u64, n_empties: u32) -> u64 {
        let total_factor = calculate_remaining_factor_sum(n_empties);
        let current_factor = get_time_allocation_factor(n_empties);

        let time_fraction = if total_factor > 0.0 {
            current_factor / total_factor
        } else {
            1.0 / n_empties.max(1) as f64
        };

        let base_budget = (main_time_ms as f64 * time_fraction) as u64;
        base_budget + increment_ms
    }

    /// Computes final limits with clamping.
    fn compute_limits(
        budget_mini: u64,
        budget_maxi: u64,
        mini_pct: u64,
        maxi_pct: u64,
        hard_limit: u64,
    ) -> (u64, u64, u64) {
        let allocated_mini = (budget_mini * mini_pct) / 100;
        let allocated_maxi = (budget_maxi * maxi_pct) / 100;

        let mini = allocated_mini.min(hard_limit);
        let maxi = allocated_maxi.min(hard_limit);

        (mini, maxi, hard_limit)
    }

    /// Calculates time limits for byoyomi-style time control.
    fn byoyomi_limits(time_per_move_ms: u64, is_endgame: bool) -> (u64, u64, u64) {
        let available = time_per_move_ms.saturating_sub(TIME_BUFFER_MS);
        Self::compute_limits(
            available,
            available,
            default_min_percent(is_endgame),
            BYOYOMI_MAX_PERCENT,
            available,
        )
    }

    /// Starts the timer for a new search.
    pub fn start(&mut self) {
        self.start_time = Instant::now();
        self.extension_steps.store(0, Ordering::Relaxed);
        let current_maxi = self.max_time_ms.load(Ordering::Relaxed);
        self.base_max_time_ms.store(current_maxi, Ordering::Relaxed);
        *self.prev_score.lock().unwrap() = None;
        self.is_endgame_mode.store(false, Ordering::Relaxed);
        self.best_move_stability.store(0, Ordering::Relaxed);
        self.prev_best_move.store(NO_PREV_MOVE, Ordering::Relaxed);
        self.skip_early_stop_once.store(false, Ordering::Relaxed);
    }

    /// Reports the iteration result: tracks best move stability and extends time on instability.
    pub fn report_iteration(&self, sq: Square, current_score: f32, depth: Depth) {
        let pv_changed = if depth >= MIN_STABILITY_CHECK_DEPTH {
            let prev_raw = self.prev_best_move.swap(sq as u8, Ordering::Relaxed);
            let pv_changed = prev_raw != NO_PREV_MOVE && prev_raw != sq as u8;

            if !pv_changed {
                if prev_raw != NO_PREV_MOVE {
                    self.best_move_stability.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                self.best_move_stability.store(0, Ordering::Relaxed);
            }

            if is_debug_enabled() {
                eprintln!(
                    "[TimeManager] Best move: {:?}, stability={}",
                    sq,
                    self.best_move_stability.load(Ordering::Relaxed)
                );
            }

            pv_changed
        } else {
            false
        };

        self.try_extend_time(current_score, pv_changed, depth);
    }

    /// Returns the elapsed time in milliseconds since search started.
    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Checks whether the search has exceeded the maximum time limit.
    #[inline]
    pub fn is_time_up(&self) -> bool {
        if self.mode == TimeControlMode::Infinite {
            return false;
        }
        self.elapsed_ms() >= self.max_time_ms.load(Ordering::Relaxed)
    }

    /// Returns true if the current mode uses a shared time bank (Fischer or MovesToGo).
    fn has_time_bank(&self) -> bool {
        matches!(
            self.mode,
            TimeControlMode::Fischer { .. } | TimeControlMode::MovesToGo { .. }
        )
    }

    /// Returns a scaling factor for min_time based on best move stability.
    ///
    /// Higher stability → lower scale → min_time is reduced, enabling earlier stop.
    /// Only meaningful for time-bank modes (Fischer/MovesToGo).
    fn stability_time_scale(&self) -> f64 {
        if !self.has_time_bank() {
            return 1.0;
        }

        const SCALE: [f64; 5] = [1.0, 1.0, 0.70, 0.50, 0.35];
        let idx = (self.best_move_stability.load(Ordering::Relaxed) as usize).min(SCALE.len() - 1);
        SCALE[idx]
    }

    /// Checks whether the search should continue to the next iteration.
    pub fn should_continue_iteration(&self) -> bool {
        if self.mode == TimeControlMode::Infinite {
            return true;
        }

        let elapsed = self.elapsed_ms();
        let scale = self.stability_time_scale();
        let effective_min = (self.min_time_ms.load(Ordering::Relaxed) as f64 * scale) as u64;
        if elapsed < effective_min {
            return true;
        }
        let skip_early_stop = self.skip_early_stop_once.swap(false, Ordering::Relaxed);

        // Early stop: best move has been stable for several iterations
        let stability = self.best_move_stability.load(Ordering::Relaxed);
        if self.has_time_bank() && stability >= STABILITY_THRESHOLD {
            if skip_early_stop {
                if is_debug_enabled() {
                    eprintln!(
                        "[TimeManager] Continue after score-drop extension: elapsed={}ms, effective_min={}ms, stability={}",
                        elapsed, effective_min, stability
                    );
                }
                return true;
            }

            if is_debug_enabled() {
                eprintln!(
                    "[TimeManager] Early stop (stable best move): elapsed={}ms, effective_min={}ms, stability={}",
                    elapsed, effective_min, stability
                );
            }
            return false;
        }

        let should_continue = (elapsed as f64 * 1.5) < self.maxi_time_ms() as f64;
        if !should_continue && is_debug_enabled() {
            eprintln!(
                "[TimeManager] Stopping iteration: elapsed={}ms, maxi={}ms",
                elapsed,
                self.max_time_ms.load(Ordering::Relaxed)
            );
        }

        should_continue
    }

    /// Attempts to extend the search time when the search becomes unstable.
    fn try_extend_time(&self, current_score: f32, pv_changed: bool, depth: Depth) -> bool {
        if self.mode == TimeControlMode::Infinite {
            *self.prev_score.lock().unwrap() = Some(current_score);
            return false;
        }

        let used_steps = self.extension_steps.load(Ordering::Relaxed);
        if used_steps >= MAX_EXTENSION_STEPS {
            *self.prev_score.lock().unwrap() = Some(current_score);
            return false;
        }

        let (should_extend, reason, prev_value) = {
            let mut prev_guard = self.prev_score.lock().unwrap();
            let prev = *prev_guard;
            *prev_guard = Some(current_score); // Always update

            let mut extend = false;
            let mut r = ExtensionReason::PvChange;

            if let Some(p) = prev
                && current_score < p - SCORE_DROP_THRESHOLD
            {
                extend = true;
                r = ExtensionReason::ScoreDrop;
            }
            if !extend && pv_changed && depth >= MIN_STABILITY_CHECK_DEPTH {
                extend = true;
                r = ExtensionReason::PvChange;
            }
            (extend, r, prev)
        };

        if !should_extend {
            return false;
        }

        self.apply_extension(reason, used_steps, prev_value, current_score)
    }

    fn apply_extension(
        &self,
        reason: ExtensionReason,
        used_steps: u8,
        prev_value: Option<f32>,
        current_score: f32,
    ) -> bool {
        let base_maxi = self.base_max_time_ms.load(Ordering::Relaxed);
        let hard_limit = self.hard_time_limit_ms.load(Ordering::Relaxed);
        let old_maxi = self.max_time_ms.load(Ordering::Relaxed);

        // In Japanese Byoyomi main time, we treat the hard limit as a soft limit for extensions
        // because falling into byoyomi is acceptable.
        let target_maxi = if matches!(self.mode, TimeControlMode::JapaneseByo { main_time_ms, .. } if main_time_ms > 0)
        {
            // Allow using up to 25% of the remaining reserve (allowance before hard limit)
            let reserve = hard_limit.saturating_sub(base_maxi);
            base_maxi.saturating_add(reserve / 4).min(hard_limit)
        } else {
            let reserve = hard_limit.saturating_sub(base_maxi);
            let extension_amount = ((reserve as f64) * EXTENSION_RESERVE_RATIO) as u64;
            base_maxi.saturating_add(extension_amount).min(hard_limit)
        };

        if old_maxi >= target_maxi {
            return false;
        }

        let remaining_steps = (MAX_EXTENSION_STEPS - used_steps) as u64;
        let remaining_budget = target_maxi.saturating_sub(old_maxi);
        let base_step = remaining_budget.div_ceil(remaining_steps);

        if base_step == 0 {
            return false;
        }

        // Score drops get a double-sized step and consume 2 extension steps
        let is_score_drop = reason == ExtensionReason::ScoreDrop;
        let got_double_step = is_score_drop && remaining_steps >= 2;
        let (step_increment, steps_consumed) = if got_double_step {
            ((base_step * 2).min(remaining_budget), 2u8)
        } else {
            (base_step, 1u8)
        };

        let new_maxi = old_maxi.saturating_add(step_increment).min(target_maxi);
        self.max_time_ms.store(new_maxi, Ordering::Relaxed);
        self.extension_steps
            .fetch_add(steps_consumed, Ordering::Relaxed);
        self.skip_early_stop_once
            .store(got_double_step, Ordering::Relaxed);

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] Time extended ({reason:?}, +{steps_consumed} step, {}/{}): \
                 {:.2} -> {:.2}, old={}ms, new={}ms, limit={}ms",
                used_steps + steps_consumed,
                MAX_EXTENSION_STEPS,
                prev_value.unwrap_or(current_score),
                current_score,
                old_maxi,
                new_maxi,
                hard_limit
            );
        }

        true
    }

    /// Signals the search to abort due to time-out.
    pub fn signal_abort(&self) {
        self.abort_flag.store(true, Ordering::Release);
    }

    /// Checks whether abort has been signaled.
    #[inline]
    pub fn is_aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }

    /// Checks whether time is up and signals abort if so.
    #[inline]
    pub fn check_time(&self) -> bool {
        if self.is_time_up() {
            if !self.is_aborted() {
                if is_debug_enabled() {
                    eprintln!(
                        "[TimeManager] Time up! elapsed={}ms, maxi={}ms",
                        self.elapsed_ms(),
                        self.max_time_ms.load(Ordering::Relaxed)
                    );
                }
                self.signal_abort();
            }
            true
        } else {
            false
        }
    }

    /// Updates the remaining time (for Fischer/MovesToGo modes).
    pub fn update_remaining_time(&mut self, remaining_time_ms: u64, n_empties: u32) {
        self.n_empties = n_empties;

        // Update mode parameters
        match &mut self.mode {
            TimeControlMode::Fischer { main_time_ms, .. } => *main_time_ms = remaining_time_ms,
            TimeControlMode::MovesToGo { time_ms, moves } => {
                *time_ms = remaining_time_ms;
                if *moves > 0 {
                    *moves -= 1;
                }
            }
            _ => return, // No update needed for other modes
        }

        // Recalculate limits
        let is_endgame = self.is_endgame_mode.load(Ordering::Relaxed);
        let (mini, maxi, hard_limit) =
            Self::calculate_time_limits(self.mode, n_empties, is_endgame);

        self.update_limits(mini, maxi, hard_limit);

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] Updated time: remaining={}ms, empties={}, new_mini={}ms, new_maxi={}ms",
                remaining_time_ms,
                n_empties,
                self.min_time_ms.load(Ordering::Relaxed),
                self.max_time_ms.load(Ordering::Relaxed)
            );
        }
    }

    fn update_limits(&self, mini: u64, maxi: u64, hard_limit: u64) {
        self.min_time_ms.store(mini, Ordering::Relaxed);
        self.max_time_ms.store(maxi, Ordering::Relaxed);
        self.base_max_time_ms.store(maxi, Ordering::Relaxed);
        self.hard_time_limit_ms.store(hard_limit, Ordering::Relaxed);
        self.extension_steps.store(0, Ordering::Relaxed);
        self.skip_early_stop_once.store(false, Ordering::Relaxed);
    }

    /// Returns the current time control mode.
    pub fn mode(&self) -> TimeControlMode {
        self.mode
    }

    /// Returns the minimum time in milliseconds.
    pub fn mini_time_ms(&self) -> u64 {
        self.min_time_ms.load(Ordering::Relaxed)
    }

    /// Returns the maximum time in milliseconds.
    pub fn maxi_time_ms(&self) -> u64 {
        self.max_time_ms.load(Ordering::Relaxed)
    }

    /// Returns the deadline instant, or None for infinite mode.
    pub fn deadline(&self) -> Option<Instant> {
        if self.mode == TimeControlMode::Infinite {
            None
        } else {
            Some(self.start_time + Duration::from_millis(self.max_time_ms.load(Ordering::Relaxed)))
        }
    }

    /// Returns the remaining time in milliseconds.
    #[inline]
    pub fn remaining_time_ms(&self) -> u64 {
        self.max_time_ms
            .load(Ordering::Relaxed)
            .saturating_sub(self.elapsed_ms())
    }

    /// Sets whether the search is in endgame mode.
    pub fn set_endgame_mode(&self, enabled: bool) {
        self.is_endgame_mode.store(enabled, Ordering::Relaxed);
        // Recalculate limits with new mode
        let (mini, maxi, hard_limit) =
            Self::calculate_time_limits(self.mode, self.n_empties, enabled);
        self.update_limits(mini, maxi, hard_limit);

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] Endgame mode set to {}: mini={}ms, maxi={}ms",
                enabled, mini, maxi
            );
        }
    }
}

/// Determines whether to stop the current search iteration based on time control.
///
/// Returns `true` if time is up or the iteration should not continue.
/// Returns `false` if no time manager is provided (unlimited search) or there
/// is still time remaining.
#[inline]
pub fn should_stop_iteration(time_manager: &Option<Arc<TimeManager>>) -> bool {
    match time_manager {
        Some(tm) => tm.check_time() || !tm.should_continue_iteration(),
        None => false,
    }
}

fn is_debug_enabled() -> bool {
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| {
        let env_var = std::env::var("REVERSI_DEBUG_TIME").unwrap_or_default();
        env_var == "1" || env_var.to_lowercase() == "true"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fischer_tm(main_time_ms: u64, increment_ms: u64, n_empties: u32) -> TimeManager {
        let abort = Arc::new(AtomicBool::new(false));
        let mode = TimeControlMode::Fischer {
            main_time_ms,
            increment_ms,
        };
        TimeManager::new(mode, abort, n_empties)
    }

    #[test]
    fn stability_scale_returns_1_for_low_stability() {
        let tm = make_fischer_tm(60_000, 0, 40);
        assert_eq!(tm.best_move_stability.load(Ordering::Relaxed), 0);
        assert_eq!(tm.stability_time_scale(), 1.0);
    }

    #[test]
    fn stability_scale_decreases_with_higher_stability() {
        let tm = make_fischer_tm(60_000, 0, 40);
        tm.best_move_stability.store(2, Ordering::Relaxed);
        assert!(tm.stability_time_scale() < 1.0);

        tm.best_move_stability.store(3, Ordering::Relaxed);
        let scale_3 = tm.stability_time_scale();

        tm.best_move_stability.store(4, Ordering::Relaxed);
        let scale_4 = tm.stability_time_scale();

        assert!(
            scale_3 > scale_4,
            "higher stability should give lower scale"
        );
    }

    #[test]
    fn extension_budget_increased() {
        const { assert!(MAX_EXTENSION_STEPS >= 5) };
        const { assert!(EXTENSION_RESERVE_RATIO >= 0.1) };
    }

    #[test]
    fn score_drop_extends_more_than_pv_change() {
        let mut tm = make_fischer_tm(120_000, 0, 40);
        tm.start();
        let sq = Square::D3;
        tm.report_iteration(sq, 5.0, 12);
        let base_maxi = tm.maxi_time_ms();

        // PV change extension
        let mut tm_pv = make_fischer_tm(120_000, 0, 40);
        tm_pv.start();
        tm_pv.report_iteration(sq, 5.0, 12);
        let sq2 = Square::C4;
        tm_pv.report_iteration(sq2, 4.0, 13);
        let maxi_after_pv = tm_pv.maxi_time_ms();

        // Score drop extension
        let mut tm_sd = make_fischer_tm(120_000, 0, 40);
        tm_sd.start();
        tm_sd.report_iteration(sq, 5.0, 12);
        tm_sd.report_iteration(sq, 1.0, 13);
        let maxi_after_sd = tm_sd.maxi_time_ms();

        let pv_extension = maxi_after_pv - base_maxi;
        let sd_extension = maxi_after_sd - base_maxi;
        assert!(
            sd_extension >= pv_extension,
            "score drop extension ({sd_extension}) should be >= pv change extension ({pv_extension})"
        );
    }

    #[test]
    fn early_stop_respects_dynamic_min_time() {
        let mut tm = make_fischer_tm(60_000, 0, 40);
        tm.start();
        let original_min = tm.mini_time_ms();

        tm.best_move_stability.store(4, Ordering::Relaxed);
        let scale = tm.stability_time_scale();
        let effective_min = (original_min as f64 * scale) as u64;

        assert!(
            effective_min < original_min,
            "effective min ({effective_min}) should be less than original ({original_min})"
        );
    }

    #[test]
    fn stability_ignores_shallow_iterations() {
        let mut tm = make_fischer_tm(60_000, 0, 40);
        tm.start();
        let sq = Square::D3;

        tm.report_iteration(sq, 5.0, 8);
        assert_eq!(tm.prev_best_move.load(Ordering::Relaxed), NO_PREV_MOVE);
        assert_eq!(tm.best_move_stability.load(Ordering::Relaxed), 0);

        tm.report_iteration(sq, 5.0, 10);
        assert_eq!(tm.best_move_stability.load(Ordering::Relaxed), 0);

        tm.report_iteration(sq, 5.0, 11);
        assert_eq!(tm.best_move_stability.load(Ordering::Relaxed), 1);

        tm.report_iteration(sq, 5.0, 12);
        assert_eq!(tm.best_move_stability.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn score_drop_extension_skips_immediate_stable_early_stop_once() {
        let mut tm = make_fischer_tm(120_000, 0, 40);
        tm.start();
        let sq = Square::D3;

        tm.report_iteration(sq, 5.0, 12);
        tm.best_move_stability
            .store(STABILITY_THRESHOLD, Ordering::Relaxed);
        tm.start_time = Instant::now() - Duration::from_millis(tm.maxi_time_ms() / 2);

        let base_maxi = tm.maxi_time_ms();
        tm.report_iteration(sq, 1.0, 13);

        assert!(tm.maxi_time_ms() > base_maxi);
        assert!(tm.should_continue_iteration());
        assert!(!tm.should_continue_iteration());
    }

    #[test]
    fn stability_resets_on_pv_change() {
        let mut tm = make_fischer_tm(60_000, 0, 40);
        tm.start();
        let sq = Square::D3;

        // Build up stability
        tm.report_iteration(sq, 5.0, 10);
        tm.report_iteration(sq, 5.0, 11);
        tm.report_iteration(sq, 5.0, 12);
        assert_eq!(tm.best_move_stability.load(Ordering::Relaxed), 2);

        // PV change resets stability
        tm.report_iteration(Square::C4, 5.0, 13);
        assert_eq!(tm.best_move_stability.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn score_drop_with_one_remaining_step_falls_back_to_single() {
        let mut tm = make_fischer_tm(120_000, 0, 40);
        tm.start();
        let sq = Square::D3;

        // Exhaust all but 1 extension step
        tm.extension_steps
            .store(MAX_EXTENSION_STEPS - 1, Ordering::Relaxed);

        tm.report_iteration(sq, 5.0, 12);
        tm.report_iteration(sq, 1.0, 13); // score drop with 1 step remaining

        assert_eq!(
            tm.extension_steps.load(Ordering::Relaxed),
            MAX_EXTENSION_STEPS
        );
        // skip_early_stop_once should NOT be set (double-step was not granted)
        assert!(!tm.skip_early_stop_once.load(Ordering::Relaxed));
    }

    #[test]
    fn byoyomi_ignores_stability_early_stop() {
        let abort = Arc::new(AtomicBool::new(false));
        let mode = TimeControlMode::Byoyomi {
            time_per_move_ms: 10_000,
        };
        let tm = TimeManager::new(mode, abort, 40);
        assert!(!tm.has_time_bank());
    }
}
