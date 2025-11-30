//! Time control management for timed games.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::types::Depth;

/// Safety buffer in milliseconds to avoid time forfeit.
const TIME_BUFFER_MS: u64 = 50;

/// Depth threshold after which PV/score instability becomes meaningful.
const MIN_STABILITY_CHECK_DEPTH: Depth = 10;

/// Score drop (in discs) that triggers an emergency extension.
const SCORE_DROP_THRESHOLD: f32 = 3.0;

/// Additional time granted on instability (percentage of current maxi).
const EXTENSION_RATIO: f64 = 0.5;

/// Maximum number of incremental time extensions allowed per move.
const MAX_EXTENSION_STEPS: u8 = 3;

// Time allocation percentages (0-100)
const MIN_PERCENT_NORMAL: u64 = 45;
const MIN_PERCENT_ENDGAME: u64 = 80;
const BYOYOMI_MAX_PERCENT: u64 = 90;
const FISCHER_MAX_PERCENT: u64 = 90;
const MOVESTOGO_MAX_PERCENT: u64 = 95;
const JP_BYO_MAIN_MIN_PERCENT_NORMAL: u64 = 60;
const JP_BYO_MAIN_MIN_PERCENT_ENDGAME: u64 = 85;

/// Calculates a time allocation factor based on game phase.
///
/// # Arguments
///
/// * `n_empties` - Number of empty squares on the board
///
/// # Returns
///
/// A multiplier for time allocation (1.0 = base allocation)
fn get_time_allocation_factor(n_empties: u32) -> f64 {
    match n_empties {
        51..=60 => 0.5,
        45..=50 => 1.5,
        29..=44 => 2.5,
        25..=28 => 1.5,
        20..=24 => 0.8,
        _ => 0.3,
    }
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
}

impl TimeManager {
    /// Creates a new TimeManager with the specified mode and abort flag.
    ///
    /// # Arguments
    ///
    /// * `mode` - The time control mode
    /// * `abort_flag` - Shared abort flag for search termination
    /// * `n_empties` - Number of empty squares on the board
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
                let available = time_per_move_ms.saturating_sub(TIME_BUFFER_MS);
                let mini_pct = if is_endgame {
                    MIN_PERCENT_ENDGAME
                } else {
                    MIN_PERCENT_NORMAL
                };
                Self::compute_limits(
                    available,
                    available,
                    mini_pct,
                    BYOYOMI_MAX_PERCENT,
                    available,
                )
            }

            TimeControlMode::Fischer {
                main_time_ms,
                increment_ms,
            } => {
                let hard_limit = Self::calculate_safe_time(main_time_ms, n_empties);
                let budget = Self::allocate_budget(main_time_ms, increment_ms, n_empties);
                let mini_pct = if is_endgame {
                    MIN_PERCENT_ENDGAME
                } else {
                    MIN_PERCENT_NORMAL
                };

                Self::compute_limits(budget, budget, mini_pct, FISCHER_MAX_PERCENT, hard_limit)
            }

            TimeControlMode::MovesToGo { time_ms, moves } => {
                let hard_limit = time_ms.saturating_sub(TIME_BUFFER_MS);
                let moves = moves.max(1) as u64;
                let time_per_move = time_ms / moves;
                let mini_pct = if is_endgame {
                    MIN_PERCENT_ENDGAME
                } else {
                    MIN_PERCENT_NORMAL
                };

                Self::compute_limits(
                    time_per_move,
                    time_per_move,
                    mini_pct,
                    MOVESTOGO_MAX_PERCENT,
                    hard_limit,
                )
            }

            TimeControlMode::JapaneseByo {
                main_time_ms,
                time_per_move_ms,
            } => {
                if main_time_ms == 0 {
                    // Already in byoyomi
                    let available = time_per_move_ms.saturating_sub(TIME_BUFFER_MS);
                    let mini_pct = if is_endgame {
                        MIN_PERCENT_ENDGAME
                    } else {
                        MIN_PERCENT_NORMAL
                    };
                    Self::compute_limits(
                        available,
                        available,
                        mini_pct,
                        BYOYOMI_MAX_PERCENT,
                        available,
                    )
                } else {
                    // Main time phase
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

    /// Calculate budget based on time factor sum
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

    /// Compute final limits with clamping
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

    /// Starts the timer for a new search.
    pub fn start(&mut self) {
        self.start_time = Instant::now();
        self.extension_steps.store(0, Ordering::Relaxed);
        let current_maxi = self.max_time_ms.load(Ordering::Relaxed);
        self.base_max_time_ms.store(current_maxi, Ordering::Relaxed);
        *self.prev_score.lock().unwrap() = None;
        self.is_endgame_mode.store(false, Ordering::Relaxed);
    }

    /// Returns the elapsed time in milliseconds since search started.
    #[inline]
    pub fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Checks if the search has exceeded the maximum time limit.
    #[inline]
    pub fn is_time_up(&self) -> bool {
        if self.mode == TimeControlMode::Infinite {
            return false;
        }
        self.elapsed_ms() >= self.max_time_ms.load(Ordering::Relaxed)
    }

    /// Checks if we should continue to the next iteration.
    pub fn should_continue_iteration(&self) -> bool {
        if self.mode == TimeControlMode::Infinite {
            return true;
        }

        let elapsed = self.elapsed_ms();
        if elapsed < self.min_time_ms.load(Ordering::Relaxed) {
            return true;
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
    pub fn try_extend_time(&self, current_score: f32, pv_changed: bool, depth: Depth) -> bool {
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
            let mut r = "unknown";

            if let Some(p) = prev {
                if current_score < p - SCORE_DROP_THRESHOLD {
                    extend = true;
                    r = "score_drop";
                } else if pv_changed && depth >= MIN_STABILITY_CHECK_DEPTH {
                    extend = true;
                    r = "pv_change";
                }
            } else if pv_changed && depth >= MIN_STABILITY_CHECK_DEPTH {
                extend = true;
                r = "pv_change";
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
        reason: &str,
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
            let extension_amount = ((base_maxi as f64) * EXTENSION_RATIO) as u64;
            base_maxi.saturating_add(extension_amount).min(hard_limit)
        };

        if old_maxi >= target_maxi {
            return false;
        }

        let remaining_steps = (MAX_EXTENSION_STEPS - used_steps) as u64;
        let remaining_budget = target_maxi.saturating_sub(old_maxi);
        let step_increment = remaining_budget.div_ceil(remaining_steps);

        if step_increment == 0 {
            return false;
        }

        let new_maxi = old_maxi.saturating_add(step_increment).min(target_maxi);
        self.max_time_ms.store(new_maxi, Ordering::Relaxed);
        self.extension_steps.fetch_add(1, Ordering::Release);

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] Time extended ({reason}, step {}/{}): {:.2} -> {:.2}, old={}ms, new={}ms, limit={}ms",
                used_steps + 1,
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

    /// Checks if abort has been signaled.
    #[inline]
    pub fn is_aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }

    /// Checks time and signals abort if time is up.
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
    }

    /// Returns the current time control mode.
    pub fn mode(&self) -> TimeControlMode {
        self.mode
    }

    pub fn mini_time_ms(&self) -> u64 {
        self.min_time_ms.load(Ordering::Relaxed)
    }

    pub fn maxi_time_ms(&self) -> u64 {
        self.max_time_ms.load(Ordering::Relaxed)
    }

    pub fn deadline(&self) -> Option<Instant> {
        if self.mode == TimeControlMode::Infinite {
            None
        } else {
            Some(self.start_time + Duration::from_millis(self.max_time_ms.load(Ordering::Relaxed)))
        }
    }

    #[inline]
    pub fn remaining_time_ms(&self) -> u64 {
        self.max_time_ms
            .load(Ordering::Relaxed)
            .saturating_sub(self.elapsed_ms())
    }

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

fn is_debug_enabled() -> bool {
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| {
        let env_var = std::env::var("REVERSI_DEBUG_TIME").unwrap_or_default();
        env_var == "1" || env_var.to_lowercase() == "true"
    })
}
