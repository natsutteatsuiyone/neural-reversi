//! Time control management for timed games.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::probcut::Selectivity;
use crate::search::threading::AbortState;
use crate::square::Square;
use crate::types::Depth;

/// Safety buffer in milliseconds to avoid time forfeit.
const TIME_BUFFER_MS: u64 = 50;

/// Depth threshold after which PV/score instability becomes meaningful.
const MIN_STABILITY_CHECK_DEPTH: Depth = 10;

/// JpByo main time may always draw down to this fraction of the bank reserve;
/// other bank modes only in emergencies (falling >= FALLING_EMERGENCY).
const EXTENSION_RESERVE_DIVISOR: u64 = 4;

/// Decay applied to the best-move-change EMA each completed iteration.
const EMA_DECAY: f64 = 0.5;

/// Weight converting the best-move-change EMA into a time factor.
/// EMA is bounded by 1/(1-EMA_DECAY) = 2, so the factor stays within [1.0, 2.0].
const INSTABILITY_WEIGHT: f64 = 0.5;

/// Score drop (in discs) that doubles the falling-eval contribution.
const FALLING_DIVISOR: f64 = 8.0;

/// Lower clamp for the falling-eval factor (rising eval slightly shortens search).
const FALLING_MIN: f64 = 0.95;

/// Upper clamp for the falling-eval factor.
const FALLING_MAX: f64 = 1.6;

/// Falling factor at or above which bank modes may draw on the bank reserve.
const FALLING_EMERGENCY: f64 = 1.4;

/// Maximum multiple of the base allocation that scaling may reach (bank modes).
const MAX_FACTOR: u64 = 3;

/// Consecutive stable best-move iterations required for an easy-move verdict.
const EASY_STREAK_THRESHOLD: u32 = 3;

/// Stability scale applied when the easy-move conditions are met.
const EASY_SCALE: f64 = 0.5;

/// Fraction of the optimum after which an easy move may stop immediately.
const EASY_MIN_FRACTION: f64 = 0.3;

/// Stability scale by consecutive unchanged best-move iterations (saturating).
const STABILITY_SCALE: [f64; 5] = [1.0, 1.0, 0.85, 0.70, 0.60];

/// Sentinel bits marking "no previous iteration score recorded".
const NO_SCORE_BITS: u32 = f32::NEG_INFINITY.to_bits();

/// Sentinel value indicating no previous best move has been recorded.
const NO_PREV_MOVE: u8 = Square::None as u8;

/// Default next-iteration prediction factor for midgame iterative deepening.
const DEFAULT_CONTINUE_FACTOR: f64 = 1.5;

/// Endgame selectivity transition factors.
const ENDGAME_LEVEL1_CONTINUE_FACTOR: f64 = 2.582241;
const ENDGAME_LEVEL2_CONTINUE_FACTOR: f64 = 2.048957;
const ENDGAME_LEVEL3_CONTINUE_FACTOR: f64 = 4.592228;

// Time allocation percentages (0-100)
const MIN_PERCENT_NORMAL: u64 = 75;
const MIN_PERCENT_ENDGAME: u64 = 80;
const BYOYOMI_MAX_PERCENT: u64 = 98;
const JP_BYO_MAIN_MIN_PERCENT: u64 = 85;

/// Fraction of the hard limit allocated to a single endgame solve attempt (bank modes).
///
/// Once the endgame solver runs, all later moves are answered from the
/// transposition table at negligible cost, so most of the remaining bank can
/// be committed to finishing the solve now. Applying the fraction to the
/// current hard limit keeps the allocation geometric: an unsolved attempt
/// still leaves the majority of the bank for the next attempt.
const ENDGAME_BANK_PERCENT: u64 = 45;

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

/// Returns the budget for a single endgame solve attempt in bank modes.
fn endgame_bank_budget(hard_limit: u64) -> u64 {
    (hard_limit * ENDGAME_BANK_PERCENT) / 100
}

fn endgame_continue_factor(current_selectivity: Selectivity) -> f64 {
    match current_selectivity {
        Selectivity::Level1 => ENDGAME_LEVEL1_CONTINUE_FACTOR,
        Selectivity::Level2 => ENDGAME_LEVEL2_CONTINUE_FACTOR,
        Selectivity::Level3 => ENDGAME_LEVEL3_CONTINUE_FACTOR,
        Selectivity::None => f64::INFINITY,
        Selectivity::Mid => unreachable!("Mid is midgame-only and never enters the endgame ladder"),
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

/// Time allocation and tracking during a search.
#[derive(Debug)]
pub struct TimeManager {
    /// Time control mode for the current game.
    mode: TimeControlMode,

    /// Start time of the current search.
    start_time: Instant,

    /// Base allocation for this move before scaling factors (milliseconds).
    base_time_ms: AtomicU64,

    /// Current optimum time (milliseconds). Doubles as the abort deadline and
    /// is recomputed from the scaling factors after every completed iteration.
    optimum_time_ms: AtomicU64,

    /// Absolute hard limit for this move (remaining time - buffer).
    hard_time_limit_ms: AtomicU64,

    /// Reference to the abort state for signaling search termination.
    abort_state: Arc<AbortState>,

    /// Previous iteration's score (f32 bits; NO_SCORE_BITS when unset).
    prev_iter_score: AtomicU32,

    /// Number of empty squares at search start (for estimating remaining moves).
    n_empties: u32,

    /// Flag indicating if we are in endgame search mode.
    is_endgame_mode: AtomicBool,

    /// EMA of best-move changes across iterations (f64 bits).
    pv_instability: AtomicU64,

    /// Consecutive iterations where the best move has not changed.
    best_move_streak: AtomicU32,

    /// Best move from the previous iteration (raw u8, NO_PREV_MOVE if unset).
    prev_best_move: AtomicU8,

    /// Whether the last completed iteration finished inside its aspiration window.
    window_clean: AtomicBool,
}

impl TimeManager {
    /// Creates a new time manager with the specified mode and abort state.
    pub(crate) fn new(mode: TimeControlMode, abort_state: Arc<AbortState>, n_empties: u32) -> Self {
        let (base_ms, hard_ms) = Self::calculate_allocation(mode, n_empties, false);
        let optimum_ms = Self::initial_optimum(mode, base_ms, hard_ms);

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] New: mode={:?}, empties={}, base={}ms, optimum={}ms, hard_limit={}ms",
                mode, n_empties, base_ms, optimum_ms, hard_ms
            );
        }

        TimeManager {
            mode,
            start_time: Instant::now(),
            base_time_ms: AtomicU64::new(base_ms),
            optimum_time_ms: AtomicU64::new(optimum_ms),
            hard_time_limit_ms: AtomicU64::new(hard_ms),
            abort_state,
            prev_iter_score: AtomicU32::new(NO_SCORE_BITS),
            n_empties,
            is_endgame_mode: AtomicBool::new(false),
            pv_instability: AtomicU64::new(0f64.to_bits()),
            best_move_streak: AtomicU32::new(0),
            prev_best_move: AtomicU8::new(NO_PREV_MOVE),
            window_clean: AtomicBool::new(false),
        }
    }

    /// Calculates safe time limit based on time control mode.
    fn calculate_safe_time(main_time_ms: u64, n_empties: u32) -> u64 {
        let my_future_moves = n_empties.saturating_sub(1).div_ceil(2);
        let total_buffer = TIME_BUFFER_MS + ((my_future_moves as u64 * TIME_BUFFER_MS) / 2);

        main_time_ms.saturating_sub(total_buffer)
    }

    /// Calculates the base allocation and hard limit for the given mode.
    fn calculate_allocation(mode: TimeControlMode, n_empties: u32, is_endgame: bool) -> (u64, u64) {
        match mode {
            TimeControlMode::Infinite => (u64::MAX, u64::MAX),

            TimeControlMode::Byoyomi { time_per_move_ms }
            | TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms,
            } => {
                let available = time_per_move_ms.saturating_sub(TIME_BUFFER_MS);
                (available, available)
            }

            TimeControlMode::Fischer {
                main_time_ms,
                increment_ms,
            } => {
                let hard = Self::calculate_safe_time(main_time_ms, n_empties);
                let base = if is_endgame {
                    endgame_bank_budget(hard) + increment_ms
                } else {
                    Self::allocate_budget(main_time_ms, increment_ms, n_empties)
                };
                (base.min(hard), hard)
            }

            TimeControlMode::MovesToGo { time_ms, moves } => {
                let hard = time_ms.saturating_sub(TIME_BUFFER_MS);
                let moves = moves.max(1) as u64;
                let base = if is_endgame {
                    endgame_bank_budget(hard)
                } else {
                    time_ms / moves
                };
                (base.min(hard), hard)
            }

            TimeControlMode::JapaneseByo { main_time_ms, .. } => {
                let hard = Self::calculate_safe_time(main_time_ms, n_empties);
                let base = if is_endgame {
                    endgame_bank_budget(hard)
                } else {
                    Self::allocate_budget(main_time_ms, 0, n_empties)
                };
                (base.min(hard), hard)
            }
        }
    }

    /// Returns the initial optimum before any iteration feedback.
    fn initial_optimum(mode: TimeControlMode, base_ms: u64, hard_ms: u64) -> u64 {
        match mode {
            TimeControlMode::Infinite => u64::MAX,
            TimeControlMode::Byoyomi { .. }
            | TimeControlMode::JapaneseByo {
                main_time_ms: 0, ..
            } => base_ms * BYOYOMI_MAX_PERCENT / 100,
            _ => base_ms.min(hard_ms),
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

    /// Reports the iteration result: updates the scaling factors and recomputes
    /// the optimum time.
    ///
    /// `window_clean` must be true only when the iteration completed inside its
    /// initial aspiration window (no root re-search), which is evidence that the
    /// best score is stable and no alternative move overtook it.
    ///
    /// Must be called only from the search's main thread; other threads may
    /// only read.
    pub(crate) fn report_iteration(
        &self,
        sq: Square,
        current_score: f32,
        depth: Depth,
        window_clean: bool,
    ) {
        if self.mode == TimeControlMode::Infinite {
            return;
        }

        if depth >= MIN_STABILITY_CHECK_DEPTH {
            let prev_raw = self.prev_best_move.swap(sq as u8, Ordering::Relaxed);
            let pv_changed = prev_raw != NO_PREV_MOVE && prev_raw != sq as u8;

            let inst = f64::from_bits(self.pv_instability.load(Ordering::Relaxed));
            let new_inst = EMA_DECAY.mul_add(inst, if pv_changed { 1.0 } else { 0.0 });
            self.pv_instability
                .store(new_inst.to_bits(), Ordering::Relaxed);

            if pv_changed {
                self.best_move_streak.store(0, Ordering::Relaxed);
            } else if prev_raw != NO_PREV_MOVE {
                self.best_move_streak.fetch_add(1, Ordering::Relaxed);
            }
            self.window_clean.store(window_clean, Ordering::Relaxed);
        }

        let prev_iter_bits = self
            .prev_iter_score
            .swap(current_score.to_bits(), Ordering::Relaxed);
        let falling = if prev_iter_bits == NO_SCORE_BITS {
            1.0
        } else {
            let drop = f64::from(f32::from_bits(prev_iter_bits) - current_score);
            (drop / FALLING_DIVISOR + 1.0).clamp(FALLING_MIN, FALLING_MAX)
        };

        self.recompute_optimum(falling);

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] Iteration: sq={:?}, depth={}, score={:.2}, streak={}, \
                 inst={:.3}, falling={:.3}, easy={}, optimum={}ms",
                sq,
                depth,
                current_score,
                self.best_move_streak.load(Ordering::Relaxed),
                f64::from_bits(self.pv_instability.load(Ordering::Relaxed)),
                falling,
                self.is_easy_move(),
                self.maxi_time_ms(),
            );
        }
    }

    /// Returns the elapsed time in milliseconds since search started.
    #[inline]
    fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().as_millis() as u64
    }

    /// Checks whether the search has exceeded the current optimum time.
    #[inline]
    fn is_time_up(&self) -> bool {
        if self.mode == TimeControlMode::Infinite {
            return false;
        }
        self.elapsed_ms() >= self.maxi_time_ms()
    }

    /// Returns true when the continuous scaling model applies (bank modes and
    /// Japanese byoyomi main time).
    ///
    /// Pure byoyomi is excluded: unused time cannot be banked, so stopping
    /// early has no value.
    fn uses_continuous_scaling(&self) -> bool {
        match self.mode {
            TimeControlMode::Infinite | TimeControlMode::Byoyomi { .. } => false,
            TimeControlMode::JapaneseByo { main_time_ms, .. } => main_time_ms > 0,
            _ => true,
        }
    }

    /// Returns the minimum-time percentage for the current mode and phase.
    fn min_percent(&self) -> u64 {
        match self.mode {
            TimeControlMode::JapaneseByo { main_time_ms, .. } if main_time_ms > 0 => {
                JP_BYO_MAIN_MIN_PERCENT
            }
            _ if self.is_endgame_mode.load(Ordering::Relaxed) => MIN_PERCENT_ENDGAME,
            _ => MIN_PERCENT_NORMAL,
        }
    }

    /// Returns true when the easy-move conditions are met: a stable best move
    /// AND aspiration-window evidence that alternatives are clearly worse.
    fn is_easy_move(&self) -> bool {
        self.best_move_streak.load(Ordering::Relaxed) >= EASY_STREAK_THRESHOLD
            && self.window_clean.load(Ordering::Relaxed)
    }

    /// Stability scale: shrinks as the best move stays stable; smallest when
    /// the easy-move conditions are met.
    fn stability_scale(&self) -> f64 {
        if self.is_easy_move() {
            return EASY_SCALE;
        }
        let streak = self.best_move_streak.load(Ordering::Relaxed) as usize;
        STABILITY_SCALE[streak.min(STABILITY_SCALE.len() - 1)]
    }

    /// Upper cap for the scaled optimum, before the hard limit.
    fn mode_cap(&self, base: u64, hard: u64, falling: f64) -> u64 {
        let reserve_target =
            base.saturating_add(hard.saturating_sub(base) / EXTENSION_RESERVE_DIVISOR);

        match self.mode {
            // In Japanese byoyomi main time falling into byoyomi is acceptable,
            // so scaling may always draw on the bank reserve.
            TimeControlMode::JapaneseByo { main_time_ms, .. } if main_time_ms > 0 => reserve_target,
            _ => {
                let capped = base.saturating_mul(MAX_FACTOR);
                if falling >= FALLING_EMERGENCY {
                    // Emergencies may draw on the bank reserve beyond the per-move cap.
                    capped.max(reserve_target)
                } else {
                    capped
                }
            }
        }
    }

    /// Recomputes the optimum from the current scaling factors.
    fn recompute_optimum(&self, falling: f64) {
        if !self.uses_continuous_scaling() {
            return;
        }
        let base = self.base_time_ms.load(Ordering::Relaxed);
        let hard = self.hard_time_limit_ms.load(Ordering::Relaxed);
        let inst = f64::from_bits(self.pv_instability.load(Ordering::Relaxed));
        let instability_factor = INSTABILITY_WEIGHT.mul_add(inst, 1.0);

        let optimum = (base as f64 * self.stability_scale() * instability_factor * falling) as u64;
        let capped = optimum.min(self.mode_cap(base, hard, falling)).min(hard);
        self.optimum_time_ms.store(capped, Ordering::Relaxed);
    }

    /// Checks whether the search should continue to the next iteration.
    fn should_continue_iteration(&self) -> bool {
        if !self.uses_continuous_scaling() {
            return true;
        }

        let elapsed = self.elapsed_ms();
        let optimum = self.maxi_time_ms();

        // Easy moves may stop well before the min gate: the best move is stable
        // and provably ahead of every alternative.
        if self.is_easy_move() && elapsed as f64 >= optimum as f64 * EASY_MIN_FRACTION {
            if is_debug_enabled() {
                eprintln!(
                    "[TimeManager] Early stop (easy move): elapsed={}ms, optimum={}ms",
                    elapsed, optimum
                );
            }
            return false;
        }

        if elapsed < optimum * self.min_percent() / 100 {
            return true;
        }

        let should_continue = (elapsed as f64 * DEFAULT_CONTINUE_FACTOR) < optimum as f64;
        if !should_continue && is_debug_enabled() {
            eprintln!(
                "[TimeManager] Stopping iteration: elapsed={}ms, factor={DEFAULT_CONTINUE_FACTOR:.3}, optimum={}ms",
                elapsed, optimum
            );
        }

        should_continue
    }

    /// Checks whether endgame search should continue to the next selectivity level.
    fn should_continue_endgame_iteration(&self, current_selectivity: Selectivity) -> bool {
        if !self.uses_continuous_scaling() {
            return true;
        }

        let elapsed = self.elapsed_ms();
        let maxi = self.maxi_time_ms();
        let factor = endgame_continue_factor(current_selectivity);
        let should_continue = (elapsed as f64 * factor) < maxi as f64;
        if !should_continue && is_debug_enabled() {
            eprintln!(
                "[TimeManager] Stopping endgame selectivity: selectivity={current_selectivity:?}, elapsed={}ms, factor={factor:.3}, maxi={}ms",
                elapsed, maxi
            );
        }

        should_continue
    }

    /// Checks whether time is up and signals abort if so.
    #[inline]
    pub(crate) fn check_time(&self) -> bool {
        if !self.is_time_up() {
            return false;
        }
        if !self.abort_state.is_aborted() {
            if is_debug_enabled() {
                eprintln!(
                    "[TimeManager] Time up! elapsed={}ms, optimum={}ms",
                    self.elapsed_ms(),
                    self.maxi_time_ms()
                );
            }
            self.abort_state.request_abort();
        }
        true
    }

    /// Returns the minimum time in milliseconds.
    #[cfg(test)]
    fn mini_time_ms(&self) -> u64 {
        if self.mode == TimeControlMode::Infinite {
            return u64::MAX;
        }
        self.maxi_time_ms() * self.min_percent() / 100
    }

    /// Returns the maximum (optimum) time in milliseconds.
    fn maxi_time_ms(&self) -> u64 {
        self.optimum_time_ms.load(Ordering::Relaxed)
    }

    /// Returns the deadline instant, or [`None`] for infinite mode.
    pub(crate) fn deadline(&self) -> Option<Instant> {
        if self.mode == TimeControlMode::Infinite {
            None
        } else {
            Some(self.start_time + Duration::from_millis(self.maxi_time_ms()))
        }
    }

    /// Sets whether the search is in endgame mode.
    ///
    /// Recalculates the allocation and resets iteration statistics: midgame
    /// stability is not evidence about the endgame solve. A call that does not
    /// change the mode is a no-op.
    ///
    /// Must be called only from the search's main thread; other threads may
    /// only read.
    pub(crate) fn set_endgame_mode(&self, enabled: bool) {
        if self.is_endgame_mode.swap(enabled, Ordering::Relaxed) == enabled {
            return;
        }
        let (base_ms, hard_ms) = Self::calculate_allocation(self.mode, self.n_empties, enabled);
        let optimum_ms = Self::initial_optimum(self.mode, base_ms, hard_ms);
        self.base_time_ms.store(base_ms, Ordering::Relaxed);
        self.hard_time_limit_ms.store(hard_ms, Ordering::Relaxed);
        self.optimum_time_ms.store(optimum_ms, Ordering::Relaxed);
        self.reset_iteration_stats();

        if is_debug_enabled() {
            eprintln!(
                "[TimeManager] Endgame mode set to {}: base={}ms, optimum={}ms",
                enabled, base_ms, optimum_ms
            );
        }
    }

    /// Resets all per-iteration scaling state.
    fn reset_iteration_stats(&self) {
        self.pv_instability.store(0f64.to_bits(), Ordering::Relaxed);
        self.best_move_streak.store(0, Ordering::Relaxed);
        self.prev_best_move.store(NO_PREV_MOVE, Ordering::Relaxed);
        self.window_clean.store(false, Ordering::Relaxed);
        self.prev_iter_score.store(NO_SCORE_BITS, Ordering::Relaxed);
    }
}

/// Determines whether to stop the current search iteration based on time control.
///
/// Returns `true` if time is up or the iteration should not continue.
/// Returns `false` if no time manager is provided (unlimited search) or there
/// is still time remaining.
#[inline]
pub(crate) fn should_stop_iteration(time_manager: &Option<Arc<TimeManager>>) -> bool {
    time_manager
        .as_ref()
        .is_some_and(|tm| tm.check_time() || !tm.should_continue_iteration())
}

#[inline]
pub(crate) fn should_stop_endgame_iteration(
    time_manager: &Option<Arc<TimeManager>>,
    current_selectivity: Selectivity,
) -> bool {
    time_manager.as_ref().is_some_and(|tm| {
        tm.check_time() || !tm.should_continue_endgame_iteration(current_selectivity)
    })
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

    fn make_tm(mode: TimeControlMode, n_empties: u32) -> TimeManager {
        TimeManager::new(mode, Arc::new(AbortState::new()), n_empties)
    }

    fn make_fischer_tm(main_time_ms: u64, increment_ms: u64, n_empties: u32) -> TimeManager {
        make_tm(
            TimeControlMode::Fischer {
                main_time_ms,
                increment_ms,
            },
            n_empties,
        )
    }

    fn make_moves_to_go_tm(time_ms: u64, moves: u32, n_empties: u32) -> TimeManager {
        make_tm(TimeControlMode::MovesToGo { time_ms, moves }, n_empties)
    }

    fn make_japanese_byo_tm(
        main_time_ms: u64,
        time_per_move_ms: u64,
        n_empties: u32,
    ) -> TimeManager {
        make_tm(
            TimeControlMode::JapaneseByo {
                main_time_ms,
                time_per_move_ms,
            },
            n_empties,
        )
    }

    fn time_limits(tm: &TimeManager) -> (u64, u64, u64) {
        (
            tm.mini_time_ms(),
            tm.maxi_time_ms(),
            tm.hard_time_limit_ms.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn moves_to_go_budget_and_hard_limit() {
        let tm = make_moves_to_go_tm(60_000, 30, 40);
        let hard = tm.hard_time_limit_ms.load(Ordering::Relaxed);

        assert_eq!(hard, 60_000 - TIME_BUFFER_MS);
        assert!(tm.mini_time_ms() <= tm.maxi_time_ms());
        assert!(tm.maxi_time_ms() <= hard);
        assert!(tm.uses_continuous_scaling());
    }

    #[test]
    fn moves_to_go_zero_moves_is_treated_as_one() {
        let zero_moves = make_moves_to_go_tm(10_000, 0, 40);
        let one_move = make_moves_to_go_tm(10_000, 1, 40);

        assert_eq!(time_limits(&zero_moves), time_limits(&one_move));
    }

    #[test]
    fn japanese_byo_without_main_time_matches_pure_byoyomi() {
        let tm = make_japanese_byo_tm(0, 10_000, 40);
        let byoyomi = make_tm(
            TimeControlMode::Byoyomi {
                time_per_move_ms: 10_000,
            },
            40,
        );

        assert_eq!(time_limits(&tm), time_limits(&byoyomi));
        assert!(!tm.uses_continuous_scaling());
    }

    #[test]
    fn japanese_byo_main_time_caps_at_safe_time() {
        let tm = make_japanese_byo_tm(60_000, 5_000, 40);
        let hard = tm.hard_time_limit_ms.load(Ordering::Relaxed);
        let expected_hard = {
            let my_future_moves = 40u32.saturating_sub(1).div_ceil(2) as u64;
            60_000 - (TIME_BUFFER_MS + my_future_moves * TIME_BUFFER_MS / 2)
        };

        assert_eq!(hard, expected_hard);
        assert!(tm.mini_time_ms() <= tm.maxi_time_ms());
        assert!(tm.maxi_time_ms() <= hard);
        assert!(tm.uses_continuous_scaling());
    }

    #[test]
    fn endgame_continue_factor_uses_measured_p95_values() {
        assert_eq!(
            endgame_continue_factor(Selectivity::Level1),
            ENDGAME_LEVEL1_CONTINUE_FACTOR
        );
        assert_eq!(
            endgame_continue_factor(Selectivity::Level2),
            ENDGAME_LEVEL2_CONTINUE_FACTOR
        );
        assert_eq!(
            endgame_continue_factor(Selectivity::Level3),
            ENDGAME_LEVEL3_CONTINUE_FACTOR
        );
        assert!(endgame_continue_factor(Selectivity::None).is_infinite());
    }

    #[test]
    fn endgame_iteration_can_stop_before_min_time_when_factor_predicts_overrun() {
        let mut tm = make_fischer_tm(60_000, 0, 36);
        tm.set_endgame_mode(true);

        let elapsed = (tm.maxi_time_ms() as f64 / ENDGAME_LEVEL1_CONTINUE_FACTOR).ceil() as u64 + 1;
        assert!(elapsed < tm.mini_time_ms());

        tm.start_time = Instant::now() - Duration::from_millis(elapsed);

        assert!(tm.should_continue_iteration());
        assert!(!tm.should_continue_endgame_iteration(Selectivity::Level1));
    }

    #[test]
    fn stability_ignores_shallow_iterations() {
        let tm = make_fischer_tm(60_000, 0, 40);
        let sq = Square::D3;

        tm.report_iteration(sq, 5.0, 8, false);
        assert_eq!(tm.prev_best_move.load(Ordering::Relaxed), NO_PREV_MOVE);
        assert_eq!(tm.best_move_streak.load(Ordering::Relaxed), 0);

        tm.report_iteration(sq, 5.0, 10, false);
        assert_eq!(tm.best_move_streak.load(Ordering::Relaxed), 0);

        tm.report_iteration(sq, 5.0, 11, false);
        assert_eq!(tm.best_move_streak.load(Ordering::Relaxed), 1);

        tm.report_iteration(sq, 5.0, 12, false);
        assert_eq!(tm.best_move_streak.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn stability_resets_on_pv_change() {
        let tm = make_fischer_tm(60_000, 0, 40);
        let sq = Square::D3;

        // Build up stability
        tm.report_iteration(sq, 5.0, 10, false);
        tm.report_iteration(sq, 5.0, 11, false);
        tm.report_iteration(sq, 5.0, 12, false);
        assert_eq!(tm.best_move_streak.load(Ordering::Relaxed), 2);

        // PV change resets stability
        tm.report_iteration(Square::C4, 5.0, 13, false);
        assert_eq!(tm.best_move_streak.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn endgame_mode_allocates_bank_fraction() {
        let tm = make_fischer_tm(60_000, 0, 36);
        let mid_maxi = tm.maxi_time_ms();
        tm.set_endgame_mode(true);
        let end_maxi = tm.maxi_time_ms();
        let hard = tm.hard_time_limit_ms.load(Ordering::Relaxed);

        assert!(
            end_maxi > mid_maxi,
            "endgame allocation ({end_maxi}) should exceed midgame share ({mid_maxi})"
        );
        assert!(end_maxi <= hard);
    }

    #[test]
    fn pure_byoyomi_continues_until_deadline() {
        let mut tm = make_tm(
            TimeControlMode::Byoyomi {
                time_per_move_ms: 1_000,
            },
            40,
        );
        // Way past any predictive-stop threshold, but before the deadline.
        tm.start_time = Instant::now() - Duration::from_millis(tm.maxi_time_ms() - 100);

        assert!(tm.should_continue_iteration());
        assert!(!tm.is_time_up());
    }

    #[test]
    fn instability_ema_rises_on_pv_changes_and_decays_when_stable() {
        let tm = make_fischer_tm(120_000, 0, 40);
        let base = tm.maxi_time_ms();

        // Alternating best moves push the EMA (and the optimum) up.
        tm.report_iteration(Square::D3, 0.0, 10, false);
        tm.report_iteration(Square::C4, 0.0, 11, false);
        tm.report_iteration(Square::D3, 0.0, 12, false);
        let unstable_optimum = tm.maxi_time_ms();
        assert!(unstable_optimum > base);

        // A long stable run decays the EMA; stability scale shrinks further.
        for depth in 13..20 {
            tm.report_iteration(Square::D3, 0.0, depth, false);
        }
        assert!(tm.maxi_time_ms() < unstable_optimum);
    }

    #[test]
    fn optimum_capped_at_max_factor_times_base() {
        let tm = make_moves_to_go_tm(60_000, 6, 40);
        let base = tm.base_time_ms.load(Ordering::Relaxed);
        let hard = tm.hard_time_limit_ms.load(Ordering::Relaxed);
        let cap = base * MAX_FACTOR;

        // Preconditions: the emergency cap max(base*MAX_FACTOR, reserve)
        // resolves to base*MAX_FACTOR and sits below the hard limit, so the
        // per-move cap is the binding bound.
        let reserve = base + (hard - base) / EXTENSION_RESERVE_DIVISOR;
        assert!(reserve < cap && cap < hard);

        // Alternate best moves and drop the score by a full FALLING_DIVISOR
        // each iteration: falling clamps at FALLING_MAX (an emergency) and the
        // instability EMA pushes the unclamped product past the cap.
        let squares = [Square::D3, Square::C4];
        for (i, depth) in (10..25).enumerate() {
            tm.report_iteration(squares[i % 2], 60.0 - 8.0 * i as f32, depth, false);
        }

        assert_eq!(tm.maxi_time_ms(), cap);
    }

    #[test]
    fn falling_eval_increases_optimum() {
        let tm = make_fischer_tm(120_000, 0, 40);
        let base = tm.maxi_time_ms();

        // First iteration: no reference score yet, optimum stays at base.
        tm.report_iteration(Square::D3, 5.0, 10, false);
        assert_eq!(tm.maxi_time_ms(), base);

        // Second iteration drops 4 discs -> falling factor kicks in.
        tm.report_iteration(Square::D3, 1.0, 11, false);
        let expected = (base as f64 * (1.0 + 4.0 / FALLING_DIVISOR)) as u64;
        assert_eq!(tm.maxi_time_ms(), expected);
    }

    #[test]
    fn rising_eval_slightly_reduces_optimum() {
        let tm = make_fischer_tm(120_000, 0, 40);
        let base = tm.maxi_time_ms();

        tm.report_iteration(Square::D3, 1.0, 10, false);
        tm.report_iteration(Square::D3, 6.0, 11, false); // score jumps up

        let expected = (base as f64 * FALLING_MIN) as u64;
        assert_eq!(tm.maxi_time_ms(), expected);
    }

    #[test]
    fn easy_move_requires_both_streak_and_window_evidence() {
        // Streak without window evidence: not easy.
        let tm = make_fischer_tm(60_000, 0, 40);
        for depth in 10..15 {
            tm.report_iteration(Square::D3, 0.0, depth, false);
        }
        assert!(!tm.is_easy_move());

        // Window evidence without streak: not easy.
        let tm2 = make_fischer_tm(60_000, 0, 40);
        tm2.report_iteration(Square::D3, 0.0, 10, true);
        tm2.report_iteration(Square::C4, 0.0, 11, true); // PV change resets streak
        assert!(!tm2.is_easy_move());

        // Both: easy.
        let tm3 = make_fischer_tm(60_000, 0, 40);
        for depth in 10..15 {
            tm3.report_iteration(Square::D3, 0.0, depth, true);
        }
        assert!(tm3.is_easy_move());
    }

    #[test]
    fn dirty_window_clears_easy_verdict() {
        let tm = make_fischer_tm(60_000, 0, 40);
        for depth in 10..15 {
            tm.report_iteration(Square::D3, 0.0, depth, true);
        }
        assert!(tm.is_easy_move());

        tm.report_iteration(Square::D3, 0.0, 15, false); // re-search happened
        assert!(!tm.is_easy_move());
    }

    #[test]
    fn easy_move_stops_before_min_gate() {
        let mut tm = make_fischer_tm(60_000, 0, 40);
        for depth in 10..15 {
            tm.report_iteration(Square::D3, 0.0, depth, true);
        }
        assert!(tm.is_easy_move());

        let optimum = tm.maxi_time_ms();
        let elapsed = (optimum as f64 * (EASY_MIN_FRACTION + 0.1)) as u64;
        // Below the normal min gate (75% of optimum) but past the easy fraction.
        assert!(elapsed < optimum * MIN_PERCENT_NORMAL / 100);
        tm.start_time = Instant::now() - Duration::from_millis(elapsed);

        assert!(!tm.should_continue_iteration());
    }

    #[test]
    fn non_easy_move_continues_at_same_elapsed() {
        let mut tm = make_fischer_tm(60_000, 0, 40);
        for depth in 10..15 {
            tm.report_iteration(Square::D3, 0.0, depth, false); // no window evidence
        }
        assert!(!tm.is_easy_move());

        let optimum = tm.maxi_time_ms();
        let elapsed = (optimum as f64 * (EASY_MIN_FRACTION + 0.1)) as u64;
        tm.start_time = Instant::now() - Duration::from_millis(elapsed);

        assert!(tm.should_continue_iteration());
    }

    #[test]
    fn japanese_byo_main_cap_is_reserve_target() {
        let tm = make_japanese_byo_tm(60_000, 5_000, 40);
        let base = tm.base_time_ms.load(Ordering::Relaxed);
        let hard = tm.hard_time_limit_ms.load(Ordering::Relaxed);
        let reserve = base + (hard - base) / EXTENSION_RESERVE_DIVISOR;

        let squares = [Square::D3, Square::C4];
        for (i, depth) in (10..25).enumerate() {
            tm.report_iteration(squares[i % 2], -(i as f32), depth, false);
        }

        assert!(tm.maxi_time_ms() <= reserve);
    }

    #[test]
    fn endgame_mode_resets_iteration_stats() {
        let tm = make_fischer_tm(60_000, 0, 36);
        for depth in 10..15 {
            tm.report_iteration(Square::D3, 0.0, depth, true);
        }
        assert!(tm.is_easy_move());

        tm.set_endgame_mode(true);
        assert!(!tm.is_easy_move());
        assert_eq!(tm.best_move_streak.load(Ordering::Relaxed), 0);
    }
}
