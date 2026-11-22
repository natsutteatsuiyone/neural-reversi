//! Time control management for timed games.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use crate::probcut::{ENDGAME_STEPS, Selectivity};
use crate::search::threading::AbortState;
use crate::square::Square;
use crate::types::Depth;

mod endgame_budget;

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
const EASY_SCALE: f64 = 0.75;

/// Fraction of the optimum after which an easy move may stop immediately.
const EASY_MIN_FRACTION: f64 = 0.65;

/// Stability scale by consecutive unchanged best-move iterations (saturating).
const STABILITY_SCALE: [f64; 5] = [1.0, 1.0, 0.90, 0.85, 0.80];

/// Sentinel bits marking "no previous iteration score recorded".
const NO_SCORE_BITS: u32 = f32::NEG_INFINITY.to_bits();

/// Sentinel value indicating no previous best move has been recorded.
const NO_PREV_MOVE: u8 = u8::MAX;

/// Default next-iteration prediction factor for midgame iterative deepening.
const DEFAULT_CONTINUE_FACTOR: f64 = 1.5;

/// Midgame-speed nodes the endgame solve needs to finish Level2
/// (`ENDGAME_TRACKING_MIN_SELECTIVITY`) at the base empty count.
const ENDGAME_ENTER_WORK: f64 = 1.04e9;
const ENDGAME_ENTER_WORK_GROWTH: f64 = 1.65;
const ENDGAME_ENTER_WORK_BASE_EMPTIES: i32 = 38;

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
const ENDGAME_BANK_PERCENT: u64 = 55;

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

/// Time control mode for a game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeControlMode {
    /// No time limit.
    #[default]
    Infinite,

    /// Exact time limit per move, with no safety buffer or scaling.
    /// `time_per_move_ms` is the search deadline in milliseconds.
    Fixed { time_per_move_ms: u64 },

    /// Fixed time per move.
    /// `time_per_move_ms` is the time allowed for each move in milliseconds.
    Byoyomi { time_per_move_ms: u64 },

    /// Fischer time control (increment per move).
    /// `main_time_ms` is the remaining time bank.
    /// `increment_ms` is the time added after each move.
    Fischer {
        main_time_ms: u64,
        increment_ms: u64,
    },

    /// Japanese byoyomi.
    /// `main_time_ms` is the remaining time bank (0 once in byoyomi).
    /// `time_per_move_ms` is the fixed time per move after main time expires.
    JapaneseByo {
        main_time_ms: u64,
        time_per_move_ms: u64,
    },
}

/// Match-level time control configuration parsed from GTP `time_settings`.
///
/// Unlike [`TimeControlMode`], [`Self::JapaneseByo`] retains the full overtime
/// period budget and stone count so callers can track multi-stone periods and
/// derive per-move budgets from `time_left` updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeSettings {
    /// No time limit.
    #[default]
    Infinite,

    /// Fixed time per move (`time_settings 0 N 0`).
    Byoyomi {
        /// Time allowed per move in milliseconds.
        time_per_move_ms: u64,
    },

    /// Main time bank plus per-move increment (`time_settings M N 0` or sudden death `M 0 S`).
    Fischer {
        /// Initial main time bank in milliseconds.
        main_time_ms: u64,
        /// Time added after each move in milliseconds (`0` for sudden death).
        increment_ms: u64,
    },

    /// Japanese or Canadian overtime (`time_settings M N S` with `N > 0` and `S > 0`).
    JapaneseByo {
        /// Initial main time bank in milliseconds (`0` starts directly in overtime).
        main_time_ms: u64,
        /// Time budget for each overtime period in milliseconds.
        byoyomi_time_ms: u64,
        /// Number of moves required per overtime period.
        byoyomi_stones: NonZeroU32,
    },
}

impl TimeSettings {
    /// Classifies GTP `time_settings` parameters (in seconds and stones) into a [`TimeSettings`].
    ///
    /// Stone counts outside Japanese overtime (`byoyomi_time_secs == 0`) carry no
    /// meaning in GTP and are discarded.
    #[must_use]
    pub const fn from_gtp_secs(
        main_time_secs: u64,
        byoyomi_time_secs: u64,
        byoyomi_stones: u32,
    ) -> Self {
        let main_time_ms = main_time_secs.saturating_mul(1000);
        let byoyomi_time_ms = byoyomi_time_secs.saturating_mul(1000);
        match (
            main_time_ms,
            byoyomi_time_ms,
            NonZeroU32::new(byoyomi_stones),
        ) {
            (0, 0, _) => Self::Infinite,
            (0, byoyomi_time_ms, None) => Self::Byoyomi {
                time_per_move_ms: byoyomi_time_ms,
            },
            (main_time_ms, byoyomi_time_ms, Some(byoyomi_stones)) if byoyomi_time_ms > 0 => {
                Self::JapaneseByo {
                    main_time_ms,
                    byoyomi_time_ms,
                    byoyomi_stones,
                }
            }
            (main_time_ms, increment_ms, _) => Self::Fischer {
                main_time_ms,
                increment_ms,
            },
        }
    }

    /// Returns whether a time limit is active (anything other than [`Self::Infinite`]).
    #[inline]
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        !matches!(self, Self::Infinite)
    }

    /// Returns the main time setting in whole seconds for GTP `time_settings`.
    #[inline]
    #[must_use]
    pub const fn main_time_secs(self) -> u64 {
        match self {
            Self::Infinite | Self::Byoyomi { .. } => 0,
            Self::Fischer { main_time_ms, .. } | Self::JapaneseByo { main_time_ms, .. } => {
                main_time_ms / 1000
            }
        }
    }

    /// Returns the overtime or increment setting in whole seconds for GTP `time_settings`.
    #[inline]
    #[must_use]
    pub const fn byoyomi_time_secs(self) -> u64 {
        match self {
            Self::Infinite => 0,
            Self::Byoyomi { time_per_move_ms } => time_per_move_ms / 1000,
            Self::Fischer { increment_ms, .. } => increment_ms / 1000,
            Self::JapaneseByo {
                byoyomi_time_ms, ..
            } => byoyomi_time_ms / 1000,
        }
    }

    /// Returns the overtime stone count for GTP `time_settings` (`0` outside [`Self::JapaneseByo`]).
    #[inline]
    #[must_use]
    pub const fn byoyomi_stones(self) -> u32 {
        match self {
            Self::JapaneseByo { byoyomi_stones, .. } => byoyomi_stones.get(),
            Self::Infinite | Self::Byoyomi { .. } | Self::Fischer { .. } => 0,
        }
    }

    /// Returns the initial [`TimeControlMode`] before any `time_left` update is received.
    #[must_use]
    pub const fn initial_time_control(self) -> TimeControlMode {
        match self {
            Self::Infinite => TimeControlMode::Infinite,
            Self::Byoyomi { time_per_move_ms } => TimeControlMode::Byoyomi { time_per_move_ms },
            Self::Fischer {
                main_time_ms,
                increment_ms,
            } => TimeControlMode::Fischer {
                main_time_ms,
                increment_ms,
            },
            Self::JapaneseByo {
                main_time_ms,
                byoyomi_time_ms,
                byoyomi_stones,
            } => TimeControlMode::JapaneseByo {
                main_time_ms,
                time_per_move_ms: byoyomi_time_ms / byoyomi_stones.get() as u64,
            },
        }
    }

    /// Returns the single-move [`TimeControlMode`] for a player given their latest `time_left` state.
    ///
    /// In [`Self::JapaneseByo`], `byo_stones_left > 0` indicates an active overtime
    /// period and divides `remaining_time_ms` evenly across the remaining stones;
    /// `byo_stones_left == 0` indicates main time.
    #[must_use]
    pub const fn move_time_control(
        self,
        remaining_time_ms: u64,
        byo_stones_left: u32,
    ) -> TimeControlMode {
        match self {
            Self::Infinite => TimeControlMode::Infinite,
            Self::Byoyomi { time_per_move_ms } => TimeControlMode::Byoyomi { time_per_move_ms },
            Self::Fischer { increment_ms, .. } => TimeControlMode::Fischer {
                main_time_ms: remaining_time_ms,
                increment_ms,
            },
            Self::JapaneseByo {
                byoyomi_time_ms,
                byoyomi_stones,
                ..
            } => match NonZeroU32::new(byo_stones_left) {
                Some(stones_left) => TimeControlMode::JapaneseByo {
                    main_time_ms: 0,
                    time_per_move_ms: remaining_time_ms / stones_left.get() as u64,
                },
                None => TimeControlMode::JapaneseByo {
                    main_time_ms: remaining_time_ms,
                    time_per_move_ms: byoyomi_time_ms / byoyomi_stones.get() as u64,
                },
            },
        }
    }
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

    /// Absolute hard limit for this move (remaining time minus safety buffer).
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

    /// Whether running out of time aborted an iteration in progress.
    timed_out: AtomicBool,

    /// Whether a timeout check requested an abort because the deadline passed.
    deadline_reached: AtomicBool,
}

struct Allocation {
    base_ms: u64,
    optimum_ms: u64,
    hard_ms: u64,
}

impl TimeManager {
    /// Creates a new time manager with the specified mode and abort state.
    pub(crate) fn new(mode: TimeControlMode, abort_state: Arc<AbortState>, n_empties: u32) -> Self {
        let Allocation {
            base_ms,
            optimum_ms,
            hard_ms,
        } = Self::calculate_allocation(mode, n_empties, false);

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
            timed_out: AtomicBool::new(false),
            deadline_reached: AtomicBool::new(false),
        }
    }

    /// Calculates safe time limit based on time control mode.
    fn calculate_safe_time(main_time_ms: u64, n_empties: u32) -> u64 {
        let my_future_moves = n_empties.saturating_sub(1).div_ceil(2);
        let total_buffer = TIME_BUFFER_MS + ((my_future_moves as u64 * TIME_BUFFER_MS) / 2);

        main_time_ms.saturating_sub(total_buffer)
    }

    fn calculate_allocation(mode: TimeControlMode, n_empties: u32, is_endgame: bool) -> Allocation {
        match mode {
            TimeControlMode::Infinite => Allocation {
                base_ms: u64::MAX,
                optimum_ms: u64::MAX,
                hard_ms: u64::MAX,
            },

            TimeControlMode::Fixed { time_per_move_ms } => Allocation {
                base_ms: time_per_move_ms,
                optimum_ms: time_per_move_ms,
                hard_ms: time_per_move_ms,
            },

            TimeControlMode::Byoyomi { time_per_move_ms }
            | TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms,
            } => {
                let available = time_per_move_ms.saturating_sub(TIME_BUFFER_MS);
                Allocation {
                    base_ms: available,
                    optimum_ms: available * BYOYOMI_MAX_PERCENT / 100,
                    hard_ms: available,
                }
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
                let base = base.min(hard);
                Allocation {
                    base_ms: base,
                    optimum_ms: base,
                    hard_ms: hard,
                }
            }

            // Later moves are covered by byoyomi, so this move may spend the
            // whole bank plus one byoyomi period, and never plans less than
            // the period alone would give it.
            TimeControlMode::JapaneseByo {
                main_time_ms,
                time_per_move_ms,
            } => {
                let byoyomi = time_per_move_ms.saturating_sub(TIME_BUFFER_MS);
                let hard = main_time_ms.saturating_add(byoyomi);
                let base = if is_endgame {
                    endgame_bank_budget(hard)
                } else {
                    Self::allocate_budget(main_time_ms, 0, n_empties)
                };
                let base = base.max(byoyomi).min(hard);
                Allocation {
                    base_ms: base,
                    optimum_ms: base,
                    hard_ms: hard,
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

    /// Reports the iteration result: updates the scaling factors and recomputes
    /// the optimum time.
    ///
    /// `window_clean` must be true only when the iteration completed inside its
    /// initial aspiration window (no root re-search), which is evidence that the
    /// best score is stable.
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
    /// Pure byoyomi and fixed-time modes are excluded: unused time cannot be
    /// banked, so stopping early has no value.
    fn uses_continuous_scaling(&self) -> bool {
        match self.mode {
            TimeControlMode::Infinite
            | TimeControlMode::Fixed { .. }
            | TimeControlMode::Byoyomi { .. } => false,
            TimeControlMode::JapaneseByo { main_time_ms, .. } => main_time_ms > 0,
            TimeControlMode::Fischer { .. } => true,
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
    /// AND an iteration that stayed inside its initial aspiration window.
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
        // and the iteration stayed inside its initial aspiration window.
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

    pub(crate) fn should_enter_endgame(&self, nodes: u64) -> bool {
        if !self.uses_continuous_scaling() || nodes == 0 {
            return true;
        }

        let elapsed = self.elapsed_ms() as f64;
        let budget = Self::calculate_allocation(self.mode, self.n_empties, true).optimum_ms;
        let work = ENDGAME_ENTER_WORK
            * ENDGAME_ENTER_WORK_GROWTH
                .powi(self.n_empties as i32 - ENDGAME_ENTER_WORK_BASE_EMPTIES);
        let expected = elapsed + work * elapsed / nodes as f64;
        let should_enter = expected < budget as f64;
        if !should_enter && is_debug_enabled() {
            eprintln!(
                "[TimeManager] Staying in midgame: elapsed={elapsed}ms, expected={expected:.0}ms, budget={budget}ms"
            );
        }

        should_enter
    }

    pub(super) fn endgame_clock(&self) -> EndgameClock<'_> {
        let prediction = self.uses_continuous_scaling().then(|| EndgamePrediction {
            budget_ms: self.maxi_time_ms(),
            factors: endgame_budget::factors(self.n_empties),
        });
        EndgameClock {
            timer: self,
            prediction,
        }
    }

    /// Checks whether time is up and signals abort if so.
    #[inline]
    pub(crate) fn check_time(&self) -> bool {
        self.signal_time_up(|| self.abort_state.request_abort())
    }

    /// Checks whether time is up and signals abort for `generation` if so.
    pub(crate) fn check_time_for(&self, generation: u32) -> bool {
        self.signal_time_up(|| self.abort_state.request_abort_for(generation))
    }

    fn signal_time_up(&self, request_abort: impl FnOnce()) -> bool {
        if !self.is_time_up() {
            return false;
        }
        if !self.abort_state.is_aborted() && is_debug_enabled() {
            eprintln!(
                "[TimeManager] Time up! elapsed={}ms, optimum={}ms",
                self.elapsed_ms(),
                self.maxi_time_ms()
            );
        }
        self.deadline_reached.store(true, Ordering::Relaxed);
        request_abort();
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

    /// Returns whether the search runs against a clock.
    pub(crate) fn has_deadline(&self) -> bool {
        self.mode != TimeControlMode::Infinite
    }

    /// Sets whether the search is in endgame mode.
    ///
    /// Recalculates the allocation and resets iteration statistics: midgame
    /// stability is not evidence about the endgame solve. A call that does not
    /// change the mode is a no-op.
    ///
    /// Must be called only before the search starts or from the search's main
    /// thread; other threads may only read.
    pub(crate) fn set_endgame_mode(&self, enabled: bool) {
        if self.is_endgame_mode.swap(enabled, Ordering::Relaxed) == enabled {
            return;
        }
        let Allocation {
            base_ms,
            optimum_ms,
            hard_ms,
        } = Self::calculate_allocation(self.mode, self.n_empties, enabled);
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

    /// Returns whether running out of time aborted an iteration in progress.
    pub(crate) fn timed_out(&self) -> bool {
        self.timed_out.load(Ordering::Relaxed)
    }
}

/// Records an aborted iteration as timed out when the deadline caused the
/// abort, rather than an external stop. Call it where the search drops an
/// iteration because of the abort.
pub(crate) fn record_abort(time_manager: Option<&TimeManager>) {
    if let Some(tm) = time_manager
        && tm.deadline_reached.load(Ordering::Relaxed)
    {
        tm.timed_out.store(true, Ordering::Relaxed);
    }
}

/// Determines whether to stop the current search iteration based on time control.
///
/// Returns `true` if time is up or the iteration should not continue.
/// Returns `false` if no time manager is provided (unlimited search) or there
/// is still time remaining.
#[inline]
pub(crate) fn should_stop_iteration(time_manager: Option<&TimeManager>) -> bool {
    time_manager.is_some_and(|tm| tm.check_time() || !tm.should_continue_iteration())
}

struct EndgamePrediction {
    budget_ms: u64,
    factors: [f64; ENDGAME_STEPS],
}

pub(super) struct EndgameClock<'a> {
    timer: &'a TimeManager,
    prediction: Option<EndgamePrediction>,
}

impl EndgameClock<'_> {
    pub(super) fn should_stop(&self, step: usize) -> bool {
        if self.timer.check_time() {
            return true;
        }
        let Some(prediction) = &self.prediction else {
            return false;
        };
        let elapsed = self.timer.elapsed_ms();
        let factor = prediction.factors[step];
        let stop = !endgame_budget::permits(elapsed, prediction.budget_ms, factor);
        if stop && is_debug_enabled() {
            eprintln!(
                "[TimeManager] Stopping endgame selectivity: selectivity={:?}, elapsed={}ms, factor={factor:.3}, maxi={}ms",
                Selectivity::ENDGAME_LADDER[step],
                elapsed,
                prediction.budget_ms
            );
        }
        stop
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
    use std::time::Duration;

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
    fn japanese_byo_low_main_time_still_spends_the_byoyomi_period() {
        for main_time_ms in [0, 300, 1_000] {
            for endgame in [false, true] {
                let tm = make_japanese_byo_tm(main_time_ms, 10_000, 40);
                tm.set_endgame_mode(endgame);
                let (_, maxi, hard) = time_limits(&tm);
                assert!(
                    maxi >= 9_000,
                    "main={main_time_ms} endgame={endgame}: deadline {maxi}ms wastes the 10s byoyomi"
                );
                assert!(
                    hard < main_time_ms + 10_000,
                    "main={main_time_ms} endgame={endgame}: hard limit {hard}ms overruns the clock"
                );
            }
        }
    }

    #[test]
    fn endgame_iteration_can_stop_before_min_time_when_factor_predicts_overrun() {
        let mut tm = make_fischer_tm(60_000, 0, 36);
        tm.set_endgame_mode(true);

        let factor = endgame_budget::factors(36)[0];
        let elapsed = (tm.maxi_time_ms() as f64 / factor).ceil() as u64 + 1;
        assert!(elapsed < tm.mini_time_ms());

        tm.start_time = Instant::now() - Duration::from_millis(elapsed);

        assert!(tm.should_continue_iteration());
        assert!(tm.endgame_clock().should_stop(0));
    }

    #[test]
    fn endgame_clock_keeps_its_prediction_budget_after_capture() {
        let mut tm = make_fischer_tm(60_000, 0, 36);
        tm.set_endgame_mode(true);
        tm.start_time = Instant::now() - Duration::from_millis(tm.maxi_time_ms());
        let clock = tm.endgame_clock();

        tm.optimum_time_ms.store(u64::MAX, Ordering::Relaxed);

        assert!(clock.should_stop(0));
        assert!(!tm.endgame_clock().should_stop(0));
    }

    #[test]
    fn endgame_entry_needs_room_for_level2() {
        let mut tm = make_fischer_tm(60_000, 0, 38);
        tm.start_time = Instant::now() - Duration::from_millis(2_000);

        assert!(tm.should_enter_endgame(280_000_000));
        assert!(!tm.should_enter_endgame(28_000_000));
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
    fn fixed_mode_uses_exact_budget_without_buffer_or_scaling() {
        let tm = make_tm(
            TimeControlMode::Fixed {
                time_per_move_ms: 25,
            },
            40,
        );
        assert_eq!(tm.base_time_ms.load(Ordering::Relaxed), 25);
        assert_eq!(tm.maxi_time_ms(), 25);
        assert_eq!(tm.hard_time_limit_ms.load(Ordering::Relaxed), 25);
        assert!(!tm.uses_continuous_scaling());
        assert!(tm.has_deadline());

        tm.set_endgame_mode(true);
        assert_eq!(tm.maxi_time_ms(), 25);
        assert_eq!(tm.hard_time_limit_ms.load(Ordering::Relaxed), 25);
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
        let tm = make_fischer_tm(60_000, 0, 40);
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
        let elapsed = (optimum as f64 * (EASY_MIN_FRACTION + 0.05)) as u64;
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
        let elapsed = (optimum as f64 * (EASY_MIN_FRACTION + 0.05)) as u64;
        tm.start_time = Instant::now() - Duration::from_millis(elapsed);

        assert!(tm.should_continue_iteration());
    }

    #[test]
    fn japanese_byo_main_cap_is_reserve_target() {
        let tm = make_japanese_byo_tm(60_000, 5_000, 4);
        let base = tm.base_time_ms.load(Ordering::Relaxed);
        let hard = tm.hard_time_limit_ms.load(Ordering::Relaxed);
        let reserve = base + (hard - base) / EXTENSION_RESERVE_DIVISOR;
        assert!(reserve < hard);

        let squares = [Square::D3, Square::C4];
        for (i, depth) in (10..25).enumerate() {
            tm.report_iteration(squares[i % 2], 60.0 - 8.0 * i as f32, depth, false);
        }

        assert_eq!(tm.maxi_time_ms(), reserve);
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

    #[test]
    fn deadline_abort_stays_timed_out_after_the_deadline_moves() {
        let mut tm = make_fischer_tm(60_000, 0, 36);
        let generation = tm.abort_state.begin_search();
        tm.start_time = Instant::now() - Duration::from_millis(tm.maxi_time_ms());
        assert!(tm.check_time_for(generation));

        tm.optimum_time_ms
            .store(tm.maxi_time_ms() * 4, Ordering::Relaxed);
        assert!(!tm.is_time_up());
        record_abort(Some(&tm));
        assert!(tm.timed_out());
    }

    #[test]
    fn time_settings_from_gtp_secs_classifies_all_eight_sign_combinations() {
        let two = NonZeroU32::new(2).unwrap();

        let untimed_zero = TimeSettings::from_gtp_secs(0, 0, 0);
        assert_eq!(untimed_zero, TimeSettings::Infinite);
        assert!(!untimed_zero.is_enabled());
        assert_eq!(
            (
                untimed_zero.main_time_secs(),
                untimed_zero.byoyomi_time_secs(),
                untimed_zero.byoyomi_stones()
            ),
            (0, 0, 0)
        );
        assert_eq!(
            untimed_zero.initial_time_control(),
            TimeControlMode::Infinite
        );

        let untimed_stones = TimeSettings::from_gtp_secs(0, 0, 3);
        assert_eq!(untimed_stones, TimeSettings::Infinite);
        assert!(!untimed_stones.is_enabled());
        assert_eq!(untimed_stones.byoyomi_stones(), 0);
        assert_eq!(
            untimed_stones.initial_time_control(),
            TimeControlMode::Infinite
        );

        let pure_byo = TimeSettings::from_gtp_secs(0, 5, 0);
        assert_eq!(
            pure_byo,
            TimeSettings::Byoyomi {
                time_per_move_ms: 5_000
            }
        );
        assert!(pure_byo.is_enabled());
        assert_eq!(
            (
                pure_byo.main_time_secs(),
                pure_byo.byoyomi_time_secs(),
                pure_byo.byoyomi_stones()
            ),
            (0, 5, 0)
        );
        assert_eq!(
            pure_byo.initial_time_control(),
            TimeControlMode::Byoyomi {
                time_per_move_ms: 5_000
            }
        );

        let mainless_jp_byo = TimeSettings::from_gtp_secs(0, 10, 2);
        assert_eq!(
            mainless_jp_byo,
            TimeSettings::JapaneseByo {
                main_time_ms: 0,
                byoyomi_time_ms: 10_000,
                byoyomi_stones: two,
            }
        );
        assert!(mainless_jp_byo.is_enabled());
        assert_eq!(
            (
                mainless_jp_byo.main_time_secs(),
                mainless_jp_byo.byoyomi_time_secs(),
                mainless_jp_byo.byoyomi_stones()
            ),
            (0, 10, 2)
        );
        assert_eq!(
            mainless_jp_byo.initial_time_control(),
            TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms: 5_000,
            }
        );

        let sudden_death = TimeSettings::from_gtp_secs(60, 0, 0);
        assert_eq!(
            sudden_death,
            TimeSettings::Fischer {
                main_time_ms: 60_000,
                increment_ms: 0,
            }
        );
        assert!(sudden_death.is_enabled());
        assert_eq!(
            (
                sudden_death.main_time_secs(),
                sudden_death.byoyomi_time_secs(),
                sudden_death.byoyomi_stones()
            ),
            (60, 0, 0)
        );
        assert_eq!(
            sudden_death.initial_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 60_000,
                increment_ms: 0,
            }
        );

        let sudden_death_stones = TimeSettings::from_gtp_secs(60, 0, 3);
        assert_eq!(
            sudden_death_stones,
            TimeSettings::Fischer {
                main_time_ms: 60_000,
                increment_ms: 0,
            }
        );
        assert_eq!(sudden_death_stones.byoyomi_stones(), 0);
        assert_eq!(
            sudden_death_stones.initial_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 60_000,
                increment_ms: 0,
            }
        );

        let fischer = TimeSettings::from_gtp_secs(60, 5, 0);
        assert_eq!(
            fischer,
            TimeSettings::Fischer {
                main_time_ms: 60_000,
                increment_ms: 5_000,
            }
        );
        assert_eq!(
            (
                fischer.main_time_secs(),
                fischer.byoyomi_time_secs(),
                fischer.byoyomi_stones()
            ),
            (60, 5, 0)
        );
        assert_eq!(
            fischer.initial_time_control(),
            TimeControlMode::Fischer {
                main_time_ms: 60_000,
                increment_ms: 5_000,
            }
        );

        let jp_byo = TimeSettings::from_gtp_secs(60, 10, 2);
        assert_eq!(
            jp_byo,
            TimeSettings::JapaneseByo {
                main_time_ms: 60_000,
                byoyomi_time_ms: 10_000,
                byoyomi_stones: two,
            }
        );
        assert_eq!(
            (
                jp_byo.main_time_secs(),
                jp_byo.byoyomi_time_secs(),
                jp_byo.byoyomi_stones()
            ),
            (60, 10, 2)
        );
        assert_eq!(
            jp_byo.initial_time_control(),
            TimeControlMode::JapaneseByo {
                main_time_ms: 60_000,
                time_per_move_ms: 5_000,
            }
        );
    }

    #[test]
    fn time_settings_from_gtp_secs_saturates_on_u64_max() {
        assert_eq!(
            TimeSettings::from_gtp_secs(u64::MAX, 0, 0),
            TimeSettings::Fischer {
                main_time_ms: u64::MAX,
                increment_ms: 0,
            }
        );
        assert_eq!(
            TimeSettings::from_gtp_secs(0, u64::MAX, 0),
            TimeSettings::Byoyomi {
                time_per_move_ms: u64::MAX,
            }
        );
        assert_eq!(
            TimeSettings::from_gtp_secs(u64::MAX, u64::MAX, 2),
            TimeSettings::JapaneseByo {
                main_time_ms: u64::MAX,
                byoyomi_time_ms: u64::MAX,
                byoyomi_stones: NonZeroU32::new(2).unwrap(),
            }
        );
    }

    #[test]
    fn time_settings_move_time_control_handles_main_time_and_overtime_periods() {
        assert_eq!(
            TimeSettings::Infinite.move_time_control(45_000, 3),
            TimeControlMode::Infinite
        );
        assert_eq!(
            TimeSettings::from_gtp_secs(0, 5, 0).move_time_control(99_000, 4),
            TimeControlMode::Byoyomi {
                time_per_move_ms: 5_000,
            }
        );
        assert_eq!(
            TimeSettings::from_gtp_secs(60, 5, 0).move_time_control(42_000, 4),
            TimeControlMode::Fischer {
                main_time_ms: 42_000,
                increment_ms: 5_000,
            }
        );

        let jp_byo = TimeSettings::from_gtp_secs(60, 10, 2);
        assert_eq!(
            jp_byo.move_time_control(25_000, 0),
            TimeControlMode::JapaneseByo {
                main_time_ms: 25_000,
                time_per_move_ms: 5_000,
            }
        );
        assert_eq!(
            jp_byo.move_time_control(8_000, 2),
            TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms: 4_000,
            }
        );
        assert_eq!(
            jp_byo.move_time_control(3_000, 1),
            TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms: 3_000,
            }
        );
    }
}
