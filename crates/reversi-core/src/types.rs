//! Common type aliases used throughout the engine.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::constants::{SCORE_INF, SCORE_MAX, SCORE_MIN};

/// Search depth.
pub type Depth = u32;

/// Score (disc difference: -64 to +64).
pub type Score = i32;

/// Floating-point score.
pub type Scoref = f32;

/// Scaled score for internal evaluation with sub-disc precision.
///
/// In Reversi, the final score is a disc difference ranging from -64 to +64.
/// However, during evaluation, the engine needs finer granularity to distinguish
/// between positions that would otherwise appear equal. `ScaledScore` achieves this
/// by scaling values by 256 (2^8), providing 8 bits of fractional precision.
///
/// # Value Representation
///
/// - **Raw value**: The internal [`i32`] representation (scaled)
/// - **Disc difference**: The human-readable score (raw_value >> 8)
///
/// # Special Values
///
/// - [`ScaledScore::MIN`] / [`ScaledScore::MAX`]: Bounds for actual game outcomes (-64/+64 discs)
/// - [`ScaledScore::INF`]: Sentinel value for search algorithm bounds (larger than any real score)
///
/// # Examples
///
/// ```
/// use reversi_core::types::ScaledScore;
///
/// // Create from a disc difference (scaled internally by 256)
/// let score = ScaledScore::from_disc_diff(10);
/// assert_eq!(score.value(), 2560);
/// assert_eq!(score.to_disc_diff(), 10);
///
/// // Create from a raw internal value
/// let raw = ScaledScore::from_raw(2560);
/// assert_eq!(raw.to_disc_diff(), 10);
///
/// // Arithmetic preserves scaling
/// let a = ScaledScore::from_disc_diff(10);
/// let b = ScaledScore::from_disc_diff(5);
/// assert_eq!((a + b).to_disc_diff(), 15);
/// assert_eq!((-a).to_disc_diff(), -10);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ScaledScore(i32);

impl ScaledScore {
    /// Number of bits used for scaling (8 bits = 256x).
    pub const SCALE_BITS: i32 = 8;

    /// Scale factor: 256.
    pub const SCALE: i32 = 1 << Self::SCALE_BITS;

    /// Zero score.
    pub const ZERO: Self = Self(0);

    /// Maximum achievable score: +64 discs.
    pub const MAX: Self = Self(SCORE_MAX << Self::SCALE_BITS);

    /// Minimum achievable score: -64 discs.
    pub const MIN: Self = Self(SCORE_MIN << Self::SCALE_BITS);

    /// Infinity sentinel for alpha-beta search bounds.
    pub const INF: Self = Self(i16::MAX as i32);

    /// Creates a [`ScaledScore`] from a raw internal value (already scaled by 256).
    #[inline(always)]
    pub const fn from_raw(raw_value: i32) -> Self {
        debug_assert!(raw_value >= -Self::INF.0 && raw_value <= Self::INF.0);
        Self(raw_value)
    }

    /// Creates a [`ScaledScore`] from a disc difference (-64 to +64).
    #[inline(always)]
    pub const fn from_disc_diff(disc_diff: Score) -> Self {
        debug_assert!(disc_diff >= -SCORE_INF && disc_diff <= SCORE_INF);
        Self(disc_diff << Self::SCALE_BITS)
    }

    /// Returns the raw internal value (scaled by 256).
    #[inline(always)]
    pub const fn value(self) -> i32 {
        self.0
    }

    /// Converts to a disc difference score (truncated toward negative infinity).
    #[inline(always)]
    pub const fn to_disc_diff(self) -> Score {
        self.0 >> Self::SCALE_BITS
    }

    /// Converts to a floating-point disc difference with full precision.
    #[inline(always)]
    pub fn to_disc_diff_f32(self) -> Scoref {
        (self.0 as f32) / (Self::SCALE as f32)
    }
}

impl Add for ScaledScore {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for ScaledScore {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for ScaledScore {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl AddAssign for ScaledScore {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for ScaledScore {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Add<i32> for ScaledScore {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        Self(self.0 + rhs)
    }
}

impl Sub<i32> for ScaledScore {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        Self(self.0 - rhs)
    }
}

impl Mul<i32> for ScaledScore {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i32) -> Self {
        Self(self.0 * rhs)
    }
}

impl Div<i32> for ScaledScore {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: i32) -> Self {
        Self(self.0 / rhs)
    }
}

impl MulAssign<i32> for ScaledScore {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: i32) {
        self.0 *= rhs;
    }
}

impl DivAssign<i32> for ScaledScore {
    #[inline(always)]
    fn div_assign(&mut self, rhs: i32) {
        self.0 /= rhs;
    }
}

impl fmt::Display for ScaledScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.to_disc_diff_f32())
    }
}

#[cfg(test)]
mod scaled_score_tests {
    use super::*;

    const EPSILON: f32 = f32::EPSILON;

    fn assert_disc_diff_f32_eq(score: ScaledScore, expected: Scoref) {
        let actual = score.to_disc_diff_f32();
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual} for raw value {}",
            score.value()
        );
    }

    #[test]
    fn constants_preserve_score_domain_and_search_sentinel_ordering() {
        assert_eq!(ScaledScore::SCALE_BITS, 8);
        assert_eq!(ScaledScore::SCALE, 256);
        assert_eq!(ScaledScore::ZERO.value(), 0);
        assert_eq!(ScaledScore::MIN.value(), SCORE_MIN * ScaledScore::SCALE);
        assert_eq!(ScaledScore::MAX.value(), SCORE_MAX * ScaledScore::SCALE);
        assert!(ScaledScore::MIN < ScaledScore::ZERO);
        assert!(ScaledScore::ZERO < ScaledScore::MAX);
        assert!(ScaledScore::MAX < ScaledScore::INF);
        assert!(-ScaledScore::INF < ScaledScore::MIN);
    }

    #[test]
    fn from_disc_diff_scales_whole_disc_scores() {
        for disc_diff in [SCORE_MIN, -1, 0, 1, SCORE_MAX] {
            let score = ScaledScore::from_disc_diff(disc_diff);

            assert_eq!(score.value(), disc_diff * ScaledScore::SCALE);
            assert_eq!(score.to_disc_diff(), disc_diff);
            assert_disc_diff_f32_eq(score, disc_diff as Scoref);
        }
    }

    #[test]
    fn from_raw_preserves_fractional_scores() {
        for (raw_value, expected_disc_diff, expected_disc_diff_f32) in [
            (1, 0, 1.0 / 256.0),
            (127, 0, 127.0 / 256.0),
            (128, 0, 0.5),
            (255, 0, 255.0 / 256.0),
            (256, 1, 1.0),
            (257, 1, 257.0 / 256.0),
            (-1, -1, -1.0 / 256.0),
            (-127, -1, -127.0 / 256.0),
            (-128, -1, -0.5),
            (-255, -1, -255.0 / 256.0),
            (-256, -1, -1.0),
            (-257, -2, -257.0 / 256.0),
        ] {
            let score = ScaledScore::from_raw(raw_value);

            assert_eq!(score.value(), raw_value);
            assert_eq!(score.to_disc_diff(), expected_disc_diff);
            assert_disc_diff_f32_eq(score, expected_disc_diff_f32);
        }
    }

    #[test]
    fn scaled_score_arithmetic_preserves_raw_units() {
        let one_and_half = ScaledScore::from_raw(ScaledScore::SCALE + ScaledScore::SCALE / 2);
        let quarter = ScaledScore::from_raw(ScaledScore::SCALE / 4);

        assert_eq!((one_and_half + quarter).value(), 448);
        assert_eq!((one_and_half - quarter).value(), 320);
        assert_eq!((-one_and_half).value(), -384);
        assert_eq!((quarter * 3).value(), 192);
        assert_eq!((one_and_half / 3).value(), 128);
    }

    #[test]
    fn assignment_arithmetic_updates_the_raw_value_in_place() {
        let mut score = ScaledScore::from_raw(ScaledScore::SCALE);

        score += ScaledScore::from_raw(ScaledScore::SCALE / 2);
        assert_eq!(score.value(), 384);

        score -= ScaledScore::from_raw(ScaledScore::SCALE / 4);
        assert_eq!(score.value(), 320);

        score *= 3;
        assert_eq!(score.value(), 960);

        score /= 5;
        assert_eq!(score.value(), 192);
    }

    #[test]
    fn raw_i32_addition_and_subtraction_use_scaled_units_not_disc_diffs() {
        let score = ScaledScore::from_disc_diff(10);

        assert_eq!((score + 100).value(), 10 * ScaledScore::SCALE + 100);
        assert_eq!((score - 50).value(), 10 * ScaledScore::SCALE - 50);
    }

    #[test]
    fn ordering_and_min_max_follow_raw_score_ordering() {
        let lower_fraction = ScaledScore::from_raw(ScaledScore::SCALE + 1);
        let higher_fraction = ScaledScore::from_raw(ScaledScore::SCALE + 2);
        let same_as_higher = ScaledScore::from_raw(ScaledScore::SCALE + 2);

        assert!(higher_fraction > lower_fraction);
        assert_eq!(higher_fraction, same_as_higher);
        assert_eq!(higher_fraction.max(lower_fraction), higher_fraction);
        assert_eq!(higher_fraction.min(lower_fraction), lower_fraction);
    }

    #[test]
    fn display_formats_disc_difference_with_two_decimal_places() {
        assert_eq!(ScaledScore::from_disc_diff(10).to_string(), "10.00");
        assert_eq!(
            ScaledScore::from_raw(ScaledScore::SCALE + ScaledScore::SCALE / 2).to_string(),
            "1.50"
        );
        assert_eq!(
            ScaledScore::from_raw(-(ScaledScore::SCALE / 2)).to_string(),
            "-0.50"
        );
    }

    #[test]
    fn default_is_zero_score() {
        let score: ScaledScore = Default::default();

        assert_eq!(score, ScaledScore::ZERO);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn from_disc_diff_rejects_values_beyond_search_sentinel_in_debug_builds() {
        let _ = ScaledScore::from_disc_diff(SCORE_INF + 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn from_raw_rejects_values_beyond_scaled_score_sentinel_in_debug_builds() {
        let _ = ScaledScore::from_raw(ScaledScore::INF.value() + 1);
    }
}
