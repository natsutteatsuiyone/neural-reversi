//! Search scores: [`Score`] with sub-disc precision and whole-disc [`DiscScore`].

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Maximum possible score (disc difference).
pub const DISC_SCORE_MAX: DiscScore = DiscScore::new(64);

/// Minimum possible score (disc difference).
pub const DISC_SCORE_MIN: DiscScore = DiscScore::new(-64);

/// Infinity score for search algorithms.
pub const DISC_SCORE_INF: DiscScore = DiscScore::new(i8::MAX as i32);

/// Disc-difference score: -64 to +64 for real game outcomes.
///
/// Search bounds may reach [`DISC_SCORE_INF`], which lies outside that range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct DiscScore(i32);

impl DiscScore {
    /// Creates a [`DiscScore`] from a disc difference.
    #[inline(always)]
    pub const fn new(value: i32) -> Self {
        Self(value)
    }

    /// Returns the disc difference.
    #[inline(always)]
    pub const fn value(self) -> i32 {
        self.0
    }
}

impl Neg for DiscScore {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Add<i32> for DiscScore {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        Self(self.0 + rhs)
    }
}

impl Sub<i32> for DiscScore {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        Self(self.0 - rhs)
    }
}

impl fmt::Display for DiscScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Search and evaluation score with sub-disc precision.
///
/// In Reversi, the final score is a disc difference ranging from -64 to +64.
/// However, during evaluation, the engine needs finer granularity to distinguish
/// between positions that would otherwise appear equal. `Score` achieves this
/// by scaling values by 256 (2^8), providing 8 bits of fractional precision.
///
/// # Value Representation
///
/// - **Raw value**: The internal [`i32`] representation (scaled)
/// - **Disc difference**: The [`DiscScore`] (raw_value >> 8)
///
/// # Special Values
///
/// - [`Score::MIN`] / [`Score::MAX`]: Bounds for actual game outcomes (-64/+64 discs)
/// - [`Score::INF`]: Sentinel value for search algorithm bounds (larger than any real score)
///
/// # Examples
///
/// ```
/// use reversi_core::score::{DiscScore, Score};
///
/// // Create from a disc difference (scaled internally by 256)
/// let score = Score::from_disc_diff(DiscScore::new(10));
/// assert_eq!(score.value(), 2560);
/// assert_eq!(score.to_disc_diff(), DiscScore::new(10));
///
/// // Create from a raw internal value
/// let raw = Score::from_raw(2560);
/// assert_eq!(raw.to_disc_diff(), DiscScore::new(10));
///
/// // Arithmetic preserves scaling
/// let a = Score::from_disc_diff(DiscScore::new(10));
/// let b = Score::from_disc_diff(DiscScore::new(5));
/// assert_eq!((a + b).to_disc_diff(), DiscScore::new(15));
/// assert_eq!((-a).to_disc_diff(), DiscScore::new(-10));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Score(i32);

impl Score {
    /// Number of bits used for scaling (8 bits = 256x).
    pub const SCALE_BITS: i32 = 8;

    /// Scale factor: 256.
    pub const SCALE: i32 = 1 << Self::SCALE_BITS;

    /// Zero score.
    pub const ZERO: Self = Self(0);

    /// Maximum achievable score: +64 discs.
    pub const MAX: Self = Self(DISC_SCORE_MAX.value() << Self::SCALE_BITS);

    /// Minimum achievable score: -64 discs.
    pub const MIN: Self = Self(DISC_SCORE_MIN.value() << Self::SCALE_BITS);

    /// Infinity sentinel for alpha-beta search bounds.
    pub const INF: Self = Self(i16::MAX as i32);

    /// Creates a [`Score`] from a raw internal value (already scaled by 256).
    #[inline(always)]
    pub const fn from_raw(raw_value: i32) -> Self {
        debug_assert!(raw_value >= -Self::INF.0 && raw_value <= Self::INF.0);
        Self(raw_value)
    }

    /// Creates a [`Score`] from a disc difference (-64 to +64).
    #[inline(always)]
    pub const fn from_disc_diff(disc_diff: DiscScore) -> Self {
        debug_assert!(
            disc_diff.value() >= -DISC_SCORE_INF.value()
                && disc_diff.value() <= DISC_SCORE_INF.value()
        );
        Self(disc_diff.value() << Self::SCALE_BITS)
    }

    /// Returns the raw internal value (scaled by 256).
    #[inline(always)]
    pub const fn value(self) -> i32 {
        self.0
    }

    /// Converts to a disc difference score (truncated toward negative infinity).
    #[inline(always)]
    pub const fn to_disc_diff(self) -> DiscScore {
        DiscScore(self.0 >> Self::SCALE_BITS)
    }

    /// Converts to a floating-point disc difference with full precision.
    #[inline(always)]
    pub fn to_disc_diff_f32(self) -> f32 {
        (self.0 as f32) / (Self::SCALE as f32)
    }
}

impl Add for Score {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Score {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for Score {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl AddAssign for Score {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Score {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Add<i32> for Score {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: i32) -> Self {
        Self(self.0 + rhs)
    }
}

impl Sub<i32> for Score {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: i32) -> Self {
        Self(self.0 - rhs)
    }
}

impl Mul<i32> for Score {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: i32) -> Self {
        Self(self.0 * rhs)
    }
}

impl Div<i32> for Score {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: i32) -> Self {
        Self(self.0 / rhs)
    }
}

impl MulAssign<i32> for Score {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: i32) {
        self.0 *= rhs;
    }
}

impl DivAssign<i32> for Score {
    #[inline(always)]
    fn div_assign(&mut self, rhs: i32) {
        self.0 /= rhs;
    }
}

impl fmt::Display for Score {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.to_disc_diff_f32())
    }
}

// Real scores stay strictly inside the ±INF sentinels (and the TT's i16 packing).
const _: () =
    assert!(-Score::INF.value() < Score::MIN.value() && Score::MAX.value() < Score::INF.value());

#[cfg(test)]
mod score_tests {
    use super::*;

    const EPSILON: f32 = f32::EPSILON;

    fn assert_disc_diff_f32_eq(score: Score, expected: f32) {
        let actual = score.to_disc_diff_f32();
        assert!(
            (actual - expected).abs() <= EPSILON,
            "expected {expected}, got {actual} for raw value {}",
            score.value()
        );
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
            let score = Score::from_raw(raw_value);

            assert_eq!(score.value(), raw_value);
            assert_eq!(score.to_disc_diff().value(), expected_disc_diff);
            assert_disc_diff_f32_eq(score, expected_disc_diff_f32);
        }
    }

    #[test]
    fn raw_i32_addition_and_subtraction_use_scaled_units_not_disc_diffs() {
        let score = Score::from_disc_diff(DiscScore::new(10));

        assert_eq!((score + 100).value(), 10 * Score::SCALE + 100);
        assert_eq!((score - 50).value(), 10 * Score::SCALE - 50);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn from_disc_diff_rejects_values_beyond_search_sentinel_in_debug_builds() {
        let _ = Score::from_disc_diff(DISC_SCORE_INF + 1);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn from_raw_rejects_values_beyond_score_sentinel_in_debug_builds() {
        let _ = Score::from_raw(Score::INF.value() + 1);
    }
}
