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

/// Scaled score - internal evaluation value (scaled by 256).
///
/// Disc difference scores are scaled by 256 to preserve precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ScaledScore(i32);

impl ScaledScore {
    /// Number of bits used for scaling.
    pub const SCALE_BITS: i32 = 8;
    /// Scale factor (256).
    pub const SCALE: i32 = 1 << Self::SCALE_BITS;

    /// Zero score.
    pub const ZERO: Self = Self(0);
    /// Maximum scaled score (+64 discs).
    pub const MAX: Self = Self(SCORE_MAX << Self::SCALE_BITS);
    /// Minimum scaled score (-64 discs).
    pub const MIN: Self = Self(SCORE_MIN << Self::SCALE_BITS);
    /// Infinity score for search algorithms.
    pub const INF: Self = Self(SCORE_INF << Self::SCALE_BITS);

    /// Creates a ScaledScore from a raw internal value.
    ///
    /// # Arguments
    ///
    /// * `raw_value` - Raw internal value
    #[inline(always)]
    pub const fn new(raw_value: i32) -> Self {
        Self(raw_value)
    }

    /// Returns the raw internal value.
    #[inline(always)]
    pub const fn value(self) -> i32 {
        self.0
    }

    /// Creates a ScaledScore from a disc difference score.
    #[inline(always)]
    pub const fn from_disc_diff(disc_diff: Score) -> Self {
        Self(disc_diff << Self::SCALE_BITS)
    }

    /// Converts to a disc difference score (truncated).
    #[inline(always)]
    pub const fn to_disc_diff(self) -> Score {
        self.0 >> Self::SCALE_BITS
    }

    /// Converts to a floating-point score.
    #[inline(always)]
    pub fn to_disc_diff_f32(self) -> Scoref {
        (self.0 as f32) / (Self::SCALE as f32)
    }

    /// Clamps the value to the given range.
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }

    /// Returns the absolute value.
    #[inline(always)]
    pub const fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// Returns the maximum of two values.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// Returns the minimum of two values.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
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

impl AddAssign<i32> for ScaledScore {
    #[inline(always)]
    fn add_assign(&mut self, rhs: i32) {
        self.0 += rhs;
    }
}

impl SubAssign<i32> for ScaledScore {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: i32) {
        self.0 -= rhs;
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

    #[test]
    fn test_conversions() {
        let score = ScaledScore::from_disc_diff(10);
        assert_eq!(score.value(), 2560);
        assert_eq!(score.to_disc_diff(), 10);
        assert!((score.to_disc_diff_f32() - 10.0).abs() < 0.001);

        let neg_score = ScaledScore::from_disc_diff(-5);
        assert_eq!(neg_score.to_disc_diff(), -5);
    }

    #[test]
    fn test_from_raw() {
        let score = ScaledScore::new(1000);
        assert_eq!(score.value(), 1000);
        assert_eq!(score.to_disc_diff(), 3); // 1000 >> 8 = 3
    }

    #[test]
    fn test_arithmetic() {
        let a = ScaledScore::from_disc_diff(10);
        let b = ScaledScore::from_disc_diff(5);

        assert_eq!((a + b).to_disc_diff(), 15);
        assert_eq!((a - b).to_disc_diff(), 5);
        assert_eq!((-a).to_disc_diff(), -10);
    }

    #[test]
    fn test_arithmetic_assign() {
        let mut score = ScaledScore::from_disc_diff(10);
        score += ScaledScore::from_disc_diff(5);
        assert_eq!(score.to_disc_diff(), 15);

        score -= ScaledScore::from_disc_diff(3);
        assert_eq!(score.to_disc_diff(), 12);
    }

    #[test]
    fn test_i32_arithmetic() {
        let score = ScaledScore::from_disc_diff(10);

        // Adding raw value (not disc diff)
        let result = score + 100;
        assert_eq!(result.value(), 2560 + 100);

        let result = score - 50;
        assert_eq!(result.value(), 2560 - 50);
    }

    #[test]
    fn test_comparison() {
        let a = ScaledScore::from_disc_diff(10);
        let b = ScaledScore::from_disc_diff(5);
        let c = ScaledScore::from_disc_diff(10);

        assert!(a > b);
        assert!(b < a);
        assert_eq!(a, c);
        assert!(a >= c);
        assert!(a <= c);
    }

    #[test]
    fn test_abs() {
        assert_eq!(ScaledScore::from_disc_diff(10).abs().to_disc_diff(), 10);
        assert_eq!(ScaledScore::from_disc_diff(-10).abs().to_disc_diff(), 10);
    }

    #[test]
    fn test_min_max() {
        let a = ScaledScore::from_disc_diff(10);
        let b = ScaledScore::from_disc_diff(5);

        assert_eq!(a.max(b), a);
        assert_eq!(a.min(b), b);
    }

    #[test]
    fn test_clamp() {
        let score = ScaledScore::from_disc_diff(100);
        let clamped = score.clamp(ScaledScore::MIN, ScaledScore::MAX);
        assert_eq!(clamped, ScaledScore::MAX);

        let score = ScaledScore::from_disc_diff(-100);
        let clamped = score.clamp(ScaledScore::MIN, ScaledScore::MAX);
        assert_eq!(clamped, ScaledScore::MIN);
    }

    #[test]
    fn test_display() {
        let score = ScaledScore::from_disc_diff(10);
        assert_eq!(format!("{}", score), "10.00");

        let score = ScaledScore::new(256 + 128); // 1.5
        assert_eq!(format!("{}", score), "1.50");
    }

    #[test]
    fn test_default() {
        let score: ScaledScore = Default::default();
        assert_eq!(score, ScaledScore::ZERO);
    }
}
