//! Common type aliases used throughout the engine.

/// Search depth
pub type Depth = u32;

/// Score
pub type Score = i32;

/// Floating-point score
pub type Scoref = f32;

/// Selectivity level for search pruning (ProbCut confidence levels)
///
/// Lower levels are more aggressive (prune more), higher levels are more conservative.
/// `NoSelectivity` disables ProbCut entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(u8)]
pub enum Selectivity {
    /// Most aggressive: 68% confidence (t=1.0)
    #[default]
    Level0 = 0,
    /// 73% confidence (t=1.1)
    Level1 = 1,
    /// 87% confidence (t=1.5)
    Level2 = 2,
    /// 95% confidence (t=2.0)
    Level3 = 3,
    /// 98% confidence (t=2.6)
    Level4 = 4,
    /// Most conservative: 99% confidence (t=3.3)
    Level5 = 5,
    /// ProbCut disabled
    None = 6,
}

impl Selectivity {
    /// Selectivity configuration: (t_multiplier, probability_percent)
    const CONFIG: [(f64, i32); 7] = [
        (1.0, 68),    // Level0: Most aggressive
        (1.1, 73),    // Level1
        (1.5, 87),    // Level2
        (2.0, 95),    // Level3
        (2.6, 98),    // Level4
        (3.3, 99),    // Level5: Most conservative
        (999.0, 100), // NoSelectivity: Effectively disabled
    ];

    /// Get the statistical confidence multiplier (t-value)
    #[inline]
    pub fn t_value(self) -> f64 {
        Self::CONFIG[self as usize].0
    }

    /// Get the expected success probability percentage
    #[inline]
    pub fn probability(self) -> i32 {
        Self::CONFIG[self as usize].1
    }

    /// Convert to u8
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Create from u8 value
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        // SAFETY: Selectivity enum has repr(u8) with values 0-6
        unsafe { std::mem::transmute(value.min(6)) }
    }

    /// Check if ProbCut is enabled for this selectivity level
    #[inline]
    pub fn is_enabled(self) -> bool {
        self != Selectivity::None
    }
}
