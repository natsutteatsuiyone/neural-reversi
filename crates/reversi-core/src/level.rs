//! Game difficulty levels and search depth configuration.

use crate::probcut::Selectivity;
use crate::types::Depth;

/// Represents a difficulty level with associated search depths.
///
/// Each level defines:
/// - A midgame search depth for the middle portion of the game
/// - Multiple endgame depths for different selectivity levels (0-6)
///
/// Higher levels generally correspond to deeper searches and stronger play.
#[derive(Copy, Clone)]
pub struct Level {
    /// Search depth used during the midgame phase.
    pub mid_depth: Depth,
    /// Endgame search depths indexed by selectivity level.
    ///
    /// The array has 7 elements, where:
    /// - Index 0: Most aggressive pruning (Selectivity::Level0)
    /// - Index 6: ProbCut disabled (Selectivity::None)
    pub end_depth: [Depth; 7],
}

impl Level {
    /// Creates a Level with no depth restrictions (for time-controlled search).
    pub const fn unlimited() -> Self {
        Level {
            mid_depth: 60,
            end_depth: [14; 7],
        }
    }

    /// Creates a Level for perfect endgame solving.
    pub const fn perfect() -> Self {
        Level {
            mid_depth: 60,
            end_depth: [60; 7],
        }
    }

    /// Returns the endgame search depth for a given selectivity level.
    ///
    /// # Arguments
    ///
    /// * `selectivity` - The selectivity level, where lower values mean
    ///   more aggressive pruning and faster searches.
    ///
    /// # Returns
    ///
    /// The search depth to use for endgame positions at the given selectivity.
    pub fn get_end_depth(&self, selectivity: Selectivity) -> Depth {
        self.end_depth[selectivity.as_u8() as usize]
    }
}

/// Retrieves the configuration for a specific difficulty level.
///
/// # Arguments
///
/// * `lv` - The level index (0-24), where 0 is the weakest and 24 is the strongest.
///
/// # Returns
///
/// A `Level` struct containing the search depth configuration.
///
/// # Panics
///
/// Panics if `lv` is outside the valid range of 0-24.
pub fn get_level(lv: usize) -> Level {
    if lv >= LEVELS.len() {
        panic!(
            "Invalid level: {}. Valid range is 0 to {}",
            lv,
            LEVELS.len() - 1
        );
    }
    LEVELS[lv]
}

/// Pre-configured difficulty levels ranging from 0 (easiest) to 24 (hardest).
#[rustfmt::skip]
const LEVELS: [Level; 25] = [
    Level { mid_depth:  1, end_depth: [ 1, 1, 1, 1, 1, 1, 1] },
    Level { mid_depth:  1, end_depth: [ 2, 2, 2, 2, 2, 2, 2] },
    Level { mid_depth:  2, end_depth: [ 4, 4, 4, 4, 4, 4, 4] },
    Level { mid_depth:  3, end_depth: [ 6, 6, 6, 6, 6, 6, 6] },
    Level { mid_depth:  4, end_depth: [ 8, 8, 8, 8, 8, 8, 8] },
    Level { mid_depth:  5, end_depth: [10,10,10,10,10,10,10] },
    Level { mid_depth:  6, end_depth: [12,12,12,12,12,12,12] },
    Level { mid_depth:  7, end_depth: [14,14,14,14,14,14,14] },
    Level { mid_depth:  8, end_depth: [16,16,16,16,16,16,16] },
    Level { mid_depth:  9, end_depth: [18,18,18,18,18,18,18] },
    Level { mid_depth: 10, end_depth: [20,20,20,20,20,20,20] },
    Level { mid_depth: 11, end_depth: [21,21,21,21,21,21,21] },
    Level { mid_depth: 12, end_depth: [21,21,21,21,21,21,21] },
    Level { mid_depth: 13, end_depth: [22,22,22,22,22,22,22] },
    Level { mid_depth: 14, end_depth: [22,22,22,22,22,22,22] },
    Level { mid_depth: 15, end_depth: [24,24,24,24,24,24,24] },
    Level { mid_depth: 16, end_depth: [24,24,24,24,24,24,24] },
    Level { mid_depth: 17, end_depth: [24,24,24,24,24,24,24] },
    Level { mid_depth: 18, end_depth: [26,26,26,26,26,26,26] },
    Level { mid_depth: 19, end_depth: [26,26,26,26,26,26,26] },
    Level { mid_depth: 20, end_depth: [26,26,26,26,26,26,26] },
    Level { mid_depth: 21, end_depth: [28,28,28,28,28,28,28] },
    Level { mid_depth: 22, end_depth: [28,28,28,28,28,28,28] },
    Level { mid_depth: 23, end_depth: [30,30,30,30,30,30,30] },
    Level { mid_depth: 24, end_depth: [30,30,30,30,30,30,30] },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_level_valid_range() {
        // Test all valid levels
        for (i, &expected_level) in LEVELS.iter().enumerate() {
            let level = get_level(i);
            assert_eq!(level.mid_depth, expected_level.mid_depth);
            assert_eq!(level.end_depth, expected_level.end_depth);
        }
    }

    #[test]
    fn test_get_end_depth() {
        let level = Level {
            mid_depth: 10,
            end_depth: [20, 21, 22, 23, 24, 25, 26],
        };

        // Test all selectivity levels
        for i in 0..=6 {
            assert_eq!(
                level.get_end_depth(Selectivity::from_u8(i)),
                20 + i as Depth
            );
        }
    }

    #[test]
    fn test_level_progression() {
        // Verify that levels generally increase in difficulty
        for i in 0..21 {
            let current = get_level(i);
            let next = get_level(i + 1);

            // Mid depth should generally increase or stay the same
            assert!(next.mid_depth >= current.mid_depth);
        }
    }
}
