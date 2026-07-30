//! Game difficulty levels and search depth configuration.

use crate::probcut::Selectivity;
use crate::types::Depth;

/// A difficulty level with associated search depths.
///
/// Each level defines:
/// - A midgame search depth for the middle portion of the game
/// - Endgame depths for each step in [`Self::ENDGAME_SELECTIVITY`]
///
/// Higher levels generally correspond to deeper searches and stronger play.
#[derive(Copy, Clone)]
pub struct Level {
    /// Search depth used during the midgame phase.
    pub mid_depth: Depth,
    /// Endgame search depths, one per [`Self::ENDGAME_SELECTIVITY`] entry.
    pub end_depth: [Depth; 4],
}

impl Level {
    /// Selectivity sequence for iterative endgame search.
    ///
    /// Progresses from aggressive pruning to exact solving.
    pub const ENDGAME_SELECTIVITY: [Selectivity; 4] = [
        Selectivity::Level1,
        Selectivity::Level2,
        Selectivity::Level3,
        Selectivity::None,
    ];

    /// Creates a level with `mid_depth` for the midgame and the same endgame
    /// depth for every [`Self::ENDGAME_SELECTIVITY`] step.
    pub const fn uniform(mid_depth: Depth, end_depth: Depth) -> Self {
        Level {
            mid_depth,
            end_depth: [end_depth; 4],
        }
    }

    /// Creates a level with `mid_depth` for the midgame and one endgame depth
    /// per [`Self::ENDGAME_SELECTIVITY`] step.
    pub const fn with_depths(mid_depth: Depth, end_depth: [Depth; 4]) -> Self {
        Level {
            mid_depth,
            end_depth,
        }
    }

    /// Creates a [`Level`] for time-controlled search.
    ///
    /// Sets `mid_depth` to 60 (effectively unlimited) and `end_depth` to 14.
    /// The endgame depth only affects the root dispatch: positions with at
    /// most `end_depth` empties enter the endgame solver directly, while
    /// deeper positions start in the midgame driver, which hands off to the
    /// endgame solver itself once iterative deepening approaches the full
    /// depth. [`Search::run`] extends the endgame depth to 60 once a previous
    /// search has reached the endgame phase.
    ///
    /// [`Search::run`]: crate::search::Search::run
    pub const fn unlimited() -> Self {
        Self::uniform(60, 14)
    }

    /// Creates a [`Level`] for perfect endgame solving.
    pub const fn perfect() -> Self {
        Self::uniform(60, 60)
    }

    /// Returns the endgame search depth for a given [`Selectivity`] level.
    ///
    /// # Panics
    ///
    /// Panics if `selectivity` is not in [`Self::ENDGAME_SELECTIVITY`].
    pub fn get_end_depth(&self, selectivity: Selectivity) -> Depth {
        let index = Self::ENDGAME_SELECTIVITY
            .iter()
            .position(|&s| s == selectivity)
            .unwrap();
        self.end_depth[index]
    }

    /// Returns the minimum endgame search depth (the first/most aggressive selectivity).
    pub fn min_end_depth(&self) -> Depth {
        self.end_depth[0]
    }
}

/// Maximum valid level index.
pub const MAX_LEVEL: usize = LEVELS.len() - 1;

/// Returns the [`Level`] configuration for a specific difficulty level.
///
/// # Panics
///
/// Panics if `lv` is outside the valid range of 0 to [`MAX_LEVEL`].
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

/// Pre-configured difficulty levels ranging from 0 (easiest) to [`MAX_LEVEL`] (hardest).
#[rustfmt::skip]
const LEVELS: [Level; 31] = [
    Level { mid_depth:  1, end_depth: [ 1, 1, 1, 1] },
    Level { mid_depth:  1, end_depth: [ 2, 2, 2, 2] },
    Level { mid_depth:  2, end_depth: [ 4, 4, 4, 4] },
    Level { mid_depth:  3, end_depth: [ 6, 6, 6, 6] },
    Level { mid_depth:  4, end_depth: [ 8, 8, 8, 8] },
    Level { mid_depth:  5, end_depth: [10,10,10,10] },
    Level { mid_depth:  6, end_depth: [12,12,12,12] },
    Level { mid_depth:  7, end_depth: [14,14,14,14] },
    Level { mid_depth:  8, end_depth: [16,16,16,16] },
    Level { mid_depth:  9, end_depth: [18,18,18,18] },
    Level { mid_depth: 10, end_depth: [20,20,20,20] },
    Level { mid_depth: 11, end_depth: [21,21,21,21] },
    Level { mid_depth: 12, end_depth: [21,21,21,21] },
    Level { mid_depth: 13, end_depth: [22,22,22,22] },
    Level { mid_depth: 14, end_depth: [22,22,22,22] },
    Level { mid_depth: 15, end_depth: [24,24,24,24] },
    Level { mid_depth: 16, end_depth: [24,24,24,24] },
    Level { mid_depth: 17, end_depth: [24,24,24,24] },
    Level { mid_depth: 18, end_depth: [26,26,26,26] },
    Level { mid_depth: 19, end_depth: [26,26,26,26] },
    Level { mid_depth: 20, end_depth: [26,26,26,26] },
    Level { mid_depth: 21, end_depth: [28,28,28,28] },
    Level { mid_depth: 22, end_depth: [28,28,28,28] },
    Level { mid_depth: 23, end_depth: [30,30,30,30] },
    Level { mid_depth: 24, end_depth: [30,30,30,30] },
    Level { mid_depth: 25, end_depth: [30,30,30,30] },
    Level { mid_depth: 26, end_depth: [30,30,30,30] },
    Level { mid_depth: 27, end_depth: [30,30,30,30] },
    Level { mid_depth: 28, end_depth: [30,30,30,30] },
    Level { mid_depth: 29, end_depth: [30,30,30,30] },
    Level { mid_depth: 30, end_depth: [30,30,30,30] },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_end_depth() {
        let level = Level {
            mid_depth: 10,
            end_depth: [20, 22, 24, 25],
        };

        assert_eq!(level.get_end_depth(Selectivity::Level1), 20);
        assert_eq!(level.get_end_depth(Selectivity::Level2), 22);
        assert_eq!(level.get_end_depth(Selectivity::Level3), 24);
        assert_eq!(level.get_end_depth(Selectivity::None), 25);
    }

    #[test]
    fn test_level_progression() {
        // Verify that levels generally increase in difficulty
        for i in 0..LEVELS.len() - 1 {
            let current = get_level(i);
            let next = get_level(i + 1);

            // Mid depth should generally increase or stay the same
            assert!(next.mid_depth >= current.mid_depth);
        }
    }

    #[test]
    #[should_panic(expected = "Invalid level")]
    fn get_level_panics_above_the_valid_range() {
        // Pinning the message ensures the panic comes from `get_level`'s explicit
        // range guard, not from some unrelated panic source.
        let _ = get_level(LEVELS.len());
    }
}
