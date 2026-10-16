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
    pub end_depth: [Depth; ENDGAME_STEPS],
}

/// Number of steps in [`Level::ENDGAME_SELECTIVITY`].
pub const ENDGAME_STEPS: usize = 5;

impl Level {
    /// Selectivity sequence for iterative endgame search.
    ///
    /// Progresses from aggressive pruning to exact solving.
    pub const ENDGAME_SELECTIVITY: [Selectivity; ENDGAME_STEPS] = [
        Selectivity::Level1,
        Selectivity::Level2,
        Selectivity::Level3,
        Selectivity::Level4,
        Selectivity::None,
    ];

    /// Creates a level with `mid_depth` for the midgame and the same endgame
    /// depth for every [`Self::ENDGAME_SELECTIVITY`] step.
    pub const fn uniform(mid_depth: Depth, end_depth: Depth) -> Self {
        Level {
            mid_depth,
            end_depth: [end_depth; ENDGAME_STEPS],
        }
    }

    /// Creates a level with `mid_depth` for the midgame and one endgame depth
    /// per [`Self::ENDGAME_SELECTIVITY`] step.
    pub const fn with_depths(mid_depth: Depth, end_depth: [Depth; ENDGAME_STEPS]) -> Self {
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
    Level::uniform(1, 1),
    Level::uniform(1, 2),
    Level::uniform(2, 4),
    Level::uniform(3, 6),
    Level::uniform(4, 8),
    Level::uniform(5, 10),
    Level::uniform(6, 12),
    Level::uniform(7, 14),
    Level::uniform(8, 16),
    Level::uniform(9, 18),
    Level::uniform(10, 20),
    Level::uniform(11, 21),
    Level::uniform(12, 21),
    Level::uniform(13, 22),
    Level::uniform(14, 22),
    Level::uniform(15, 24),
    Level::uniform(16, 24),
    Level::uniform(17, 24),
    Level::uniform(18, 26),
    Level::uniform(19, 26),
    Level::uniform(20, 26),
    Level::uniform(21, 28),
    Level::uniform(22, 28),
    Level::uniform(23, 30),
    Level::uniform(24, 30),
    Level::uniform(25, 30),
    Level::uniform(26, 30),
    Level::uniform(27, 30),
    Level::uniform(28, 30),
    Level::uniform(29, 30),
    Level::uniform(30, 30),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_end_depth() {
        let level = Level {
            mid_depth: 10,
            end_depth: [20, 22, 23, 24, 25],
        };

        assert_eq!(level.get_end_depth(Selectivity::Level1), 20);
        assert_eq!(level.get_end_depth(Selectivity::Level2), 22);
        assert_eq!(level.get_end_depth(Selectivity::Level3), 23);
        assert_eq!(level.get_end_depth(Selectivity::Level4), 24);
        assert_eq!(level.get_end_depth(Selectivity::None), 25);
    }

    #[test]
    #[should_panic(expected = "Invalid level")]
    fn get_level_panics_above_the_valid_range() {
        // Pinning the message ensures the panic comes from `get_level`'s explicit
        // range guard, not from some unrelated panic source.
        let _ = get_level(LEVELS.len());
    }
}
