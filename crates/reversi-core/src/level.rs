//! Fixed-depth search levels.

use crate::constants::INITIAL_EMPTY_COUNT;
use crate::probcut::{ENDGAME_STEPS, Selectivity};
use crate::types::Depth;

/// Depth limits of a fixed-level search.
///
/// A position is searched `mid_depth` plies deep until few enough squares
/// are empty for the endgame solver to take over. The solver climbs
/// [`Selectivity::ENDGAME_LADDER`] as far as [`Self::end_selectivity`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    /// Deepest midgame iteration.
    pub mid_depth: Depth,
    end_depth: [Depth; ENDGAME_STEPS],
    last_step: Selectivity,
}

const FULL_DEPTH: Depth = INITIAL_EMPTY_COUNT as Depth;

/// Highest preset level.
pub const MAX_LEVEL: usize = PRESET_END_DEPTHS.len() - 1;

#[rustfmt::skip]
const PRESET_END_DEPTHS: [Depth; 31] = [
    1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
    21, 21, 22, 22, 24, 24, 24, 26, 26, 26,
    28, 28, 30, 30, 30, 30, 30, 30, 30, 30,
];

impl Level {
    /// Creates a level that solves every [`Selectivity::ENDGAME_LADDER`] step
    /// from `end_depth` empties.
    pub const fn uniform(mid_depth: Depth, end_depth: Depth) -> Self {
        Self::with_depths(mid_depth, [end_depth; ENDGAME_STEPS])
    }

    /// Creates a level that solves each [`Selectivity::ENDGAME_LADDER`] step
    /// from its own entry of `end_depth` empties.
    pub const fn with_depths(mid_depth: Depth, end_depth: [Depth; ENDGAME_STEPS]) -> Self {
        Level {
            mid_depth,
            end_depth,
            last_step: Selectivity::None,
        }
    }

    /// Creates a level that solves every position exactly.
    pub const fn perfect() -> Self {
        Self::uniform(FULL_DEPTH, FULL_DEPTH)
    }

    /// Returns preset level `n`, which searches the midgame `n` plies deep.
    ///
    /// Level 0 searches one ply like level 1; frontends offer
    /// `1..=MAX_LEVEL`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is greater than [`MAX_LEVEL`].
    pub fn preset(n: usize) -> Self {
        assert!(
            n <= MAX_LEVEL,
            "invalid level {n}: the valid range is 0 to {MAX_LEVEL}"
        );
        Self::uniform((n as Depth).max(1), PRESET_END_DEPTHS[n])
    }

    /// Returns this level with the endgame ladder capped at `last`: the
    /// steps after it are never run.
    ///
    /// The depths of the steps up to `last` are kept.
    #[must_use]
    pub fn stopping_at(self, last: Selectivity) -> Self {
        Level {
            last_step: self.last_step.min(last),
            ..self
        }
    }

    /// Returns the last [`Selectivity::ENDGAME_LADDER`] step a position with
    /// `n_empties` empties is solved to, or `None` if the position is left
    /// to the midgame search.
    ///
    /// The ladder runs in order, so a step counts only if every step before
    /// it is solved from `n_empties` empties too.
    pub fn end_selectivity(&self, n_empties: Depth) -> Option<Selectivity> {
        Selectivity::ENDGAME_LADDER
            .into_iter()
            .zip(self.end_depth)
            .take_while(|&(selectivity, end_depth)| {
                selectivity <= self.last_step && n_empties <= end_depth
            })
            .map(|(selectivity, _)| selectivity)
            .last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_selectivity_is_the_last_step_solved_from_the_empties() {
        let level = Level::with_depths(12, [28, 27, 26, 25, 24]);

        for (n_empties, expected) in [
            (29, None),
            (28, Some(Selectivity::Level1)),
            (27, Some(Selectivity::Level2)),
            (26, Some(Selectivity::Level3)),
            (25, Some(Selectivity::Level4)),
            (24, Some(Selectivity::None)),
            (1, Some(Selectivity::None)),
        ] {
            assert_eq!(level.end_selectivity(n_empties), expected, "{n_empties}");
        }
    }

    #[test]
    fn end_selectivity_stops_at_the_first_step_out_of_reach() {
        let level = Level::with_depths(12, [24, 20, 24, 24, 24]);

        assert_eq!(level.end_selectivity(25), None);
        assert_eq!(level.end_selectivity(22), Some(Selectivity::Level1));
        assert_eq!(level.end_selectivity(20), Some(Selectivity::None));
    }

    #[test]
    fn stopping_at_caps_the_ladder_and_keeps_the_earlier_depths() {
        let level = Level::perfect().stopping_at(Selectivity::Level2);

        for n_empties in [1, 30, 60] {
            assert_eq!(level.end_selectivity(n_empties), Some(Selectivity::Level2));
        }

        let level = Level::uniform(12, 24).stopping_at(Selectivity::Level4);
        assert_eq!(level.end_selectivity(25), None);
        assert_eq!(level.end_selectivity(24), Some(Selectivity::Level4));
        assert_eq!(
            level.stopping_at(Selectivity::None).end_selectivity(24),
            Some(Selectivity::Level4)
        );
    }

    #[test]
    fn presets_span_level_zero_to_max_level() {
        assert_eq!(Level::preset(0), Level::uniform(1, 1));
        assert_eq!(Level::preset(1), Level::uniform(1, 2));
        assert_eq!(Level::preset(21), Level::uniform(21, 28));
        assert_eq!(Level::preset(MAX_LEVEL), Level::uniform(30, 30));
    }

    #[test]
    #[should_panic(expected = "invalid level")]
    fn preset_panics_above_max_level() {
        let _ = Level::preset(MAX_LEVEL + 1);
    }
}
