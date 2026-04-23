//! Mapping from GGS clock fields to `reversi_core` `TimeControlMode`.

use reversi_core::search::time_control::TimeControlMode;

/// Converts GGS clock values to the matching `TimeControlMode`.
///
/// GGS clocks are `[initial]/[increment]/[extension]`. Othello commonly uses
/// the extension field as Japanese byoyomi (`15:00//02:00`). If an extension
/// is present, prefer that conservative model; otherwise use Fischer with the
/// parsed increment.
pub fn derive_mode(main_ms: u64, increment_ms: u64, byoyomi_ms: u64) -> TimeControlMode {
    if byoyomi_ms > 0 {
        TimeControlMode::JapaneseByo {
            main_time_ms: main_ms,
            time_per_move_ms: byoyomi_ms,
        }
    } else {
        TimeControlMode::Fischer {
            main_time_ms: main_ms,
            increment_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byoyomi_present_maps_to_japanese_byo() {
        assert_eq!(
            derive_mode(56_000, 0, 120_000),
            TimeControlMode::JapaneseByo {
                main_time_ms: 56_000,
                time_per_move_ms: 120_000,
            }
        );
    }

    #[test]
    fn byoyomi_zero_maps_to_fischer_sudden_death() {
        assert_eq!(
            derive_mode(300_000, 0, 0),
            TimeControlMode::Fischer {
                main_time_ms: 300_000,
                increment_ms: 0
            },
        );
    }

    #[test]
    fn increment_without_extension_maps_to_fischer() {
        assert_eq!(
            derive_mode(300_000, 5_000, 0),
            TimeControlMode::Fischer {
                main_time_ms: 300_000,
                increment_ms: 5_000
            },
        );
    }

    #[test]
    fn main_zero_and_byoyomi_present_still_japanese_byo() {
        assert_eq!(
            derive_mode(0, 0, 120_000),
            TimeControlMode::JapaneseByo {
                main_time_ms: 0,
                time_per_move_ms: 120_000
            },
        );
    }
}
