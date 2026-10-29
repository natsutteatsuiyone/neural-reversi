use reversi_core::types::Depth;

use crate::probcut::Selectivity;

pub(crate) const DEFAULT_TT_MB: usize = 128;
pub(crate) const DEFAULT_MID_DEPTH: Depth = 12;
pub(crate) const MIDGAME_SELECTIVITY: Selectivity = Selectivity::Level1;
pub(crate) const MIN_MID_DEPTH: u8 = 1;
pub(crate) const MAX_MID_DEPTH: u8 = 24;
