use reversi_core::probcut::Selectivity;
use reversi_core::types::Depth;

pub(crate) const DEFAULT_TT_MB: usize = 128;
pub(crate) const DEFAULT_MID_DEPTH: Depth = 12;
pub(crate) const MIDGAME_SELECTIVITY: Selectivity = Selectivity::Level1;
pub(crate) const MIN_MID_DEPTH: u8 = 1;
pub(crate) const MAX_MID_DEPTH: u8 = 24;
