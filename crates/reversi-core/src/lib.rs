#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_i8mm))]
#![feature(hint_prefetch)]

pub mod bitboard;
pub mod board;
pub mod constants;
pub mod count_last_flip;
pub mod disc;
pub mod empty_list;
pub mod eval;
pub mod flip;
pub mod game_state;
pub mod level;
pub mod move_list;
pub mod obf;
pub mod perft;
pub mod probcut;
pub mod search;
pub mod square;
pub mod stability;
pub mod transposition_table;
pub mod types;
mod util;
