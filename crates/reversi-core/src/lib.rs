#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_i8mm))]
#![feature(hint_prefetch)]
#![feature(sync_unsafe_cell)]

pub mod analysis;
pub mod bitboard;
pub mod board;
pub mod constants;
pub mod disc;
pub mod empty_list;
pub mod eval;
pub mod flip;
pub mod game_state;
pub mod legal_moves;
pub mod level;
pub mod move_list;
pub mod obf;
pub mod perft;
pub mod player_color;
pub mod probcut;
pub mod score;
pub mod search;
pub mod square;
pub mod stability;
pub mod transposition_table;
pub mod types;
mod util;
