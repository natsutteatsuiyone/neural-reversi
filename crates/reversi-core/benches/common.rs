#![allow(dead_code)]

// Each Criterion target is compiled as a separate crate and imports a subset of these helpers.

use std::hint::black_box;
use std::path::PathBuf;

use criterion::{BenchmarkGroup, BenchmarkId, measurement::WallTime};
use rand::{RngExt, rngs::StdRng};

use reversi_core::bitboard::Bitboard;
use reversi_core::board::Board;
use reversi_core::constants::INITIAL_EMPTY_COUNT;
use reversi_core::disc::Disc;
use reversi_core::square::Square;

pub(crate) fn board_from_rows(rows: [&str; 8]) -> Board {
    let mut board_string = String::with_capacity(64);
    for row in rows {
        board_string.push_str(row);
    }
    Board::from_string(&board_string, Disc::Black).expect("benchmark board must parse")
}

pub(crate) fn weights_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(file_name)
}

pub(crate) fn board_ply(board: &Board) -> usize {
    INITIAL_EMPTY_COUNT - board.get_empty_count() as usize
}

pub(crate) fn random_square(rng: &mut StdRng) -> Square {
    let square_index: u32 = rng.random_range(0..64);
    Square::from_u32(square_index).expect("random square index must be in 0..64")
}

pub(crate) fn random_disjoint_bitboards(rng: &mut StdRng) -> (u64, u64) {
    let player: u64 = rng.random();
    let opponent: u64 = rng.random::<u64>() & !player;
    (player, opponent)
}

pub(crate) fn choose_square(squares: Bitboard, rng: &mut StdRng) -> Square {
    let index = rng.random_range(0..squares.count()) as usize;
    squares
        .iter()
        .nth(index)
        .expect("bitboard must contain the chosen square index")
}

pub(crate) fn add_random_bit(rng: &mut StdRng, occupied: &mut u64) -> u64 {
    loop {
        debug_assert_ne!(*occupied, u64::MAX);
        let bit = 1u64 << rng.random_range(0..64);
        if (*occupied & bit) == 0 {
            *occupied |= bit;
            return bit;
        }
    }
}

pub(crate) fn bench_case_set<T, R, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    cases: &[T],
    checksum: F,
) where
    F: Fn(&[T]) -> R,
{
    group.bench_with_input(BenchmarkId::from_parameter(name), cases, move |b, cases| {
        b.iter(|| black_box(checksum(black_box(cases))))
    });
}
