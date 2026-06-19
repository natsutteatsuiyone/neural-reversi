//! BMI2 version of counting flipped discs for the last move.
//!
//! Rust port of the Edax PEXT layout, with per-square masks built at compile
//! time instead of hand-written.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/count_last_flip_bmi2.c>

use std::arch::x86_64::_pext_u64;

use crate::constants::SCORE_MAX;
use crate::square::Square;
use crate::types::Score;
use crate::util::align::Align64;

const COUNT_FLIP_LEN: usize = 8 * 256;
const FILE_A: u64 = 0x0101_0101_0101_0101;
const RANK_1: u64 = 0x0000_0000_0000_00ff;

const fn line_count(pos: usize, bits: usize) -> u8 {
    let mut n = 0usize;

    // Lower-index side: nearest own disc brackets every intervening opponent disc.
    let mut i = pos;
    while i > 0 {
        i -= 1;
        if ((bits >> i) & 1) != 0 {
            n += pos - i - 1;
            break;
        }
    }

    // Higher-index side.
    i = pos + 1;
    while i < 8 {
        if ((bits >> i) & 1) != 0 {
            n += i - pos - 1;
            break;
        }
        i += 1;
    }

    (n << 1) as u8
}

const fn build_count_flip() -> [u8; COUNT_FLIP_LEN] {
    let mut out = [0u8; COUNT_FLIP_LEN];
    let mut pos = 0usize;
    while pos < 8 {
        let mut bits = 0usize;
        while bits < 256 {
            out[(pos << 8) | bits] = line_count(pos, bits);
            bits += 1;
        }
        pos += 1;
    }
    out
}

// Anti-diagonal, ascending in bit-index order: H1..A8 for the longest line.
const fn diag0_mask(sq: usize) -> u64 {
    let mut x = sq & 7;
    let mut y = sq >> 3;

    while x < 7 && y > 0 {
        x += 1;
        y -= 1;
    }

    let mut mask = 0u64;
    loop {
        mask |= 1u64 << ((y << 3) | x);
        if x == 0 || y == 7 {
            break;
        }
        x -= 1;
        y += 1;
    }
    mask
}

// Main diagonal, ascending in bit-index order: A1..H8 for the longest line.
const fn diag1_mask(sq: usize) -> u64 {
    let mut x = sq & 7;
    let mut y = sq >> 3;

    while x > 0 && y > 0 {
        x -= 1;
        y -= 1;
    }

    let mut mask = 0u64;
    loop {
        mask |= 1u64 << ((y << 3) | x);
        if x == 7 || y == 7 {
            break;
        }
        x += 1;
        y += 1;
    }
    mask
}

const fn diag0_pos(sq: usize) -> usize {
    let x = sq & 7;
    let y = sq >> 3;
    if 7 - x < y { 7 - x } else { y }
}

const fn diag1_pos(sq: usize) -> usize {
    let x = sq & 7;
    let y = sq >> 3;
    if x < y { x } else { y }
}

const fn rank_mask(sq: usize) -> u64 {
    RANK_1 << (sq & 0x38)
}

const fn file_mask(sq: usize) -> u64 {
    FILE_A << (sq & 7)
}

/// Sets up to `pad_count` extra 1-bits at positions below the lowest bit of
/// `mask`, skipping bits already covered by `union`. The padded mask shifts
/// the diagonal so that PEXT yields a value whose bit indices match the
/// square's rank, letting all four directions share `COUNT_FLIP`'s rank row.
const fn pad_before_line(mask: u64, union: u64, pad_count: usize) -> u64 {
    let first = mask.trailing_zeros() as usize;
    let mut out = mask;
    let mut bit = 0usize;
    let mut padded = 0usize;

    while bit < first && padded < pad_count {
        let bit_mask = 1u64 << bit;
        if (union & bit_mask) == 0 {
            out |= bit_mask;
            padded += 1;
        }
        bit += 1;
    }

    out
}

/// Per-square hot-path masks for the last-flip count.
///
/// `mask_diag{0,1}` includes padding bits before the real diagonal line, which
/// makes the square position match its board rank. `mask_union` excludes those
/// padding bits, so `player & mask_union` feeds zeroes into the padded PEXT
/// lanes and all four directions can use the same rank-indexed `COUNT_FLIP`
/// row.
#[derive(Copy, Clone)]
struct SquareTable {
    mask_diag0: u64,
    mask_diag1: u64,
    mask_file: u64,
    mask_union: u64,
}

const _: () = assert!(core::mem::size_of::<SquareTable>() == 32);

/// Per-square hot-path offsets, kept separate from [`SquareTable`] to preserve
/// its 32-byte layout.
#[derive(Copy, Clone)]
struct SquareMeta {
    count_file_offset: u16,
    count_rank_offset: u16,
    flip_mask: u16,
    rank_shift: u8,
}

const _: () = assert!(core::mem::size_of::<SquareMeta>() == 8);

const fn pext_u8(value: u64, mut mask: u64) -> u8 {
    let mut out = 0usize;
    let mut out_bit = 1usize;

    while mask != 0 {
        let bit = mask.isolate_lowest_one();
        if (value & bit) != 0 {
            out |= out_bit;
        }
        mask &= mask - 1;
        out_bit <<= 1;
    }

    out as u8
}

const fn build_square_table() -> [SquareTable; 64] {
    let zero = SquareTable {
        mask_diag0: 0,
        mask_diag1: 0,
        mask_file: 0,
        mask_union: 0,
    };
    let mut out = [zero; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let mask_diag0 = diag0_mask(sq);
        let mask_diag1 = diag1_mask(sq);
        let mask_file = file_mask(sq);
        let mask_union = rank_mask(sq) | mask_file | mask_diag0 | mask_diag1;
        let rank = sq >> 3;
        let padded_diag0 = pad_before_line(mask_diag0, mask_union, rank - diag0_pos(sq));
        let padded_diag1 = pad_before_line(mask_diag1, mask_union, rank - diag1_pos(sq));

        out[sq] = SquareTable {
            mask_diag0: padded_diag0,
            mask_diag1: padded_diag1,
            mask_file,
            mask_union,
        };
        sq += 1;
    }
    out
}

const fn build_square_meta(table: &[SquareTable; 64]) -> [SquareMeta; 64] {
    let zero = SquareMeta {
        count_file_offset: 0,
        count_rank_offset: 0,
        flip_mask: 0,
        rank_shift: 0,
    };
    let mut out = [zero; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let entry = &table[sq];
        let rank_shift = sq & 0x38;
        let diag0_flip_mask = pext_u8(entry.mask_union, entry.mask_diag0) as u16;
        let diag1_flip_mask = pext_u8(entry.mask_union, entry.mask_diag1) as u16;
        out[sq] = SquareMeta {
            count_file_offset: ((sq & 7) << 8) as u16,
            count_rank_offset: (rank_shift << 5) as u16,
            flip_mask: diag0_flip_mask | (diag1_flip_mask << 8),
            rank_shift: rank_shift as u8,
        };
        sq += 1;
    }
    out
}

const COUNT_FLIP_RAW: [u8; COUNT_FLIP_LEN] = build_count_flip();
const SQUARE_TABLE_RAW: [SquareTable; 64] = build_square_table();
const SQUARE_META_RAW: [SquareMeta; 64] = build_square_meta(&SQUARE_TABLE_RAW);

static COUNT_FLIP: Align64<[u8; COUNT_FLIP_LEN]> = Align64(COUNT_FLIP_RAW);
static SQUARE_TABLE: Align64<[SquareTable; 64]> = Align64(SQUARE_TABLE_RAW);
static SQUARE_META: Align64<[SquareMeta; 64]> = Align64(SQUARE_META_RAW);

/// Counts the number of discs that would be flipped by the last move.
///
/// Returns twice the actual number of flipped discs for optimization purposes.
#[inline(always)]
pub(super) fn count_last_flip(player: u64, sq: Square) -> i32 {
    unsafe {
        let sq_idx = sq.index();
        debug_assert!(sq_idx < 64);

        let entry = &*SQUARE_TABLE.0.as_ptr().add(sq_idx);
        let meta = &*SQUARE_META.0.as_ptr().add(sq_idx);
        let mask_diag0 = entry.mask_diag0;
        let mask_diag1 = entry.mask_diag1;
        let mask_file = entry.mask_file;
        let mask_union = entry.mask_union;

        let rank_shift = meta.rank_shift as usize;
        let row_idx = ((player >> rank_shift) & 0xff) as usize;
        let file_idx = _pext_u64(player, mask_file) as usize;
        let masked = player & mask_union;
        let diag1_idx = _pext_u64(masked, mask_diag1) as usize;
        let diag0_idx = _pext_u64(masked, mask_diag0) as usize;

        let count_base = COUNT_FLIP.0.as_ptr();
        let count_file_row = count_base.add(meta.count_file_offset as usize);
        let count_rank_row = count_base.add(meta.count_rank_offset as usize);
        let row_count = *count_file_row.add(row_idx) as u32;

        (row_count
            + *count_rank_row.add(diag0_idx) as u32
            + *count_rank_row.add(diag1_idx) as u32
            + *count_rank_row.add(file_idx) as u32) as i32
    }
}

/// Scores a position with exactly one empty square.
#[inline(always)]
pub(super) fn solve1(player: u64, alpha: Score, sq: Square) -> Score {
    unsafe {
        let sq_idx = sq.index();
        debug_assert!(sq_idx < 64);

        let player_count = player.count_ones() as Score;
        let entry = &*SQUARE_TABLE.0.as_ptr().add(sq_idx);
        let meta = &*SQUARE_META.0.as_ptr().add(sq_idx);
        let mask_diag0 = entry.mask_diag0;
        let mask_diag1 = entry.mask_diag1;
        let mask_file = entry.mask_file;
        let mask_union = entry.mask_union;

        let rank_shift = meta.rank_shift as usize;
        let row_idx = ((player >> rank_shift) & 0xff) as usize;
        let file_idx = _pext_u64(player, mask_file) as usize;
        let masked = player & mask_union;
        let diag1_idx = _pext_u64(masked, mask_diag1) as usize;
        let diag0_idx = _pext_u64(masked, mask_diag0) as usize;

        let count_base = COUNT_FLIP.0.as_ptr();
        let count_file_row = count_base.add(meta.count_file_offset as usize);
        let count_rank_row = count_base.add(meta.count_rank_offset as usize);
        let row_count = *count_file_row.add(row_idx) as u32;
        let n_flipped = (row_count
            + *count_rank_row.add(diag0_idx) as u32
            + *count_rank_row.add(diag1_idx) as u32
            + *count_rank_row.add(file_idx) as u32) as i32;
        let score = 2 * player_count - SCORE_MAX + 2 + n_flipped;

        if n_flipped != 0 {
            return score;
        }

        let score_if_opp_passes = if score > 0 { score } else { score - 2 };
        if score_if_opp_passes <= alpha {
            score_if_opp_passes
        } else {
            let flip_masks = meta.flip_mask as usize;
            let opp_row_idx = row_idx ^ 0xff;
            let opp_file_idx = file_idx ^ 0xff;
            let opp_diag0_idx = diag0_idx ^ (flip_masks & 0xff);
            let opp_diag1_idx = diag1_idx ^ (flip_masks >> 8);
            let opp_row_count = *count_file_row.add(opp_row_idx) as u32;
            let opp_n_flipped = (opp_row_count
                + *count_rank_row.add(opp_diag0_idx) as u32
                + *count_rank_row.add(opp_diag1_idx) as u32
                + *count_rank_row.add(opp_file_idx) as u32) as i32;

            if opp_n_flipped > 0 {
                score - 2 - opp_n_flipped
            } else {
                score_if_opp_passes
            }
        }
    }
}
