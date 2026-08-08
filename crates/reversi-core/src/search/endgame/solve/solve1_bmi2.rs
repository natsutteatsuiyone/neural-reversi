//! BMI2 last-empty-square scoring.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/count_last_flip_bmi2.c>

use std::arch::x86_64::_pext_u64;

use crate::constants::SCORE_MAX;
use crate::square::Square;
use crate::types::Score;
use crate::util::align::Align64;

use super::solve1_tables::{COUNT_FLIP, COUNT_FLIP_RAW};

/// Upper bound on the indexed cells in the union of a square's two diagonals,
/// excluding the move square itself.
const DIAG_UNION_MAX_BITS: usize = 13;
const FILE_A: u64 = 0x0101_0101_0101_0101;

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

// The move square bit does not affect line counts, so the solve1 union table
// omits it and halves the number of indexed occupancies.
const fn diag_union_index_mask(sq: usize) -> u64 {
    (diag0_mask(sq) | diag1_mask(sq)) & !(1u64 << sq)
}

const fn diag_union_entry_count(sq: usize) -> usize {
    1usize << diag_union_index_mask(sq).count_ones() as usize
}

const fn diag_union_offset(sq: usize) -> usize {
    let mut offset = 0usize;
    let mut prev_sq = 0usize;

    while prev_sq < sq {
        offset += diag_union_entry_count(prev_sq);
        prev_sq += 1;
    }

    offset
}

/// Per-square metadata used only by `solve1`.
///
/// This keeps the straight-line DU13 algorithm, but avoids loading
/// multiple per-square tables in the inlined solve1 hot path. The record is
/// exactly 32 bytes, so two squares fit in one cache line.
#[repr(C, align(32))]
#[derive(Copy, Clone)]
struct Solve1Table {
    mask_file: u64,
    diag_mask: u64,
    diag_offset: u32,
    diag_index_flip_mask: u16,
    count_file_offset: u16,
    count_rank_offset: u16,
    rank_shift: u8,
    _pad: [u8; 5],
}

const _: () = assert!(core::mem::size_of::<Solve1Table>() == 32);

const fn build_solve1_table() -> [Solve1Table; 64] {
    let zero = Solve1Table {
        mask_file: 0,
        diag_mask: 0,
        diag_offset: 0,
        diag_index_flip_mask: 0,
        count_file_offset: 0,
        count_rank_offset: 0,
        rank_shift: 0,
        _pad: [0; 5],
    };
    let mut out = [zero; 64];
    let mut sq = 0usize;

    while sq < 64 {
        let rank_shift = sq & 0x38;
        out[sq] = Solve1Table {
            mask_file: FILE_A << (sq & 7),
            diag_mask: diag_union_index_mask(sq),
            diag_offset: diag_union_offset(sq) as u32,
            diag_index_flip_mask: (diag_union_entry_count(sq) - 1) as u16,
            count_file_offset: ((sq & 7) << 8) as u16,
            count_rank_offset: (rank_shift << 5) as u16,
            rank_shift: rank_shift as u8,
            _pad: [0; 5],
        };
        sq += 1;
    }

    out
}

/// Maps each union cell (low to high bit of `mask`) to the `diag0` / `diag1`
/// sub-index bit it occupies, leaving zero where the cell is not on that
/// diagonal. Returns both per-cell mask tables and the union cell count.
const fn diag_union_index_masks(
    index_mask: u64,
    diag0: u64,
    diag1: u64,
) -> (
    [usize; DIAG_UNION_MAX_BITS],
    [usize; DIAG_UNION_MAX_BITS],
    usize,
) {
    let mut diag0_masks = [0usize; DIAG_UNION_MAX_BITS];
    let mut diag1_masks = [0usize; DIAG_UNION_MAX_BITS];
    let mut union_bits = diag0 | diag1;
    let mut index_pos = 0usize;
    let mut diag0_pos = 0usize;
    let mut diag1_pos = 0usize;

    while union_bits != 0 {
        let bit = 1u64 << union_bits.trailing_zeros();
        let mut diag0_index_mask = 0usize;
        let mut diag1_index_mask = 0usize;

        if (diag0 & bit) != 0 {
            diag0_index_mask = 1usize << diag0_pos;
            diag0_pos += 1;
        }
        if (diag1 & bit) != 0 {
            diag1_index_mask = 1usize << diag1_pos;
            diag1_pos += 1;
        }
        if (index_mask & bit) != 0 {
            diag0_masks[index_pos] = diag0_index_mask;
            diag1_masks[index_pos] = diag1_index_mask;
            index_pos += 1;
        }
        union_bits &= union_bits - 1;
    }

    (diag0_masks, diag1_masks, index_pos)
}

const DIAG_UNION_COUNT_LEN: usize = {
    let mut total = 0usize;
    let mut sq = 0usize;

    while sq < 64 {
        total += diag_union_entry_count(sq);
        sq += 1;
    }

    total
};

/// Precomputes `DIAG_UNION_COUNT`: for every square and every occupancy of its
/// indexed diagonal union, the combined flip count of both diagonals. Union
/// occupancies are enumerated as a plain counter; each increment toggles only
/// the changed bits, so the two diagonal sub-indices are maintained
/// incrementally instead of being recomputed per entry.
const fn build_diag_union_count() -> [u8; DIAG_UNION_COUNT_LEN] {
    let mut out = [0u8; DIAG_UNION_COUNT_LEN];
    let mut sq = 0usize;

    while sq < 64 {
        let x = sq & 7;
        let y = sq >> 3;
        let diag0_count_pos = if 7 - x < y { 7 - x } else { y };
        let diag1_count_pos = if x < y { x } else { y };
        let diag0 = diag0_mask(sq);
        let diag1 = diag1_mask(sq);
        let mask = diag_union_index_mask(sq);
        let offset = diag_union_offset(sq);
        let (diag0_masks, diag1_masks, len) = diag_union_index_masks(mask, diag0, diag1);
        let end = 1usize << len;
        let mut idx = 0usize;
        let mut diag0_idx = 0usize;
        let mut diag1_idx = 0usize;

        while idx < end {
            out[offset + idx] = COUNT_FLIP_RAW[diag0_count_pos][diag0_idx]
                + COUNT_FLIP_RAW[diag1_count_pos][diag1_idx];
            let next = idx + 1;
            if next < end {
                let mut changed = idx ^ next;
                while changed != 0 {
                    let bit_pos = changed.trailing_zeros() as usize;
                    diag0_idx ^= diag0_masks[bit_pos];
                    diag1_idx ^= diag1_masks[bit_pos];
                    changed &= changed - 1;
                }
            }
            idx = next;
        }

        sq += 1;
    }

    out
}

const SOLVE1_TABLE_RAW: [Solve1Table; 64] = build_solve1_table();

static SOLVE1_TABLE: Align64<[Solve1Table; 64]> = Align64(SOLVE1_TABLE_RAW);
static DIAG_UNION_COUNT: Align64<[u8; DIAG_UNION_COUNT_LEN]> = Align64(build_diag_union_count());

/// Scores a position with exactly one empty square.
#[inline(always)]
pub(super) fn solve1(player: u64, alpha: Score, sq: Square) -> Score {
    let sq_idx = sq.index();
    // Start POPCNT before the independent table-address work.
    let score_base = 2 * player.count_ones() as Score - SCORE_MAX + 2;

    // SAFETY: `solve1` is called with a real board square. This module is
    // compiled only when BMI2 is enabled, and all table offsets/indices come
    // from the same const-generated masks.
    unsafe {
        let entry = &*SOLVE1_TABLE.0.as_ptr().add(sq_idx);
        let row_idx = ((player >> (entry.rank_shift as usize)) & 0xff) as usize;

        let count_base = COUNT_FLIP.0[0].as_ptr();
        let count_file_row = count_base.add(entry.count_file_offset as usize);
        // Let the row lookup overlap both PEXT-dependent lookups.
        let row_count = *count_file_row.add(row_idx) as u32;

        let file_idx = _pext_u64(player, entry.mask_file) as usize;
        let diag_idx = _pext_u64(player, entry.diag_mask) as usize;
        let count_rank_row = count_base.add(entry.count_rank_offset as usize);
        let diag_count_row = DIAG_UNION_COUNT.0.as_ptr().add(entry.diag_offset as usize);

        let n_flipped = (row_count
            + *count_rank_row.add(file_idx) as u32
            + *diag_count_row.add(diag_idx) as u32) as i32;
        let score = score_base + n_flipped;

        if n_flipped != 0 {
            return score;
        }

        let score_if_opp_passes = if score_base > 0 {
            score_base
        } else {
            score_base - 2
        };
        if score_if_opp_passes <= alpha {
            score_if_opp_passes
        } else {
            let opp_row_idx = row_idx ^ 0xff;
            let opp_file_idx = file_idx ^ 0xff;
            let opp_diag_idx = diag_idx ^ entry.diag_index_flip_mask as usize;
            let opp_n_flipped = (*count_file_row.add(opp_row_idx) as u32
                + *count_rank_row.add(opp_file_idx) as u32
                + *diag_count_row.add(opp_diag_idx) as u32) as i32;

            if opp_n_flipped > 0 {
                score_base - 2 - opp_n_flipped
            } else {
                score_if_opp_passes
            }
        }
    }
}
