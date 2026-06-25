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
/// Upper bound on the indexed cells in the union of a square's two diagonals,
/// excluding the move square itself.
const DIAG_UNION_MAX_BITS: usize = 13;
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

// Union of both diagonal lines through a square.
const fn diag_union_mask(sq: usize) -> u64 {
    diag0_mask(sq) | diag1_mask(sq)
}

// The move square bit does not affect `line_count`, so the solve1 union table
// omits it and halves the number of indexed occupancies.
const fn diag_union_index_mask(sq: usize) -> u64 {
    diag_union_mask(sq) & !(1u64 << sq)
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
    rank_shift: u8,
    _pad: u16,
}

const _: () = assert!(core::mem::size_of::<SquareMeta>() == 8);

/// Combined table for the two diagonal lines through a square.
///
/// `solve1` needs both the current-player diagonal count and, in the pass
/// continuation, the opponent diagonal count for the same square. Extracting
/// the union of both diagonals with one PEXT replaces two diagonal PEXTs and
/// two small COUNT_FLIP loads with one PEXT and one per-square table load.
///
/// `mask` excludes the move square bit because that bit is ignored by
/// [`line_count`]. `offset` locates the square's slice in `DIAG_UNION_COUNT`.
/// `index_flip_mask` is `(1 << indexed_cells) - 1`; XORing the union index with
/// it complements every indexed cell, yielding the opponent's index for the
/// pass continuation.
#[derive(Copy, Clone)]
struct DiagUnionTable {
    mask: u64,
    offset: u32,
    index_flip_mask: u16,
}

const _: () = assert!(core::mem::size_of::<DiagUnionTable>() == 16);

/// Per-square metadata used only by `solve1`.
///
/// This keeps the straight-line DU13 algorithm, but avoids loading
/// `SQUARE_TABLE`, `SQUARE_META`, and `DIAG_UNION_TABLE` separately in the
/// inlined solve1 hot path. The record is exactly 32 bytes, so two squares fit
/// in one cache line and the extra static data is only 1 KiB over the uploaded
/// DU13 implementation.
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

const fn diag_union_count_entries() -> usize {
    let mut total = 0usize;
    let mut sq = 0usize;

    while sq < 64 {
        total += 1usize << diag_union_index_mask(sq).count_ones();
        sq += 1;
    }

    total
}

const DIAG_UNION_COUNT_LEN: usize = diag_union_count_entries();

const fn build_diag_union_table() -> [DiagUnionTable; 64] {
    let zero = DiagUnionTable {
        mask: 0,
        offset: 0,
        index_flip_mask: 0,
    };
    let mut out = [zero; 64];
    let mut offset = 0usize;
    let mut sq = 0usize;

    while sq < 64 {
        let mask = diag_union_index_mask(sq);
        let len = mask.count_ones() as usize;
        out[sq] = DiagUnionTable {
            mask,
            offset: offset as u32,
            index_flip_mask: ((1usize << len) - 1) as u16,
        };
        offset += 1usize << len;
        sq += 1;
    }

    out
}

/// Precomputes `DIAG_UNION_COUNT`: for every square and every occupancy of its
/// indexed diagonal union, the combined flip count of both diagonals. Union
/// occupancies are enumerated as a plain counter; each increment toggles only
/// the changed bits, so the two diagonal sub-indices are maintained
/// incrementally instead of being recomputed per entry.
const fn build_diag_union_count() -> [u8; DIAG_UNION_COUNT_LEN] {
    let mut out = [0u8; DIAG_UNION_COUNT_LEN];
    let mut sq = 0usize;

    while sq < 64 {
        let diag0 = diag0_mask(sq);
        let diag1 = diag1_mask(sq);
        let mask = diag_union_index_mask(sq);
        let offset = DIAG_UNION_TABLE_RAW[sq].offset as usize;
        let (diag0_masks, diag1_masks, len) = diag_union_index_masks(mask, diag0, diag1);
        let end = 1usize << len;
        let mut idx = 0usize;
        let mut diag0_idx = 0usize;
        let mut diag1_idx = 0usize;

        while idx < end {
            out[offset + idx] = COUNT_FLIP_RAW[(diag0_pos(sq) << 8) | diag0_idx]
                + COUNT_FLIP_RAW[(diag1_pos(sq) << 8) | diag1_idx];
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

const fn build_square_meta() -> [SquareMeta; 64] {
    let zero = SquareMeta {
        count_file_offset: 0,
        count_rank_offset: 0,
        rank_shift: 0,
        _pad: 0,
    };
    let mut out = [zero; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let rank_shift = sq & 0x38;
        out[sq] = SquareMeta {
            count_file_offset: ((sq & 7) << 8) as u16,
            count_rank_offset: (rank_shift << 5) as u16,
            rank_shift: rank_shift as u8,
            _pad: 0,
        };
        sq += 1;
    }
    out
}

const fn build_solve1_table(
    square_table: &[SquareTable; 64],
    square_meta: &[SquareMeta; 64],
    diag_table: &[DiagUnionTable; 64],
) -> [Solve1Table; 64] {
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
        out[sq] = Solve1Table {
            mask_file: square_table[sq].mask_file,
            diag_mask: diag_table[sq].mask,
            diag_offset: diag_table[sq].offset,
            diag_index_flip_mask: diag_table[sq].index_flip_mask,
            count_file_offset: square_meta[sq].count_file_offset,
            count_rank_offset: square_meta[sq].count_rank_offset,
            rank_shift: square_meta[sq].rank_shift,
            _pad: [0; 5],
        };
        sq += 1;
    }

    out
}

const COUNT_FLIP_RAW: [u8; COUNT_FLIP_LEN] = build_count_flip();
const SQUARE_TABLE_RAW: [SquareTable; 64] = build_square_table();
const SQUARE_META_RAW: [SquareMeta; 64] = build_square_meta();
const DIAG_UNION_TABLE_RAW: [DiagUnionTable; 64] = build_diag_union_table();
const SOLVE1_TABLE_RAW: [Solve1Table; 64] =
    build_solve1_table(&SQUARE_TABLE_RAW, &SQUARE_META_RAW, &DIAG_UNION_TABLE_RAW);

static COUNT_FLIP: Align64<[u8; COUNT_FLIP_LEN]> = Align64(COUNT_FLIP_RAW);
static SQUARE_TABLE: Align64<[SquareTable; 64]> = Align64(SQUARE_TABLE_RAW);
static SQUARE_META: Align64<[SquareMeta; 64]> = Align64(SQUARE_META_RAW);
static SOLVE1_TABLE: Align64<[Solve1Table; 64]> = Align64(SOLVE1_TABLE_RAW);
static DIAG_UNION_COUNT: Align64<[u8; DIAG_UNION_COUNT_LEN]> = Align64(build_diag_union_count());

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

        let entry = &*SOLVE1_TABLE.0.as_ptr().add(sq_idx);
        let row_idx = ((player >> (entry.rank_shift as usize)) & 0xff) as usize;
        let file_idx = _pext_u64(player, entry.mask_file) as usize;
        let diag_idx = _pext_u64(player, entry.diag_mask) as usize;

        let count_base = COUNT_FLIP.0.as_ptr();
        let count_file_row = count_base.add(entry.count_file_offset as usize);
        let count_rank_row = count_base.add(entry.count_rank_offset as usize);
        let diag_count_row = DIAG_UNION_COUNT.0.as_ptr().add(entry.diag_offset as usize);

        let n_flipped = (*count_file_row.add(row_idx) as u32
            + *count_rank_row.add(file_idx) as u32
            + *diag_count_row.add(diag_idx) as u32) as i32;
        let score_base = 2 * player.count_ones() as Score - SCORE_MAX + 2;
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
