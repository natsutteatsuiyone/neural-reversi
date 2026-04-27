//! Portable scalar last-flip count: a single data-driven function that
//! eliminates the 64-way function-pointer dispatch used by the per-square
//! kindergarten unrolling. Math is the kindergarten magic-multiply (no SIMD,
//! no platform intrinsics); the win is keeping `sq` as a runtime index into
//! a parameter table instead of branching to one of 64 specialized fns.
//!
//! The original kindergarten code has three sub-patterns per square (simple
//! diagonal, addend-and-mask diagonal, two separate diagonals). All three
//! are handled by a single uniform formula (see `count_one`).
//!
//! Squares that don't need the second diagonal set `mask_d9 = mult_d9 = 0`,
//! which makes `idx3 = 0` and `COUNT_FLIP[t3][0] = 0` (always true for the
//! kindergarten table), contributing nothing to the sum. The fourth load
//! happens but overlaps with the other three in flight.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/count_last_flip_kindergarten.c>

use crate::square::Square;

#[rustfmt::skip]
static COUNT_FLIP: [[i8; 256]; 8] = [
    [
         0,  0,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
        10, 10,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
        12, 12,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
        10, 10,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
         8,  8,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,  6,  6,  0,  0,  2,  2,  0,  0,  4,  4,  0,  0,  2,  2,  0,  0,
    ],
    [
         0,  0,  0,  0,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         8,  8,  8,  8,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
        10, 10, 10, 10,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         8,  8,  8,  8,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
         6,  6,  6,  6,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,  4,  4,  4,  4,  0,  0,  0,  0,  2,  2,  2,  2,  0,  0,  0,  0,
    ],
    [
         0,  2,  0,  0,  0,  2,  0,  0,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         6,  8,  6,  6,  6,  8,  6,  6,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         8, 10,  8,  8,  8, 10,  8,  8,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         6,  8,  6,  6,  6,  8,  6,  6,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
         4,  6,  4,  4,  4,  6,  4,  4,  0,  2,  0,  0,  0,  2,  0,  0,  2,  4,  2,  2,  2,  4,  2,  2,  0,  2,  0,  0,  0,  2,  0,  0,
    ],
    [
         0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         4,  8,  6,  6,  4,  4,  4,  4,  4,  8,  6,  6,  4,  4,  4,  4,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         6, 10,  8,  8,  6,  6,  6,  6,  6, 10,  8,  8,  6,  6,  6,  6,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         4,  8,  6,  6,  4,  4,  4,  4,  4,  8,  6,  6,  4,  4,  4,  4,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
         2,  6,  4,  4,  2,  2,  2,  2,  2,  6,  4,  4,  2,  2,  2,  2,  0,  4,  2,  2,  0,  0,  0,  0,  0,  4,  2,  2,  0,  0,  0,  0,
    ],
    [
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         4, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  4, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
         2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  6,  4,  4,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
    [
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         2, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         2, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  8,  6,  6,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
    [
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 10,  8,  8,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
    [
         0, 12, 10, 10,  8,  8,  8,  8,  6,  6,  6,  6,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, 12, 10, 10,  8,  8,  8,  8,  6,  6,  6,  6,  6,  6,  6,  6,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    ],
];

#[derive(Clone, Copy)]
struct SqParams {
    mask_v: u64,
    mult_v: u64,
    mask_d7: u64,
    addend7: u64,
    post_mask7: u64,
    mult_d7: u64,
    mask_d9: u64,
    mult_d9: u64,
    row_shift: u8,
    t0: u8,
    t1: u8,
    t2: u8,
    t3: u8,
}

#[rustfmt::skip]
const PARAMS: [SqParams; 64] = [
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 0, t2: 0, t3: 0 }, // a1
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0080402010080402, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 1, t2: 1, t3: 0 }, // b1
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0000804020110a04, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 2, t2: 2, t3: 0 }, // c1
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0000008041221408, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 3, t2: 3, t3: 0 }, // d1
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0000000182442810, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 4, t2: 4, t3: 0 }, // e1
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0000010204885020, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 5, t2: 5, t3: 0 }, // f1
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 6, t2: 6, t3: 0 }, // g1
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 7, t2: 7, t3: 0 }, // h1
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x4020100804020100, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 0, t2: 0, t3: 0 }, // a2
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 1, t2: 1, t3: 0 }, // b2
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x00804020110a0400, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 2, t2: 2, t3: 0 }, // c2
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0000804122140800, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 3, t2: 3, t3: 0 }, // d2
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0000018244281000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 4, t2: 4, t3: 0 }, // e2
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0001020488502000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 5, t2: 5, t3: 0 }, // f2
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 6, t2: 6, t3: 0 }, // g2
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 7, t2: 7, t3: 0 }, // h2
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x2010080402010204, addend7: 0x6070787c7e7f7e7c, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 0, t2: 2, t3: 0 }, // a3
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x4020100804020408, addend7: 0x406070787c7e7c78, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 1, t2: 2, t3: 0 }, // b3
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0000000102040810, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, mult_d9: 0x0101010101010101, row_shift: 16, t0: 2, t1: 2, t2: 2, t3: 2 }, // c3
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0000010204081020, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0080402010080402, mult_d9: 0x0101010101010101, row_shift: 16, t0: 2, t1: 3, t2: 3, t3: 3 }, // d3
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000804020100804, mult_d9: 0x0101010101010101, row_shift: 16, t0: 2, t1: 4, t2: 4, t3: 4 }, // e3
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000008040201008, mult_d9: 0x0101010101010101, row_shift: 16, t0: 2, t1: 5, t2: 5, t3: 5 }, // f3
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0204081020402010, addend7: 0x7e7c787060406070, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 6, t2: 2, t3: 0 }, // g3
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0408102040804020, addend7: 0x7c78706040004060, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 7, t2: 2, t3: 0 }, // h3
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x1008040201020408, addend7: 0x70787c7e7f7e7c78, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 0, t2: 3, t3: 0 }, // a4
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x2010080402040810, addend7: 0x6070787c7e7c7870, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 1, t2: 3, t3: 0 }, // b4
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0000010204081020, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x4020100804020100, mult_d9: 0x0101010101010101, row_shift: 24, t0: 3, t1: 2, t2: 2, t3: 2 }, // c4
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, mult_d9: 0x0101010101010101, row_shift: 24, t0: 3, t1: 3, t2: 3, t3: 3 }, // d4
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0080402010080402, mult_d9: 0x0101010101010101, row_shift: 24, t0: 3, t1: 4, t2: 4, t3: 4 }, // e4
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000804020100804, mult_d9: 0x0101010101010101, row_shift: 24, t0: 3, t1: 5, t2: 5, t3: 5 }, // f4
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0408102040201008, addend7: 0x7c78706040607078, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 6, t2: 3, t3: 0 }, // g4
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0810204080402010, addend7: 0x7870604000406070, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 7, t2: 3, t3: 0 }, // h4
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0804020102040810, addend7: 0x787c7e7f7e7c7870, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 0, t2: 4, t3: 0 }, // a5
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x1008040204081020, addend7: 0x70787c7e7c787060, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 1, t2: 4, t3: 0 }, // b5
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x2010080402010000, mult_d9: 0x0101010101010101, row_shift: 32, t0: 4, t1: 2, t2: 2, t3: 2 }, // c5
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x4020100804020100, mult_d9: 0x0101010101010101, row_shift: 32, t0: 4, t1: 3, t2: 3, t3: 3 }, // d5
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, mult_d9: 0x0101010101010101, row_shift: 32, t0: 4, t1: 4, t2: 4, t3: 4 }, // e5
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0408102040800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0080402010080402, mult_d9: 0x0101010101010101, row_shift: 32, t0: 4, t1: 5, t2: 5, t3: 5 }, // f5
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0810204020100804, addend7: 0x787060406070787c, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 6, t2: 4, t3: 0 }, // g5
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x1020408040201008, addend7: 0x7060400040607078, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 7, t2: 4, t3: 0 }, // h5
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0402010204081020, addend7: 0x7c7e7f7e7c787060, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 0, t2: 5, t3: 0 }, // a6
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0804020408102040, addend7: 0x787c7e7c78706040, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 1, t2: 5, t3: 0 }, // b6
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x1008040201000000, mult_d9: 0x0101010101010101, row_shift: 40, t0: 5, t1: 2, t2: 2, t3: 2 }, // c6
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x2010080402010000, mult_d9: 0x0101010101010101, row_shift: 40, t0: 5, t1: 3, t2: 3, t3: 3 }, // d6
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0408102040800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x4020100804020100, mult_d9: 0x0101010101010101, row_shift: 40, t0: 5, t1: 4, t2: 4, t3: 4 }, // e6
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0810204080000000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, mult_d9: 0x0101010101010101, row_shift: 40, t0: 5, t1: 5, t2: 5, t3: 5 }, // f6
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x1020402010080402, addend7: 0x7060406070787c7e, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 6, t2: 5, t3: 0 }, // g6
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x2040804020100804, addend7: 0x604000406070787c, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 7, t2: 5, t3: 0 }, // h6
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 0, t2: 0, t3: 0 }, // a7
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 1, t2: 1, t3: 0 }, // b7
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x00040a1120408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 2, t2: 2, t3: 0 }, // c7
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0008142241800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 3, t2: 3, t3: 0 }, // d7
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0010284482010000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 4, t2: 4, t3: 0 }, // e7
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0020508804020100, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 5, t2: 5, t3: 0 }, // f7
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 6, t2: 6, t3: 0 }, // g7
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0080402010080402, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 7, t2: 7, t3: 0 }, // h7
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 0, t2: 0, t3: 0 }, // a8
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 1, t2: 1, t3: 0 }, // b8
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x040a112040800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 2, t2: 2, t3: 0 }, // c8
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0814224180000000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 3, t2: 3, t3: 0 }, // d8
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x1028448201000000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 4, t2: 4, t3: 0 }, // e8
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x2050880402010000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 5, t2: 5, t3: 0 }, // f8
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x4020100804020100, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 6, t2: 6, t3: 0 }, // g8
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, mult_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 7, t2: 7, t3: 0 }, // h8
];

/// Sum the four COUNT_FLIP lookups for one bitboard, given pre-loaded params.
#[inline(always)]
fn count_one(p: u64, pp: &SqParams) -> i32 {
    let idx0 = ((p & pp.mask_v).wrapping_mul(pp.mult_v) >> 56) as usize;
    let idx1 = ((p >> pp.row_shift) & 0xff) as usize;
    let idx2 = ((((p & pp.mask_d7).wrapping_add(pp.addend7)) & pp.post_mask7)
        .wrapping_mul(pp.mult_d7)
        >> 56) as usize;
    let idx3 = ((p & pp.mask_d9).wrapping_mul(pp.mult_d9) >> 56) as usize;

    // SAFETY: t0..=t3 are 0..=7 (verified at table-construction time);
    // idx0..=idx3 are 0..=255 (one byte by `>> 56` or `& 0xff`).
    unsafe {
        *COUNT_FLIP.get_unchecked(pp.t0 as usize).get_unchecked(idx0) as i32
            + *COUNT_FLIP.get_unchecked(pp.t1 as usize).get_unchecked(idx1) as i32
            + *COUNT_FLIP.get_unchecked(pp.t2 as usize).get_unchecked(idx2) as i32
            + *COUNT_FLIP.get_unchecked(pp.t3 as usize).get_unchecked(idx3) as i32
    }
}

#[inline]
pub fn count_last_flip(p: u64, sq: Square) -> i32 {
    // SAFETY: Square::index() returns 0..=63 by construction.
    let pp = unsafe { PARAMS.get_unchecked(sq.index()) };
    count_one(p, pp)
}
