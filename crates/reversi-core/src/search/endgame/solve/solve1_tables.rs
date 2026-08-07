//! Lookup tables shared by last-empty-square scoring.
//!
//! `COUNT_FLIP_RAW` is the canonical 8x256 line-count table. The scalar path
//! uses its aligned `COUNT_FLIP` view for indexed lookups, the BMI2 path reads
//! the raw table at compile time when building its PEXT-specific per-square
//! table, and NEON reads it at compile time to build `COUNT_PAIR`.
//!
//! The remaining entries are kindergarten multiply/shift extractor parameters
//! for scalar and NEON builds. They are cfg-excluded from BMI2 builds, where
//! PEXT provides a different extraction strategy.
//!
//! The parameter table keeps `sq` as a runtime index instead of branching to
//! one of 64 specialized functions. It encodes the original kindergarten
//! variants: simple diagonal, addend-and-mask diagonal, and two-diagonal cases.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/count_last_flip_kindergarten.c>

use crate::util::align::Align64;

/// Builds the 8x256 last-flip count table: `[pos][line]` is twice the number
/// of discs flipped on an 8-cell line when the mover plays at `pos`, where
/// `line` is the mover's disc mask and every other non-empty cell belongs to
/// the opponent.
const fn build_count_flip() -> [[u8; 256]; 8] {
    let mut out = [[0u8; 256]; 8];
    let mut pos = 0;
    while pos < 8 {
        let mut line: usize = 0;
        while line < 256 {
            let mut count = 0u8;
            let above = line >> (pos + 1);
            if above != 0 {
                count += 2 * above.trailing_zeros() as u8;
            }
            let below = line & ((1 << pos) - 1);
            if below != 0 {
                count += 2 * (pos - 1 - below.ilog2() as usize) as u8;
            }
            out[pos][line] = count;
            line += 1;
        }
        pos += 1;
    }
    out
}

pub(super) const COUNT_FLIP_RAW: [[u8; 256]; 8] = build_count_flip();

#[cfg(not(target_arch = "aarch64"))]
pub(super) static COUNT_FLIP: Align64<[[u8; 256]; 8]> = Align64(COUNT_FLIP_RAW);

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
pub(super) const MULT_D9: u64 = 0x0101_0101_0101_0101;

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[derive(Clone, Copy)]
#[repr(align(64))]
pub(super) struct SqParams {
    pub(super) mask_v: u64,
    pub(super) mult_v: u64,
    pub(super) mask_d7: u64,
    pub(super) addend7: u64,
    pub(super) post_mask7: u64,
    pub(super) mult_d7: u64,
    pub(super) mask_d9: u64,
    pub(super) row_shift: u8,
    pub(super) t0: u8,
    pub(super) t1: u8,
    pub(super) t2: u8,
    pub(super) t3: u8,
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
#[rustfmt::skip]
pub(super) const PARAMS: [SqParams; 64] = [
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 0, t2: 0, t3: 0 }, // a1
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0080402010080402, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 1, t2: 1, t3: 0 }, // b1
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0000804020110a04, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 2, t2: 2, t3: 0 }, // c1
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0000008041221408, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 3, t2: 3, t3: 0 }, // d1
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0000000182442810, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 4, t2: 4, t3: 0 }, // e1
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0000010204885020, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 5, t2: 5, t3: 0 }, // f1
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 6, t2: 6, t3: 0 }, // g1
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 0, t0: 0, t1: 7, t2: 7, t3: 0 }, // h1
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x4020100804020100, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 0, t2: 0, t3: 0 }, // a2
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 1, t2: 1, t3: 0 }, // b2
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x00804020110a0400, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 2, t2: 2, t3: 0 }, // c2
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0000804122140800, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 3, t2: 3, t3: 0 }, // d2
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0000018244281000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 4, t2: 4, t3: 0 }, // e2
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0001020488502000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 5, t2: 5, t3: 0 }, // f2
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 6, t2: 6, t3: 0 }, // g2
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 8, t0: 1, t1: 7, t2: 7, t3: 0 }, // h2
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x2010080402010204, addend7: 0x6070787c7e7f7e7c, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 0, t2: 2, t3: 0 }, // a3
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x4020100804020408, addend7: 0x406070787c7e7c78, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 1, t2: 2, t3: 0 }, // b3
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0000000102040810, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, row_shift: 16, t0: 2, t1: 2, t2: 2, t3: 2 }, // c3
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0000010204081020, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0080402010080402, row_shift: 16, t0: 2, t1: 3, t2: 3, t3: 3 }, // d3
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000804020100804, row_shift: 16, t0: 2, t1: 4, t2: 4, t3: 4 }, // e3
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000008040201008, row_shift: 16, t0: 2, t1: 5, t2: 5, t3: 5 }, // f3
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0204081020402010, addend7: 0x7e7c787060406070, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 6, t2: 2, t3: 0 }, // g3
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0408102040804020, addend7: 0x7c78706040004060, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 16, t0: 2, t1: 7, t2: 2, t3: 0 }, // h3
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x1008040201020408, addend7: 0x70787c7e7f7e7c78, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 0, t2: 3, t3: 0 }, // a4
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x2010080402040810, addend7: 0x6070787c7e7c7870, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 1, t2: 3, t3: 0 }, // b4
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0000010204081020, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x4020100804020100, row_shift: 24, t0: 3, t1: 2, t2: 2, t3: 2 }, // c4
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, row_shift: 24, t0: 3, t1: 3, t2: 3, t3: 3 }, // d4
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0080402010080402, row_shift: 24, t0: 3, t1: 4, t2: 4, t3: 4 }, // e4
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000804020100804, row_shift: 24, t0: 3, t1: 5, t2: 5, t3: 5 }, // f4
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0408102040201008, addend7: 0x7c78706040607078, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 6, t2: 3, t3: 0 }, // g4
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0810204080402010, addend7: 0x7870604000406070, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 24, t0: 3, t1: 7, t2: 3, t3: 0 }, // h4
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0804020102040810, addend7: 0x787c7e7f7e7c7870, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 0, t2: 4, t3: 0 }, // a5
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x1008040204081020, addend7: 0x70787c7e7c787060, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 1, t2: 4, t3: 0 }, // b5
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x2010080402010000, row_shift: 32, t0: 4, t1: 2, t2: 2, t3: 2 }, // c5
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x4020100804020100, row_shift: 32, t0: 4, t1: 3, t2: 3, t3: 3 }, // d5
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, row_shift: 32, t0: 4, t1: 4, t2: 4, t3: 4 }, // e5
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0408102040800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0080402010080402, row_shift: 32, t0: 4, t1: 5, t2: 5, t3: 5 }, // f5
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x0810204020100804, addend7: 0x787060406070787c, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 6, t2: 4, t3: 0 }, // g5
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x1020408040201008, addend7: 0x7060400040607078, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 32, t0: 4, t1: 7, t2: 4, t3: 0 }, // h5
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0402010204081020, addend7: 0x7c7e7f7e7c787060, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 0, t2: 5, t3: 0 }, // a6
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0804020408102040, addend7: 0x787c7e7c78706040, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 1, t2: 5, t3: 0 }, // b6
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x1008040201000000, row_shift: 40, t0: 5, t1: 2, t2: 2, t3: 2 }, // c6
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x2010080402010000, row_shift: 40, t0: 5, t1: 3, t2: 3, t3: 3 }, // d6
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0408102040800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x4020100804020100, row_shift: 40, t0: 5, t1: 4, t2: 4, t3: 4 }, // e6
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0810204080000000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x8040201008040201, row_shift: 40, t0: 5, t1: 5, t2: 5, t3: 5 }, // f6
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x1020402010080402, addend7: 0x7060406070787c7e, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 6, t2: 5, t3: 0 }, // g6
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x2040804020100804, addend7: 0x604000406070787c, post_mask7: 0x8080808080808080, mult_d7: 0x0002040810204081, mask_d9: 0x0000000000000000, row_shift: 40, t0: 5, t1: 7, t2: 5, t3: 0 }, // h6
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0001020408102040, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 0, t2: 0, t3: 0 }, // a7
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 1, t2: 1, t3: 0 }, // b7
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x00040a1120408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 2, t2: 2, t3: 0 }, // c7
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0008142241800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 3, t2: 3, t3: 0 }, // d7
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x0010284482010000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 4, t2: 4, t3: 0 }, // e7
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x0020508804020100, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 5, t2: 5, t3: 0 }, // f7
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 6, t2: 6, t3: 0 }, // g7
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x0080402010080402, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 48, t0: 6, t1: 7, t2: 7, t3: 0 }, // h7
    SqParams { mask_v: 0x0101010101010101, mult_v: 0x0102040810204080, mask_d7: 0x0102040810204080, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 0, t2: 0, t3: 0 }, // a8
    SqParams { mask_v: 0x0202020202020202, mult_v: 0x0081020408102040, mask_d7: 0x0204081020408000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 1, t2: 1, t3: 0 }, // b8
    SqParams { mask_v: 0x0404040404040404, mult_v: 0x0040810204081020, mask_d7: 0x040a112040800000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 2, t2: 2, t3: 0 }, // c8
    SqParams { mask_v: 0x0808080808080808, mult_v: 0x0020408102040810, mask_d7: 0x0814224180000000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 3, t2: 3, t3: 0 }, // d8
    SqParams { mask_v: 0x1010101010101010, mult_v: 0x0010204081020408, mask_d7: 0x1028448201000000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 4, t2: 4, t3: 0 }, // e8
    SqParams { mask_v: 0x2020202020202020, mult_v: 0x0008102040810204, mask_d7: 0x2050880402010000, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 5, t2: 5, t3: 0 }, // f8
    SqParams { mask_v: 0x4040404040404040, mult_v: 0x0004081020408102, mask_d7: 0x4020100804020100, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 6, t2: 6, t3: 0 }, // g8
    SqParams { mask_v: 0x8080808080808080, mult_v: 0x0002040810204081, mask_d7: 0x8040201008040201, addend7: 0x0000000000000000, post_mask7: 0xffffffffffffffff, mult_d7: 0x0101010101010101, mask_d9: 0x0000000000000000, row_shift: 56, t0: 7, t1: 7, t2: 7, t3: 0 }, // h8
];

/// Number of distinct `(line position, valid-cell mask)` pairs used by the
/// four extraction slots of all 64 squares.
#[cfg(target_arch = "aarch64")]
const PAIR_ROWS: usize = 24;

/// `meta` bit marking the squares whose two diagonals need separate lookups.
#[cfg(target_arch = "aarch64")]
pub(super) const META_HAS_D9: u64 = 1 << 63;

/// XOR that turns a player's D7 line index into the opponent's: the set of
/// line positions backed by a real board cell.
#[cfg(target_arch = "aarch64")]
const fn d7_index_xor(pp: &SqParams) -> u8 {
    let empty = (pp.addend7 & pp.post_mask7).wrapping_mul(pp.mult_d7);
    let full = (pp.mask_d7.wrapping_add(pp.addend7) & pp.post_mask7).wrapping_mul(pp.mult_d7);
    ((empty ^ full) >> 56) as u8
}

#[cfg(target_arch = "aarch64")]
const fn d9_index_xor(pp: &SqParams) -> u8 {
    (pp.mask_d9.wrapping_mul(MULT_D9) >> 56) as u8
}

/// Collects the `(line position, valid-cell mask)` key of every extraction slot,
/// deduplicated. Slot order is vertical, row, D7, D9.
#[cfg(target_arch = "aarch64")]
const fn pair_class_keys() -> [(u8, u8); PAIR_ROWS] {
    let mut keys = [(0u8, 0u8); PAIR_ROWS];
    let mut len = 0;
    let mut sq = 0;
    while sq < PARAMS.len() {
        let pp = &PARAMS[sq];
        let slots = [
            (pp.t0, 0xff),
            (pp.t1, 0xff),
            (pp.t2, d7_index_xor(pp)),
            (pp.t3, d9_index_xor(pp)),
        ];
        let used = if pp.mask_d9 != 0 { 4 } else { 3 };
        let mut slot = 0;
        while slot < used {
            let (t, xor) = slots[slot];
            let mut seen = false;
            let mut c = 0;
            while c < len {
                if keys[c].0 == t && keys[c].1 == xor {
                    seen = true;
                    break;
                }
                c += 1;
            }
            if !seen {
                assert!(len < PAIR_ROWS, "PAIR_ROWS is too small");
                keys[len] = (t, xor);
                len += 1;
            }
            slot += 1;
        }
        sq += 1;
    }
    assert!(
        len == PAIR_ROWS,
        "PAIR_ROWS does not match the used classes"
    );
    keys
}

#[cfg(target_arch = "aarch64")]
const PAIR_KEYS: [(u8, u8); PAIR_ROWS] = pair_class_keys();

#[cfg(target_arch = "aarch64")]
const fn pair_class(t: u8, xor: u8) -> u64 {
    let mut c = 0;
    while c < PAIR_ROWS {
        if PAIR_KEYS[c].0 == t && PAIR_KEYS[c].1 == xor {
            return c as u64;
        }
        c += 1;
    }
    panic!("extraction slot without a COUNT_PAIR class")
}

#[cfg(target_arch = "aarch64")]
const fn build_count_pair() -> [[u16; 256]; PAIR_ROWS] {
    let mut out = [[0u16; 256]; PAIR_ROWS];
    let mut c = 0;
    while c < PAIR_ROWS {
        let (t, xor) = PAIR_KEYS[c];
        let row = &COUNT_FLIP_RAW[t as usize];
        let mut idx = 0;
        while idx < 256 {
            out[c][idx] = row[idx] as u16 | ((row[idx ^ xor as usize] as u16) << 8);
            idx += 1;
        }
        c += 1;
    }
    out
}

/// Line counts for both sides in one entry: the low byte is the mover's flip
/// count for line index `idx`, the high byte is the opponent's count for the
/// same line. With one empty square the opponent holds exactly the line cells
/// the mover does not, so the opponent's index is `idx ^ mask` for the class's
/// valid-cell mask. Summing four entries never carries between the bytes
/// because a single line flips at most 12.
#[cfg(target_arch = "aarch64")]
pub(super) static COUNT_PAIR: Align64<[[u16; 256]; PAIR_ROWS]> = Align64(build_count_pair());

/// One cache line of AArch64 extraction data.
///
/// `meta` stores, from low to high byte: row shift and the four `COUNT_PAIR`
/// class indices. Bit 63 is [`META_HAS_D9`].
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
#[repr(align(64))]
pub(super) struct NeonSqParams {
    pub(super) mask_v: u64,
    pub(super) mult_v: u64,
    pub(super) mask_d7: u64,
    pub(super) addend7: u64,
    pub(super) post_mask7: u64,
    pub(super) mult_d7: u64,
    pub(super) mask_d9: u64,
    pub(super) meta: u64,
}

#[cfg(target_arch = "aarch64")]
const _: () = assert!(core::mem::size_of::<NeonSqParams>() == 64);

#[cfg(target_arch = "aarch64")]
const fn build_neon_params() -> [NeonSqParams; 64] {
    let zero = NeonSqParams {
        mask_v: 0,
        mult_v: 0,
        mask_d7: 0,
        addend7: 0,
        post_mask7: 0,
        mult_d7: 0,
        mask_d9: 0,
        meta: 0,
    };
    let mut params = [zero; 64];
    let mut sq = 0;
    while sq < PARAMS.len() {
        let pp = PARAMS[sq];
        let d9 = if pp.mask_d9 != 0 {
            (pair_class(pp.t3, d9_index_xor(&pp)) << 32) | META_HAS_D9
        } else {
            0
        };
        params[sq] = NeonSqParams {
            mask_v: pp.mask_v,
            mult_v: pp.mult_v,
            mask_d7: pp.mask_d7,
            addend7: pp.addend7,
            post_mask7: pp.post_mask7,
            mult_d7: pp.mult_d7,
            mask_d9: pp.mask_d9,
            meta: pp.row_shift as u64
                | (pair_class(pp.t0, 0xff) << 8)
                | (pair_class(pp.t1, 0xff) << 16)
                | (pair_class(pp.t2, d7_index_xor(&pp)) << 24)
                | d9,
        };
        sq += 1;
    }
    params
}

#[cfg(target_arch = "aarch64")]
pub(super) static NEON_PARAMS: [NeonSqParams; 64] = build_neon_params();
