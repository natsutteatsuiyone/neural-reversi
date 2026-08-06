//! WebAssembly SIMD flip backend for single-square and shared-board batch paths.
//!
//! This follows the NEON backend's bit-reversal scheme: the eight direction
//! masks are processed as four `v128` pairs. Right-side masks are
//! pre-bit-reversed so the same LSB-to-MSB outflank primitive handles both
//! directions.

use super::lrmask::LRMASK;
use crate::square::Square;
use core::arch::wasm32::*;

#[repr(align(64))]
#[derive(Copy, Clone)]
struct WasmMaskEntry([u64; 8]);

static WASM_MASK: [WasmMaskEntry; 66] = build_wasm_masks();

const fn build_wasm_masks() -> [WasmMaskEntry; 66] {
    let mut out = [WasmMaskEntry([0; 8]); 66];
    let mut i = 0;
    while i < 66 {
        let mut j = 0;
        while j < 4 {
            out[i].0[j] = LRMASK[i].0[j];
            out[i].0[j + 4] = LRMASK[i].0[j + 4].reverse_bits();
            j += 1;
        }
        i += 1;
    }
    out
}

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
#[target_feature(enable = "simd128")]
#[inline]
pub fn flip(sq: Square, player: u64, opponent: u64) -> u64 {
    BoardCtx::new(player, opponent).flip1(sq.index())
}

/// SIMD board context for runtime squares that share the same `(player,
/// opponent)` board.
#[derive(Copy, Clone)]
pub(super) struct BoardCtx {
    pp: v128,
    no: v128,
    pp_rev: v128,
    no_rev: v128,
    zero: v128,
    one: v128,
}

impl BoardCtx {
    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn new(player: u64, opponent: u64) -> Self {
        let not_opponent = !opponent;
        let rev = bit_reverse_u64x2(u64x2(player, not_opponent));
        Self {
            pp: u64x2_splat(player),
            no: u64x2_splat(not_opponent),
            pp_rev: i64x2_shuffle::<0, 0>(rev, rev),
            no_rev: i64x2_shuffle::<1, 1>(rev, rev),
            zero: u64x2_splat(0),
            one: u64x2_splat(1),
        }
    }

    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn flip1(&self, pos: usize) -> u64 {
        unsafe {
            flip_index_prepared(
                pos,
                self.pp,
                self.no,
                self.pp_rev,
                self.no_rev,
                self.zero,
                self.one,
            )
        }
    }

    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn flip2(&self, x0: usize, x1: usize) -> (u64, u64) {
        (self.flip1(x0), self.flip1(x1))
    }

    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn flip3(&self, x0: usize, x1: usize, x2: usize) -> (u64, u64, u64) {
        (self.flip1(x0), self.flip1(x1), self.flip1(x2))
    }

    #[target_feature(enable = "simd128")]
    #[inline]
    pub fn flip4(&self, x0: usize, x1: usize, x2: usize, x3: usize) -> (u64, u64, u64, u64) {
        (
            self.flip1(x0),
            self.flip1(x1),
            self.flip1(x2),
            self.flip1(x3),
        )
    }
}

#[target_feature(enable = "simd128")]
#[inline]
unsafe fn flip_index_prepared(
    pos: usize,
    pp: v128,
    no: v128,
    pp_rev: v128,
    no_rev: v128,
    zero: v128,
    one: v128,
) -> u64 {
    let mask_ptr = unsafe { WASM_MASK.get_unchecked(pos).0.as_ptr() as *const v128 };

    let mask_l_a = unsafe { v128_load(mask_ptr) };
    let mask_l_b = unsafe { v128_load(mask_ptr.add(1)) };
    let flip_l_a = flip_left_pair(mask_l_a, pp, no, zero, one);
    let flip_l_b = flip_left_pair(mask_l_b, pp, no, zero, one);

    let mask_rr_a = unsafe { v128_load(mask_ptr.add(2)) };
    let mask_rr_b = unsafe { v128_load(mask_ptr.add(3)) };
    let flip_rr_a = flip_left_pair(mask_rr_a, pp_rev, no_rev, zero, one);
    let flip_rr_b = flip_left_pair(mask_rr_b, pp_rev, no_rev, zero, one);

    let flip_l = v128_or(flip_l_a, flip_l_b);
    let flip_rr = bit_reverse_u64x2(v128_or(flip_rr_a, flip_rr_b));
    fold_or_pair(v128_or(flip_l, flip_rr))
}

/// Computes a pair of LSB-first rays. Native left-side masks use this form
/// directly; right-side masks are bit-reversed before reaching this helper.
/// The closest square is the least significant bit in each mask.
#[inline]
#[target_feature(enable = "simd128")]
fn flip_left_pair(mask: v128, pp: v128, no: v128, zero: v128, one: v128) -> v128 {
    let non_opponent = v128_and(mask, no);
    let outflank = v128_and(v128_and(non_opponent, u64x2_sub(zero, non_opponent)), pp);
    let nonzero = v128_not(u64x2_eq(outflank, zero));
    v128_and(v128_and(mask, u64x2_sub(outflank, one)), nonzero)
}

#[inline]
#[target_feature(enable = "simd128")]
fn fold_or_pair(x: v128) -> u64 {
    u64x2_extract_lane::<0>(x) | u64x2_extract_lane::<1>(x)
}

/// Reverses the bits within each 64-bit lane (lane `i` becomes
/// `reverse_bits(lane_i)`). wasm SIMD has no bit-reversal opcode, so this
/// reverses the byte order within each lane (`i8x16.swizzle`) and then the
/// bits within each byte via a 4-bit nibble lookup table (two more swizzles).
/// One short, parallel SIMD sequence in place of the scalar
/// `u64::reverse_bits` shift/mask chain.
#[inline]
#[target_feature(enable = "simd128")]
fn bit_reverse_u64x2(x: v128) -> v128 {
    let byte_rev = i8x16_swizzle(
        x,
        i8x16(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8),
    );
    // `lut[n]` is the 4-bit reversal of nibble `n` (e.g. 0b0001 -> 0b1000).
    let lut = i8x16(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);
    let low_mask = u8x16_splat(0x0f);
    let lo = v128_and(byte_rev, low_mask);
    let hi = v128_and(u8x16_shr(byte_rev, 4), low_mask);
    // The reversed low nibble moves to the high half of the byte and vice versa.
    let rev_lo = i8x16_swizzle(lut, lo);
    let rev_hi = i8x16_swizzle(lut, hi);
    v128_or(i8x16_shl(rev_lo, 4), rev_hi)
}
