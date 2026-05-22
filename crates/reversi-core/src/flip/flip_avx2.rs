//! AVX2 variant of flip function.
//! Based on flip_avx_ppseq.c from edax-reversi.
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/flip_avx_ppseq.c>

use crate::square::Square;
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn mm_flip_prepared(pp: __m256i, oo: __m256i, pos: usize) -> __m128i {
    // SAFETY: this module is compiled only for AVX2 targets. `pos` comes from
    // `Square` or the sentinel pseudo-squares, indexing the 66-entry,
    // 64-byte-aligned `LRMASK` table; each entry contains two aligned YMM masks.
    unsafe {
        let mask_ptr = super::lrmask::LRMASK.get_unchecked(pos).0.as_ptr() as *const __m256i;
        let right_mask = _mm256_load_si256(mask_ptr.add(1));
        let left_mask = _mm256_load_si256(mask_ptr);

        // Left: non-opponent BLSMSK
        let mut lo = _mm256_andnot_si256(oo, left_mask);
        lo = _mm256_and_si256(
            _mm256_xor_si256(_mm256_add_epi64(lo, _mm256_set1_epi64x(-1)), lo),
            left_mask,
        );

        // Right side computations
        // Right: shadow mask lower than leftmost P
        let rp: __m256i = _mm256_and_si256(pp, right_mask);
        let mut rs = _mm256_or_si256(rp, _mm256_srlv_epi64(rp, _mm256_set_epi64x(7, 9, 8, 1)));
        // Clear MSB of BLSMSK if it is P
        let lf: __m256i = _mm256_andnot_si256(pp, lo);
        // Erase lf if lo = lf (i.e., MSB is not P)
        let left_flip = _mm256_andnot_si256(_mm256_cmpeq_epi64(lf, lo), lf);
        rs = _mm256_or_si256(rs, _mm256_srlv_epi64(rs, _mm256_set_epi64x(14, 18, 16, 2)));
        rs = _mm256_or_si256(rs, _mm256_srlv_epi64(rs, _mm256_set_epi64x(28, 36, 32, 4)));
        // Apply flip if leftmost non-opponent is P
        let re: __m256i = _mm256_xor_si256(_mm256_andnot_si256(oo, right_mask), rp); // Masked Empty
        let right_flip = _mm256_and_si256(
            _mm256_andnot_si256(rs, right_mask),
            _mm256_cmpgt_epi64(rp, re),
        );

        let flip = _mm256_or_si256(right_flip, left_flip);

        // Combine the lower and higher 128-bit lanes of the flip pattern
        _mm_or_si128(
            _mm256_castsi256_si128(flip),
            _mm256_extracti128_si256(flip, 1),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn flip_prepared(pp: __m256i, oo: __m256i, pos: usize) -> u64 {
    let flip = mm_flip_prepared(pp, oo, pos);
    // SAFETY: this module is compiled only for AVX2 targets, which include the
    // SSE2 operations used for the final horizontal OR reduction.
    unsafe { _mm_cvtsi128_si64(_mm_or_si128(flip, _mm_shuffle_epi32(flip, 0x4e))) as u64 }
}

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub fn flip(sq: Square, player: u64, opponent: u64) -> u64 {
    BoardCtx::new(player, opponent).flip1(sq.index())
}

/// SIMD board context for runtime squares that share the same `(player,
/// opponent)` board.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
pub(super) struct BoardCtx {
    pp: __m256i,
    oo: __m256i,
}

#[cfg(target_arch = "x86_64")]
impl BoardCtx {
    #[inline(always)]
    pub fn new(player: u64, opponent: u64) -> Self {
        // SAFETY: this module is compiled only for AVX2 targets.
        unsafe {
            Self {
                pp: _mm256_set1_epi64x(player as i64),
                oo: _mm256_set1_epi64x(opponent as i64),
            }
        }
    }

    #[inline(always)]
    pub fn flip1(&self, pos: usize) -> u64 {
        flip_prepared(self.pp, self.oo, pos)
    }

    #[inline(always)]
    pub fn flip2(&self, x0: usize, x1: usize) -> (u64, u64) {
        (self.flip1(x0), self.flip1(x1))
    }

    #[inline(always)]
    pub fn flip3(&self, x0: usize, x1: usize, x2: usize) -> (u64, u64, u64) {
        (self.flip1(x0), self.flip1(x1), self.flip1(x2))
    }

    #[inline(always)]
    pub fn flip4(&self, x0: usize, x1: usize, x2: usize, x3: usize) -> (u64, u64, u64, u64) {
        (
            self.flip1(x0),
            self.flip1(x1),
            self.flip1(x2),
            self.flip1(x3),
        )
    }
}
