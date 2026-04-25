//! AVX2 variant of flip function.
//! Based on flip_avx_ppseq.c from edax-reversi.
//! Reference: <https://github.com/abulmo/edax-reversi/blob/ce77e7a7da45282799e61871882ecac07b3884aa/src/flip_avx_ppseq.c>

use crate::square::Square;
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn mm_flip(op: __m128i, pos: usize) -> __m128i {
    let mut flip: __m256i;
    let mut mask: __m256i;
    let mut rs: __m256i;
    let mut lo: __m256i;

    // Broadcast the player's and opponent's patterns
    let pp: __m256i = _mm256_broadcastq_epi64(op);
    let oo: __m256i = _mm256_broadcastq_epi64(_mm_unpackhi_epi64(op, op));

    let mask_ptr = unsafe { super::lrmask::LRMASK.get_unchecked(pos).0.as_ptr() as *const __m256i };

    // Right side computations
    mask = unsafe { _mm256_load_si256(mask_ptr.add(1)) };
    // Right: shadow mask lower than leftmost P
    let rp: __m256i = _mm256_and_si256(pp, mask);
    rs = _mm256_or_si256(rp, _mm256_srlv_epi64(rp, _mm256_set_epi64x(7, 9, 8, 1)));
    rs = _mm256_or_si256(rs, _mm256_srlv_epi64(rs, _mm256_set_epi64x(14, 18, 16, 2)));
    rs = _mm256_or_si256(rs, _mm256_srlv_epi64(rs, _mm256_set_epi64x(28, 36, 32, 4)));
    // Apply flip if leftmost non-opponent is P
    let re: __m256i = _mm256_xor_si256(_mm256_andnot_si256(oo, mask), rp); // Masked Empty
    flip = _mm256_and_si256(_mm256_andnot_si256(rs, mask), _mm256_cmpgt_epi64(rp, re));

    // Left side computations
    mask = unsafe { _mm256_load_si256(mask_ptr) };
    // Left: non-opponent BLSMSK
    lo = _mm256_andnot_si256(oo, mask);
    lo = _mm256_and_si256(
        _mm256_xor_si256(_mm256_add_epi64(lo, _mm256_set1_epi64x(-1)), lo),
        mask,
    );
    // Clear MSB of BLSMSK if it is P
    let lf: __m256i = _mm256_andnot_si256(pp, lo);
    // Erase lf if lo = lf (i.e., MSB is not P)
    flip = _mm256_or_si256(flip, _mm256_andnot_si256(_mm256_cmpeq_epi64(lf, lo), lf));

    // Combine the lower and higher 128-bit lanes of the flip pattern
    _mm_or_si128(
        _mm256_castsi256_si128(flip),
        _mm256_extracti128_si256(flip, 1),
    )
}

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub fn flip(sq: Square, player: u64, opponent: u64) -> u64 {
    let op = _mm_set_epi64x(opponent as i64, player as i64);
    let flip = mm_flip(op, sq.index());
    _mm_cvtsi128_si64(_mm_or_si128(flip, _mm_shuffle_epi32(flip, 0x4e))) as u64
}
