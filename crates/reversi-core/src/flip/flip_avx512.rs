//! AVX-512 variant of flip function.
//! Based on flip_avx512cd.c from edax-reversi.
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/flip_avx512cd.c>

use crate::square::Square;
use std::arch::x86_64::*;

/// Emits a plain `vpsrlvq` and relies on the hardware-defined zero result for shift
/// counts >= 64. Since Rust 1.93 (LLVM 21+) the `_mm256_srlv_epi64` intrinsic wrapper
/// produces `vptestmq + vpsrlvq{k}{z}`, which is wasted work when the algorithm already
/// tolerates the >=64 case (see `mm_flip`'s `lzcnt` → `srlv` pattern).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512cd,avx512vl")]
#[inline]
unsafe fn vpsrlvq_raw(src: __m256i, cnt: __m256i) -> __m256i {
    let out: __m256i;
    unsafe {
        std::arch::asm!(
            "vpsrlvq {out}, {src}, {cnt}",
            out = lateout(ymm_reg) out,
            src = in(ymm_reg) src,
            cnt = in(ymm_reg) cnt,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512cd,avx512vl")]
#[inline]
fn mm_flip(op: __m128i, x: usize) -> __m128i {
    let pp = _mm256_broadcastq_epi64(op);
    let oo = _mm256_broadcastq_epi64(_mm_unpackhi_epi64(op, op));

    let mask_ptr = unsafe { super::lrmask::LRMASK.get_unchecked(x).0.as_ptr() as *const __m256i };
    let right_mask = unsafe { _mm256_load_si256(mask_ptr.add(1)) };

    let mut t0 = _mm256_lzcnt_epi64(_mm256_andnot_si256(oo, right_mask));
    t0 = _mm256_and_si256(
        unsafe { vpsrlvq_raw(_mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64), t0) },
        pp,
    );
    let right_flank = _mm256_ternarylogic_epi64(
        _mm256_sub_epi64(_mm256_setzero_si256(), t0),
        t0,
        right_mask,
        0x28,
    );

    let left_mask = unsafe { _mm256_load_si256(mask_ptr) };
    let mut l_o = _mm256_andnot_si256(oo, left_mask);
    l_o = _mm256_ternarylogic_epi64(l_o, _mm256_sub_epi64(_mm256_setzero_si256(), l_o), pp, 0x80);

    let ff = _mm256_ternarylogic_epi64(
        right_flank,
        _mm256_sub_epi64(_mm256_cmpeq_epi64(l_o, _mm256_setzero_si256()), l_o),
        left_mask,
        0xF2,
    );

    _mm_or_si128(_mm256_castsi256_si128(ff), _mm256_extracti128_si256(ff, 1))
}

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512cd,avx512vl")]
#[inline]
pub fn flip(sq: Square, p: u64, o: u64) -> u64 {
    let op = _mm_set_epi64x(o as i64, p as i64);
    let flip = mm_flip(op, sq.index());
    let rflip = _mm_or_si128(flip, _mm_shuffle_epi32(flip, 0x4E));
    _mm_cvtsi128_si64(rflip) as u64
}
