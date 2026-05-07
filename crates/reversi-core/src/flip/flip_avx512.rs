//! AVX-512 flip backend with a real batch fast path.
//!
//! Based on flip_avx512cd.c from edax-reversi.
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/flip_avx512cd.c>
//!
//! The single-square `flip(sq, p, o)` API is kept, but the important addition is
//! the pairwise batch path: it evaluates two runtime squares at once when the
//! board `(p, o)` is shared by a node's legal moves.  No const square
//! specialization is used.

// The shared bodies wrap each individual intrinsic in its own `unsafe { ... }`
// so they remain self-contained when invoked from a context that has not been
// declared `unsafe` (e.g. an Edition 2024 `#[target_feature]` function). When
// invoked from an outer `unsafe { ... }` (the variant-1 single-square API
// without `#[target_feature]`), those inner blocks become redundant, which
// would otherwise fire `unused_unsafe`.
#![allow(unused_unsafe)]

use crate::square::Square;
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
macro_rules! vpsrlvq_raw_ymm {
    ($src:expr, $cnt:expr) => {{
        let out: __m256i;
        unsafe {
            std::arch::asm!(
                "vpsrlvq {out}, {src}, {cnt}",
                out = lateout(ymm_reg) out,
                src = in(ymm_reg) $src,
                cnt = in(ymm_reg) $cnt,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        out
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! vpsrlvq_raw_zmm {
    ($src:expr, $cnt:expr) => {{
        let out: __m512i;
        unsafe {
            std::arch::asm!(
                "vpsrlvq {out}, {src}, {cnt}",
                out = lateout(zmm_reg) out,
                src = in(zmm_reg) $src,
                cnt = in(zmm_reg) $cnt,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        out
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! reduce4_or_u64 {
    ($v:expr) => {{
        let x = _mm_or_si128(_mm256_castsi256_si128($v), _mm256_extracti128_si256($v, 1));
        let x = _mm_or_si128(x, _mm_shuffle_epi32(x, 0x4e));
        _mm_cvtsi128_si64(x) as u64
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! flip_prepared_ymm_body {
    ($x:expr, $pp:expr, $no:expr, $zero:expr, $msb:expr) => {{
        debug_assert!($x < 66);
        let mask_ptr =
            unsafe { super::lrmask::LRMASK.get_unchecked($x).0.as_ptr() as *const __m256i };

        // Start the high-latency right-side LZCNT chain first.
        let right_mask = unsafe { _mm256_load_si256(mask_ptr.add(1)) };
        let mut right_bit = _mm256_lzcnt_epi64(_mm256_and_si256($no, right_mask));

        // The left-side LS1B chain is independent and fills part of the
        // LZCNT -> VPSRLVQ gap.
        let left_mask = unsafe { _mm256_load_si256(mask_ptr) };
        let mut left_bit = _mm256_and_si256($no, left_mask);
        left_bit =
            _mm256_ternarylogic_epi64(left_bit, _mm256_sub_epi64($zero, left_bit), $pp, 0x80);
        let left_flank = _mm256_sub_epi64(_mm256_cmpeq_epi64(left_bit, $zero), left_bit);

        right_bit = _mm256_and_si256(vpsrlvq_raw_ymm!($msb, right_bit), $pp);
        let right_flips = _mm256_ternarylogic_epi64(
            _mm256_sub_epi64($zero, right_bit),
            right_bit,
            right_mask,
            0x28,
        );

        // right_flips | (left_mask & !left_flank)
        reduce4_or_u64!(_mm256_ternarylogic_epi64(
            right_flips,
            left_flank,
            left_mask,
            0xf2,
        ))
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! flip_pair_zmm2_body {
    ($x0:expr, $x1:expr, $pp:expr, $no:expr, $zero:expr, $msb:expr, $all_ones:expr) => {{
        debug_assert!($x0 < 66 && $x1 < 66);
        let m0 = unsafe { super::lrmask::LRMASK.get_unchecked($x0).0.as_ptr() as *const __m256i };
        let m1 = unsafe { super::lrmask::LRMASK.get_unchecked($x1).0.as_ptr() as *const __m256i };

        // Pack square0 into low 256 bits and square1 into high 256 bits.
        let left_mask = _mm512_castsi256_si512(unsafe { _mm256_load_si256(m0) });
        let left_mask = _mm512_inserti64x4(left_mask, unsafe { _mm256_load_si256(m1) }, 1);
        let right_mask = _mm512_castsi256_si512(unsafe { _mm256_load_si256(m0.add(1)) });
        let right_mask = _mm512_inserti64x4(right_mask, unsafe { _mm256_load_si256(m1.add(1)) }, 1);

        let mut right_bit = _mm512_lzcnt_epi64(_mm512_and_si512($no, right_mask));
        let mut left_bit = _mm512_and_si512($no, left_mask);
        left_bit =
            _mm512_ternarylogic_epi64(left_bit, _mm512_sub_epi64($zero, left_bit), $pp, 0x80);

        let eq_zero = _mm512_cmpeq_epi64_mask(left_bit, $zero);
        let eq_vec = _mm512_maskz_mov_epi64(eq_zero, $all_ones);
        let left_flank = _mm512_sub_epi64(eq_vec, left_bit);

        right_bit = _mm512_and_si512(vpsrlvq_raw_zmm!($msb, right_bit), $pp);
        let right_flips = _mm512_ternarylogic_epi64(
            _mm512_sub_epi64($zero, right_bit),
            right_bit,
            right_mask,
            0x28,
        );

        let flips = _mm512_ternarylogic_epi64(right_flips, left_flank, left_mask, 0xf2);
        let lo = _mm512_castsi512_si256(flips);
        let hi = _mm512_extracti64x4_epi64(flips, 1);
        (reduce4_or_u64!(lo), reduce4_or_u64!(hi))
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! flip_runtime_body {
    ($x:expr, $p:expr, $o:expr) => {{
        let pp = _mm256_set1_epi64x($p as i64);
        let no = _mm256_set1_epi64x((!$o) as i64);
        let zero = _mm256_setzero_si256();
        let msb = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
        flip_prepared_ymm_body!($x, pp, no, zero, msb)
    }};
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512cd",
    target_feature = "avx512vl"
))]
#[inline(always)]
pub fn flip_index(x: usize, p: u64, o: u64) -> u64 {
    unsafe { flip_runtime_body!(x, p, o) }
}

#[cfg(all(
    target_arch = "x86_64",
    not(all(
        target_feature = "avx512f",
        target_feature = "avx512cd",
        target_feature = "avx512vl"
    ))
))]
#[target_feature(enable = "avx512f,avx512cd,avx512vl")]
#[inline]
pub fn flip_index(x: usize, p: u64, o: u64) -> u64 {
    unsafe { flip_runtime_body!(x, p, o) }
}

/// Computes the bitboard of discs flipped by placing a disc at `sq`.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn flip(sq: Square, p: u64, o: u64) -> u64 {
    flip_index(sq.index(), p, o)
}

/// Prepared context for one-at-a-time calls that share the same board.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
pub struct FlipPrepared {
    pp: __m256i,
    no: __m256i,
}

#[cfg(target_arch = "x86_64")]
impl FlipPrepared {
    #[target_feature(enable = "avx512f,avx512cd,avx512vl")]
    #[inline]
    pub fn new(p: u64, o: u64) -> Self {
        Self {
            pp: _mm256_set1_epi64x(p as i64),
            no: _mm256_set1_epi64x((!o) as i64),
        }
    }

    #[target_feature(enable = "avx512f,avx512cd,avx512vl")]
    #[inline]
    pub fn flip_index(self, x: usize) -> u64 {
        let zero = _mm256_setzero_si256();
        let msb = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
        unsafe { flip_prepared_ymm_body!(x, self.pp, self.no, zero, msb) }
    }

    #[target_feature(enable = "avx512f,avx512cd,avx512vl")]
    #[inline]
    pub fn flip(self, sq: Square) -> u64 {
        self.flip_index(sq.index())
    }
}

/// SIMD board context: `(p, !o)` and helper constants broadcast once and held
/// in registers across an entire move-generation loop.
///
/// Built with [`BoardCtx::new`] (5 wide broadcasts). The 256-bit constants
/// used by the single-square path are derived for free with
/// `_mm512_castsi512_si256`, so callers pay no extra setup for the trailing
/// odd move. All methods are `#[inline(always)]` and contain no
/// `#[target_feature]` gate, so they fold into the caller.
#[cfg(target_arch = "x86_64")]
#[derive(Copy, Clone)]
pub struct BoardCtx {
    pp: __m512i,
    no: __m512i,
    zero: __m512i,
    msb: __m512i,
    all_ones: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl BoardCtx {
    /// Broadcast `(p, !o)` and the working constants into wide vector lanes.
    #[inline(always)]
    pub fn new(p: u64, o: u64) -> Self {
        unsafe {
            Self {
                pp: _mm512_set1_epi64(p as i64),
                no: _mm512_set1_epi64((!o) as i64),
                zero: _mm512_setzero_si512(),
                msb: _mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64),
                all_ones: _mm512_set1_epi64(-1),
            }
        }
    }

    /// Flip masks for two squares in a single batch (`(x0, x1)` packed into
    /// the low / high 256-bit halves).
    #[inline(always)]
    pub fn flip_pair(&self, x0: usize, x1: usize) -> (u64, u64) {
        unsafe {
            flip_pair_zmm2_body!(x0, x1, self.pp, self.no, self.zero, self.msb, self.all_ones)
        }
    }

    /// Flip mask for a single square via the one-at-a-time path. Reuses the wide
    /// constants by truncating to 256 bits — no extra broadcasts.
    #[inline(always)]
    pub fn flip_one(&self, x: usize) -> u64 {
        unsafe {
            let pp = _mm512_castsi512_si256(self.pp);
            let no = _mm512_castsi512_si256(self.no);
            let zero = _mm512_castsi512_si256(self.zero);
            let msb = _mm512_castsi512_si256(self.msb);
            flip_prepared_ymm_body!(x, pp, no, zero, msb)
        }
    }
}

/// Computes flips for every set bit in `moves`, reusing `(p, !o)` and processing
/// two runtime squares at a time.
///
/// Results are written in increasing square order, exactly like a repeated
/// `trailing_zeros` loop.  The return value is the number of written entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512cd,avx512vl")]
#[inline]
pub fn flip_moves_pairwise(mut moves: u64, p: u64, o: u64, out: &mut [u64; 64]) -> usize {
    let pp = _mm512_set1_epi64(p as i64);
    let no = _mm512_set1_epi64((!o) as i64);
    let zero = _mm512_setzero_si512();
    let msb = _mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64);
    let all_ones = _mm512_set1_epi64(-1);

    let pp_ymm = _mm512_castsi512_si256(pp);
    let no_ymm = _mm512_castsi512_si256(no);
    let zero_ymm = _mm512_castsi512_si256(zero);
    let msb_ymm = _mm512_castsi512_si256(msb);

    let mut n = 0usize;
    let pair_count = moves.count_ones() as usize / 2;
    for _ in 0..pair_count {
        let x0 = moves.trailing_zeros() as usize;
        moves &= moves - 1;

        let x1 = moves.trailing_zeros() as usize;
        moves &= moves - 1;

        let (f0, f1) = flip_pair_zmm2_body!(x0, x1, pp, no, zero, msb, all_ones);
        out[n] = f0;
        out[n + 1] = f1;
        n += 2;
    }
    if moves != 0 {
        let x = moves.trailing_zeros() as usize;
        out[n] = flip_prepared_ymm_body!(x, pp_ymm, no_ymm, zero_ymm, msb_ymm);
        n += 1;
    }
    n
}

/// Same as `flip_moves_pairwise`, but also records the square indices
/// corresponding to each output flip mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512cd,avx512vl")]
#[inline]
pub fn flip_moves_pairwise_with_squares(
    mut moves: u64,
    p: u64,
    o: u64,
    squares: &mut [u8; 64],
    flips: &mut [u64; 64],
) -> usize {
    let pp = _mm512_set1_epi64(p as i64);
    let no = _mm512_set1_epi64((!o) as i64);
    let zero = _mm512_setzero_si512();
    let msb = _mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64);
    let all_ones = _mm512_set1_epi64(-1);

    let pp_ymm = _mm512_castsi512_si256(pp);
    let no_ymm = _mm512_castsi512_si256(no);
    let zero_ymm = _mm512_castsi512_si256(zero);
    let msb_ymm = _mm512_castsi512_si256(msb);

    let mut n = 0usize;
    let pair_count = moves.count_ones() as usize / 2;
    for _ in 0..pair_count {
        let x0 = moves.trailing_zeros() as usize;
        moves &= moves - 1;

        let x1 = moves.trailing_zeros() as usize;
        moves &= moves - 1;

        let (f0, f1) = flip_pair_zmm2_body!(x0, x1, pp, no, zero, msb, all_ones);
        squares[n] = x0 as u8;
        flips[n] = f0;
        squares[n + 1] = x1 as u8;
        flips[n + 1] = f1;
        n += 2;
    }
    if moves != 0 {
        let x = moves.trailing_zeros() as usize;
        squares[n] = x as u8;
        flips[n] = flip_prepared_ymm_body!(x, pp_ymm, no_ymm, zero_ymm, msb_ymm);
        n += 1;
    }
    n
}

/// One-at-a-time batch path. Useful as an A/B switch against `flip_moves_pairwise`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512cd,avx512vl")]
#[inline]
pub fn flip_moves_one_by_one(mut moves: u64, p: u64, o: u64, out: &mut [u64; 64]) -> usize {
    let pp = _mm256_set1_epi64x(p as i64);
    let no = _mm256_set1_epi64x((!o) as i64);
    let zero = _mm256_setzero_si256();
    let msb = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);

    let mut n = 0usize;
    while moves != 0 {
        let x = moves.trailing_zeros() as usize;
        moves &= moves - 1;
        out[n] = flip_prepared_ymm_body!(x, pp, no, zero, msb);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_moves_pairwise_matches_single_square_path() {
        let mut seed = 0x5125_1f1a_aa55_9669u64;

        for _ in 0..512 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let p = seed;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let o = seed & !p;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let moves = seed & !(p | o);

            let mut flips = [0u64; 64];
            let mut flips_with_squares = [0u64; 64];
            let mut squares = [0u8; 64];
            // SAFETY: this test module is compiled only with the AVX-512 backend.
            let n = unsafe { flip_moves_pairwise(moves, p, o, &mut flips) };
            // SAFETY: this test module is compiled only with the AVX-512 backend.
            let n_with_squares = unsafe {
                flip_moves_pairwise_with_squares(moves, p, o, &mut squares, &mut flips_with_squares)
            };
            assert_eq!(n, n_with_squares);
            assert_eq!(n, moves.count_ones() as usize);
            assert_eq!(flips[..n], flips_with_squares[..n]);

            let mut bb = moves;
            for i in 0..n {
                let x = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                assert_eq!(squares[i] as usize, x);
                assert_eq!(flips_with_squares[i], flip_index(x, p, o));
            }
            assert_eq!(bb, 0);
        }
    }
}
