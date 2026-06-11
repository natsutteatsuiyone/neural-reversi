//! AVX-512 flip backend for single-square and shared-board batch paths.
//!
//! Based on flip_avx512cd.c from edax-reversi.
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/flip_avx512cd.c>
//!
//! The single-square `flip(sq, p, o)` API is still the public dispatch target.
//! [`BoardCtx`] is the shared-board fast path: it broadcasts `(p, !o)` once and
//! evaluates one to four runtime squares for move-list and shallow endgame
//! callers. No const-square specialization is used.

// Some helper macros contain local `unsafe { ... }` blocks so they remain
// self-contained when expanded from safe `#[target_feature]` functions. When
// those same helpers are expanded inside a larger unsafe block, the local
// blocks become redundant and would otherwise fire `unused_unsafe`.
#![allow(unused_unsafe)]

use crate::square::Square;
use std::arch::x86_64::*;

// Raw variable shifts.
//
// Keep raw `vpsrlvq`: count 64 from `vplzcntq` must produce 0, but the
// intrinsic goes through LLVM `lshr` poison handling and adds a mask/test.
// Re-run the cargo-asm A/B before replacing this with an intrinsic.
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

// Same raw-shift rationale as `vpsrlvq_raw_ymm!`.
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

// Horizontal reductions.
#[cfg(target_arch = "x86_64")]
macro_rules! reduce_ymm_or_u64 {
    ($v:expr) => {{
        let x = _mm_or_si128(_mm256_castsi256_si128($v), _mm256_extracti128_si256($v, 1));
        let x = _mm_or_si128(x, _mm_shuffle_epi32(x, 0x4e));
        _mm_cvtsi128_si64(x) as u64
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! reduce_zmm_pair_or_u64 {
    ($flips:expr) => {{
        // OR-reduce each 256-bit half independently: f0 lands in lane 0, f1 in lane 4.
        let swap64 = _mm512_shuffle_epi32::<0x4e>($flips);
        let or64 = _mm512_or_si512($flips, swap64);
        let swap128 = _mm512_shuffle_i64x2::<0xb1>(or64, or64);
        let reduced = _mm512_or_si512(or64, swap128);
        let f0 = _mm_cvtsi128_si64(_mm512_castsi512_si128(reduced)) as u64;
        let f1 = _mm_cvtsi128_si64(_mm256_castsi256_si128(_mm512_extracti64x4_epi64::<1>(
            reduced,
        ))) as u64;
        (f0, f1)
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! reduce_zmm_two_pairs_or_u64 {
    ($flips01:expr, $flips23:expr) => {{
        let swap64_01 = _mm512_shuffle_epi32::<0x4e>($flips01);
        let swap64_23 = _mm512_shuffle_epi32::<0x4e>($flips23);
        let or64_01 = _mm512_or_si512($flips01, swap64_01);
        let or64_23 = _mm512_or_si512($flips23, swap64_23);

        let swap128_01 = _mm512_shuffle_i64x2::<0xb1>(or64_01, or64_01);
        let swap128_23 = _mm512_shuffle_i64x2::<0xb1>(or64_23, or64_23);
        let reduced01 = _mm512_or_si512(or64_01, swap128_01);
        let reduced23 = _mm512_or_si512(or64_23, swap128_23);

        let f0 = _mm_cvtsi128_si64(_mm512_castsi512_si128(reduced01)) as u64;
        let f1 = _mm_cvtsi128_si64(_mm256_castsi256_si128(_mm512_extracti64x4_epi64::<1>(
            reduced01,
        ))) as u64;
        let f2 = _mm_cvtsi128_si64(_mm512_castsi512_si128(reduced23)) as u64;
        let f3 = _mm_cvtsi128_si64(_mm256_castsi256_si128(_mm512_extracti64x4_epi64::<1>(
            reduced23,
        ))) as u64;
        (f0, f1, f2, f3)
    }};
}

// Flip kernels.
// The multi-square kernels intentionally keep the right-side LZCNT chains and
// left-side LS1B chains in one body. Splitting those chains into helper macros
// makes the code shorter but tends to hide the scheduling that protects this
// path from latency regressions.
#[cfg(target_arch = "x86_64")]
macro_rules! flip_prepared_ymm_body {
    ($x:expr, $pp:expr, $no:expr, $zero:expr, $msb:expr) => {{
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
        reduce_ymm_or_u64!(_mm256_ternarylogic_epi64(
            right_flips,
            left_flank,
            left_mask,
            0xf2,
        ))
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! flip_pair_body {
    ($x0:expr, $x1:expr, $pp:expr, $no:expr, $zero:expr, $msb:expr, $all_ones:expr) => {{
        // Each `LrmaskEntry` is one 64-byte-aligned cache line laid out as
        // `[left(4 x u64), right(4 x u64)]`. For the latency-sensitive
        // endgame `flip2` path, load right-side masks first so the LZCNT chain
        // starts before the independent left-side loads.
        let mask_ptr0 =
            unsafe { super::lrmask::LRMASK.get_unchecked($x0).0.as_ptr() as *const __m256i };
        let mask_ptr1 =
            unsafe { super::lrmask::LRMASK.get_unchecked($x1).0.as_ptr() as *const __m256i };
        let right0 = unsafe { _mm256_load_si256(mask_ptr0.add(1)) };
        let right1 = unsafe { _mm256_load_si256(mask_ptr1.add(1)) };
        let right_mask = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(right0), right1);

        let mut right_bit = _mm512_lzcnt_epi64(_mm512_and_si512($no, right_mask));

        let left0 = unsafe { _mm256_load_si256(mask_ptr0) };
        let left1 = unsafe { _mm256_load_si256(mask_ptr1) };
        let left_mask = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(left0), left1);
        let mut left_bit = _mm512_and_si512($no, left_mask);
        left_bit =
            _mm512_ternarylogic_epi64(left_bit, _mm512_sub_epi64($zero, left_bit), $pp, 0x80);

        // left_flank = signed-min(-left_bit, -1) = -1 if left_bit == 0 else -left_bit.
        // Relies on -left_bit being <= 0 for any non-negative left_bit so the
        // i64 min picks the more-negative side.
        let left_flank = _mm512_min_epi64(_mm512_sub_epi64($zero, left_bit), $all_ones);

        right_bit = _mm512_and_si512(vpsrlvq_raw_zmm!($msb, right_bit), $pp);
        let right_flips = _mm512_ternarylogic_epi64(
            _mm512_sub_epi64($zero, right_bit),
            right_bit,
            right_mask,
            0x28,
        );

        let flips = _mm512_ternarylogic_epi64(right_flips, left_flank, left_mask, 0xf2);
        reduce_zmm_pair_or_u64!(flips)
    }};
}

#[cfg(target_arch = "x86_64")]
macro_rules! flip_pair_wide_load_body {
    ($x0:expr, $x1:expr, $pp:expr, $no:expr, $zero:expr, $msb:expr, $all_ones:expr) => {{
        // Move-list generation calls `flip2` in a dense loop. Two 64B loads
        // plus lane shuffles have lower instruction count and benchmark better
        // there than the latency-first split-load schedule above.
        let z0 = unsafe {
            _mm512_load_si512(super::lrmask::LRMASK.get_unchecked($x0).0.as_ptr() as *const __m512i)
        };
        let z1 = unsafe {
            _mm512_load_si512(super::lrmask::LRMASK.get_unchecked($x1).0.as_ptr() as *const __m512i)
        };
        // 0x44 = 0b01_00_01_00: keep the low 256 (left halves) from each source.
        let left_mask = _mm512_shuffle_i64x2::<0x44>(z0, z1);
        // 0xee = 0b11_10_11_10: keep the high 256 (right halves) from each source.
        let right_mask = _mm512_shuffle_i64x2::<0xee>(z0, z1);

        let mut right_bit = _mm512_lzcnt_epi64(_mm512_and_si512($no, right_mask));

        let mut left_bit = _mm512_and_si512($no, left_mask);
        left_bit =
            _mm512_ternarylogic_epi64(left_bit, _mm512_sub_epi64($zero, left_bit), $pp, 0x80);

        // left_flank = signed-min(-left_bit, -1) = -1 if left_bit == 0 else -left_bit.
        // Relies on -left_bit being <= 0 for any non-negative left_bit so the
        // i64 min picks the more-negative side.
        let left_flank = _mm512_min_epi64(_mm512_sub_epi64($zero, left_bit), $all_ones);

        right_bit = _mm512_and_si512(vpsrlvq_raw_zmm!($msb, right_bit), $pp);
        let right_flips = _mm512_ternarylogic_epi64(
            _mm512_sub_epi64($zero, right_bit),
            right_bit,
            right_mask,
            0x28,
        );

        let flips = _mm512_ternarylogic_epi64(right_flips, left_flank, left_mask, 0xf2);
        reduce_zmm_pair_or_u64!(flips)
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

/// SIMD board context for runtime squares that share the same `(p, o)` board.
///
/// [`BoardCtx::new`] broadcasts `(p, !o)` and helper constants once. `flip2`,
/// `flip3`, and `flip4` reuse those broadcasts for paired ZMM work; `flip1`
/// derives the YMM constants with `_mm512_castsi512_si256` for the trailing
/// single square in move-list generation. All methods are `#[inline(always)]`
/// and are intended to fold into AVX-512-gated callers.
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
    /// Broadcasts `(p, !o)` and the working constants into wide vector lanes.
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

    /// Computes the flip mask for one runtime square.
    ///
    /// Reuses the wide constants by truncating to 256 bits; no extra broadcasts.
    #[inline(always)]
    pub fn flip1(&self, x: usize) -> u64 {
        unsafe {
            let pp = _mm512_castsi512_si256(self.pp);
            let no = _mm512_castsi512_si256(self.no);
            let zero = _mm512_castsi512_si256(self.zero);
            let msb = _mm512_castsi512_si256(self.msb);
            flip_prepared_ymm_body!(x, pp, no, zero, msb)
        }
    }

    /// Computes flip masks for two runtime squares sharing this board.
    ///
    /// The internal ZMM lanes are arranged as `(x0, x1)` in the low and high
    /// 256-bit halves.
    #[inline(always)]
    pub fn flip2(&self, x0: usize, x1: usize) -> (u64, u64) {
        unsafe { flip_pair_body!(x0, x1, self.pp, self.no, self.zero, self.msb, self.all_ones) }
    }

    /// Computes two flip masks with the load schedule that is fastest in the
    /// dense move-list loop.
    #[inline(always)]
    pub fn flip2_wide_load(&self, x0: usize, x1: usize) -> (u64, u64) {
        unsafe {
            flip_pair_wide_load_body!(x0, x1, self.pp, self.no, self.zero, self.msb, self.all_ones)
        }
    }

    /// Computes flip masks for three runtime squares sharing this board.
    ///
    /// `x0` uses the YMM one-at-a-time path, while `(x1, x2)` use the paired
    /// 512-bit path. `x0` gets the YMM chain because its reduction tail is
    /// shorter than the paired ZMM one, and callers consume the first flip
    /// first: shallow-solve callers feed `f0` straight into the next board.
    ///
    /// Both chains are issued from one body so the scheduler overlaps the
    /// paired `LZCNT` latency with the independent single-square work.
    #[inline(always)]
    pub fn flip3(&self, x0: usize, x1: usize, x2: usize) -> (u64, u64, u64) {
        unsafe {
            let mask_ptr0 = super::lrmask::LRMASK.get_unchecked(x0).0.as_ptr() as *const __m256i;
            let right_mask0 = _mm256_load_si256(mask_ptr0.add(1));

            let pp0 = _mm512_castsi512_si256(self.pp);
            let no0 = _mm512_castsi512_si256(self.no);
            let zero0 = _mm512_castsi512_si256(self.zero);
            let msb0 = _mm512_castsi512_si256(self.msb);
            let all_ones0 = _mm512_castsi512_si256(self.all_ones);

            // Start both high-latency right-side chains before doing either left side.
            let mut right_bit0 = _mm256_lzcnt_epi64(_mm256_and_si256(no0, right_mask0));

            let mask_ptr1 = super::lrmask::LRMASK.get_unchecked(x1).0.as_ptr() as *const __m256i;
            let mask_ptr2 = super::lrmask::LRMASK.get_unchecked(x2).0.as_ptr() as *const __m256i;
            let right1 = _mm256_load_si256(mask_ptr1.add(1));
            let right2 = _mm256_load_si256(mask_ptr2.add(1));
            let right_mask12 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(right1), right2);
            let mut right_bit12 = _mm512_lzcnt_epi64(_mm512_and_si512(self.no, right_mask12));

            let left_mask0 = _mm256_load_si256(mask_ptr0);
            let left1 = _mm256_load_si256(mask_ptr1);
            let left2 = _mm256_load_si256(mask_ptr2);
            let left_mask12 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(left1), left2);

            let mut left_bit0 = _mm256_and_si256(no0, left_mask0);
            let mut left_bit12 = _mm512_and_si512(self.no, left_mask12);
            left_bit0 =
                _mm256_ternarylogic_epi64(left_bit0, _mm256_sub_epi64(zero0, left_bit0), pp0, 0x80);
            left_bit12 = _mm512_ternarylogic_epi64(
                left_bit12,
                _mm512_sub_epi64(self.zero, left_bit12),
                self.pp,
                0x80,
            );

            let left_flank0 = _mm256_min_epi64(_mm256_sub_epi64(zero0, left_bit0), all_ones0);
            let left_flank12 =
                _mm512_min_epi64(_mm512_sub_epi64(self.zero, left_bit12), self.all_ones);

            right_bit0 = _mm256_and_si256(vpsrlvq_raw_ymm!(msb0, right_bit0), pp0);
            right_bit12 = _mm512_and_si512(vpsrlvq_raw_zmm!(self.msb, right_bit12), self.pp);

            let right_flips0 = _mm256_ternarylogic_epi64(
                _mm256_sub_epi64(zero0, right_bit0),
                right_bit0,
                right_mask0,
                0x28,
            );
            let right_flips12 = _mm512_ternarylogic_epi64(
                _mm512_sub_epi64(self.zero, right_bit12),
                right_bit12,
                right_mask12,
                0x28,
            );

            let flips0 = _mm256_ternarylogic_epi64(right_flips0, left_flank0, left_mask0, 0xf2);
            let flips12 = _mm512_ternarylogic_epi64(right_flips12, left_flank12, left_mask12, 0xf2);
            let f0 = reduce_ymm_or_u64!(flips0);
            let (f1, f2) = reduce_zmm_pair_or_u64!(flips12);
            (f0, f1, f2)
        }
    }

    /// Computes flip masks for four runtime squares sharing this board.
    ///
    /// The implementation runs two paired 512-bit chains over `(x0, x1)` and
    /// `(x2, x3)`.
    ///
    /// The two passes share one set of broadcast constants and are issued
    /// from one body, so the scheduler interleaves their independent
    /// dependency chains for instruction-level parallelism.
    #[inline(always)]
    pub fn flip4(&self, x0: usize, x1: usize, x2: usize, x3: usize) -> (u64, u64, u64, u64) {
        unsafe {
            let mask_ptr0 = super::lrmask::LRMASK.get_unchecked(x0).0.as_ptr() as *const __m256i;
            let mask_ptr1 = super::lrmask::LRMASK.get_unchecked(x1).0.as_ptr() as *const __m256i;
            let right0 = _mm256_load_si256(mask_ptr0.add(1));
            let right1 = _mm256_load_si256(mask_ptr1.add(1));
            let right_mask01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(right0), right1);
            let mut right_bit01 = _mm512_lzcnt_epi64(_mm512_and_si512(self.no, right_mask01));

            let mask_ptr2 = super::lrmask::LRMASK.get_unchecked(x2).0.as_ptr() as *const __m256i;
            let mask_ptr3 = super::lrmask::LRMASK.get_unchecked(x3).0.as_ptr() as *const __m256i;
            let right2 = _mm256_load_si256(mask_ptr2.add(1));
            let right3 = _mm256_load_si256(mask_ptr3.add(1));
            let right_mask23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(right2), right3);

            // Fire both right-side LZCNT chains before either waits on VPSRLVQ.
            let mut right_bit23 = _mm512_lzcnt_epi64(_mm512_and_si512(self.no, right_mask23));

            let left0 = _mm256_load_si256(mask_ptr0);
            let left1 = _mm256_load_si256(mask_ptr1);
            let left_mask01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(left0), left1);
            let left2 = _mm256_load_si256(mask_ptr2);
            let left3 = _mm256_load_si256(mask_ptr3);
            let left_mask23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(left2), left3);

            let mut left_bit01 = _mm512_and_si512(self.no, left_mask01);
            let mut left_bit23 = _mm512_and_si512(self.no, left_mask23);
            left_bit01 = _mm512_ternarylogic_epi64(
                left_bit01,
                _mm512_sub_epi64(self.zero, left_bit01),
                self.pp,
                0x80,
            );
            left_bit23 = _mm512_ternarylogic_epi64(
                left_bit23,
                _mm512_sub_epi64(self.zero, left_bit23),
                self.pp,
                0x80,
            );

            let left_flank01 =
                _mm512_min_epi64(_mm512_sub_epi64(self.zero, left_bit01), self.all_ones);
            let left_flank23 =
                _mm512_min_epi64(_mm512_sub_epi64(self.zero, left_bit23), self.all_ones);

            right_bit01 = _mm512_and_si512(vpsrlvq_raw_zmm!(self.msb, right_bit01), self.pp);
            right_bit23 = _mm512_and_si512(vpsrlvq_raw_zmm!(self.msb, right_bit23), self.pp);

            let right_flips01 = _mm512_ternarylogic_epi64(
                _mm512_sub_epi64(self.zero, right_bit01),
                right_bit01,
                right_mask01,
                0x28,
            );
            let right_flips23 = _mm512_ternarylogic_epi64(
                _mm512_sub_epi64(self.zero, right_bit23),
                right_bit23,
                right_mask23,
                0x28,
            );

            let flips01 = _mm512_ternarylogic_epi64(right_flips01, left_flank01, left_mask01, 0xf2);
            let flips23 = _mm512_ternarylogic_epi64(right_flips23, left_flank23, left_mask23, 0xf2);
            reduce_zmm_two_pairs_or_u64!(flips01, flips23)
        }
    }
}
