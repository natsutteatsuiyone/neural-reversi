//! Stability detection module.
//!
//! Stable discs are pieces that can never be flipped for the rest of the game.
//! This module provides fast stability estimation using a pre-computed edge
//! stability lookup table combined with full-line and contact-based analysis.
//!
//! The stability count is used for search pruning (stability cutoff): if the
//! opponent has enough stable discs to guarantee a score at or below alpha,
//! the subtree can be pruned without further search.
//!
//! The edge stability lookup table is generated at compile time.
//!
//! Reference: <https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c>
use crate::{
    bitboard::Bitboard, board::Board, score::DISC_SCORE_MAX, score::DiscScore, util::align::Align64,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::__m256i;

/// Size of the edge stability lookup table (256 * 256 for all possible edge configurations).
const EDGE_STABILITY_SIZE: usize = 256 * 256;
const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const FILE_A_INNER: u64 = 0x0001_0101_0101_0100;
const FILE_H_INNER: u64 = 0x0080_8080_8080_8000;
const CENTRAL_MASK: u64 = 0x007e_7e7e_7e7e_7e00;
const FILES_AB: u64 = 0x0303_0303_0303_0303;
const FILES_GH: u64 = 0xc0c0_c0c0_c0c0_c0c0;
const FILES_AD: u64 = 0x0f0f_0f0f_0f0f_0f0f;
const FILES_EH: u64 = 0xf0f0_f0f0_f0f0_f0f0;

/// Global edge stability lookup table.
static EDGE_STABILITY: Align64<[u8; EDGE_STABILITY_SIZE]> = Align64(init_edge_stability());

/// Simulates placing a disc at position `x` (0-7) on an 8-bit edge and applies
/// flips in both directions.
///
/// Returns `(player_after, opponent_after)`.
const fn apply_edge_move(x: u8, player: u8, opponent: u8) -> (u8, u8) {
    let mut p = player | x_to_bit(x);
    let mut o = opponent;
    if x > 1 {
        let mut y = x - 1;
        while y > 0 && (o & x_to_bit(y)) != 0 {
            y -= 1;
        }
        if (p & x_to_bit(y)) != 0 {
            let mut y = x - 1;
            while y > 0 && (o & x_to_bit(y)) != 0 {
                o ^= x_to_bit(y);
                p ^= x_to_bit(y);
                y -= 1;
            }
        }
    }
    if x < 6 {
        let mut y = x + 1;
        while y < 8 && (o & x_to_bit(y)) != 0 {
            y += 1;
        }
        if y < 8 && (p & x_to_bit(y)) != 0 {
            let mut y = x + 1;
            while y < 8 && (o & x_to_bit(y)) != 0 {
                o ^= x_to_bit(y);
                p ^= x_to_bit(y);
                y += 1;
            }
        }
    }
    (p, o)
}

/// Computes stable discs for one edge from child states whose occupancy is one
/// disc higher.
const fn find_edge_stable_from_table(
    old_p: u8,
    old_o: u8,
    table: &[u8; EDGE_STABILITY_SIZE],
) -> u8 {
    let mut stable = old_p;
    let e = !(old_p | old_o);

    if stable == 0 || e == 0 {
        return stable;
    }

    let mut e_remaining = e;
    while e_remaining != 0 {
        let x = e_remaining.trailing_zeros() as u8;

        // Simulate player's move at position x.
        let (p, o) = apply_edge_move(x, old_p, old_o);
        stable &= table[edge_stability_index(p, o)];
        if stable == 0 {
            return stable;
        }

        // Simulate opponent's move at position x.
        let (o, p) = apply_edge_move(x, old_o, old_p);
        stable &= table[edge_stability_index(p, o)];
        if stable == 0 {
            return stable;
        }

        e_remaining &= e_remaining - 1;
    }

    stable
}

/// Converts a position index (0-7) to its corresponding bit mask.
const fn x_to_bit(x: u8) -> u8 {
    1u8 << x
}

const fn edge_stability_index(player_edge: u8, opponent_edge: u8) -> usize {
    ((player_edge as usize) << 8) | opponent_edge as usize
}

/// Computes the edge stability lookup table.
///
/// Returns a 65536-entry table indexed by `player_edge * 256 + opponent_edge`.
const fn init_edge_stability() -> [u8; EDGE_STABILITY_SIZE] {
    let mut table: [u8; EDGE_STABILITY_SIZE] = [0; EDGE_STABILITY_SIZE];

    // Build occupancies high to low so each lookup reads an entry already filled.
    // Within an occupancy, walk the disjoint (p, o = m & !p) splits via
    // carry-rippler subset enumeration over m.
    let mut occupied: usize = 8;
    loop {
        let mut m_u16: u16 = 0;
        while m_u16 < 256 {
            let m = m_u16 as u8;
            if m.count_ones() as usize == occupied {
                let mut p: u8 = 0;
                loop {
                    let o = m & !p;
                    table[edge_stability_index(p, o)] = find_edge_stable_from_table(p, o, &table);
                    p = p.wrapping_sub(m) & m;
                    if p == 0 {
                        break;
                    }
                }
            }
            m_u16 += 1;
        }

        if occupied == 0 {
            break;
        }
        occupied -= 1;
    }

    table
}

/// Unpacks bits 1-6 of an edge stability byte to the A2-A7 squares on the board.
#[inline]
fn unpack_a2a7(x: u8) -> u64 {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "bmi2") => {
            use std::arch::x86_64::_pdep_u64;

            // SAFETY: this branch is compiled only when BMI2 is enabled for
            // the current target.
            unsafe { _pdep_u64(u64::from((x & 0x7e) >> 1), FILE_A_INNER) }
        }
        _ => {
            let a = (x & 0x7e) as u64; // Extract bits 1-6
            (a.wrapping_mul(0x0000_0408_1020_4080u64)) & FILE_A_INNER
        }
    }
}

/// Unpacks bits 1-6 of an edge stability byte to the H2-H7 squares on the board.
#[inline]
fn unpack_h2h7(x: u8) -> u64 {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "bmi2") => {
            use std::arch::x86_64::_pdep_u64;

            // SAFETY: this branch is compiled only when BMI2 is enabled for
            // the current target.
            unsafe { _pdep_u64(u64::from((x & 0x7e) >> 1), FILE_H_INNER) }
        }
        _ => {
            let a = (x & 0x7e) as u64; // Extract bits 1-6
            (a.wrapping_mul(0x0002_0408_1020_4000u64)) & FILE_H_INNER
        }
    }
}

/// Packs the A-file (A1-A8) of a bitboard into a single byte.
#[inline]
fn pack_a1a8(x: u64) -> usize {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "bmi2") => {
            use std::arch::x86_64::_pext_u64;

            // SAFETY: this branch is compiled only when BMI2 is enabled for
            // the current target.
            unsafe { _pext_u64(x, FILE_A) as usize }
        }
        _ => {
            let a = x & FILE_A; // Mask A-file
            ((a.wrapping_mul(0x0102_0408_1020_4080u64)) >> 56) as usize
        }
    }
}

/// Packs the H-file (H1-H8) of a bitboard into a single byte.
#[inline]
fn pack_h1h8(x: u64) -> usize {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "bmi2") => {
            use std::arch::x86_64::_pext_u64;

            // SAFETY: this branch is compiled only when BMI2 is enabled for
            // the current target.
            unsafe { _pext_u64(x, FILE_H) as usize }
        }
        _ => {
            let a = x & FILE_H; // Mask H-file
            ((a.wrapping_mul(0x0002_0408_1020_4081u64)) >> 56) as usize
        }
    }
}

#[inline(always)]
fn edge_stability(player_edge: usize, opponent_edge: usize) -> u8 {
    debug_assert!(player_edge < 256);
    debug_assert!(opponent_edge < 256);

    // SAFETY: callers pass packed edges, so each index component is 0..256 and
    // the combined table index is within EDGE_STABILITY_SIZE.
    unsafe {
        *EDGE_STABILITY
            .0
            .get_unchecked((player_edge << 8) | opponent_edge)
    }
}

/// Returns stable discs along all four edges (rank 1, rank 8, A-file, H-file).
///
/// Uses the pre-computed edge stability table for fast lookup.
#[inline]
fn get_stable_edge(p: u64, o: u64) -> u64 {
    u64::from(edge_stability((p & 0xff) as usize, (o & 0xff) as usize))
        | u64::from(edge_stability((p >> 56) as usize, (o >> 56) as usize)) << 56
        | unpack_a2a7(edge_stability(pack_a1a8(p), pack_a1a8(o)))
        | unpack_h2h7(edge_stability(pack_h1h8(p), pack_h1h8(o)))
}

#[derive(Clone, Copy)]
struct LineMasks {
    horizontal: u64,
    vertical: u64,
    diag_7: u64,
    diag_9: u64,
}

impl LineMasks {
    #[inline(always)]
    fn all_directions(self) -> u64 {
        self.horizontal & self.vertical & self.diag_7 & self.diag_9
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl LineMasks {
    #[target_feature(enable = "avx2")]
    #[inline]
    fn to_lanes(self) -> __m256i {
        std::arch::x86_64::_mm256_set_epi64x(
            self.diag_9 as i64,
            self.diag_7 as i64,
            self.vertical as i64,
            self.horizontal as i64,
        )
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    fn from_lanes(lanes: __m256i) -> Self {
        use std::arch::x86_64::*;
        let lo = _mm256_castsi256_si128(lanes);
        let hi = _mm256_extracti128_si256(lanes, 1);
        Self {
            horizontal: _mm_cvtsi128_si64(lo) as u64,
            vertical: _mm_extract_epi64(lo, 1) as u64,
            diag_7: _mm_cvtsi128_si64(hi) as u64,
            diag_9: _mm_extract_epi64(hi, 1) as u64,
        }
    }
}

/// Detects completely filled lines in all four directions.
fn get_full_lines(disc: u64) -> LineMasks {
    let mut h = disc; // Horizontal
    let mut v = disc; // Vertical
    let mut l7 = disc; // Left diagonal (/), shift by 7
    let mut l9 = disc; // Left diagonal (\), shift by 9
    let mut r7 = disc; // Right diagonal (/), shift by 7
    let mut r9 = disc; // Right diagonal (\), shift by 9

    // Check horizontal lines: fold all 8 bits together
    h &= h >> 1; // Check 2 consecutive discs
    h &= h >> 2; // Check 4 consecutive discs
    h &= h >> 4; // Check 8 consecutive discs
    let horizontal = (h & 0x0101010101010101) * 0xff;

    // Check vertical lines: fold all 8 rows together
    v &= v.rotate_right(8); // Check 2 consecutive discs
    v &= v.rotate_right(16); // Check 4 consecutive discs
    v &= v.rotate_left(32); // Check 8 consecutive discs
    let vertical = v;

    l7 &= 0xff01010101010101 | (l7 >> 7);
    r7 &= 0x80808080808080ff | (r7 << 7);
    l7 &= 0xffff030303030303 | (l7 >> 14);
    r7 &= 0xc0c0c0c0c0c0ffff | (r7 << 14);
    l7 &= 0xffffffff0f0f0f0f | (l7 >> 28);
    r7 &= 0xf0f0f0f0ffffffff | (r7 << 28);
    let diag_7 = l7 & r7;

    l9 &= 0xff80808080808080 | (l9 >> 9);
    r9 &= 0x01010101010101ff | (r9 << 9);
    l9 &= 0xffffc0c0c0c0c0c0 | (l9 >> 18);
    r9 &= 0x030303030303ffff | (r9 << 18);
    let diag_9 = l9 & r9 & (0x0f0f0f0ff0f0f0f0 | (l9 >> 36) | (r9 << 36));

    LineMasks {
        horizontal,
        vertical,
        diag_7,
        diag_9,
    }
}

/// Expands the stable disc set by contact propagation.
fn get_stable_by_contact(central_mask: u64, previous_stable: u64, safe_lines: LineMasks) -> u64 {
    let mut stable_h: u64; // Stable in horizontal direction
    let mut stable_v: u64; // Stable in vertical direction
    let mut stable_d7: u64; // Stable in diagonal (/) direction
    let mut stable_d9: u64; // Stable in diagonal (\) direction
    let mut old_stable = 0;
    let mut stable = previous_stable | (safe_lines.all_directions() & central_mask);

    // Iteratively expand stable region until no new stable discs are found
    while stable != old_stable {
        old_stable = stable;

        stable_h = (stable >> 1) | (stable << 1) | safe_lines.horizontal;
        stable_v = (stable >> 8) | (stable << 8) | safe_lines.vertical;
        stable_d7 = (stable >> 7) | (stable << 7) | safe_lines.diag_7;
        stable_d9 = (stable >> 9) | (stable << 9) | safe_lines.diag_9;

        // A disc is stable if it's stable in all four directions
        stable |= stable_h & stable_v & stable_d7 & stable_d9 & central_mask;
    }
    stable
}

/// Returns a conservative subset of the player's stable discs.
///
/// Stable discs are pieces that can never be flipped for the rest of the game.
/// The result may underestimate but never overestimates: every returned disc
/// is guaranteed stable. The pipeline runs in three stages:
/// 1. Edge stability from a pre-computed lookup table
/// 2. Full-line detection (discs on completely filled lines)
/// 3. Contact propagation (discs adjacent to already-stable discs)
#[inline]
pub fn get_stable_discs(player: Bitboard, opponent: Bitboard) -> Bitboard {
    let central_mask = player.bits() & CENTRAL_MASK;
    let full = get_full_lines(player.bits() | opponent.bits());
    let stable = get_stable_edge(player.bits(), opponent.bits());
    Bitboard::new(get_stable_by_contact(central_mask, stable, full))
}

/// Threshold values for null window search (NWS) stability cutoff.
#[rustfmt::skip]
const NWS_STABILITY_THRESHOLD: [i8; 64] = [
    99, 99, 99, 99,  6,  8, 10, 12,  // 0-7 empties
    14, 16, 20, 22, 24, 26, 28, 30,  // 8-15 empties
    32, 34, 36, 38, 40, 42, 44, 46,  // 16-23 empties
    48, 48, 50, 50, 52, 52, 54, 54,  // 24-31 empties
    56, 56, 58, 58, 60, 60, 62, 62,  // 32-39 empties
    64, 64, 64, 64, 64, 64, 64, 64,  // 40-47 empties
    99, 99, 99, 99, 99, 99, 99, 99,  // 48-55 empties (no stable squares)
    99, 99, 99, 99, 99, 99, 99, 99   // 56-63 empties (no stable squares)
];

const STABILITY_CUTOFF_DISC_MARGIN: i32 = 8;
const SINGLE_EMPTY_MAX_SHORTFALL: i32 = 4;

/// Returns whether the opponent has enough discs that stability can plausibly
/// prove `score <= alpha`.
///
/// Requires the opponent to clear the strict cutoff bound by
/// [`STABILITY_CUTOFF_DISC_MARGIN`]: `get_stable_discs` typically returns
/// far fewer stable discs than `opponent.count()`, so the sweep is skipped
/// when surplus is thin.
#[inline(always)]
fn has_enough_opponent_discs_for_cutoff(opponent_count: u32, alpha: DiscScore) -> bool {
    2 * opponent_count as i32
        >= DISC_SCORE_MAX.value() - alpha.value() + 2 * STABILITY_CUTOFF_DISC_MARGIN
}

struct AxisSpec {
    step_1: u32,
    no_wrap_l1: u64,
    no_wrap_l2: u64,
    no_wrap_l4: u64,
    no_wrap_r1: u64,
    no_wrap_r2: u64,
    no_wrap_r4: u64,
}

impl AxisSpec {
    #[inline(always)]
    const fn step_2(&self) -> u32 {
        self.step_1 * 2
    }

    #[inline(always)]
    const fn step_4(&self) -> u32 {
        self.step_1 * 4
    }
}

const AXIS_H: AxisSpec = AxisSpec {
    step_1: 1,
    no_wrap_l1: !FILE_A,
    no_wrap_l2: !FILES_AB,
    no_wrap_l4: !FILES_AD,
    no_wrap_r1: !FILE_H,
    no_wrap_r2: !FILES_GH,
    no_wrap_r4: !FILES_EH,
};
const AXIS_V: AxisSpec = AxisSpec {
    step_1: 8,
    no_wrap_l1: !0,
    no_wrap_l2: !0,
    no_wrap_l4: !0,
    no_wrap_r1: !0,
    no_wrap_r2: !0,
    no_wrap_r4: !0,
};
const AXIS_D7: AxisSpec = AxisSpec {
    step_1: 7,
    no_wrap_l1: !FILE_H,
    no_wrap_l2: !FILES_GH,
    no_wrap_l4: !FILES_EH,
    no_wrap_r1: !FILE_A,
    no_wrap_r2: !FILES_AB,
    no_wrap_r4: !FILES_AD,
};
const AXIS_D9: AxisSpec = AxisSpec {
    step_1: 9,
    no_wrap_l1: !FILE_A,
    no_wrap_l2: !FILES_AB,
    no_wrap_l4: !FILES_AD,
    no_wrap_r1: !FILE_H,
    no_wrap_r2: !FILES_GH,
    no_wrap_r4: !FILES_EH,
};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
fn axis_lanes(value: impl Fn(&AxisSpec) -> u64) -> __m256i {
    LineMasks {
        horizontal: value(&AXIS_H),
        vertical: value(&AXIS_V),
        diag_7: value(&AXIS_D7),
        diag_9: value(&AXIS_D9),
    }
    .to_lanes()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
fn scan_step_avx2(
    c1: __m256i,
    c2: __m256i,
    sh1: __m256i,
    sh2: __m256i,
    no_wrap: __m256i,
) -> (__m256i, __m256i) {
    use std::arch::x86_64::*;

    let next_c2 = _mm256_or_si256(
        _mm256_or_si256(c2, _mm256_and_si256(_mm256_and_si256(c1, sh1), no_wrap)),
        _mm256_and_si256(sh2, no_wrap),
    );
    let next_c1 = _mm256_or_si256(c1, _mm256_and_si256(sh1, no_wrap));
    (next_c1, next_c2)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
fn scan_empties_avx2(empties: __m256i) -> (__m256i, __m256i, __m256i) {
    use std::arch::x86_64::*;

    let s1 = axis_lanes(|axis| u64::from(axis.step_1));
    let s2 = axis_lanes(|axis| u64::from(axis.step_2()));
    let s4 = axis_lanes(|axis| u64::from(axis.step_4()));
    let nl1 = axis_lanes(|axis| axis.no_wrap_l1);
    let nl2 = axis_lanes(|axis| axis.no_wrap_l2);
    let nl4 = axis_lanes(|axis| axis.no_wrap_l4);
    let nr1 = axis_lanes(|axis| axis.no_wrap_r1);
    let nr2 = axis_lanes(|axis| axis.no_wrap_r2);
    let nr4 = axis_lanes(|axis| axis.no_wrap_r4);

    let sh_l1 = _mm256_sllv_epi64(empties, s1);
    let l1 = _mm256_or_si256(empties, _mm256_and_si256(sh_l1, nl1));
    let l2 = _mm256_and_si256(_mm256_and_si256(empties, sh_l1), nl1);
    let (l1, l2) = scan_step_avx2(
        l1,
        l2,
        _mm256_sllv_epi64(l1, s2),
        _mm256_sllv_epi64(l2, s2),
        nl2,
    );
    let (l1, l2) = scan_step_avx2(
        l1,
        l2,
        _mm256_sllv_epi64(l1, s4),
        _mm256_sllv_epi64(l2, s4),
        nl4,
    );

    let sh_r1 = _mm256_srlv_epi64(empties, s1);
    let r1 = _mm256_or_si256(empties, _mm256_and_si256(sh_r1, nr1));
    let r2 = _mm256_and_si256(_mm256_and_si256(empties, sh_r1), nr1);
    let (r1, r2) = scan_step_avx2(
        r1,
        r2,
        _mm256_srlv_epi64(r1, s2),
        _mm256_srlv_epi64(r2, s2),
        nr2,
    );
    let (r1, r2) = scan_step_avx2(
        r1,
        r2,
        _mm256_srlv_epi64(r1, s4),
        _mm256_srlv_epi64(r2, s4),
        nr4,
    );

    (
        _mm256_or_si256(l1, r1),
        _mm256_andnot_si256(_mm256_or_si256(l2, r1), l1),
        _mm256_andnot_si256(_mm256_or_si256(r2, l1), r1),
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
fn adjacent_lanes_avx2(bits: __m256i) -> __m256i {
    use std::arch::x86_64::*;

    let s1 = axis_lanes(|axis| u64::from(axis.step_1));
    let l = _mm256_and_si256(
        _mm256_sllv_epi64(bits, s1),
        axis_lanes(|axis| axis.no_wrap_l1),
    );
    let r = _mm256_and_si256(
        _mm256_srlv_epi64(bits, s1),
        axis_lanes(|axis| axis.no_wrap_r1),
    );
    _mm256_or_si256(l, r)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
fn lock_other_axes_avx2(discs: u64, unsafe_lanes: __m256i) -> __m256i {
    use std::arch::x86_64::*;

    let discs = _mm256_set1_epi64x(discs as i64);
    let swap_pair = _mm256_shuffle_epi32(unsafe_lanes, 0x4e);
    let swap_halves = _mm256_permute4x64_epi64(unsafe_lanes, 0x4e);
    let reverse = _mm256_permute4x64_epi64(unsafe_lanes, 0x1b);
    let safe_pair = _mm256_andnot_si256(swap_pair, discs);
    _mm256_andnot_si256(_mm256_or_si256(swap_halves, reverse), safe_pair)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
#[inline(always)]
fn propagate_avx2(locked: __m256i, one_l: __m256i, one_r: __m256i) -> (__m256i, __m256i) {
    use std::arch::x86_64::*;

    let s1 = axis_lanes(|axis| u64::from(axis.step_1));
    let s2 = axis_lanes(|axis| u64::from(axis.step_2()));
    let s4 = axis_lanes(|axis| u64::from(axis.step_4()));

    let mut l = _mm256_and_si256(locked, one_l);
    l = _mm256_or_si256(
        l,
        _mm256_and_si256(_mm256_sllv_epi64(l, s1), axis_lanes(|axis| axis.no_wrap_l1)),
    );
    l = _mm256_or_si256(
        l,
        _mm256_and_si256(_mm256_sllv_epi64(l, s2), axis_lanes(|axis| axis.no_wrap_l2)),
    );
    l = _mm256_or_si256(
        l,
        _mm256_and_si256(_mm256_sllv_epi64(l, s4), axis_lanes(|axis| axis.no_wrap_l4)),
    );

    let mut r = _mm256_and_si256(locked, one_r);
    r = _mm256_or_si256(
        r,
        _mm256_and_si256(_mm256_srlv_epi64(r, s1), axis_lanes(|axis| axis.no_wrap_r1)),
    );
    r = _mm256_or_si256(
        r,
        _mm256_and_si256(_mm256_srlv_epi64(r, s2), axis_lanes(|axis| axis.no_wrap_r2)),
    );
    r = _mm256_or_si256(
        r,
        _mm256_and_si256(_mm256_srlv_epi64(r, s4), axis_lanes(|axis| axis.no_wrap_r4)),
    );

    (l, r)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
fn single_empty_line_stable_avx2(player: u64, opponent: u64, stable: u64) -> u64 {
    use std::arch::x86_64::*;

    let opp_edge = get_stable_edge(opponent, player);
    let empties = _mm256_set1_epi64x(!(player | opponent) as i64);
    let (has_empty, one_l, one_r) = scan_empties_avx2(empties);

    let prev_touch = adjacent_lanes_avx2(_mm256_set1_epi64x(stable as i64));
    let p_unsafe = _mm256_andnot_si256(prev_touch, has_empty);
    let p_locked = lock_other_axes_avx2(player, p_unsafe);
    let (pl, pr) = propagate_avx2(p_locked, one_l, one_r);

    let opp_edge_touch = adjacent_lanes_avx2(_mm256_set1_epi64x(opp_edge as i64));
    let opp_unsafe = _mm256_andnot_si256(
        _mm256_or_si256(pl, pr),
        _mm256_andnot_si256(opp_edge_touch, has_empty),
    );
    let opp_locked = lock_other_axes_avx2(opponent, opp_unsafe);
    let (bl, br) = propagate_avx2(opp_locked, one_l, one_r);

    let base_safe = _mm256_xor_si256(p_unsafe, _mm256_set1_epi64x(-1));
    let safe_lines = LineMasks::from_lanes(_mm256_or_si256(base_safe, _mm256_or_si256(bl, br)));

    get_stable_by_contact(player & CENTRAL_MASK, stable, safe_lines)
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn scan_axis_scalar(e: u64, axis: &AxisSpec) -> (u64, u64, u64) {
    let ml = (e << axis.step_1) & axis.no_wrap_l1;
    let mut l1 = e | ml;
    let mut l2 = e & ml;
    let ml = (l1 << axis.step_2()) & axis.no_wrap_l2;
    l2 = l2 | (l1 & ml) | ((l2 << axis.step_2()) & axis.no_wrap_l2);
    l1 |= ml;
    let ml = (l1 << axis.step_4()) & axis.no_wrap_l4;
    l2 = l2 | (l1 & ml) | ((l2 << axis.step_4()) & axis.no_wrap_l4);
    l1 |= ml;

    let mr = (e >> axis.step_1) & axis.no_wrap_r1;
    let mut r1 = e | mr;
    let mut r2 = e & mr;
    let mr = (r1 >> axis.step_2()) & axis.no_wrap_r2;
    r2 = r2 | (r1 & mr) | ((r2 >> axis.step_2()) & axis.no_wrap_r2);
    r1 |= mr;
    let mr = (r1 >> axis.step_4()) & axis.no_wrap_r4;
    r2 = r2 | (r1 & mr) | ((r2 >> axis.step_4()) & axis.no_wrap_r4);
    r1 |= mr;

    (l1 | r1, l1 & !(l2 | r1), r1 & !(r2 | l1))
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn adjacent_axis(bits: u64, axis: &AxisSpec) -> u64 {
    ((bits << axis.step_1) & axis.no_wrap_l1) | ((bits >> axis.step_1) & axis.no_wrap_r1)
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn lock_other_axes_scalar(discs: u64, unsafe_lines: LineMasks) -> LineMasks {
    let no_hv = discs & !(unsafe_lines.horizontal | unsafe_lines.vertical);
    let no_diag = discs & !(unsafe_lines.diag_7 | unsafe_lines.diag_9);
    LineMasks {
        horizontal: no_diag & !unsafe_lines.vertical,
        vertical: no_diag & !unsafe_lines.horizontal,
        diag_7: no_hv & !unsafe_lines.diag_9,
        diag_9: no_hv & !unsafe_lines.diag_7,
    }
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
#[inline(always)]
fn propagate_axis(locked: u64, one_l: u64, one_r: u64, axis: &AxisSpec) -> u64 {
    let mut l = locked & one_l;
    l |= (l << axis.step_1) & axis.no_wrap_l1;
    l |= (l << axis.step_2()) & axis.no_wrap_l2;
    l |= (l << axis.step_4()) & axis.no_wrap_l4;

    let mut r = locked & one_r;
    r |= (r >> axis.step_1) & axis.no_wrap_r1;
    r |= (r >> axis.step_2()) & axis.no_wrap_r2;
    r |= (r >> axis.step_4()) & axis.no_wrap_r4;

    l | r
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "avx2"))))]
fn single_empty_line_stable_scalar(player: u64, opponent: u64, stable: u64) -> u64 {
    let empties = !(player | opponent);
    let opp_edge = get_stable_edge(opponent, player);
    let (has_h, one_hl, one_hr) = scan_axis_scalar(empties, &AXIS_H);
    let (has_v, one_vl, one_vr) = scan_axis_scalar(empties, &AXIS_V);
    let (has_d7, one_7l, one_7r) = scan_axis_scalar(empties, &AXIS_D7);
    let (has_d9, one_9l, one_9r) = scan_axis_scalar(empties, &AXIS_D9);

    let p_unsafe = LineMasks {
        horizontal: has_h & !adjacent_axis(stable, &AXIS_H),
        vertical: has_v & !adjacent_axis(stable, &AXIS_V),
        diag_7: has_d7 & !adjacent_axis(stable, &AXIS_D7),
        diag_9: has_d9 & !adjacent_axis(stable, &AXIS_D9),
    };
    let p_locked = lock_other_axes_scalar(player, p_unsafe);

    let opp_unsafe = LineMasks {
        horizontal: has_h
            & !(adjacent_axis(opp_edge, &AXIS_H)
                | propagate_axis(p_locked.horizontal, one_hl, one_hr, &AXIS_H)),
        vertical: has_v
            & !(adjacent_axis(opp_edge, &AXIS_V)
                | propagate_axis(p_locked.vertical, one_vl, one_vr, &AXIS_V)),
        diag_7: has_d7
            & !(adjacent_axis(opp_edge, &AXIS_D7)
                | propagate_axis(p_locked.diag_7, one_7l, one_7r, &AXIS_D7)),
        diag_9: has_d9
            & !(adjacent_axis(opp_edge, &AXIS_D9)
                | propagate_axis(p_locked.diag_9, one_9l, one_9r, &AXIS_D9)),
    };
    let opp_locked = lock_other_axes_scalar(opponent, opp_unsafe);

    let safe_lines = LineMasks {
        horizontal: !p_unsafe.horizontal
            | propagate_axis(opp_locked.horizontal, one_hl, one_hr, &AXIS_H),
        vertical: !p_unsafe.vertical | propagate_axis(opp_locked.vertical, one_vl, one_vr, &AXIS_V),
        diag_7: !p_unsafe.diag_7 | propagate_axis(opp_locked.diag_7, one_7l, one_7r, &AXIS_D7),
        diag_9: !p_unsafe.diag_9 | propagate_axis(opp_locked.diag_9, one_9l, one_9r, &AXIS_D9),
    };

    get_stable_by_contact(player & CENTRAL_MASK, stable, safe_lines)
}

#[inline(always)]
fn single_empty_line_stable_impl(player: u64, opponent: u64, stable: u64) -> u64 {
    cfg_select! {
        all(target_arch = "x86_64", target_feature = "avx2") => unsafe {
            single_empty_line_stable_avx2(player, opponent, stable)
        },
        _ => single_empty_line_stable_scalar(player, opponent, stable),
    }
}

#[cold]
#[inline(never)]
fn stability_cutoff_single_empty(
    player: u64,
    opponent: u64,
    stable: u64,
    alpha: DiscScore,
) -> Option<DiscScore> {
    let stable = single_empty_line_stable_impl(player, opponent, stable);
    let score = DiscScore::new(DISC_SCORE_MAX.value() - 2 * stable.count_ones() as i32);
    (score <= alpha).then_some(score)
}

/// Attempts to prove an alpha cutoff using stability analysis.
///
/// If the opponent has enough stable discs to guarantee a final score at or
/// below `alpha`, returns that score immediately.
///
/// Returns [`None`] if the cutoff cannot be proven.
#[inline(always)]
pub fn stability_cutoff(board: &Board, n_empties: u32, alpha: DiscScore) -> Option<DiscScore> {
    let &threshold = NWS_STABILITY_THRESHOLD.get(n_empties as usize)?;
    if alpha.value() >= i32::from(threshold) {
        if !has_enough_opponent_discs_for_cutoff(board.opponent().count(), alpha) {
            return None;
        }

        let stable = get_stable_discs(board.opponent(), board.player());
        let score = DiscScore::new(DISC_SCORE_MAX.value() - 2 * stable.count() as i32);
        if score <= alpha {
            return Some(score);
        }
        if score.value() - alpha.value() <= 2 * SINGLE_EMPTY_MAX_SHORTFALL {
            return stability_cutoff_single_empty(
                board.opponent().bits(),
                board.player().bits(),
                stable.bits(),
                alpha,
            );
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    #[test]
    fn stability_cutoff_returns_none_on_the_empty_board() {
        let board = Board::from_bitboards(0_u64, 0_u64);
        assert_eq!(board.empty_count(), 64);
        assert_eq!(
            stability_cutoff(&board, board.empty_count(), DiscScore::new(64)),
            None
        );
    }

    fn reference_find_edge_stable(old_p: u8, old_o: u8, mut stable: u8) -> u8 {
        let e = !(old_p | old_o);

        stable &= old_p;
        if stable == 0 || e == 0 {
            return stable;
        }

        for x in 0..8 {
            if (e & x_to_bit(x)) == 0 {
                continue;
            }

            let (p, o) = apply_edge_move(x, old_p, old_o);
            stable = reference_find_edge_stable(p, o, stable);
            if stable == 0 {
                return stable;
            }

            let (o, p) = apply_edge_move(x, old_o, old_p);
            stable = reference_find_edge_stable(p, o, stable);
            if stable == 0 {
                return stable;
            }
        }

        stable
    }

    #[test]
    fn edge_table_matches_recursive_reference() {
        let cases = [
            (0x00, 0x00),
            (0xff, 0x00),
            (0x00, 0xff),
            (0x81, 0x00),
            (0x7e, 0x00),
            (0x18, 0x24),
            (0x42, 0x3c),
            (0xa5, 0x5a),
            (0x55, 0xaa),
        ];

        for (p, o) in cases {
            assert_eq!(
                edge_stability(p as usize, o as usize),
                reference_find_edge_stable(p, o, p),
                "edge stability mismatch for p={p:#04x} o={o:#04x}",
            );
        }

        let mut rng = StdRng::seed_from_u64(0x0057_ab1e);
        for _ in 0..512 {
            let p: u8 = rng.random();
            let o: u8 = rng.random::<u8>() & !p;
            assert_eq!(
                edge_stability(p as usize, o as usize),
                reference_find_edge_stable(p, o, p),
                "edge stability mismatch for p={p:#04x} o={o:#04x}",
            );
        }
    }

    #[test]
    fn file_packers_match_expected_bit_order() {
        for byte in 0..=255u64 {
            let mut file_a = 0u64;
            let mut file_h = 0u64;

            for rank in 0..8 {
                if ((byte >> rank) & 1) != 0 {
                    file_a |= 1u64 << (rank * 8);
                    file_h |= 1u64 << (rank * 8 + 7);
                }
            }

            assert_eq!(
                pack_a1a8(file_a | (0xa5a5_a5a5_a5a5_a5a5 & !FILE_A)),
                byte as usize
            );
            assert_eq!(
                pack_h1h8(file_h | (0xa5a5_a5a5_a5a5_a5a5 & !FILE_H)),
                byte as usize
            );
            assert_eq!(unpack_a2a7(byte as u8), file_a & FILE_A_INNER);
            assert_eq!(unpack_h2h7(byte as u8), file_h & FILE_H_INNER);
        }
    }

    #[test]
    fn stability_cutoff_candidate_uses_disc_margin_heuristic() {
        let m = STABILITY_CUTOFF_DISC_MARGIN as u32;
        assert!(!has_enough_opponent_discs_for_cutoff(
            31 + m,
            DiscScore::new(1)
        ));
        assert!(has_enough_opponent_discs_for_cutoff(
            31 + m,
            DiscScore::new(2)
        ));
        assert!(!has_enough_opponent_discs_for_cutoff(
            8 + m,
            DiscScore::new(47)
        ));
        assert!(has_enough_opponent_discs_for_cutoff(
            8 + m,
            DiscScore::new(48)
        ));
    }

    /// Returns whether some line of play from `board` flips a disc of `discs`.
    ///
    /// `discs` belong to the side to move if `owned_by_mover`, otherwise to
    /// the other side.
    fn reference_can_be_flipped(board: Board, discs: u64, owned_by_mover: bool) -> bool {
        let owner = if owned_by_mover {
            board.player()
        } else {
            board.opponent()
        };
        if discs & !owner.bits() != 0 {
            return true;
        }

        let moves = board.legal_moves();
        if moves.is_empty() {
            let passed = board.switch_players();
            return passed.has_legal_moves()
                && reference_can_be_flipped(passed, discs, !owned_by_mover);
        }
        moves
            .iter()
            .any(|sq| reference_can_be_flipped(board.make_move(sq), discs, !owned_by_mover))
    }

    /// Returns a board with `n_empties` empty squares and random discs.
    fn random_filled_board(rng: &mut StdRng, n_empties: u32) -> Board {
        let mut empties = 0u64;
        while empties.count_ones() < n_empties {
            empties |= 1 << rng.random_range(0..64);
        }
        let player = rng.random::<u64>() & !empties;
        Board::from_bitboards(player, !player & !empties)
    }

    /// Returns a board reached by random legal moves from the initial
    /// position, or `None` if the game ends before `n_empties` are left.
    fn random_played_board(rng: &mut StdRng, n_empties: u32) -> Option<Board> {
        let mut board = Board::new();
        while board.empty_count() > n_empties {
            if !board.has_legal_moves() {
                board = board.switch_players();
                if !board.has_legal_moves() {
                    return None;
                }
            }
            let moves = board.legal_moves();
            let sq = moves
                .iter()
                .nth(rng.random_range(0..moves.count() as usize))?;
            board = board.make_move(sq);
        }
        Some(board)
    }

    fn count_checked_stable_discs(stable_discs: impl Fn(Bitboard, Bitboard) -> u64) -> u32 {
        let mut rng = StdRng::seed_from_u64(0x5741_424c_4553);
        let mut n_stable = 0;

        for case in 0..600 {
            let n_empties = 1 + case % 7;
            let board = if case % 2 == 0 {
                random_filled_board(&mut rng, n_empties)
            } else {
                match random_played_board(&mut rng, n_empties) {
                    Some(board) => board,
                    None => continue,
                }
            };

            for (board, name) in [
                (board, "side to move"),
                (board.switch_players(), "other side"),
            ] {
                let stable = stable_discs(board.player(), board.opponent());
                n_stable += stable.count_ones();

                assert_eq!(stable & !board.player().bits(), 0, "{name} of {board:?}");
                // Stability does not depend on whose turn it is.
                assert!(
                    !reference_can_be_flipped(board, stable, true)
                        && !reference_can_be_flipped(board.switch_players(), stable, false),
                    "{stable:#018x} is not stable for the {name} of {board:?}"
                );
            }
        }

        n_stable
    }

    #[test]
    fn stable_discs_are_never_flipped_in_any_line_of_play() {
        let n_stable = count_checked_stable_discs(|player, opponent| {
            get_stable_discs(player, opponent).bits()
        });
        assert!(n_stable > 5000, "only {n_stable} stable discs were checked");
    }

    fn single_empty_line_stable(player: Bitboard, opponent: Bitboard) -> u64 {
        single_empty_line_stable_impl(
            player.bits(),
            opponent.bits(),
            get_stable_discs(player, opponent).bits(),
        )
    }

    #[test]
    fn single_empty_line_discs_are_never_flipped_in_any_line_of_play() {
        let n_contact = count_checked_stable_discs(|player, opponent| {
            get_stable_discs(player, opponent).bits()
        });
        let n_stable = count_checked_stable_discs(single_empty_line_stable);
        assert!(
            n_stable > n_contact + 1000,
            "the pass added only {} stable discs",
            n_stable - n_contact
        );
    }

    fn reference_single_empty_line_stable(player: u64, opponent: u64, stable: u64) -> u64 {
        const STEPS: [(i32, i32); 4] = [(1, 0), (0, 1), (-1, 1), (1, 1)];
        let empties = !(player | opponent);
        let opp_edge = get_stable_edge(opponent, player);
        let has = |bits: u64, sq: i32| bits >> sq & 1 != 0;
        let ray = |sq: i32, dir: usize, sign: i32| {
            let (dx, dy) = STEPS[dir];
            let (mut x, mut y) = (sq % 8 + sign * dx, sq / 8 + sign * dy);
            let mut squares = Vec::new();
            while (0..8).contains(&x) && (0..8).contains(&y) {
                squares.push(y * 8 + x);
                x += sign * dx;
                y += sign * dy;
            }
            squares
        };
        let n_empties_on_line = |sq: i32, dir: usize| {
            ray(sq, dir, -1)
                .into_iter()
                .chain(ray(sq, dir, 1))
                .chain([sq])
                .filter(|&s| has(empties, s))
                .count()
        };
        let touches = |bits: u64, sq: i32, dir: usize| {
            [-1, 1]
                .into_iter()
                .any(|sign| ray(sq, dir, sign).first().is_some_and(|&s| has(bits, s)))
        };
        let player_safe_on_dir =
            |sq: i32, dir: usize| n_empties_on_line(sq, dir) == 0 || touches(stable, sq, dir);
        let player_keeps_color =
            |sq: i32, dir: usize| (0..4).all(|other| other == dir || player_safe_on_dir(sq, other));
        let opp_safe_on_dir = |sq: i32, dir: usize| {
            if n_empties_on_line(sq, dir) == 0 || touches(opp_edge, sq, dir) {
                return true;
            }
            if n_empties_on_line(sq, dir) == 1 {
                let sign = if ray(sq, dir, 1).iter().any(|&s| has(empties, s)) {
                    1
                } else {
                    -1
                };
                return ray(sq, dir, sign)
                    .iter()
                    .take_while(|&&s| !has(empties, s))
                    .any(|&s| has(player, s) && player_keeps_color(s, dir));
            }
            false
        };
        let opp_keeps_color =
            |sq: i32, dir: usize| (0..4).all(|other| other == dir || opp_safe_on_dir(sq, other));

        let mut safe = [0u64; 4];
        for (dir, safe) in safe.iter_mut().enumerate() {
            for sq in (0..64).filter(|&sq| has(player, sq)) {
                let is_safe = match n_empties_on_line(sq, dir) {
                    0 => true,
                    1 => {
                        let sign = if ray(sq, dir, 1).iter().any(|&s| has(empties, s)) {
                            1
                        } else {
                            -1
                        };
                        ray(sq, dir, sign)
                            .iter()
                            .take_while(|&&s| !has(empties, s))
                            .any(|&s| has(opponent, s) && opp_keeps_color(s, dir))
                    }
                    _ => false,
                };
                if is_safe {
                    *safe |= 1 << sq;
                }
            }
        }

        let mut stable = stable;
        loop {
            let mut next = stable;
            for sq in (0..64).filter(|&sq| has(player, sq)) {
                let on_edge = sq % 8 == 0 || sq % 8 == 7 || sq / 8 == 0 || sq / 8 == 7;
                let safe_everywhere =
                    (0..4).all(|dir| has(safe[dir], sq) || touches(stable, sq, dir));
                if !on_edge && safe_everywhere {
                    next |= 1 << sq;
                }
            }
            if next == stable {
                return stable;
            }
            stable = next;
        }
    }

    #[test]
    fn single_empty_line_pass_matches_reference() {
        let mut rng = StdRng::seed_from_u64(0x0051_4e47_4c45);
        let mut n_added = 0;

        for case in 0..3000 {
            let n_empties = 1 + case % 12;
            let board = if case % 2 == 0 {
                random_filled_board(&mut rng, n_empties)
            } else {
                match random_played_board(&mut rng, n_empties) {
                    Some(board) => board,
                    None => continue,
                }
            };

            for board in [board, board.switch_players()] {
                let (player, opponent) = (board.player().bits(), board.opponent().bits());
                let stable = get_stable_discs(board.player(), board.opponent()).bits();
                let expected = reference_single_empty_line_stable(player, opponent, stable);
                let extended = single_empty_line_stable_impl(player, opponent, stable);
                assert_eq!(extended, expected, "{board:?}");
                assert_eq!(
                    single_empty_line_stable_scalar(player, opponent, stable),
                    expected,
                    "scalar mismatch on {board:?}"
                );
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                assert_eq!(
                    unsafe { single_empty_line_stable_avx2(player, opponent, stable) },
                    expected,
                    "avx2 mismatch on {board:?}"
                );
                n_added += (extended & !stable).count_ones();
            }
        }

        assert!(n_added > 3000, "the pass added only {n_added} stable discs");
    }

    #[test]
    fn single_empty_line_pass_finds_a_disc_shielded_by_a_color_keeping_disc() {
        let board = Board::from_string(
            "--XXXX-X\
             OOOOXX-X\
             XOXXXXOX\
             XOXXOXXX\
             XXXOXXXX\
             XOOOXXOX\
             XOOOOOXX\
             XXXXXXXX",
            crate::player_color::PlayerColor::Black,
        )
        .unwrap();
        let b5 = crate::square::Square::B5.bitboard().bits();

        assert_eq!(
            get_stable_discs(board.player(), board.opponent()).bits() & b5,
            0
        );
        assert_ne!(
            single_empty_line_stable(board.player(), board.opponent()) & b5,
            0
        );

        let white = board.switch_players();
        assert_eq!(
            stability_cutoff(&white, white.empty_count(), DiscScore::new(6)),
            Some(DiscScore::new(6))
        );
    }

    #[test]
    fn stable_discs_include_full_edge_but_not_unprotected_inner_disc() {
        let player = Bitboard::new(0xff) | crate::square::Square::B2.bitboard();
        let stable = get_stable_discs(player, Bitboard::new(0));

        assert_eq!(stable.bits(), 0xff);
    }
}
