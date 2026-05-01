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
    bitboard::Bitboard, board::Board, constants::SCORE_MAX, types::Score, util::align::Align64,
};

/// Size of the edge stability lookup table (256 * 256 for all possible edge configurations).
const EDGE_STABILITY_SIZE: usize = 256 * 256;
const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = 0x8080_8080_8080_8080;
const FILE_A_INNER: u64 = 0x0001_0101_0101_0100;
const FILE_H_INNER: u64 = 0x0080_8080_8080_8000;

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
/// disc higher. This is the iterative equivalent of the Edax recursive search.
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

    let mut x = 0;
    while x < 8 {
        if (e & x_to_bit(x)) != 0 {
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
        }
        x += 1;
    }

    stable
}

/// Converts a position index (0-7) to its corresponding bit mask.
const fn x_to_bit(x: u8) -> u8 {
    1u8 << x
}

const fn popcount8(mut x: u8) -> usize {
    let mut n = 0;
    while x != 0 {
        x &= x - 1;
        n += 1;
    }
    n
}

const fn edge_stability_index(player_edge: u8, opponent_edge: u8) -> usize {
    ((player_edge as usize) << 8) | opponent_edge as usize
}

/// Computes the edge stability lookup table.
///
/// Returns a 65536-entry table indexed by `player_edge * 256 + opponent_edge`.
const fn init_edge_stability() -> [u8; EDGE_STABILITY_SIZE] {
    let mut table: [u8; EDGE_STABILITY_SIZE] = [0; EDGE_STABILITY_SIZE];

    let mut occupied = 8usize;
    loop {
        let mut p = 0usize;
        while p < 256 {
            let mut o = 0usize;
            while o < 256 {
                if (p & o) == 0 && popcount8((p | o) as u8) == occupied {
                    table[(p << 8) | o] = find_edge_stable_from_table(p as u8, o as u8, &table);
                }
                o += 1;
            }
            p += 1;
        }

        if occupied == 0 {
            break;
        }
        occupied -= 1;
    }

    table
}

#[doc(hidden)]
pub fn build_edge_stability_table_for_bench() -> [u8; EDGE_STABILITY_SIZE] {
    init_edge_stability()
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

/// Detects completely filled lines in all four directions.
///
/// Populates `full` with bitmasks for each direction: horizontal (`full[0]`),
/// vertical (`full[1]`), diagonal `/` (`full[2]`), and diagonal `\` (`full[3]`).
/// Returns the intersection (squares on full lines in every direction).
fn get_full_lines(disc: u64, full: &mut [u64; 4]) -> u64 {
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
    full[0] = (h & 0x0101010101010101) * 0xff; // Expand to full rows

    // Check vertical lines: fold all 8 rows together
    v &= v.rotate_right(8); // Check 2 consecutive discs
    v &= v.rotate_right(16); // Check 4 consecutive discs
    v &= v.rotate_left(32); // Check 8 consecutive discs
    full[1] = v;

    l7 &= 0xff01010101010101 | (l7 >> 7);
    r7 &= 0x80808080808080ff | (r7 << 7);
    l7 &= 0xffff030303030303 | (l7 >> 14);
    r7 &= 0xc0c0c0c0c0c0ffff | (r7 << 14);
    l7 &= 0xffffffff0f0f0f0f | (l7 >> 28);
    r7 &= 0xf0f0f0f0ffffffff | (r7 << 28);
    full[2] = l7 & r7;

    l9 &= 0xff80808080808080 | (l9 >> 9);
    r9 &= 0x01010101010101ff | (r9 << 9);
    l9 &= 0xffffc0c0c0c0c0c0 | (l9 >> 18);
    r9 &= 0x030303030303ffff | (r9 << 18);
    full[3] = l9 & r9 & (0x0f0f0f0ff0f0f0f0 | (l9 >> 36) | (r9 << 36));

    full[0] & full[1] & full[2] & full[3]
}

/// Expands the stable disc set by contact propagation.
///
/// Starting from `previous_stable` (edge-stable and full-line-stable discs),
/// iteratively marks central discs in `central_mask` as stable when, in each
/// of the four directions, they are adjacent to a stable disc or lie on a
/// full line.
fn get_stable_by_contact(central_mask: u64, previous_stable: u64, full: &[u64; 4]) -> u64 {
    let mut stable_h: u64; // Stable in horizontal direction
    let mut stable_v: u64; // Stable in vertical direction
    let mut stable_d7: u64; // Stable in diagonal (/) direction
    let mut stable_d9: u64; // Stable in diagonal (\) direction
    let mut old_stable = 0;
    let mut stable = previous_stable;

    // Iteratively expand stable region until no new stable discs are found
    while stable != old_stable {
        old_stable = stable;

        // A disc is stable in a direction if it touches a stable disc or is on a full line
        stable_h = (stable >> 1) | (stable << 1) | full[0];
        stable_v = (stable >> 8) | (stable << 8) | full[1];
        stable_d7 = (stable >> 7) | (stable << 7) | full[2];
        stable_d9 = (stable >> 9) | (stable << 9) | full[3];

        // A disc is stable if it's stable in all four directions
        stable |= stable_h & stable_v & stable_d7 & stable_d9 & central_mask;
    }
    stable
}

/// Returns a lower-bound estimate of the player's stable discs.
///
/// Stable discs are pieces that can never be flipped for the rest of the game.
/// The estimate combines three techniques:
/// 1. Edge stability from a pre-computed lookup table
/// 2. Full-line detection (discs on completely filled lines)
/// 3. Contact propagation (discs adjacent to already-stable discs)
///
#[inline]
pub fn stable_discs_lower_bound(player: Bitboard, opponent: Bitboard) -> Bitboard {
    let central_mask = player.bits() & 0x007e7e7e7e7e7e00;
    let mut full: [u64; 4] = [0; 4];

    let mut stable = get_stable_edge(player.bits(), opponent.bits());
    if stable == 0 {
        return Bitboard::new(0);
    }

    stable |= get_full_lines(player.bits() | opponent.bits(), &mut full) & central_mask;
    Bitboard::new(get_stable_by_contact(central_mask, stable, &full))
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

/// Returns whether the opponent has enough discs that stability can possibly
/// prove `score <= alpha`.
#[inline(always)]
fn has_enough_opponent_discs_for_cutoff(opponent_count: u32, alpha: Score) -> bool {
    SCORE_MAX - 2 * opponent_count as Score <= alpha
}

/// Attempts to prove an alpha cutoff using stability analysis.
///
/// If the opponent has enough stable discs to guarantee a final score at or
/// below `alpha`, returns that score immediately.
///
/// Returns [`None`] if the cutoff cannot be proven.
///
#[inline(always)]
pub fn stability_cutoff(board: &Board, n_empties: u32, alpha: Score) -> Option<Score> {
    if alpha >= NWS_STABILITY_THRESHOLD[n_empties as usize] as Score {
        if n_empties >= 8 && !has_enough_opponent_discs_for_cutoff(board.opponent.count(), alpha) {
            return None;
        }

        let score =
            SCORE_MAX - 2 * stable_discs_lower_bound(board.opponent, board.player).count() as Score;
        if score <= alpha {
            return Some(score);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

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
    fn stability_cutoff_candidate_uses_opponent_disc_upper_bound() {
        assert!(!has_enough_opponent_discs_for_cutoff(31, 1));
        assert!(has_enough_opponent_discs_for_cutoff(31, 2));
        assert!(!has_enough_opponent_discs_for_cutoff(8, 47));
        assert!(has_enough_opponent_discs_for_cutoff(8, 48));
    }

    #[test]
    fn stable_discs_are_available_without_runtime_init() {
        let player = Bitboard::new(0xff);
        let stable = stable_discs_lower_bound(player, Bitboard::new(0));

        assert_eq!(stable.bits() & 0xff, 0xff);
    }
}
