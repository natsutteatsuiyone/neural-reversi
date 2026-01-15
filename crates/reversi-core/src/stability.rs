//! Stability detection module
//!
//! Reference: https://github.com/abulmo/edax-reversi/blob/14f048c05ddfa385b6bf954a9c2905bbe677e9d3/src/board.c
use std::sync::OnceLock;

use crate::{bitboard::Bitboard, board::Board, constants::SCORE_MAX, types::Score};

/// Size of the edge stability lookup table (256 * 256 for all possible edge configurations)
const EDGE_STABILITY_SIZE: usize = 256 * 256;

/// Global edge stability lookup table, initialized once at startup.
static EDGE_STABILITY: OnceLock<[u8; EDGE_STABILITY_SIZE]> = OnceLock::new();

/// Recursively finds stable discs along an edge by simulating all possible move sequences.
///
/// # Arguments
///
/// * `old_p` - Player's disc configuration on the edge (8 bits, one per square)
/// * `old_o` - Opponent's disc configuration on the edge (8 bits, one per square)
/// * `stable` - Current stable disc configuration to refine
///
/// # Returns
///
/// A bitmask of stable discs for the player on this edge.
fn find_edge_stable(old_p: i32, old_o: i32, mut stable: i32) -> i32 {
    // Calculate empty squares on the edge
    let e: i32 = !(old_p | old_o);

    // Only player's discs can be stable
    stable &= old_p;
    if stable == 0 || e == 0 {
        return stable;
    }

    // Try placing a disc on each empty square and check stability
    for x in 0..8 {
        if (e & x_to_bit(x)) == 0 {
            continue;
        }

        // Simulate player's move at position x
        let mut o = old_o;
        let mut p = old_p | x_to_bit(x);
        if x > 1 {
            // Check for flips to the left
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
            // Check for flips to the right
            let mut y = x + 1;
            while y < 8 && (o & x_to_bit(y)) != 0 {
                y += 1;
            }
            if (p & x_to_bit(y)) != 0 {
                let mut y = x + 1;
                while y < 8 && (o & x_to_bit(y)) != 0 {
                    o ^= x_to_bit(y);
                    p ^= x_to_bit(y);
                    y += 1;
                }
            }
        }
        // Recursively check stability after player's move
        stable = find_edge_stable(p, o, stable);
        if stable == 0 {
            return stable;
        }

        // Now simulate opponent's move at position x
        let mut p = old_p;
        let mut o = old_o | x_to_bit(x);
        if x > 1 {
            let mut y = x - 1;
            while y > 0 && (p & x_to_bit(y)) != 0 {
                y -= 1;
            }
            if (o & x_to_bit(y)) != 0 {
                let mut y = x - 1;
                while y > 0 && (p & x_to_bit(y)) != 0 {
                    o ^= x_to_bit(y);
                    p ^= x_to_bit(y);
                    y -= 1;
                }
            }
        }
        if x < 6 {
            let mut y = x + 1;
            while y < 8 && (p & x_to_bit(y)) != 0 {
                y += 1;
            }
            if (o & x_to_bit(y)) != 0 {
                let mut y = x + 1;
                while y < 8 && (p & x_to_bit(y)) != 0 {
                    o ^= x_to_bit(y);
                    p ^= x_to_bit(y);
                    y += 1;
                }
            }
        }
        stable = find_edge_stable(p, o, stable);
        if stable == 0 {
            return stable;
        }
    }

    stable
}

/// Converts a position index (0-7) to its corresponding bit mask.
///
/// # Arguments
///
/// * `x` - Position index from 0 to 7
///
/// # Returns
///
/// Bit mask with only the bit at position x set.
fn x_to_bit(x: i32) -> i32 {
    1 << x
}

/// Initializes the edge stability lookup table.
///
/// # Returns
///
/// A 65536-entry lookup table indexed by [player_edge * 256 + opponent_edge].
fn init_edge_stability() -> [u8; EDGE_STABILITY_SIZE] {
    let mut table: [u8; EDGE_STABILITY_SIZE] = [0; EDGE_STABILITY_SIZE];
    for p in 0..256 {
        for o in 0..256 {
            if p & o != 0 {
                // Illegal positions (same square occupied by both players)
                table[p * 256 + o] = 0;
            } else {
                // Compute stable discs for this edge configuration
                table[p * 256 + o] = find_edge_stable(p as i32, o as i32, p as i32) as u8;
            }
        }
    }
    table
}

/// Initializes the stability module by computing the edge stability table.
pub fn init() {
    let _ = EDGE_STABILITY.set(init_edge_stability());
}

/// Unpacks bits 1-6 of a byte to the A2-A7 squares on the board.
///
/// # Arguments
///
/// * `x` - Edge byte containing stability information
///
/// # Returns
///
/// 64-bit board with bits set at A2-A7 positions.
#[inline]
fn unpack_a2a7(x: u8) -> u64 {
    let a = (x & 0x7e) as u64; // Extract bits 1-6
    (a.wrapping_mul(0x0000_0408_1020_4080u64)) & 0x0001_0101_0101_0100
}

/// Unpacks bits 1-6 of a byte to the H2-H7 squares on the board.
///
/// # Arguments
///
/// * `x` - Edge byte containing stability information
///
/// # Returns
///
/// 64-bit board with bits set at H2-H7 positions.
#[inline]
fn unpack_h2h7(x: u8) -> u64 {
    let a = (x & 0x7e) as u64; // Extract bits 1-6
    (a.wrapping_mul(0x0002_0408_1020_4000u64)) & 0x0080_8080_8080_8000
}

/// Packs the A-file (A1-A8) of a bitboard into a single byte.
///
/// # Arguments
///
/// * `x` - 64-bit bitboard
///
/// # Returns
///
/// 8-bit value with each bit representing a square in the A-file.
#[inline]
fn pack_a1a8(x: u64) -> usize {
    let a = x & 0x0101_0101_0101_0101; // Mask A-file
    ((a.wrapping_mul(0x0102_0408_1020_4080u64)) >> 56) as usize
}

/// Packs the H-file (H1-H8) of a bitboard into a single byte.
///
/// # Arguments
///
/// * `x` - 64-bit bitboard
///
/// # Returns
///
/// 8-bit value with each bit representing a square in the H-file.
#[inline]
fn pack_h1h8(x: u64) -> usize {
    let a = x & 0x8080_8080_8080_8080; // Mask H-file
    ((a.wrapping_mul(0x0002_0408_1020_4081u64)) >> 56) as usize
}

/// Gets stable discs along all four edges of the board.
///
/// This function uses the pre-computed edge stability table to quickly
/// determine which edge discs are stable. It handles all four edges:
/// - Bottom edge (rank 1)
/// - Top edge (rank 8)
/// - Left edge (A-file)
/// - Right edge (H-file)
///
/// # Arguments
///
/// * `p` - Player's bitboard
/// * `o` - Opponent's bitboard
///
/// # Returns
///
/// Bitboard with stable edge discs marked.
#[inline]
fn get_stable_edge(p: u64, o: u64) -> u64 {
    let table = EDGE_STABILITY.get().unwrap();
    table[((p & 0xff) * 256 + (o & 0xff)) as usize] as u64
        | (table[((p >> 56) * 256 + (o >> 56)) as usize] as u64) << 56
        | unpack_a2a7(table[pack_a1a8(p) * 256 + pack_a1a8(o)])
        | unpack_h2h7(table[pack_h1h8(p) * 256 + pack_h1h8(o)])
}

/// Detects completely filled lines in all four directions.
///
/// # Arguments
///
/// * `disc` - Combined bitboard of all occupied squares (player | opponent)
/// * `full` - Output array to store full lines in each direction:
///   - `full[0]`: Horizontal full lines
///   - `full[1]`: Vertical full lines
///   - `full[2]`: Diagonal (/) full lines
///   - `full[3]`: Diagonal (\) full lines
///
/// # Returns
///
/// Bitboard with all squares that are on full lines in all four directions.
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

/// Gets stable discs by contact with other stable discs.
///
/// # Arguments
///
/// * `central_mask` - Mask of potentially stable discs (excludes edges)
/// * `previous_stable` - Initially stable discs (from edges and full lines)
/// * `full` - Full lines in each direction (from `get_full_lines`)
///
/// # Returns
///
/// Bitboard with stable discs marked.
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

/// Estimates the stable discs.
///
/// # Arguments
///
/// * `p` - Player's bitboard
/// * `o` - Opponent's bitboard
///
/// # Returns
///
/// Bitboard with stable discs for the player.
pub fn get_stable_discs(player: Bitboard, opponent: Bitboard) -> Bitboard {
    let central_mask = player.0 & 0x007e7e7e7e7e7e00;
    let mut full: [u64; 4] = [0; 4];

    let mut stable = get_stable_edge(player.0, opponent.0);
    stable |= get_full_lines(player.0 | opponent.0, &mut full) & central_mask;
    Bitboard(get_stable_by_contact(central_mask, stable, &full))
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

/// Attempts to prove an alpha cutoff using stability analysis.
///
/// This function implements a stability-based pruning technique. If we can prove
/// that the opponent has enough stable discs to guarantee a score <= alpha,
/// we can immediately return that score without further search.
///
/// # Arguments
///
/// * `board` - Current board position
/// * `n_empties` - Number of empty squares remaining
/// * `alpha` - Alpha value from alpha-beta search
///
/// # Returns
///
/// Some(score) if a cutoff is possible, None otherwise.
pub fn stability_cutoff(board: &Board, n_empties: u32, alpha: Score) -> Option<Score> {
    if alpha >= NWS_STABILITY_THRESHOLD[n_empties as usize] as Score {
        let score = SCORE_MAX - 2 * board.switch_players().get_stability() as Score;
        if score <= alpha {
            return Some(score);
        }
    }
    None
}
