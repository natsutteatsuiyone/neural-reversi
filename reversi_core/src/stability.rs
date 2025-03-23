use std::sync::OnceLock;

use crate::{board::Board, constants::SCORE_MAX, types::Score};

const EDGE_STABILITY_SIZE: usize = 256 * 256;
static EDGE_STABILITY: OnceLock<[u8; EDGE_STABILITY_SIZE]> = OnceLock::new();

fn find_edge_stable(old_p: i32, old_o: i32, mut stable: i32) -> i32 {
    let e: i32 = !(old_p | old_o);

    stable &= old_p;
    if stable == 0 || e == 0 {
        return stable;
    }

    for x in 0..8 {
        if (e & x_to_bit(x)) == 0 {
            continue;
        }

        let mut o = old_o;
        let mut p = old_p | x_to_bit(x);
        if x > 1 {
            // flip left discs
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
            // flip right discs
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
        stable = find_edge_stable(p, o, stable); // next move
        if stable == 0 {
            return stable;
        }

        let mut p = old_p;
        let mut o = old_o | x_to_bit(x); // opponent plays on it
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

fn x_to_bit(x: i32) -> i32 {
    1 << x
}

fn init_edge_stability() -> [u8; EDGE_STABILITY_SIZE] {
    let mut table: [u8; EDGE_STABILITY_SIZE] = [0; EDGE_STABILITY_SIZE];
    for p in 0..256 {
        for o in 0..256 {
            if p & o != 0 {
                //illegal positions
                table[p * 256 + o] = 0;
            } else {
                table[p * 256 + o] = find_edge_stable(p as i32, o as i32, p as i32) as u8;
            }
        }
    }
    table
}

pub fn init() {
    let _ = EDGE_STABILITY
        .set(init_edge_stability());
}

#[inline]
fn unpack_a2a7(x: u8) -> u64 {
    let a = (x & 0x7e) as u64;
    (a.wrapping_mul(0x0000_0408_1020_4080u64)) & 0x0001_0101_0101_0100
}

#[inline]
fn unpack_h2h7(x: u8) -> u64 {
    let a = (x & 0x7e) as u64;
    (a.wrapping_mul(0x0002_0408_1020_4000u64)) & 0x0080_8080_8080_8000
}

#[inline]
fn pack_a1a8(x: u64) -> usize {
    let a = x & 0x0101_0101_0101_0101;
    ((a.wrapping_mul(0x0102_0408_1020_4080u64)) >> 56) as usize
}

#[inline]
fn pack_h1h8(x: u64) -> usize {
    let a = x & 0x8080_8080_8080_8080;
    ((a.wrapping_mul(0x0002_0408_1020_4081u64)) >> 56) as usize
}

#[inline]
fn get_stable_edge(p: u64, o: u64) -> u64 {
    let table = EDGE_STABILITY.get().unwrap();

    if is_x86_feature_detected!("avx2") {
        unsafe {
            use std::arch::x86_64::*;
            let p0 = _mm_cvtsi64_si128(p as i64);
            let o0 = _mm_cvtsi64_si128(o as i64);
            let po = _mm_unpacklo_epi8(o0, p0);
            let mut stable_edge  =
                *table.get_unchecked(_mm_extract_epi16(po, 0) as usize) as u64
                | ((*table.get_unchecked(_mm_extract_epi16(po, 7) as usize) as u64) << 56);

            let po = _mm_unpacklo_epi64(o0, p0);
            let a1a8 = table.get_unchecked(_mm_movemask_epi8(_mm_slli_epi64(po, 7)) as usize);
            let h1h8 = table.get_unchecked(_mm_movemask_epi8(po) as usize);
            stable_edge |= unpack_a2a7(*a1a8) | unpack_h2h7(*h1h8);

            return stable_edge;
        }
    }

    table[((p & 0xff) * 256 + (o & 0xff)) as usize] as u64
        | (table[((p >> 56) * 256 + (o >> 56)) as usize] as u64) << 56
        | unpack_a2a7(table[pack_a1a8(p) * 256 + pack_a1a8(o)])
        | unpack_h2h7(table[pack_h1h8(p) * 256 + pack_h1h8(o)])
}

fn get_full_lines(disc: u64, full: &mut [u64; 4]) -> u64 {
    let mut h = disc;
    let mut v = disc;
    let mut l7 = disc;
    let mut l9 = disc;
    let mut r7 = disc;
    let mut r9 = disc;

    h &= h >> 1;
    h &= h >> 2;
    h &= h >> 4;
    full[0] = (h & 0x0101010101010101) * 0xff;

    v &= v.rotate_right(8); // ror 8
    v &= v.rotate_right(16); // ror 16
    v &= v.rotate_left(32); // ror 32
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

fn get_stable_by_contact(central_mask: u64, previous_stable: u64, full: &[u64; 4]) -> u64 {
    let mut stable_h: u64;
    let mut stable_v: u64;
    let mut stable_d7: u64;
    let mut stable_d9: u64;
    let mut old_stable = 0;
    let mut stable = previous_stable;

    while stable != old_stable {
        old_stable = stable;
        stable_h = (stable >> 1) | (stable << 1) | full[0];
        stable_v = (stable >> 8) | (stable << 8) | full[1];
        stable_d7 = (stable >> 7) | (stable << 7) | full[2];
        stable_d9 = (stable >> 9) | (stable << 9) | full[3];
        stable |= stable_h & stable_v & stable_d7 & stable_d9 & central_mask;
    }
    stable
}

pub fn get_stable_discs(p: u64, o: u64) -> u64 {
    let central_mask = p & 0x007e7e7e7e7e7e00;
    let mut full: [u64; 4] = [0; 4];

    let mut stable = get_stable_edge(p, o);
    stable |= get_full_lines(p | o, &mut full) & central_mask;

    get_stable_by_contact(central_mask, stable, &full)
}

#[rustfmt::skip]
const NWS_STABILITY_THRESHOLD: [i8; 64] = [
    99, 99, 99, 99,  6,  8, 10, 12,
    14, 16, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 38, 40, 42, 44, 46,
    48, 48, 50, 50, 52, 52, 54, 54,
    56, 56, 58, 58, 60, 60, 62, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
    99, 99, 99, 99, 99, 99, 99, 99, // no stable square at those depths
    99, 99, 99, 99, 99, 99, 99, 99
];

pub fn stability_cutoff(board: &Board, n_empties: u32, alpha: Score) -> Option<Score> {
    if alpha >= NWS_STABILITY_THRESHOLD[n_empties as usize] as Score {
        let score = SCORE_MAX - 2 * board.switch_players().get_stability();
        if score <= alpha {
            return Some(score);
        }
    }
    None
}
