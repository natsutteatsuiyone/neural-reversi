use std::io::{self, Write};

use flip_scalar_difftest::opt_or_dash;

include!(concat!(env!("OUT_DIR"), "/stability_source.rs"));

fn write_edge_cases(out: &mut impl Write) -> io::Result<()> {
    for p in 0..=u8::MAX {
        for o in 0..=u8::MAX {
            let index = edge_stability_index(p, o);
            let stable = if p & !o == 0 {
                edge_stability(p as usize, o as usize)
            } else {
                EDGE_STABILITY.0[((o as usize) << 8) | p as usize]
            };
            writeln!(out, "table {p} {o} {index} {stable}")?;
            if p & o != 0 {
                continue;
            }
            for sq in 0..8 {
                if (p | o) & (1 << sq) == 0 {
                    let (next_p, next_o) = apply_edge_move(sq, p, o);
                    writeln!(out, "move {p} {o} {sq} {} {next_p} {next_o}", x_to_bit(sq))?;
                }
            }
        }
    }
    for x in 0..=u8::MAX {
        writeln!(out, "unpack {x} {} {}", unpack_a2a7(x), unpack_h2h7(x))?;
    }
    Ok(())
}

fn write_cutoff_case(out: &mut impl Write, board: &Board, n: u32, alpha: i32) -> io::Result<()> {
    let alpha = DiscScore::new(alpha);
    let p = board.player().bits();
    let o = board.opponent().bits();
    let margin = opt_or_dash(
        (alpha.value() >= 6)
            .then(|| u8::from(has_enough_opponent_discs_for_cutoff(o.count_ones(), alpha))),
    );
    let cutoff = opt_or_dash(stability_cutoff(board, n, alpha).map(DiscScore::value));
    writeln!(out, "cut {p} {o} {n} {} {margin} {cutoff}", alpha.value())
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    write_edge_cases(out)?;
    let full_opponent = Board::from_bitboards(0, u64::MAX);
    for n in (0..66).chain([u32::MAX]) {
        for alpha in 5..=100 {
            write_cutoff_case(out, &full_opponent, n, alpha)?;
        }
    }

    let words = [0, u64::MAX, 0x8181_8181_8181_8181, 0xaa55_aa55_aa55_aa55]
        .into_iter()
        .chain((0..64).flat_map(|bit| [1u64 << bit, !(1u64 << bit)]))
        .chain((0..=64).map(|n| u64::MAX.checked_shr(n).unwrap_or(0)))
        .chain((1u64..=192).map(|n| n.wrapping_mul(0x9e37_79b9_7f4a_7c15)));
    for p in words {
        writeln!(out, "pack {p} {} {}", pack_a1a8(p), pack_h1h8(p))?;
        for o in [0, !p, p.rotate_left(17) & !p] {
            let board = Board::from_bitboards(p, o);
            let occ = p | o;
            let edge = get_stable_edge(p, occ);
            let full = get_full_lines(occ);
            let intersection = full.all_directions();
            let (h, v, d7, d9) = (full.horizontal, full.vertical, full.diag_7, full.diag_9);
            let stable = get_stable_discs(board.player(), board.opponent()).bits();
            writeln!(
                out,
                "board {p} {o} {edge} {h} {v} {d7} {d9} {intersection} {stable}"
            )?;
            for seed in [0, edge, edge | intersection] {
                let expanded = get_stable_by_contact(p, seed, full);
                writeln!(out, "contact {p} {seed} {h} {v} {d7} {d9} {expanded}")?;
            }
            for alpha in [i32::MIN, -4, 6, 32, 48, 64, 98, 99, i32::MAX] {
                write_cutoff_case(out, &board, board.empty_count(), alpha)?;
            }
        }
    }
    Ok(())
}
