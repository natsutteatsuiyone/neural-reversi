use std::io::{self, Write};

use crate::Board;
use crate::engine::bitboard::Bitboard;
use crate::engine::empties::{Empties, odd_quadrants, quadrant};
use crate::square::Square;

fn listed(mut empties: Empties) -> Vec<Square> {
    (0..empties.count())
        .map(|_| {
            let [sq] = empties.first::<1>();
            empties = empties.without(sq);
            sq
        })
        .collect()
}

fn observe(out: &mut impl Write, empties: Empties) -> io::Result<()> {
    write!(out, "{}", empties.count())?;
    for sq in listed(empties) {
        write!(out, " {}", sq.index())?;
    }
    Ok(())
}

fn write_case(out: &mut impl Write, empty: u64, remove_index: usize) -> io::Result<()> {
    let occupied = !empty;
    let colors = 0x55aa_55aa_55aa_55aa;
    let board = Board::from_bitboards(occupied & colors, occupied & !colors);
    let empties = Empties::new(&board);
    let odd = odd_quadrants(board.empty_squares()).bits();
    let removed = listed(empties).get(remove_index).copied();
    write!(out, "{empty} {}|", removed.map_or(64, Square::index))?;
    observe(out, empties)?;
    write!(out, "|{odd}")?;
    if let Some(sq) = removed {
        write!(out, "|")?;
        observe(out, empties.without(sq))?;
        let child = Bitboard::new(empty & !(1 << sq.index()));
        write!(
            out,
            "|{}|{}",
            odd ^ quadrant(sq).bits(),
            odd_quadrants(child).bits()
        )?;
    }
    writeln!(out)
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    write_case(out, 0, 0)?;
    for count in 1..=64 {
        let run = u64::MAX >> (64 - count);
        for (rotation, index) in [(0, 0), (13, count / 2), (47, count - 1)] {
            write_case(out, run.rotate_left(rotation), index)?;
        }
    }
    for empty in [
        0x8100_0000_0000_0081,
        0x0000_0018_1800_0000,
        0x0f0f_0f0f_0000_0000,
        0x1248_1248_8421_8421,
    ] {
        for index in 0..4 {
            write_case(out, empty, index)?;
        }
    }
    Ok(())
}
