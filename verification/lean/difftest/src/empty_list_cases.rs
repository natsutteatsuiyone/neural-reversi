use std::io::{self, Write};

use crate::Board;
use crate::engine::empty_list::EmptyList;
use crate::square::Square;

fn observe(out: &mut impl Write, list: &EmptyList) -> io::Result<()> {
    let count = list.count();
    write!(out, "{count} {}", list.parity())?;
    if count > 0 {
        let (mut square, quadrant) = list.first_and_quad_id();
        write!(out, " {}:{quadrant}", square.index())?;
        for _ in 1..count {
            let (next, quadrant) = list.next_and_quad_id(square);
            write!(out, " {}:{quadrant}", next.index())?;
            square = next;
        }
    }
    Ok(())
}

fn write_case(out: &mut impl Write, empty: u64, remove_index: usize) -> io::Result<()> {
    let occupied = !empty;
    let colors = 0x55aa_55aa_55aa_55aa;
    let board = Board::from_bitboards(occupied & colors, occupied & !colors);
    let mut list = EmptyList::new(&board);
    let removed = (empty != 0).then(|| {
        let mut square = list.first();
        for _ in 0..remove_index {
            square = list.next(square);
        }
        square
    });
    let encoded = removed.map_or(64, Square::index);
    write!(out, "{empty} {encoded}|")?;
    observe(out, &list)?;
    if let Some(square) = removed {
        list.remove(square);
        write!(out, "|")?;
        observe(out, &list)?;
        list.restore(square);
        write!(out, "|")?;
        observe(out, &list)?;
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
    Ok(())
}
