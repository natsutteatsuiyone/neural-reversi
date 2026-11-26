use std::io::{self, Write};

use flip_scalar_difftest::csv;

use crate::constants::MAX_PLY;
use crate::search_stack::SearchStack;
use crate::square::Square;

fn encode(row: &[Option<Square>; MAX_PLY]) -> String {
    csv(row.iter().map(|entry| entry.map_or(64, Square::index)))
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let old = std::array::from_fn(|i| Square::from_u8(((i * 29 + 47) % 64) as u8));
    let empty = [None; MAX_PLY];
    let selected = Square::D3;
    let selected_index = selected.index();
    for ply in [0, 1, MAX_PLY - 2] {
        for first_none in 0..=MAX_PLY {
            let child = std::array::from_fn(|i| {
                (i != first_none)
                    .then(|| Square::from_u8(((i * 13 + first_none + ply) % 64) as u8).unwrap())
            });
            let mut stack = SearchStack::new();
            stack.set_pv(ply, &old);
            stack.set_pv(ply + 1, &child);
            stack.update_pv(selected, ply);
            writeln!(
                out,
                "u {ply} {selected_index} {} {} {} {}",
                encode(&old),
                encode(&child),
                encode(stack.pv(ply)),
                encode(stack.pv(ply + 1)),
            )?;
        }
    }
    for ply in [0, 1, MAX_PLY - 2, MAX_PLY - 1] {
        let mut stack = SearchStack::new();
        stack.set_pv(ply, &old);
        let child = if ply + 1 < MAX_PLY {
            stack.set_pv(ply + 1, &old);
            &old
        } else {
            &empty
        };
        stack.prepare_pv(ply);
        let next = if ply + 1 < MAX_PLY {
            stack.pv(ply + 1)
        } else {
            &empty
        };
        writeln!(
            out,
            "p {ply} {selected_index} {} {} {} {}",
            encode(&old),
            encode(child),
            encode(stack.pv(ply)),
            encode(next),
        )?;
    }

    for ply in 0..MAX_PLY - 1 {
        for length in [0, MAX_PLY / 2, MAX_PLY] {
            let child = std::array::from_fn(|i| if i < length { old[i] } else { None });
            let mut stack = SearchStack::new();
            for row in 0..MAX_PLY {
                assert_eq!(*stack.pv(row), empty);
                stack.set_pv(row, &old);
            }
            stack.set_pv(ply + 1, &child);
            stack.update_pv(Square::A1, ply);
            // A depth-zero sibling clears the child row and returns no continuation.
            stack.clear_pv(ply + 1);
            stack.update_pv(selected, ply);
            for row in 0..MAX_PLY {
                if row != ply && row != ply + 1 {
                    assert_eq!(*stack.pv(row), old, "unrelated row {row} changed");
                }
            }
            writeln!(
                out,
                "b {ply} {selected_index} {} {} {} {}",
                encode(&old),
                encode(&child),
                encode(stack.pv(ply)),
                encode(stack.pv(ply + 1)),
            )?;
        }
    }
    Ok(())
}
