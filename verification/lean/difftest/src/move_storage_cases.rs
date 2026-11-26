use std::io::{self, Write};

use flip_scalar_difftest::csv;

#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../../../crates/reversi-core/src/move_list"]
mod engine {
    pub mod iterator;
    pub mod move_array;

    #[derive(Clone, Copy, Debug)]
    #[repr(C)]
    pub struct Move {
        payload: u64,
        pub value: i32,
        pub id: u8,
    }

    impl Move {
        pub fn checked_id(&self) -> u8 {
            assert_eq!(self.payload, 0x1234_5678_9abc_def0 ^ u64::from(self.id));
            self.id
        }
    }

    pub struct MoveList {
        moves: move_array::MoveArray,
    }

    pub fn build(values: &[i32]) -> MoveList {
        assert!(values.len() <= crate::legal_moves::LegalMoves::MAX_COUNT);
        let mut raw = std::mem::MaybeUninit::<move_array::MoveArray>::uninit();
        // SAFETY: The capacity check bounds the writes, and len covers exactly that prefix.
        unsafe {
            let data = move_array::MoveArray::data_ptr_from(raw.as_mut_ptr());
            for (i, &value) in values.iter().enumerate() {
                data.add(i).write(Move {
                    payload: 0x1234_5678_9abc_def0 ^ i as u64,
                    value,
                    id: u8::try_from(i).unwrap(),
                });
            }
            move_array::MoveArray::len_ptr_from(raw.as_mut_ptr()).write(values.len());
            MoveList {
                moves: raw.assume_init(),
            }
        }
    }

    pub fn sorted_ids(list: &MoveList, values: &[i32]) -> Vec<u8> {
        let mut sorted = list.moves.clone();
        sorted.sort_by_key_desc(|mv| mv.value);
        sorted
            .iter()
            .map(|mv| {
                let id = mv.checked_id();
                assert_eq!(mv.value, values[usize::from(id)]);
                id
            })
            .collect()
    }

    pub fn retained_ids(list: &mut MoveList, mask: u64) -> Vec<u8> {
        list.moves.retain(|mv| mask & (1 << usize::from(mv.id)) != 0);
        list.moves.iter().map(Move::checked_id).collect()
    }
}

use crate::legal_moves::LegalMoves;
use engine::iterator::ConcurrentMoveIterator;

fn write_order(out: &mut impl Write, values: &[i32]) -> io::Result<()> {
    let mut list = engine::build(values);
    let sorted = engine::sorted_ids(&list, values);
    let mut iter = list.best_first_iter();
    let mut best = Vec::new();
    for remaining in (1..=values.len()).rev() {
        assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
        let chosen = iter.next().unwrap();
        let id = chosen.checked_id();
        assert_eq!(chosen.value, values[usize::from(id)]);
        best.push(id);
    }
    assert_eq!(iter.size_hint(), (0, Some(0)));
    assert!(iter.next().is_none());
    assert!(iter.next().is_none());
    writeln!(out, "order {} {} {}", csv(values), csv(&sorted), csv(&best))
}

fn write_claims(out: &mut impl Write, len: usize, start: usize) -> io::Result<()> {
    let list = engine::build(&vec![0; len]);
    let iter = ConcurrentMoveIterator::from_offset(list, start);
    let mut claims = Vec::new();
    while let Some((mv, ordinal)) = iter.next() {
        claims.push(format!("{}:{ordinal}", mv.checked_id()));
    }
    assert!(iter.next().is_none());
    writeln!(out, "claims {} {len} {start} {}", usize::BITS, csv(&claims))
}

fn write_retain(out: &mut impl Write, len: usize, mask: u64) -> io::Result<()> {
    let mut list = engine::build(&vec![0; len]);
    let ids = engine::retained_ids(&mut list, mask);
    writeln!(out, "retain {len} {mask} {}", csv(&ids))
}

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    for len in 0..=3 {
        for ordinal in 0..3usize.pow(len) {
            let values = (0..len)
                .map(|i| i32::try_from(ordinal / 3usize.pow(i) % 3).unwrap() - 1)
                .collect::<Vec<_>>();
            write_order(out, &values)?;
        }
    }
    let scores = [i32::MIN, -2, -1, 0, 1, 2, i32::MAX];
    for len in 4..=LegalMoves::MAX_COUNT {
        for offset in 0..scores.len() {
            let values = (0..len)
                .map(|i| scores[(i * i + offset * (i + 1)) % scores.len()])
                .collect::<Vec<_>>();
            write_order(out, &values)?;
        }
        write_order(out, &vec![0; len])?;
    }
    for len in 0..=LegalMoves::MAX_COUNT {
        for start in [0, 1, len / 2, len, len + 1, usize::MAX] {
            write_claims(out, len, start)?;
        }
    }
    for len in 0..=5 {
        for mask in 0..(1 << len) {
            write_retain(out, len, mask)?;
        }
    }
    for mask in [0, 0xaaaa_aaaa_aaaa_aaaa, u64::MAX] {
        write_retain(out, LegalMoves::MAX_COUNT, mask)?;
    }
    Ok(())
}
