#[path = "../../../../../crates/reversi-core/src/util/bitset.rs"]
mod bitset;
#[path = "../../../../../crates/reversi-core/src/util/spinlock.rs"]
mod spinlock;

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::thread;

use bitset::AtomicBitSet;
use spinlock::SpinLock;

fn snapshot(mask: &AtomicBitSet) -> String {
    let bits = (0..64).fold(0, |bits, i| bits | (u64::from(mask.test(i)) << i));
    format!("{bits} {} {}", mask.count(), u8::from(mask.none()))
}

fn mask_cases(out: &mut impl Write) -> io::Result<()> {
    for first in 0..64 {
        for second in 0..64 {
            let start = [0, u64::MAX, 0x55aa_55aa_55aa_55aa][(first + second) % 3];
            let mask = AtomicBitSet::new();
            let empty = snapshot(&mask);
            for i in 0..64 {
                if start & (1 << i) != 0 {
                    mask.set(i);
                }
            }
            mask.set(first);
            mask.set(second);
            let booked = snapshot(&mask);
            mask.reset(first);
            let partial = snapshot(&mask);
            mask.reset(second);
            let finished = snapshot(&mask);
            mask.clear();
            let cleared = snapshot(&mask);
            writeln!(
                out,
                "mask {start} {first} {second}|{empty}|{booked}|{partial}|{finished}|{cleared}"
            )?;
        }
    }
    Ok(())
}

fn guard_handoff(value: u64, transfer: bool) -> u64 {
    let lock = &SpinLock::new();
    let started = &AtomicBool::new(false);
    let payload = &AtomicU64::new(0);
    thread::scope(|scope| {
        if transfer {
            let (send, receive) = mpsc::channel();
            scope.spawn(move || {
                let guard = lock.lock();
                payload.store(value, Ordering::Relaxed);
                started.store(true, Ordering::Relaxed);
                send.send(guard).unwrap();
            });
            scope.spawn(move || drop(receive.recv().unwrap()));
        } else {
            scope.spawn(move || {
                let _guard = lock.lock();
                payload.store(value, Ordering::Relaxed);
                started.store(true, Ordering::Relaxed);
            });
        }
        while !started.load(Ordering::Relaxed) {
            thread::yield_now();
        }
        let _read_guard = lock.lock();
        payload.load(Ordering::Relaxed)
    })
}

fn main() -> io::Result<()> {
    flip_scalar_difftest::run_buffered(|out| {
        mask_cases(out)?;
        for transfer in [false, true] {
            for value in [1, 31, 47, u64::MAX] {
                let observed = guard_handoff(value, transfer);
                writeln!(out, "guard {} {value} {observed}", u8::from(transfer))?;
            }
        }
        Ok(())
    })
}
