use std::io;

#[path = "../../../../../crates/reversi-core/src/search"]
mod search {
    #[path = "threading/in_flight.rs"]
    mod in_flight;

    use in_flight::{Claim, InFlightTable};
    use std::io::{self, Write};

    fn code(claim: &Claim<'_>) -> char {
        match claim {
            Claim::Owned(_) => 'o',
            Claim::Busy => 'b',
            Claim::Untracked => 'u',
        }
    }

    pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
        let table = InFlightTable::new();
        for first in [0, 1, 42, 0xffff, 0x1_0000, 0xffff_ffff, 1 << 63, u64::MAX] {
            for difference in [0, 1, 2, 1 << 15, 1 << 16, 1 << 32, 1 << 63, u64::MAX] {
                let second = first ^ difference;
                for drop_first in [false, true] {
                    let a = table.claim(first);
                    let b = table.claim(second);
                    write!(
                        out,
                        "{first} {second} {} {}{}",
                        u8::from(drop_first),
                        code(&a),
                        code(&b)
                    )?;
                    let probe = |out: &mut dyn Write| {
                        let first = code(&table.claim(first));
                        let second = code(&table.claim(second));
                        write!(out, "{first}{second}")
                    };
                    probe(out)?;
                    let (released, kept) = if drop_first { (a, b) } else { (b, a) };
                    drop(released);
                    probe(out)?;
                    drop(kept);
                    probe(out)?;
                    writeln!(out)?;
                }
            }
        }
        Ok(())
    }
}

fn main() -> io::Result<()> {
    flip_scalar_difftest::run_buffered(search::write_cases)
}
