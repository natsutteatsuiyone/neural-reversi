use std::fmt::{self, Write as _};
use std::io::{self, Write as _};

pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

pub fn csv<I>(items: I) -> String
where
    I: IntoIterator,
    I::Item: fmt::Display,
{
    let mut iter = items.into_iter();
    let Some(first) = iter.next() else {
        return "-".to_owned();
    };
    let mut out = first.to_string();
    for item in iter {
        let _ = write!(out, ",{item}");
    }
    out
}

pub fn opt_or_dash<T: fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| value.to_string())
}

pub fn run_buffered(
    write: impl FnOnce(&mut io::BufWriter<io::StdoutLock<'static>>) -> io::Result<()>,
) -> io::Result<()> {
    let mut out = io::BufWriter::new(io::stdout().lock());
    write(&mut out)?;
    out.flush()
}
