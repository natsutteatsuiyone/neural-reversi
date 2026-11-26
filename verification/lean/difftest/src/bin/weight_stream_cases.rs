use byteorder::{LittleEndian, ReadBytesExt};
use flip_scalar_difftest::{csv, run_buffered};
use std::fs::File;
use std::io::{self, Cursor, Read, Write};
use std::path::Path;

include!(concat!(env!("OUT_DIR"), "/weight_eof_source.rs"));

struct Fragmented<R> {
    inner: R,
    maximum: usize,
    interrupts: usize,
    pending_interrupts: usize,
    eof_error: Option<io::ErrorKind>,
    consumed: usize,
}

impl<R: Read> Read for Fragmented<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pending_interrupts > 0 {
            self.pending_interrupts -= 1;
            return Err(io::ErrorKind::Interrupted.into());
        }
        let cap = out.len().min(self.maximum);
        let read = self.inner.read(&mut out[..cap])?;
        self.pending_interrupts = self.interrupts;
        self.consumed += read;
        match self.eof_error.filter(|_| read == 0) {
            Some(error) => Err(error.into()),
            None => Ok(read),
        }
    }
}

fn read_fields(reader: &mut impl Read) -> io::Result<[i32; 5]> {
    let mut pair = [0i16; 2];
    reader.read_i16_into::<LittleEndian>(&mut pair)?;
    let wide = reader.read_i32::<LittleEndian>()?;
    let signed = reader.read_i8()?;
    let last = reader.read_i16::<LittleEndian>()?;
    expect_end_of_weights(reader)?;
    Ok([
        pair[0].into(),
        pair[1].into(),
        wide,
        signed.into(),
        last.into(),
    ])
}

fn error_code(error: io::ErrorKind) -> &'static str {
    match error {
        io::ErrorKind::UnexpectedEof => "eof",
        io::ErrorKind::InvalidData => "invalid",
        io::ErrorKind::PermissionDenied => "denied",
        io::ErrorKind::Other => "other",
        _ => panic!("unexpected native reader error: {error:?}"),
    }
}

fn write_case(
    out: &mut impl Write,
    decoded: &[u8],
    terminal: Option<io::ErrorKind>,
    kind: &str,
    mut reader: Fragmented<impl Read>,
) -> io::Result<()> {
    let result = read_fields(&mut reader);
    let observed = match result {
        Ok(values) => csv(values),
        Err(error) => error_code(error.kind()).to_owned(),
    };
    let terminal = terminal.map_or("clean", error_code);
    writeln!(
        out,
        "{kind} {} {terminal} {} {} {} {observed}",
        csv(decoded),
        reader.maximum,
        reader.interrupts,
        reader.consumed
    )
}

fn write_frames(out: &mut impl Write, payload: &[u8; 11]) -> io::Result<()> {
    let encode = |bytes: &[u8]| {
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 0)?;
        encoder.include_checksum(true)?;
        encoder.write_all(bytes)?;
        encoder.finish()
    };
    let frame = encode(payload)?;
    let empty = encode(&[])?;
    let cases = [
        ("complete-frame", frame.clone(), payload.to_vec(), None),
        (
            "split-frames",
            [encode(&payload[..5])?, encode(&payload[5..])?].concat(),
            payload.to_vec(),
            None,
        ),
        (
            "empty-frame",
            [frame.clone(), empty].concat(),
            payload.to_vec(),
            None,
        ),
        (
            "short-frame",
            encode(&payload[..10])?,
            payload[..10].to_vec(),
            None,
        ),
        (
            "truncated-checksum",
            frame[..frame.len() - 1].to_vec(),
            payload.to_vec(),
            Some(io::ErrorKind::UnexpectedEof),
        ),
        (
            "continued-frame",
            [frame, encode(&[234])?].concat(),
            [payload.as_slice(), &[234]].concat(),
            None,
        ),
    ];
    for (name, encoded, decoded, terminal) in cases {
        for (maximum, interrupts) in [(1, 0), (3, 1), (usize::MAX, 0)] {
            let reader = Fragmented {
                inner: zstd::stream::read::Decoder::new(encoded.as_slice())?,
                maximum,
                interrupts,
                pending_interrupts: interrupts,
                eof_error: None,
                consumed: 0,
            };
            write_case(out, &decoded, terminal, name, reader)?;
        }
    }
    Ok(())
}

fn write_cases(out: &mut impl Write) -> io::Result<()> {
    let payload = [255, 127, 0, 128, 18, 52, 86, 128, 129, 254, 255, 171, 205];
    for mask in [0, 85, 170, 255] {
        let encoded = payload.map(|byte| byte ^ mask);
        for len in 0..=encoded.len() {
            let bytes = &encoded[..len];
            for maximum in [1, 2, 3, 4, 11] {
                for interrupts in [0, 1, 4] {
                    for terminal in [None, Some(io::ErrorKind::PermissionDenied)] {
                        let reader = Fragmented {
                            inner: Cursor::new(bytes),
                            maximum,
                            interrupts,
                            pending_interrupts: interrupts,
                            eof_error: terminal,
                            consumed: 0,
                        };
                        write_case(out, bytes, terminal, "reader", reader)?;
                    }
                }
            }
        }
    }
    write_frames(out, payload[..11].try_into().unwrap())
}

fn main() -> io::Result<()> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    run_buffered(|out| {
        if let [main, small] = args.as_slice() {
            for (network, path) in [("main", Path::new(main)), ("small", Path::new(small))] {
                let mut decoder = zstd::stream::read::Decoder::new(File::open(path)?)?;
                let count = io::copy(&mut decoder, &mut io::sink())?;
                writeln!(out, "layout {network} {count}")?;
            }
            return Ok(());
        }
        if !args.is_empty() {
            return Err(io::Error::other(
                "expected no arguments, or main and small weight paths",
            ));
        }
        write_cases(out)
    })
}
