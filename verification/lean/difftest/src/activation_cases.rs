use std::io::{self, Write};

use crate::SplitMix64;
use crate::activations::{screlu, screlu_store_order, sqr_clipped_and_clipped_relu_16};
use crate::align::Align64;

const FUSED_BACKEND: &str = cfg_select! {
    all(
        target_arch = "x86_64",
        target_feature = "avx512bw",
        target_feature = "avx512vl"
    ) => "avx512",
    all(target_arch = "x86_64", target_feature = "avx2") => "avx2",
    all(target_arch = "aarch64", target_feature = "neon") => "neon",
    _ => "scalar",
};

const SCRELU_BACKEND: &str = if cfg!(all(target_arch = "x86_64", target_feature = "avx512bw")) {
    "avx512"
} else {
    FUSED_BACKEND
};

const EDGES: [i32; 20] = [
    i32::MIN,
    i16::MIN as i32 - 1,
    i16::MIN as i32,
    i16::MIN as i32 + 1,
    -16320,
    -1025,
    -1024,
    -1,
    0,
    63,
    64,
    1023,
    1024,
    16319,
    16320,
    16321,
    i16::MAX as i32 - 1,
    i16::MAX as i32,
    i16::MAX as i32 + 1,
    i32::MAX,
];

pub fn write_cases(out: &mut impl Write) -> io::Result<()> {
    writeln!(out, "{FUSED_BACKEND} {SCRELU_BACKEND}")?;
    let mut rng = SplitMix64(0x7265_6c75_2d63_6173);
    for case in 0..96 {
        let input = make_input::<16>(case, &mut rng);
        let mut output = Align64([0xa5; 32]);
        sqr_clipped_and_clipped_relu_16(&input, &mut output);
        write!(out, "fused|")?;
        write_values(out, &input[..])?;
        write!(out, "|")?;
        write_values(out, &output[..])?;
        writeln!(out)?;
    }

    write_sc_cases::<0>(out, &mut rng)?;
    write_sc_cases::<1>(out, &mut rng)?;
    write_sc_cases::<16>(out, &mut rng)?;
    write_sc_cases::<17>(out, &mut rng)?;
    write_sc_cases::<32>(out, &mut rng)?;
    write_sc_cases::<37>(out, &mut rng)?;
    write_sc_cases::<64>(out, &mut rng)?;
    write_sc_cases::<128>(out, &mut rng)
}

fn make_input<const SIZE: usize>(case: usize, rng: &mut SplitMix64) -> Align64<[i32; SIZE]> {
    Align64(std::array::from_fn(|i| {
        if case < EDGES.len() {
            EDGES[(case + i) % EDGES.len()]
        } else if case == EDGES.len() {
            7680 + 68 * i as i32
        } else if case.is_multiple_of(2) {
            (rng.next() % 50_000) as i32 - 25_000
        } else {
            rng.next() as i32
        }
    }))
}

fn write_sc_cases<const SIZE: usize>(out: &mut impl Write, rng: &mut SplitMix64) -> io::Result<()> {
    let order = screlu_store_order::<SIZE>();
    for case in 0..64 {
        let input = make_input::<SIZE>(case, rng);
        let mut output = Align64([0xa5; SIZE]);
        screlu(&input, &mut output);
        write!(out, "screlu|")?;
        write_values(out, &input[..])?;
        write!(out, "|")?;
        write_values(out, &output[..])?;
        write!(out, "|")?;
        write_values(out, &order)?;
        writeln!(out)?;
    }
    Ok(())
}

fn write_values(out: &mut impl Write, values: &[impl std::fmt::Display]) -> io::Result<()> {
    for (i, value) in values.iter().enumerate() {
        if i != 0 {
            write!(out, " ")?;
        }
        write!(out, "{value}")?;
    }
    Ok(())
}
