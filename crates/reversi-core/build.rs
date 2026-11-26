//! Emits `EVAL_FEATURE_U16_SUM` (16 MiB) to `$OUT_DIR/eval_feature_u16_sum.bin`
//! in the target's byte order when it enables AVX2/AVX-512BW or AArch64 NEON.
//! Generating the table as a `const fn` exceeds rustc's `long_running_const_eval`
//! budget. The data and the index helper live in `src/eval/pattern_table_data.rs`
//! so this script and the runtime module use the same definitions.
//!
//! Also emits the midgame ProbCut coefficient table to
//! `$OUT_DIR/midgame_probcut_table.rs`, tabulated from
//! the model in `src/probcut/midgame_params.rs`; `exp` and `sqrt` are not
//! `const fn`.

use std::env;
use std::fmt::Write;
use std::fs;
use std::path::PathBuf;

include!("src/eval/pattern_table_data.rs");
include!("src/probcut/midgame_params.rs");

fn build_eval_feature_rows() -> Vec<[u16; FEATURE_VECTOR_SIZE]> {
    let mut result = vec![[0u16; FEATURE_VECTOR_SIZE]; BOARD_SQUARES];
    for (sq_idx, row) in result.iter_mut().enumerate() {
        let board = 1u64 << sq_idx;
        for (row_slot, &(n_square, squares)) in row.iter_mut().zip(EVAL_F2X_RAW.iter()) {
            *row_slot = compute_pattern_feature_index_raw(board, n_square, squares) as u16;
        }
    }
    result
}

fn generate_u16_sum_table() -> Vec<[u16; FEATURE_VECTOR_SIZE]> {
    let ef = build_eval_feature_rows();
    let mut table = vec![[0u16; FEATURE_VECTOR_SIZE]; FLIP_U16_TABLES * FLIP_U16_VALUES];

    // Slot 0 is left zero so the SIMD update paths can unconditionally load
    // all four chunks; an empty chunk then contributes the identity.
    for chunk_idx in 0..FLIP_U16_TABLES {
        for mask in 1..FLIP_U16_VALUES {
            let prev_mask = mask & (mask - 1);
            let bit = (mask ^ prev_mask).trailing_zeros() as usize;
            let square_idx = chunk_idx * FLIP_U16_BITS + bit;

            let prev = table[chunk_idx * FLIP_U16_VALUES + prev_mask];
            let cur = &mut table[chunk_idx * FLIP_U16_VALUES + mask];
            for ((c, p), e) in cur.iter_mut().zip(prev.iter()).zip(ef[square_idx].iter()) {
                *c = p.wrapping_add(*e);
            }
        }
    }
    table
}

fn write_u16_sum_table(path: &std::path::Path, target_is_little_endian: bool) {
    let mut table = generate_u16_sum_table();
    // The target transmutates the bytes directly into u16 lanes.
    if cfg!(target_endian = "little") != target_is_little_endian {
        for row in &mut table {
            for v in row {
                *v = v.swap_bytes();
            }
        }
    }
    // SAFETY: `[u16; FEATURE_VECTOR_SIZE]` is `Copy` and contains no padding,
    // so reinterpreting the contiguous `Vec` storage as bytes is sound.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            table.as_ptr() as *const u8,
            std::mem::size_of_val(table.as_slice()),
        )
    };
    fs::write(path, bytes).expect("failed to write eval_feature_u16_sum.bin");
}

fn probcut_depth(deep: u32) -> u32 {
    2 * (deep / 6)
}

fn build_midgame_table() -> Vec<[[f64; 4]; NUM_DEPTH]> {
    let scale = f64::from(SCORE_SCALE);
    PROBCUT_PARAMS
        .iter()
        .map(|params| {
            std::array::from_fn(|deep| {
                let shallow = f64::from(probcut_depth(deep as u32));
                let deep = deep as f64;
                [
                    params.mean(shallow, deep) * scale,
                    params.sigma(shallow, deep) * scale,
                    params.mean(0.0, deep) * scale,
                    params.sigma(0.0, deep) * scale,
                ]
            })
        })
        .collect()
}

fn write_midgame_table(path: &std::path::Path) {
    let mut source = String::new();
    writeln!(source, "const MIDGAME_TABLE_PLY: usize = {NUM_PLY};").unwrap();
    writeln!(source, "const MIDGAME_TABLE_DEPTH: usize = {NUM_DEPTH};").unwrap();
    writeln!(source, "const _: () = {{").unwrap();
    writeln!(source, "    assert!(Score::SCALE == {SCORE_SCALE});").unwrap();
    write!(
        source,
        "    const EXPECTED_SHALLOW: [Depth; MIDGAME_TABLE_DEPTH] = ["
    )
    .unwrap();
    for deep in 0..NUM_DEPTH {
        if deep > 0 {
            source.push_str(", ");
        }
        write!(source, "{}", probcut_depth(deep as u32)).unwrap();
    }
    writeln!(source, "];").unwrap();
    writeln!(source, "    let mut deep = 0;").unwrap();
    writeln!(source, "    while deep < MIDGAME_TABLE_DEPTH {{").unwrap();
    writeln!(
        source,
        "        assert!(probcut_depth(deep as Depth) == EXPECTED_SHALLOW[deep]);"
    )
    .unwrap();
    writeln!(source, "        deep += 1;").unwrap();
    writeln!(source, "    }}").unwrap();
    writeln!(source, "}};").unwrap();
    writeln!(
        source,
        "static MIDGAME_TABLE: [[ProbcutCoefficients; MIDGAME_TABLE_DEPTH]; MIDGAME_TABLE_PLY] = ["
    )
    .unwrap();
    for row in build_midgame_table() {
        source.push('[');
        for [mean, sigma, mean0, sigma0] in row {
            write!(
                source,
                "ProbcutCoefficients{{mean:f64::from_bits({:#x}),sigma:f64::from_bits({:#x}),\
                 mean0:f64::from_bits({:#x}),sigma0:f64::from_bits({:#x})}},",
                mean.to_bits(),
                sigma.to_bits(),
                mean0.to_bits(),
                sigma0.to_bits(),
            )
            .unwrap();
        }
        source.push_str("],\n");
    }
    source.push_str("];\n");
    fs::write(path, source).expect("failed to write midgame_probcut_table.rs");
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/eval/pattern_table_data.rs");
    println!("cargo::rerun-if-changed=src/probcut/midgame_params.rs");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ENDIAN");
    generate_tables();
}

pub(crate) fn generate_tables() {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let needs_table = match arch.as_str() {
        "x86_64" => features.split(',').any(|f| f == "avx2" || f == "avx512bw"),
        "aarch64" => features.split(',').any(|f| f == "neon"),
        _ => false,
    };

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    write_midgame_table(&out_dir.join("midgame_probcut_table.rs"));
    let path = out_dir.join("eval_feature_u16_sum.bin");

    if needs_table {
        let endian = env::var("CARGO_CFG_TARGET_ENDIAN").expect("CARGO_CFG_TARGET_ENDIAN not set");
        write_u16_sum_table(&path, endian == "little");
    } else {
        // Drop any stale output from a previous build so the .bin doesn't
        // survive a cfg toggle.
        let _ = fs::remove_file(&path);
    }
}
