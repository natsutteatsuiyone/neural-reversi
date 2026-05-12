//! Emits `EVAL_FEATURE_U16_SUM` (16 MiB) to `$OUT_DIR/eval_feature_u16_sum.bin`
//! when the target enables `avx2` or `avx512bw`. Generating the table as a
//! `const fn` exceeds rustc's `long_running_const_eval` budget; the data and
//! the index helper live in `src/eval/pattern_table_data.rs` so both this
//! script and the runtime module use the same definitions.

use std::env;
use std::fs;
use std::path::PathBuf;

include!("src/eval/pattern_table_data.rs");

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

fn write_u16_sum_table(path: &std::path::Path) {
    let mut table = generate_u16_sum_table();
    // Force LE byte order so the embedded blob is portable across build hosts.
    if cfg!(target_endian = "big") {
        for row in &mut table {
            for v in row {
                *v = v.to_le();
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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/eval/pattern_table_data.rs");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");

    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let needs_table =
        arch == "x86_64" && features.split(',').any(|f| f == "avx2" || f == "avx512bw");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let path = out_dir.join("eval_feature_u16_sum.bin");

    if needs_table {
        write_u16_sum_table(&path);
    } else {
        // Drop any stale output from a previous build so the .bin doesn't
        // survive a cfg toggle.
        let _ = fs::remove_file(&path);
    }
}
