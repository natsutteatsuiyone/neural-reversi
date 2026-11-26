use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "../../../crates/reversi-core/build.rs"]
mod engine_build;

const CORE: &str = "../../../crates/reversi-core";

fn read_core(rel: &str) -> String {
    let path = format!("{CORE}/{rel}");
    println!("cargo::rerun-if-changed={path}");
    std::fs::read_to_string(path).unwrap().replace("\r\n", "\n")
}

fn slice_fn<'a>(source: &'a str, start: &str, end_fn: Option<&str>) -> &'a str {
    let begin = source.find(start).unwrap();
    let anchor = end_fn.map_or(begin, |m| begin + source[begin..].find(m).unwrap());
    let end = anchor + source[anchor..].find("\n}\n").unwrap() + 3;
    &source[begin..end]
}

fn write_out(out: &Path, name: &str, contents: impl AsRef<[u8]>) {
    std::fs::write(out.join(name), contents).unwrap();
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(feature, values(\"probcut-trace\"))");
    println!("cargo::rerun-if-changed={CORE}/build.rs");
    println!("cargo::rerun-if-changed={CORE}/src/eval/pattern_table_data.rs");
    println!("cargo::rerun-if-changed={CORE}/src/probcut/midgame_params.rs");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    engine_build::generate_tables();

    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());

    let pvs = read_core("src/search/pvs.rs");
    let lmr = slice_fn(
        &pvs,
        "pub const LMR_MIN_DEPTH:",
        Some("\nfn lmr_max_reduction("),
    );
    write_out(&out, "lmr_source.rs", lmr);
    let (_, worker) = pvs.split_once("pub(super) fn search_split_point").unwrap();
    let (_, publish) = worker.split_once("        // Best score update\n").unwrap();
    let (publish, _) = publish
        .split_once("        if cutoff {\n            break;\n")
        .unwrap();
    write_out(
        &out,
        "split_report_source.rs",
        format!("{{\n{publish}cutoff\n}}"),
    );

    let endgame = read_core("src/search/endgame.rs");
    let delta = endgame
        .lines()
        .find(|l| l.starts_with("const ASPIRATION_DELTA:"))
        .unwrap();
    let asp = slice_fn(&endgame, "fn aspiration_search(", None);
    write_out(
        &out,
        "endgame_aspiration_source.rs",
        format!("{delta}\n{asp}"),
    );

    let large_pages = read_core("src/util/large_pages.rs");
    let (_, windows) = large_pages.split_once("#[cfg(windows)]\n").unwrap();
    let (windows, _) = windows.split_once("#[cfg(target_os = \"linux\")]").unwrap();
    write_out(&out, "windows_large_pages.rs", windows);

    let eval_util = read_core("src/eval/util.rs");
    let eof = slice_fn(&eval_util, "pub fn expect_end_of_weights<", None);
    write_out(&out, "weight_eof_source.rs", eof);

    let move_list = read_core("src/move_list.rs");
    let (_, move_list) = move_list
        .split_once("mod iterator;\nmod move_array;\nmod ordering;\n")
        .unwrap();
    let (before, after) = move_list
        .split_once("pub(crate) use ordering::fast_move_value;\n")
        .unwrap();
    let (after, _) = after.split_once("\n#[cfg(test)]\nmod tests;").unwrap();
    let iterator = std::fs::canonicalize(format!("{CORE}/src/move_list/iterator.rs")).unwrap();
    let array = std::fs::canonicalize(format!("{CORE}/src/move_list/move_array.rs")).unwrap();
    let move_list_src = format!(
        "#[path = {iterator:?}]\nmod iterator;\n#[path = {array:?}]\nmod move_array;\n{before}{after}"
    );
    write_out(&out, "move_list_source.rs", move_list_src);

    let tt = read_core("src/transposition_table.rs");
    let first = tt
        .find("/// Size of each cluster in the transposition table.")
        .unwrap();
    let (slot, _) = tt[first..]
        .split_once("\n#[cfg(test)]\nimpl TTEntry {")
        .unwrap();
    let (before, after) = slot
        .split_once(
            "// SAFETY: an all-zero `TTEntry` represents an empty entry.\nunsafe impl Zeroable for TTEntry {}\n",
        )
        .unwrap();
    write_out(&out, "tt_slot_source.rs", format!("{before}{after}"));

    let stability = read_core("src/stability.rs");
    let first = stability.find("use crate::{").unwrap();
    let (production, _) = stability[first..].split_once("\n#[cfg(test)]").unwrap();
    write_out(&out, "stability_source.rs", production);
}
