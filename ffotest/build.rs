fn main() {
    let files_to_copy = [
        ("../eval.zst", "eval.zst"),
        ("../eval_sm.zst", "eval_sm.zst"),
    ];

    if let Err(e) = build_common::copy_files_to_target_profile_dir(&files_to_copy) {
        eprintln!("cargo:warning=Error in build_common: {}", e);
    }

    println!("cargo:rerun-if-changed=build.rs");
}
