fn main() {
    tauri_build::build();

    if let Err(e) = build_common::copy_weight_files("../..") {
        eprintln!("cargo:warning=Error in build_common: {}", e);
    }

    println!("cargo:rerun-if-changed=build.rs");
}
