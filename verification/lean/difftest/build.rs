#[allow(dead_code)]
#[path = "../../../crates/reversi-core/build.rs"]
mod engine_build;

fn main() {
    println!("cargo::rerun-if-changed=../../../crates/reversi-core/build.rs");
    println!("cargo::rerun-if-changed=../../../crates/reversi-core/src/eval/pattern_table_data.rs");
    println!("cargo::rerun-if-changed=../../../crates/reversi-core/src/probcut/midgame_params.rs");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
    engine_build::generate_tables();
}
