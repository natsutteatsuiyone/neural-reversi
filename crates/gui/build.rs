fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    // rust-i18n reads locales/ from its macro without telling Cargo.
    println!("cargo::rerun-if-changed=locales");
    println!("cargo::rerun-if-changed=locales/app.yml");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_OS");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo::rerun-if-changed=assets/icon.ico");
        winresource::WindowsResource::new()
            .set_icon("assets/icon.ico")
            .compile()
            .expect("compile Windows application icon");
    }
}
