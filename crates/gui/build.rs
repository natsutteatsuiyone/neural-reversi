fn main() {
    // rust-i18n reads locales/ from its macro without telling Cargo.
    println!("cargo:rerun-if-changed=locales");
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=assets/icon.ico");
        winresource::WindowsResource::new()
            .set_icon("assets/icon.ico")
            .compile()
            .expect("compile Windows application icon");
    }
}
