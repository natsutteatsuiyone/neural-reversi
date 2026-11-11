fn main() {
    println!(
        "cargo::rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap()
    );

    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let mut flags = rustflags.split('\x1f');
    let mut target_cpu = None;
    while let Some(flag) = flags.next() {
        let codegen = match flag {
            "-C" | "--codegen" => flags.next(),
            _ => flag
                .strip_prefix("-C")
                .or_else(|| flag.strip_prefix("--codegen=")),
        };
        if let Some(cpu) = codegen.and_then(|c| c.strip_prefix("target-cpu=")) {
            target_cpu = Some(cpu);
        }
    }
    println!("cargo::rustc-env=TARGET_CPU={}", target_cpu.unwrap_or(""));
}
